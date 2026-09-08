#!/usr/bin/env python3
"""
AI-Powered CI Failure Triage Engine
===================================

Architecture:
    Jenkins Build
        -> Data Collection
        -> Failure Preprocessing
        -> Correlation Engine
        -> AI Triage Engine
        -> Risk Assessment
        -> Report Generator

Inputs:
    * Allure *-result.json files
    * Jenkins console log (optional)
    * Jenkins/build environment metadata

Outputs:
    * Markdown engineering report
    * Professional HTML dashboard

Environment variables:
    OPENAI_API_KEY        Required for AI analysis
    JENKINS_USER          Optional, for authenticated Jenkins access
    JENKINS_API_TOKEN     Optional, for authenticated Jenkins access
"""

import argparse
import base64
import hashlib
import html
import json
import os
import re
import sys
import urllib.request
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI


# ============================================================
# 1. CONFIGURATION
# ============================================================

AI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
MAX_TRACE_LENGTH = 4000
MAX_CONSOLE_LINES = 250
MAX_FAILURES_FOR_AI = 100

CATEGORIES = [
    "ui-locator",
    "timeout",
    "api-contract",
    "concurrency",
    "null-safety",
    "bounds-check",
    "logic-bug",
    "flaky",
    "environment",
    "build",
    "infrastructure",
    "assertion",
    "unknown",
]

SEVERITY_ORDER = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1}
CONFIDENCE_ORDER = {"High": 3, "Medium": 2, "Low": 1}


# ============================================================
# 2. DATA MODELS
# ============================================================

@dataclass
class BuildMetadata:
    build_url: str
    build_number: str = "N/A"
    job_name: str = "N/A"
    pipeline_status: str = "UNKNOWN"
    timestamp: str = ""


@dataclass
class Failure:
    name: str
    status: str
    message: str
    trace: str
    module: str = "Unknown"
    exception_type: str = "Unknown"
    normalized_signature: str = ""
    fingerprint: str = ""
    rule_category: str = "unknown"
    duplicate_count: int = 1


@dataclass
class FailureCluster:
    cluster_id: str
    fingerprint: str
    failures: List[Failure] = field(default_factory=list)
    affected_tests: int = 0
    blast_radius: str = "Low"
    representative_error: str = ""
    probable_pattern: str = ""


@dataclass
class BuildMetrics:
    total: int = 0
    passed: int = 0
    failed: int = 0
    broken: int = 0
    skipped: int = 0

    @property
    def pass_rate(self) -> float:
        return round((self.passed / self.total) * 100, 1) if self.total else 100.0


# ============================================================
# 3. DATA COLLECTION LAYER
# ============================================================

def load_allure_results(report_dir: Path) -> List[Dict[str, Any]]:
    """Load valid Allure test result JSON files."""
    results = []

    for file_path in sorted(report_dir.glob("*-result.json")):
        try:
            results.append(json.loads(file_path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
            print(
                f"Warning: skipping unreadable Allure result {file_path.name}: {exc}",
                file=sys.stderr,
            )

    return results


def extract_module(result: Dict[str, Any]) -> str:
    """Extract module/package information from Allure labels."""
    labels = result.get("labels") or []

    preferred = ["package", "testClass", "parentSuite", "suite"]
    for label_type in preferred:
        for label in labels:
            if label.get("name") == label_type and label.get("value"):
                value = str(label["value"])
                if label_type == "package" and "." in value:
                    return value.rsplit(".", 1)[0]
                return value

    full_name = result.get("fullName") or ""
    if "." in full_name:
        return full_name.rsplit(".", 1)[0]

    return "Unknown"


def summarize_allure_results(results: List[Dict[str, Any]]):
    """Build metrics and structured failures from Allure results."""
    metrics = BuildMetrics()
    failures: List[Failure] = []

    for result in results:
        status = (result.get("status") or "").lower()
        if not status:
            continue

        metrics.total += 1

        if status == "passed":
            metrics.passed += 1
        elif status == "failed":
            metrics.failed += 1
        elif status == "broken":
            metrics.broken += 1
        elif status in {"skipped", "unknown"}:
            metrics.skipped += 1

        if status in {"failed", "broken"}:
            details = result.get("statusDetails") or {}
            failures.append(
                Failure(
                    name=result.get("fullName") or result.get("name") or "Unnamed Test",
                    status=status,
                    message=str(details.get("message") or ""),
                    trace=str(details.get("trace") or "")[:MAX_TRACE_LENGTH],
                    module=extract_module(result),
                )
            )

    return metrics, failures


def fetch_console_log(build_url: str, max_lines: int = MAX_CONSOLE_LINES) -> str:
    """Fetch Jenkins console log, optionally using Basic authentication."""
    if not build_url:
        return ""

    url = build_url.rstrip("/") + "/consoleText"
    request = urllib.request.Request(url)

    user = os.getenv("JENKINS_USER")
    token = os.getenv("JENKINS_API_TOKEN")

    if user and token:
        credentials = base64.b64encode(f"{user}:{token}".encode()).decode()
        request.add_header("Authorization", f"Basic {credentials}")

    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            text = response.read().decode("utf-8", errors="replace")

        lines = text.splitlines()
        return "\n".join(lines[-max_lines:])

    except Exception as exc:
        print(f"Warning: unable to fetch Jenkins console log: {exc}", file=sys.stderr)
        return ""


def collect_build_metadata(build_url: str, console_log: str) -> BuildMetadata:
    """Collect Jenkins metadata from environment and available console evidence."""
    status = "UNKNOWN"
    console_upper = console_log.upper()

    if "BUILD SUCCESS" in console_upper or "SUCCESS" in console_upper[-1000:]:
        status = "SUCCESS"
    elif "BUILD FAILURE" in console_upper or "FAILURE" in console_upper[-1500:]:
        status = "FAILURE"
    elif os.getenv("BUILD_RESULT"):
        status = os.getenv("BUILD_RESULT", "UNKNOWN").upper()

    return BuildMetadata(
        build_url=build_url,
        build_number=os.getenv("BUILD_NUMBER", "N/A"),
        job_name=os.getenv("JOB_NAME", "N/A"),
        pipeline_status=status,
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    )


# ============================================================
# 4. FAILURE PREPROCESSING LAYER
# ============================================================

def extract_exception_type(message: str, trace: str) -> str:
    """Extract the most useful exception/error type from failure evidence."""
    source = f"{message}\n{trace}"

    patterns = [
        r"\b([A-Za-z_][A-Za-z0-9_]*(?:Exception|Error|Failure))\b",
        r"\b(AssertionError)\b",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, source)
        if matches:
            return matches[-1]

    return "Unknown"


def normalize_failure_text(text: str) -> str:
    """
    Normalize volatile values so failures with the same underlying pattern
    can be clustered together.
    """
    if not text:
        return ""

    normalized = text.lower()

    # URLs, UUIDs, timestamps, hexadecimal values and numbers.
    normalized = re.sub(r"https?://[^\s]+", "<url>", normalized)
    normalized = re.sub(
        r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
        "<uuid>",
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(
        r"\b\d{4}-\d{2}-\d{2}[t\s]\d{2}:\d{2}:\d{2}(?:\.\d+)?z?\b",
        "<timestamp>",
        normalized,
    )
    normalized = re.sub(r"0x[0-9a-f]+", "<hex>", normalized)
    normalized = re.sub(r"\b\d+\b", "<num>", normalized)

    # Java stack line numbers and volatile file paths.
    normalized = re.sub(r":\d+\)", ":<line>)", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()

    return normalized


def generate_fingerprint(message: str, trace: str, exception_type: str) -> tuple[str, str]:
    """Generate a stable fingerprint from the strongest available evidence."""
    primary = message.strip() or trace.strip()
    normalized = normalize_failure_text(primary)

    if exception_type and exception_type != "Unknown":
        normalized = f"{exception_type}|{normalized}"

    # Limit input to prevent huge traces from dominating signatures.
    normalized = normalized[:1200]
    fingerprint = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]

    return normalized, fingerprint


def classify_failure_rules(message: str, trace: str, exception_type: str) -> str:
    """Deterministic first-pass classification used as evidence for AI."""
    source = f"{message} {trace} {exception_type}".lower()

    rules = [
        ("ui-locator", [
            "nosuchelementexception", "elementnotfound", "unable to locate",
            "locator", "selector", "staleelementreference", "webelement",
        ]),
        ("timeout", [
            "timeoutexception", "timeout", "timed out", "waited for",
            "read timed out", "sockettimeout",
        ]),
        ("api-contract", [
            "unexpected status", "http status", "status code", "schema validation",
            "response body", "jsonpath", "contract", "expected:<", "expected:",
        ]),
        ("concurrency", [
            "concurrentmodificationexception", "deadlock", "race condition",
            "lock acquisition", "thread",
        ]),
        ("null-safety", [
            "nullpointerexception", "cannot invoke", " is null", "none type",
        ]),
        ("bounds-check", [
            "indexoutofboundsexception", "arrayindexoutofboundsexception",
            "stringindexoutofboundsexception",
        ]),
        ("environment", [
            "connection refused", "unknown host", "dns", "certificate",
            "ssl", "unable to connect", "service unavailable",
        ]),
        ("infrastructure", [
            "agent offline", "workspace", "disk space", "out of memory",
            "docker", "kubernetes", "node lost",
        ]),
        ("build", [
            "compilation failure", "could not resolve dependencies",
            "maven", "gradle", "dependency resolution",
        ]),
        ("assertion", [
            "assertionerror", "assert failed", "expected", "but was",
        ]),
    ]

    for category, keywords in rules:
        if any(keyword in source for keyword in keywords):
            return category

    return "unknown"


def preprocess_failures(failures: List[Failure]) -> List[Failure]:
    """Normalize, fingerprint and classify every failure."""
    fingerprint_counts = Counter()

    for failure in failures:
        failure.exception_type = extract_exception_type(
            failure.message, failure.trace
        )

        signature, fingerprint = generate_fingerprint(
            failure.message,
            failure.trace,
            failure.exception_type,
        )

        failure.normalized_signature = signature
        failure.fingerprint = fingerprint
        failure.rule_category = classify_failure_rules(
            failure.message,
            failure.trace,
            failure.exception_type,
        )

        fingerprint_counts[fingerprint] += 1

    for failure in failures:
        failure.duplicate_count = fingerprint_counts[failure.fingerprint]

    return failures


# ============================================================
# 5. CORRELATION ENGINE
# ============================================================

def determine_blast_radius(count: int, total_failures: int) -> str:
    if count >= 10 or (total_failures and count / total_failures >= 0.5):
        return "High"
    if count >= 3:
        return "Medium"
    return "Low"


def detect_common_pattern(failures: List[Failure]) -> str:
    categories = Counter(f.rule_category for f in failures)
    exceptions = Counter(
        f.exception_type for f in failures if f.exception_type != "Unknown"
    )

    category = categories.most_common(1)[0][0] if categories else "unknown"
    exception = exceptions.most_common(1)[0][0] if exceptions else ""

    if exception:
        return f"Shared {category} pattern with recurring {exception}"
    return f"Shared {category} failure pattern"


def cluster_failures(failures: List[Failure]) -> List[FailureCluster]:
    """Group failures by normalized fingerprint."""
    grouped: Dict[str, List[Failure]] = defaultdict(list)

    for failure in failures:
        grouped[failure.fingerprint].append(failure)

    clusters = []

    for index, (fingerprint, grouped_failures) in enumerate(
        sorted(grouped.items(), key=lambda item: len(item[1]), reverse=True),
        start=1,
    ):
        representative = grouped_failures[0]

        clusters.append(
            FailureCluster(
                cluster_id=f"CL-{index:02d}",
                fingerprint=fingerprint,
                failures=grouped_failures,
                affected_tests=len(grouped_failures),
                blast_radius=determine_blast_radius(
                    len(grouped_failures), len(failures)
                ),
                representative_error=(
                    representative.message
                    or representative.trace[:300]
                    or "No detailed error message available"
                ),
                probable_pattern=detect_common_pattern(grouped_failures),
            )
        )

    return clusters


def console_pipeline_signals(console_log: str) -> List[str]:
    """Extract pipeline-level failure signals without inventing a root cause."""
    if not console_log:
        return []

    patterns = [
        (r"COMPILATION FAILURE", "Compilation failure detected"),
        (r"Could not resolve dependencies", "Dependency resolution failure detected"),
        (r"OutOfMemoryError", "Out-of-memory signal detected"),
        (r"Connection refused", "Connectivity failure signal detected"),
        (r"BUILD FAILURE", "Jenkins/Maven build failure marker detected"),
        (r"ERROR.*Exception", "Unhandled exception signal detected"),
    ]

    findings = []
    for pattern, label in patterns:
        if re.search(pattern, console_log, flags=re.IGNORECASE):
            findings.append(label)

    return findings


# ============================================================
# 6. AI TRIAGE ENGINE
# ============================================================

TRIAGE_SYSTEM_PROMPT = """
You are a Principal QA Engineer and CI Reliability Analyst performing evidence-based
failure triage for an engineering organization.

Your task is to analyze structured CI failure data and return STRICT JSON ONLY.
Do not use Markdown. Do not wrap JSON in code fences.

Critical evidence rules:
1. Never invent files, line numbers, APIs, deployments, incidents, or root causes.
2. A root cause is a hypothesis unless directly proven by the evidence.
3. Use High confidence only when the evidence strongly supports the conclusion.
4. Distinguish correlated failures from independent failures.
5. Severity reflects user/business/release impact, not merely exception type.
6. Prioritize systemic issues with larger blast radius when evidence supports it.
7. If evidence is insufficient, explicitly say "Insufficient evidence".
8. Suggested fixes must be concrete but must not claim knowledge of unavailable code.
9. Use only these categories:
   ui-locator, timeout, api-contract, concurrency, null-safety, bounds-check,
   logic-bug, flaky, environment, build, infrastructure, assertion, unknown
10. Severity must be one of: Critical, High, Medium, Low.
11. Confidence must be one of: High, Medium, Low.
12. Release recommendation must be one of: GO, CONDITIONAL_GO, HOLD, NO_GO.

Return this schema:
{
  "executive_summary": "string",
  "release_assessment": {
    "recommendation": "GO|CONDITIONAL_GO|HOLD|NO_GO",
    "risk_level": "Critical|High|Medium|Low",
    "reason": "string"
  },
  "clusters": [
    {
      "cluster_id": "CL-01",
      "root_cause_hypothesis": "string",
      "category": "allowed category",
      "severity": "Critical|High|Medium|Low",
      "confidence": "High|Medium|Low",
      "suggested_owner": "string",
      "suggested_fix": "string"
    }
  ],
  "correlation_analysis": "string",
  "remediation_plan": [
    {
      "priority": "P0|P1|P2|P3",
      "action": "string",
      "why": "string",
      "effort": "Small|Medium|Large",
      "owner": "string"
    }
  ],
  "detailed_findings": [
    {
      "test_name": "exact test name",
      "cluster_id": "CL-01",
      "category": "allowed category",
      "severity": "Critical|High|Medium|Low",
      "confidence": "High|Medium|Low",
      "root_cause_hypothesis": "string",
      "evidence": "string",
      "suggested_fix": "string",
      "suggested_owner": "string"
    }
  ]
}
"""


def build_ai_payload(
    metadata: BuildMetadata,
    metrics: BuildMetrics,
    failures: List[Failure],
    clusters: List[FailureCluster],
    console_log: str,
) -> Dict[str, Any]:
    """Build compact structured evidence for AI analysis."""
    limited_failures = failures[:MAX_FAILURES_FOR_AI]

    return {
        "build": asdict(metadata),
        "metrics": asdict(metrics),
        "pipeline_signals": console_pipeline_signals(console_log),
        "failure_clusters": [
            {
                "cluster_id": cluster.cluster_id,
                "affected_tests": cluster.affected_tests,
                "blast_radius": cluster.blast_radius,
                "probable_pattern": cluster.probable_pattern,
                "representative_error": cluster.representative_error[:1000],
                "tests": [failure.name for failure in cluster.failures[:20]],
            }
            for cluster in clusters
        ],
        "failures": [
            {
                "name": failure.name,
                "status": failure.status,
                "module": failure.module,
                "message": failure.message[:1500],
                "trace": failure.trace[:2000],
                "exception_type": failure.exception_type,
                "rule_based_category": failure.rule_category,
                "fingerprint": failure.fingerprint,
            }
            for failure in limited_failures
        ],
        "console_log_tail": console_log[-12000:] if console_log else "",
    }


def extract_json(text: str) -> Dict[str, Any]:
    """Safely extract JSON even if a model accidentally returns code fences."""
    cleaned = (text or "").strip()

    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start == -1 or end == -1:
        raise ValueError("AI response did not contain a JSON object")

    return json.loads(cleaned[start:end + 1])


def fallback_analysis(
    failures: List[Failure],
    clusters: List[FailureCluster],
    metadata: BuildMetadata,
) -> Dict[str, Any]:
    """
    Deterministic fallback. This ensures a report is still produced if the
    AI service is unavailable.
    """
    findings = []
    cluster_analysis = []

    for cluster in clusters:
        representative = cluster.failures[0]
        severity = "High" if cluster.blast_radius == "High" else (
            "Medium" if cluster.affected_tests > 1 else "Low"
        )

        cluster_analysis.append({
            "cluster_id": cluster.cluster_id,
            "root_cause_hypothesis": (
                f"Recurring {representative.rule_category} failure pattern. "
                "AI analysis was unavailable; root cause requires engineering review."
            ),
            "category": representative.rule_category,
            "severity": severity,
            "confidence": "Low",
            "suggested_owner": "Engineering / QA Automation",
            "suggested_fix": (
                "Review the representative error and stack trace, reproduce the failure, "
                "and validate the affected dependency or application behavior."
            ),
        })

        for failure in cluster.failures:
            findings.append({
                "test_name": failure.name,
                "cluster_id": cluster.cluster_id,
                "category": failure.rule_category,
                "severity": severity,
                "confidence": "Low",
                "root_cause_hypothesis": "Insufficient evidence for a high-confidence root cause.",
                "evidence": failure.message or failure.trace[:300] or "No detailed evidence available.",
                "suggested_fix": "Investigate the exact exception and reproduce the failure.",
                "suggested_owner": "Engineering / QA Automation",
            })

    recommendation = "GO" if not failures else "HOLD"

    return {
        "executive_summary": (
            "AI analysis was unavailable, so this report uses deterministic failure "
            "clustering and requires engineering review for root-cause confirmation."
        ),
        "release_assessment": {
            "recommendation": recommendation,
            "risk_level": "Medium" if failures else "Low",
            "reason": (
                "Release decision is conservative because automated root-cause analysis "
                "could not be completed."
            ),
        },
        "clusters": cluster_analysis,
        "correlation_analysis": (
            f"{len(clusters)} distinct failure cluster(s) were identified from "
            f"{len(failures)} failing/broken test(s)."
        ),
        "remediation_plan": [
            {
                "priority": "P1",
                "action": "Investigate the largest failure cluster first.",
                "why": "It has the highest potential blast radius.",
                "effort": "Medium",
                "owner": "Engineering / QA Automation",
            }
        ] if failures else [],
        "detailed_findings": findings,
    }


def perform_ai_triage(
    metadata: BuildMetadata,
    metrics: BuildMetrics,
    failures: List[Failure],
    clusters: List[FailureCluster],
    console_log: str,
) -> Dict[str, Any]:
    """Perform evidence-based AI analysis with graceful fallback."""
    if not failures and not console_log:
        return {
            "executive_summary": "All collected tests passed and no failure evidence was detected.",
            "release_assessment": {
                "recommendation": "GO",
                "risk_level": "Low",
                "reason": "No failing or broken tests were identified.",
            },
            "clusters": [],
            "correlation_analysis": "No failure correlations exist because no failures were collected.",
            "remediation_plan": [],
            "detailed_findings": [],
        }

    payload = build_ai_payload(
        metadata, metrics, failures, clusters, console_log
    )

    try:
        client = OpenAI()

        response = client.chat.completions.create(
            model=AI_MODEL,
            temperature=0.1,
            max_tokens=8000,
            messages=[
                {"role": "system", "content": TRIAGE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Analyze the following CI evidence and return the required JSON only:\n\n"
                        + json.dumps(payload, indent=2)
                    ),
                },
            ],
        )

        content = response.choices[0].message.content or ""
        analysis = extract_json(content)
        return analysis

    except Exception as exc:
        print(
            f"Warning: AI analysis unavailable ({exc}). Using deterministic fallback.",
            file=sys.stderr,
        )
        return fallback_analysis(failures, clusters, metadata)


# ============================================================
# 7. RISK ASSESSMENT ENGINE
# ============================================================

def normalize_recommendation(value: str) -> str:
    value = (value or "").upper().replace("-", "_").replace(" ", "_")
    valid = {"GO", "CONDITIONAL_GO", "HOLD", "NO_GO"}
    return value if value in valid else "HOLD"


def assess_risk(
    analysis: Dict[str, Any],
    clusters: List[FailureCluster],
    failures: List[Failure],
) -> Dict[str, Any]:
    """
    Apply deterministic guardrails on top of AI recommendations.
    The AI provides reasoning; this layer ensures the dashboard has
    consistent, explainable risk information.
    """
    release = analysis.get("release_assessment") or {}
    recommendation = normalize_recommendation(release.get("recommendation", "HOLD"))
    risk_level = release.get("risk_level", "Medium").title()

    if risk_level not in SEVERITY_ORDER:
        risk_level = "Medium"

    cluster_data = analysis.get("clusters") or []
    severities = [
        item.get("severity", "Low").title()
        for item in cluster_data
        if item.get("severity", "Low").title() in SEVERITY_ORDER
    ]

    highest_severity = max(
        severities,
        key=lambda value: SEVERITY_ORDER.get(value, 1),
        default="Low",
    )

    largest_cluster = max(
        (cluster.affected_tests for cluster in clusters),
        default=0,
    )

    # Conservative deterministic guardrails.
    if highest_severity == "Critical":
        recommendation = "NO_GO"
        risk_level = "Critical"
    elif highest_severity == "High" and largest_cluster >= 3:
        if recommendation == "GO":
            recommendation = "HOLD"
        risk_level = max(
            [risk_level, "High"],
            key=lambda value: SEVERITY_ORDER.get(value, 1),
        )
    elif failures and recommendation == "GO":
        recommendation = "CONDITIONAL_GO"

    priority_score = (
        SEVERITY_ORDER.get(highest_severity, 1)
        * max(1, largest_cluster)
        * max(1, len(failures))
    )

    return {
        "recommendation": recommendation,
        "risk_level": risk_level,
        "reason": release.get("reason")
        or "Risk assessment based on failure severity and blast radius.",
        "highest_severity": highest_severity,
        "largest_cluster_size": largest_cluster,
        "priority_score": priority_score,
    }


# ============================================================
# 8. MARKDOWN REPORT GENERATOR
# ============================================================

def recommendation_label(value: str) -> str:
    return value.replace("_", " ")


def safe(value: Any, default: str = "N/A") -> str:
    if value is None:
        return default
    value = str(value).strip()
    return value if value else default


def generate_markdown_report(
    metadata: BuildMetadata,
    metrics: BuildMetrics,
    failures: List[Failure],
    clusters: List[FailureCluster],
    analysis: Dict[str, Any],
    risk: Dict[str, Any],
) -> str:
    lines = [
        "# CI Failure Triage Report",
        "",
        f"**Build:** {metadata.build_url or 'N/A'}",
        f"**Job:** {metadata.job_name}",
        f"**Build Number:** {metadata.build_number}",
        f"**Pipeline Status:** {metadata.pipeline_status}",
        f"**Generated:** {metadata.timestamp}",
        "",
        "## Release Assessment",
        "",
        f"**Recommendation:** {recommendation_label(risk['recommendation'])}",
        f"**Risk Level:** {risk['risk_level']}",
        f"**Reason:** {safe(risk['reason'])}",
        "",
        "## Executive Summary",
        "",
        safe(analysis.get("executive_summary")),
        "",
        "## Build Metrics",
        "",
        "| Total | Passed | Failed | Broken | Skipped | Pass Rate |",
        "|---:|---:|---:|---:|---:|---:|",
        (
            f"| {metrics.total} | {metrics.passed} | {metrics.failed} | "
            f"{metrics.broken} | {metrics.skipped} | {metrics.pass_rate}% |"
        ),
        "",
        "## Failure Clusters",
        "",
        "| Cluster | Affected Tests | Blast Radius | Pattern |",
        "|---|---:|---|---|",
    ]

    if clusters:
        for cluster in clusters:
            lines.append(
                f"| {cluster.cluster_id} | {cluster.affected_tests} | "
                f"{cluster.blast_radius} | {cluster.probable_pattern} |"
            )
    else:
        lines.append("| None | 0 | Low | No failures detected |")

    lines.extend([
        "",
        "## Correlation Analysis",
        "",
        safe(analysis.get("correlation_analysis")),
        "",
        "## Prioritized Remediation Plan",
        "",
    ])

    remediation = analysis.get("remediation_plan") or []
    if remediation:
        for item in remediation:
            lines.extend([
                f"### {safe(item.get('priority'))} — {safe(item.get('action'))}",
                f"- **Why:** {safe(item.get('why'))}",
                f"- **Effort:** {safe(item.get('effort'))}",
                f"- **Suggested Owner:** {safe(item.get('owner'))}",
                "",
            ])
    else:
        lines.append("No remediation actions required based on the collected evidence.")
        lines.append("")

    lines.extend([
        "## Detailed Findings",
        "",
    ])

    findings = analysis.get("detailed_findings") or []
    if findings:
        for finding in findings:
            lines.extend([
                f"### {safe(finding.get('test_name'))}",
                f"- **Cluster:** {safe(finding.get('cluster_id'))}",
                f"- **Category:** {safe(finding.get('category'))}",
                f"- **Severity:** {safe(finding.get('severity'))}",
                f"- **Confidence:** {safe(finding.get('confidence'))}",
                f"- **Root Cause Hypothesis:** {safe(finding.get('root_cause_hypothesis'))}",
                f"- **Evidence:** {safe(finding.get('evidence'))}",
                f"- **Suggested Fix:** {safe(finding.get('suggested_fix'))}",
                f"- **Suggested Owner:** {safe(finding.get('suggested_owner'))}",
                "",
            ])
    else:
        lines.append("No detailed failure findings were generated.")

    return "\n".join(lines)


# ============================================================
# 9. HTML DASHBOARD GENERATOR
# ============================================================

def escape(value: Any) -> str:
    return html.escape(safe(value))


def severity_class(value: str) -> str:
    return safe(value, "low").lower().replace(" ", "-")


def recommendation_class(value: str) -> str:
    return safe(value, "hold").lower().replace("_", "-")


def metric_card(label: str, value: Any, css_class: str = "") -> str:
    return f"""
    <div class="metric-card {css_class}">
        <div class="metric-value">{escape(value)}</div>
        <div class="metric-label">{escape(label)}</div>
    </div>
    """


def generate_html_dashboard(
    metadata: BuildMetadata,
    metrics: BuildMetrics,
    failures: List[Failure],
    clusters: List[FailureCluster],
    analysis: Dict[str, Any],
    risk: Dict[str, Any],
) -> str:
    category_counts = Counter(
        safe(item.get("category"), "unknown")
        for item in (analysis.get("detailed_findings") or [])
    )

    if not category_counts:
        category_counts = Counter(f.rule_category for f in failures)

    severity_counts = Counter(
        safe(item.get("severity"), "Low")
        for item in (analysis.get("detailed_findings") or [])
    )

    if not severity_counts and failures:
        severity_counts["Medium"] = len(failures)

    recommendation = recommendation_label(risk["recommendation"])

    metrics_html = "\n".join([
        metric_card("Total Tests", metrics.total, "blue"),
        metric_card("Passed", metrics.passed, "green"),
        metric_card("Failed", metrics.failed, "red"),
        metric_card("Broken", metrics.broken, "red"),
        metric_card("Pass Rate", f"{metrics.pass_rate}%", "blue"),
        metric_card("Failure Clusters", len(clusters), "amber"),
    ])

    cluster_analysis_map = {
        item.get("cluster_id"): item
        for item in (analysis.get("clusters") or [])
    }

    cluster_cards = []
    for cluster in clusters:
        ai_cluster = cluster_analysis_map.get(cluster.cluster_id, {})
        cluster_cards.append(f"""
        <div class="cluster-card">
            <div class="cluster-top">
                <span class="cluster-id">{escape(cluster.cluster_id)}</span>
                <span class="badge blast-{severity_class(cluster.blast_radius)}">
                    Blast Radius: {escape(cluster.blast_radius)}
                </span>
            </div>
            <h3>{escape(cluster.probable_pattern)}</h3>
            <div class="cluster-impact">{cluster.affected_tests} affected test(s)</div>
            <p><strong>Root Cause Hypothesis:</strong> {escape(ai_cluster.get("root_cause_hypothesis", "Pending analysis"))}</p>
            <div class="tag-row">
                <span class="badge severity-{severity_class(ai_cluster.get("severity", "Medium"))}">
                    {escape(ai_cluster.get("severity", "Medium"))}
                </span>
                <span class="badge confidence-{severity_class(ai_cluster.get("confidence", "Low"))}">
                    {escape(ai_cluster.get("confidence", "Low"))} confidence
                </span>
            </div>
        </div>
        """)

    if not cluster_cards:
        cluster_cards.append("""
        <div class="empty-state">
            <div class="empty-icon">✓</div>
            <h3>No Failure Clusters</h3>
            <p>No failing or broken tests were identified in the collected Allure results.</p>
        </div>
        """)

    remediation_items = []
    for item in (analysis.get("remediation_plan") or []):
        remediation_items.append(f"""
        <div class="remediation-item">
            <div class="priority">{escape(item.get("priority"))}</div>
            <div class="remediation-content">
                <h3>{escape(item.get("action"))}</h3>
                <p>{escape(item.get("why"))}</p>
                <div class="meta">
                    <span>Effort: <strong>{escape(item.get("effort"))}</strong></span>
                    <span>Owner: <strong>{escape(item.get("owner"))}</strong></span>
                </div>
            </div>
        </div>
        """)

    if not remediation_items:
        remediation_items.append("<p class='muted'>No remediation actions required.</p>")

    findings_html = []
    for index, finding in enumerate(analysis.get("detailed_findings") or [], start=1):
        finding_id = f"finding-{index}"
        findings_html.append(f"""
        <details class="finding">
            <summary>
                <div>
                    <span class="finding-name">{escape(finding.get("test_name"))}</span>
                    <div class="finding-meta">
                        <span>{escape(finding.get("cluster_id"))}</span>
                        <span>{escape(finding.get("category"))}</span>
                    </div>
                </div>
                <div class="tag-row">
                    <span class="badge severity-{severity_class(finding.get("severity"))}">
                        {escape(finding.get("severity"))}
                    </span>
                    <span class="badge confidence-{severity_class(finding.get("confidence"))}">
                        {escape(finding.get("confidence"))}
                    </span>
                </div>
            </summary>
            <div class="finding-body">
                <div class="finding-grid">
                    <div>
                        <h4>Root Cause Hypothesis</h4>
                        <p>{escape(finding.get("root_cause_hypothesis"))}</p>
                    </div>
                    <div>
                        <h4>Evidence</h4>
                        <p class="evidence">{escape(finding.get("evidence"))}</p>
                    </div>
                    <div>
                        <h4>Suggested Fix</h4>
                        <p>{escape(finding.get("suggested_fix"))}</p>
                    </div>
                    <div>
                        <h4>Suggested Owner</h4>
                        <p>{escape(finding.get("suggested_owner"))}</p>
                    </div>
                </div>
            </div>
        </details>
        """)

    if not findings_html:
        findings_html.append("<p class='muted'>No detailed findings available.</p>")

    category_rows = "".join(
        f"<div class='distribution-row'><span>{escape(category)}</span>"
        f"<div class='bar-track'><div class='bar' style='width:{min(100, count / max(category_counts.values(), default=1) * 100):.0f}%'></div></div>"
        f"<strong>{count}</strong></div>"
        for category, count in category_counts.most_common()
    ) or "<p class='muted'>No category distribution available.</p>"

    severity_rows = "".join(
        f"<div class='distribution-row'><span>{escape(severity)}</span>"
        f"<div class='bar-track'><div class='bar severity-bar {severity_class(severity)}' style='width:{min(100, count / max(severity_counts.values(), default=1) * 100):.0f}%'></div></div>"
        f"<strong>{count}</strong></div>"
        for severity, count in severity_counts.most_common()
    ) or "<p class='muted'>No severity distribution available.</p>"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CI Failure Triage Report</title>
<style>
:root {{
    --bg: #f4f7fb;
    --surface: #ffffff;
    --surface-alt: #f8fafc;
    --text: #172033;
    --muted: #64748b;
    --border: #e2e8f0;
    --blue: #2563eb;
    --green: #16a34a;
    --amber: #d97706;
    --red: #dc2626;
    --critical: #991b1b;
    --shadow: 0 10px 30px rgba(15, 23, 42, .07);
}}

* {{ box-sizing: border-box; }}

body {{
    margin: 0;
    background: var(--bg);
    color: var(--text);
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    line-height: 1.55;
}}

.container {{
    max-width: 1240px;
    margin: 0 auto;
    padding: 32px 22px 64px;
}}

.hero {{
    background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 55%, #2563eb 100%);
    color: white;
    border-radius: 20px;
    padding: 34px;
    box-shadow: 0 16px 40px rgba(30, 58, 138, .22);
    margin-bottom: 24px;
}}

.hero-top {{
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 20px;
    flex-wrap: wrap;
}}

.eyebrow {{
    text-transform: uppercase;
    letter-spacing: .12em;
    font-size: 11px;
    font-weight: 800;
    opacity: .72;
}}

h1 {{
    margin: 6px 0 8px;
    font-size: clamp(28px, 4vw, 40px);
    letter-spacing: -.03em;
}}

.hero-sub {{
    margin: 0;
    opacity: .8;
}}

.build-meta {{
    margin-top: 20px;
    display: flex;
    flex-wrap: wrap;
    gap: 8px 20px;
    font-size: 13px;
    opacity: .88;
}}

.build-meta a {{
    color: #dbeafe;
    word-break: break-all;
}}

.status-pill {{
    padding: 10px 16px;
    border-radius: 999px;
    font-size: 13px;
    font-weight: 800;
    background: rgba(255,255,255,.15);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255,255,255,.2);
}}

.release-banner {{
    border-radius: 16px;
    padding: 22px 24px;
    margin-bottom: 24px;
    display: flex;
    align-items: flex-start;
    gap: 16px;
    border: 1px solid var(--border);
    background: var(--surface);
    box-shadow: var(--shadow);
}}

.release-banner.no-go, .release-banner.hold {{
    border-left: 6px solid var(--red);
}}

.release-banner.go {{
    border-left: 6px solid var(--green);
}}

.release-banner.conditional-go {{
    border-left: 6px solid var(--amber);
}}

.release-icon {{
    font-size: 28px;
    line-height: 1;
}}

.release-banner h2 {{
    margin: 0 0 4px;
    font-size: 20px;
}}

.release-banner p {{
    margin: 0;
    color: var(--muted);
}}

.section {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 26px;
    margin-bottom: 22px;
    box-shadow: var(--shadow);
}}

.section-header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    margin-bottom: 18px;
}}

.section h2 {{
    margin: 0;
    font-size: 19px;
    letter-spacing: -.01em;
}}

.section p {{
    color: #475569;
}}

.metrics-grid {{
    display: grid;
    grid-template-columns: repeat(6, minmax(130px, 1fr));
    gap: 14px;
    margin-bottom: 22px;
}}

.metric-card {{
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 18px;
    background: linear-gradient(180deg, #fff, #f8fafc);
}}

.metric-value {{
    font-size: 28px;
    font-weight: 800;
    letter-spacing: -.03em;
}}

.metric-label {{
    color: var(--muted);
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: .08em;
    margin-top: 5px;
    font-weight: 700;
}}

.metric-card.green .metric-value {{ color: var(--green); }}
.metric-card.red .metric-value {{ color: var(--red); }}
.metric-card.blue .metric-value {{ color: var(--blue); }}
.metric-card.amber .metric-value {{ color: var(--amber); }}

.two-column {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 22px;
}}

.distribution-row {{
    display: grid;
    grid-template-columns: 135px 1fr 36px;
    gap: 12px;
    align-items: center;
    margin: 13px 0;
    font-size: 13px;
}}

.bar-track {{
    height: 10px;
    border-radius: 999px;
    background: #edf2f7;
    overflow: hidden;
}}

.bar {{
    height: 100%;
    background: var(--blue);
    border-radius: inherit;
}}

.bar.critical {{ background: var(--critical); }}
.bar.high {{ background: var(--red); }}
.bar.medium {{ background: var(--amber); }}
.bar.low {{ background: var(--green); }}

.cluster-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(310px, 1fr));
    gap: 16px;
}}

.cluster-card {{
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 20px;
    background: var(--surface-alt);
}}

.cluster-top {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 8px;
}}

.cluster-id {{
    font-weight: 800;
    color: var(--blue);
    font-size: 13px;
}}

.cluster-card h3 {{
    margin: 14px 0 4px;
    font-size: 16px;
}}

.cluster-impact {{
    color: var(--muted);
    font-size: 13px;
    margin-bottom: 14px;
}}

.cluster-card p {{
    font-size: 13px;
    margin-bottom: 14px;
}}

.tag-row {{
    display: flex;
    flex-wrap: wrap;
    gap: 7px;
}}

.badge {{
    display: inline-flex;
    align-items: center;
    padding: 5px 9px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 800;
}}

.severity-critical {{ background: #fee2e2; color: #991b1b; }}
.severity-high {{ background: #fee2e2; color: #b91c1c; }}
.severity-medium {{ background: #fef3c7; color: #92400e; }}
.severity-low {{ background: #dcfce7; color: #166534; }}

.confidence-high {{ background: #dbeafe; color: #1d4ed8; }}
.confidence-medium {{ background: #ede9fe; color: #6d28d9; }}
.confidence-low {{ background: #f1f5f9; color: #475569; }}

.blast-high {{ background: #fee2e2; color: #b91c1c; }}
.blast-medium {{ background: #fef3c7; color: #92400e; }}
.blast-low {{ background: #dcfce7; color: #166534; }}

.remediation-item {{
    display: grid;
    grid-template-columns: 58px 1fr;
    gap: 16px;
    padding: 18px 0;
    border-bottom: 1px solid var(--border);
}}

.remediation-item:last-child {{ border-bottom: none; }}

.priority {{
    width: 46px;
    height: 46px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 12px;
    background: #eff6ff;
    color: var(--blue);
    font-weight: 900;
}}

.remediation-content h3 {{
    margin: 0 0 5px;
    font-size: 15px;
}}

.remediation-content p {{
    margin: 0 0 9px;
    font-size: 13px;
}}

.meta {{
    display: flex;
    gap: 20px;
    flex-wrap: wrap;
    color: var(--muted);
    font-size: 12px;
}}

.finding {{
    border: 1px solid var(--border);
    border-radius: 12px;
    margin-bottom: 12px;
    overflow: hidden;
}}

.finding summary {{
    cursor: pointer;
    padding: 17px 18px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 18px;
    list-style: none;
}}

.finding summary::-webkit-details-marker {{ display: none; }}

.finding-name {{
    font-weight: 750;
    display: block;
    word-break: break-word;
}}

.finding-meta {{
    display: flex;
    gap: 10px;
    color: var(--muted);
    font-size: 12px;
    margin-top: 4px;
}}

.finding-body {{
    border-top: 1px solid var(--border);
    padding: 20px;
    background: var(--surface-alt);
}}

.finding-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 18px;
}}

.finding-grid h4 {{
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: .07em;
    color: var(--muted);
    margin: 0 0 6px;
}}

.finding-grid p {{
    margin: 0;
    font-size: 13px;
    white-space: pre-wrap;
    word-break: break-word;
}}

.evidence {{
    background: #fff;
    border: 1px solid var(--border);
    padding: 10px;
    border-radius: 8px;
    font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
}}

.empty-state {{
    text-align: center;
    padding: 34px;
    color: var(--muted);
}}

.empty-icon {{
    color: var(--green);
    font-size: 38px;
    font-weight: 900;
}}

.muted {{ color: var(--muted); }}

.footer {{
    text-align: center;
    color: var(--muted);
    font-size: 12px;
    padding-top: 8px;
}}

@media (max-width: 950px) {{
    .metrics-grid {{ grid-template-columns: repeat(3, 1fr); }}
}}

@media (max-width: 700px) {{
    .container {{ padding: 16px 12px 40px; }}
    .hero {{ padding: 24px; }}
    .section {{ padding: 20px; }}
    .metrics-grid {{ grid-template-columns: repeat(2, 1fr); }}
    .two-column, .finding-grid {{ grid-template-columns: 1fr; }}
    .finding summary {{ align-items: flex-start; flex-direction: column; }}
    .distribution-row {{ grid-template-columns: 110px 1fr 30px; }}
}}

@media print {{
    body {{ background: white; }}
    .container {{ max-width: none; padding: 0; }}
    .hero, .section, .release-banner {{ box-shadow: none; }}
    .finding {{ break-inside: avoid; }}
    details {{ display: block; }}
    details .finding-body {{ display: block !important; }}
}}
</style>
</head>
<body>
<div class="container">

    <header class="hero">
        <div class="hero-top">
            <div>
                <div class="eyebrow">Automated CI Intelligence</div>
                <h1>CI Failure Triage Report</h1>
                <p class="hero-sub">Evidence-based failure correlation, AI analysis and release risk assessment</p>
            </div>
            <div class="status-pill">Pipeline: {escape(metadata.pipeline_status)}</div>
        </div>

        <div class="build-meta">
            <span><strong>Job:</strong> {escape(metadata.job_name)}</span>
            <span><strong>Build:</strong> {escape(metadata.build_number)}</span>
            <span><strong>Generated:</strong> {escape(metadata.timestamp)}</span>
            {"<a href='" + html.escape(metadata.build_url, quote=True) + "'>Open Jenkins Build</a>" if metadata.build_url else ""}
        </div>
    </header>

    <section class="release-banner {recommendation_class(risk["recommendation"])}">
        <div class="release-icon">
            {"✓" if risk["recommendation"] == "GO" else "⚠" if risk["recommendation"] == "CONDITIONAL_GO" else "✕"}
        </div>
        <div>
            <div class="eyebrow">Release Recommendation</div>
            <h2>{escape(recommendation)}</h2>
            <p><strong>Risk: {escape(risk["risk_level"])}</strong> — {escape(risk["reason"])}</p>
        </div>
    </section>

    <div class="metrics-grid">
        {metrics_html}
    </div>

    <section class="section">
        <div class="section-header">
            <h2>Executive Summary</h2>
        </div>
        <p>{escape(analysis.get("executive_summary"))}</p>
    </section>

    <div class="two-column">
        <section class="section">
            <div class="section-header"><h2>Failure Categories</h2></div>
            {category_rows}
        </section>

        <section class="section">
            <div class="section-header"><h2>Severity Distribution</h2></div>
            {severity_rows}
        </section>
    </div>

    <section class="section">
        <div class="section-header">
            <h2>Failure Clusters & Blast Radius</h2>
            <span class="muted">{len(clusters)} cluster(s)</span>
        </div>
        <div class="cluster-grid">
            {"".join(cluster_cards)}
        </div>
    </section>

    <section class="section">
        <div class="section-header"><h2>Correlation Analysis</h2></div>
        <p>{escape(analysis.get("correlation_analysis"))}</p>
    </section>

    <section class="section">
        <div class="section-header"><h2>Prioritized Remediation Plan</h2></div>
        {"".join(remediation_items)}
    </section>

    <section class="section">
        <div class="section-header">
            <h2>Detailed Findings</h2>
            <span class="muted">{len(analysis.get("detailed_findings") or [])} finding(s)</span>
        </div>
        {"".join(findings_html)}
    </section>

    <div class="footer">
        Generated automatically by the AI-Powered CI Failure Triage Engine
    </div>

</div>
</body>
</html>
"""


def write_reports(
    markdown_content: str,
    html_content: str,
    markdown_path: str,
    html_path: str,
) -> None:
    Path(markdown_path).parent.mkdir(parents=True, exist_ok=True)
    Path(html_path).parent.mkdir(parents=True, exist_ok=True)

    Path(markdown_path).write_text(markdown_content, encoding="utf-8")
    Path(html_path).write_text(html_content, encoding="utf-8")


# ============================================================
# 10. ORCHESTRATION / MAIN
# ============================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="AI-powered CI failure triage engine"
    )

    parser.add_argument(
        "--report-dir",
        required=True,
        help="Directory containing Allure *-result.json files",
    )
    parser.add_argument(
        "--build-url",
        default=os.getenv("BUILD_URL", ""),
        help="Jenkins build URL (defaults to BUILD_URL environment variable)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path for Markdown report",
    )
    parser.add_argument(
        "--html-output",
        required=True,
        help="Output path for HTML dashboard",
    )
    parser.add_argument(
        "--no-console-log",
        action="store_true",
        help="Skip Jenkins console log collection",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    report_dir = Path(args.report_dir)

    if not report_dir.exists():
        print(
            f"Error: report directory not found: {report_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    print("=" * 62)
    print("AI-POWERED CI FAILURE TRIAGE ENGINE")
    print("=" * 62)

    # --------------------------------------------------------
    # STEP 1: DATA COLLECTION
    # --------------------------------------------------------
    print("[1/6] Collecting CI evidence...")

    allure_results = load_allure_results(report_dir)
    metrics, failures = summarize_allure_results(allure_results)

    console_log = ""
    if not args.no_console_log:
        console_log = fetch_console_log(args.build_url)

    metadata = collect_build_metadata(args.build_url, console_log)

    print(
        f"      Tests={metrics.total}, Passed={metrics.passed}, "
        f"Failed={metrics.failed}, Broken={metrics.broken}"
    )

    # --------------------------------------------------------
    # STEP 2: FAILURE PREPROCESSING
    # --------------------------------------------------------
    print("[2/6] Preprocessing failures...")
    processed_failures = preprocess_failures(failures)

    # --------------------------------------------------------
    # STEP 3: CORRELATION ENGINE
    # --------------------------------------------------------
    print("[3/6] Correlating failures...")
    clusters = cluster_failures(processed_failures)
    print(f"      Identified {len(clusters)} failure cluster(s)")

    # --------------------------------------------------------
    # STEP 4: AI TRIAGE ENGINE
    # --------------------------------------------------------
    print("[4/6] Performing AI triage analysis...")
    analysis = perform_ai_triage(
        metadata=metadata,
        metrics=metrics,
        failures=processed_failures,
        clusters=clusters,
        console_log=console_log,
    )

    # --------------------------------------------------------
    # STEP 5: RISK ASSESSMENT
    # --------------------------------------------------------
    print("[5/6] Assessing release risk...")
    risk = assess_risk(
        analysis=analysis,
        clusters=clusters,
        failures=processed_failures,
    )

    # --------------------------------------------------------
    # STEP 6: REPORT GENERATION
    # --------------------------------------------------------
    print("[6/6] Generating reports...")

    markdown_report = generate_markdown_report(
        metadata, metrics, processed_failures, clusters, analysis, risk
    )

    html_report = generate_html_dashboard(
        metadata, metrics, processed_failures, clusters, analysis, risk
    )

    write_reports(
        markdown_content=markdown_report,
        html_content=html_report,
        markdown_path=args.output,
        html_path=args.html_output,
    )

    print("=" * 62)
    print("TRIAGE COMPLETED")
    print(f"Release Recommendation : {recommendation_label(risk['recommendation'])}")
    print(f"Risk Level             : {risk['risk_level']}")
    print(f"Markdown Report        : {args.output}")
    print(f"HTML Dashboard         : {args.html_output}")
    print("=" * 62)


if __name__ == "__main__":
    main()
