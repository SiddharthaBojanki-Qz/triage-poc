#!/usr/bin/env python3
"""
AI-Powered CI Failure Triage Engine
===================================

Single-file implementation for Jenkins + Allure CI failure triage.

Pipeline:
    Jenkins Build
        -> Data Collection
        -> Failure Preprocessing
        -> Correlation Engine
        -> AI Triage Engine
        -> Risk Assessment
        -> Interactive Report Generator

Report UX:
    Overview -> Category -> Failed Tests -> Test Detail

The HTML report is intentionally concise on first load. Detailed technical
information is progressively disclosed only when the user drills into a
category and then a failed test.

Final failure taxonomy (and ONLY these values may appear in the report):
    1. Application Defect
    2. Environment Issue
    3. Test Data Issue
    4. Script Defect
    5. Unknown

Environment variables:
    OPENAI_API_KEY        Required for AI analysis
    OPENAI_MODEL          Optional; default: gpt-4o
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
MAX_TRACE_LENGTH = 6000
MAX_CONSOLE_LINES = 300
MAX_FAILURES_FOR_AI = 100
MAX_CONSOLE_FOR_AI = 12000

CATEGORIES = [
    "Application Defect",
    "Environment Issue",
    "Test Data Issue",
    "Script Defect",
    "Unknown",
]

SEVERITIES = ["Critical", "High", "Medium", "Low"]
CONFIDENCES = ["High", "Medium", "Low"]
SEVERITY_ORDER = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1}
CONFIDENCE_ORDER = {"High": 3, "Medium": 2, "Low": 1}

# Safety net for older/internal labels that may still be returned by a model
# or exist in a previously developed rule set. These values never reach the UI.
LEGACY_CATEGORY_MAP = {
    "logic-bug": "Application Defect",
    "null-safety": "Application Defect",
    "bounds-check": "Application Defect",
    "concurrency": "Application Defect",
    "api-contract": "Application Defect",
    "assertion": "Application Defect",
    "business-logic": "Application Defect",
    "application": "Application Defect",
    "product-defect": "Application Defect",
    "timeout": "Environment Issue",
    "environment": "Environment Issue",
    "infrastructure": "Environment Issue",
    "build": "Environment Issue",
    "dependency": "Environment Issue",
    "network": "Environment Issue",
    "ui-locator": "Script Defect",
    "locator": "Script Defect",
    "selector": "Script Defect",
    "automation": "Script Defect",
    "flaky": "Script Defect",
    "test-data": "Test Data Issue",
    "test_data": "Test Data Issue",
    "data": "Test Data Issue",
    "unknown": "Unknown",
}


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
    suite: str = "Unknown"
    feature: str = "Unknown"
    duration_ms: int = 0
    test_id: str = ""
    history_id: str = ""
    exception_type: str = "Unknown"
    normalized_signature: str = ""
    fingerprint: str = ""
    rule_category: str = "Unknown"
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
# 3. NORMALIZATION / SAFETY HELPERS
# ============================================================

def safe(value: Any, default: str = "N/A") -> str:
    if value is None:
        return default
    value = str(value).strip()
    return value if value else default


def normalize_category(value: Any) -> str:
    """Return one of the five approved business categories, always."""
    raw = safe(value, "Unknown").strip()
    if raw in CATEGORIES:
        return raw

    key = raw.lower().replace(" ", "-").replace("_", "-")
    if key in LEGACY_CATEGORY_MAP:
        return LEGACY_CATEGORY_MAP[key]

    # A few tolerant aliases. Anything else deliberately becomes Unknown.
    aliases = {
        "application defect": "Application Defect",
        "environment issue": "Environment Issue",
        "test data issue": "Test Data Issue",
        "script defect": "Script Defect",
    }
    return aliases.get(raw.lower(), "Unknown")


def normalize_severity(value: Any, default: str = "Medium") -> str:
    value = safe(value, default).title()
    return value if value in SEVERITIES else default


def normalize_confidence(value: Any, default: str = "Low") -> str:
    value = safe(value, default).title()
    return value if value in CONFIDENCES else default


def compact_text(value: str, limit: int) -> str:
    value = safe(value, "")
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 3)] + "..."


def html_escape(value: Any, default: str = "N/A") -> str:
    return html.escape(safe(value, default), quote=True)


def js_json(value: Any) -> str:
    """Serialize data safely for embedding inside a script tag."""
    return json.dumps(value, ensure_ascii=False).replace("</", "<\\/")


# ============================================================
# 4. DATA COLLECTION LAYER
# ============================================================

def load_allure_results(report_dir: Path) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    if not report_dir.exists():
        return results

    for file_path in sorted(report_dir.glob("*-result.json")):
        try:
            results.append(json.loads(file_path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
            print(
                f"Warning: skipping unreadable Allure result {file_path.name}: {exc}",
                file=sys.stderr,
            )
    return results


def labels_map(result: Dict[str, Any]) -> Dict[str, List[str]]:
    output: Dict[str, List[str]] = defaultdict(list)
    for label in result.get("labels") or []:
        if label.get("name") and label.get("value") is not None:
            output[str(label["name"])].append(str(label["value"]))
    return output


def first_label(labels: Dict[str, List[str]], *names: str) -> str:
    for name in names:
        values = labels.get(name) or []
        if values:
            return values[0]
    return "Unknown"


def extract_module(result: Dict[str, Any]) -> str:
    labels = labels_map(result)
    package = first_label(labels, "package")
    if package != "Unknown" and "." in package:
        return package.rsplit(".", 1)[0]
    return first_label(labels, "testClass", "parentSuite", "suite")


def extract_suite(result: Dict[str, Any]) -> str:
    labels = labels_map(result)
    return first_label(labels, "suite", "parentSuite", "subSuite")


def extract_feature(result: Dict[str, Any]) -> str:
    labels = labels_map(result)
    return first_label(labels, "feature", "epic", "story")


def summarize_allure_results(
    results: List[Dict[str, Any]],
) -> tuple[BuildMetrics, List[Failure]]:
    metrics = BuildMetrics()
    failures: List[Failure] = []

    for result in results:
        status = str(result.get("status") or "").lower()
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

        if status not in {"failed", "broken"}:
            continue

        details = result.get("statusDetails") or {}
        labels = labels_map(result)
        failures.append(
            Failure(
                name=str(result.get("fullName") or result.get("name") or "Unnamed Test"),
                status=status,
                message=str(details.get("message") or ""),
                trace=str(details.get("trace") or "")[:MAX_TRACE_LENGTH],
                module=extract_module(result),
                suite=extract_suite(result),
                feature=extract_feature(result),
                duration_ms=int(result.get("duration") or 0),
                test_id=str(result.get("uuid") or ""),
                history_id=str(result.get("historyId") or ""),
            )
        )

    return metrics, failures


def fetch_console_log(build_url: str, max_lines: int = MAX_CONSOLE_LINES) -> str:
    if not build_url:
        return ""

    request = urllib.request.Request(build_url.rstrip("/") + "/consoleText")
    user = os.getenv("JENKINS_USER")
    token = os.getenv("JENKINS_API_TOKEN")

    if user and token:
        credentials = base64.b64encode(f"{user}:{token}".encode()).decode()
        request.add_header("Authorization", f"Basic {credentials}")

    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            text = response.read().decode("utf-8", errors="replace")
        return "\n".join(text.splitlines()[-max_lines:])
    except Exception as exc:
        print(f"Warning: unable to fetch Jenkins console log: {exc}", file=sys.stderr)
        return ""


def collect_build_metadata(build_url: str, console_log: str) -> BuildMetadata:
    env_status = os.getenv("BUILD_RESULT", "").upper()
    console_upper = console_log.upper()
    status = env_status or "UNKNOWN"

    if "BUILD FAILURE" in console_upper:
        status = "FAILURE"
    elif "BUILD SUCCESS" in console_upper:
        status = "SUCCESS"

    return BuildMetadata(
        build_url=build_url,
        build_number=os.getenv("BUILD_NUMBER", "N/A"),
        job_name=os.getenv("JOB_NAME", "N/A"),
        pipeline_status=status,
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    )


# ============================================================
# 5. FAILURE PREPROCESSING LAYER
# ============================================================

def extract_exception_type(message: str, trace: str) -> str:
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
    if not text:
        return ""
    normalized = text.lower()
    normalized = re.sub(r"https?://[^\s]+", "<url>", normalized)
    normalized = re.sub(
        r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
        "<uuid>", normalized, flags=re.IGNORECASE,
    )
    normalized = re.sub(
        r"\b\d{4}-\d{2}-\d{2}[t\s]\d{2}:\d{2}:\d{2}(?:\.\d+)?z?\b",
        "<timestamp>", normalized,
    )
    normalized = re.sub(r"0x[0-9a-f]+", "<hex>", normalized)
    normalized = re.sub(r":\d+\)", ":<line>)", normalized)
    normalized = re.sub(r"\b\d+\b", "<num>", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def generate_fingerprint(message: str, trace: str, exception_type: str) -> tuple[str, str]:
    primary = message.strip() or trace.strip() or "no failure message"
    normalized = normalize_failure_text(primary)
    if exception_type and exception_type != "Unknown":
        normalized = f"{exception_type}|{normalized}"
    normalized = normalized[:1500]
    fingerprint = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
    return normalized, fingerprint


def classify_failure_rules(message: str, trace: str, exception_type: str) -> str:
    """Conservative deterministic classifier used as AI evidence."""
    source = f"{message} {trace} {exception_type}".lower()

    test_data_keywords = [
        "test data", "testdata", "fixture", "dataset", "seed data",
        "missing data", "invalid data", "duplicate data", "duplicate key",
        "data not found", "record not found", "no data available",
        "invalid input data", "data setup", "precondition data",
        "unique constraint", "foreign key constraint", "expected record not found",
    ]
    environment_keywords = [
        "connection refused", "unknown host", "unknownhostexception", "dns",
        "certificate", "ssl", "tls", "unable to connect", "service unavailable",
        "agent offline", "workspace", "disk space", "out of memory",
        "outofmemoryerror", "docker", "kubernetes", "node lost",
        "compilation failure", "could not resolve dependencies", "dependency resolution",
        "network is unreachable", "connection reset", "sockettimeout", "read timed out",
        "connection timed out", "host unreachable", "502 bad gateway", "503 service unavailable",
    ]
    script_keywords = [
        "nosuchelementexception", "elementnotfound", "unable to locate",
        "locator", "selector", "staleelementreference", "webelement",
        "invalid selector", "element click intercepted", "test script",
        "automation script", "page object", "assertion in test", "test automation",
    ]
    application_keywords = [
        "nullpointerexception", "cannot invoke", " is null", "none type",
        "indexoutofboundsexception", "arrayindexoutofboundsexception",
        "stringindexoutofboundsexception", "concurrentmodificationexception",
        "deadlock", "race condition", "unexpected status", "http status",
        "status code", "schema validation", "response body", "contract violation",
        "business rule", "internal server error", "500 internal", "application error",
        "incorrect response", "unexpected response", "wrong business result",
    ]

    if any(k in source for k in test_data_keywords):
        return "Test Data Issue"
    if any(k in source for k in environment_keywords):
        return "Environment Issue"
    if any(k in source for k in script_keywords):
        return "Script Defect"
    if any(k in source for k in application_keywords):
        return "Application Defect"
    return "Unknown"


def preprocess_failures(failures: List[Failure]) -> List[Failure]:
    fingerprint_counts = Counter()
    for failure in failures:
        failure.exception_type = extract_exception_type(failure.message, failure.trace)
        signature, fingerprint = generate_fingerprint(
            failure.message, failure.trace, failure.exception_type
        )
        failure.normalized_signature = signature
        failure.fingerprint = fingerprint
        failure.rule_category = normalize_category(
            classify_failure_rules(failure.message, failure.trace, failure.exception_type)
        )
        fingerprint_counts[fingerprint] += 1

    for failure in failures:
        failure.duplicate_count = fingerprint_counts[failure.fingerprint]
    return failures


# ============================================================
# 6. CORRELATION ENGINE
# ============================================================

def determine_blast_radius(count: int, total_failures: int) -> str:
    if count >= 10 or (total_failures and count / total_failures >= 0.5):
        return "High"
    if count >= 3:
        return "Medium"
    return "Low"


def detect_common_pattern(failures: List[Failure]) -> str:
    exceptions = Counter(
        f.exception_type for f in failures if f.exception_type != "Unknown"
    )
    exception = exceptions.most_common(1)[0][0] if exceptions else ""
    category = Counter(f.rule_category for f in failures).most_common(1)
    category_name = category[0][0] if category else "Unknown"
    if exception:
        return f"Recurring {exception} pattern"
    return f"Shared {category_name} failure pattern"


def cluster_failures(failures: List[Failure]) -> List[FailureCluster]:
    grouped: Dict[str, List[Failure]] = defaultdict(list)
    for failure in failures:
        grouped[failure.fingerprint].append(failure)

    clusters: List[FailureCluster] = []
    ordered = sorted(grouped.items(), key=lambda item: len(item[1]), reverse=True)
    for index, (fingerprint, grouped_failures) in enumerate(ordered, start=1):
        representative = grouped_failures[0]
        clusters.append(
            FailureCluster(
                cluster_id=f"CL-{index:02d}",
                fingerprint=fingerprint,
                failures=grouped_failures,
                affected_tests=len(grouped_failures),
                blast_radius=determine_blast_radius(len(grouped_failures), len(failures)),
                representative_error=representative.message or representative.trace[:400],
                probable_pattern=detect_common_pattern(grouped_failures),
            )
        )
    return clusters


def console_pipeline_signals(console_log: str) -> List[str]:
    if not console_log:
        return []
    patterns = [
        (r"COMPILATION FAILURE", "Compilation failure detected"),
        (r"Could not resolve dependencies", "Dependency resolution failure detected"),
        (r"OutOfMemoryError", "Out-of-memory signal detected"),
        (r"Connection refused", "Connectivity failure signal detected"),
        (r"BUILD FAILURE", "Build failure marker detected"),
        (r"ERROR.*Exception", "Unhandled exception signal detected"),
    ]
    return [label for pattern, label in patterns if re.search(pattern, console_log, re.I)]


# ============================================================
# 7. AI TRIAGE ENGINE
# ============================================================

TRIAGE_SYSTEM_PROMPT = """
You are a Principal QA Engineer performing evidence-based CI failure triage.

Your job is NOT to write a long report. Your job is to produce concise, structured
triage findings for each failed/broken test so a UI can progressively disclose detail.

STRICT RULES:
1. Return JSON only.
2. Never invent files, line numbers, APIs, deployments, incidents, test steps, or facts.
3. A root cause is a hypothesis unless the supplied evidence directly proves it.
4. If evidence is insufficient, say "Insufficient evidence" and use Unknown when appropriate.
5. Use ONLY these five categories, with exact spelling:
   - Application Defect
   - Environment Issue
   - Test Data Issue
   - Script Defect
   - Unknown
6. Do not use any technical/internal category names such as logic-bug, ui-locator,
   concurrency, timeout, api-contract, null-safety, flaky, infrastructure, etc.
7. Severity: Critical, High, Medium, Low. Base it on likely impact, not merely exception type.
8. Confidence: High, Medium, Low. High requires strong evidence.
9. Suggested fixes must be actionable but must not pretend that unavailable source code was inspected.
10. Evidence must be grounded in the supplied test message, stack trace, metadata, or console evidence.
11. Keep each field concise. Prefer 1-3 sentences per field.
12. Return one finding for every supplied failed/broken test. Preserve the exact test_name.
13. cluster_id must be one of the supplied cluster IDs.
14. If multiple tests share a fingerprint, their findings may share the same cluster_id and
    should reflect the common failure pattern where supported.

JSON schema:
{
  "release_assessment": {
    "recommendation": "GO|CONDITIONAL_GO|HOLD|NO_GO",
    "risk_level": "Critical|High|Medium|Low",
    "reason": "short reason"
  },
  "findings": [
    {
      "test_name": "exact supplied test name",
      "cluster_id": "CL-01",
      "category": "Application Defect|Environment Issue|Test Data Issue|Script Defect|Unknown",
      "severity": "Critical|High|Medium|Low",
      "confidence": "High|Medium|Low",
      "root_cause": "concise evidence-based hypothesis",
      "evidence": "specific evidence from input",
      "suggested_fix": "concise actionable fix",
      "recommended_action": "what the team should do next",
      "suggested_owner": "Application|QA Automation|QA|DevOps|Data|Engineering|Unknown"
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
    limited_failures = failures[:MAX_FAILURES_FOR_AI]
    return {
        "build": asdict(metadata),
        "metrics": asdict(metrics),
        "pipeline_signals": console_pipeline_signals(console_log),
        "clusters": [
            {
                "cluster_id": cluster.cluster_id,
                "affected_tests": cluster.affected_tests,
                "blast_radius": cluster.blast_radius,
                "pattern": cluster.probable_pattern,
                "tests": [f.name for f in cluster.failures[:30]],
            }
            for cluster in clusters
        ],
        "failed_tests": [
            {
                "test_name": f.name,
                "status": f.status,
                "module": f.module,
                "suite": f.suite,
                "feature": f.feature,
                "duration_ms": f.duration_ms,
                "message": compact_text(f.message, 2000),
                "stack_trace": compact_text(f.trace, 4000),
                "exception_type": f.exception_type,
                "rule_based_category": f.rule_category,
                "fingerprint": f.fingerprint,
            }
            for f in limited_failures
        ],
        "console_log_tail": console_log[-MAX_CONSOLE_FOR_AI:] if console_log else "",
    }


def extract_json(text: str) -> Dict[str, Any]:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end < 0:
        raise ValueError("AI response did not contain a JSON object")
    return json.loads(cleaned[start:end + 1])


def default_finding(failure: Failure, cluster: Optional[FailureCluster] = None) -> Dict[str, Any]:
    severity = "High" if cluster and cluster.blast_radius == "High" else (
        "Medium" if cluster and cluster.affected_tests > 1 else "Low"
    )
    return {
        "test_name": failure.name,
        "cluster_id": cluster.cluster_id if cluster else "CL-01",
        "category": normalize_category(failure.rule_category),
        "severity": severity,
        "confidence": "Low",
        "root_cause": "Insufficient evidence to establish a confirmed root cause.",
        "evidence": compact_text(
            failure.message or failure.trace or "No detailed failure evidence was available.",
            900,
        ),
        "suggested_fix": "Reproduce the failure and review the available error evidence before applying a code or environment change.",
        "recommended_action": "Investigate the failure with the owning engineering or QA team.",
        "suggested_owner": "Unknown",
    }


def fallback_analysis(
    failures: List[Failure],
    clusters: List[FailureCluster],
) -> Dict[str, Any]:
    cluster_by_fp = {c.fingerprint: c for c in clusters}
    findings = [default_finding(f, cluster_by_fp.get(f.fingerprint)) for f in failures]
    return {
        "release_assessment": {
            "recommendation": "HOLD" if failures else "GO",
            "risk_level": "High" if failures else "Low",
            "reason": (
                "Automated AI analysis was unavailable; review failure evidence before release."
                if failures else "No failing or broken tests were detected."
            ),
        },
        "findings": findings,
    }


def perform_ai_triage(
    metadata: BuildMetadata,
    metrics: BuildMetrics,
    failures: List[Failure],
    clusters: List[FailureCluster],
    console_log: str,
) -> Dict[str, Any]:
    if not failures:
        return {
            "release_assessment": {
                "recommendation": "GO",
                "risk_level": "Low",
                "reason": "No failing or broken tests were detected in the collected Allure results.",
            },
            "findings": [],
        }

    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY is not set. Using deterministic fallback.", file=sys.stderr)
        return fallback_analysis(failures, clusters)

    payload = build_ai_payload(metadata, metrics, failures, clusters, console_log)

    try:
        client = OpenAI()
        response = client.chat.completions.create(
            model=AI_MODEL,
            temperature=0.1,
            max_tokens=9000,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": TRIAGE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": "Analyze this CI evidence and return the required JSON:\n\n"
                    + json.dumps(payload, ensure_ascii=False, indent=2),
                },
            ],
        )
        content = response.choices[0].message.content or ""
        return extract_json(content)
    except Exception as exc:
        print(f"Warning: AI analysis unavailable ({exc}). Using deterministic fallback.", file=sys.stderr)
        return fallback_analysis(failures, clusters)


# ============================================================
# 8. FINAL ANALYSIS VALIDATION / ENRICHMENT
# ============================================================

def build_validated_findings(
    failures: List[Failure],
    clusters: List[FailureCluster],
    analysis: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Make the UI data complete and enforce the five-category contract."""
    failure_by_name: Dict[str, Failure] = {f.name: f for f in failures}
    cluster_by_id = {c.cluster_id: c for c in clusters}
    ai_by_name: Dict[str, Dict[str, Any]] = {}

    for item in analysis.get("findings") or []:
        if not isinstance(item, dict):
            continue
        name = safe(item.get("test_name"), "")
        if name:
            ai_by_name[name] = item

    validated: List[Dict[str, Any]] = []
    for failure in failures:
        ai = ai_by_name.get(failure.name, {})
        cluster = cluster_by_id.get(safe(ai.get("cluster_id"), ""))
        if cluster is None:
            cluster = next((c for c in clusters if failure in c.failures), None)

        base = default_finding(failure, cluster)
        finding = {
            "test_name": failure.name,
            "test_id": failure.test_id,
            "history_id": failure.history_id,
            "status": failure.status.upper(),
            "module": failure.module,
            "suite": failure.suite,
            "feature": failure.feature,
            "duration_ms": failure.duration_ms,
            "exception_type": failure.exception_type,
            "failure_message": failure.message,
            "stack_trace": failure.trace,
            "fingerprint": failure.fingerprint,
            "duplicate_count": failure.duplicate_count,
            "cluster_id": safe(ai.get("cluster_id"), base["cluster_id"]),
            "category": normalize_category(ai.get("category", base["category"])),
            "severity": normalize_severity(ai.get("severity", base["severity"]), base["severity"]),
            "confidence": normalize_confidence(ai.get("confidence", base["confidence"]), base["confidence"]),
            "root_cause": safe(ai.get("root_cause"), base["root_cause"]),
            "evidence": safe(ai.get("evidence"), base["evidence"]),
            "suggested_fix": safe(ai.get("suggested_fix"), base["suggested_fix"]),
            "recommended_action": safe(ai.get("recommended_action"), base["recommended_action"]),
            "suggested_owner": safe(ai.get("suggested_owner"), base["suggested_owner"]),
        }
        validated.append(finding)

    # Defensive final pass: no arbitrary category can enter the report.
    for finding in validated:
        finding["category"] = normalize_category(finding.get("category"))

    return validated


# ============================================================
# 9. RISK ASSESSMENT
# ============================================================

def normalize_recommendation(value: Any) -> str:
    normalized = safe(value, "HOLD").upper().replace("-", "_").replace(" ", "_")
    return normalized if normalized in {"GO", "CONDITIONAL_GO", "HOLD", "NO_GO"} else "HOLD"


def assess_risk(
    analysis: Dict[str, Any],
    findings: List[Dict[str, Any]],
) -> Dict[str, Any]:
    release = analysis.get("release_assessment") or {}
    recommendation = normalize_recommendation(release.get("recommendation"))

    highest_severity = max(
        (normalize_severity(f.get("severity")) for f in findings),
        key=lambda x: SEVERITY_ORDER.get(x, 1),
        default="Low",
    )
    largest_cluster = max(Counter(f.get("cluster_id") for f in findings).values(), default=0)

    risk_level = normalize_severity(release.get("risk_level"), "Low")
    if SEVERITY_ORDER[highest_severity] > SEVERITY_ORDER[risk_level]:
        risk_level = highest_severity

    # Deterministic safety guardrails. The AI cannot override these into an unsafe GO.
    if highest_severity == "Critical":
        recommendation = "NO_GO"
        risk_level = "Critical"
    elif highest_severity == "High" and largest_cluster >= 3:
        recommendation = "HOLD" if recommendation == "GO" else recommendation
        risk_level = "High"
    elif findings and recommendation == "GO":
        recommendation = "CONDITIONAL_GO"

    return {
        "recommendation": recommendation,
        "risk_level": risk_level,
        "reason": safe(
            release.get("reason"),
            "Release assessment is based on failure severity, confidence and correlated impact.",
        ),
        "highest_severity": highest_severity,
        "affected_failures": len(findings),
        "largest_cluster": largest_cluster,
    }


# ============================================================
# 10. CONCISE MARKDOWN REPORT
# ============================================================

def generate_markdown_report(
    metadata: BuildMetadata,
    metrics: BuildMetrics,
    findings: List[Dict[str, Any]],
    risk: Dict[str, Any],
) -> str:
    category_counts = Counter(f["category"] for f in findings)
    lines = [
        "# CI Failure Triage Report",
        "",
        f"**Job:** {metadata.job_name}",
        f"**Build:** {metadata.build_number}",
        f"**Pipeline:** {metadata.pipeline_status}",
        f"**Generated:** {metadata.timestamp}",
        "",
        f"**Tests:** {metrics.total}  |  **Passed:** {metrics.passed}  |  **Failed:** {metrics.failed + metrics.broken}  |  **Pass Rate:** {metrics.pass_rate}%",
        f"**Release Assessment:** {risk['recommendation']}  |  **Risk:** {risk['risk_level']}",
        "",
        "## Failure Categories",
        "",
    ]
    for category in CATEGORIES:
        lines.append(f"- **{category}:** {category_counts.get(category, 0)}")

    lines.extend(["", "## Failed Tests", ""])
    if not findings:
        lines.append("No failed or broken tests detected.")
    else:
        for category in CATEGORIES:
            category_findings = [f for f in findings if f["category"] == category]
            if not category_findings:
                continue
            lines.extend([f"### {category}", ""])
            for finding in category_findings:
                lines.extend([
                    f"#### {finding['test_name']}",
                    f"- **Severity:** {finding['severity']}",
                    f"- **Confidence:** {finding['confidence']}",
                    f"- **Root Cause:** {finding['root_cause']}",
                    f"- **Suggested Fix:** {finding['suggested_fix']}",
                    "",
                ])
    return "\n".join(lines)


# ============================================================
# 11. INTERACTIVE HTML DASHBOARD
# ============================================================

CSS = r"""
:root {
  --bg:#F7F8FA; --surface:#FFFFFF; --surface-2:#F4F5F7; --text:#172B4D; --text-strong:#0F1F3D;
  --muted:#5E6C84; --muted-2:#7A869A; --border:#DFE1E6; --primary:#0C66E4; --primary-dark:#0747A6;
  --primary-soft:#E9F2FF; --danger:#AE2E24; --danger-soft:#FFEBE9; --warning:#974F0C; --warning-soft:#FFF4E5;
  --success:#216E4E; --success-soft:#DCFFF1; --shadow:0 8px 24px rgba(9,30,66,.06); --shadow-hover:0 12px 28px rgba(9,30,66,.10);
  --radius:14px;
}
*{box-sizing:border-box} html{scroll-behavior:smooth;background:var(--bg)}
body{margin:0;background:var(--bg);color:var(--text);font-family:"Lato",-apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif;line-height:1.52;-webkit-font-smoothing:antialiased}
a{color:inherit}.app{min-height:100vh}
.topbar{background:#fff;border-bottom:1px solid var(--border);position:sticky;top:0;z-index:20}.topbar-inner{max-width:1180px;margin:0 auto;padding:14px 24px;display:flex;align-items:center;justify-content:space-between;gap:20px}
.brand{display:flex;align-items:center;gap:12px}.brand-mark{width:38px;height:38px;border-radius:11px;background:var(--primary);color:#fff;display:grid;place-items:center;font-weight:900}.brand-title{color:var(--text-strong);font-weight:900;font-size:16px}.brand-sub{color:var(--muted);font-size:12px}
.status{padding:7px 12px;border-radius:999px;font-size:12px;font-weight:800;background:var(--surface-2);border:1px solid var(--border)}.status.success{color:var(--success);background:var(--success-soft)}.status.failure{color:var(--danger);background:var(--danger-soft)}
.container{max-width:1180px;margin:0 auto;padding:30px 24px 64px}.hero{display:flex;align-items:flex-end;justify-content:space-between;gap:24px;margin-bottom:20px}
.eyebrow{color:var(--muted);text-transform:uppercase;letter-spacing:.11em;font-size:10px;font-weight:900}h1{margin:4px 0 5px;color:var(--text-strong);font-size:31px;line-height:1.15;font-weight:900;letter-spacing:-.035em}.build-number{color:var(--muted-2);font-weight:700}.hero-meta{color:var(--muted);font-size:13px}.hero-meta a{color:var(--primary);text-decoration:none;font-weight:800}
.release{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:16px 19px;display:flex;align-items:center;justify-content:space-between;gap:18px;box-shadow:var(--shadow);margin-bottom:22px}.release-main{display:flex;align-items:center;gap:13px}.release-icon{width:40px;height:40px;border-radius:12px;display:grid;place-items:center;font-weight:900;background:var(--surface-2)}.release.go{border-left:4px solid var(--success)}.release.conditional-go{border-left:4px solid var(--warning)}.release.hold,.release.no-go{border-left:4px solid var(--danger)}.release-title{color:var(--text-strong);font-weight:900}.release-reason{color:var(--muted);font-size:12px;margin-top:2px}.risk-pill{padding:7px 11px;border-radius:999px;background:var(--surface-2);border:1px solid var(--border);color:var(--text);font-size:12px;font-weight:800;white-space:nowrap}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:24px}.stat{background:var(--surface);border:1px solid var(--border);border-radius:13px;padding:15px 17px;box-shadow:var(--shadow)}.stat-value{color:var(--text-strong);font-size:25px;font-weight:900;letter-spacing:-.035em}.stat-label{color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:.09em;font-weight:800;margin-top:3px}
.section-title{margin:0 0 4px;color:var(--text-strong);font-size:19px;font-weight:900;letter-spacing:-.02em}.section-subtitle{margin:0 0 15px;color:var(--muted);font-size:13px}.category-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:13px}.category-link{display:block;text-decoration:none}
.category-card{position:relative;background:var(--surface);border:1px solid var(--border);border-radius:15px;padding:18px;min-height:142px;box-shadow:var(--shadow);overflow:hidden;transition:transform .16s ease,border-color .16s ease,box-shadow .16s ease,background .16s ease}.category-card::before{content:"";position:absolute;inset:0 auto 0 0;width:4px;background:var(--primary)}.category-link:hover .category-card{transform:translateY(-3px);border-color:#B7C9E8;box-shadow:var(--shadow-hover);background:#FBFDFF}.category-name{color:var(--text-strong);font-weight:800;font-size:14px;min-height:39px;padding-right:18px}.category-count{color:var(--text-strong);font-size:34px;line-height:1;font-weight:900;letter-spacing:-.05em;margin-top:13px}.category-label{color:var(--muted);font-size:11px;margin-top:5px}.category-cta{position:absolute;right:17px;bottom:15px;color:var(--primary);font-size:12px;font-weight:900}
.category-section,.detail{scroll-margin-top:80px;margin-top:26px}.panel{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);box-shadow:var(--shadow);overflow:hidden}.panel-head{padding:18px 20px;border-bottom:1px solid var(--border);display:flex;align-items:center;justify-content:space-between;gap:14px}.panel-title{color:var(--text-strong);font-size:18px;font-weight:900}.panel-count{color:var(--muted);font-size:12px;font-weight:700}.back-link{color:var(--primary);text-decoration:none;font-size:12px;font-weight:800}.back-link:hover{text-decoration:underline}.test-list{padding:8px}.test-link{display:block;text-decoration:none}
.test-row{background:var(--surface);border:1px solid transparent;border-radius:11px;padding:14px 13px;display:flex;align-items:center;justify-content:space-between;gap:16px;transition:background .14s ease,border-color .14s ease}.test-link:hover .test-row{background:var(--surface-2);border-color:var(--border)}.test-main{min-width:0}.test-name{color:var(--text-strong);font-weight:800;font-size:14px;overflow-wrap:anywhere}.test-secondary{color:var(--muted);font-size:12px;margin-top:4px;display:flex;flex-wrap:wrap;gap:9px}.test-side{display:flex;align-items:center;gap:8px;flex-shrink:0}.badge{display:inline-flex;align-items:center;padding:5px 8px;border-radius:999px;font-size:10px;font-weight:900;white-space:nowrap}.sev-critical,.sev-high{color:var(--danger);background:var(--danger-soft)}.sev-medium{color:var(--warning);background:var(--warning-soft)}.sev-low{color:var(--success);background:var(--success-soft)}.conf-high{color:var(--primary-dark);background:var(--primary-soft)}.conf-medium{color:#5E4DB2;background:#F3F0FF}.conf-low{color:var(--muted);background:var(--surface-2)}
.detail-head{padding:20px 21px;border-bottom:1px solid var(--border);background:#FCFDFE}.detail-title{color:var(--text-strong);font-size:22px;line-height:1.25;font-weight:900;letter-spacing:-.025em;overflow-wrap:anywhere;margin-top:10px}.detail-meta{display:flex;flex-wrap:wrap;gap:8px;margin-top:10px}.detail-body{padding:21px}.detail-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}.info-card{border:1px solid var(--border);border-radius:13px;padding:15px;background:var(--surface-2)}.info-card.full{grid-column:1/-1}.info-label{color:var(--muted);text-transform:uppercase;letter-spacing:.07em;font-size:10px;font-weight:900;margin-bottom:6px}.info-value{color:var(--text);font-size:13px;white-space:pre-wrap;overflow-wrap:anywhere}.info-value.evidence,.trace{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:12px}.disclosure{margin-top:14px;border:1px solid var(--border);border-radius:12px;overflow:hidden}.disclosure summary{cursor:pointer;padding:12px 14px;font-weight:800;font-size:12px;color:var(--text-strong);background:#fff}.disclosure summary:hover{background:var(--surface-2)}.disclosure .tech-block{padding:14px;background:var(--surface-2);border-top:1px solid var(--border)}.disclosure pre{margin:10px 0 0;padding:14px;background:#172B4D;color:#F1F5F9;max-height:360px;overflow:auto;white-space:pre-wrap;overflow-wrap:anywhere;border-radius:9px}.empty{padding:38px 20px;text-align:center;color:var(--muted)}.footer{color:var(--muted);text-align:center;font-size:11px;margin-top:28px}
@media (max-width:1000px){.category-grid{grid-template-columns:repeat(3,1fr)}}@media (max-width:700px){.container{padding:20px 13px 44px}.topbar-inner{padding:13px 14px}.hero{align-items:flex-start;flex-direction:column}h1{font-size:27px}.category-grid{grid-template-columns:1fr 1fr}.stats{grid-template-columns:repeat(2,1fr)}.detail-grid{grid-template-columns:1fr}.info-card.full{grid-column:auto}.test-row{align-items:flex-start;flex-direction:column}.test-side{width:100%;justify-content:flex-start}.release{align-items:flex-start;flex-direction:column}}@media (max-width:460px){.category-grid,.stats{grid-template-columns:1fr}}@media print{body{background:#fff}.topbar{position:static}.category-section,.detail{break-inside:avoid}}
"""


def severity_css_class(value: Any) -> str:
    return "sev-" + str(value or "low").strip().lower().replace(" ", "-")


def confidence_css_class(value: Any) -> str:
    return "conf-" + str(value or "low").strip().lower().replace(" ", "-")


def info_card_html(label: str, value: Any, full: bool = False, extra: str = "") -> str:
    classes = "info-card" + (" full" if full else "")
    value_class = "info-value" + ((" " + extra) if extra else "")
    return f'<div class="{classes}"><div class="info-label">{html_escape(label)}</div><div class="{value_class}">{html_escape(value or "Not available")}</div></div>'


def test_info_text(finding: Dict[str, Any]) -> str:
    duration = finding.get("duration_ms")
    duration_text = f"{float(duration)/1000:.2f} s" if duration not in (None, "") else "Not available"
    return (f"Status: {finding.get('status') or 'Not available'}\n"
            f"Module: {finding.get('module') or 'Unknown'}\n"
            f"Suite: {finding.get('suite') or 'Unknown'}\n"
            f"Feature: {finding.get('feature') or 'Unknown'}\n"
            f"Duration: {duration_text}\n"
            f"Cluster: {finding.get('cluster_id') or 'Not available'}\n"
            f"Related failures: {finding.get('duplicate_count') or 1}")


def category_card(category: str, count: int, anchor: str) -> str:
    plural = "s" if count != 1 else ""
    return f'<a class="category-link" href="#{html_escape(anchor)}" aria-label="Open {html_escape(category)} category"><div class="category-card"><div class="category-name">{html_escape(category)}</div><div class="category-count">{count}</div><div class="category-label">failed test{plural}</div><div class="category-cta">View tests →</div></div></a>'


def category_anchor(category: str) -> str:
    return "category-" + re.sub(r"[^a-z0-9]+", "-", category.lower()).strip("-")


def finding_anchor(finding: Dict[str, Any], index: int) -> str:
    digest = hashlib.sha1(f"{finding.get('test_name','')}|{index}".encode("utf-8")).hexdigest()[:10]
    return f"test-{digest}"


def finding_category_map(findings: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped = {category: [] for category in CATEGORIES}
    for finding in findings:
        category = finding.get("category") if finding.get("category") in CATEGORIES else "Unknown"
        grouped[category].append(finding)
    return grouped


def generate_html_dashboard(metadata: BuildMetadata, metrics: BuildMetrics, findings: List[Dict[str, Any]], risk: Dict[str, Any]) -> str:
    grouped = finding_category_map(findings)
    recommendation = risk["recommendation"]
    rec_class = recommendation.lower().replace("_", "-")
    rec_icon = "✓" if recommendation == "GO" else "⚠" if recommendation == "CONDITIONAL_GO" else "!"
    pipeline_class = "success" if metadata.pipeline_status == "SUCCESS" else "failure" if metadata.pipeline_status in {"FAILURE", "FAILED"} else ""
    build_link = f'<a href="{html_escape(metadata.build_url)}" target="_blank" rel="noopener">Open Jenkins build ↗</a>' if metadata.build_url else ""
    category_cards = "\n".join(category_card(c, len(grouped[c]), category_anchor(c)) for c in CATEGORIES)

    category_sections = []
    index = 0
    for category in CATEGORIES:
        items = grouped[category]
        rows = []
        plural = "s" if len(items) != 1 else ""
        for finding in items:
            index += 1
            anchor = finding_anchor(finding, index)
            module_text = finding.get("module") or finding.get("suite") or "Unknown"
            related = f'<span>•</span><span>{finding["duplicate_count"]} related</span>' if finding.get("duplicate_count", 1) > 1 else ""
            rows.append(f'<a class="test-link" href="#{anchor}"><div class="test-row"><div class="test-main"><div class="test-name">{html_escape(finding.get("test_name"))}</div><div class="test-secondary"><span>{html_escape(module_text)}</span><span>•</span><span>{html_escape(finding.get("exception_type") or "Failure")}</span>{related}</div></div><div class="test-side"><span class="badge {severity_css_class(finding.get("severity"))}">{html_escape(finding.get("severity"))}</span><span class="badge {confidence_css_class(finding.get("confidence"))}">{html_escape(finding.get("confidence"))} confidence</span></div></div></a>')
        body = "".join(rows) if rows else '<div class="empty">No failed tests in this category.</div>'
        category_sections.append(f'<section class="category-section" id="{category_anchor(category)}"><div class="panel"><div class="panel-head"><div><div class="panel-title">{html_escape(category)}</div><div class="panel-count">{len(items)} failed test{plural}</div></div><a class="back-link" href="#categories">Back to categories</a></div><div class="test-list">{body}</div></div></section>')

    detail_sections = []
    index = 0
    for category in CATEGORIES:
        for finding in grouped[category]:
            index += 1
            anchor = finding_anchor(finding, index)
            detail_sections.append(f'<section class="detail" id="{anchor}"><div class="panel"><div class="detail-head"><a class="back-link" href="#{category_anchor(category)}">← Back to {html_escape(category)}</a><div class="detail-title">{html_escape(finding.get("test_name"))}</div><div class="detail-meta"><span class="badge {severity_css_class(finding.get("severity"))}">{html_escape(finding.get("severity"))} severity</span><span class="badge {confidence_css_class(finding.get("confidence"))}">{html_escape(finding.get("confidence"))} confidence</span></div></div><div class="detail-body"><div class="detail-grid">{info_card_html("Root Cause Analysis", finding.get("root_cause"), True)}{info_card_html("Suggested Fix", finding.get("suggested_fix"), True)}{info_card_html("Recommended Action", finding.get("recommended_action"))}{info_card_html("Suggested Owner", finding.get("suggested_owner"))}{info_card_html("Evidence", finding.get("evidence"), True, "evidence")}{info_card_html("Test Information", test_info_text(finding), True)}</div><details class="disclosure"><summary>Technical failure evidence</summary><div class="tech-block"><div class="info-label">Failure message</div><div class="info-value trace">{html_escape(finding.get("failure_message") or "No failure message available.")}</div><pre>{html_escape(finding.get("stack_trace") or "No stack trace available.")}</pre></div></details></div></div></section>')

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CI Failure Triage — Build {html_escape(metadata.build_number)}</title>
<style>{CSS}</style>
</head>
<body>
<div class="app">
<header class="topbar"><div class="topbar-inner"><div class="brand"><div class="brand-mark">T</div><div><div class="brand-title">CI Failure Triage</div><div class="brand-sub">Failure classification &amp; root-cause analysis</div></div></div><div class="status {pipeline_class}">Pipeline: {html_escape(metadata.pipeline_status)}</div></div></header>
<main class="container">
<section class="hero"><div><div class="eyebrow">Build overview</div><h1>{html_escape(metadata.job_name)} <span class="build-number">#{html_escape(metadata.build_number)}</span></h1><div class="hero-meta">Generated {html_escape(metadata.timestamp)} &nbsp; {build_link}</div></div></section>
<section class="release {rec_class}"><div class="release-main"><div class="release-icon">{rec_icon}</div><div><div class="release-title">Release recommendation: {html_escape(recommendation.replace("_", " "))}</div><div class="release-reason">{html_escape(risk['reason'])}</div></div></div><div class="risk-pill">Risk: {html_escape(risk['risk_level'])}</div></section>
<section class="stats"><div class="stat"><div class="stat-value">{metrics.total}</div><div class="stat-label">Total tests</div></div><div class="stat"><div class="stat-value">{metrics.passed}</div><div class="stat-label">Passed</div></div><div class="stat"><div class="stat-value">{metrics.failed + metrics.broken}</div><div class="stat-label">Failed / broken</div></div><div class="stat"><div class="stat-value">{metrics.pass_rate}%</div><div class="stat-label">Pass rate</div></div></section>
<section id="categories"><h2 class="section-title">Failure Categories</h2><p class="section-subtitle">Select a category to see its failed tests.</p><div class="category-grid">{category_cards}</div></section>
{''.join(category_sections)}
{''.join(detail_sections)}
<div class="footer">Generated by the AI-Powered CI Failure Triage Engine</div>
</main></div>
</body>
</html>'''


# ============================================================
# 12. OUTPUT / ORCHESTRATION
# ============================================================

def write_reports(markdown: str, html_report: str, markdown_path: str, html_path: str) -> None:
    Path(markdown_path).parent.mkdir(parents=True, exist_ok=True)
    Path(html_path).parent.mkdir(parents=True, exist_ok=True)
    Path(markdown_path).write_text(markdown, encoding="utf-8")
    Path(html_path).write_text(html_report, encoding="utf-8")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI-powered CI failure triage engine")
    parser.add_argument("--report-dir", required=True, help="Directory containing Allure *-result.json files")
    parser.add_argument("--build-url", default=os.getenv("BUILD_URL", ""), help="Jenkins build URL")
    parser.add_argument("--output", required=True, help="Output path for Markdown report")
    parser.add_argument("--html-output", required=True, help="Output path for HTML dashboard")
    parser.add_argument("--no-console-log", action="store_true", help="Skip Jenkins console log collection")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    report_dir = Path(args.report_dir)

    if not report_dir.exists():
        print(f"Error: report directory not found: {report_dir}", file=sys.stderr)
        sys.exit(1)

    print("=" * 68)
    print("AI-POWERED CI FAILURE TRIAGE ENGINE")
    print("=" * 68)

    print("[1/6] Collecting CI evidence...")
    allure_results = load_allure_results(report_dir)
    metrics, failures = summarize_allure_results(allure_results)
    console_log = "" if args.no_console_log else fetch_console_log(args.build_url)
    metadata = collect_build_metadata(args.build_url, console_log)
    print(f"      Tests={metrics.total}, Passed={metrics.passed}, Failed={metrics.failed}, Broken={metrics.broken}")

    print("[2/6] Preprocessing failures...")
    processed_failures = preprocess_failures(failures)

    print("[3/6] Correlating failures...")
    clusters = cluster_failures(processed_failures)
    print(f"      Identified {len(clusters)} failure cluster(s)")

    print("[4/6] Performing AI triage analysis...")
    raw_analysis = perform_ai_triage(metadata, metrics, processed_failures, clusters, console_log)
    findings = build_validated_findings(processed_failures, clusters, raw_analysis)

    print("[5/6] Assessing release risk...")
    risk = assess_risk(raw_analysis, findings)

    print("[6/6] Generating reports...")
    markdown_report = generate_markdown_report(metadata, metrics, findings, risk)
    html_report = generate_html_dashboard(metadata, metrics, findings, risk)
    write_reports(markdown_report, html_report, args.output, args.html_output)

    print("=" * 68)
    print("TRIAGE COMPLETED")
    print(f"Release Recommendation : {risk['recommendation']}")
    print(f"Risk Level             : {risk['risk_level']}")
    print(f"Failed/ Broken Tests   : {len(findings)}")
    print(f"Markdown Report        : {args.output}")
    print(f"HTML Dashboard         : {args.html_output}")
    print("=" * 68)


if __name__ == "__main__":
    main()
