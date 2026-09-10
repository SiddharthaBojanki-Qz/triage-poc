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
    suite: str = ""
    feature: str = ""
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

def load_allure_results(
    report_dir: Path,
    build_start_file: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Load only Allure results belonging to the current Jenkins build.

    When build_start_file is supplied, only result files whose modification
    time is at or after the marker timestamp are considered. This prevents
    stale Allure JSON files from previous builds in a reused Jenkins workspace
    from contaminating the current build's triage report.
    """
    results: List[Dict[str, Any]] = []
    if not report_dir.exists():
        return results

    build_start_ts: Optional[float] = None
    if build_start_file:
        try:
            build_start_ts = float(build_start_file.read_text(encoding="utf-8").strip())
        except (OSError, ValueError) as exc:
            raise RuntimeError(
                f"Invalid or unreadable build start marker {build_start_file}: {exc}"
            ) from exc

    result_files = sorted(report_dir.glob("*-result.json"))
    skipped_stale = 0

    for file_path in result_files:
        try:
            if build_start_ts is not None and file_path.stat().st_mtime < build_start_ts:
                skipped_stale += 1
                continue

            results.append(json.loads(file_path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
            print(
                f"Warning: skipping unreadable Allure result {file_path.name}: {exc}",
                file=sys.stderr,
            )

    if build_start_ts is not None:
        print(
            f"      Current-build Allure results={len(results)} "
            f"(ignored {skipped_stale} stale result file(s))"
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
    """Return a real Allure feature/story when one exists; otherwise keep it absent."""
    labels = labels_map(result)
    for name in ("feature", "story", "epic"):
        values = labels.get(name) or []
        if values and values[0].strip():
            return values[0].strip()
    return ""


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
        detail = str(exc)
        if "403" in detail:
            print(
                "Warning: Jenkins console log access returned HTTP 403. "
                "Set JENKINS_USER and JENKINS_API_TOKEN (or Jenkins credentials "
                "binding) if console-log context is required. Continuing without it.",
                file=sys.stderr,
            )
        else:
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
        "cluster_id": cluster.cluster_id if cluster else "",
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
        from openai import OpenAI
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
            "failure_pattern": (
                cluster.probable_pattern
                if cluster is not None and cluster.probable_pattern
                else "Individual failure pattern"
            ),
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
            "Release assessment is based on failure severity and correlated impact.",
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
                    f"- **Root Cause:** {finding['root_cause']}",
                    f"- **Suggested Fix:** {finding['suggested_fix']}",
                    "",
                ])
    return "\n".join(lines)



def finding_category_map(findings: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group validated findings by the five approved user-facing categories."""
    grouped: Dict[str, List[Dict[str, Any]]] = {category: [] for category in CATEGORIES}
    for finding in findings:
        category = normalize_category(finding.get("category"))
        finding["category"] = category
        grouped[category].append(finding)
    return grouped


# ============================================================
# 11. INTERACTIVE HTML DASHBOARD
# ============================================================

CSS = r"""
:root{--bg:#f7f8fb;--surface:#ffffff;--surface-alt:#fbfcfe;--text:#24324a;--strong:#10213b;--muted:#718096;--faint:#9aa6b7;--border:#e3e8ef;--border-strong:#d5dce6;--primary:#1f5fbf;--primary-dark:#174a99;--primary-soft:#eef4fc;--danger:#c53030;--danger-soft:#fff4f4;--warning:#a15c00;--warning-soft:#fff8ed;--success:#237a4b;--success-soft:#edf8f1;--shadow:0 1px 2px rgba(16,33,59,.03),0 6px 22px rgba(16,33,59,.05);--shadow-open:0 10px 30px rgba(16,33,59,.08)}
*{box-sizing:border-box}html,body{margin:0;min-height:100%;background:var(--bg)}body{font-family:"Montserrat",-apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif;color:var(--text);font-size:14px;line-height:1.55;-webkit-font-smoothing:antialiased;text-rendering:optimizeLegibility}button,label,summary{font:inherit}a{color:inherit}
.topbar{height:64px;background:rgba(255,255,255,.96);border-bottom:1px solid var(--border);display:flex;align-items:center}.topbar-inner{width:100%;max-width:1280px;margin:0 auto;padding:0 28px;display:flex;align-items:center;justify-content:space-between;gap:20px}.brand{display:flex;align-items:center;gap:11px}.brand-mark{width:34px;height:34px;border-radius:9px;background:var(--strong);color:#fff;display:grid;place-items:center;font-size:13px;font-weight:900;letter-spacing:-.02em}.brand-title{font-size:14px;font-weight:800;color:var(--strong);letter-spacing:-.02em}.brand-sub{font-size:10.5px;color:var(--muted);margin-top:1px}.status{font-size:10px;font-weight:800;padding:6px 10px;border-radius:999px;border:1px solid var(--border);background:#fff;color:var(--muted)}.status.failure{color:var(--danger);background:var(--danger-soft);border-color:#f2cdcd}.status.success{color:var(--success);background:var(--success-soft);border-color:#cde7d8}
.container{max-width:1280px;margin:0 auto;padding:30px 28px 44px}.hero{display:flex;align-items:flex-end;justify-content:space-between;gap:24px;margin-bottom:20px}.eyebrow{font-size:9.5px;font-weight:800;letter-spacing:.14em;text-transform:uppercase;color:var(--muted)}h1{margin:5px 0 7px;font-size:31px;line-height:1.08;letter-spacing:-.045em;color:var(--strong);font-weight:800}.build-number{color:#6d7b91;font-weight:700}.hero-meta{font-size:11.5px;color:var(--muted)}.hero-meta a{color:var(--primary);font-weight:750;text-decoration:none}.hero-meta a:hover{text-decoration:underline}
.release{display:flex;align-items:center;justify-content:space-between;gap:18px;background:linear-gradient(180deg,#fff,#fcfdff);border:1px solid var(--border);border-left:3px solid var(--primary);border-radius:13px;padding:15px 18px;margin-bottom:15px;box-shadow:var(--shadow)}.release.hold,.release.no-go{border-left-color:var(--danger)}.release.conditional-go{border-left-color:var(--warning)}.release.go{border-left-color:var(--success)}.release-main{display:flex;align-items:center;gap:12px;min-width:0}.release-icon{font-size:16px;font-weight:900;line-height:1;color:var(--muted)}.release-title{font-size:12.5px;color:var(--strong);font-weight:800}.release-reason{font-size:11.5px;color:var(--muted);margin-top:3px;line-height:1.45}.risk-pill{padding:5px 9px;border:1px solid var(--border);background:#fff;border-radius:999px;font-size:9.5px;font-weight:800;white-space:nowrap;color:var(--muted)}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:28px}.stat{position:relative;background:var(--surface);border:1px solid var(--border);border-radius:13px;padding:18px 18px 16px;min-height:92px;display:flex;flex-direction:column;justify-content:center;overflow:hidden;box-shadow:0 1px 2px rgba(16,33,59,.025)}.stat:after{content:"";position:absolute;left:0;right:0;bottom:0;height:2px;background:#e9edf3}.stat-value{color:var(--strong);font-size:28px;line-height:1;font-weight:800;letter-spacing:-.05em}.stat-label{font-size:9.5px;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);font-weight:800;margin-top:8px}.stat:nth-child(3) .stat-value{color:var(--danger)}
.analysis-heading{display:flex;align-items:flex-end;justify-content:space-between;gap:20px;margin-bottom:12px}.analysis-title{font-size:18px;line-height:1.2;color:var(--strong);font-weight:800;letter-spacing:-.03em}.analysis-subtitle{font-size:11px;color:var(--muted);margin-top:3px}.analysis-hint{font-size:10.5px;color:var(--faint);text-align:right}
.category-workspace{margin-top:0}.category-bar{background:rgba(255,255,255,.88);border:1px solid var(--border);border-bottom:0;border-radius:14px 14px 0 0;box-shadow:var(--shadow);overflow:hidden}.category-tabs{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));width:100%}.category-option{display:block;text-decoration:none;min-width:0;border-right:1px solid var(--border)}.category-option:last-child{border-right:0}.category-card{position:relative;display:flex;align-items:center;justify-content:space-between;gap:10px;padding:15px 16px 14px;min-height:62px;background:#fff;cursor:pointer;white-space:nowrap;transition:background .16s ease,box-shadow .16s ease}.category-card:hover{background:#fbfcfe}.category-card:after{content:"";position:absolute;left:16px;right:16px;bottom:0;height:3px;border-radius:3px 3px 0 0;background:transparent;transition:background .16s ease}.category-card-title{font-size:11.5px;color:#44536b;font-weight:800;overflow:hidden;text-overflow:ellipsis;letter-spacing:-.015em}.category-card-count{display:inline-flex;align-items:center;justify-content:center;min-width:25px;height:23px;padding:0 7px;border-radius:7px;background:#f3f5f8;border:1px solid #e5e9ef;font-size:10px;font-weight:800;color:#6d7b90;flex:0 0 auto}.cat-radio:focus-visible~.category-bar{outline:3px solid rgba(31,95,191,.15);outline-offset:2px}
#cat-app:checked~.category-bar .tab-app,#cat-env:checked~.category-bar .tab-env,#cat-data:checked~.category-bar .tab-data,#cat-script:checked~.category-bar .tab-script,#cat-unknown:checked~.category-bar .tab-unknown{background:#fff;box-shadow:0 1px 0 #fff inset}.cat-radio:checked~.category-bar .category-card:after{background:transparent}#cat-app:checked~.category-bar .tab-app:after,#cat-env:checked~.category-bar .tab-env:after,#cat-data:checked~.category-bar .tab-data:after,#cat-script:checked~.category-bar .tab-script:after,#cat-unknown:checked~.category-bar .tab-unknown:after{background:var(--primary)}#cat-app:checked~.category-bar .tab-app .category-card-title,#cat-env:checked~.category-bar .tab-env .category-card-title,#cat-data:checked~.category-bar .tab-data .category-card-title,#cat-script:checked~.category-bar .tab-script .category-card-title,#cat-unknown:checked~.category-bar .tab-unknown .category-card-title{color:var(--strong)}#cat-app:checked~.category-bar .tab-app .category-card-count,#cat-env:checked~.category-bar .tab-env .category-card-count,#cat-data:checked~.category-bar .tab-data .category-card-count,#cat-script:checked~.category-bar .tab-script .category-card-count,#cat-unknown:checked~.category-bar .tab-unknown .category-card-count{background:var(--primary-soft);border-color:#d6e4f7;color:var(--primary-dark)}
.failures-section{background:#fff;border:1px solid var(--border);border-top:0;border-radius:0 0 14px 14px;box-shadow:var(--shadow);padding:18px 18px 20px}.category-panel{display:none}.category-panel-header{display:flex;align-items:center;justify-content:space-between;gap:18px;padding:3px 2px 14px;margin-bottom:10px;border-bottom:1px solid var(--border)}.category-panel-title{font-size:17px;line-height:1.2;font-weight:800;color:var(--strong);letter-spacing:-.025em}.category-panel-count{font-size:10.5px;color:var(--muted);font-weight:700}.test-list{display:flex;flex-direction:column;gap:7px}.test-item{border:1px solid var(--border);border-radius:11px;background:#fff;overflow:hidden;transition:border-color .16s ease,box-shadow .16s ease,transform .16s ease}.test-item:hover{border-color:#c8d5e6;box-shadow:0 4px 14px rgba(16,33,59,.05)}.test-item[open]{border-color:#b9cae1;box-shadow:var(--shadow-open)}.test-summary{list-style:none;display:flex;align-items:center;justify-content:space-between;gap:18px;padding:15px 16px;cursor:pointer}.test-summary::-webkit-details-marker{display:none}.test-summary:before{content:"";width:7px;height:7px;border-right:2px solid #7f8b9d;border-bottom:2px solid #7f8b9d;transform:rotate(-45deg);flex:0 0 auto;transition:transform .16s ease,border-color .16s ease;margin-left:1px}.test-item[open]>.test-summary:before{transform:rotate(45deg);border-color:var(--primary)}.test-main{min-width:0;flex:1}.test-name{display:block;font-size:13.5px;line-height:1.45;font-weight:800;color:var(--strong);overflow-wrap:anywhere}.test-secondary{margin-top:5px;display:flex;gap:7px;flex-wrap:wrap;color:var(--muted);font-size:10.5px}.test-secondary span:nth-child(even){color:#b6beca}.test-side{display:flex;align-items:center;flex-wrap:wrap;justify-content:flex-end;gap:6px;flex-shrink:0}.badge{display:inline-flex;align-items:center;padding:5px 9px;border-radius:999px;font-size:9.5px;font-weight:800;white-space:nowrap;border:1px solid transparent}.sev-critical,.sev-high{color:var(--danger);background:var(--danger-soft);border-color:#f5d1d1}.sev-medium{color:var(--warning);background:var(--warning-soft);border-color:#f1dfbf}.sev-low{color:var(--success);background:var(--success-soft);border-color:#d1e9dc}.conf-high{color:var(--primary-dark);background:var(--primary-soft);border-color:#d6e4f7}.conf-medium{color:#5f55a3;background:#f3f1fd;border-color:#ded9f6}.conf-low{color:var(--muted);background:#f6f7f9;border-color:#e6e9ee}
.test-detail{border-top:1px solid var(--border);padding:18px 20px 20px;background:linear-gradient(180deg,#fbfcfe,#f9fbfd)}.failure-pattern{border:1px solid #dbe3ee;border-radius:10px;background:#fff;padding:13px 14px 14px;margin-bottom:12px}.failure-pattern .info-label{margin-bottom:6px}.pattern-text{display:block;color:var(--strong);font-size:12.5px;font-weight:700;line-height:1.55}.detail-grid{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:9px}.snapshot-grid{margin-bottom:14px}.snapshot-grid .info-card:nth-child(1),.snapshot-grid .info-card:nth-child(2),.snapshot-grid .info-card:nth-child(3){grid-column:span 2}.snapshot-grid .info-card:nth-child(n+4){grid-column:span 3}.analysis-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}.analysis-grid .analysis-owner{align-self:start}.info-card{border:1px solid var(--border);border-radius:10px;background:#fff;padding:13px 14px}.info-card.full{grid-column:1/-1}.snapshot-grid .info-card{padding:11px 12px;background:#fbfcfe}.snapshot-heading{font-size:10px;font-weight:800;letter-spacing:.11em;text-transform:uppercase;color:var(--muted);margin:0 0 8px}.analysis-grid .info-card{padding:15px 16px}.analysis-grid .analysis{background:#fff}.analysis-grid .analysis .info-label{color:var(--primary-dark)}.info-label{font-size:8.75px;font-weight:800;text-transform:uppercase;letter-spacing:.11em;color:var(--muted);margin-bottom:6px}.info-value{font-size:12.5px;line-height:1.65;color:var(--text);white-space:pre-wrap;overflow-wrap:anywhere}.trace{font:10.5px/1.55 ui-monospace,SFMono-Regular,Consolas,monospace}.disclosure{margin-top:14px;border:1px solid var(--border);border-radius:10px;overflow:hidden;background:#fff}.disclosure summary{cursor:pointer;padding:12px 14px;font-size:10.5px;font-weight:800;color:var(--strong);background:#fff}.disclosure summary:hover{background:#fbfcfe}.disclosure .tech-block{border-top:1px solid var(--border);padding:13px 14px;background:#f7f8fa}.disclosure pre{margin:8px 0 0;padding:12px;border-radius:8px;background:#18283f;color:#eef3f8;max-height:280px;overflow:auto;white-space:pre-wrap;overflow-wrap:anywhere;font:10.5px/1.55 ui-monospace,SFMono-Regular,Consolas,monospace}.tech-label{margin-top:12px}.empty{padding:48px 20px;text-align:center;color:var(--muted);font-size:11.5px}.footer{color:var(--faint);text-align:center;font-size:9.5px;margin-top:18px}
.cat-radio{position:absolute;opacity:0;pointer-events:none}#cat-app:checked~.failures-section .panel-app,#cat-env:checked~.failures-section .panel-env,#cat-data:checked~.failures-section .panel-data,#cat-script:checked~.failures-section .panel-script,#cat-unknown:checked~.failures-section .panel-unknown{display:block}
@media (max-width:900px){.container{padding:22px 16px 30px}.topbar-inner{padding:0 16px}.category-card{padding-left:12px;padding-right:12px}.category-card-title{font-size:10.5px}.analysis-hint{display:none}}
@media (max-width:700px){.container{padding:16px 10px 24px}.topbar{height:60px}.brand-sub{display:none}.stats{grid-template-columns:repeat(2,1fr);gap:9px;margin-bottom:22px}.stat{min-height:78px;padding:14px}.stat-value{font-size:24px}.hero{flex-direction:column;align-items:flex-start;gap:6px;margin-bottom:16px}h1{font-size:26px}.category-bar{overflow-x:auto}.category-tabs{grid-template-columns:repeat(5,minmax(170px,1fr));min-width:850px}.failures-section{padding:14px}.category-panel-header{align-items:flex-start;flex-direction:column;gap:4px}.test-summary{align-items:flex-start}.test-side{justify-content:flex-start}.detail-grid{grid-template-columns:1fr}.snapshot-grid .info-card:nth-child(n){grid-column:auto}.analysis-grid{grid-template-columns:1fr}.info-card.full{grid-column:auto}.release{align-items:flex-start;flex-direction:column}.risk-pill{align-self:flex-start}}
@media print{.category-panel{display:block!important}.category-bar{box-shadow:none}.failures-section{box-shadow:none}.test-item{break-inside:avoid}}
"""

def severity_css_class(value: Any) -> str:
    return "sev-" + str(value or "low").strip().lower().replace(" ", "-")



def info_card_html(label: str, value: Any, full: bool = False, extra: str = "") -> str:
    classes = "info-card" + (" full" if full else "")
    value_class = "info-value" + ((" " + extra) if extra else "")
    return f'<div class="{classes}"><div class="info-label">{html_escape(label)}</div><div class="{value_class}">{html_escape(value or "Not available")}</div></div>'


def generate_html_dashboard(metadata: BuildMetadata, metrics: BuildMetrics, findings: List[Dict[str, Any]], risk: Dict[str, Any]) -> str:
    """Generate a compact tab-style failure dashboard without JavaScript.

    Five categories are displayed in one horizontal tab row. The first category
    is selected by default. Selecting another category switches the content below
    to that category's failed tests. Each failed test is a native HTML disclosure
    row (<details>) so its analysis expands inline without JavaScript, keeping the
    report compatible with restrictive Jenkins CSP settings.
    """
    grouped = finding_category_map(findings)
    recommendation = risk["recommendation"]
    rec_class = recommendation.lower().replace("_", "-")
    rec_icon = "✓" if recommendation == "GO" else "⚠" if recommendation == "CONDITIONAL_GO" else "!"
    pipeline_class = (
        "success" if metadata.pipeline_status == "SUCCESS"
        else "failure" if metadata.pipeline_status in {"FAILURE", "FAILED"}
        else ""
    )
    build_link = (
        f'<a href="{html_escape(metadata.build_url)}" target="_blank" rel="noopener">'
        'Open Jenkins build ↗</a>'
        if metadata.build_url else ""
    )

    category_defs = [
        ("Application Defect", "cat-app", "panel-app"),
        ("Environment Issue", "cat-env", "panel-env"),
        ("Test Data Issue", "cat-data", "panel-data"),
        ("Script Defect", "cat-script", "panel-script"),
        ("Unknown", "cat-unknown", "panel-unknown"),
    ]

    category_inputs = "".join(
        f'<input class="cat-radio" type="radio" name="category" id="{radio}" '
        f'{"checked" if idx == 0 else ""}>'
        for idx, (_, radio, _) in enumerate(category_defs)
    )

    category_tabs = "".join(
        f'<label class="category-option" for="{radio}">'
        f'<span class="category-card tab-{radio[4:]}">'
        f'<span class="category-card-title">{html_escape(category)}</span>'
        f'<span class="category-card-count">{len(grouped[category])}</span>'
        f'</span></label>'
        for category, radio, _ in category_defs
    )

    def render_detail(finding: Dict[str, Any]) -> str:
        related_count = max(0, int(finding.get("duplicate_count", 1)) - 1)
        snapshot_cards = [
            info_card_html("Status", finding.get("status")),
            info_card_html("Severity", finding.get("severity")),
        ]
        if finding.get("module"):
            snapshot_cards.append(info_card_html("Module", finding.get("module")))
        if finding.get("suite"):
            snapshot_cards.append(info_card_html("Suite", finding.get("suite")))
        if finding.get("exception_type"):
            snapshot_cards.append(info_card_html("Exception", finding.get("exception_type")))
        if finding.get("feature"):
            snapshot_cards.append(info_card_html("Feature / Story", finding.get("feature")))
        if related_count:
            snapshot_cards.append(
                info_card_html(
                    "Related Tests",
                    f'{related_count} other test{"s" if related_count != 1 else ""} show the same failure pattern',
                    True,
                )
            )

        tech = (
            '<details class="disclosure"><summary>View technical error details</summary>'
            '<div class="tech-block">'
            '<div class="tech-section"><div class="info-label">Failure message</div>'
            f'<div class="info-value trace">{html_escape(finding.get("failure_message") or "No failure message available.")}</div></div>'
            '<div class="tech-section"><div class="info-label">Stack trace</div>'
            f'<pre>{html_escape(finding.get("stack_trace") or "No stack trace available.")}</pre></div>'
            '</div></details>'
        )
        return (
            f'<div class="test-detail">'
            f'<div class="snapshot-heading">Test snapshot</div>'
            f'<div class="detail-grid snapshot-grid">{"".join(snapshot_cards)}</div>'
            f'<div class="analysis-grid">'
            f'{info_card_html("Root Cause Analysis", finding.get("root_cause"), True, "analysis")}'
            f'{info_card_html("Suggested Fix", finding.get("suggested_fix"), True, "analysis")}'
            f'{info_card_html("Evidence", finding.get("evidence"), True, "analysis")}'
            f'</div>{tech}'
            f'</div>'
        )

    def render_test_item(finding: Dict[str, Any]) -> str:
        module = finding.get("module") or finding.get("suite")
        secondary = []
        if module:
            secondary.append(html_escape(module))
        if finding.get("exception_type"):
            secondary.append(html_escape(finding.get("exception_type")))
        related_count = max(0, int(finding.get("duplicate_count", 1)) - 1)
        if related_count:
            secondary.append(f'{related_count} similar test{"s" if related_count != 1 else ""}')
        secondary_html = "".join(
            f'<span>{item}</span><span>•</span>' for item in secondary[:-1]
        )
        if secondary:
            secondary_html += f'<span>{secondary[-1]}</span>'
        return (
            f'<details class="test-item">'
            f'<summary class="test-summary">'
            f'<span class="test-main">'
            f'<span class="test-name">{html_escape(finding.get("test_name"))}</span>'
            f'<span class="test-secondary">{secondary_html}</span>'
            f'</span>'
            f'<span class="test-side">'
            f'<span class="badge {severity_css_class(finding.get("severity"))}">{html_escape(finding.get("severity"))}</span>'
            f'</span>'
            f'</summary>'
            f'{render_detail(finding)}'
            f'</details>'
        )

    panels = []
    for category, _, panel_class in category_defs:
        items = grouped[category]
        body = "".join(render_test_item(finding) for finding in items)
        if not body:
            body = '<div class="empty">No failed tests in this category.</div>'
        plural = "s" if len(items) != 1 else ""
        panels.append(
            f'<section class="category-panel {panel_class}">'
            f'<div class="category-panel-header">'
            f'<div><div class="eyebrow">Failure category</div>'
            f'<div class="category-panel-title">{html_escape(category)}</div></div>'
            f'<div class="category-panel-count">{len(items)} failed test{plural}</div>'
            f'</div>'
            f'<div class="test-list">{body}</div>'
            f'</section>'
        )

    return f"""<!DOCTYPE html>
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
<section class="release {rec_class}"><div class="release-main"><div class="release-icon">{rec_icon}</div><div><div class="release-title">Release recommendation: {html_escape(recommendation.replace("_", " "))}</div><div class="release-reason">{html_escape(risk["reason"])}</div></div></div><div class="risk-pill">Risk: {html_escape(risk["risk_level"])}</div></section>
<section class="stats"><div class="stat"><div class="stat-value">{metrics.total}</div><div class="stat-label">Total tests</div></div><div class="stat"><div class="stat-value">{metrics.passed}</div><div class="stat-label">Passed</div></div><div class="stat"><div class="stat-value">{metrics.failed + metrics.broken}</div><div class="stat-label">Failed / broken</div></div><div class="stat"><div class="stat-value">{metrics.pass_rate}%</div><div class="stat-label">Pass rate</div></div></section>
<div class="analysis-heading"><div><div class="analysis-title">Failure analysis</div><div class="analysis-subtitle">Select a category to review failed tests, then expand a test for the analysis.</div></div><div class="analysis-hint">5 failure categories</div></div>
<div class="category-workspace">
{category_inputs}
<section class="category-bar"><div class="category-tabs">{category_tabs}</div></section>
<section class="failures-section">{"".join(panels)}</section>
</div>
<div class="footer">Generated by the AI-Powered CI Failure Triage Engine</div>
</main></div>
</body></html>"""


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
    parser.add_argument(
        "--build-start-file",
        help=(
            "File containing the current Jenkins build start timestamp. "
            "When provided, only Allure result files created/updated from that "
            "timestamp onward are included."
        ),
    )
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
    build_start_file = Path(args.build_start_file) if args.build_start_file else None
    allure_results = load_allure_results(report_dir, build_start_file)
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
