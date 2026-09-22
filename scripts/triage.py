#!/usr/bin/env python3
"""
CI Failure Triage Engine
========================

Single-file engine. The script extracts evidence and renders reports.
OpenAI performs classification, grouping, and root-cause analysis.

    python scripts/triage.py --report-dir "$WORKSPACE/allure-report"

    Jenkins flow:
      1. collect Allure evidence directly from --report-dir
      2. reason with OpenAI
      3. render the existing HTML dashboard

Each zip is extracted and reported under a folder named after that zip:
    input_data/_extracted/<name>/
    report/<name>/

If input_data/ contains exactly one zip, --zip may be omitted. If several zips
are present, --zip is required so the same archive is used for extract and render.

No keyword-only classifier is used for semantic triage; OpenAI performs evidence-based reasoning.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import os
import re
import shutil
import sys
import urllib.error
import urllib.request
import webbrowser
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_DIR = PROJECT_ROOT / "input_data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "report"
MAX_CONSOLE_LINES = 300

CATEGORIES = [
    "Application Defect",
    "Environment Issue",
    "Test Data Issue",
    "Script Defect",
    "Unknown",
]
SKIPPED_CATEGORY = "Skipped"
DISPLAY_CATEGORIES = CATEGORIES + [SKIPPED_CATEGORY]
SEVERITIES = ["Critical", "High", "Medium", "Low"]
CONFIDENCES = ["High", "Medium", "Low"]
SEVERITY_ORDER = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1}
OWNERS = {
    "Application",
    "QA Automation",
    "QA",
    "DevOps",
    "Data",
    "Engineering",
    "Unknown",
}
RECOMMENDATIONS = {"GO", "CONDITIONAL_GO", "HOLD", "NO_GO"}


# ============================================================
# Helpers
# ============================================================

def safe(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def html_escape(value: Any, default: str = "N/A") -> str:
    return html.escape(safe(value, default) or default, quote=True)


def normalize_category(value: Any) -> str:
    raw = safe(value, "Unknown")
    if raw in CATEGORIES:
        return raw
    aliases = {
        "application defect": "Application Defect",
        "product defect": "Application Defect",
        "environment issue": "Environment Issue",
        "test data issue": "Test Data Issue",
        "script defect": "Script Defect",
        "unknown": "Unknown",
    }
    return aliases.get(raw.lower(), "Unknown")


def normalize_severity(value: Any, default: str = "Medium") -> str:
    value = safe(value, default).title()
    return value if value in SEVERITIES else default


def normalize_confidence(value: Any, default: str = "Medium") -> str:
    value = safe(value, default).title()
    return value if value in CONFIDENCES else default


def normalize_recommendation(value: Any) -> str:
    normalized = safe(value, "HOLD").upper().replace("-", "_").replace(" ", "_")
    return normalized if normalized in RECOMMENDATIONS else "HOLD"


def normalize_owner(value: Any) -> str:
    raw = safe(value, "Unknown")
    return raw if raw in OWNERS else "Unknown"


def exception_class(message: str, trace: str) -> str:
    source = f"{message}\n{trace}"
    match = re.search(r"\b([A-Za-z_][\w.]*(?:Exception|Error|Failure))\b", source)
    return match.group(1) if match else ""


def labels_map(result: Dict[str, Any]) -> Dict[str, List[str]]:
    output: Dict[str, List[str]] = defaultdict(list)
    for label in result.get("labels") or []:
        name = label.get("name")
        if name and label.get("value") is not None:
            output[str(name)].append(str(label["value"]))
    return output


def first_label(labels: Dict[str, List[str]], *names: str) -> str:
    for name in names:
        values = labels.get(name) or []
        if values and values[0].strip():
            return values[0].strip()
    return ""


def extract_module(labels: Dict[str, List[str]]) -> str:
    package = first_label(labels, "package")
    if package and "." in package:
        return package.rsplit(".", 1)[0]
    return first_label(labels, "testClass", "parentSuite", "suite") or "Unknown"


def flatten_steps(steps: Any) -> List[Dict[str, str]]:
    flattened: List[Dict[str, str]] = []
    for step in steps or []:
        if not isinstance(step, dict):
            continue
        details = step.get("statusDetails") or {}
        flattened.append({
            "name": safe(step.get("name"), "(unnamed step)"),
            "status": safe(step.get("status"), "unknown").lower(),
            "message": safe(step.get("statusMessage") or details.get("message")),
        })
        flattened.extend(flatten_steps(step.get("steps") or []))
    return flattened


def walk_attachments(node: Any, found: List[Dict[str, str]]) -> None:
    if isinstance(node, list):
        for item in node:
            walk_attachments(item, found)
        return
    if not isinstance(node, dict):
        return
    for attachment in node.get("attachments") or []:
        if isinstance(attachment, dict):
            found.append({
                "name": safe(attachment.get("name")),
                "source": safe(attachment.get("source")),
                "type": safe(attachment.get("type")),
            })
    for key in ("steps", "beforeStages", "afterStages"):
        walk_attachments(node.get(key), found)
    walk_attachments(node.get("testStage"), found)


def parameters_map(record: Dict[str, Any]) -> Dict[str, str]:
    params: Dict[str, str] = {}
    for item in record.get("parameters") or []:
        if isinstance(item, dict) and item.get("name"):
            params[str(item["name"])] = str(item.get("value") or "")
    return params


def build_index(root: Path) -> Dict[str, str]:
    index: Dict[str, str] = {}
    for path in root.rglob("*"):
        if path.is_file():
            index.setdefault(path.name, str(path.resolve()))
    return index


def resolve_screenshots(
    attachments: List[Dict[str, str]],
    file_index: Dict[str, str],
) -> List[str]:
    paths: List[str] = []
    for attachment in attachments:
        source = attachment.get("source") or ""
        kind = (attachment.get("type") or "").lower()
        name = (attachment.get("name") or "").lower()
        is_image = (
            kind.startswith("image/")
            or source.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".webp"))
            or "screenshot" in name
        )
        if not is_image or not source:
            continue
        resolved = file_index.get(source) or file_index.get(Path(source).name)
        if resolved and resolved not in paths:
            paths.append(resolved)
    return paths


def extract_build_from_url(url: str) -> str:
    match = re.search(r"/(\d+)/allure", url or "")
    if match:
        return f"#{match.group(1)}"
    match = re.search(r"/(\d+)/", url or "")
    return f"#{match.group(1)}" if match else ""


def summarize_history(extra: Dict[str, Any], case: Dict[str, Any]) -> Dict[str, Any]:
    history = (extra or {}).get("history") or {}
    statistic = history.get("statistic") or {}
    items = history.get("items") or []
    recent = [safe(item.get("status")).lower() for item in items if isinstance(item, dict)]
    last_passed = ""
    for item in items:
        if isinstance(item, dict) and safe(item.get("status")).lower() == "passed":
            last_passed = extract_build_from_url(safe(item.get("reportUrl")))
            break
    current_failed = safe(case.get("status")).lower() in {"failed", "broken"}
    consecutive = 0
    if current_failed:
        consecutive = 1
        for status in recent:
            if status in {"failed", "broken"}:
                consecutive += 1
            else:
                break
    return {
        "total_runs": int(statistic.get("total") or len(items) or 0),
        "statistic": {
            "failed": int(statistic.get("failed") or 0),
            "broken": int(statistic.get("broken") or 0),
            "passed": int(statistic.get("passed") or 0),
            "skipped": int(statistic.get("skipped") or 0),
        },
        "recent_statuses": recent[:20],
        "last_passed_build": last_passed or None,
        "consecutive_failures": consecutive,
        "flaky": bool(case.get("flaky")),
        "new_failed": bool(case.get("newFailed")),
        "new_broken": bool(case.get("newBroken")),
        "retries_count": int(case.get("retriesCount") or 0),
    }


def retry_info(extra: Dict[str, Any], message: str) -> Tuple[bool, bool]:
    retries = (extra or {}).get("retries") or []
    if not retries:
        return False, False
    first = retries[0] if isinstance(retries[0], dict) else {}
    details = safe(first.get("statusDetails") or first.get("statusMessage"))
    same = bool(details) and details.strip() == (message or "").strip()
    return True, same


# ============================================================
# Zip + Allure collection
# ============================================================

def extract_zip(zip_path: Path, dest: Path) -> Path:
    if not zip_path.exists():
        raise FileNotFoundError(f"Zip file not found: {zip_path}")
    if not zipfile.is_zipfile(zip_path):
        raise ValueError(f"Not a valid zip archive: {zip_path}")

    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    dest = dest.resolve()

    with zipfile.ZipFile(zip_path, "r") as archive:
        for info in archive.infolist():
            target = (dest / info.filename).resolve()
            try:
                target.relative_to(dest)
            except ValueError as exc:
                raise ValueError(f"Unsafe zip path rejected: {info.filename}") from exc
        archive.extractall(dest)

    print(f"      Extracted {zip_path.name} -> {dest}")
    return dest


def list_input_zips(input_dir: Path) -> List[Path]:
    if not input_dir.exists():
        return []
    return sorted(
        (path for path in input_dir.glob("*.zip") if path.is_file()),
        key=lambda path: path.name.lower(),
    )


def format_zip_list(zips: Iterable[Path]) -> str:
    names = [path.name for path in zips]
    return ", ".join(names) if names else "(none)"


def sanitize_run_name(name: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*]', "_", name).strip(" .")
    return cleaned or "unnamed-report"


def run_slug(zip_path: Path) -> str:
    return sanitize_run_name(zip_path.stem)


def load_console_log(root: Path, max_lines: int = MAX_CONSOLE_LINES) -> str:
    names = {
        "consoletext",
        "console.log",
        "console.txt",
        "build.log",
        "jenkins.log",
        "pipeline.log",
    }
    candidates = [
        path for path in root.rglob("*")
        if path.is_file() and path.name.lower() in names
    ]
    if not candidates:
        return ""
    text = candidates[0].read_text(encoding="utf-8", errors="replace")
    print(f"      Using local console log: {candidates[0].name}")
    return "\n".join(text.splitlines()[-max_lines:])


def load_json_any(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return None


def metadata_from_allure_executor(extracted_root: Path) -> Tuple[str, str, str]:
    """Read Jenkins/Allure build id from executors.json, then trend files."""
    for path in extracted_root.rglob("executors.json"):
        data = load_json_any(path)
        rows = data if isinstance(data, list) else [data] if isinstance(data, dict) else []
        for row in rows:
            if not isinstance(row, dict):
                continue
            build_order = row.get("buildOrder")
            build_url = safe(row.get("buildUrl") or row.get("reportUrl"))
            number = str(build_order).strip() if build_order is not None else ""
            if not number and build_url:
                match = re.search(r"/(\d+)(?:/allure)?/?$", build_url)
                if match:
                    number = match.group(1)
            if not number:
                continue
            build_name = safe(row.get("buildName"))
            job_name = build_name.split("#")[0].strip() if build_name else safe(row.get("name"))
            host = safe(row.get("url") or row.get("buildUrl"))
            return job_name, number, host

    for name in ("history-trend.json", "categories-trend.json", "duration-trend.json", "retry-trend.json"):
        for path in extracted_root.rglob(name):
            data = load_json_any(path)
            rows = data if isinstance(data, list) else []
            if not rows or not isinstance(rows[0], dict):
                continue
            build_order = rows[0].get("buildOrder")
            if build_order is None:
                continue
            report_url = safe(rows[0].get("reportUrl"))
            job_name = ""
            job_match = re.search(r"/job/([^/]+)/", report_url)
            if job_match:
                job_name = job_match.group(1)
            return job_name, str(build_order).strip(), report_url
    return "", "N/A", ""


def apply_executor_metadata(build: Dict[str, Any], extracted_root: Path) -> None:
    job_name, build_number, host = metadata_from_allure_executor(extracted_root)
    if build_number and build_number != "N/A":
        build["build_number"] = build_number
    if job_name:
        build["job_name"] = job_name
    if host:
        build["host"] = host


def infer_metadata(zip_path: Optional[Path], extracted_root: Path) -> Tuple[str, str, str]:
    job_name = zip_path.stem if zip_path else "Allure Report"
    build_number = "N/A"
    host = ""

    env_files = list(extracted_root.rglob("environment.properties")) + list(
        extracted_root.rglob("environment.xml")
    )
    for env_file in env_files:
        try:
            text = env_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        job_match = re.search(r"(?:job[_-]?name|JOB_NAME)\s*[=:]\s*(.+)", text, re.I)
        build_match = re.search(r"(?:build[_-]?number|BUILD_NUMBER)\s*[=:]\s*(.+)", text, re.I)
        host_match = re.search(r"(?:host|HOST)\s*[=:]\s*(.+)", text, re.I)
        if job_match:
            job_name = job_match.group(1).strip()
        if build_match:
            build_number = build_match.group(1).strip()
        if host_match:
            host = host_match.group(1).strip()

    exec_job, exec_build, exec_host = metadata_from_allure_executor(extracted_root)
    if exec_build and exec_build != "N/A":
        build_number = exec_build
    if exec_job:
        job_name = exec_job
    if exec_host:
        host = exec_host

    if zip_path and build_number == "N/A":
        match = re.search(r"(?:build|#)[-_]?(\d+)", zip_path.stem, re.I)
        if match:
            build_number = match.group(1)
    return job_name, build_number, host


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
        print(f"Warning: skipping unreadable file {path.name}: {exc}", file=sys.stderr)
        return None
    return data if isinstance(data, dict) else None


def iter_report_cases(root: Path) -> Iterable[Tuple[Path, Dict[str, Any]]]:
    for directory in root.rglob("test-cases"):
        if not directory.is_dir():
            continue
        for file_path in sorted(directory.glob("*.json")):
            case = load_json(file_path)
            if case:
                yield file_path, case


def iter_raw_results(root: Path) -> Iterable[Tuple[Path, Dict[str, Any]]]:
    for file_path in sorted(root.rglob("*-result.json")):
        case = load_json(file_path)
        if case:
            yield file_path, case


def dossier_from_report_case(
    case: Dict[str, Any],
    file_index: Dict[str, str],
) -> Optional[Dict[str, Any]]:
    status = safe(case.get("status")).lower()
    if not status:
        return None
    extra = case.get("extra") or {}
    labels = labels_map(case)
    message = safe(case.get("statusMessage"))
    trace = safe(case.get("statusTrace"))
    attachments: List[Dict[str, str]] = []
    walk_attachments(case, attachments)
    retried, retry_same = retry_info(extra, message)
    timing = case.get("time") or {}
    return {
        "test_name": safe(case.get("name") or case.get("fullName"), "Unnamed Test"),
        "full_name": safe(case.get("fullName") or case.get("name")),
        "status": status,
        "suite": first_label(labels, "suite", "parentSuite", "subSuite"),
        "feature": first_label(labels, "feature", "story", "epic"),
        "module": extract_module(labels),
        "duration_ms": int(timing.get("duration") or 0),
        "retry": bool(case.get("retry")),
        "uid": safe(case.get("uid")),
        "history_id": safe(case.get("historyId")),
        "error": {
            "message": message,
            "trace": trace,
            "exception_class": exception_class(message, trace),
        },
        "steps": flatten_steps((case.get("testStage") or {}).get("steps") or []),
        "parameters": parameters_map(case),
        "screenshots": resolve_screenshots(attachments, file_index),
        "attachments": attachments,
        "history": summarize_history(extra, case),
        "retried": retried or int(case.get("retriesCount") or 0) > 0,
        "retry_same_error": retry_same,
    }


def dossier_from_raw_result(
    result: Dict[str, Any],
    file_index: Dict[str, str],
) -> Optional[Dict[str, Any]]:
    status = safe(result.get("status")).lower()
    if not status:
        return None
    details = result.get("statusDetails") or {}
    labels = labels_map(result)
    message = safe(details.get("message"))
    trace = safe(details.get("trace"))
    attachments: List[Dict[str, str]] = []
    walk_attachments(result, attachments)
    return {
        "test_name": safe(result.get("name") or result.get("fullName"), "Unnamed Test"),
        "full_name": safe(result.get("fullName") or result.get("name")),
        "status": status,
        "suite": first_label(labels, "suite", "parentSuite", "subSuite"),
        "feature": first_label(labels, "feature", "story", "epic"),
        "module": extract_module(labels),
        "duration_ms": int(result.get("duration") or ((result.get("stop") or 0) - (result.get("start") or 0))),
        "retry": False,
        "uid": safe(result.get("uuid") or result.get("uid")),
        "history_id": safe(result.get("historyId")),
        "error": {
            "message": message,
            "trace": trace,
            "exception_class": exception_class(message, trace),
        },
        "steps": flatten_steps(result.get("steps") or []),
        "parameters": parameters_map(result),
        "screenshots": resolve_screenshots(attachments, file_index),
        "attachments": attachments,
        "history": summarize_history({}, result),
        "retried": False,
        "retry_same_error": False,
    }


def collect_tests(extracted_root: Path) -> Tuple[List[Dict[str, Any]], str]:
    file_index = build_index(extracted_root)
    report_cases = list(iter_report_cases(extracted_root))
    if report_cases:
        tests = []
        skipped_retries = 0
        for _, case in report_cases:
            record = dossier_from_report_case(case, file_index)
            if not record:
                continue
            if record["retry"]:
                skipped_retries += 1
                continue
            tests.append(record)
        print(
            f"      Allure report test cases={len(tests)} "
            f"(skipped {skipped_retries} superseded retry attempt(s))"
        )
        return tests, "allure-report"
    raw = []
    for _, result in iter_raw_results(extracted_root):
        record = dossier_from_raw_result(result, file_index)
        if record:
            raw.append(record)
    if not raw:
        raise FileNotFoundError(
            f"No Allure test data found under {extracted_root}. "
            "Provide either raw *-result.json files or a report with data/test-cases/*.json."
        )
    print(f"      Allure raw results={len(raw)}")
    return raw, "allure-results"


def compute_metrics(tests: List[Dict[str, Any]]) -> Dict[str, Any]:
    metrics = {
        "total": 0,
        "passed": 0,
        "failed": 0,
        "broken": 0,
        "skipped": 0,
        "pass_rate": 100.0,
    }
    for test in tests:
        status = test["status"]
        metrics["total"] += 1
        if status == "passed":
            metrics["passed"] += 1
        elif status == "failed":
            metrics["failed"] += 1
        elif status == "broken":
            metrics["broken"] += 1
        elif status in {"skipped", "unknown"}:
            metrics["skipped"] += 1
    if metrics["total"]:
        metrics["pass_rate"] = round((metrics["passed"] / metrics["total"]) * 100, 1)
    return metrics


def build_evidence(
    extracted_root: Path,
    zip_path: Optional[Path],
    tests: List[Dict[str, Any]],
    source_format: str,
) -> Dict[str, Any]:
    metrics = compute_metrics(tests)
    job_name, build_number, host = infer_metadata(zip_path, extracted_root)
    if job_name == "N/A":
        job_name = zip_path.stem if zip_path else "Allure Report"
    pipeline = "FAILURE" if (metrics["failed"] or metrics["broken"]) else "SUCCESS"
    failing = [test for test in tests if test["status"] in {"failed", "broken"}]
    skipped = [test for test in tests if test["status"] == "skipped"]
    return {
        "build": {
            "job_name": job_name,
            "build_number": build_number,
            "host": host,
            "pipeline_status": pipeline,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "source_zip": zip_path.name if zip_path else "",
            "source_format": source_format,
        },
        "metrics": metrics,
        "console_log": load_console_log(extracted_root),
        "extracted_root": str(extracted_root.resolve()),
        "report_dir": "",
        "failing_tests": failing,
        "skipped_tests": skipped,
    }



# ============================================================
# OpenAI triage reasoning
# ============================================================

TRIAGE_SYSTEM_PROMPT = r"""
You are the CI Failure Triage reasoning engine for an enterprise Allure/Jenkins
test automation system.

Your job is NOT to write HTML, parse files, invent data, or perform generic
keyword classification. Python has already collected the evidence into a
structured dossier. You must reason from that evidence and the supplied
screenshots, then return ONLY the JSON object required by the output schema.

The report is read by a QA lead or release manager. If root cause and evidence
are not clear in one read, the analysis has failed.

NON-NEGOTIABLE RULES
1. Use only facts present in the evidence dossier and screenshots.
2. Never invent files, line numbers, APIs, incidents, expected behavior,
   parameters, locators, builds, or application details.
3. Every category, root cause, severity, and confidence decision must be
   traceable to a specific evidence fact, step, parameter, history record,
   console fact, or opened screenshot.
4. Complete all three reasoning phases before writing finding text.
5. Exact test_name values must be copied from evidence.
6. Preserve exact technical details that support the conclusion: step names,
   error text, exception class, parameters, locators, history counts, and
   screenshot name/source.
7. A generic Selenium exception is not a root cause.
8. If evidence is insufficient, use Unknown and lower confidence rather than
   guessing.
9. Group only when there is a shared observable mechanism: the same exact
   error message, the same failing step with the same mechanism, or a clearly
   shared suite/history/environment pattern supported by evidence.
10. Every failed or broken test must have exactly one finding.

INPUT
The user message contains a JSON evidence dossier. It includes:
- build metadata
- metrics
- console_log
- failing_tests
- skipped_tests
Each failing test can contain:
- test_name
- status
- suite/module/feature
- error.message
- error.trace
- error.exception_class
- steps
- parameters
- screenshots
- attachments
- history
- retry_same_error

Screenshots are supplied as image blocks after the evidence text. A screenshot
manifest maps each image to its Allure attachment name/source. When a screenshot
path exists, inspect the corresponding supplied image before finalizing the
finding.

REASONING PHASE 1 — BUILD-WIDE RECONNAISSANCE
A. Read the entire console log if it is non-empty. If an infrastructure error
   explains multiple failures, use that as build-wide evidence.
B. Map every distinct error.message. The same exact message is a strong signal
   for grouping; analyze that root cause once and apply it only where the
   evidence supports the same mechanism.
C. Do not trust generic Selenium messages such as timeout,
   NoSuchElementException, or ElementClickInterceptedException as root cause.
   Use the screenshot and the failure boundary.
D. Read history:
   - consecutive_failures >= 5 => chronic
   - consecutive_failures <= 2 with a recent last-passed build => regression
   - mixed recent_statuses => intermittent
E. retry_same_error=true means the failure is stable/reproducible, not a
   one-off race.
F. Draft groups before writing findings. Every group must have a shared
   observable fact. Use G-01, G-02, etc.

REASONING PHASE 2 — CAUSAL CHAIN
For every group, reason backwards:

[Quoted error.message]
    -> Why? cite step, parameter, or screenshot
[Proximate cause]
    -> Why? cite history, parameter, console, or screenshot
[Root cause]

Failure boundary is the last step with status "passed" and the first step with
status "failed" or "broken". The first failed/broken step is what the test was
trying to do when it died.

When a screenshot exists, inspect it. Describe what a person sees: page/tab,
filled fields, empty or populated grid, error banner text, spinner, overlay,
wrong page, etc. The screenshot outweighs a generic Selenium exception.

Never treat these as root causes:
- Element not found / timeout waiting for an element
- Assertion failed
- Click intercepted

Instead ask WHY the element/value/click failed.

Evidence-grounded cues (confirm with the actual evidence):
- Correctly loaded page + locator missed => Script Defect
- Application error/banner on screen => usually Application Defect or Test Data
  Issue, not Script Defect
- "Could not find a row where [column] equals [value]" + matching test
  parameter + chronic history => Test Data Issue
- "Expected X but was Y" on a business value + recent last-passed build =>
  Application Defect
- Config/connection/service unavailable across multiple suites =>
  Environment Issue
- Backend NullPointerException/IndexOutOfBoundsException in application
  packages => Application Defect

REASONING PHASE 3 — DIFFERENTIAL DIAGNOSIS (MANDATORY)
For every group, explicitly accept or reject ALL FOUR categories internally,
with evidence:

Application Defect:
- wrong business value
- backend exception in trace
- invalid UI state for valid input
Accept/reject based on evidence.

Environment Issue:
- connection error
- service unavailable
- missing shared config/permissions
- SSL/host timeout affecting relevant scope
Accept/reject based on evidence.

Test Data Issue:
- named account/security/rate/record not found
- seeded data absent/stale/locked/wrong state
- prerequisite workflow not completed
Accept/reject based on evidence and named parameters.

Script Defect:
- automation is brittle/outdated while page and data are correct
- screenshot confirms a normal ready screen
Accept/reject based on evidence.

The surviving hypothesis is the category.
If two hypotheses survive, choose the one with stronger evidence and set
confidence Medium.
If none survive, use Unknown.

When evidence genuinely leaves multiple hypotheses possible, do not manufacture
certainty.

CATEGORY NAMES — USE EXACTLY
Application Defect
Environment Issue
Test Data Issue
Script Defect
Unknown

CATEGORY DEFINITIONS
Application Defect = production code bug: wrong business data, wrong status,
backend exception, or UI showing an incorrect state.
Environment Issue = infrastructure/configuration problem: connection refused,
5xx, missing shared configuration/permissions, SSL problem, host timeout.
Test Data Issue = application is behaving as expected but a seeded record is
missing, stale, locked, or in the wrong state.
Script Defect = automation is brittle/outdated while the application page/data
is correct.
Unknown = message, steps, and screenshot remain ambiguous.

TIE-BREAK ORDER
1. Application Defect
2. Environment Issue
3. Test Data Issue
4. Script Defect
5. Unknown

Tie-break rules:
- Correct page + failed locator => Script Defect.
- "Not found" for a specific account/security/rate => Test Data Issue.
- "Not configured" for a shared service => Environment Issue.
- Expected X but was Y => Application Defect even if wrapped in RuntimeException.
- Broken 5+ consecutive builds with a missing-record or pending-state message
  => Test Data Issue.
- Application error banner + same banner on retries => not Script Defect.

SEVERITY
Critical = blocks release, core journey, or data corruption.
High = significant feature, many users, or large failure group.
Medium = non-critical and workaround exists.
Low = cosmetic, edge case, or single test.

CONFIDENCE
High = all four alternatives rejected and screenshot + error + steps agree.
Medium = primary evidence fits but one alternative is not fully ruled out.
Low = generic error, no screenshot, or steps add no context. Explain why.

RELEASE ASSESSMENT
NO_GO = any Critical Application Defect, or a cluster of Critical failures.
HOLD = High severity, or 3+ tests in a High-severity group.
CONDITIONAL_GO = only Medium/Low issues.
GO = no application defects; remaining issues do not affect users.

REPORT WRITING
Write complete English sentences. Do not dump internal reasoning into fields.
The UI reader must understand the cause without opening Allure.

root_cause must state:
1. what the test was doing, including named parameters when relevant;
2. what the application/data actually did, including exact visible UI text when
   available;
3. why the automation check failed, including the exact locator/assertion/error
   without treating a generic Selenium timeout as the underlying cause.

evidence MUST contain exactly five labeled lines:
Last passed: "<full last-passed step name>"
Failed at: "<full first-failed step name>"
Error: <exception class> — "<short error text, not full xpath>"
Screenshot: <full attachments[].name> (<source>) — <one-line description>
History: consecutive_failures=N; retries_count=N; retry_same_error=true/false; last_passed_build=#X or none

If there is no screenshot:
Screenshot: No screenshot was attached.

suggested_fix = one concrete action, normally one or two sentences.
recommended_action = role + concrete next step; never merely "investigate".
shared_root_cause = one plain-English mechanism supported for every test in group.

OUTPUT JSON — RETURN NOTHING ELSE
{
  "release_assessment": {
    "recommendation": "GO|CONDITIONAL_GO|HOLD|NO_GO",
    "risk_level": "Critical|High|Medium|Low",
    "reason": "One-sentence plain-English reason"
  },
  "groups": [
    {
      "group_id": "G-01",
      "title": "Short title of the shared root cause",
      "category": "Application Defect|Environment Issue|Test Data Issue|Script Defect|Unknown",
      "severity": "Critical|High|Medium|Low",
      "shared_root_cause": "Plain-English mechanism for every test in this group.",
      "suggested_fix": "Specific, actionable fix.",
      "suggested_owner": "Application|QA Automation|QA|DevOps|Data|Engineering",
      "tests": ["exact test name 1", "exact test name 2"]
    }
  ],
  "findings": [
    {
      "test_name": "exact test name from triage-evidence.json",
      "group_id": "G-01",
      "category": "Application Defect|Environment Issue|Test Data Issue|Script Defect|Unknown",
      "severity": "Critical|High|Medium|Low",
      "confidence": "High|Medium|Low",
      "root_cause": "English mechanism including parameters, quoted UI text, locator, and exact error.",
      "evidence": "Exactly five labeled lines as specified above.",
      "suggested_fix": "Specific actionable fix.",
      "recommended_action": "Who does what next.",
      "suggested_owner": "Application|QA Automation|QA|DevOps|Data|Engineering|Unknown"
    }
  ]
}
"""

def _extract_openai_text(response: Any) -> str:
    """Extract text from an OpenAI Responses API response."""
    text = getattr(response, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()

    # Defensive fallback for SDK/model response shapes.
    parts: List[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            value = getattr(content, "text", None)
            if isinstance(value, str):
                parts.append(value)
    return "\n".join(parts).strip()


def _extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            raise ValueError("OpenAI did not return a JSON object.")
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("OpenAI response must be a JSON object.")
    return value


def _image_format(path: Path) -> Optional[str]:
    suffix = path.suffix.lower()
    return {
        ".png": "png",
        ".jpg": "jpeg",
        ".jpeg": "jpeg",
        ".webp": "webp",
    }.get(suffix)


def _openai_image_inputs(
    evidence: Dict[str, Any], max_images: int
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Build Responses API image inputs as data URLs.

    The manifest uses the exact Allure attachment filename so the model can
    connect the image to the evidence dossier.
    """
    paths: List[Path] = []
    seen = set()

    for test in evidence.get("failing_tests") or []:
        for raw in test.get("screenshots") or []:
            p = Path(raw)
            if p.exists() and p.is_file():
                resolved = str(p.resolve())
                fmt = _image_format(p)
                if fmt and resolved not in seen:
                    paths.append(p)
                    seen.add(resolved)

    paths = paths[:max_images]
    inputs: List[Dict[str, Any]] = []
    manifest: List[str] = []

    for idx, path in enumerate(paths, 1):
        fmt = _image_format(path)
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        data_url = f"data:image/{fmt};base64,{encoded}"

        manifest.append(
            f"IMAGE {idx}: attachment={path.name}; source={path.name}"
        )

        inputs.append({
            "type": "input_image",
            "image_url": data_url,
            "detail": os.getenv("OPENAI_IMAGE_DETAIL", "high"),
        })

    return inputs, "\n".join(manifest)


def call_openai(evidence: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run the same triage reasoning through the OpenAI Responses API.

    Environment variables:
      OPENAI_API_KEY       required
      OPENAI_MODEL         optional; default gpt-5.6-luna
      OPENAI_MAX_SCREENSHOTS
      OPENAI_MAX_OUTPUT_TOKENS
      OPENAI_REASONING_EFFORT
    """
    if OpenAI is None:
        raise RuntimeError(
            "The openai package is required. Install it with: pip install openai"
        )

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Example: export OPENAI_API_KEY='your-api-key'"
        )

    model = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")
    max_images = int(os.getenv("OPENAI_MAX_SCREENSHOTS", "100"))
    max_output_tokens = int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "20000"))
    reasoning_effort = os.getenv("OPENAI_REASONING_EFFORT", "high")

    image_inputs, manifest = _openai_image_inputs(evidence, max_images)

    # Do not send local filesystem paths as if they were accessible to the
    # model. Keep only the attachment filename in the JSON dossier.
    evidence_for_model = json.loads(json.dumps(evidence))
    for test in evidence_for_model.get("failing_tests") or []:
        test["screenshots"] = [
            Path(x).name for x in test.get("screenshots") or []
        ]

    user_text = (
        "Analyze this CI failure evidence. Follow the system instructions "
        "exactly. Inspect every supplied screenshot that corresponds to an "
        "attachment name in the evidence. Return ONLY the required JSON object.\n\n"
        "SCREENSHOT MANIFEST:\n"
        f"{manifest or 'No screenshots were supplied.'}\n\n"
        "EVIDENCE DOSSIER:\n"
        f"{json.dumps(evidence_for_model, ensure_ascii=False, indent=2)}"
    )

    content: List[Dict[str, Any]] = [
        {"type": "input_text", "text": user_text}
    ]
    content.extend(image_inputs)

    client = OpenAI()

    kwargs: Dict[str, Any] = {
        "model": model,
        "instructions": TRIAGE_SYSTEM_PROMPT,
        "input": [{"role": "user", "content": content}],
        "max_output_tokens": max_output_tokens,
    }

    # Reasoning-capable models use reasoning.effort rather than temperature.
    # Keep this configurable so the eventual Bedrock migration can map the
    # same quality/latency intent to the client's model settings.
    if reasoning_effort:
        kwargs["reasoning"] = {"effort": reasoning_effort}

    response = client.responses.create(**kwargs)
    text = _extract_openai_text(response)

    if not text:
        raise ValueError("OpenAI returned an empty response.")

    return _extract_json(text)


def call_model(evidence: Dict[str, Any]) -> Dict[str, Any]:
    """Provider-neutral entry point used by the pipeline."""
    return call_openai(evidence)

def fetch_jenkins_console_log(max_lines: int = MAX_CONSOLE_LINES) -> str:
    build_url = safe(os.getenv("BUILD_URL"))
    if not build_url:
        return ""
    base = build_url.rstrip("/") + "/consoleText"
    user = os.getenv("JENKINS_USER")
    token = os.getenv("JENKINS_API_TOKEN") or os.getenv("JENKINS_TOKEN")
    request = urllib.request.Request(base)
    if user and token:
        credentials = base64.b64encode(f"{user}:{token}".encode()).decode()
        request.add_header("Authorization", f"Basic {credentials}")
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            text = response.read().decode("utf-8", errors="replace")
        return "\n".join(text.splitlines()[-max_lines:])
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(f"      Jenkins console log unavailable: {exc}")
        return ""


def apply_jenkins_metadata(evidence: Dict[str, Any]) -> None:
    build = evidence.setdefault("build", {})
    env_map = {
        "JOB_NAME": "job_name",
        "BUILD_NUMBER": "build_number",
        "BUILD_URL": "build_url",
    }
    for env_name, field in env_map.items():
        value = safe(os.getenv(env_name))
        if value:
            build[field] = value
    if safe(os.getenv("BUILD_URL")):
        build["host"] = re.match(r"^https?://[^/]+", os.getenv("BUILD_URL", "")).group(0)
    build["pipeline_status"] = "FAILURE" if (
        int(evidence.get("metrics", {}).get("failed") or 0)
        + int(evidence.get("metrics", {}).get("broken") or 0)
    ) else "SUCCESS"
    evidence["console_log"] = evidence.get("console_log") or fetch_jenkins_console_log()
    evidence["build"]["source_format"] = "jenkins-report-dir"
    evidence["build"]["source_zip"] = ""


def run_openai(args: argparse.Namespace) -> Dict[str, Any]:
    evidence_path = Path(args.evidence_file)
    analysis_path = Path(args.analysis_file)
    evidence = load_json_object(evidence_path)
    print("=" * 68)
    print("CI FAILURE TRIAGE — OPENAI REASONING")
    print("=" * 68)
    print(f"Evidence dossier        : {evidence_path}")
    print(f"OpenAI model            : {os.getenv('OPENAI_MODEL', 'gpt-5.6-luna')}")
    analysis = call_model(evidence)
    analysis_path.parent.mkdir(parents=True, exist_ok=True)
    analysis_path.write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"OpenAI analysis         : {analysis_path}")
    print(f"Findings returned       : {len(analysis.get('findings') or [])}")
    print("=" * 68)
    return analysis


# ============================================================
# Analysis merge + risk
# ============================================================

def load_json_object(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def finding_lookup(analysis: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    for item in analysis.get("findings") or []:
        if not isinstance(item, dict):
            continue
        name = safe(item.get("test_name"))
        if name:
            lookup[name] = item
            lookup.setdefault(name.strip(), item)
    return lookup


def group_lookup(analysis: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    for group in analysis.get("groups") or []:
        if isinstance(group, dict) and group.get("group_id"):
            lookup[str(group["group_id"])] = group
    return lookup


def sibling_test_names(group: Optional[Dict[str, Any]], current_name: str) -> List[str]:
    """Other tests in the same agent group (`groups[].tests` sharing `group_id`).

    Similarity is that grouping only — not hash clustering or a guessed
    failure-pattern match. A test with no group, or a group of one, has
    no siblings.
    """
    current = safe(current_name).strip()
    siblings: List[str] = []
    seen = {current} if current else set()
    for other in (group or {}).get("tests") or []:
        other_name = safe(other)
        key = other_name.strip()
        if not key or key in seen:
            continue
        siblings.append(other_name)
        seen.add(key)
    return siblings


def merge_findings(evidence: Dict[str, Any], analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
    by_name = finding_lookup(analysis)
    groups = group_lookup(analysis)
    findings: List[Dict[str, Any]] = []
    for test in evidence.get("failing_tests") or []:
        name = test["test_name"]
        ai = by_name.get(name) or by_name.get(name.strip(), {})
        group = groups.get(safe(ai.get("group_id")))
        group_tests = (group or {}).get("tests") or []
        # duplicate_count = agent group size; similar_tests = the other members.
        group_size = len(group_tests)
        siblings = sibling_test_names(group, name)
        findings.append({
            "test_name": name,
            "status": test["status"].upper(),
            "module": test.get("module") or "Unknown",
            "suite": test.get("suite") or "",
            "feature": test.get("feature") or "",
            "duration_ms": test.get("duration_ms") or 0,
            "exception_type": (test.get("error") or {}).get("exception_class") or "",
            "failure_message": (test.get("error") or {}).get("message") or "",
            "stack_trace": (test.get("error") or {}).get("trace") or "",
            "duplicate_count": max(1, group_size),
            "similar_tests": siblings,
            "group_id": safe(ai.get("group_id") or (group or {}).get("group_id")),
            "failure_pattern": safe(
                (group or {}).get("title") or ai.get("failure_pattern"),
                "Individual failure",
            ),
            "category": normalize_category(ai.get("category") or (group or {}).get("category")),
            "severity": normalize_severity(
                ai.get("severity") or (group or {}).get("severity"),
                "Medium",
            ),
            "confidence": normalize_confidence(ai.get("confidence"), "Medium"),
            "root_cause": safe(
                ai.get("root_cause") or (group or {}).get("shared_root_cause"),
                "Insufficient evidence to establish a confirmed root cause.",
            ),
            "evidence": safe(ai.get("evidence"), "No agent evidence was supplied for this test."),
            "suggested_fix": safe(
                ai.get("suggested_fix") or (group or {}).get("suggested_fix"),
                "Reproduce the failure using the captured evidence before applying a change.",
            ),
            "recommended_action": safe(ai.get("recommended_action"), "Review the evidence with the owning team."),
            "suggested_owner": normalize_owner(
                ai.get("suggested_owner") or (group or {}).get("suggested_owner")
            ),
            "screenshots": list(test.get("screenshots") or []),
            "screenshot_srcs": [],
        })
    return findings


def skip_reason_key(message: str) -> str:
    return re.sub(r"\s+", " ", safe(message)).strip() or "(no skip reason recorded)"


def skip_group_title(reason: str) -> str:
    text = skip_reason_key(reason)
    if text == "(no skip reason recorded)":
        return "Skipped with no reason recorded"
    stripped = re.sub(r"^Skipping because\s+", "", text, flags=re.I).strip()
    if not stripped:
        return text
    return stripped[0].upper() + stripped[1:]


def skip_suggested_fix(reason: str) -> str:
    lower = reason.lower()
    if "inventory" in lower and "api" in lower:
        return (
            "No action. These UI inventory tests are skipped because inventory "
            "is seeded via API in this run."
        )
    if "day of the week" in lower or "days to skip" in lower:
        return (
            "No action. This scenario is scheduled to run only on days that are "
            "not in the skip list."
        )
    return "No action unless this skip was unexpected for this environment or schedule."


def load_skipped_tests(evidence: Dict[str, Any]) -> List[Dict[str, Any]]:
    skipped = evidence.get("skipped_tests")
    expected = int((evidence.get("metrics") or {}).get("skipped") or 0)
    if isinstance(skipped, list) and (skipped or expected == 0):
        return skipped
    extracted = Path(evidence.get("extracted_root") or "")
    if not extracted.exists():
        return list(skipped or [])
    tests, _ = collect_tests(extracted)
    recovered = [test for test in tests if test.get("status") == "skipped"]
    evidence["skipped_tests"] = recovered
    return recovered


def merge_skipped_findings(evidence: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Group skipped tests by the Allure skip reason (statusMessage)."""
    skipped_tests = load_skipped_tests(evidence)
    reason_ids: Dict[str, str] = {}
    reason_order: List[str] = []
    for test in skipped_tests:
        key = skip_reason_key((test.get("error") or {}).get("message") or "")
        if key not in reason_ids:
            reason_ids[key] = f"S-{len(reason_order) + 1:02d}"
            reason_order.append(key)

    by_reason: Dict[str, List[str]] = defaultdict(list)
    for test in skipped_tests:
        key = skip_reason_key((test.get("error") or {}).get("message") or "")
        by_reason[key].append(test["test_name"])

    findings: List[Dict[str, Any]] = []
    for test in skipped_tests:
        reason = skip_reason_key((test.get("error") or {}).get("message") or "")
        group_id = reason_ids[reason]
        members = by_reason[reason]
        title = skip_group_title(reason)
        findings.append({
            "test_name": test["test_name"],
            "status": "SKIPPED",
            "module": test.get("module") or "Unknown",
            "suite": test.get("suite") or "",
            "feature": test.get("feature") or "",
            "duration_ms": test.get("duration_ms") or 0,
            "exception_type": "",
            "failure_message": reason,
            "stack_trace": "",
            "duplicate_count": max(1, len(members)),
            "similar_tests": [name for name in members if name != test["test_name"]],
            "group_id": group_id,
            "failure_pattern": title,
            "category": SKIPPED_CATEGORY,
            "severity": "Low",
            "confidence": "High",
            "root_cause": f"The test did not run. TestNG skipped it with: {reason}.",
            "evidence": (
                f"Status: skipped\n"
                f"Reason: {reason}\n"
                f"Error: {(test.get('error') or {}).get('exception_class') or 'org.testng.SkipException'} — \"{reason}\"\n"
                f"Screenshot: No screenshot was attached.\n"
                f"History: consecutive_failures=0; retries_count={int((test.get('history') or {}).get('retries_count') or 0)}; retry_same_error=false; last_passed_build=none"
            ),
            "suggested_fix": skip_suggested_fix(reason),
            "recommended_action": "Confirm the skip condition is expected for this build, then ignore unless the suite should have run.",
            "suggested_owner": "QA",
            "screenshots": list(test.get("screenshots") or []),
            "screenshot_srcs": [],
        })
    return findings


def assess_risk(analysis: Dict[str, Any], findings: List[Dict[str, Any]]) -> Dict[str, Any]:
    release = analysis.get("release_assessment") or {}
    recommendation = normalize_recommendation(release.get("recommendation"))
    highest = max(
        (normalize_severity(item.get("severity")) for item in findings),
        key=lambda item: SEVERITY_ORDER.get(item, 1),
        default="Low",
    )
    largest_group = max(Counter(item.get("group_id") for item in findings).values(), default=0)
    risk_level = normalize_severity(release.get("risk_level"), "Low")
    if SEVERITY_ORDER[highest] > SEVERITY_ORDER[risk_level]:
        risk_level = highest
    if highest == "Critical":
        recommendation = "NO_GO"
        risk_level = "Critical"
    elif highest == "High" and largest_group >= 3:
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
        "highest_severity": highest,
        "affected_failures": len(findings),
        "largest_cluster": largest_group,
    }


# ============================================================
# Reports — HTML matches the existing dashboard structure
# ============================================================

CSS = r"""
:root{--bg:#f7f8fb;--surface:#ffffff;--surface-alt:#fbfcfe;--text:#24324a;--strong:#10213b;--muted:#718096;--faint:#9aa6b7;--border:#e3e8ef;--border-strong:#d5dce6;--primary:#1f5fbf;--primary-dark:#174a99;--primary-soft:#eef4fc;--danger:#c53030;--danger-soft:#fff4f4;--warning:#a15c00;--warning-soft:#fff8ed;--success:#237a4b;--success-soft:#edf8f1;--shadow:0 1px 2px rgba(16,33,59,.03),0 6px 22px rgba(16,33,59,.05);--shadow-open:0 10px 30px rgba(16,33,59,.08)}
*{box-sizing:border-box}html,body{margin:0;min-height:100%;background:var(--bg)}body{font-family:"Montserrat",-apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif;color:var(--text);font-size:14px;line-height:1.55;-webkit-font-smoothing:antialiased;text-rendering:optimizeLegibility}button,label,summary{font:inherit}a{color:inherit}
.topbar{height:64px;background:rgba(255,255,255,.96);border-bottom:1px solid var(--border);display:flex;align-items:center}.topbar-inner{width:100%;max-width:1280px;margin:0 auto;padding:0 28px;display:flex;align-items:center;justify-content:space-between;gap:20px}.brand{display:flex;align-items:center;gap:11px}.brand-mark{width:34px;height:34px;border-radius:9px;background:var(--strong);color:#fff;display:grid;place-items:center;font-size:13px;font-weight:900;letter-spacing:-.02em}.brand-title{font-size:14px;font-weight:800;color:var(--strong);letter-spacing:-.02em}.brand-sub{font-size:10.5px;color:var(--muted);margin-top:1px}.status{font-size:10px;font-weight:800;padding:6px 10px;border-radius:999px;border:1px solid var(--border);background:#fff;color:var(--muted)}.status.failure{color:var(--danger);background:var(--danger-soft);border-color:#f2cdcd}.status.success{color:var(--success);background:var(--success-soft);border-color:#cde7d8}
.container{max-width:1280px;margin:0 auto;padding:30px 28px 44px}.hero{display:flex;align-items:flex-end;justify-content:space-between;gap:24px;margin-bottom:20px}.eyebrow{font-size:9.5px;font-weight:800;letter-spacing:.14em;text-transform:uppercase;color:var(--muted)}h1{margin:5px 0 7px;font-size:31px;line-height:1.08;letter-spacing:-.045em;color:var(--strong);font-weight:800}.build-number{color:#6d7b91;font-weight:700}.hero-meta{font-size:11.5px;color:var(--muted)}.hero-meta a{color:var(--primary);font-weight:750;text-decoration:none}.hero-meta a:hover{text-decoration:underline}
.release{display:flex;align-items:center;justify-content:space-between;gap:18px;background:linear-gradient(180deg,#fff,#fcfdff);border:1px solid var(--border);border-left:3px solid var(--primary);border-radius:13px;padding:15px 18px;margin-bottom:15px;box-shadow:var(--shadow)}.release.hold,.release.no-go{border-left-color:var(--danger)}.release.conditional-go{border-left-color:var(--warning)}.release.go{border-left-color:var(--success)}.release-main{display:flex;align-items:center;gap:12px;min-width:0}.release-icon{font-size:16px;font-weight:900;line-height:1;color:var(--muted)}.release-title{font-size:12.5px;color:var(--strong);font-weight:800}.release-reason{font-size:11.5px;color:var(--muted);margin-top:3px;line-height:1.45}.risk-pill{padding:5px 9px;border:1px solid var(--border);background:#fff;border-radius:999px;font-size:9.5px;font-weight:800;white-space:nowrap;color:var(--muted)}
.stats{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin-bottom:28px}.stat{position:relative;background:var(--surface);border:1px solid var(--border);border-radius:13px;padding:18px 18px 16px;min-height:92px;display:flex;flex-direction:column;justify-content:center;overflow:hidden;box-shadow:0 1px 2px rgba(16,33,59,.025)}.stat:after{content:"";position:absolute;left:0;right:0;bottom:0;height:2px;background:#e9edf3}.stat-value{color:var(--strong);font-size:28px;line-height:1;font-weight:800;letter-spacing:-.05em}.stat-label{font-size:9.5px;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);font-weight:800;margin-top:8px}.stat:nth-child(2) .stat-value{color:var(--success)}.stat:nth-child(3) .stat-value{color:var(--danger)}.stat:nth-child(4) .stat-value{color:var(--muted)}.stat-skip-link{cursor:pointer}.stat-skip-link:hover{border-color:#c8d5e6;box-shadow:0 4px 14px rgba(16,33,59,.05)}#cat-skip:checked~.stats .stat-skip-link,#cat-skip:checked~.category-workspace .stats .stat-skip-link{border-color:#d6e4f7}
.analysis-heading{display:flex;align-items:flex-end;justify-content:space-between;gap:20px;margin-bottom:12px}.analysis-title{font-size:18px;line-height:1.2;color:var(--strong);font-weight:800;letter-spacing:-.03em}.analysis-subtitle{font-size:11px;color:var(--muted);margin-top:3px}.analysis-hint{font-size:10.5px;color:var(--faint);text-align:right}
.category-workspace{margin-top:0}.category-bar{background:rgba(255,255,255,.88);border:1px solid var(--border);border-bottom:0;border-radius:14px 14px 0 0;box-shadow:var(--shadow);overflow:hidden}.category-tabs{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));width:100%}.category-tabs.has-skip{grid-template-columns:repeat(6,minmax(0,1fr))}.category-option{display:block;text-decoration:none;min-width:0;border-right:1px solid var(--border)}.category-option:last-child{border-right:0}.category-card{position:relative;display:flex;align-items:center;justify-content:space-between;gap:10px;padding:15px 16px 14px;min-height:62px;background:#fff;cursor:pointer;white-space:nowrap;transition:background .16s ease,box-shadow .16s ease}.category-card:hover{background:#fbfcfe}.category-card:after{content:"";position:absolute;left:16px;right:16px;bottom:0;height:3px;border-radius:3px 3px 0 0;background:transparent;transition:background .16s ease}.category-card-title{font-size:11.5px;color:#44536b;font-weight:800;overflow:hidden;text-overflow:ellipsis;letter-spacing:-.015em}.category-card-count{display:inline-flex;align-items:center;justify-content:center;min-width:25px;height:23px;padding:0 7px;border-radius:7px;background:#f3f5f8;border:1px solid #e5e9ef;font-size:10px;font-weight:800;color:#6d7b90;flex:0 0 auto}.cat-radio:focus-visible~.category-bar{outline:3px solid rgba(31,95,191,.15);outline-offset:2px}
#cat-app:checked~.category-bar .tab-app,#cat-env:checked~.category-bar .tab-env,#cat-data:checked~.category-bar .tab-data,#cat-script:checked~.category-bar .tab-script,#cat-unknown:checked~.category-bar .tab-unknown,#cat-skip:checked~.category-bar .tab-skip{background:#fff;box-shadow:0 1px 0 #fff inset}.cat-radio:checked~.category-bar .category-card:after{background:transparent}#cat-app:checked~.category-bar .tab-app:after,#cat-env:checked~.category-bar .tab-env:after,#cat-data:checked~.category-bar .tab-data:after,#cat-script:checked~.category-bar .tab-script:after,#cat-unknown:checked~.category-bar .tab-unknown:after,#cat-skip:checked~.category-bar .tab-skip:after{background:var(--primary)}#cat-app:checked~.category-bar .tab-app .category-card-title,#cat-env:checked~.category-bar .tab-env .category-card-title,#cat-data:checked~.category-bar .tab-data .category-card-title,#cat-script:checked~.category-bar .tab-script .category-card-title,#cat-unknown:checked~.category-bar .tab-unknown .category-card-title,#cat-skip:checked~.category-bar .tab-skip .category-card-title{color:var(--strong)}#cat-app:checked~.category-bar .tab-app .category-card-count,#cat-env:checked~.category-bar .tab-env .category-card-count,#cat-data:checked~.category-bar .tab-data .category-card-count,#cat-script:checked~.category-bar .tab-script .category-card-count,#cat-unknown:checked~.category-bar .tab-unknown .category-card-count,#cat-skip:checked~.category-bar .tab-skip .category-card-count{background:var(--primary-soft);border-color:#d6e4f7;color:var(--primary-dark)}
.failures-section{background:#fff;border:1px solid var(--border);border-top:0;border-radius:0 0 14px 14px;box-shadow:var(--shadow);padding:18px 18px 20px}.category-panel{display:none}.category-panel-header{display:flex;align-items:center;justify-content:space-between;gap:18px;padding:3px 2px 14px;margin-bottom:10px;border-bottom:1px solid var(--border)}.category-panel-title{font-size:17px;line-height:1.2;font-weight:800;color:var(--strong);letter-spacing:-.025em}.category-panel-count{font-size:10.5px;color:var(--muted);font-weight:700}.test-list{display:flex;flex-direction:column;gap:7px}.group-item{border:1px solid var(--border);border-radius:12px;background:#fbfcfe;overflow:hidden;transition:border-color .16s ease,box-shadow .16s ease}.group-item:hover{border-color:#c8d5e6}.group-item[open]{border-color:#b9cae1;box-shadow:var(--shadow-open);background:#fff}.group-summary{list-style:none;display:flex;align-items:center;justify-content:space-between;gap:18px;padding:15px 16px;cursor:pointer}.group-summary::-webkit-details-marker{display:none}.group-summary:before{content:"";width:7px;height:7px;border-right:2px solid #7f8b9d;border-bottom:2px solid #7f8b9d;transform:rotate(-45deg);flex:0 0 auto;transition:transform .16s ease,border-color .16s ease;margin-left:1px}.group-item[open]>.group-summary:before{transform:rotate(45deg);border-color:var(--primary)}.group-body{border-top:1px solid var(--border);padding:10px;background:linear-gradient(180deg,#fbfcfe,#f9fbfd)}.group-body .test-list{gap:7px}.test-item{border:1px solid var(--border);border-radius:11px;background:#fff;overflow:hidden;transition:border-color .16s ease,box-shadow .16s ease,transform .16s ease}.test-item:hover{border-color:#c8d5e6;box-shadow:0 4px 14px rgba(16,33,59,.05)}.test-item[open]{border-color:#b9cae1;box-shadow:var(--shadow-open)}.test-summary{list-style:none;display:flex;align-items:center;justify-content:space-between;gap:18px;padding:15px 16px;cursor:pointer}.test-summary::-webkit-details-marker{display:none}.test-summary:before{content:"";width:7px;height:7px;border-right:2px solid #7f8b9d;border-bottom:2px solid #7f8b9d;transform:rotate(-45deg);flex:0 0 auto;transition:transform .16s ease,border-color .16s ease;margin-left:1px}.test-item[open]>.test-summary:before{transform:rotate(45deg);border-color:var(--primary)}.test-main{min-width:0;flex:1}.test-name{display:block;font-size:13.5px;line-height:1.45;font-weight:800;color:var(--strong);overflow-wrap:anywhere}.test-secondary{margin-top:5px;display:flex;gap:7px;flex-wrap:wrap;color:var(--muted);font-size:10.5px}.test-secondary span:nth-child(even){color:#b6beca}.test-side{display:flex;align-items:center;flex-wrap:wrap;justify-content:flex-end;gap:6px;flex-shrink:0}.badge{display:inline-flex;align-items:center;padding:5px 9px;border-radius:999px;font-size:9.5px;font-weight:800;white-space:nowrap;border:1px solid transparent}.sev-critical,.sev-high{color:var(--danger);background:var(--danger-soft);border-color:#f5d1d1}.sev-medium{color:var(--warning);background:var(--warning-soft);border-color:#f1dfbf}.sev-low{color:var(--success);background:var(--success-soft);border-color:#d1e9dc}.badge.skipped{color:var(--muted);background:#f6f7f9;border-color:#e6e9ee}.conf-high{color:var(--primary-dark);background:var(--primary-soft);border-color:#d6e4f7}.conf-medium{color:#5f55a3;background:#f3f1fd;border-color:#ded9f6}.conf-low{color:var(--muted);background:#f6f7f9;border-color:#e6e9ee}
.test-detail{border-top:1px solid var(--border);padding:18px 20px 20px;background:linear-gradient(180deg,#fbfcfe,#f9fbfd)}.failure-pattern{border:1px solid #dbe3ee;border-radius:10px;background:#fff;padding:13px 14px 14px;margin-bottom:12px}.failure-pattern .info-label{margin-bottom:6px}.pattern-text{display:block;color:var(--strong);font-size:12.5px;font-weight:700;line-height:1.55}.detail-grid{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:9px}.snapshot-grid{margin-bottom:14px}.snapshot-grid .info-card:nth-child(1),.snapshot-grid .info-card:nth-child(2),.snapshot-grid .info-card:nth-child(3){grid-column:span 2}.snapshot-grid .info-card:nth-child(n+4){grid-column:span 3}.analysis-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}.analysis-grid .analysis-owner{align-self:start}.info-card{border:1px solid var(--border);border-radius:10px;background:#fff;padding:13px 14px}.info-card.full{grid-column:1/-1}.snapshot-grid .info-card{padding:11px 12px;background:#fbfcfe}.snapshot-heading{font-size:10px;font-weight:800;letter-spacing:.11em;text-transform:uppercase;color:var(--muted);margin:0 0 8px}.analysis-grid .info-card{padding:15px 16px}.analysis-grid .analysis{background:#fff}.analysis-grid .analysis .info-label{color:var(--primary-dark)}.info-label{font-size:8.75px;font-weight:800;text-transform:uppercase;letter-spacing:.11em;color:var(--muted);margin-bottom:6px}.info-value{font-size:12.5px;line-height:1.65;color:var(--text);white-space:pre-wrap;overflow-wrap:anywhere}.trace{font:10.5px/1.55 ui-monospace,SFMono-Regular,Consolas,monospace}.disclosure{margin-top:14px;border:1px solid var(--border);border-radius:10px;overflow:hidden;background:#fff}.disclosure+.disclosure{margin-top:8px}.disclosure summary{cursor:pointer;padding:12px 14px;font-size:10.5px;font-weight:800;color:var(--strong);background:#fff}.disclosure summary:hover{background:#fbfcfe}.disclosure .tech-block{border-top:1px solid var(--border);padding:13px 14px;background:#f7f8fa}.disclosure pre{margin:8px 0 0;padding:12px;border-radius:8px;background:#18283f;color:#eef3f8;max-height:280px;overflow:auto;white-space:pre-wrap;overflow-wrap:anywhere;font:10.5px/1.55 ui-monospace,SFMono-Regular,Consolas,monospace}.tech-label{margin-top:12px}.tech-section+.tech-section{margin-top:12px}.evidence-list{margin:0;display:grid;grid-template-columns:108px minmax(0,1fr);gap:7px 14px;align-items:start}.evidence-list dt{margin:0;font-size:8.75px;font-weight:800;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);padding-top:2px}.evidence-list dd{margin:0;font-size:12px;line-height:1.5;color:var(--text);overflow-wrap:anywhere}.evidence-list dt.shot,.evidence-list dd.shot{grid-column:1/-1}.evidence-shot{display:block;width:100%;max-height:360px;object-fit:contain;object-position:top left;border:1px solid var(--border);border-radius:8px;background:#fff}.evidence-caption{margin-top:6px;font-size:11.5px;color:var(--muted);line-height:1.45}.empty{padding:48px 20px;text-align:center;color:var(--muted);font-size:11.5px}.footer{color:var(--faint);text-align:center;font-size:9.5px;margin-top:18px}
.cat-radio{position:absolute;opacity:0;pointer-events:none}#cat-app:checked~.failures-section .panel-app,#cat-env:checked~.failures-section .panel-env,#cat-data:checked~.failures-section .panel-data,#cat-script:checked~.failures-section .panel-script,#cat-unknown:checked~.failures-section .panel-unknown,#cat-skip:checked~.failures-section .panel-skip{display:block}
@media (max-width:900px){.container{padding:22px 16px 30px}.topbar-inner{padding:0 16px}.stats{grid-template-columns:repeat(3,1fr)}.category-card{padding-left:12px;padding-right:12px}.category-card-title{font-size:10.5px}.analysis-hint{display:none}}
@media (max-width:700px){.container{padding:16px 10px 24px}.topbar{height:60px}.brand-sub{display:none}.stats{grid-template-columns:repeat(2,1fr);gap:9px;margin-bottom:22px}.stat{min-height:78px;padding:14px}.stat-value{font-size:24px}.hero{flex-direction:column;align-items:flex-start;gap:6px;margin-bottom:16px}h1{font-size:26px}.category-bar{overflow-x:auto}.category-tabs{grid-template-columns:repeat(5,minmax(150px,1fr));min-width:800px}.category-tabs.has-skip{grid-template-columns:repeat(6,minmax(150px,1fr));min-width:960px}.failures-section{padding:14px}.category-panel-header{align-items:flex-start;flex-direction:column;gap:4px}.test-summary,.group-summary{align-items:flex-start}.test-side{justify-content:flex-start}.detail-grid{grid-template-columns:1fr}.snapshot-grid .info-card:nth-child(n){grid-column:auto}.analysis-grid{grid-template-columns:1fr}.info-card.full{grid-column:auto}.release{align-items:flex-start;flex-direction:column}.risk-pill{align-self:flex-start}}
@media print{.category-panel{display:block!important}.category-bar{box-shadow:none}.failures-section{box-shadow:none}.test-item,.group-item{break-inside:avoid}}
"""


def severity_css_class(value: Any) -> str:
    return "sev-" + str(value or "low").strip().lower().replace(" ", "-")


def info_card_html(label: str, value: Any, full: bool = False, extra: str = "") -> str:
    classes = "info-card" + (" full" if full else "")
    value_class = "info-value" + ((" " + extra) if extra else "")
    return (
        f'<div class="{classes}"><div class="info-label">{html_escape(label)}</div>'
        f'<div class="{value_class}">{html_escape(value or "Not available")}</div></div>'
    )


def cluster_category_findings(items: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """Groups first (first-seen order), then singleton tests (first-seen order)."""
    groups: List[List[Dict[str, Any]]] = []
    singles: List[List[Dict[str, Any]]] = []
    seen = set()
    for finding in items:
        group_id = safe(finding.get("group_id"))
        if not group_id:
            singles.append([finding])
            continue
        if group_id in seen:
            continue
        seen.add(group_id)
        members = [item for item in items if safe(item.get("group_id")) == group_id]
        if len(members) > 1:
            groups.append(members)
        else:
            singles.append(members or [finding])
    return groups + singles


def compact_error_text(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    if not cleaned:
        return ""
    cut = re.search(
        r"(.*?)(?:waiting for presence of element located by:|By\.xpath:)",
        cleaned,
        re.I,
    )
    if cut and cut.group(1).strip():
        base = cut.group(1).strip().rstrip(":").rstrip().rstrip('"').rstrip()
        return base + " (full locator in Failure message below)"
    if len(cleaned) > 280:
        return cleaned[:277].rstrip() + "..."
    return cleaned


def compact_history_text(text: str) -> str:
    source = text or ""
    source = re.sub(r"recent_statuses\s*=\s*\[[^\]]*\]", "", source)
    parts: List[str] = []
    if "chronic" in source.lower():
        parts.append("chronic")
    consecutive = re.search(r"consecutive_failures\s*=\s*(\d+)", source)
    if consecutive:
        parts.append(f"consecutive_failures={consecutive.group(1)}")
    retries = re.search(r"retries_count\s*=\s*(\d+)", source)
    if retries:
        parts.append(f"retries_count={retries.group(1)}")
    if re.search(r"retry_same_error\s*=\s*true", source, re.I):
        parts.append("retry_same_error=true")
    last_passed = re.search(
        r"last_passed_build[^\n#]*?(#[\w.-]+|none[^\s,;.]*)",
        source,
        re.I,
    )
    if last_passed:
        parts.append(f"last_passed_build={last_passed.group(1)}")
    elif re.search(r"no last passed", source, re.I):
        parts.append("last_passed_build=none")
    if parts:
        return "; ".join(parts)
    cleaned = re.sub(r"\s+", " ", source).strip(" .;")
    return cleaned[:220] + ("..." if len(cleaned) > 220 else "")


def slice_after(text: str, start: str, stops: List[str]) -> str:
    index = text.find(start)
    if index < 0:
        return ""
    rest = text[index + len(start):]
    end = len(rest)
    for stop in stops:
        found = rest.find(stop)
        if found >= 0:
            end = min(end, found)
    return rest[:end].strip().strip(".")


def parse_evidence_rows(text: str) -> List[Tuple[str, str]]:
    raw = (text or "").strip()
    if not raw:
        return []

    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    labeled = True
    parsed: List[Tuple[str, str]] = []
    if len(lines) >= 2:
        for line in lines:
            if ":" not in line:
                labeled = False
                break
            key, _, value = line.partition(":")
            if not value.strip() or key.lower().startswith("the "):
                labeled = False
                break
            parsed.append((key.strip(), value.strip()))
        if labeled and parsed:
            return parsed

    last_passed = slice_after(
        raw,
        "The last passed step was ",
        ["The first failed step was", "The error was", "Screenshot "],
    ).strip().strip('"')
    failed_at = slice_after(
        raw,
        "The first failed step was ",
        ["The error was", "Screenshot "],
    ).strip().strip('"')
    error = slice_after(raw, "The error was ", ["Screenshot "])
    screenshot = ""
    history = ""
    shot_at = raw.find("Screenshot ")
    if shot_at >= 0:
        after = raw[shot_at + len("Screenshot "):]
        history_at = len(after)
        for marker in (
            " This run failed",
            " this failure is",
            " History:",
            " consecutive_failures",
            " this case has failed",
            " This case has failed",
        ):
            found = after.find(marker)
            if found >= 0:
                history_at = min(history_at, found)
        screenshot = after[:history_at].strip().strip(".")
        history = compact_history_text(after[history_at:])
    else:
        history = compact_history_text(raw)

    rows: List[Tuple[str, str]] = []
    if last_passed:
        rows.append(("Last passed", last_passed))
    if failed_at:
        rows.append(("Failed at", failed_at))
    if error:
        rows.append(("Error", compact_error_text(error)))
    if screenshot:
        rows.append(("Screenshot", screenshot))
    if history:
        rows.append(("History", history))
    return rows or [("Notes", raw)]


def screenshot_caption(text: str) -> str:
    raw = (text or "").strip()
    if not raw:
        return ""
    for marker in (" shows ", " — ", " - "):
        found = raw.find(marker)
        if found >= 0:
            caption = raw[found + len(marker):].strip().strip(".")
            if caption and not caption.lower().endswith(".png"):
                return caption
    if re.search(r"\.(png|jpe?g|gif|webp)\b", raw, re.I) or "No screenshot" in raw:
        return ""
    return raw


def screenshot_row_html(caption: str, sources: List[str]) -> str:
    if not sources:
        body = html_escape(caption) if caption else "No screenshot was attached."
        return f'<dt class="shot">Screenshot</dt><dd class="shot">{body}</dd>'
    images = "".join(
        f'<img class="evidence-shot" src="{html_escape(src)}" alt="{html_escape(caption or "Failure screenshot")}">'
        for src in sources
    )
    note = f'<div class="evidence-caption">{html_escape(caption)}</div>' if caption else ""
    return f'<dt class="shot">Screenshot</dt><dd class="shot">{images}{note}</dd>'


def evidence_html(text: str, screenshot_srcs: Optional[List[str]] = None) -> str:
    rows = parse_evidence_rows(text)
    sources = [safe(src) for src in (screenshot_srcs or []) if safe(src)]
    items: List[str] = []
    saw_shot = False
    for label, value in rows:
        if label.lower() == "screenshot":
            saw_shot = True
            items.append(screenshot_row_html(screenshot_caption(value), sources))
            continue
        items.append(f"<dt>{html_escape(label)}</dt><dd>{html_escape(value)}</dd>")
    if sources and not saw_shot:
        items.append(screenshot_row_html("", sources))
    return f'<dl class="evidence-list">{"".join(items)}</dl>'


def copy_report_screenshots(findings: List[Dict[str, Any]], html_path: Path) -> None:
    dest_dir = html_path.parent / "screenshots"
    dest_dir.mkdir(parents=True, exist_ok=True)
    for finding in findings:
        rels: List[str] = []
        for raw in finding.get("screenshots") or []:
            src = Path(str(raw))
            if not src.is_file():
                continue
            dest = dest_dir / src.name
            if not dest.exists() or dest.stat().st_size != src.stat().st_size:
                shutil.copy2(src, dest)
            rels.append(f"screenshots/{src.name}")
        finding["screenshot_srcs"] = rels


def finding_category_map(findings: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {category: [] for category in DISPLAY_CATEGORIES}
    for finding in findings:
        category = safe(finding.get("category"))
        if category != SKIPPED_CATEGORY:
            category = normalize_category(category)
        finding["category"] = category
        grouped.setdefault(category, []).append(finding)
    return grouped


def generate_html_dashboard(
    build: Dict[str, Any],
    metrics: Dict[str, Any],
    findings: List[Dict[str, Any]],
    risk: Dict[str, Any],
) -> str:
    grouped = finding_category_map(findings)
    recommendation = risk["recommendation"]
    rec_class = recommendation.lower().replace("_", "-")
    rec_icon = "✓" if recommendation == "GO" else "⚠" if recommendation == "CONDITIONAL_GO" else "!"
    pipeline_status = safe(build.get("pipeline_status"), "UNKNOWN")
    pipeline_class = (
        "success" if pipeline_status == "SUCCESS"
        else "failure" if pipeline_status in {"FAILURE", "FAILED"}
        else ""
    )
    job_name = safe(build.get("job_name"), "Allure Report")
    build_number = safe(build.get("build_number"), "N/A")
    timestamp = safe(build.get("timestamp"))

    has_skipped = bool(grouped.get(SKIPPED_CATEGORY))
    category_defs = [
        ("Application Defect", "cat-app", "panel-app"),
        ("Environment Issue", "cat-env", "panel-env"),
        ("Test Data Issue", "cat-data", "panel-data"),
        ("Script Defect", "cat-script", "panel-script"),
        ("Unknown", "cat-unknown", "panel-unknown"),
    ]
    if has_skipped:
        category_defs.append((SKIPPED_CATEGORY, "cat-skip", "panel-skip"))
    default_index = next(
        (idx for idx, (category, _, _) in enumerate(category_defs) if grouped[category]),
        0,
    )
    category_inputs = "".join(
        f'<input class="cat-radio" type="radio" name="category" id="{radio}" '
        f'{"checked" if idx == default_index else ""}>'
        for idx, (_, radio, _) in enumerate(category_defs)
    )
    category_tabs = "".join(
        f'<label class="category-option" for="{radio}">'
        f'<span class="category-card tab-{radio[4:]}">'
        f'<span class="category-card-title">{html_escape(category)}</span>'
        f'<span class="category-card-count">{len(grouped[category])}</span>'
        f"</span></label>"
        for category, radio, _ in category_defs
    )

    def render_detail(finding: Dict[str, Any], in_group: bool = False) -> str:
        snapshot_cards = [
            info_card_html("Status", finding.get("status")),
            info_card_html("Severity", finding.get("severity")),
        ]
        if finding.get("module"):
            snapshot_cards.append(info_card_html("Module", finding.get("module")))
        if finding.get("suite"):
            snapshot_cards.append(info_card_html("Suite", finding.get("suite")))
        is_skip = finding.get("category") == SKIPPED_CATEGORY
        if finding.get("exception_type") and not is_skip:
            snapshot_cards.append(info_card_html("Exception", finding.get("exception_type")))
        if finding.get("feature"):
            snapshot_cards.append(info_card_html("Feature / Story", finding.get("feature")))
        pattern = finding.get("failure_pattern") or ""
        pattern_label = "Skip reason" if is_skip else "Failure pattern"
        cause_label = "Skip reason" if is_skip else "Root Cause Analysis"
        pattern_html = (
            f'<div class="failure-pattern"><div class="info-label">{pattern_label}</div>'
            f'<span class="pattern-text">{html_escape(pattern)}</span></div>'
            if pattern and pattern != "Individual failure" and not in_group
            else ""
        )
        tech = ""
        if not is_skip:
            tech = (
                '<details class="disclosure"><summary>View technical error details</summary>'
                '<div class="tech-block">'
                '<div class="tech-section"><div class="info-label">Evidence</div>'
                f"{evidence_html(finding.get('evidence'), finding.get('screenshot_srcs'))}</div>"
                '<div class="tech-section"><div class="info-label">Failure message</div>'
                f'<div class="info-value trace">{html_escape(finding.get("failure_message") or "No failure message available.")}</div></div>'
                '<div class="tech-section"><div class="info-label">Stack trace</div>'
                f'<pre>{html_escape(finding.get("stack_trace") or "No stack trace available.")}</pre></div>'
                "</div></details>"
            )
        return (
            f'<div class="test-detail">{pattern_html}'
            f'<div class="snapshot-heading">Test snapshot</div>'
            f'<div class="detail-grid snapshot-grid">{"".join(snapshot_cards)}</div>'
            f'<div class="analysis-grid">'
            f'{info_card_html(cause_label, finding.get("root_cause"), True, "analysis")}'
            f'{info_card_html("Suggested Fix", finding.get("suggested_fix"), True, "analysis")}'
            f"</div>{tech}</div>"
        )

    def render_test_item(finding: Dict[str, Any], in_group: bool = False) -> str:
        module = finding.get("module") or finding.get("suite")
        secondary = []
        if module:
            secondary.append(html_escape(module))
        if finding.get("exception_type") and finding.get("category") != SKIPPED_CATEGORY:
            secondary.append(html_escape(finding.get("exception_type")))
        secondary_html = "".join(f"<span>{item}</span><span>•</span>" for item in secondary[:-1])
        if secondary:
            secondary_html += f"<span>{secondary[-1]}</span>"
        is_skip = finding.get("category") == SKIPPED_CATEGORY
        badge_class = "badge skipped" if is_skip else f'badge {severity_css_class(finding.get("severity"))}'
        badge_text = "Skipped" if is_skip else html_escape(finding.get("severity"))
        return (
            f'<details class="test-item">'
            f'<summary class="test-summary">'
            f'<span class="test-main">'
            f'<span class="test-name">{html_escape(finding.get("test_name"))}</span>'
            f'<span class="test-secondary">{secondary_html}</span>'
            f"</span>"
            f'<span class="test-side">'
            f'<span class="{badge_class}">{badge_text}</span>'
            f"</span></summary>{render_detail(finding, in_group=in_group)}</details>"
        )

    def render_group_item(members: List[Dict[str, Any]]) -> str:
        title = safe(members[0].get("failure_pattern")) or "Related failures"
        if title == "Individual failure":
            title = "Related failures"
        count = len(members)
        noun = "test" if count == 1 else "tests"
        secondary_html = f"<span>{count} {noun}</span>"
        tests = "".join(render_test_item(item, in_group=True) for item in members)
        return (
            f'<details class="group-item">'
            f'<summary class="group-summary">'
            f'<span class="test-main">'
            f'<span class="test-name">{html_escape(title)}</span>'
            f'<span class="test-secondary">{secondary_html}</span>'
            f"</span></summary>"
            f'<div class="group-body"><div class="test-list">{tests}</div></div>'
            f"</details>"
        )

    def render_category_body(items: List[Dict[str, Any]], force_groups: bool = False) -> str:
        parts: List[str] = []
        for cluster in cluster_category_findings(items):
            if len(cluster) == 1 and not force_groups:
                parts.append(render_test_item(cluster[0]))
            else:
                parts.append(render_group_item(cluster))
        return "".join(parts)

    panels = []
    for category, _, panel_class in category_defs:
        items = grouped[category]
        is_skip = category == SKIPPED_CATEGORY
        body = render_category_body(items, force_groups=is_skip)
        if not body:
            body = (
                '<div class="empty">No skipped tests in this run.</div>'
                if is_skip
                else '<div class="empty">No failed tests in this category.</div>'
            )
        plural = "s" if len(items) != 1 else ""
        eyebrow = "Skip category" if is_skip else "Failure category"
        count_label = (
            f"{len(items)} skipped test{plural}"
            if is_skip
            else f"{len(items)} failed test{plural}"
        )
        panels.append(
            f'<section class="category-panel {panel_class}">'
            f'<div class="category-panel-header">'
            f'<div><div class="eyebrow">{eyebrow}</div>'
            f'<div class="category-panel-title">{html_escape(category)}</div></div>'
            f'<div class="category-panel-count">{count_label}</div>'
            f"</div><div class=\"test-list\">{body}</div></section>"
        )

    failed = int(metrics.get("failed") or 0) + int(metrics.get("broken") or 0)
    skipped = int(metrics.get("skipped") or 0)
    skipped_stat = (
        f'<label class="stat stat-skip-link" for="cat-skip"><div class="stat-value">{skipped}</div>'
        f'<div class="stat-label">Skipped</div></label>'
        if has_skipped
        else (
            f'<div class="stat"><div class="stat-value">{skipped}</div>'
            f'<div class="stat-label">Skipped</div></div>'
        )
    )
    analysis_subtitle = (
        "Select a category, expand a group to see related tests, then expand a test for the analysis. "
        "Skipped tests are grouped by skip reason."
        if has_skipped
        else "Select a category, expand a group to see related tests, then expand a test for the analysis."
    )
    analysis_hint = "5 failure categories + skipped" if has_skipped else "5 failure categories"
    tabs_class = "category-tabs has-skip" if has_skipped else "category-tabs"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CI Failure Triage — Build {html_escape(build_number)}</title>
<style>{CSS}</style>
</head>
<body>
<div class="app">
<header class="topbar"><div class="topbar-inner"><div class="brand"><div class="brand-mark">T</div><div><div class="brand-title">CI Failure Triage</div><div class="brand-sub">Failure classification &amp; root-cause analysis</div></div></div><div class="status {pipeline_class}">Pipeline: {html_escape(pipeline_status)}</div></div></header>
<main class="container">
<section class="hero"><div><div class="eyebrow">Build overview</div><h1>{html_escape(job_name)} <span class="build-number">#{html_escape(build_number)}</span></h1><div class="hero-meta">Generated {html_escape(timestamp)}</div></div></section>
<section class="release {rec_class}"><div class="release-main"><div class="release-icon">{rec_icon}</div><div><div class="release-title">Release recommendation: {html_escape(recommendation.replace("_", " "))}</div><div class="release-reason">{html_escape(risk["reason"])}</div></div></div><div class="risk-pill">Risk: {html_escape(risk["risk_level"])}</div></section>
<section class="stats"><div class="stat"><div class="stat-value">{metrics.get("total", 0)}</div><div class="stat-label">Total tests</div></div><div class="stat"><div class="stat-value">{metrics.get("passed", 0)}</div><div class="stat-label">Passed</div></div><div class="stat"><div class="stat-value">{failed}</div><div class="stat-label">Failed / broken</div></div>{skipped_stat}<div class="stat"><div class="stat-value">{metrics.get("pass_rate", 0)}%</div><div class="stat-label">Pass rate</div></div></section>
<div class="analysis-heading"><div><div class="analysis-title">Failure analysis</div><div class="analysis-subtitle">{analysis_subtitle}</div></div><div class="analysis-hint">{analysis_hint}</div></div>
<div class="category-workspace">
{category_inputs}
<section class="category-bar"><div class="{tabs_class}">{category_tabs}</div></section>
<section class="failures-section">{"".join(panels)}</section>
</div>
<div class="footer">Generated by the CI Failure Triage Engine</div>
</main></div>
</body></html>"""


def generate_markdown_report(
    build: Dict[str, Any],
    metrics: Dict[str, Any],
    findings: List[Dict[str, Any]],
    risk: Dict[str, Any],
) -> str:
    category_counts = Counter(
        item["category"] for item in findings if item.get("category") != SKIPPED_CATEGORY
    )
    skipped_findings = [item for item in findings if item.get("category") == SKIPPED_CATEGORY]
    failed = int(metrics.get("failed") or 0) + int(metrics.get("broken") or 0)
    skipped = int(metrics.get("skipped") or 0)
    lines = [
        "# CI Failure Triage Report",
        "",
        f"**Job:** {build.get('job_name')}",
        f"**Build:** {build.get('build_number')}",
        f"**Pipeline:** {build.get('pipeline_status')}",
        f"**Generated:** {build.get('timestamp')}",
        "",
        f"**Tests:** {metrics.get('total')}  |  **Passed:** {metrics.get('passed')}  |  "
        f"**Failed:** {failed}  |  **Skipped:** {skipped}  |  **Pass Rate:** {metrics.get('pass_rate')}%",
        f"**Release Assessment:** {risk['recommendation']}  |  **Risk:** {risk['risk_level']}",
        "",
        "## Failure Categories",
        "",
    ]
    for category in CATEGORIES:
        lines.append(f"- **{category}:** {category_counts.get(category, 0)}")
    if skipped_findings:
        lines.append(f"- **Skipped:** {len(skipped_findings)}")
    lines.extend(["", "## Failed Tests", ""])
    if not [item for item in findings if item.get("category") != SKIPPED_CATEGORY]:
        lines.append("No failed or broken tests detected.")
    else:
        for category in CATEGORIES:
            category_findings = [item for item in findings if item["category"] == category]
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
    if skipped_findings:
        lines.extend(["", "## Skipped Tests", ""])
        seen_groups = []
        for finding in skipped_findings:
            group_id = safe(finding.get("group_id"))
            if group_id and group_id in seen_groups:
                continue
            if group_id:
                seen_groups.append(group_id)
            members = [
                item for item in skipped_findings
                if safe(item.get("group_id")) == group_id
            ] if group_id else [finding]
            title = safe(finding.get("failure_pattern")) or skip_group_title(
                finding.get("failure_message") or ""
            )
            lines.extend([f"### {title} ({len(members)})", ""])
            for member in members:
                lines.extend([
                    f"- **{member['test_name']}**",
                    f"  - **Reason:** {member.get('failure_message')}",
                    "",
                ])
    return "\n".join(lines)


# ============================================================
# CLI
# ============================================================

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CI failure triage engine")
    parser.add_argument("--serve", action="store_true", help="Run the Jenkins HTTP API")
    parser.add_argument("--host", default="0.0.0.0", help="API bind address (with --serve)")
    parser.add_argument("--port", type=int, default=8088, help="API port (with --serve)")
    parser.add_argument("--prepare", action="store_true", help="Collect evidence and write triage-evidence.json")
    parser.add_argument("--openai", action="store_true", help="Run OpenAI reasoning from triage-evidence.json")
    parser.add_argument("--render", action="store_true", help="Render HTML/Markdown from triage-analysis.json")
    parser.add_argument(
        "--report-dir",
        help="Existing Jenkins/Allure results directory. No ZIP extraction is performed.",
    )
    parser.add_argument(
        "--zip",
        help="Zip archive to triage (filename or path). Required when input_data/ has more than one zip.",
    )
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--evidence-file", help="Path for the evidence dossier JSON")
    parser.add_argument("--analysis-file", help="Path for agent analysis JSON")
    parser.add_argument("--output", help="Markdown report path")
    parser.add_argument("--html-output", help="HTML dashboard path")
    parser.add_argument("--no-open", action="store_true", help="Do not open the HTML report")
    args = parser.parse_args()
    args.resolved_zip = ""
    args.extract_dir = ""
    args.run_dir = ""
    return args


def resolve_named_zip(raw: str, input_dir: Path) -> Path:
    given = Path(raw)
    if given.exists() and given.is_file():
        if not zipfile.is_zipfile(given):
            raise ValueError(f"Not a valid zip archive: {given}")
        return given.resolve()

    zips = list_input_zips(input_dir)
    wanted = given.name.lower()
    wanted_zip = wanted if wanted.endswith(".zip") else f"{wanted}.zip"
    matches = [
        path for path in zips
        if path.name.lower() in {wanted, wanted_zip} or path.stem.lower() == Path(wanted_zip).stem.lower()
    ]
    unique = list(dict.fromkeys(matches))
    if len(unique) == 1:
        return unique[0].resolve()
    raise FileNotFoundError(
        f"Zip file not found: {raw}. "
        f"Available in {input_dir.resolve()}: {format_zip_list(zips)}"
    )


def resolve_zip(args: argparse.Namespace, required: bool) -> Optional[Path]:
    input_dir = Path(args.input_dir)
    zips = list_input_zips(input_dir)
    if args.zip:
        return resolve_named_zip(args.zip, input_dir)
    if len(zips) == 1:
        return zips[0]
    if not zips:
        if required:
            raise FileNotFoundError(
                f"No zip file found in {input_dir.resolve()}. "
                "Drop an Allure report zip into the input_data folder."
            )
        return None
    raise ValueError(
        "Multiple zip files found. Pass --zip with the archive you want to triage. "
        f"Available: {format_zip_list(zips)}"
    )


def apply_run_paths(args: argparse.Namespace, zip_path: Optional[Path]) -> None:
    output_root = Path(args.output_dir)
    if zip_path:
        slug = run_slug(zip_path)
        run_dir = output_root / slug
        args.resolved_zip = str(zip_path)
        args.extract_dir = str(Path(args.input_dir) / "_extracted" / slug)
    else:
        run_dir = output_root
        args.resolved_zip = ""
        args.extract_dir = str(Path(args.input_dir) / "_extracted")
    args.run_dir = str(run_dir)
    args.evidence_file = args.evidence_file or str(run_dir / "triage-evidence.json")
    args.analysis_file = args.analysis_file or str(run_dir / "triage-analysis.json")
    args.output = args.output or str(run_dir / "triage-report.md")
    args.html_output = args.html_output or str(run_dir / "triage-report.html")


def bind_report_dir_context(args: argparse.Namespace) -> Path:
    report_dir = Path(args.report_dir).expanduser().resolve()
    if not report_dir.exists() or not report_dir.is_dir():
        raise FileNotFoundError(f"Allure report directory not found: {report_dir}")
    output_root = Path(args.output_dir)
    # Jenkins BUILD_NUMBER keeps each build isolated when available.
    slug = safe(os.getenv("BUILD_NUMBER"))
    if not slug:
        slug = report_dir.name or "current-build"
    run_dir = output_root / slug
    args.resolved_zip = ""
    args.extract_dir = str(report_dir)
    args.run_dir = str(run_dir)
    args.evidence_file = args.evidence_file or str(run_dir / "triage-evidence.json")
    args.analysis_file = args.analysis_file or str(run_dir / "triage-analysis.json")
    args.output = args.output or str(run_dir / "triage-report.md")
    args.html_output = args.html_output or str(run_dir / "triage-report.html")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    return report_dir


def run_prepare_report_dir(args: argparse.Namespace) -> Dict[str, Any]:
    report_dir = Path(args.extract_dir)
    print("=" * 68)
    print("CI FAILURE TRIAGE — COLLECT EVIDENCE")
    print("=" * 68)
    print(f"Allure results          : {report_dir}")
    tests, source_format = collect_tests(report_dir)
    evidence = build_evidence(report_dir, None, tests, source_format)
    apply_jenkins_metadata(evidence)
    evidence["extracted_root"] = str(report_dir)
    evidence["report_dir"] = str(Path(args.run_dir).resolve())
    evidence["build"]["timestamp"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    evidence_path = Path(args.evidence_file)
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    failed = evidence["metrics"]["failed"] + evidence["metrics"]["broken"]
    print(f"Tests                   : {evidence['metrics']['total']}")
    print(f"Failed / broken         : {failed}")
    print(f"Skipped                 : {evidence['metrics']['skipped']}")
    print(f"Evidence dossier        : {evidence_path}")
    print("=" * 68)
    return evidence


def bind_run_context(args: argparse.Namespace, require_zip: bool) -> Optional[Path]:
    zip_path = resolve_zip(args, required=require_zip)
    apply_run_paths(args, zip_path)
    Path(args.input_dir).mkdir(parents=True, exist_ok=True)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.run_dir).mkdir(parents=True, exist_ok=True)
    return zip_path


def run_prepare(args: argparse.Namespace) -> Dict[str, Any]:
    zip_path = Path(args.resolved_zip)
    extract_dir = Path(args.extract_dir)
    print("=" * 68)
    print("CI FAILURE TRIAGE — PREPARE EVIDENCE")
    print("=" * 68)
    print(f"[1/2] Extracting {zip_path.name}...")
    extracted_root = extract_zip(zip_path, extract_dir)
    print("[2/2] Parsing Allure results...")
    tests, source_format = collect_tests(extracted_root)
    evidence = build_evidence(extracted_root, zip_path, tests, source_format)
    evidence["report_dir"] = str(Path(args.run_dir).resolve())
    evidence_path = Path(args.evidence_file)
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    failed = evidence["metrics"]["failed"] + evidence["metrics"]["broken"]
    print("=" * 68)
    print("EVIDENCE PREPARED FOR AGENT TRIAGE")
    print(f"Source zip             : {zip_path.name}")
    print(f"Extracted to           : {extracted_root}")
    print(f"Report folder          : {Path(args.run_dir).resolve()}")
    print(f"Tests                  : {evidence['metrics']['total']}")
    print(f"Failed / broken        : {failed}")
    print(f"Skipped                : {evidence['metrics']['skipped']}")
    print(f"Evidence dossier       : {evidence_path}")
    print(f"Write analysis to      : {args.analysis_file}")
    print("=" * 68)
    if failed == 0:
        print("No failing tests found. Nothing to analyse.")
    return evidence


def run_render(args: argparse.Namespace, evidence: Optional[Dict[str, Any]] = None) -> None:
    evidence_path = Path(args.evidence_file)
    analysis_path = Path(args.analysis_file)
    zip_hint = f" --zip {Path(args.resolved_zip).name}" if args.resolved_zip else ""
    if evidence is None:
        if not evidence_path.exists():
            raise FileNotFoundError(
                f"{evidence_path} not found. "
                f"Run `python scripts/triage.py --prepare{zip_hint}` first."
            )
        evidence = load_json_object(evidence_path)
    extracted_root = Path(evidence.get("extracted_root") or args.extract_dir)
    if extracted_root.exists():
        build = evidence.setdefault("build", {})
        apply_executor_metadata(build, extracted_root)
        evidence_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    expected_zip = Path(args.resolved_zip).name if args.resolved_zip else ""
    actual_zip = safe((evidence.get("build") or {}).get("source_zip"))
    if expected_zip and actual_zip and expected_zip != actual_zip:
        raise ValueError(
            f"Evidence in {evidence_path} was extracted from {actual_zip}, "
            f"not {expected_zip}. Re-run --prepare --zip {expected_zip}."
        )
    if not analysis_path.exists():
        raise FileNotFoundError(
            f"{analysis_path} not found. Write the agent analysis before rendering."
        )
    analysis = load_json_object(analysis_path)
    if not isinstance(analysis.get("findings"), list):
        raise ValueError(f"{analysis_path} must contain a 'findings' list")

    print("=" * 68)
    print("CI FAILURE TRIAGE — RENDER REPORT")
    print("=" * 68)
    findings = merge_findings(evidence, analysis)
    skipped_findings = merge_skipped_findings(evidence)
    if "skipped_tests" in evidence:
        evidence_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    html_path = Path(args.html_output)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    copy_report_screenshots(findings + skipped_findings, html_path)
    risk = assess_risk(analysis, findings)
    report_findings = findings + skipped_findings
    markdown = generate_markdown_report(evidence["build"], evidence["metrics"], report_findings, risk)
    html_report = generate_html_dashboard(evidence["build"], evidence["metrics"], report_findings, risk)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(markdown, encoding="utf-8")
    html_path.write_text(html_report, encoding="utf-8")
    if not args.no_open:
        report = Path(args.html_output).resolve()
        webbrowser.open(report.as_uri())
        print(f"HTML Dashboard opened : {report}")
    print("=" * 68)
    print("TRIAGE COMPLETED")
    print(f"Source zip             : {actual_zip or expected_zip or 'N/A'}")
    print(f"Report folder          : {Path(args.run_dir).resolve()}")
    print(f"Release Recommendation : {risk['recommendation']}")
    print(f"Risk Level             : {risk['risk_level']}")
    print(f"Failed / broken tests  : {len(findings)}")
    print(f"Skipped tests          : {len(skipped_findings)}")
    print(f"Markdown Report        : {args.output}")
    print(f"HTML Dashboard         : {args.html_output}")
    print("=" * 68)


def main() -> None:
    args = parse_arguments()
    try:
        if args.serve:
            from server import serve
            serve(args.host, args.port)
            return

        # New Jenkins/Bedrock flow: --report-dir never extracts a ZIP.
        if args.report_dir:
            bind_report_dir_context(args)

            if args.prepare:
                run_prepare_report_dir(args)

            if args.openai:
                if not Path(args.evidence_file).exists():
                    run_prepare_report_dir(args)
                run_openai(args)

            if args.render:
                run_render(args)

            # Convenience: --report-dir alone runs the complete flow.
            if not (args.prepare or args.openai or args.render):
                run_prepare_report_dir(args)
                run_openai(args)
                run_render(args)
            return

        # Legacy ZIP/agent flow remains available for local regression testing.
        if args.openai:
            bind_run_context(args, require_zip=False)
            if not Path(args.evidence_file).exists():
                raise FileNotFoundError(
                    f"{args.evidence_file} not found. Use --report-dir for Jenkins "
                    "or run --prepare first."
                )
            run_openai(args)
            if args.render:
                run_render(args)
            return

        if args.render and not args.prepare:
            bind_run_context(args, require_zip=False)
            run_render(args)
        elif args.prepare and args.render:
            bind_run_context(args, require_zip=True)
            evidence = run_prepare(args)
            run_render(args, evidence)
        else:
            bind_run_context(args, require_zip=True)
            run_prepare(args)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
