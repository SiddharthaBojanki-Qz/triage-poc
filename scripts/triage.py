#!/usr/bin/env python3
"""
CI Failure Triage Script
=========================
Reads Allure test results (either raw *-result.json files, or a generated
report's data/test-cases/*.json files — auto-detected), builds a rich
per-test evidence dossier (labels, history, retries, exception class), sends
it to an LLM for structured triage analysis, and renders a Markdown + HTML
report.

Usage (as run from a Jenkins Post Step):
    python3 scripts/triage.py \
        --report-dir target/allure-results \
        --build-url $BUILD_URL \
        --output triage-report.md \
        --html-output triage-report.html
"""

import argparse
import html
import json
import os
import re
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

# ============================================================
# Fixed taxonomy (validated, never trusted blindly from the model)
# ============================================================

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
OWNERS = ["Application", "QA Automation", "Environment/Infra", "Data", "Unknown"]
RECOMMENDATIONS = ["GO", "CONDITIONAL_GO", "HOLD", "NO_GO"]

MAX_CONSOLE_LINES = 150


# ============================================================
# Small helpers
# ============================================================

def safe(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def esc(value: Any, default: str = "N/A") -> str:
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
    }
    return aliases.get(raw.lower(), "Unknown")


def normalize_choice(value: Any, allowed: List[str], default: str) -> str:
    raw = safe(value, default).title()
    return raw if raw in allowed else default


def normalize_owner(value: Any) -> str:
    raw = safe(value, "Unknown")
    return raw if raw in OWNERS else "Unknown"


def normalize_recommendation(value: Any) -> str:
    raw = safe(value, "HOLD").upper().replace("-", "_").replace(" ", "_")
    return raw if raw in RECOMMENDATIONS else "HOLD"


def exception_class(message: str, trace: str) -> str:
    source = f"{message}\n{trace}"
    match = re.search(r"\b([A-Za-z_][\w.]*(?:Exception|Error|Failure))\b", source)
    return match.group(1) if match else ""


def labels_map(result: Dict[str, Any]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = defaultdict(list)
    for label in result.get("labels") or []:
        name = label.get("name")
        if name and label.get("value") is not None:
            out[str(name)].append(str(label["value"]))
    return out


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


def count_attachments(node: Any) -> int:
    count = 0
    if isinstance(node, list):
        for item in node:
            count += count_attachments(item)
        return count
    if not isinstance(node, dict):
        return 0
    count += len(node.get("attachments") or [])
    for key in ("steps", "beforeStages", "afterStages"):
        count += count_attachments(node.get(key))
    count += count_attachments(node.get("testStage"))
    return count


def summarize_history(extra: Dict[str, Any], case: Dict[str, Any]) -> Dict[str, Any]:
    history = (extra or {}).get("history") or {}
    statistic = history.get("statistic") or {}
    items = history.get("items") or []
    recent = [safe(item.get("status")).lower() for item in items if isinstance(item, dict)]
    current_failed = safe(case.get("status")).lower() in {"failed", "broken"}
    consecutive = 0
    if current_failed:
        consecutive = 1
        for status in recent:
            if status in {"failed", "broken"}:
                consecutive += 1
            else:
                break
    total_runs = int(statistic.get("total") or len(items) or 0)
    bad_runs = int(statistic.get("failed") or 0) + int(statistic.get("broken") or 0)
    return {
        "total_runs": total_runs,
        "bad_runs": bad_runs,
        "consecutive_failures": consecutive,
        "flaky": bool(case.get("flaky")),
        "new_failed": bool(case.get("newFailed")),
        "new_broken": bool(case.get("newBroken")),
    }


def retry_info(extra: Dict[str, Any], message: str) -> Tuple[bool, bool]:
    retries = (extra or {}).get("retries") or []
    if not retries:
        return False, False
    first = retries[0] if isinstance(retries[0], dict) else {}
    details = safe(first.get("statusDetails") or first.get("statusMessage"))
    same = bool(details) and details.strip() == (message or "").strip()
    return True, same


def existing_allure_category(extra: Dict[str, Any]) -> str:
    categories = (extra or {}).get("categories") or []
    if categories and isinstance(categories[0], dict):
        return safe(categories[0].get("name"))
    return ""


# ============================================================
# Dual-format Allure parsing: generated-report test-cases/*.json,
# or raw *-result.json — auto-detected, whichever is present.
# ============================================================

def load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
        print(f"Warning: skipping unreadable file {path.name}: {exc}", file=sys.stderr)
        return None
    return data if isinstance(data, dict) else None


def dossier_from_report_case(case: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    status = safe(case.get("status")).lower()
    if not status:
        return None
    extra = case.get("extra") or {}
    labels = labels_map(case)
    message = safe(case.get("statusMessage"))
    trace = safe(case.get("statusTrace"))
    retried, retry_same = retry_info(extra, message)
    return {
        "test_name": safe(case.get("name") or case.get("fullName"), "Unnamed Test"),
        "status": status,
        "suite": first_label(labels, "suite", "parentSuite", "subSuite"),
        "feature": first_label(labels, "feature", "story", "epic"),
        "module": extract_module(labels),
        "message": message,
        "trace": trace[:3000],
        "exception_class": exception_class(message, trace),
        "history": summarize_history(extra, case),
        "retried": retried,
        "retry_same_error": retry_same,
        "existing_allure_category": existing_allure_category(extra),
        "attachments_count": count_attachments(case),
    }


def dossier_from_raw_result(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    status = safe(result.get("status")).lower()
    if not status:
        return None
    details = result.get("statusDetails") or {}
    labels = labels_map(result)
    message = safe(details.get("message"))
    trace = safe(details.get("trace"))
    return {
        "test_name": safe(result.get("name") or result.get("fullName"), "Unnamed Test"),
        "status": status,
        "suite": first_label(labels, "suite", "parentSuite", "subSuite"),
        "feature": first_label(labels, "feature", "story", "epic"),
        "module": extract_module(labels),
        "message": message,
        "trace": trace[:3000],
        "exception_class": exception_class(message, trace),
        "history": summarize_history({}, result),
        "retried": False,
        "retry_same_error": False,
        "existing_allure_category": "",
        "attachments_count": count_attachments(result),
    }


def collect_tests(report_dir: Path) -> Tuple[List[Dict[str, Any]], str]:
    """Auto-detects generated-report format (data/test-cases/*.json) vs raw
    Allure results (*-result.json) under report_dir, parses whichever is found."""
    case_dirs = [d for d in report_dir.rglob("test-cases") if d.is_dir()]
    if case_dirs:
        tests = []
        for case_dir in case_dirs:
            for file_path in sorted(case_dir.glob("*.json")):
                case = load_json(file_path)
                if not case:
                    continue
                record = dossier_from_report_case(case)
                if record and not case.get("retry"):
                    tests.append(record)
        if tests:
            return tests, "allure-report"

    raw = []
    for file_path in sorted(report_dir.rglob("*-result.json")):
        result = load_json(file_path)
        if not result:
            continue
        record = dossier_from_raw_result(result)
        if record:
            raw.append(record)
    return raw, "allure-results"


def compute_metrics(tests: List[Dict[str, Any]]) -> Dict[str, Any]:
    m = {"total": 0, "passed": 0, "failed": 0, "broken": 0, "skipped": 0}
    for t in tests:
        m["total"] += 1
        status = t["status"]
        if status == "passed":
            m["passed"] += 1
        elif status == "failed":
            m["failed"] += 1
        elif status == "broken":
            m["broken"] += 1
        else:
            m["skipped"] += 1
    m["pass_rate"] = round((m["passed"] / m["total"]) * 100, 1) if m["total"] else 100.0
    return m


# ============================================================
# Console log fallback (build-level failures with no test data at all)
# ============================================================

def fetch_console_log_tail(build_url: str, max_lines: int = MAX_CONSOLE_LINES) -> str:
    url = build_url.rstrip("/") + "/consoleText"
    try:
        req = urllib.request.Request(url)
        user = os.environ.get("JENKINS_USER")
        token = os.environ.get("JENKINS_API_TOKEN")
        if user and token:
            import base64
            creds = base64.b64encode(f"{user}:{token}".encode()).decode()
            req.add_header("Authorization", f"Basic {creds}")
        with urllib.request.urlopen(req, timeout=10) as resp:
            text = resp.read().decode("utf-8", errors="replace")
        return "\n".join(text.splitlines()[-max_lines:])
    except Exception as e:
        print(f"Warning: could not fetch console log ({e})", file=sys.stderr)
        return ""


# ============================================================
# LLM call — structured JSON output, validated against fixed taxonomy
# ============================================================

TRIAGE_SYSTEM_PROMPT = """You are a Principal QA Engineer performing a formal triage review of a
CI build's failing tests. Respond with ONLY a single JSON object (no prose, no markdown fences)
matching exactly this schema:

{
  "executive_summary": "2-4 sentences: overall build health, dominant failure theme, whether
    failures look release-blocking",
  "release_recommendation": "one of GO, CONDITIONAL_GO, HOLD, NO_GO",
  "correlation_notes": "explicit reasoning about which failures likely share a root cause versus
    are independent/unrelated. Say so plainly either way; never assume correlation without
    evidence.",
  "findings": [
    {
      "test_name": "must exactly match a test name given in the input",
      "category": "one of: Application Defect, Environment Issue, Test Data Issue, Script Defect, Unknown",
      "severity": "one of: Critical, High, Medium, Low — based on likely user/business impact",
      "confidence": "one of: High, Medium, Low",
      "root_cause": "specific, evidence-based hypothesis citing the actual error/message given",
      "evidence": "a short quote or paraphrase of the specific error text supporting the hypothesis",
      "suggested_fix": "concrete, technical, actionable — not generic advice",
      "suggested_owner": "one of: Application, QA Automation, Environment/Infra, Data, Unknown"
    }
  ]
}

Category definitions:
- Application Defect: a genuine bug in the product/application code being tested.
- Environment Issue: failure caused by the test environment itself (config, connectivity,
  infra, service availability) rather than the application or the test.
- Test Data Issue: failure caused by missing/invalid/stale test data or fixtures.
- Script Defect: failure caused by a bug in the test automation script/locator/logic itself,
  not the application under test.
- Unknown: insufficient evidence to confidently assign one of the above.

When "history" data is provided for a test (total_runs, bad_runs, consecutive_failures), weight
it heavily: a test that has failed with the same message across many recent runs is a chronic,
well-understood issue — usually higher confidence, though not necessarily lower severity. A test
with no failure history, or a message that differs from its usual pattern, is more likely a fresh
regression and deserves closer scrutiny. If an "existing_allure_category" tag is present, treat it
as a strong hint (the client's own QA team already classified it) but still apply your own
judgment against the 5 fixed categories above — do not just copy it verbatim if it doesn't fit.

Never invent details, file names, or line numbers not present in the input. If evidence is
insufficient for a confident root cause, say so explicitly and set confidence to Low rather than
guessing."""


def build_user_prompt(tests: List[Dict[str, Any]], build_url: str, metrics: Dict[str, Any],
                       console_tail: str) -> str:
    failures = [t for t in tests if t["status"] in ("failed", "broken")]
    parts = [
        f"Build: {build_url}",
        f"Total tests: {metrics['total']}, Passed: {metrics['passed']}, "
        f"Failed/broken: {metrics['failed'] + metrics['broken']}, Skipped: {metrics['skipped']}",
        "",
    ]
    for t in failures:
        parts.append(f"### {t['test_name']}")
        parts.append(f"Status: {t['status']}")
        if t["module"]:
            parts.append(f"Module: {t['module']}")
        if t["feature"] or t["suite"]:
            parts.append(f"Feature/Suite: {t['feature']} / {t['suite']}")
        parts.append(f"Message: {t['message']}")
        if t["exception_class"]:
            parts.append(f"Exception class: {t['exception_class']}")
        h = t["history"]
        if h["total_runs"] > 0:
            parts.append(
                f"History: failed/broken in {h['bad_runs']} of the last {h['total_runs']} runs "
                f"of this test; {h['consecutive_failures']} consecutive failure(s) including this one."
            )
        if t["retried"]:
            parts.append(f"Retried this run: yes, same error on retry: {t['retry_same_error']}")
        if t["existing_allure_category"]:
            parts.append(f"existing_allure_category: {t['existing_allure_category']}")
        if t["attachments_count"]:
            parts.append(f"({t['attachments_count']} attachment(s)/screenshot(s) captured, see Allure report)")
        parts.append(f"Trace:\n{t['trace']}")
        parts.append("")
    if console_tail:
        parts.append("### Console log (tail, for extra context)")
        parts.append(f"```\n{console_tail}\n```")
    return "\n".join(parts)


def call_openai(prompt: str) -> Dict[str, Any]:
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=4000,
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": TRIAGE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    raw = response.choices[0].message.content
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"executive_summary": raw, "release_recommendation": "HOLD",
                "correlation_notes": "", "findings": []}


def normalize_analysis(analysis: Dict[str, Any], tests: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Validate every AI-supplied field against our fixed taxonomy — a category, severity,
    confidence, or owner can never drift from the allowed lists, regardless of what the model
    returned."""
    by_name = {t["test_name"]: t for t in tests}
    findings = []
    for item in analysis.get("findings") or []:
        if not isinstance(item, dict):
            continue
        name = safe(item.get("test_name"))
        test = by_name.get(name, {})
        findings.append({
            "test_name": name or "Unknown test",
            "module": test.get("module", ""),
            "category": normalize_category(item.get("category")),
            "severity": normalize_choice(item.get("severity"), SEVERITIES, "Medium"),
            "confidence": normalize_choice(item.get("confidence"), CONFIDENCES, "Medium"),
            "root_cause": safe(item.get("root_cause"), "Insufficient evidence for a confirmed root cause."),
            "evidence": safe(item.get("evidence"), "No supporting evidence provided."),
            "suggested_fix": safe(item.get("suggested_fix"), "Reproduce the failure before applying a change."),
            "suggested_owner": normalize_owner(item.get("suggested_owner")),
        })
    highest = max((SEVERITY_ORDER[f["severity"]] for f in findings), default=1)
    highest_severity = next((s for s, v in SEVERITY_ORDER.items() if v == highest), "Low")
    recommendation = normalize_recommendation(analysis.get("release_recommendation"))
    if highest_severity == "Critical":
        recommendation = "NO_GO"
    return {
        "executive_summary": safe(analysis.get("executive_summary"), "No summary provided."),
        "recommendation": recommendation,
        "highest_severity": highest_severity,
        "correlation_notes": safe(analysis.get("correlation_notes"), "No correlation notes provided."),
        "findings": findings,
    }


# ============================================================
# Report rendering
# ============================================================

MD_TEMPLATE = """# CI Triage Report

**Build:** {build_url}
**Release recommendation:** {recommendation}
**Overall risk:** {highest_severity}

## Executive Summary

{executive_summary}

## Failure Overview

| Test | Category | Severity | Confidence | Cause |
|---|---|---|---|---|
{failure_rows}

## Correlation Analysis

{correlation_notes}

## Detailed Findings

{detailed_findings}
"""

HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>CI Triage Report</title>
<style>
  :root {{
    --blue: #2563eb; --blue-light: #eff6ff;
    --red: #dc2626; --red-light: #fef2f2;
    --amber: #d97706; --amber-light: #fffbeb;
    --green: #16a34a; --green-light: #f0fdf4;
    --gray-bg: #f8fafc; --border: #e2e8f0; --text: #1e293b; --muted: #64748b;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    font-family: -apple-system, "Segoe UI", Roboto, Arial, sans-serif;
    background: var(--gray-bg); color: var(--text); margin: 0; padding: 0;
  }}
  .wrapper {{ max-width: 980px; margin: 0 auto; padding: 40px 24px 80px; }}
  .header {{
    background: linear-gradient(135deg, #1e3a8a, #2563eb);
    color: white; border-radius: 14px; padding: 28px 32px; margin-bottom: 20px;
    box-shadow: 0 4px 16px rgba(37,99,235,0.25);
  }}
  .header h1 {{ margin: 0 0 8px; font-size: 26px; }}
  .header .sub {{ opacity: 0.85; font-size: 14px; margin-bottom: 10px; }}
  .header a {{ color: #dbeafe; text-decoration: underline; font-size: 14px; }}
  .badge {{
    display: inline-block; padding: 4px 12px; border-radius: 999px; font-weight: 700;
    font-size: 13px; margin-top: 8px;
  }}
  .badge.GO {{ background: var(--green-light); color: var(--green); }}
  .badge.CONDITIONAL_GO {{ background: var(--amber-light); color: var(--amber); }}
  .badge.HOLD {{ background: var(--amber-light); color: var(--amber); }}
  .badge.NO_GO {{ background: var(--red-light); color: var(--red); }}
  .stat-row {{ display: flex; gap: 14px; margin-bottom: 24px; flex-wrap: wrap; }}
  .stat {{
    flex: 1; min-width: 130px; background: white; border: 1px solid var(--border);
    border-radius: 10px; padding: 16px 18px; text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
  }}
  .stat .num {{ font-size: 26px; font-weight: 700; }}
  .stat .label {{ font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.04em; margin-top: 4px; }}
  .stat.pass .num {{ color: var(--green); }}
  .stat.fail .num {{ color: var(--red); }}
  .stat.total .num {{ color: var(--blue); }}
  .card {{
    background: white; border: 1px solid var(--border); border-radius: 12px;
    padding: 24px 28px; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.04);
  }}
  .card h2 {{
    margin-top: 0; font-size: 18px; color: var(--blue);
    border-bottom: 2px solid var(--blue-light); padding-bottom: 10px;
  }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 12px; font-size: 13.5px; }}
  th {{
    background: var(--blue-light); color: var(--blue); text-align: left;
    padding: 10px 12px; border-bottom: 2px solid var(--border);
  }}
  td {{ padding: 10px 12px; border-bottom: 1px solid var(--border); vertical-align: top; }}
  tr:last-child td {{ border-bottom: none; }}
  .sev {{ display: inline-block; padding: 2px 9px; border-radius: 999px; font-size: 12px; font-weight: 700; }}
  .sev.Critical {{ background: var(--red-light); color: var(--red); }}
  .sev.High {{ background: var(--red-light); color: var(--red); }}
  .sev.Medium {{ background: var(--amber-light); color: var(--amber); }}
  .sev.Low {{ background: var(--green-light); color: var(--green); }}
  code {{ background: #f1f5f9; padding: 2px 6px; border-radius: 4px; font-size: 0.88em; color: #be185d; }}
  pre {{ background: #0f172a; color: #e2e8f0; padding: 14px 16px; border-radius: 8px; overflow-x: auto; font-size: 13px; line-height: 1.5; white-space: pre-wrap; }}
  h3 {{ color: var(--text); font-size: 16px; margin: 22px 0 8px; padding-top: 12px; border-top: 1px dashed var(--border); }}
  .kv {{ margin: 4px 0; }}
  .kv b {{ color: var(--muted); font-weight: 600; }}
  .footer {{ text-align: center; color: var(--muted); font-size: 12px; margin-top: 30px; }}
</style>
</head>
<body>
<div class="wrapper">
  <div class="header">
    <h1>&#128269; CI Triage Report</h1>
    <div class="sub">Automated AI-powered failure analysis</div>
    <a href="{build_url}">{build_url}</a><br>
    <span class="badge {recommendation}">{recommendation_label}</span>
  </div>
  <div class="stat-row">
    <div class="stat total"><div class="num">{total}</div><div class="label">Total Tests</div></div>
    <div class="stat pass"><div class="num">{passed}</div><div class="label">Passed</div></div>
    <div class="stat fail"><div class="num">{failed}</div><div class="label">Failed</div></div>
    <div class="stat"><div class="num">{pass_rate}%</div><div class="label">Pass Rate</div></div>
  </div>
  <div class="card">
    <h2>Executive Summary</h2>
    <p>{executive_summary}</p>
  </div>
  <div class="card">
    <h2>Failure Overview</h2>
    <table>
      <tr><th>Test</th><th>Category</th><th>Severity</th><th>Confidence</th><th>Cause</th></tr>
      {failure_rows_html}
    </table>
  </div>
  <div class="card">
    <h2>Correlation Analysis</h2>
    <p>{correlation_notes}</p>
  </div>
  <div class="card">
    <h2>Detailed Findings</h2>
    {detailed_findings_html}
  </div>
  <div class="footer">Generated automatically by the AI Triage Agent</div>
</div>
</body>
</html>
"""


def render_reports(analysis: Dict[str, Any], build_url: str, metrics: Dict[str, Any],
                    md_path: str, html_path: str) -> None:
    findings = analysis["findings"]

    md_rows = "\n".join(
        f"| {f['test_name']} | {f['category']} | {f['severity']} | {f['confidence']} | "
        f"{f['root_cause'][:80]}... |" if len(f['root_cause']) > 80 else
        f"| {f['test_name']} | {f['category']} | {f['severity']} | {f['confidence']} | {f['root_cause']} |"
        for f in findings
    ) or "| _No failures_ | | | | |"

    md_detail = "\n\n".join(
        f"### {f['test_name']}\n"
        f"- **Category:** {f['category']}  \n"
        f"- **Severity:** {f['severity']} (confidence: {f['confidence']})  \n"
        f"- **Root cause:** {f['root_cause']}  \n"
        f"- **Evidence:** {f['evidence']}  \n"
        f"- **Suggested fix:** {f['suggested_fix']}  \n"
        f"- **Suggested owner:** {f['suggested_owner']}"
        for f in findings
    ) or "_No failures to detail._"

    md = MD_TEMPLATE.format(
        build_url=build_url,
        recommendation=analysis["recommendation"],
        highest_severity=analysis["highest_severity"],
        executive_summary=analysis["executive_summary"],
        failure_rows=md_rows,
        correlation_notes=analysis["correlation_notes"],
        detailed_findings=md_detail,
    )
    Path(md_path).write_text(md, encoding="utf-8")

    html_rows = "".join(
        f"<tr><td>{esc(f['test_name'])}</td><td>{esc(f['category'])}</td>"
        f"<td><span class='sev {f['severity']}'>{esc(f['severity'])}</span></td>"
        f"<td>{esc(f['confidence'])}</td><td>{esc(f['root_cause'])}</td></tr>"
        for f in findings
    ) or "<tr><td colspan='5'>No failures</td></tr>"

    html_detail = "".join(
        f"<h3>{esc(f['test_name'])}</h3>"
        f"<div class='kv'><b>Category:</b> {esc(f['category'])}</div>"
        f"<div class='kv'><b>Severity:</b> <span class='sev {f['severity']}'>{esc(f['severity'])}</span> "
        f"(confidence: {esc(f['confidence'])})</div>"
        f"<div class='kv'><b>Root cause:</b> {esc(f['root_cause'])}</div>"
        f"<div class='kv'><b>Evidence:</b> {esc(f['evidence'])}</div>"
        f"<div class='kv'><b>Suggested fix:</b> {esc(f['suggested_fix'])}</div>"
        f"<div class='kv'><b>Suggested owner:</b> {esc(f['suggested_owner'])}</div>"
        for f in findings
    ) or "<p>No failures to detail.</p>"

    rec = analysis["recommendation"]
    rec_labels = {"GO": "GO", "CONDITIONAL_GO": "CONDITIONAL GO", "HOLD": "HOLD", "NO_GO": "NO-GO"}

    html_out = HTML_TEMPLATE.format(
        build_url=esc(build_url, build_url),
        recommendation=rec,
        recommendation_label=rec_labels.get(rec, rec),
        total=metrics["total"], passed=metrics["passed"],
        failed=metrics["failed"] + metrics["broken"], pass_rate=metrics["pass_rate"],
        executive_summary=esc(analysis["executive_summary"]),
        failure_rows_html=html_rows,
        correlation_notes=esc(analysis["correlation_notes"]),
        detailed_findings_html=html_detail,
    )
    Path(html_path).write_text(html_out, encoding="utf-8")


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", required=True,
                         help="Root of either a generated Allure report or a raw allure-results dir")
    parser.add_argument("--build-url", required=True)
    parser.add_argument("--output", required=True, help="Markdown output path")
    parser.add_argument("--html-output", required=True, help="HTML output path")
    parser.add_argument("--no-console-log", action="store_true")
    args = parser.parse_args()

    report_dir = Path(args.report_dir)
    if not report_dir.exists():
        print(f"Report dir not found: {report_dir}", file=sys.stderr)
        sys.exit(1)

    tests, source_format = collect_tests(report_dir)
    metrics = compute_metrics(tests)
    print(f"Parsed {metrics['total']} tests ({source_format}): "
          f"{metrics['passed']} passed, {metrics['failed']} failed, {metrics['broken']} broken")

    console_tail = "" if args.no_console_log else fetch_console_log_tail(args.build_url)

    failures = [t for t in tests if t["status"] in ("failed", "broken")]
    if not failures:
        if not console_tail:
            analysis = normalize_analysis({
                "executive_summary": "All tests passed. No triage needed.",
                "release_recommendation": "GO",
                "correlation_notes": "",
                "findings": [],
            }, tests)
            render_reports(analysis, args.build_url, metrics, args.output, args.html_output)
            print("No failures found — wrote a clean-bill-of-health report.")
            return
        prompt = (f"Build: {args.build_url}\n\nNo individual test failures were recorded, but the "
                  f"build may have failed at an earlier stage. Console log tail:\n```\n{console_tail}\n```")
    else:
        prompt = build_user_prompt(tests, args.build_url, metrics, console_tail)

    raw_analysis = call_openai(prompt)
    analysis = normalize_analysis(raw_analysis, tests)
    render_reports(analysis, args.build_url, metrics, args.output, args.html_output)
    print(f"Wrote triage reports to {args.output} and {args.html_output}")


if __name__ == "__main__":
    main()
