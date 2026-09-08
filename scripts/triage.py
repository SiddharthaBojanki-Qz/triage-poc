#!/usr/bin/env python3
import argparse
import json
import sys
import os
import urllib.request
from pathlib import Path

from openai import OpenAI

TRIAGE_SYSTEM_PROMPT = """You are a Principal QA Engineer performing a formal triage review of a CI
build's failing tests for an engineering team. Your audience includes both engineers and
non-technical stakeholders (release managers, product owners), so the report must be precise for
engineers but scannable for a five-minute stakeholder read.

You will receive structured per-test failure data (test name, exception/assertion message, stack
trace) and possibly a tail of the raw Jenkins console log for additional build-level context.

Produce a Markdown report with EXACTLY these sections, in this order:

## Executive Summary
2-4 sentences: overall build health, whether failures are release-blocking, and the dominant theme
(e.g. "isolated UI locator drift" vs "systemic API contract regression"). State a clear go/no-go
recommendation for release if the evidence supports one.

## Failure Overview
A Markdown table with columns: Test | Module | Category | Severity | Confidence | One-line Cause.
- Category: one of [ui-locator, timeout, api-contract, concurrency, null-safety, bounds-check,
  logic-bug, flaky, environment, unknown]
- Severity: Critical / High / Medium / Low, based on likely user/business impact, not just whether
  it's an exception vs assertion failure.
- Confidence: High / Medium / Low — how confident you are in the root cause given the evidence.

## Correlation Analysis
Explicitly state which failures are likely related (shared root cause) versus which are independent,
isolated issues. If two failures could plausibly share a cause, say so and explain the reasoning; if
they are unrelated, say that explicitly too. Do not assume correlation without evidence.

## Prioritized Remediation Plan
A numbered, ordered list of what to fix first and why, considering severity, confidence, and blast
radius (how many other tests/features a fix might affect). Include a rough relative effort estimate
(Small/Medium/Large) for each.

## Detailed Findings
For each failure, in the same order as the table:
### <Test Name>
- **Root cause hypothesis:** specific, evidence-based, cites the exact exception/message/line given.
- **Evidence:** quote the specific error text that supports the hypothesis.
- **Suggested fix:** concrete and technical (code-level or config-level), not generic advice.
- **Suggested owner:** which team would likely own this (e.g. Frontend, Backend/API, QA Automation,
  Infra) based on the nature of the failure.

Rules:
- Never invent details, file names, or line numbers not present in the input.
- If evidence is insufficient to determine a root cause confidently, say so explicitly and mark
  Confidence as Low rather than guessing.
- Be direct and specific — avoid vague phrases like "there might be an issue" without grounding them
  in the actual evidence given."""

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
  code {{
    background: #f1f5f9; padding: 2px 6px; border-radius: 4px;
    font-size: 0.88em; color: #be185d;
  }}
  pre {{
    background: #0f172a; color: #e2e8f0; padding: 14px 16px; border-radius: 8px;
    overflow-x: auto; font-size: 13px; line-height: 1.5;
  }}
  h3 {{ color: var(--text); font-size: 16px; margin: 22px 0 8px; padding-top: 12px; border-top: 1px dashed var(--border); }}
  ul, ol {{ padding-left: 20px; }}
  li {{ margin-bottom: 10px; line-height: 1.6; }}
  .footer {{ text-align: center; color: var(--muted); font-size: 12px; margin-top: 30px; }}
</style>
</head>
<body>
<div class="wrapper">
  <div class="header">
    <h1>🔍 CI Triage Report</h1>
    <div class="sub">Automated AI-powered failure analysis</div>
    <a href="{build_url}">{build_url}</a>
  </div>
  <div class="stat-row">
    <div class="stat total"><div class="num">{total}</div><div class="label">Total Tests</div></div>
    <div class="stat pass"><div class="num">{passed}</div><div class="label">Passed</div></div>
    <div class="stat fail"><div class="num">{failed}</div><div class="label">Failed</div></div>
    <div class="stat"><div class="num">{pass_rate}%</div><div class="label">Pass Rate</div></div>
  </div>
  <div class="card">
    {content}
  </div>
  <div class="footer">Generated automatically by the AI Triage Agent</div>
</div>
</body>
</html>
"""


def load_allure_results(report_dir: Path):
    results = []
    for f in report_dir.glob("*-result.json"):
        try:
            results.append(json.loads(f.read_text()))
        except json.JSONDecodeError:
            continue
    return results


def summarize_failures(results):
    failures = []
    for r in results:
        status = r.get("status")
        if status in ("failed", "broken"):
            failures.append({
                "name": r.get("fullName") or r.get("name"),
                "status": status,
                "message": (r.get("statusDetails") or {}).get("message", ""),
                "trace": (r.get("statusDetails") or {}).get("trace", "")[:3000],
            })
    return failures


def fetch_console_log_tail(build_url: str, max_lines: int = 150) -> str:
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
        lines = text.splitlines()
        return "\n".join(lines[-max_lines:])
    except Exception as e:
        print(f"Warning: could not fetch console log ({e})", file=sys.stderr)
        return ""


def build_prompt(failures, build_url, console_tail, total, passed):
    parts = [
        f"Build: {build_url}",
        f"Total tests: {total}, Passed: {passed}, Failed/broken: {len(failures)}",
        "",
    ]
    for f in failures:
        parts.append(f"### {f['name']} ({f['status']})")
        parts.append(f"Message: {f['message']}")
        parts.append(f"Trace:\n{f['trace']}")
        parts.append("")
    if console_tail:
        parts.append("### Console log (tail, for extra context)")
        parts.append(f"```\n{console_tail}\n```")
    return "\n".join(parts)


def call_openai(prompt: str) -> str:
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=4000,
        temperature=0.2,
        messages=[
            {"role": "system", "content": TRIAGE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content


def markdown_to_html(md_text: str) -> str:
    try:
        import markdown
        return markdown.markdown(md_text, extensions=["tables", "fenced_code"])
    except ImportError:
        escaped = md_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        return f"<pre>{escaped}</pre>"


def write_reports(body_markdown, build_url, md_path, html_path, total, passed, failed):
    Path(md_path).write_text(f"# Triage Report\n\nBuild: {build_url}\n\n{body_markdown}\n")
    html_content = markdown_to_html(body_markdown)
    pass_rate = round((passed / total) * 100, 1) if total else 100
    Path(html_path).write_text(HTML_TEMPLATE.format(
        build_url=build_url, content=html_content,
        total=total, passed=passed, failed=failed, pass_rate=pass_rate,
    ))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--build-url", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--html-output", required=True)
    parser.add_argument("--no-console-log", action="store_true")
    args = parser.parse_args()

    report_dir = Path(args.report_dir)
    if not report_dir.exists():
        print(f"Report dir not found: {report_dir}", file=sys.stderr)
        sys.exit(1)

    results = load_allure_results(report_dir)
    # Only count actual test cases (skip container/before/after entries without a status)
    test_results = [r for r in results if r.get("status") is not None]
    total = len(test_results)
    failures = summarize_failures(test_results)
    passed = total - len(failures)

    console_tail = "" if args.no_console_log else fetch_console_log_tail(args.build_url)

    if not failures and not console_tail:
        write_reports("All tests passed. No triage needed.", args.build_url,
                       args.output, args.html_output, total, passed, 0)
        print("No failures found — wrote a clean-bill-of-health report.")
        return

    if not failures:
        prompt = (f"Build: {args.build_url}\n\nNo individual test failures were recorded, but the "
                  f"build may have failed at an earlier stage. Console log tail:\n```\n{console_tail}\n```")
    else:
        prompt = build_prompt(failures, args.build_url, console_tail, total, passed)

    report_body = call_openai(prompt)

    write_reports(report_body, args.build_url, args.output, args.html_output,
                  total, passed, len(failures))
    print(f"Wrote triage reports to {args.output} and {args.html_output}")


if __name__ == "__main__":
    main()
