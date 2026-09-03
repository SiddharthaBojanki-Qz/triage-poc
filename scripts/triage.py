#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

from openai import OpenAI

TRIAGE_SYSTEM_PROMPT = """You are a CI triage assistant. Given failing test details, produce a
concise Markdown report with these sections:
1. Summary (1-2 sentences, overall build health)
2. Failures table: test name | likely category (regression/flaky/env/test-bug) | one-line cause
3. For each failure: root cause hypothesis and a suggested fix
Be specific and reference the actual error messages/stack traces given. Do not invent details
not present in the input."""

HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>CI Triage Report</title>
<style>
  :root {{
    --blue: #2563eb; --blue-light: #eff6ff;
    --red: #dc2626; --red-light: #fef2f2;
    --gray-bg: #f8fafc; --border: #e2e8f0; --text: #1e293b; --muted: #64748b;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    font-family: -apple-system, "Segoe UI", Roboto, Arial, sans-serif;
    background: var(--gray-bg); color: var(--text); margin: 0; padding: 0;
  }}
  .wrapper {{ max-width: 920px; margin: 0 auto; padding: 40px 24px 80px; }}
  .header {{
    background: linear-gradient(135deg, #1e3a8a, #2563eb);
    color: white; border-radius: 14px; padding: 28px 32px; margin-bottom: 28px;
    box-shadow: 0 4px 16px rgba(37,99,235,0.25);
  }}
  .header h1 {{ margin: 0 0 8px; font-size: 26px; }}
  .header a {{ color: #dbeafe; text-decoration: underline; font-size: 14px; }}
  .card {{
    background: white; border: 1px solid var(--border); border-radius: 12px;
    padding: 24px 28px; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.04);
  }}
  .card h2 {{
    margin-top: 0; font-size: 18px; color: var(--blue);
    border-bottom: 2px solid var(--blue-light); padding-bottom: 10px;
  }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 12px; font-size: 14px; }}
  th {{
    background: var(--blue-light); color: var(--blue); text-align: left;
    padding: 10px 14px; border-bottom: 2px solid var(--border);
  }}
  td {{ padding: 10px 14px; border-bottom: 1px solid var(--border); vertical-align: top; }}
  tr:last-child td {{ border-bottom: none; }}
  .badge {{
    display: inline-block; padding: 3px 10px; border-radius: 999px;
    font-size: 12px; font-weight: 600; background: var(--red-light); color: var(--red);
  }}
  code {{
    background: #f1f5f9; padding: 2px 6px; border-radius: 4px;
    font-size: 0.88em; color: #be185d;
  }}
  pre {{
    background: #0f172a; color: #e2e8f0; padding: 14px 16px; border-radius: 8px;
    overflow-x: auto; font-size: 13px; line-height: 1.5;
  }}
  h3 {{ color: var(--text); font-size: 16px; margin: 20px 0 8px; }}
  ul {{ padding-left: 20px; }}
  li {{ margin-bottom: 10px; line-height: 1.6; }}
  .footer {{ text-align: center; color: var(--muted); font-size: 12px; margin-top: 30px; }}
</style>
</head>
<body>
<div class="wrapper">
  <div class="header">
    <h1>🔍 CI Triage Report</h1>
    <a href="{build_url}">{build_url}</a>
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


def build_prompt(failures, build_url):
    parts = [f"Build: {build_url}", f"Failed/broken test count: {len(failures)}", ""]
    for f in failures:
        parts.append(f"### {f['name']} ({f['status']})")
        parts.append(f"Message: {f['message']}")
        parts.append(f"Trace:\n{f['trace']}")
        parts.append("")
    return "\n".join(parts)


def call_openai(prompt: str) -> str:
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=2000,
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


def write_reports(body_markdown: str, build_url: str, md_path: str, html_path: str):
    Path(md_path).write_text(f"# Triage Report\n\nBuild: {build_url}\n\n{body_markdown}\n")
    html_content = markdown_to_html(body_markdown)
    Path(html_path).write_text(HTML_TEMPLATE.format(build_url=build_url, content=html_content))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--build-url", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--html-output", required=True)
    args = parser.parse_args()

    report_dir = Path(args.report_dir)
    if not report_dir.exists():
        print(f"Report dir not found: {report_dir}", file=sys.stderr)
        sys.exit(1)

    results = load_allure_results(report_dir)
    failures = summarize_failures(results)

    if not failures:
        write_reports("All tests passed. No triage needed.", args.build_url, args.output, args.html_output)
        print("No failures found — wrote a clean-bill-of-health report.")
        return

    prompt = build_prompt(failures, args.build_url)
    report_body = call_openai(prompt)

    write_reports(report_body, args.build_url, args.output, args.html_output)
    print(f"Wrote triage reports to {args.output} and {args.html_output}")


if __name__ == "__main__":
    main()
