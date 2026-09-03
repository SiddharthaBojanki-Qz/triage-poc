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
  body {{ font-family: -apple-system, Segoe UI, Arial, sans-serif; max-width: 900px;
         margin: 40px auto; padding: 0 20px; color: #1a1a1a; line-height: 1.6; }}
  h1 {{ border-bottom: 3px solid #2563eb; padding-bottom: 10px; }}
  h2 {{ color: #2563eb; margin-top: 30px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
  th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
  th {{ background: #f0f4ff; }}
  code, pre {{ background: #f5f5f5; padding: 2px 6px; border-radius: 4px; font-size: 0.9em; }}
  pre {{ padding: 12px; overflow-x: auto; }}
  .meta {{ color: #666; font-size: 0.9em; margin-bottom: 25px; }}
</style>
</head>
<body>
<h1>CI Triage Report</h1>
<p class="meta">Build: <a href="{build_url}">{build_url}</a></p>
{content}
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
    """Minimal Markdown -> HTML conversion (headers, bold, tables, code, paragraphs)."""
    try:
        import markdown
        return markdown.markdown(md_text, extensions=["tables", "fenced_code"])
    except ImportError:
        # Fallback: wrap as preformatted text if the 'markdown' package isn't available
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
    parser.add_argument("--output", required=True, help="Path for the .md output")
    parser.add_argument("--html-output", required=True, help="Path for the .html output")
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
