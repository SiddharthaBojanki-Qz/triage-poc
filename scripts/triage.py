#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

from openai import OpenAI


TRIAGE_SYSTEM_PROMPT = """You are a CI triage assistant. Given failing test details, produce a
concise Markdown report with exactly these sections:

## Summary
Provide a concise 1-2 sentence overview of the overall build health.

## Failures Table
Create a Markdown table with these columns:
Test Name | Likely Category | One-Line Cause

Likely Category must be one of:
regression, flaky, environment, test-bug

## Detailed Failure Analysis
For each failure, create a subsection using the test name as the heading and provide:

- **Root Cause Hypothesis:** Explain the likely cause using only the actual error messages and stack traces provided.
- **Suggested Fix:** Provide a practical recommendation.

Be specific and reference the actual error messages/stack traces given.
Do not invent details not present in the input.
"""


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">

<head>

<meta charset="UTF-8">

<meta name="viewport" content="width=device-width, initial-scale=1.0">

<title>CI Triage Report</title>

<style>

* {{
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}}

body {{
    font-family: Arial, Helvetica, sans-serif;
    background: #f4f7fb;
    color: #1e293b;
    line-height: 1.6;
}}

.wrapper {{
    max-width: 1100px;
    margin: 0 auto;
    padding: 40px 25px 70px;
}}


/* =========================
   HEADER
========================= */

.header {{
    background: #1e40af;
    color: white;
    padding: 35px;
    border-radius: 16px;
    margin-bottom: 30px;
    box-shadow: 0 8px 25px rgba(30, 64, 175, 0.25);
}}

.header-top {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 20px;
}}

.header h1 {{
    font-size: 30px;
    margin-bottom: 8px;
}}

.header p {{
    color: #dbeafe;
    font-size: 15px;
}}

.status-badge {{
    background: #fee2e2;
    color: #dc2626;
    padding: 9px 16px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: bold;
}}

.build-link {{
    display: inline-block;
    margin-top: 20px;
    padding: 10px 16px;
    background: #ffffff;
    color: #1d4ed8;
    text-decoration: none;
    border-radius: 8px;
    font-weight: bold;
    font-size: 14px;
}}


/* =========================
   CONTENT CARDS
========================= */

.card {{
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 30px;
    margin-bottom: 24px;
    box-shadow: 0 3px 12px rgba(0, 0, 0, 0.06);
}}

.card h2 {{
    font-size: 21px;
    color: #1e3a8a;
    margin-bottom: 20px;
    padding-bottom: 12px;
    border-bottom: 2px solid #dbeafe;
}}

.card h3 {{
    color: #0f172a;
    font-size: 17px;
    margin-top: 28px;
    margin-bottom: 15px;
    padding-bottom: 10px;
    border-bottom: 1px solid #e2e8f0;
}}


/* =========================
   PARAGRAPHS
========================= */

p {{
    margin-bottom: 15px;
    color: #475569;
    font-size: 15px;
}}


/* =========================
   TABLE
========================= */

.table-wrapper {{
    overflow-x: auto;
}}

table {{
    width: 100%;
    border-collapse: collapse;
    margin-top: 15px;
    font-size: 14px;
}}

th {{
    background: #eff6ff;
    color: #1d4ed8;
    text-align: left;
    padding: 14px;
    font-weight: bold;
    border-bottom: 2px solid #bfdbfe;
}}

td {{
    padding: 14px;
    border-bottom: 1px solid #e2e8f0;
    vertical-align: top;
    color: #475569;
}}

tr:hover {{
    background: #f8fafc;
}}

tr:last-child td {{
    border-bottom: none;
}}


/* =========================
   LISTS
========================= */

ul {{
    list-style: none;
    padding: 0;
}}

li {{
    background: #f8fafc;
    border-left: 4px solid #2563eb;
    padding: 16px;
    margin-bottom: 14px;
    border-radius: 8px;
    color: #475569;
}}

li strong {{
    color: #0f172a;
}}


/* =========================
   CODE
========================= */

code {{
    background: #f1f5f9;
    color: #be123c;
    padding: 3px 6px;
    border-radius: 4px;
    font-family: Consolas, monospace;
    font-size: 13px;
}}

pre {{
    background: #0f172a;
    color: #e2e8f0;
    padding: 18px;
    border-radius: 10px;
    overflow-x: auto;
    margin: 15px 0;
    font-size: 13px;
}}


/* =========================
   FOOTER
========================= */

.footer {{
    text-align: center;
    color: #64748b;
    font-size: 13px;
    margin-top: 35px;
    padding-top: 20px;
}}


/* =========================
   RESPONSIVE
========================= */

@media (max-width: 700px) {{

    .wrapper {{
        padding: 20px 15px;
    }}

    .header {{
        padding: 25px;
    }}

    .header h1 {{
        font-size: 24px;
    }}

    .card {{
        padding: 20px;
    }}

}}

</style>

</head>


<body>

<div class="wrapper">


    <!-- HEADER -->

    <div class="header">

        <div class="header-top">

            <div>

                <h1>CI Triage Report</h1>

                <p>
                    Automated AI-powered analysis of CI test failures
                </p>

            </div>


            <div class="status-badge">
                BUILD FAILURE
            </div>

        </div>


        <a class="build-link" href="{build_url}">
            View Jenkins Build
        </a>

    </div>


    <!-- REPORT CONTENT -->

    <div class="card">

        {content}

    </div>


    <!-- FOOTER -->

    <div class="footer">

        Generated automatically by the AI Triage Agent

    </div>


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

            status_details = r.get("statusDetails") or {}

            failures.append({
                "name": r.get("fullName") or r.get("name"),
                "status": status,
                "message": status_details.get("message", ""),
                "trace": status_details.get("trace", "")[:3000],
            })

    return failures


def build_prompt(failures, build_url):

    parts = [
        f"Build: {build_url}",
        f"Failed/broken test count: {len(failures)}",
        ""
    ]

    for failure in failures:

        parts.append(
            f"### {failure['name']} ({failure['status']})"
        )

        parts.append(
            f"Message: {failure['message']}"
        )

        parts.append(
            f"Trace:\n{failure['trace']}"
        )

        parts.append("")

    return "\n".join(parts)


def call_openai(prompt: str) -> str:

    client = OpenAI()

    response = client.chat.completions.create(

        model="gpt-4o",

        max_tokens=2000,

        messages=[
            {
                "role": "system",
                "content": TRIAGE_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": prompt
            },
        ],

    )

    return response.choices[0].message.content


def markdown_to_html(md_text: str) -> str:

    try:

        import markdown

        return markdown.markdown(

            md_text,

            extensions=[
                "tables",
                "fenced_code"
            ]

        )

    except ImportError:

        escaped = (
            md_text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )

        return f"<pre>{escaped}</pre>"


def write_reports(
    body_markdown: str,
    build_url: str,
    md_path: str,
    html_path: str
):

    markdown_content = (
        f"# CI Triage Report\n\n"
        f"Build: {build_url}\n\n"
        f"{body_markdown}\n"
    )

    Path(md_path).write_text(
        markdown_content,
        encoding="utf-8"
    )


    html_content = markdown_to_html(body_markdown)


    final_html = HTML_TEMPLATE.format(

        build_url=build_url,

        content=html_content

    )


    Path(html_path).write_text(
        final_html,
        encoding="utf-8"
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--report-dir",
        required=True
    )

    parser.add_argument(
        "--build-url",
        required=True
    )

    parser.add_argument(
        "--output",
        required=True
    )

    parser.add_argument(
        "--html-output",
        required=True
    )


    args = parser.parse_args()


    report_dir = Path(args.report_dir)


    if not report_dir.exists():

        print(
            f"Report dir not found: {report_dir}",
            file=sys.stderr
        )

        sys.exit(1)


    results = load_allure_results(report_dir)


    failures = summarize_failures(results)


    if not failures:

        write_reports(

            body_markdown="""
## Summary

All tests passed successfully. No failures were detected during this build.

## Failures Table

No failed or broken tests were found.

## Detailed Failure Analysis

No failure analysis is required because the build completed without test failures.
""",

            build_url=args.build_url,

            md_path=args.output,

            html_path=args.html_output

        )


        print(
            "No failures found — wrote a clean-bill-of-health report."
        )

        return


    prompt = build_prompt(

        failures,

        args.build_url

    )


    report_body = call_openai(prompt)


    write_reports(

        body_markdown=report_body,

        build_url=args.build_url,

        md_path=args.output,

        html_path=args.html_output

    )


    print(
        f"Wrote triage reports to "
        f"{args.output} and {args.html_output}"
    )


if __name__ == "__main__":
    main()
