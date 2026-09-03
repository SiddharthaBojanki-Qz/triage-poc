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
    client = OpenAI()  # reads OPENAI_API_KEY from the environment
    response = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=2000,
        messages=[
            {"role": "system", "content": TRIAGE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", required=True, help="Path to allure-results (raw JSON) dir")
    parser.add_argument("--build-url", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    report_dir = Path(args.report_dir)
    if not report_dir.exists():
        print(f"Report dir not found: {report_dir}", file=sys.stderr)
        sys.exit(1)

    results = load_allure_results(report_dir)
    failures = summarize_failures(results)

    if not failures:
        Path(args.output).write_text(
            f"# Triage Report\n\nBuild: {args.build_url}\n\nAll tests passed. No triage needed.\n"
        )
        print("No failures found — wrote a clean-bill-of-health report.")
        return

    prompt = build_prompt(failures, args.build_url)
    report_body = call_openai(prompt)

    Path(args.output).write_text(f"# Triage Report\n\n{report_body}\n")
    print(f"Wrote triage report to {args.output}")


if __name__ == "__main__":
    main()
