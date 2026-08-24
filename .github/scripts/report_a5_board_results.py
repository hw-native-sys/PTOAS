#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BoardSummary:
    results_found: bool
    counts: Counter[str]
    failed_cases: tuple[str, ...]

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    @property
    def passed(self) -> bool:
        return (
            self.results_found
            and self.total > 0
            and self.counts.get("FAIL", 0) == 0
            and set(self.counts).issubset({"OK", "SKIP"})
        )


def load_results(path: Path) -> BoardSummary:
    if not path.is_file():
        return BoardSummary(False, Counter(), ())

    counts: Counter[str] = Counter()
    failed_cases: list[str] = []
    with path.open(encoding="utf-8", errors="replace", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        for row in reader:
            status = (row.get("status") or "UNKNOWN").strip() or "UNKNOWN"
            testcase = (row.get("testcase") or "<unknown>").strip() or "<unknown>"
            counts[status] += 1
            if status == "FAIL":
                failed_cases.append(testcase)
    return BoardSummary(True, counts, tuple(failed_cases))


def render_markdown(
    summary: BoardSummary,
    *,
    conclusion: str,
    run_url: str,
    sha: str,
) -> str:
    succeeded = conclusion == "success" and summary.passed
    lines = [
        "## A5 nightly board validation",
        "",
        f"- Status: **{'PASS' if succeeded else 'FAIL'}**",
        f"- Commit: `{sha[:12] or 'unknown'}`",
        f"- Results: OK `{summary.counts.get('OK', 0)}` / "
        f"FAIL `{summary.counts.get('FAIL', 0)}` / "
        f"SKIP `{summary.counts.get('SKIP', 0)}` / TOTAL `{summary.total}`",
    ]
    if run_url:
        lines.append(f"- Run: {run_url}")
    if not summary.results_found:
        lines.extend(["", "No board result TSV was produced. Inspect the workflow log."])
    elif summary.failed_cases:
        lines.extend(["", "### Failed cases", ""])
        lines.extend(f"- `{case}`" for case in summary.failed_cases[:50])
        if len(summary.failed_cases) > 50:
            lines.append(f"- ... and {len(summary.failed_cases) - 50} more")
    return "\n".join(lines) + "\n"


def build_feishu_payload(
    summary: BoardSummary,
    *,
    conclusion: str,
    run_url: str,
    sha: str,
) -> dict[str, object]:
    succeeded = conclusion == "success" and summary.passed
    status = "PASS" if succeeded else "FAIL"
    detail_lines = [
        f"**Status**: {status}",
        f"**Commit**: `{sha[:12] or 'unknown'}`",
        f"**Results**: OK `{summary.counts.get('OK', 0)}` / "
        f"FAIL `{summary.counts.get('FAIL', 0)}` / "
        f"SKIP `{summary.counts.get('SKIP', 0)}` / TOTAL `{summary.total}`",
    ]
    if not summary.results_found:
        detail_lines.append("**Error**: result TSV was not produced")
    elif summary.failed_cases:
        failed = ", ".join(summary.failed_cases[:20])
        if len(summary.failed_cases) > 20:
            failed += f", ... (+{len(summary.failed_cases) - 20})"
        detail_lines.append(f"**Failed cases**: {failed}")

    elements: list[dict[str, object]] = [
        {
            "tag": "div",
            "text": {"tag": "lark_md", "content": "\n".join(detail_lines)},
        }
    ]
    if run_url:
        elements.append(
            {
                "tag": "action",
                "actions": [
                    {
                        "tag": "button",
                        "text": {"tag": "plain_text", "content": "Open GitHub run"},
                        "url": run_url,
                        "type": "primary",
                    }
                ],
            }
        )
    return {
        "msg_type": "interactive",
        "card": {
            "header": {
                "template": "green" if succeeded else "red",
                "title": {
                    "tag": "plain_text",
                    "content": f"PTOAS A5 nightly board validation: {status}",
                },
            },
            "elements": elements,
        },
    }


def append_file(path_text: str, content: str) -> None:
    if not path_text:
        return
    with Path(path_text).open("a", encoding="utf-8") as stream:
        stream.write(content)


def send_feishu(webhook_url: str, payload: dict[str, object]) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        webhook_url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        response.read()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize an A5 board result TSV and optionally notify Feishu."
    )
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--conclusion", default="failure")
    parser.add_argument("--run-url", default="")
    parser.add_argument("--sha", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = load_results(args.results)
    markdown = render_markdown(
        summary,
        conclusion=args.conclusion,
        run_url=args.run_url,
        sha=args.sha,
    )
    print(markdown, end="")
    append_file(os.environ.get("GITHUB_STEP_SUMMARY", ""), markdown)
    append_file(
        os.environ.get("GITHUB_OUTPUT", ""),
        "\n".join(
            [
                f"ok={summary.counts.get('OK', 0)}",
                f"fail={summary.counts.get('FAIL', 0)}",
                f"skip={summary.counts.get('SKIP', 0)}",
                f"total={summary.total}",
            ]
        )
        + "\n",
    )

    webhook_url = os.environ.get("A5_FEISHU_WEBHOOK_URL", "").strip()
    if webhook_url:
        payload = build_feishu_payload(
            summary,
            conclusion=args.conclusion,
            run_url=args.run_url,
            sha=args.sha,
        )
        try:
            send_feishu(webhook_url, payload)
        except (OSError, ValueError, urllib.error.URLError) as exc:
            print(
                "WARNING: failed to send Feishu notification "
                f"({type(exc).__name__})",
                file=sys.stderr,
            )
    return 0 if args.conclusion == "success" and summary.passed else 1


if __name__ == "__main__":
    sys.exit(main())
