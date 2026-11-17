#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import hmac
import json
import os
import re
import sys
import time
import urllib.request
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path


@dataclass(frozen=True)
class BoardSummary:
    results_found: bool
    counts: Counter[str]
    failed_cases: tuple[str, ...]
    failure_details: tuple[tuple[str, str], ...] = ()
    log_errors: tuple[str, ...] = ()
    # Board results that were collected but did not validate anything. They are
    # reported explicitly: a silent filter looks like a smaller suite instead of
    # a known gap, which is exactly what a CIE role must not do.
    skips: tuple[tuple[str, str], ...] = ()
    not_generated: tuple[tuple[str, str], ...] = ()

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


def _case_key_from_marker(path_text: str) -> str:
    """Convert a run log CASE path to the TSV's ``Sample/testcase`` key."""
    case_path = Path(path_text.strip())
    filename = case_path.name
    if filename.endswith("-pto.cpp"):
        filename = filename[: -len("-pto.cpp")]
    return f"{case_path.parent.name}/{filename}"


_ERROR_LINE = re.compile(
    r"(?:\[ERROR\]|\bERROR:\s*testcase failed|\b(?:fatal )?error:|"
    r"static assertion failed|Segmentation fault|core dumped|CMake Error|"
    r"aclrtSetDevice|halDeviceOpen|TsdOpen|open device .* failed|"
    r"Mismatch:|Packed mask mismatch:|compare failed|gmake: \*\*)",
    re.IGNORECASE,
)
_ERROR_PRIORITIES = (
    re.compile(r"Mismatch:|Packed mask mismatch", re.IGNORECASE),
    re.compile(r"Segmentation fault|core dumped", re.IGNORECASE),
    re.compile(r"aclrtSetDevice|halDeviceOpen|TsdOpen|open device .* failed", re.IGNORECASE),
    re.compile(r"no member named|undefined reference|CMake Error", re.IGNORECASE),
    re.compile(r"static assertion failed|compare failed", re.IGNORECASE),
    re.compile(r"ERROR:\s*testcase failed|gmake: \*\*", re.IGNORECASE),
)


def _select_error_lines(lines: list[str]) -> list[str]:
    candidates: list[str] = []
    for line in lines:
        text = line.strip()
        if text and _ERROR_LINE.search(text) and text not in candidates:
            candidates.append(text)
    selected: list[str] = []
    for priority in _ERROR_PRIORITIES:
        selected.extend(line for line in candidates if priority.search(line) and line not in selected[:])
        if len(selected) >= 8:
            break
    if len(selected) < 8:
        selected.extend(line for line in candidates if line not in selected)
    return selected[:10]


def load_log_failure_details(path: Path | None) -> dict[str, str]:
    """Extract concise, actionable error lines from each CASE section."""
    if path is None or not path.is_file():
        return {}

    sections: dict[str, list[str]] = {}
    current: list[str] | None = None
    current_key = ""
    case_marker = re.compile(r"=== CASE:\s*(.*?)\s*===")
    with path.open(encoding="utf-8", errors="replace") as stream:
        for raw_line in stream:
            marker = case_marker.search(raw_line)
            if marker:
                current_key = _case_key_from_marker(marker.group(1))
                current = sections.setdefault(current_key, [])
                continue
            if current is not None:
                current.append(raw_line.rstrip())

    details: dict[str, str] = {}
    for key, lines in sections.items():
        selected = _select_error_lines(lines)
        if selected:
            # Keep one testcase's excerpt compact enough for a Feishu card.
            details[key] = "\n".join(selected)[:1400]
    return details


def load_log_errors(path: Path | None, *, limit: int = 12) -> tuple[str, ...]:
    """Extract global errors for failures that happen before a CASE section."""
    if path is None or not path.is_file():
        return ()
    errors: list[str] = []
    with path.open(encoding="utf-8", errors="replace") as stream:
        for raw_line in stream:
            text = raw_line.strip()
            if text and _ERROR_LINE.search(text) and text not in errors:
                errors.append(text)
            if len(errors) >= limit:
                break
    return tuple(errors)


def load_results(path: Path, log_path: Path | None = None) -> BoardSummary:
    if not path.is_file():
        return BoardSummary(False, Counter(), (), (), load_log_errors(log_path))

    counts: Counter[str] = Counter()
    failed_cases: list[str] = []
    failed_info: dict[str, str] = {}
    skips: list[tuple[str, str]] = []
    with path.open(encoding="utf-8", errors="replace", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        for row in reader:
            status = (row.get("status") or "UNKNOWN").strip() or "UNKNOWN"
            testcase = (row.get("testcase") or "<unknown>").strip() or "<unknown>"
            counts[status] += 1
            if status == "FAIL":
                failed_cases.append(testcase)
                failed_info[testcase] = (row.get("info") or "exit status unavailable").strip()
            elif status == "SKIP":
                skips.append((testcase, (row.get("info") or "").strip()))
    log_details = load_log_failure_details(log_path)
    failure_details = tuple(
        (
            testcase,
            log_details.get(
                testcase,
                "No recognized error line found in board-validation.log; "
                f"result info: {failed_info.get(testcase, 'exit status unavailable')}",
            ),
        )
        for testcase in failed_cases
    )
    return BoardSummary(
        True, counts, tuple(failed_cases), failure_details, (), tuple(skips)
    )


def load_not_generated(path: Path | None) -> tuple[tuple[str, str], ...]:
    """Read the driver's deliberate generation exclusions (``case<TAB>reason``)."""
    if path is None or not path.is_file():
        return ()
    rows: list[tuple[str, str]] = []
    with path.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            case, _, reason = line.partition("\t")
            if case.strip():
                rows.append((case.strip(), reason.strip() or "excluded before generation"))
    return tuple(rows)


def skip_groups(
    summary: BoardSummary, *, limit: int | None = None
) -> list[tuple[str, int]]:
    counts = Counter(reason or "no reason recorded" for _, reason in summary.skips)
    groups = counts.most_common(limit)
    if limit is not None and len(counts) > limit:
        rest = sum(count for _, count in counts.most_common()[limit:])
        groups.append((f"other reasons ({len(counts) - limit} kinds)", rest))
    return groups


def render_not_validated(summary: BoardSummary) -> list[str]:
    """Markdown block listing everything that was not actually validated."""
    if not summary.skips and not summary.not_generated:
        return []
    lines = ["", "### Not validated", ""]
    if summary.skips:
        lines.append(f"SKIP ({len(summary.skips)}), grouped by the reason the runner recorded:")
        lines.append("")
        lines.extend(f"- `{count}` {reason}" for reason, count in skip_groups(summary, limit=20))
        lines.append("")
    if summary.not_generated:
        lines.append(f"Not generated ({len(summary.not_generated)}), so absent from the manifest:")
        lines.append("")
        lines.extend(f"- `{case}` — {reason}" for case, reason in summary.not_generated)
        lines.append("")
    return lines


def render_failure_details(summary: BoardSummary, *, max_chars: int | None = None) -> str:
    """Render per-case excerpts, optionally bounded for webhook card limits."""
    if not summary.failure_details:
        return ""
    blocks: list[str] = []
    used = 0
    for testcase, detail in summary.failure_details:
        block = f"- `{testcase}`\n```text\n{detail}\n```"
        if max_chars is not None and blocks and used + len(block) + 1 > max_chars:
            remaining = len(summary.failure_details) - len(blocks)
            blocks.append(f"- ... and {remaining} more; open the GitHub run for the full log")
            break
        blocks.append(block)
        used += len(block) + 1
    return "\n\n".join(blocks)


def render_failure_details_inline(
    summary: BoardSummary, *, max_chars: int | None = None
) -> str:
    """Card variant: one inline-code line per error.

    The bot card client renders fenced blocks literally, so a card built with
    ``` fences shows the fence markers text; inline code is rendered reliably.
    """
    if not summary.failure_details:
        return ""
    blocks: list[str] = []
    used = 0
    for testcase, detail in summary.failure_details:
        lines = [f"**{testcase}**"]
        lines.extend(f"`{line}`" for line in detail.splitlines() if line.strip())
        block = "\n".join(lines)
        if max_chars is not None and blocks and used + len(block) + 1 > max_chars:
            remaining = len(summary.failure_details) - len(blocks)
            blocks.append(f"... and {remaining} more; open the run log for the full output")
            break
        blocks.append(block)
        used += len(block) + 1
    return "\n\n".join(blocks)


def render_markdown(
    summary: BoardSummary,
    *,
    conclusion: str,
    run_url: str,
    sha: str,
    run_id: str = "",
) -> str:
    succeeded = conclusion == "success" and summary.passed
    lines = [
        "## A5 nightly board validation",
        "",
        f"- Status: **{'PASS' if succeeded else 'FAIL'}**",
        f"- Commit: `{sha[:12] or 'unknown'}`",
        *([f"- Run ID: `{run_id}`"] if run_id else []),
        f"- Results: OK `{summary.counts.get('OK', 0)}` / "
        f"FAIL `{summary.counts.get('FAIL', 0)}` / "
        f"SKIP `{summary.counts.get('SKIP', 0)}` / TOTAL `{summary.total}`",
    ]
    if summary.not_generated:
        lines.append(
            f"- A5 case inventory: `{summary.total}` generated"
            f" + `{len(summary.not_generated)}` not generated"
            f" = `{summary.total + len(summary.not_generated)}`"
        )
    if run_url:
        lines.append(f"- Run: {run_url}")
    if not summary.results_found:
        lines.extend(["", "No board result TSV was produced. Inspect the workflow log."])
        if summary.log_errors:
            lines.extend(["", "### Concrete errors", ""])
            lines.extend(f"```text\n{error}\n```" for error in summary.log_errors)
    elif summary.failed_cases:
        lines.extend(["", "### Failed cases", ""])
        lines.extend(f"- `{case}`" for case in summary.failed_cases[:50])
        if len(summary.failed_cases) > 50:
            lines.append(f"- ... and {len(summary.failed_cases) - 50} more")
        details = render_failure_details(summary)
        if details:
            lines.extend(["", "### Concrete errors", "", details])
    if summary.results_found:
        lines.extend(render_not_validated(summary))
    return "\n".join(lines) + "\n"


def _feishu_summary_lines(
    summary: BoardSummary,
    *,
    conclusion: str,
    sha: str,
    run_id: str,
) -> tuple[list[str], bool]:
    succeeded = conclusion == "success" and summary.passed
    status = "PASS" if succeeded else "FAIL"
    lines = [
        f"**Status**: {status}",
        f"**Commit**: `{sha[:12] or 'unknown'}`",
        *([f"**Run**: `{run_id}`"] if run_id else []),
        f"**Results**: OK `{summary.counts.get('OK', 0)}` / "
        f"FAIL `{summary.counts.get('FAIL', 0)}` / "
        f"SKIP `{summary.counts.get('SKIP', 0)}` / TOTAL `{summary.total}`",
    ]
    if not summary.results_found:
        lines.append("**Error**: result TSV was not produced")
        if summary.log_errors:
            lines.append(
                "**Concrete errors**:\n```text\n"
                + "\n".join(summary.log_errors)
                + "\n```"
            )
    return lines, succeeded


def _append_feishu_validation_gaps(
    lines: list[str], summary: BoardSummary
) -> None:
    if summary.failed_cases:
        failed = ", ".join(summary.failed_cases[:20])
        if len(summary.failed_cases) > 20:
            failed += f", ... (+{len(summary.failed_cases) - 20})"
        lines.append(f"**Failed cases**: {failed}")
        details = render_failure_details_inline(summary, max_chars=7600)
        if details:
            lines.append(f"**Concrete errors**:\n{details}")
    # Everything the run did not validate must stay visible: hiding skipped
    # or ungenerated cases turns a gap into a silently smaller suite.
    if summary.not_generated:
        lines.append(
            f"**A5 用例口径**: 生成 `{summary.total}` + 未生成 "
            f"`{len(summary.not_generated)}` = "
            f"`{summary.total + len(summary.not_generated)}`"
        )
    if summary.skips:
        groups = " · ".join(
            f"{reason} `{count}`" for reason, count in skip_groups(summary, limit=5)
        )
        lines.append(f"**未验证(SKIP {len(summary.skips)})**: {groups}")
    for case, reason in summary.not_generated[:5]:
        lines.append(f"**未生成**: `{case}` — {reason}")
    if len(summary.not_generated) > 5:
        lines.append(f"**未生成**: ... 另有 {len(summary.not_generated) - 5} 例")


def _action_element(run_url: str, *, label: str) -> dict[str, object] | None:
    if not run_url:
        return None
    return {
        "tag": "action",
        "actions": [
            {
                "tag": "button",
                "text": {"tag": "plain_text", "content": label},
                "url": run_url,
                "type": "primary",
            }
        ],
    }


def build_feishu_payload(
    summary: BoardSummary,
    *,
    conclusion: str,
    run_url: str,
    sha: str,
    run_id: str = "",
) -> dict[str, object]:
    detail_lines, succeeded = _feishu_summary_lines(
        summary, conclusion=conclusion, sha=sha, run_id=run_id
    )
    if summary.results_found:
        _append_feishu_validation_gaps(detail_lines, summary)
    elements: list[dict[str, object]] = [
        {
            "tag": "div",
            "text": {"tag": "lark_md", "content": "\n".join(detail_lines)},
        }
    ]
    action = _action_element(run_url, label="Open GitHub run")
    if action is not None:
        elements.append(action)
    status = "PASS" if succeeded else "FAIL"
    return {
        "msg_type": "interactive",
        "card": {
            "header": {
                "template": "green" if succeeded else "red",
                "title": {
                    "tag": "plain_text",
                    "content": f"PTOAS A5 夜间看护：{status}",
                },
            },
            "elements": elements,
        },
    }


def _group_failure_details(
    details: tuple[tuple[str, str], ...],
) -> list[list[tuple[str, str]]]:
    # Keep each card comfortably below Feishu's interactive-card text limit.
    groups: list[list[tuple[str, str]]] = []
    current: list[tuple[str, str]] = []
    current_chars = 0
    for testcase, detail in details:
        block_chars = len(testcase) + len(detail) + 40
        if current and current_chars + block_chars > 7000:
            groups.append(current)
            current = []
            current_chars = 0
        current.append((testcase, detail))
        current_chars += block_chars
    if current:
        groups.append(current)
    return groups


def _render_detail_blocks(group: list[tuple[str, str]]) -> str:
    return "\n\n".join(
        "**{case}**\n{lines}".format(
            case=testcase,
            lines="\n".join(
                f"`{line}`" for line in detail.splitlines() if line.strip()
            ),
        )
        for testcase, detail in group
    )


def _detail_card(
    *,
    index: int,
    total: int,
    blocks: str,
    run_url: str,
) -> dict[str, object]:
    elements: list[dict[str, object]] = [
        {"tag": "div", "text": {"tag": "lark_md", "content": blocks}}
    ]
    action = _action_element(run_url, label="打开 GitHub 运行")
    if action is not None:
        elements.append(action)
    return {
        "msg_type": "interactive",
        "card": {
            "config": {"wide_screen_mode": True},
            "header": {
                "template": "red",
                "title": {
                    "tag": "plain_text",
                    "content": f"PTOAS A5 夜间看护失败详情 {index}/{total}",
                },
            },
            "elements": elements,
        },
    }


def build_feishu_detail_payloads(
    summary: BoardSummary,
    *,
    run_url: str,
) -> list[dict[str, object]]:
    """Build bounded cards so every failed case gets its own concrete excerpt."""
    if not summary.failure_details:
        return []
    groups = _group_failure_details(summary.failure_details)
    total = len(groups)
    return [
        _detail_card(
            index=index,
            total=total,
            blocks=_render_detail_blocks(group),
            run_url=run_url,
        )
        for index, group in enumerate(groups, start=1)
    ]


def append_file(path_text: str, content: str) -> None:
    if not path_text:
        return
    with Path(path_text).open("a", encoding="utf-8") as stream:
        stream.write(content)


def sign_feishu_payload(payload: dict[str, object]) -> dict[str, object]:
    """Add Feishu bot timestamp/sign when A5_FEISHU_SECRET is configured.

    Bots created with "签名校验" reject unsigned requests; the signature is
    base64(HMAC-SHA256(key="<timestamp>\\n<secret>", msg="")).
    """

    secret = os.environ.get("A5_FEISHU_SECRET", "").strip()
    if not secret:
        return payload
    timestamp = str(int(time.time()))
    string_to_sign = "{}\n{}".format(timestamp, secret).encode("utf-8")
    digest = hmac.new(string_to_sign, digestmod=hashlib.sha256).digest()
    signed = dict(payload)
    signed["timestamp"] = timestamp
    signed["sign"] = base64.b64encode(digest).decode("ascii")
    return signed


def send_feishu(webhook_url: str, payload: dict[str, object]) -> None:
    body = json.dumps(sign_feishu_payload(payload), ensure_ascii=False).encode("utf-8")
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
    parser.add_argument("--run-id", default="")
    parser.add_argument("--sha", default="")
    parser.add_argument(
        "--log",
        type=Path,
        default=None,
        help="Optional board-validation.log used to include concrete failure lines.",
    )
    parser.add_argument(
        "--not-generated",
        type=Path,
        default=None,
        help=(
            "Optional TSV of cases deliberately excluded before generation "
            "(case<TAB>reason); they are reported as not validated."
        ),
    )
    parser.add_argument(
        "--notify-status",
        type=Path,
        default=None,
        help=(
            "Optional file that receives ok/failed/skipped after the Feishu "
            "delivery attempt, so the caller can distinguish 'card sent' from "
            "'card rendered but delivery failed'."
        ),
    )
    return parser.parse_args()


def _load_summary(args: argparse.Namespace) -> BoardSummary:
    summary = load_results(args.results, args.log)
    not_generated = load_not_generated(args.not_generated)
    if not_generated:
        summary = replace(summary, not_generated=not_generated)
    return summary


def _emit_summary(summary: BoardSummary, args: argparse.Namespace) -> None:
    markdown = render_markdown(
        summary,
        conclusion=args.conclusion,
        run_url=args.run_url,
        run_id=args.run_id,
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


def _notify(summary: BoardSummary, args: argparse.Namespace) -> str:
    webhook_url = os.environ.get("A5_FEISHU_WEBHOOK_URL", "").strip()
    if not webhook_url:
        return "skipped"
    payload = build_feishu_payload(
        summary,
        conclusion=args.conclusion,
        run_url=args.run_url,
        run_id=args.run_id,
        sha=args.sha,
    )
    payloads = [payload] + build_feishu_detail_payloads(summary, run_url=args.run_url)
    notify_failures = 0
    for index, card_payload in enumerate(payloads, start=1):
        try:
            send_feishu(webhook_url, card_payload)
        except (OSError, ValueError) as exc:
            notify_failures += 1
            print(
                "WARNING: failed to send Feishu notification "
                f"part {index}/{len(payloads)} ({type(exc).__name__})",
                file=sys.stderr,
            )
    return "failed" if notify_failures else "ok"


def _write_notify_status(path: Path | None, status: str) -> None:
    if path is None:
        return
    try:
        path.write_text(status + "\n", encoding="utf-8")
    except OSError as exc:
        print(
            f"WARNING: cannot write notify status to {path} ({exc})",
            file=sys.stderr,
        )


def main() -> int:
    args = parse_args()
    summary = _load_summary(args)
    _emit_summary(summary, args)
    _write_notify_status(args.notify_status, _notify(summary, args))
    return 0 if args.conclusion == "success" and summary.passed else 1


if __name__ == "__main__":
    sys.exit(main())
