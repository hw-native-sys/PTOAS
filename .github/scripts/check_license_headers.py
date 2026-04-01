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
import json
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

HEADER_BODY = [
    "Copyright (c) 2026 Huawei Technologies Co., Ltd.",
    "This program is free software, you can redistribute it and/or modify it under the terms and conditions of",
    'CANN Open Software License Agreement Version 2.0 (the "License").',
    "Please refer to the License for details. You may not use this file except in compliance with the License.",
    "THIS SOFTWARE IS PROVIDED ON AN \"AS IS\" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,",
    "INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.",
    "See LICENSE in the root of the software repository for the full text of the License.",
]
HASH_HEADER = [f"# {line}" for line in HEADER_BODY]
SLASH_HEADER = [f"// {line}" for line in HEADER_BODY]
HASH_FILE_SUFFIXES = {".py", ".sh", ".cmake"}
SLASH_FILE_SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx", ".td"}
HASH_FILE_BASENAMES = {"CMakeLists.txt"}
SHEBANG_SUFFIXES = {".py", ".sh"}
ZERO_SHA = "0" * 40


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check PR386-style license headers on changed files.")
    parser.add_argument("--repo", required=True, help="owner/repo for GitHub API lookups")
    parser.add_argument("--event-name", required=True, help="GitHub event name")
    parser.add_argument("--pr-number", default="", help="Pull request number for pull_request events")
    parser.add_argument("--base-sha", default="", help="Git base SHA for push events")
    parser.add_argument("--head-sha", default="HEAD", help="Git head SHA for push events")
    parser.add_argument("--github-token", default="", help="GitHub token used for PR file listing")
    return parser.parse_args()


def comment_style_for(path_str: str) -> str | None:
    path = Path(path_str)
    suffix = path.suffix.lower()
    if path.name in HASH_FILE_BASENAMES or suffix in HASH_FILE_SUFFIXES:
        return "#"
    if suffix in SLASH_FILE_SUFFIXES:
        return "//"
    return None


def expected_header(style: str) -> list[str]:
    return HASH_HEADER if style == "#" else SLASH_HEADER


def git_output(*args: str) -> list[str]:
    proc = subprocess.run(
        ["git", *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def changed_files_from_git(base_sha: str, head_sha: str) -> list[str]:
    if base_sha and base_sha != ZERO_SHA:
        try:
            return git_output("diff", "--name-only", "--diff-filter=ACMR", base_sha, head_sha)
        except subprocess.CalledProcessError:
            pass
    return git_output("diff-tree", "--no-commit-id", "--name-only", "--diff-filter=ACMR", "-r", head_sha)


def github_api_json(url: str, token: str) -> list[dict]:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def changed_files_from_pr(repo: str, pr_number: str, token: str) -> list[str]:
    files: list[str] = []
    page = 1
    while True:
        url = (
            f"https://api.github.com/repos/{urllib.parse.quote(repo, safe='/')}/pulls/"
            f"{pr_number}/files?per_page=100&page={page}"
        )
        page_items = github_api_json(url, token)
        if not page_items:
            break
        for item in page_items:
            if item.get("status") == "removed":
                continue
            filename = str(item.get("filename") or "").strip()
            if filename:
                files.append(filename)
        page += 1
    return files


def normalize_lines(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if lines:
        lines[0] = lines[0].lstrip("\ufeff")
    return lines


def header_start_index(path: Path, lines: list[str]) -> int:
    if lines and path.suffix.lower() in SHEBANG_SUFFIXES and lines[0].startswith("#!"):
        return 1
    return 0


def has_expected_header(path_str: str, style: str) -> bool:
    path = Path(path_str)
    if not path.exists():
        return True
    lines = normalize_lines(path)
    start = header_start_index(path, lines)
    expected = expected_header(style)
    return lines[start : start + len(expected)] == expected


def main() -> int:
    args = parse_args()
    try:
        if args.event_name == "pull_request" and args.pr_number:
            changed_files = changed_files_from_pr(args.repo, args.pr_number, args.github_token)
        else:
            changed_files = changed_files_from_git(args.base_sha, args.head_sha)
    except urllib.error.URLError as exc:
        print(f"Failed to query GitHub API: {exc}", file=sys.stderr)
        return 2
    except subprocess.CalledProcessError as exc:
        print(exc.stderr or str(exc), file=sys.stderr)
        return 2

    relevant_files: list[tuple[str, str]] = []
    for path_str in changed_files:
        style = comment_style_for(path_str)
        if style is None:
            continue
        relevant_files.append((path_str, style))

    if not relevant_files:
        print("No changed source/script files require the PR386 license header.")
        return 0

    missing: list[tuple[str, str]] = []
    for path_str, style in relevant_files:
        if not has_expected_header(path_str, style):
            missing.append((path_str, style))

    if missing:
        print("Missing PR386 license header in changed files:", file=sys.stderr)
        for path_str, style in missing:
            print(f"- {path_str}", file=sys.stderr)
            for line in expected_header(style):
                print(f"    {line}", file=sys.stderr)
        return 1

    print(f"Checked {len(relevant_files)} changed source/script files: all headers present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
