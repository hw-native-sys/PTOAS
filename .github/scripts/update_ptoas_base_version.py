#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import argparse
import pathlib
import re
import subprocess
import sys


PROJECT_VERSION_RE = re.compile(
    r"(project\s*\(\s*ptoas\s+VERSION\s+)([0-9]+\.[0-9]+)(\s*\))"
)
TAG_VERSION_RE = re.compile(r"v?([0-9]+)\.([0-9]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Update the base PTOAS version in the top-level CMakeLists.txt."
    )
    parser.add_argument(
        "--cmake-file",
        default="CMakeLists.txt",
        help="Path to the top-level CMakeLists.txt file.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--version",
        help="Released version to write back, e.g. v0.8 or 0.8.",
    )
    group.add_argument(
        "--from-git-tags",
        action="store_true",
        help="Infer the base version from the latest valid git tag (vX.Y).",
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root used when scanning git tags.",
    )
    return parser.parse_args()


def normalize_version(version: str) -> str:
    normalized = version.strip()
    if normalized.startswith("v"):
        normalized = normalized[1:]
    if not re.fullmatch(r"[0-9]+\.[0-9]+", normalized):
        raise ValueError(f"invalid PTOAS version '{version}'")
    return normalized


def read_base_version(cmake_file: pathlib.Path) -> str:
    content = cmake_file.read_text(encoding="utf-8")
    match = PROJECT_VERSION_RE.search(content)
    if not match:
        raise ValueError(
            f"could not find 'project(ptoas VERSION x.y)' in {cmake_file}"
        )
    return match.group(2)


def latest_git_tag_version(repo_root: pathlib.Path) -> str | None:
    try:
        output = subprocess.check_output(
            ["git", "-C", str(repo_root), "tag", "--list"],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except subprocess.CalledProcessError as exc:
        raise ValueError(
            f"failed to list git tags under {repo_root}: {exc.output.strip()}"
        ) from exc

    latest: tuple[int, int] | None = None
    for raw_tag in output.splitlines():
        tag = raw_tag.strip()
        match = TAG_VERSION_RE.fullmatch(tag)
        if not match:
            continue
        candidate = (int(match.group(1)), int(match.group(2)))
        if latest is None or candidate > latest:
            latest = candidate

    if latest is None:
        return None
    return f"{latest[0]}.{latest[1]}"


def update_base_version(cmake_file: pathlib.Path, version: str) -> bool:
    content = cmake_file.read_text(encoding="utf-8")

    def repl(match: re.Match[str]) -> str:
        return f"{match.group(1)}{version}{match.group(3)}"

    updated, count = PROJECT_VERSION_RE.subn(repl, content, count=1)
    if count != 1:
        raise ValueError(
            f"could not find 'project(ptoas VERSION x.y)' in {cmake_file}"
        )
    if updated == content:
        return False
    cmake_file.write_text(updated, encoding="utf-8")
    return True


def main() -> int:
    args = parse_args()
    cmake_file = pathlib.Path(args.cmake_file)
    if args.version is not None:
        version = normalize_version(args.version)
    else:
        repo_root = pathlib.Path(args.repo_root)
        version = latest_git_tag_version(repo_root)
        if version is None:
            version = read_base_version(cmake_file)
            print(
                f"warning: no valid git tags found under {repo_root}; "
                f"keeping current base version {version}",
                file=sys.stderr,
            )

    update_base_version(cmake_file, version)
    print(version)
    return 0


if __name__ == "__main__":
    sys.exit(main())
