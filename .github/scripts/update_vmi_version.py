#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Update the next VMI package version after a VMI release."""

from __future__ import annotations

import argparse
import pathlib
import re
import sys


VERSION_RE = re.compile(r'^(\+version = ")([0-9]+\.[0-9]+\.[0-9]+)(")$', re.M)
TAG_RE = re.compile(r"^vmi-v([0-9]+)\.([0-9]+)\.([0-9]+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patch-file", default="packaging/ptoas-vmi/pyproject.toml.patch")
    parser.add_argument("--version", required=True, help="Released VMI tag, for example vmi-v0.1.5.")
    parser.add_argument("--next", action="store_true", help="Advance the released patch version by one.")
    return parser.parse_args()


def normalize_tag(tag: str) -> str:
    match = TAG_RE.fullmatch(tag.strip())
    if not match:
        raise ValueError(f"invalid VMI release tag '{tag}'")
    return ".".join(match.groups())


def read_version(patch_file: pathlib.Path) -> tuple[str, re.Match[str]]:
    content = patch_file.read_text(encoding="utf-8")
    matches = list(VERSION_RE.finditer(content))
    if len(matches) != 1:
        raise ValueError(f"VMI metadata patch must contain exactly one +version line: {patch_file}")
    return matches[0].group(2), matches[0]


def bump_version(version: str) -> str:
    major, minor, patch = (int(part) for part in version.split("."))
    return f"{major}.{minor}.{patch + 1}"


def update_version(patch_file: pathlib.Path, released_version: str, advance: bool) -> str:
    content = patch_file.read_text(encoding="utf-8")
    current_version, match = read_version(patch_file)
    if current_version != released_version:
        raise ValueError(f"metadata version {current_version} does not match released version {released_version}")
    next_version = bump_version(released_version) if advance else released_version
    updated = content[: match.start(2)] + next_version + content[match.end(2) :]
    if updated != content:
        patch_file.write_text(updated, encoding="utf-8")
    return next_version


def main() -> int:
    args = parse_args()
    try:
        released_version = normalize_tag(args.version)
        next_version = update_version(pathlib.Path(args.patch_file), released_version, args.next)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(next_version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
