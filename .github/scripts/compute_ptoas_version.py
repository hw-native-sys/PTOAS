#!/usr/bin/env python3
# coding=utf-8
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
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute the ptoas CLI version from the top-level CMakeLists.txt."
    )
    parser.add_argument(
        "--cmake-file",
        default="CMakeLists.txt",
        help="Path to the top-level CMakeLists.txt file.",
    )
    parser.add_argument(
        "--mode",
        choices=("dev", "release"),
        default="dev",
        help="Both dev and release modes use the base version from CMakeLists.txt; release mode exists to validate release tags against that version.",
    )
    parser.add_argument(
        "--check-tag",
        help="Optional release tag to validate, e.g. v0.8 or ptoas-v0.8.",
    )
    return parser.parse_args()


def read_base_version(cmake_file: pathlib.Path) -> str:
    content = cmake_file.read_text(encoding="utf-8")
    match = re.search(r"project\s*\(\s*ptoas\s+VERSION\s+([0-9]+\.[0-9]+)\s*\)", content)
    if not match:
        raise ValueError(
            f"could not find 'project(ptoas VERSION x.y)' in {cmake_file}"
        )
    return match.group(1)

def normalize_tag(tag: str) -> str:
    normalized = tag.strip()
    if normalized.startswith("ptoas-"):
        normalized = normalized[len("ptoas-"):]
    if normalized.startswith("v"):
        normalized = normalized[1:]
    return normalized


def main() -> int:
    args = parse_args()
    cmake_file = pathlib.Path(args.cmake_file)
    base_version = read_base_version(cmake_file)
    version = base_version

    if args.check_tag is not None:
        normalized_tag = normalize_tag(args.check_tag.strip())
        if normalized_tag != version:
            print(
                f"release tag '{args.check_tag}' does not match computed version '{version}'",
                file=sys.stderr,
            )
            return 1

    print(version)
    return 0


if __name__ == "__main__":
    sys.exit(main())
