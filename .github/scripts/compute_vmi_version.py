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
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute the VMI release version from a version file."
    )
    parser.add_argument(
        "--version-file",
        default="docs/release/VMI_VERSION",
        help="Path to the VMI version file.",
    )
    parser.add_argument(
        "--check-tag",
        help="Optional release tag to validate, e.g. vmi-v0.1.0.",
    )
    return parser.parse_args()


def read_version(version_file: pathlib.Path) -> str:
    version = version_file.read_text(encoding="utf-8").strip()
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", version):
        raise ValueError(f"invalid VMI version '{version}' in {version_file}")
    return version


def normalize_tag(tag: str) -> str:
    normalized = tag.strip()
    if normalized.startswith("vmi-"):
        normalized = normalized[len("vmi-"):]
    if normalized.startswith("v"):
        normalized = normalized[1:]
    return normalized


def main() -> int:
    args = parse_args()
    version_file = pathlib.Path(args.version_file)
    version = read_version(version_file)

    if args.check_tag is not None:
        normalized_tag = normalize_tag(args.check_tag)
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
