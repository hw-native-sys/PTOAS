#!/usr/bin/env python3

import argparse
import pathlib
import re
import sys


PROJECT_VERSION_RE = re.compile(
    r"project\s*\(\s*ptoas\s+VERSION\s+([0-9]+\.[0-9]+)\s*\)"
)
VERSION_RE = re.compile(r"[0-9]+\.[0-9]+")


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
        help="release mode increments the minor component by 1 (for example, 0.7 -> 0.8 and 0.10 -> 0.11).",
    )
    parser.add_argument(
        "--check-tag",
        help="Optional release tag to validate, e.g. v0.8 or 0.8.",
    )
    return parser.parse_args()


def read_base_version(cmake_file: pathlib.Path) -> str:
    content = cmake_file.read_text(encoding="utf-8")
    match = PROJECT_VERSION_RE.search(content)
    if not match:
        raise ValueError(
            f"could not find 'project(ptoas VERSION x.y)' in {cmake_file}"
        )
    return match.group(1)


def bump_version(base_version: str) -> str:
    major_str, minor_str = base_version.split(".")
    major = int(major_str)
    minor = int(minor_str) + 1
    return f"{major}.{minor}"


def normalize_tag(tag: str) -> str:
    normalized = tag[1:] if tag.startswith("v") else tag
    if not VERSION_RE.fullmatch(normalized):
        raise ValueError(f"invalid PTOAS release tag '{tag}'")
    return normalized


def compute_version(base_version: str, mode: str, check_tag: str | None = None) -> str:
    if mode == "dev":
        version = base_version
    else:
        next_release_version = bump_version(base_version)
        version = next_release_version

        if check_tag is not None:
            normalized_tag = normalize_tag(check_tag.strip())
            valid_versions = (next_release_version, base_version)
            if normalized_tag not in valid_versions:
                raise ValueError(
                    "release tag "
                    f"'{check_tag}' does not match next release version "
                    f"'{next_release_version}' or current base version "
                    f"'{base_version}'"
                )
            version = normalized_tag

    return version


def main() -> int:
    args = parse_args()
    cmake_file = pathlib.Path(args.cmake_file)
    base_version = read_base_version(cmake_file)

    try:
        version = compute_version(base_version, args.mode, args.check_tag)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(version)
    return 0


if __name__ == "__main__":
    sys.exit(main())
