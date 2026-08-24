#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Prepare the complete source tree used to build the ``ptoas-vmi`` wheel."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
REPOSITORY_ROOT = SCRIPT_PATH.parents[2]
METADATA_PATCH = SCRIPT_PATH.with_name("pyproject.toml.patch")
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / ".work" / "ptoas-vmi-source"
VERSION_PATTERN = re.compile(r'^\+version = "([0-9]+\.[0-9]+\.[0-9]+)"$', re.M)
# Agent instructions and compatibility discovery links are development-only
# configuration, not part of the VMI source tree. Source packaging continues
# to reject symbolic links everywhere else in the repository.
ARCHIVE_EXCLUDES = (
    ":(exclude).agents",
    ":(exclude).claude",
    ":(exclude).codex",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Staging directory to replace (default: .work/ptoas-vmi-source).",
    )
    parser.add_argument(
        "--revision",
        default="HEAD",
        help="Git revision to archive (default: HEAD).",
    )
    parser.add_argument(
        "--print-version",
        action="store_true",
        help="Print the configured VMI version without preparing a source tree.",
    )
    return parser.parse_args()


def read_vmi_version() -> str:
    versions = VERSION_PATTERN.findall(METADATA_PATCH.read_text(encoding="utf-8"))
    if len(versions) != 1:
        raise ValueError("VMI metadata patch does not set one static X.Y.Z version")
    return versions[0]


def _check_output_dir(output_dir: Path) -> None:
    try:
        relative = output_dir.relative_to(REPOSITORY_ROOT)
    except ValueError:
        if output_dir.exists() and (
            not output_dir.is_dir() or any(output_dir.iterdir())
        ):
            raise ValueError(
                f"external output directory must be absent or empty: {output_dir}"
            )
        return
    if len(relative.parts) < 2 or relative.parts[0] != ".work":
        raise ValueError(
            f"repository-local output directory must be below .work: {output_dir}"
        )


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def prepare_source(output_dir: Path, revision: str) -> dict[str, str]:
    output_dir = output_dir.expanduser().resolve()
    _check_output_dir(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent)
    )
    archive_path: Path | None = None
    try:
        try:
            resolved_revision = subprocess.run(
                [
                    "git",
                    "-C",
                    str(REPOSITORY_ROOT),
                    "rev-parse",
                    "--verify",
                    f"{revision}^{{commit}}",
                ],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except subprocess.CalledProcessError:
            # Shallow clone fallback: HEAD^{commit} may fail when the
            # commit object is not fully resolved (e.g. fetch-depth: 1).
            resolved_revision = subprocess.run(
                [
                    "git",
                    "-C",
                    str(REPOSITORY_ROOT),
                    "rev-parse",
                    "--verify",
                    revision,
                ],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        with tempfile.NamedTemporaryFile(
            prefix="ptoas-vmi-source-",
            suffix=".tar",
            dir=output_dir.parent,
            delete=False,
        ) as archive_file:
            archive_path = Path(archive_file.name)

        subprocess.run(
            [
                "git",
                "-C",
                str(REPOSITORY_ROOT),
                "archive",
                "--format=tar",
                f"--output={archive_path}",
                resolved_revision,
                "--",
                *ARCHIVE_EXCLUDES,
            ],
            check=True,
        )
        with tarfile.open(archive_path, mode="r:") as archive:
            for member in archive.getmembers():
                member_path = Path(member.name)
                if member_path.is_absolute() or ".." in member_path.parts:
                    raise ValueError(
                        f"git archive contains an unsafe path: {member.name}"
                    )
                if member.issym() or member.islnk():
                    raise ValueError(
                        f"git archive contains an unsupported link: {member.name}"
                    )
            archive.extractall(temporary_dir)

        apply_environment = os.environ.copy()
        apply_environment.update(
            {
                "GIT_DIR": str(temporary_dir / ".git-not-present"),
                "GIT_WORK_TREE": str(temporary_dir),
            }
        )
        subprocess.run(
            ["git", "apply", str(METADATA_PATCH)],
            cwd=temporary_dir,
            env=apply_environment,
            check=True,
        )

        version = read_vmi_version()
        staged_text = (temporary_dir / "pyproject.toml").read_text(encoding="utf-8")
        required_fragments = (
            'name = "ptoas-vmi"',
            f'version = "{version}"',
            'PTOAS_CLI_VERSION_LABEL = "vmi"',
            'sdist.inclusion-mode = "manual"',
        )
        missing = [item for item in required_fragments if item not in staged_text]
        if missing:
            raise ValueError(f"VMI metadata patch is incomplete: {missing}")
        if "../" in staged_text:
            raise ValueError("staged pyproject.toml contains an external relative path")

        _remove_path(output_dir)
        temporary_dir.replace(output_dir)
    except BaseException:
        _remove_path(temporary_dir)
        raise
    finally:
        if archive_path is not None:
            archive_path.unlink(missing_ok=True)

    return {"version": version, "source": str(output_dir)}


def main() -> int:
    args = parse_args()
    try:
        if args.print_version:
            print(read_vmi_version())
            return 0
        outputs = prepare_source(args.output_dir, args.revision)
        print(outputs["source"])
    except (OSError, RuntimeError, subprocess.CalledProcessError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
