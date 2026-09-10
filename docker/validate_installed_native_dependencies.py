#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Validate the installed PTOAS native dependency ownership boundary."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


COMPILER_LIBRARY_PREFIX = "libPTOASCompiler"
OBSOLETE_LIBRARY_PREFIX = "libPTOASPythonCAPI"
EXTERNAL_IMPLEMENTATION_PATTERN = re.compile(
    r"^lib(?:LLVM|MLIR|PTOIR|PTOTransforms)"
)


def _dependencies(binary: Path) -> set[str]:
    if sys.platform == "darwin":
        output = subprocess.run(
            ["otool", "-L", str(binary)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        lines = output.splitlines()[1:]
        return {Path(line.strip().split(" ", 1)[0]).name for line in lines}

    output = subprocess.run(
        ["readelf", "-d", str(binary)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return set(re.findall(r"Shared library: \[([^]]+)\]", output))


def _native_extensions(package_root: Path) -> list[Path]:
    mlir_native_root = package_root / "mlir" / "_mlir_libs"
    candidates = list(package_root.glob("_core*.so"))
    candidates.extend(package_root.glob("_core*.dylib"))
    candidates.extend(mlir_native_root.glob("_*.so"))
    candidates.extend(mlir_native_root.glob("_*.dylib"))
    return sorted(path for path in candidates if path.is_file())


def validate(package_root: Path, expect_static_llvm: bool) -> None:
    mlir_native_root = package_root / "mlir" / "_mlir_libs"
    compiler_libraries = sorted(
        path
        for path in mlir_native_root.glob(f"{COMPILER_LIBRARY_PREFIX}*")
        if path.is_file()
    )
    if len(compiler_libraries) != 1:
        raise SystemExit(
            f"expected exactly one {COMPILER_LIBRARY_PREFIX} under "
            f"{mlir_native_root}, found {compiler_libraries}"
        )
    if any(mlir_native_root.glob(f"{OBSOLETE_LIBRARY_PREFIX}*")):
        raise SystemExit(
            f"obsolete {OBSOLETE_LIBRARY_PREFIX} remains under {mlir_native_root}"
        )

    extensions = _native_extensions(package_root)
    if not extensions:
        raise SystemExit(f"no PTOAS native extensions found under {package_root}")

    failures: list[str] = []
    for extension in extensions:
        dependencies = _dependencies(extension)
        if not any(dep.startswith(COMPILER_LIBRARY_PREFIX) for dep in dependencies):
            failures.append(f"{extension}: does not resolve {COMPILER_LIBRARY_PREFIX}")
        if expect_static_llvm:
            external = sorted(
                dep
                for dep in dependencies
                if EXTERNAL_IMPLEMENTATION_PATTERN.match(dep)
            )
            if external:
                failures.append(
                    f"{extension}: directly depends on static-profile external "
                    f"implementation libraries {external}"
                )

    if expect_static_llvm:
        compiler_dependencies = _dependencies(compiler_libraries[0])
        external = sorted(
            dep
            for dep in compiler_dependencies
            if EXTERNAL_IMPLEMENTATION_PATTERN.match(dep)
        )
        if external:
            failures.append(
                f"{compiler_libraries[0]}: static profile still depends on {external}"
            )

    if failures:
        raise SystemExit("native dependency validation failed:\n  " + "\n  ".join(failures))

    profile = "static LLVM" if expect_static_llvm else "shared LLVM"
    print(
        f"validated {len(extensions)} native extensions and "
        f"{COMPILER_LIBRARY_PREFIX} ownership ({profile} profile)"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate an installed PTOAS native dependency graph."
    )
    parser.add_argument("package_root", type=Path)
    parser.add_argument("--expect-static-llvm", action="store_true")
    args = parser.parse_args()
    validate(args.package_root.resolve(), args.expect_static_llvm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
