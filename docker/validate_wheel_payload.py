#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Validate the packaged PTOAS wheel payload and launcher contract."""

from __future__ import annotations

import argparse
import importlib.machinery
import zipfile
from pathlib import Path


REQUIRED_FILES = {
    "ptoas/__init__.py",
    "ptoas/_cli.py",
    "ptoas/_runtime/share/ptoas/TileOps/__init__.py",
    "ptoas/_runtime/share/ptoas/SoftOps/__init__.py",
    "ptoas/mlir/ir.py",
    "ptoas/mlir/dialects/pto.py",
}
FORBIDDEN_FILES = {
    "ptoas/_runtime/bin/ptoas",
}
PTOAS_ENTRYPOINT_TARGET = "ptoas._cli:main"
WHEEL_GLOB = "ptoas*.whl"
NATIVE_MODULE_PATHS = {
    f"ptoas/_core{suffix}"
    for suffix in importlib.machinery.EXTENSION_SUFFIXES
}
MLIR_NATIVE_MODULE_PREFIX = "ptoas/mlir/_mlir_libs/"
PTOAS_COMPILER_LIBRARY_PREFIX = (
    f"{MLIR_NATIVE_MODULE_PREFIX}libPTOASCompiler"
)
OBSOLETE_COMMON_CAPI_PREFIX = (
    f"{MLIR_NATIVE_MODULE_PREFIX}libPTOASPythonCAPI"
)
MLIR_NATIVE_MODULE_PATHS = {
    f"{MLIR_NATIVE_MODULE_PREFIX}_mlir{suffix}"
    for suffix in importlib.machinery.EXTENSION_SUFFIXES
}
SITE_INITIALIZER_PATHS = {
    f"{MLIR_NATIVE_MODULE_PREFIX}_site_initialize_0{suffix}"
    for suffix in importlib.machinery.EXTENSION_SUFFIXES
}


def _resolve_wheel(candidate: str) -> Path:
    path = Path(candidate)
    if path.is_file():
        return path
    if path.is_dir():
        wheels = sorted(path.glob(WHEEL_GLOB))
        if len(wheels) != 1:
            raise SystemExit(
                f"expected exactly one wheel in {path}, found {len(wheels)}"
            )
        return wheels[0]
    matches = sorted(path.parent.glob(path.name))
    if len(matches) != 1:
        raise SystemExit(
            f"expected exactly one wheel matching {candidate!r}, found {len(matches)}"
        )
    return matches[0]


def _parse_console_scripts(entry_points: str) -> dict[str, str]:
    in_console_scripts = False
    scripts: dict[str, str] = {}
    for raw_line in entry_points.splitlines():
        line = raw_line.strip()
        if not line or line.startswith(("#", ";")):
            continue
        if line.startswith("[") and line.endswith("]"):
            in_console_scripts = line == "[console_scripts]"
            continue
        if not in_console_scripts:
            continue
        if "=" not in line:
            raise SystemExit(
                "wheel console_scripts entry point is malformed: "
                f"{raw_line!r}"
            )
        name, target = [part.strip() for part in line.split("=", 1)]
        if name in scripts:
            raise SystemExit(
                "wheel console_scripts contains duplicate entry point: "
                f"{name}"
            )
        scripts[name] = target
    return scripts


def validate_wheel_payload(wheel: Path) -> None:
    with zipfile.ZipFile(wheel) as zf:
        names = set(zf.namelist())
        missing = sorted(REQUIRED_FILES - names)
        if missing:
            raise SystemExit(f"wheel is missing required payload files: {missing}")

        native_modules = sorted(NATIVE_MODULE_PATHS & names)
        if len(native_modules) != 1:
            raise SystemExit(
                "wheel must contain exactly one importable ptoas._core extension, "
                f"found {native_modules}"
            )

        mlir_native_modules = sorted(MLIR_NATIVE_MODULE_PATHS & names)
        if len(mlir_native_modules) != 1:
            raise SystemExit(
                "wheel must contain exactly one ptoas-owned MLIR native module, "
                f"found {mlir_native_modules}"
            )

        compiler_libraries = sorted(
            name for name in names if name.startswith(PTOAS_COMPILER_LIBRARY_PREFIX)
        )
        if len(compiler_libraries) != 1:
            raise SystemExit(
                "wheel must contain exactly one PTOAS compiler library, "
                f"found {compiler_libraries}"
            )

        obsolete_common_capi = sorted(
            name for name in names if name.startswith(OBSOLETE_COMMON_CAPI_PREFIX)
        )
        if obsolete_common_capi:
            raise SystemExit(
                "wheel contains the obsolete PTOAS Python CAPI library: "
                f"{obsolete_common_capi}"
            )

        site_initializers = sorted(SITE_INITIALIZER_PATHS & names)
        if len(site_initializers) != 1:
            raise SystemExit(
                "wheel must contain exactly one PTOAS MLIR site initializer, "
                f"found {site_initializers}"
            )

        top_level_mlir = sorted(name for name in names if name.startswith("mlir/"))
        if top_level_mlir:
            raise SystemExit(
                "wheel must not install a top-level mlir package; "
                f"found {top_level_mlir[:10]}"
            )

        present_forbidden = sorted(FORBIDDEN_FILES & names)
        if present_forbidden:
            raise SystemExit(
                "wheel unexpectedly contains forbidden payload files: "
                f"{present_forbidden}"
            )

        entry_points_names = sorted(
            name
            for name in names
            if name.endswith(".dist-info/entry_points.txt")
        )
        if len(entry_points_names) != 1:
            raise SystemExit(
                "wheel must contain exactly one dist-info/entry_points.txt, "
                f"found {len(entry_points_names)}"
            )
        entry_points_name = entry_points_names[0]

        entry_points = zf.read(entry_points_name).decode("utf-8")
        console_scripts = _parse_console_scripts(entry_points)
        if console_scripts.get("ptoas") != PTOAS_ENTRYPOINT_TARGET:
            raise SystemExit(
                "wheel entry points do not route ptoas through "
                f"{PTOAS_ENTRYPOINT_TARGET}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate the packaged PTOAS wheel payload."
    )
    parser.add_argument(
        "wheel",
        help="Wheel file, directory containing exactly one ptoas*.whl, or glob.",
    )
    args = parser.parse_args()

    wheel = _resolve_wheel(args.wheel)
    validate_wheel_payload(wheel)
    print(f"validated wheel payload and launcher contract: {wheel.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
