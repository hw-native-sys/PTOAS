# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Resolve external toolchain binaries and CANN paths."""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def resolve_ptoas_binary() -> Path:
    for var in ("PTO_BUILD_DIR", "PTO_INSTALL_DIR"):
        if var in os.environ:
            cand = Path(os.environ[var]) / "tools" / "ptoas" / "ptoas"
            if cand.is_file():
                return cand
            cand2 = Path(os.environ[var]) / "bin" / "ptoas"
            if cand2.is_file():
                return cand2

    repo_root = Path(__file__).resolve().parents[4]
    candidates = [
        repo_root / "build" / "tools" / "ptoas" / "ptoas",
        repo_root / "install" / "bin" / "ptoas",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    for start in [Path.cwd()] + list(Path.cwd().parents):
        for pattern in [start / "install" / "bin" / "ptoas",
                        start / "build" / "tools" / "ptoas" / "ptoas"]:
            if pattern.is_file():
                return pattern

    from_path = shutil.which("ptoas")
    if from_path:
        return Path(from_path)

    raise FileNotFoundError(
        "unable to locate ptoas; build ptoas or add it to PATH after sourcing scripts/ptoas_env.sh"
    )


def resolve_bisheng() -> str:
    ascend_home = ascend_home_path()
    candidate = ascend_home / "bin" / "bisheng"
    if candidate.is_file():
        return str(candidate)

    found = shutil.which("bisheng")
    if found:
        return found

    raise FileNotFoundError("bisheng compiler not found; source ASCEND setenv.bash first")


def ascend_home_path() -> Path:
    home = os.environ.get("ASCEND_HOME_PATH")
    if not home:
        raise EnvironmentError("ASCEND_HOME_PATH is not set; source CANN setenv.bash first")
    return Path(home)


def ascend_driver_path() -> Path:
    return Path(os.environ.get("ASCEND_DRIVER_PATH", "/usr/local/Ascend/driver"))


def common_include_flags() -> list[str]:
    ascend = ascend_home_path()
    driver = ascend_driver_path()
    return [
        f"-I{ascend}/include",
        f"-I{driver}/kernel/inc",
        f"-I{ascend}/pkg_inc",
        f"-I{ascend}/pkg_inc/profiling",
        f"-I{ascend}/pkg_inc/runtime/runtime",
    ]


def aicore_arch_for_kernel_kind(kernel_kind: str, target_arch: str = "a5") -> str:
    target = target_arch.lower()
    if kernel_kind == "vector":
        if target in {"a2", "a3"}:
            return "dav-c220-vec"
        return "dav-c310-vec"
    if kernel_kind == "cube":
        if target in {"a2", "a3"}:
            return "dav-c220-cube"
        return "dav-c310-cube"
    raise ValueError(f"unsupported kernel_kind for native build: {kernel_kind!r}")


__all__ = [
    "aicore_arch_for_kernel_kind",
    "ascend_driver_path",
    "ascend_home_path",
    "common_include_flags",
    "resolve_bisheng",
    "resolve_ptoas_binary",
]
