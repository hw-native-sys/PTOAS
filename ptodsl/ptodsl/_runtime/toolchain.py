# coding=utf-8
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
    env_override = os.environ.get("PTOAS_BIN")
    if env_override:
        candidate = Path(env_override)
        if candidate.is_file():
            return candidate
        from_path = shutil.which(env_override)
        if from_path:
            return Path(from_path)
        raise FileNotFoundError(
            f"PTOAS_BIN is set but does not resolve to an existing executable: {env_override}"
        )

    from_path = shutil.which("ptoas")
    if from_path:
        return Path(from_path)

    raise FileNotFoundError(
        "unable to locate ptoas; install it, add it to PATH, or set PTOAS_BIN"
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


def _append_include_flag(flags: list[str], path: Path) -> None:
    try:
        is_dir = path.is_dir()
    except OSError:
        return
    if not is_dir:
        return
    flag = f"-I{path}"
    if flag not in flags:
        flags.append(flag)


def _has_pto_isa_header(include_dir: Path) -> bool:
    return (include_dir / "pto" / "pto-inst.hpp").is_file()


def _pto_isa_include_dirs() -> list[Path]:
    pto_isa_root = os.environ.get("PTO_ISA_PATH") or os.environ.get("PTO_ISA_ROOT")
    if not pto_isa_root:
        return []

    root = Path(pto_isa_root)
    dirs: list[Path] = []
    include_dir = root / "include"
    if _has_pto_isa_header(include_dir):
        dirs.append(include_dir)
    common_dir = root / "tests" / "common"
    if common_dir.is_dir():
        dirs.append(common_dir)
    if _has_pto_isa_header(root):
        dirs.append(root)
    return dirs


def common_include_flags() -> list[str]:
    ascend = ascend_home_path()
    driver = ascend_driver_path()
    flags: list[str] = []
    for include_dir in _pto_isa_include_dirs():
        _append_include_flag(flags, include_dir)
    _append_include_flag(flags, ascend / "include")
    _append_include_flag(flags, driver / "kernel" / "inc")
    _append_include_flag(flags, ascend / "pkg_inc")
    _append_include_flag(flags, ascend / "pkg_inc" / "profiling")
    _append_include_flag(flags, ascend / "pkg_inc" / "runtime" / "runtime")
    return flags


def _append_unique_dir(dirs: list[Path], path: Path) -> None:
    if not path.is_dir():
        return
    if path in dirs:
        return
    dirs.append(path)


def simulator_library_dirs() -> list[Path]:
    ascend = ascend_home_path()
    dirs: list[Path] = []

    _append_unique_dir(
        dirs, ascend / "tools" / "simulator" / "Ascend950PR_9599" / "lib"
    )

    sim_lib_dir = os.environ.get("SIM_LIB_DIR")
    if sim_lib_dir:
        _append_unique_dir(dirs, Path(sim_lib_dir))
        return dirs

    _append_unique_dir(dirs, ascend / "x86_64-linux" / "simulator" / "dav_3510" / "lib")
    for candidate in sorted(ascend.rglob("simulator/dav_3510/lib")):
        _append_unique_dir(dirs, candidate)
    return dirs


def runtime_library_flags(*, sim_mode: bool = False) -> list[str]:
    ascend = ascend_home_path()
    lib_dirs = [ascend / "lib64"]
    if sim_mode:
        lib_dirs.extend(simulator_library_dirs())

    flags: list[str] = []
    for lib_dir in lib_dirs:
        flags.extend([f"-L{lib_dir}", f"-Wl,-rpath,{lib_dir}"])

    flags.append("-Wl,--no-as-needed")
    flags.append("-lruntime_camodel" if sim_mode else "-lruntime")
    return flags


def aicore_arch_for_kernel_kind(kernel_kind: str | None, target_arch: str) -> str:
    target = target_arch.lower()
    if kernel_kind is None:
        return "dav-c220" if target in {"a2", "a3"} else "dav-c310"
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
    "runtime_library_flags",
    "simulator_library_dirs",
    "resolve_bisheng",
    "resolve_ptoas_binary",
]
