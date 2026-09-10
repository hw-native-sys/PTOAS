# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""CCE backend for the rope kernel."""

from __future__ import annotations

import ctypes
import os
import subprocess
from pathlib import Path

from kernel_test.backends import RunPurpose
from kernel_test.npu_runtime import ensure_runtime, stream_ptr, sync

from ..runtime import RopeLaunchArgs, prepare_launch_args
from ..tile_config import DTYPES, MODES, sim_fn_name

_CCE_ROOT = Path(__file__).resolve().parent
_BUILD_DIR = _CCE_ROOT / "build"
_KERNEL_SOURCE = _CCE_ROOT / "rope_cce_kernel.cpp"
_LIB_PATH = _BUILD_DIR / "librope_cce.so"
_CMAKE_FILE = _CCE_ROOT / "CMakeLists.txt"
_SIM_SYMBOLS: tuple[str, ...] = tuple(
    sim_fn_name(mode, dtype, cycle=cycle)
    for cycle in (False, True)
    for dtype in DTYPES
    for mode in MODES
)
_LIB: ctypes.CDLL | None = None


def _ascend_home() -> Path:
    home = os.environ.get("ASCEND_HOME_PATH") or os.environ.get("ASCEND_TOOLKIT_HOME")
    if not home:
        raise EnvironmentError("ASCEND_HOME_PATH is not set. Source CANN setenv.bash first.")
    return Path(home)


def _run(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


def _build_lib(force: bool = False) -> Path:
    _BUILD_DIR.mkdir(parents=True, exist_ok=True)
    if _LIB_PATH.is_file() and not force:
        return _LIB_PATH

    ascend = _ascend_home()
    driver = os.environ.get("ASCEND_DRIVER_PATH", "/usr/local/Ascend/driver")
    _run(
        [
            "cmake",
            "-S",
            str(_CCE_ROOT),
            "-B",
            str(_BUILD_DIR),
            f"-DASCEND_HOME_PATH={ascend}",
            f"-DASCEND_DRIVER_PATH={driver}",
        ],
        cwd=_CCE_ROOT,
    )
    _run(["cmake", "--build", str(_BUILD_DIR), "--target", "rope_cce"], cwd=_CCE_ROOT)
    return _LIB_PATH


def _bind_lib(lib: ctypes.CDLL) -> None:
    argtypes = [ctypes.c_void_p] * 6
    for name in _SIM_SYMBOLS:
        fn = getattr(lib, name)
        fn.argtypes = argtypes
        fn.restype = None


def _load_lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is None:
        _LIB = ctypes.CDLL(str(_build_lib()))
        _bind_lib(_LIB)
    return _LIB


def _vp(t) -> ctypes.c_void_p:
    return ctypes.c_void_p(t.data_ptr())

def _launch_cce(lib: ctypes.CDLL, launch_args: RopeLaunchArgs) -> object:
    fn = getattr(lib, launch_args.fn_name)
    fn(
        stream_ptr(),
        _vp(launch_args.x),
        _vp(launch_args.cos),
        _vp(launch_args.sin),
        _vp(launch_args.y),
        _vp(launch_args.params),
    )
    sync()
    return launch_args.y


def rope_f16(launch_args: RopeLaunchArgs) -> object:
    """Launch the local rope f16 CCE kernel."""

    return _launch_cce(_load_lib(), launch_args)


def rope_bf16(launch_args: RopeLaunchArgs) -> object:
    """Launch the local rope bf16 CCE kernel."""

    return _launch_cce(_load_lib(), launch_args)


def rope_f32(launch_args: RopeLaunchArgs) -> object:
    """Launch the local rope f32 CCE kernel."""

    return _launch_cce(_load_lib(), launch_args)


class RopeCceBackend:
    """CCE rope backend implemented locally under kernel-test."""

    name = "cce"
    _launchers = {
        "f16": rope_f16,
        "bf16": rope_bf16,
        "f32": rope_f32,
    }

    def is_supported(self, case: object, *, purpose: RunPurpose) -> tuple[bool, str | None]:
        del case, purpose
        return True, None

    def launch(self, case: object, *, purpose: RunPurpose) -> object:
        ensure_runtime("rope")
        launch_args = prepare_launch_args(case, cycle=purpose == "cycle")
        return self._launchers[launch_args.dtype](launch_args)
