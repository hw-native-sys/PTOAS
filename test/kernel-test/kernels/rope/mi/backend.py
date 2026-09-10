# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""PTODSL MI backend for the rope kernel."""

import os
from pathlib import Path

from ptodsl import pto

from kernel_test.backends import ArtifactPlan, RunPurpose
from kernel_test.npu_runtime import ensure_runtime, stream_ptr, sync

from ..runtime import RopeLaunchArgs, artifact_case_dir, prepare_launch_args
from ..tile_config import MAX_N, MAX_S, SIM_D

_MI_ROOT = Path(__file__).resolve().parent
_GENERATED_DIR = _MI_ROOT.parent / "generated"
_MAX_XY_ROWS = MAX_S * MAX_N
_CS_ELEMS = MAX_S * SIM_D
_XY_ELEMS = _MAX_XY_ROWS * SIM_D
_UB_BASE_2B_COS = 0
_UB_BASE_2B_SIN = _UB_BASE_2B_COS + _CS_ELEMS * 2
_UB_BASE_2B_X = _UB_BASE_2B_SIN + _CS_ELEMS * 2
_UB_BASE_2B_Y = _UB_BASE_2B_X + _XY_ELEMS * 2
_UB_BASE_4B_COS = 0
_UB_BASE_4B_SIN = _UB_BASE_4B_COS + _CS_ELEMS * 4
_UB_BASE_4B_X = _UB_BASE_4B_SIN + _CS_ELEMS * 4
_UB_BASE_4B_Y = _UB_BASE_4B_X + _XY_ELEMS * 4


@pto.jit(
    name="rope_mi_f16",
    target="a5",
    backend="vpto",
    mode="explicit",
    kernel_kind="vector",
    insert_sync=False,
)
def rope_mi_f16(
    x_gm: pto.ptr(pto.f16, "gm"),
    cos_gm: pto.ptr(pto.f16, "gm"),
    sin_gm: pto.ptr(pto.f16, "gm"),
    y_gm: pto.ptr(pto.f16, "gm"),
    s_count: pto.i32,
    n_count: pto.i32,
    *,
    MODE: pto.const_expr = 0,
):
    rows = s_count * n_count
    row_bytes = SIM_D * 2
    cs_bytes = SIM_D * 2

    cos_ptr = pto.castptr(pto.const(_UB_BASE_2B_COS, dtype=pto.ui64), pto.ptr(pto.f16, "ub"))
    sin_ptr = pto.castptr(pto.const(_UB_BASE_2B_SIN, dtype=pto.ui64), pto.ptr(pto.f16, "ub"))
    x_ptr = pto.castptr(pto.const(_UB_BASE_2B_X, dtype=pto.ui64), pto.ptr(pto.f16, "ub"))
    y_ptr = pto.castptr(pto.const(_UB_BASE_2B_Y, dtype=pto.ui64), pto.ptr(pto.f16, "ub"))

    pto.mte_gm_ub(cos_gm, cos_ptr, 0, cs_bytes, nburst=(s_count, cs_bytes, cs_bytes))
    pto.mte_gm_ub(sin_gm, sin_ptr, 0, cs_bytes, nburst=(s_count, cs_bytes, cs_bytes))
    pto.mte_gm_ub(x_gm, x_ptr, 0, row_bytes, nburst=(rows, row_bytes, row_bytes))

    pto.set_flag(pto.Pipe.MTE2, pto.Pipe.V, event_id=0)
    pto.wait_flag(pto.Pipe.MTE2, pto.Pipe.V, event_id=0)

    half_mask = pto.pge_b16(pto.MaskPattern.VL32)
    interleave_mask = pto.pge_b16(pto.MaskPattern.VL64)
    x_s_step = n_count * SIM_D

    if MODE == 0:
        for s in range(0, s_count, 1):
            x_s_off = s * x_s_step
            cs_off = s * SIM_D
            cs_hi_off = cs_off + 32

            cos_lo = pto.vlds(cos_ptr, cs_off)
            cos_hi = pto.vlds(cos_ptr, cs_hi_off)
            sin_lo = pto.vlds(sin_ptr, cs_off)
            sin_hi = pto.vlds(sin_ptr, cs_hi_off)

            for n in range(0, n_count, 1):
                row_off = x_s_off + n * SIM_D
                x_hi_off = row_off + 32

                x_lo = pto.vlds(x_ptr, row_off)
                x_hi = pto.vlds(x_ptr, x_hi_off)

                y_lo = pto.vsub(
                    pto.vmul(cos_lo, x_lo, half_mask),
                    pto.vmul(sin_lo, x_hi, half_mask),
                    half_mask,
                )
                y_hi = pto.vadd(
                    pto.vmul(cos_hi, x_hi, half_mask),
                    pto.vmul(sin_hi, x_lo, half_mask),
                    half_mask,
                )

                pto.vsts(y_lo, y_ptr, row_off, half_mask)
                pto.vsts(y_hi, y_ptr, x_hi_off, half_mask)
    else:
        for s in range(0, s_count, 1):
            x_s_off = s * x_s_step
            cs_off = s * SIM_D

            cos_even, cos_odd = pto.vldsx2(
                cos_ptr,
                cs_off,
                pto.DeinterleaveDist.DINTLV_B16,
            )
            sin_even, sin_odd = pto.vldsx2(
                sin_ptr,
                cs_off,
                pto.DeinterleaveDist.DINTLV_B16,
            )

            for n in range(0, n_count, 1):
                row_off = x_s_off + n * SIM_D
                x_even, x_odd = pto.vldsx2(
                    x_ptr,
                    row_off,
                    pto.DeinterleaveDist.DINTLV_B16,
                )

                y_even = pto.vsub(
                    pto.vmul(x_even, cos_even, half_mask),
                    pto.vmul(x_odd, sin_even, half_mask),
                    half_mask,
                )
                y_odd = pto.vadd(
                    pto.vmul(x_odd, cos_odd, half_mask),
                    pto.vmul(x_even, sin_odd, half_mask),
                    half_mask,
                )

                pto.vstsx2(
                    y_even,
                    y_odd,
                    y_ptr,
                    row_off,
                    pto.InterleaveDist.INTLV_B16,
                    interleave_mask,
                )

    pto.set_flag(pto.Pipe.V, pto.Pipe.MTE3, event_id=0)
    pto.wait_flag(pto.Pipe.V, pto.Pipe.MTE3, event_id=0)
    pto.mte_ub_gm(y_ptr, y_gm, row_bytes, nburst=(rows, row_bytes, row_bytes))
    pto.pipe_barrier(pto.Pipe.ALL)


@pto.jit(
    name="rope_mi_bf16",
    target="a5",
    backend="vpto",
    mode="explicit",
    kernel_kind="vector",
    insert_sync=False,
)
def rope_mi_bf16(
    x_gm: pto.ptr(pto.bf16, "gm"),
    cos_gm: pto.ptr(pto.f16, "gm"),
    sin_gm: pto.ptr(pto.f16, "gm"),
    y_gm: pto.ptr(pto.bf16, "gm"),
    s_count: pto.i32,
    n_count: pto.i32,
    *,
    MODE: pto.const_expr = 0,
):
    rows = s_count * n_count
    row_bytes = SIM_D * 2
    cs_bytes = SIM_D * 2

    cos_ptr = pto.castptr(pto.const(_UB_BASE_2B_COS, dtype=pto.ui64), pto.ptr(pto.f16, "ub"))
    sin_ptr = pto.castptr(pto.const(_UB_BASE_2B_SIN, dtype=pto.ui64), pto.ptr(pto.f16, "ub"))
    x_ptr = pto.castptr(pto.const(_UB_BASE_2B_X, dtype=pto.ui64), pto.ptr(pto.bf16, "ub"))
    y_ptr = pto.castptr(pto.const(_UB_BASE_2B_Y, dtype=pto.ui64), pto.ptr(pto.bf16, "ub"))

    pto.mte_gm_ub(cos_gm, cos_ptr, 0, cs_bytes, nburst=(s_count, cs_bytes, cs_bytes))
    pto.mte_gm_ub(sin_gm, sin_ptr, 0, cs_bytes, nburst=(s_count, cs_bytes, cs_bytes))
    pto.mte_gm_ub(x_gm, x_ptr, 0, row_bytes, nburst=(rows, row_bytes, row_bytes))

    pto.set_flag(pto.Pipe.MTE2, pto.Pipe.V, event_id=0)
    pto.wait_flag(pto.Pipe.MTE2, pto.Pipe.V, event_id=0)

    mask16_all = pto.pset_b16(pto.MaskPattern.ALL)
    half_mask = pto.pge_b32(pto.MaskPattern.VL32)
    full_mask32 = pto.pset_b32(pto.MaskPattern.ALL)
    x_s_step = n_count * SIM_D

    if MODE == 0:
        for s in range(0, s_count, 1):
            x_s_off = s * x_s_step
            cs_off = s * SIM_D
            cs_hi_off = cs_off + 32

            cos_lo = pto.vcvt(pto.vlds(cos_ptr, cs_off, dist="UNPK_B16"), pto.f32, mask16_all, part="EVEN")
            cos_hi = pto.vcvt(pto.vlds(cos_ptr, cs_hi_off, dist="UNPK_B16"), pto.f32, mask16_all, part="EVEN")
            sin_lo = pto.vcvt(pto.vlds(sin_ptr, cs_off, dist="UNPK_B16"), pto.f32, mask16_all, part="EVEN")
            sin_hi = pto.vcvt(pto.vlds(sin_ptr, cs_hi_off, dist="UNPK_B16"), pto.f32, mask16_all, part="EVEN")

            for n in range(0, n_count, 1):
                row_off = x_s_off + n * SIM_D
                x_hi_off = row_off + 32

                x_lo = pto.vcvt(pto.vlds(x_ptr, row_off, dist="UNPK_B16"), pto.f32, mask16_all, part="EVEN")
                x_hi = pto.vcvt(pto.vlds(x_ptr, x_hi_off, dist="UNPK_B16"), pto.f32, mask16_all, part="EVEN")

                y_lo = pto.vsub(
                    pto.vmul(cos_lo, x_lo, half_mask),
                    pto.vmul(sin_lo, x_hi, half_mask),
                    half_mask,
                )
                y_hi = pto.vadd(
                    pto.vmul(cos_hi, x_hi, half_mask),
                    pto.vmul(sin_hi, x_lo, half_mask),
                    half_mask,
                )

                pto.vsts(
                    pto.vcvt(y_lo, pto.bf16, half_mask, rnd="R", sat="SAT", part="EVEN"),
                    y_ptr,
                    row_off,
                    half_mask,
                    dist=pto.VStoreDist.PK_B32,
                )
                pto.vsts(
                    pto.vcvt(y_hi, pto.bf16, half_mask, rnd="R", sat="SAT", part="EVEN"),
                    y_ptr,
                    x_hi_off,
                    half_mask,
                    dist=pto.VStoreDist.PK_B32,
                )
    else:
        for s in range(0, s_count, 1):
            x_s_off = s * x_s_step
            cs_off = s * SIM_D

            cos = pto.vcvt(pto.vlds(cos_ptr, cs_off, dist="UNPK_B16"), pto.f32, mask16_all, part="EVEN")
            sin = pto.vcvt(pto.vlds(sin_ptr, cs_off, dist="UNPK_B16"), pto.f32, mask16_all, part="EVEN")

            for n in range(0, n_count, 1):
                row_off = x_s_off + n * SIM_D
                x = pto.vcvt(pto.vlds(x_ptr, row_off, dist="UNPK_B16"), pto.f32, mask16_all, part="EVEN")
                x_even, x_odd = pto.vdintlv(x, x)
                rot, _ = pto.vintlv(pto.vneg(x_odd, half_mask), x_even)
                y = pto.vadd(
                    pto.vmul(x, cos, full_mask32),
                    pto.vmul(rot, sin, full_mask32),
                    full_mask32,
                )

                pto.vsts(
                    pto.vcvt(y, pto.bf16, full_mask32, rnd="R", sat="SAT", part="EVEN"),
                    y_ptr,
                    row_off,
                    full_mask32,
                    dist=pto.VStoreDist.PK_B32,
                )

    pto.set_flag(pto.Pipe.V, pto.Pipe.MTE3, event_id=0)
    pto.wait_flag(pto.Pipe.V, pto.Pipe.MTE3, event_id=0)
    pto.mte_ub_gm(y_ptr, y_gm, row_bytes, nburst=(rows, row_bytes, row_bytes))
    pto.pipe_barrier(pto.Pipe.ALL)


@pto.jit(
    name="rope_mi_f32",
    target="a5",
    backend="vpto",
    mode="explicit",
    kernel_kind="vector",
    insert_sync=False,
)
def rope_mi_f32(
    x_gm: pto.ptr(pto.f32, "gm"),
    cos_gm: pto.ptr(pto.f32, "gm"),
    sin_gm: pto.ptr(pto.f32, "gm"),
    y_gm: pto.ptr(pto.f32, "gm"),
    s_count: pto.i32,
    n_count: pto.i32,
    *,
    MODE: pto.const_expr = 0,
):
    rows = s_count * n_count
    row_bytes = SIM_D * 4
    cs_bytes = SIM_D * 4

    cos_ptr = pto.castptr(pto.const(_UB_BASE_4B_COS, dtype=pto.ui64), pto.ptr(pto.f32, "ub"))
    sin_ptr = pto.castptr(pto.const(_UB_BASE_4B_SIN, dtype=pto.ui64), pto.ptr(pto.f32, "ub"))
    x_ptr = pto.castptr(pto.const(_UB_BASE_4B_X, dtype=pto.ui64), pto.ptr(pto.f32, "ub"))
    y_ptr = pto.castptr(pto.const(_UB_BASE_4B_Y, dtype=pto.ui64), pto.ptr(pto.f32, "ub"))

    pto.mte_gm_ub(cos_gm, cos_ptr, 0, cs_bytes, nburst=(s_count, cs_bytes, cs_bytes))
    pto.mte_gm_ub(sin_gm, sin_ptr, 0, cs_bytes, nburst=(s_count, cs_bytes, cs_bytes))
    pto.mte_gm_ub(x_gm, x_ptr, 0, row_bytes, nburst=(rows, row_bytes, row_bytes))

    pto.set_flag(pto.Pipe.MTE2, pto.Pipe.V, event_id=0)
    pto.wait_flag(pto.Pipe.MTE2, pto.Pipe.V, event_id=0)

    half_mask = pto.pge_b32(pto.MaskPattern.VL32)
    interleave_mask = pto.pset_b32(pto.MaskPattern.ALL)
    x_s_step = n_count * SIM_D

    if MODE == 0:
        for s in range(0, s_count, 1):
            x_s_off = s * x_s_step
            cs_off = s * SIM_D
            cs_hi_off = cs_off + 32

            cos_lo = pto.vlds(cos_ptr, cs_off)
            cos_hi = pto.vlds(cos_ptr, cs_hi_off)
            sin_lo = pto.vlds(sin_ptr, cs_off)
            sin_hi = pto.vlds(sin_ptr, cs_hi_off)

            for n in range(0, n_count, 1):
                row_off = x_s_off + n * SIM_D
                x_hi_off = row_off + 32

                x_lo = pto.vlds(x_ptr, row_off)
                x_hi = pto.vlds(x_ptr, x_hi_off)

                y_lo = pto.vsub(
                    pto.vmul(cos_lo, x_lo, half_mask),
                    pto.vmul(sin_lo, x_hi, half_mask),
                    half_mask,
                )
                y_hi = pto.vadd(
                    pto.vmul(cos_hi, x_hi, half_mask),
                    pto.vmul(sin_hi, x_lo, half_mask),
                    half_mask,
                )

                pto.vsts(y_lo, y_ptr, row_off, half_mask)
                pto.vsts(y_hi, y_ptr, x_hi_off, half_mask)
    else:
        for s in range(0, s_count, 1):
            x_s_off = s * x_s_step
            cs_off = s * SIM_D

            cos_even, cos_odd = pto.vldsx2(
                cos_ptr,
                cs_off,
                pto.DeinterleaveDist.DINTLV_B32,
            )
            sin_even, sin_odd = pto.vldsx2(
                sin_ptr,
                cs_off,
                pto.DeinterleaveDist.DINTLV_B32,
            )

            for n in range(0, n_count, 1):
                row_off = x_s_off + n * SIM_D
                x_even, x_odd = pto.vldsx2(
                    x_ptr,
                    row_off,
                    pto.DeinterleaveDist.DINTLV_B32,
                )

                y_even = pto.vsub(
                    pto.vmul(x_even, cos_even, half_mask),
                    pto.vmul(x_odd, sin_even, half_mask),
                    half_mask,
                )
                y_odd = pto.vadd(
                    pto.vmul(x_odd, cos_odd, half_mask),
                    pto.vmul(x_even, sin_odd, half_mask),
                    half_mask,
                )

                pto.vstsx2(
                    y_even,
                    y_odd,
                    y_ptr,
                    row_off,
                    pto.InterleaveDist.INTLV_B32,
                    interleave_mask,
                )

    pto.set_flag(pto.Pipe.V, pto.Pipe.MTE3, event_id=0)
    pto.wait_flag(pto.Pipe.V, pto.Pipe.MTE3, event_id=0)
    pto.mte_ub_gm(y_ptr, y_gm, row_bytes, nburst=(rows, row_bytes, row_bytes))
    pto.pipe_barrier(pto.Pipe.ALL)


_COMPILED: dict[tuple[str, int], object] = {}


def _kernel_for_dtype(dtype: str):
    if dtype == "f16":
        return "f16", rope_mi_f16
    if dtype == "bf16":
        return "bf16", rope_mi_bf16
    if dtype == "f32":
        return "f32", rope_mi_f32
    raise ValueError(f"unsupported mi dtype: {dtype}")


def _prepare(dtype: str, mode_value: int) -> object:
    dtype_key, kernel = _kernel_for_dtype(dtype)
    key = (dtype_key, mode_value)
    compiled = _COMPILED.get(key)
    if compiled is None:
        compiled = kernel.compile(MODE=mode_value)
        _COMPILED[key] = compiled
    return compiled


def _launch(launch_args: RopeLaunchArgs):
    compiled = _prepare(launch_args.dtype, launch_args.mode_value)
    compiled[1, stream_ptr()](
        launch_args.x.data_ptr(),
        launch_args.cos.data_ptr(),
        launch_args.sin.data_ptr(),
        launch_args.y.data_ptr(),
        launch_args.s_count,
        launch_args.n_count,
    )
    sync()
    return launch_args.y


def _build_artifact_plan(case: dict[str, object]) -> ArtifactPlan:
    compiled = _prepare(case["dtype"], 0 if case["mode"] == "half" else 1)
    case_dir = artifact_case_dir(_GENERATED_DIR, case, backend_name="mi")
    return ArtifactPlan(
        generated_dir=_GENERATED_DIR,
        case_dir=case_dir,
        mi_text=compiled.mlir_text(),
    )


def rope_f16(launch_args: RopeLaunchArgs):
    """Launch the local rope f16 MI kernel."""

    return _launch(launch_args)


def rope_bf16(launch_args: RopeLaunchArgs):
    """Launch the local rope bf16 MI kernel."""

    return _launch(launch_args)


def rope_f32(launch_args: RopeLaunchArgs):
    """Launch the local rope f32 MI kernel."""

    return _launch(launch_args)


class RopeMiBackend:
    """Pure-PTODSL MI backend for rope."""

    name = "mi"
    _launchers = {
        "f16": rope_f16,
        "bf16": rope_bf16,
        "f32": rope_f32,
    }

    def is_supported(self, case: object, *, purpose: RunPurpose) -> tuple[bool, str | None]:
        del purpose
        supported = case["dtype"] in {"f16", "bf16", "f32"} and case["mode"] in {"half", "interleave"}
        if supported:
            return True, None
        return False, "backend=mi not wired for this case"

    def launch(self, case: object, *, purpose: RunPurpose) -> object:
        ensure_runtime("rope")
        launch_args = prepare_launch_args(case, cycle=purpose == "cycle")
        return self._launchers[launch_args.dtype](launch_args)

    def cache_tag(self) -> str:
        backend_py = _MI_ROOT / "backend.py"
        return f"mi:{backend_py}:{os.path.getmtime(backend_py):.0f}"

    def build_artifact_plan(self, case_id: str, case: object) -> ArtifactPlan:
        del case_id
        return _build_artifact_plan(case)
