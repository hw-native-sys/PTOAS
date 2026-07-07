# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""
PTODSL softmax helper for Flash Attention vector kernels.

This file now keeps two layers:

- ``fa_softmax_init_vpto_kernel`` / ``fa_softmax_update_vpto_kernel``:
  ptr-ABI VPTO child modules intended to become separate backend objects
- ``fa_softmax_init_vpto`` / ``fa_softmax_update_vpto``:
  Tile-ABI ``@pto.tileop`` adapters that materialize ``as_ptr()`` internally
- ``fa_softmax_vpto_probe``: minimal entry wrapper for compile-only inspection

The intended structure is:

- auto-mode callers only see Tile arguments
- the ``@pto.tileop`` adapter bridges Tile -> ptr
- the explicit VPTO kernel module owns the micro-instruction body
"""

import argparse
from pathlib import Path
import time

import numpy as np

from ptodsl import pto


def _inv_sqrt(head_size: int) -> float:
    if head_size <= 0:
        raise ValueError("head_size must be positive")
    return float(head_size) ** -0.5


def softmax_init_reference(
    qk: np.ndarray,
    *,
    scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(p_nz_f16, running_max, running_sum)`` for the init pass."""
    qk_f32 = np.asarray(qk, dtype=np.float32)
    if qk_f32.ndim != 2:
        raise ValueError(f"qk must be 2D, got shape {qk_f32.shape}")
    running_max = np.max(qk_f32, axis=1, keepdims=True)
    shifted = qk_f32 * np.float32(scale) - running_max * np.float32(scale)
    probs = np.exp(shifted, dtype=np.float32)
    running_sum = np.sum(probs, axis=1, keepdims=True, dtype=np.float32)
    p_nz_f16 = probs.astype(np.float16)
    return p_nz_f16.copy(), running_max.astype(np.float32), running_sum.astype(np.float32)


def softmax_update_reference(
    qk: np.ndarray,
    running_max: np.ndarray,
    running_sum: np.ndarray,
    *,
    scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(p_nz_f16, new_running_max, new_running_sum, exp_scale)``."""
    qk_f32 = np.asarray(qk, dtype=np.float32)
    if qk_f32.ndim != 2:
        raise ValueError(f"qk must be 2D, got shape {qk_f32.shape}")
    rows = qk_f32.shape[0]
    running_max_f32 = np.asarray(running_max, dtype=np.float32).reshape(rows, 1)
    running_sum_f32 = np.asarray(running_sum, dtype=np.float32).reshape(rows, 1)

    local_max = np.max(qk_f32, axis=1, keepdims=True)
    new_running_max = np.maximum(local_max, running_max_f32)
    shifted = qk_f32 * np.float32(scale) - new_running_max * np.float32(scale)
    probs = np.exp(shifted, dtype=np.float32)
    exp_scale = np.exp(
        running_max_f32 * np.float32(scale) - new_running_max * np.float32(scale),
        dtype=np.float32,
    )
    new_running_sum = running_sum_f32 * exp_scale + np.sum(
        probs,
        axis=1,
        keepdims=True,
        dtype=np.float32,
    )
    p_nz_f16 = probs.astype(np.float16)
    return (
        p_nz_f16.copy(),
        new_running_max.astype(np.float32),
        new_running_sum.astype(np.float32),
        exp_scale.astype(np.float32),
    )


def _softmax_pointer_plan():
    return {
        "f32_lanes": pto.elements_per_vreg(pto.f32),
        "f16_lanes": pto.elements_per_vreg(pto.f16),
    }


def _pack_f32_chunk_to_u16(chunk, mask32):
    packed_f16 = pto.vcvt(chunk, pto.f16, mask32, rnd="R", sat="SAT", part="EVEN")
    return pto.vpack(pto.vbitcast(packed_f16, pto.ui32), pto.VPackPart.LOWER)


@pto.jit(
    name="fa_softmax_init_vpto_kernel",
    target="a5",
    entry=False,
    backend="vpto",
    mode="explicit",
    kernel_kind="vector",
    insert_sync=False,
)
def fa_softmax_init_vpto_kernel(
    qk_ptr: pto.ptr(pto.f32, "ub"),
    p_nz_ptr: pto.ptr(pto.f16, "ub"),
    running_max_ptr: pto.ptr(pto.f32, "ub"),
    running_sum_ptr: pto.ptr(pto.f32, "ub"),
    rows: pto.i32,
    cols: pto.i32,
    scale: pto.f32,
):
    ptr_plan = _softmax_pointer_plan()
    f32_lanes = ptr_plan["f32_lanes"]
    active32 = pto.pset_b32(pto.MaskPattern.ALL)
    active16, _ = pto.make_mask(pto.f16, f32_lanes)
    one32, _ = pto.make_mask(pto.f32, 1)
    p_nz_u16_ptr = pto.castptr(p_nz_ptr, pto.ptr(pto.ui16, "ub"))

    for row in range(0, rows, 1):
        row_base = row * cols
        row_max = pto.vdup(pto.f32(-3.4028235e38), active32)

        for col in range(0, cols, f32_lanes):
            vec = pto.vlds(qk_ptr, row_base + col, dist="NORM")
            chunk_max = pto.vcmax(vec, active32)
            chunk_max = pto.vdup(chunk_max, active32)
            row_max = pto.vmax(row_max, chunk_max, active32)
        row_max_scaled = pto.vmuls(row_max, scale, active32)
        row_sum = pto.vdup(pto.f32(0.0), active32)

        for col in range(0, cols, 2 * f32_lanes):
            vec0 = pto.vlds(qk_ptr, row_base + col, dist="NORM")
            vec1 = pto.vlds(qk_ptr, row_base + col + f32_lanes, dist="NORM")
            vec0_scaled = pto.vmuls(vec0, scale, active32)
            vec1_scaled = pto.vmuls(vec1, scale, active32)
            exp0 = pto.vexpdif(vec0_scaled, row_max_scaled, active32, part="ODD")
            exp1 = pto.vexpdif(vec1_scaled, row_max_scaled, active32, part="ODD")
            sum0 = pto.vdup(pto.vcadd(exp0, active32), active32)
            sum1 = pto.vdup(pto.vcadd(exp1, active32), active32)
            row_sum = pto.vadd(row_sum, sum0, active32)
            row_sum = pto.vadd(row_sum, sum1, active32)
            packed0 = _pack_f32_chunk_to_u16(exp0, active32)
            packed1 = _pack_f32_chunk_to_u16(exp1, active32)
            pto.vsts(packed0, p_nz_u16_ptr, row_base + col, active16, dist="NORM_B16")
            pto.vsts(packed1, p_nz_u16_ptr, row_base + col + f32_lanes, active16, dist="NORM_B16")
        pto.vsts(row_max, running_max_ptr, row, one32, dist="1PT_B32")
        pto.vsts(row_sum, running_sum_ptr, row, one32, dist="1PT_B32")


@pto.jit(
    name="fa_softmax_update_vpto_kernel",
    target="a5",
    entry=False,
    backend="vpto",
    mode="explicit",
    kernel_kind="vector",
    insert_sync=False,
)
def fa_softmax_update_vpto_kernel(
    qk_ptr: pto.ptr(pto.f32, "ub"),
    p_nz_ptr: pto.ptr(pto.f16, "ub"),
    running_max_ptr: pto.ptr(pto.f32, "ub"),
    running_sum_ptr: pto.ptr(pto.f32, "ub"),
    exp_scale_ptr: pto.ptr(pto.f32, "ub"),
    rows: pto.i32,
    cols: pto.i32,
    scale: pto.f32,
):
    ptr_plan = _softmax_pointer_plan()
    f32_lanes = ptr_plan["f32_lanes"]
    active32 = pto.pset_b32(pto.MaskPattern.ALL)
    active16, _ = pto.make_mask(pto.f16, f32_lanes)
    one32, _ = pto.make_mask(pto.f32, 1)
    p_nz_u16_ptr = pto.castptr(p_nz_ptr, pto.ptr(pto.ui16, "ub"))

    for row in range(0, rows, 1):
        row_base = row * cols
        old_max = pto.vlds(running_max_ptr, row, dist="BRC_B32")
        old_sum = pto.vlds(running_sum_ptr, row, dist="BRC_B32")
        local_max = pto.vdup(pto.f32(-3.4028235e38), active32)

        for col in range(0, cols, f32_lanes):
            vec = pto.vlds(qk_ptr, row_base + col, dist="NORM")
            chunk_max = pto.vcmax(vec, active32)
            chunk_max = pto.vdup(chunk_max, active32)
            local_max = pto.vmax(local_max, chunk_max, active32)
        final_max = pto.vmax(local_max, old_max, active32)
        scaled_old_max = pto.vmuls(old_max, scale, active32)
        scaled_final_max = pto.vmuls(final_max, scale, active32)
        exp_scale = pto.vexpdif(scaled_old_max, scaled_final_max, active32, part="ODD")
        row_sum = pto.vmul(old_sum, exp_scale, active32)

        for col in range(0, cols, 2 * f32_lanes):
            vec0 = pto.vlds(qk_ptr, row_base + col, dist="NORM")
            vec1 = pto.vlds(qk_ptr, row_base + col + f32_lanes, dist="NORM")
            vec0_scaled = pto.vmuls(vec0, scale, active32)
            vec1_scaled = pto.vmuls(vec1, scale, active32)
            exp0 = pto.vexpdif(vec0_scaled, scaled_final_max, active32, part="ODD")
            exp1 = pto.vexpdif(vec1_scaled, scaled_final_max, active32, part="ODD")
            sum0 = pto.vdup(pto.vcadd(exp0, active32), active32)
            sum1 = pto.vdup(pto.vcadd(exp1, active32), active32)
            row_sum = pto.vadd(row_sum, sum0, active32)
            row_sum = pto.vadd(row_sum, sum1, active32)
            packed0 = _pack_f32_chunk_to_u16(exp0, active32)
            packed1 = _pack_f32_chunk_to_u16(exp1, active32)
            pto.vsts(packed0, p_nz_u16_ptr, row_base + col, active16, dist="NORM_B16")
            pto.vsts(packed1, p_nz_u16_ptr, row_base + col + f32_lanes, active16, dist="NORM_B16")
        pto.vsts(final_max, running_max_ptr, row, one32, dist="1PT_B32")
        pto.vsts(row_sum, running_sum_ptr, row, one32, dist="1PT_B32")
        pto.vsts(exp_scale, exp_scale_ptr, row, one32, dist="1PT_B32")


@pto.tileop
def fa_softmax_init_vpto(
    qk: pto.Tile,
    p_nz: pto.Tile,
    running_max: pto.Tile,
    running_sum: pto.Tile,
    scale: pto.f32,
):
    rows, cols = qk.valid_shape
    fa_softmax_init_vpto_kernel(
        qk.as_ptr(),
        p_nz.as_ptr(),
        running_max.as_ptr(),
        running_sum.as_ptr(),
        rows,
        cols,
        scale,
    )


@pto.tileop
def fa_softmax_update_vpto(
    qk: pto.Tile,
    p_nz: pto.Tile,
    running_max: pto.Tile,
    running_sum: pto.Tile,
    exp_scale: pto.Tile,
    scale: pto.f32,
):
    rows, cols = qk.valid_shape
    fa_softmax_update_vpto_kernel(
        qk.as_ptr(),
        p_nz.as_ptr(),
        running_max.as_ptr(),
        running_sum.as_ptr(),
        exp_scale.as_ptr(),
        rows,
        cols,
        scale,
    )


@pto.jit(entry=True, target="a5", backend="emitc", mode="auto", insert_sync=True)
def fa_softmax_init_vpto_validate(
    qk_gm: pto.ptr(pto.f32, "gm"),
    p_nz_gm: pto.ptr(pto.f16, "gm"),
    running_max_gm: pto.ptr(pto.f32, "gm"),
    running_sum_gm: pto.ptr(pto.f32, "gm"),
    scale: pto.f32,
    *,
    BR: pto.const_expr = 32,
    BC: pto.const_expr = 256,
):
    qk_view = pto.make_tensor_view(
        qk_gm,
        shape=[1, 1, 1, BR, BC],
        strides=[BR * BC, BR * BC, BR * BC, BC, 1],
    )
    p_nz_view = pto.make_tensor_view(
        p_nz_gm,
        shape=[1, 1, 1, BR, BC],
        strides=[BR * BC, BR * BC, BR * BC, BC, 1],
    )
    running_max_view = pto.make_tensor_view(
        running_max_gm,
        shape=[1, 1, 1, BR, 1],
        strides=[BR, BR, BR, 1, 1],
    )
    running_sum_view = pto.make_tensor_view(
        running_sum_gm,
        shape=[1, 1, 1, BR, 1],
        strides=[BR, BR, BR, 1, 1],
    )

    qk = pto.alloc_tile(shape=[BR, BC], dtype=pto.f32, valid_shape=[BR, BC])
    p_nz = pto.alloc_tile(shape=[BR, BC], dtype=pto.f16, valid_shape=[BR, BC])
    running_max = pto.alloc_tile(shape=[BR, 1], dtype=pto.f32, valid_shape=[BR, 1], blayout="ColMajor")
    running_sum = pto.alloc_tile(shape=[BR, 1], dtype=pto.f32, valid_shape=[BR, 1], blayout="ColMajor")

    pto.tile.load(
        pto.partition_view(qk_view, offsets=[0, 0, 0, 0, 0], sizes=[1, 1, 1, BR, BC]),
        qk,
    )
    fa_softmax_init_vpto(qk, p_nz, running_max, running_sum, scale)
    pto.tile.store(
        p_nz,
        pto.partition_view(p_nz_view, offsets=[0, 0, 0, 0, 0], sizes=[1, 1, 1, BR, BC]),
    )
    pto.tile.store(
        running_max,
        pto.partition_view(running_max_view, offsets=[0, 0, 0, 0, 0], sizes=[1, 1, 1, BR, 1]),
    )
    pto.tile.store(
        running_sum,
        pto.partition_view(running_sum_view, offsets=[0, 0, 0, 0, 0], sizes=[1, 1, 1, BR, 1]),
    )


@pto.jit(entry=True, target="a5", backend="emitc", mode="auto", insert_sync=True)
def fa_softmax_update_vpto_validate(
    qk_gm: pto.ptr(pto.f32, "gm"),
    p_nz_gm: pto.ptr(pto.f16, "gm"),
    running_max_gm: pto.ptr(pto.f32, "gm"),
    running_sum_gm: pto.ptr(pto.f32, "gm"),
    exp_scale_gm: pto.ptr(pto.f32, "gm"),
    scale: pto.f32,
    *,
    BR: pto.const_expr = 32,
    BC: pto.const_expr = 256,
):
    qk_view = pto.make_tensor_view(
        qk_gm,
        shape=[1, 1, 1, BR, BC],
        strides=[BR * BC, BR * BC, BR * BC, BC, 1],
    )
    p_nz_view = pto.make_tensor_view(
        p_nz_gm,
        shape=[1, 1, 1, BR, BC],
        strides=[BR * BC, BR * BC, BR * BC, BC, 1],
    )
    running_max_view = pto.make_tensor_view(
        running_max_gm,
        shape=[1, 1, 1, BR, 1],
        strides=[BR, BR, BR, 1, 1],
    )
    running_sum_view = pto.make_tensor_view(
        running_sum_gm,
        shape=[1, 1, 1, BR, 1],
        strides=[BR, BR, BR, 1, 1],
    )
    exp_scale_view = pto.make_tensor_view(
        exp_scale_gm,
        shape=[1, 1, 1, BR, 1],
        strides=[BR, BR, BR, 1, 1],
    )

    qk = pto.alloc_tile(shape=[BR, BC], dtype=pto.f32, valid_shape=[BR, BC])
    p_nz = pto.alloc_tile(shape=[BR, BC], dtype=pto.f16, valid_shape=[BR, BC])
    running_max = pto.alloc_tile(shape=[BR, 1], dtype=pto.f32, valid_shape=[BR, 1], blayout="ColMajor")
    running_sum = pto.alloc_tile(shape=[BR, 1], dtype=pto.f32, valid_shape=[BR, 1], blayout="ColMajor")
    exp_scale = pto.alloc_tile(shape=[BR, 1], dtype=pto.f32, valid_shape=[BR, 1], blayout="ColMajor")

    pto.tile.load(
        pto.partition_view(qk_view, offsets=[0, 0, 0, 0, 0], sizes=[1, 1, 1, BR, BC]),
        qk,
    )
    pto.tile.load(
        pto.partition_view(running_max_view, offsets=[0, 0, 0, 0, 0], sizes=[1, 1, 1, BR, 1]),
        running_max,
    )
    pto.tile.load(
        pto.partition_view(running_sum_view, offsets=[0, 0, 0, 0, 0], sizes=[1, 1, 1, BR, 1]),
        running_sum,
    )
    fa_softmax_update_vpto(qk, p_nz, running_max, running_sum, exp_scale, scale)
    pto.tile.store(
        p_nz,
        pto.partition_view(p_nz_view, offsets=[0, 0, 0, 0, 0], sizes=[1, 1, 1, BR, BC]),
    )
    pto.tile.store(
        running_max,
        pto.partition_view(running_max_view, offsets=[0, 0, 0, 0, 0], sizes=[1, 1, 1, BR, 1]),
    )
    pto.tile.store(
        running_sum,
        pto.partition_view(running_sum_view, offsets=[0, 0, 0, 0, 0], sizes=[1, 1, 1, BR, 1]),
    )
    pto.tile.store(
        exp_scale,
        pto.partition_view(exp_scale_view, offsets=[0, 0, 0, 0, 0], sizes=[1, 1, 1, BR, 1]),
    )


@pto.jit(target="a5", mode="explicit")
def fa_softmax_vpto_probe(
    *,
    BR: pto.const_expr = 8,
    BC: pto.const_expr = 64,
    INIT: pto.const_expr = False,
    HEAD_SIZE: pto.const_expr = 64,
):
    softmax_scale = _inv_sqrt(HEAD_SIZE)

    qk = pto.alloc_tile(shape=[BR, BC], dtype=pto.f32, valid_shape=[BR, BC])
    p_nz = pto.alloc_tile(shape=[BR, BC], dtype=pto.f16, valid_shape=[BR, BC])
    running_max = pto.alloc_tile(shape=[BR, 1], dtype=pto.f32, valid_shape=[BR, 1], blayout="ColMajor")
    running_sum = pto.alloc_tile(shape=[BR, 1], dtype=pto.f32, valid_shape=[BR, 1], blayout="ColMajor")

    if INIT:
        fa_softmax_init_vpto(
            qk,
            p_nz,
            running_max,
            running_sum,
            softmax_scale,
        )
    else:
        exp_scale = pto.alloc_tile(shape=[BR, 1], dtype=pto.f32, valid_shape=[BR, 1], blayout="ColMajor")
        fa_softmax_update_vpto(
            qk,
            p_nz,
            running_max,
            running_sum,
            exp_scale,
            softmax_scale,
        )

__all__ = [
    "fa_softmax_init_vpto_kernel",
    "fa_softmax_update_vpto_kernel",
    "fa_softmax_init_vpto",
    "fa_softmax_update_vpto",
    "fa_softmax_init_vpto_validate",
    "fa_softmax_update_vpto_validate",
    "softmax_init_reference",
    "softmax_update_reference",
    "fa_softmax_vpto_probe",
]


_DEVICE = "npu:0"


def emit_softmax_mlir(*, init: bool, br: int, bc: int) -> str:
    compiled = (
        fa_softmax_init_vpto_validate.compile(BR=br, BC=bc)
        if init
        else fa_softmax_update_vpto_validate.compile(BR=br, BC=bc)
    )
    return compiled.mlir_text()


def compile_softmax_kernel(*, init: bool, br: int, bc: int):
    return (
        fa_softmax_init_vpto_validate.compile(BR=br, BC=bc)
        if init
        else fa_softmax_update_vpto_validate.compile(BR=br, BC=bc)
    )


def init_runtime():
    try:
        import torch
        import torch_npu  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "softmax launch validation requires a Python environment with both "
            "`torch` and `torch_npu` installed"
        ) from exc

    torch.npu.config.allow_internal_format = False
    torch_npu.npu.set_compile_mode(jit_compile=False)
    torch.npu.set_device(_DEVICE)
    return torch


def npu_stream(torch):
    return torch.npu.current_stream()._as_parameter_  # noqa: SLF001


def _to_device(torch, array: np.ndarray):
    return torch.from_numpy(np.ascontiguousarray(array)).to(_DEVICE)


def _assert_close(name: str, got: np.ndarray, ref: np.ndarray, *, rtol: float, atol: float) -> None:
    try:
        np.testing.assert_allclose(got, ref, rtol=rtol, atol=atol)
    except AssertionError as exc:
        diff = np.max(np.abs(got - ref))
        raise AssertionError(f"{name} mismatch, max_abs_diff={diff}\n{exc}") from exc


def run_demo(
    *,
    init: bool,
    br: int,
    bc: int,
    head_size: int,
    seed: int = 20260605,
) -> None:
    torch = init_runtime()
    rng = np.random.RandomState(seed)
    scale = np.float32(_inv_sqrt(head_size))
    stream = npu_stream(torch)

    t0 = time.perf_counter()
    compiled = compile_softmax_kernel(init=init, br=br, bc=bc)
    compile_s = time.perf_counter() - t0

    if init:
        qk = rng.uniform(-4.0, 4.0, size=(br, bc)).astype(np.float32)
        ref_p_nz, ref_running_max, ref_running_sum = softmax_init_reference(qk, scale=scale)

        qk_t = _to_device(torch, qk)
        p_nz_t = _to_device(torch, np.zeros((br, bc), dtype=np.float16))
        running_max_t = _to_device(torch, np.zeros((br, 1), dtype=np.float32))
        running_sum_t = _to_device(torch, np.zeros((br, 1), dtype=np.float32))

        t0 = time.perf_counter()
        compiled[1, stream](
            qk_t.data_ptr(),
            p_nz_t.data_ptr(),
            running_max_t.data_ptr(),
            running_sum_t.data_ptr(),
            float(scale),
        )
        torch.npu.synchronize()
        launch_s = time.perf_counter() - t0

        got_p_nz = p_nz_t.cpu().numpy()
        got_running_max = running_max_t.cpu().numpy()
        got_running_sum = running_sum_t.cpu().numpy()

        _assert_close(
            "init.p_nz",
            got_p_nz.astype(np.float32),
            ref_p_nz.astype(np.float32),
            rtol=2e-3,
            atol=2e-3,
        )
        _assert_close("init.running_max", got_running_max, ref_running_max, rtol=1e-6, atol=1e-6)
        _assert_close("init.running_sum", got_running_sum, ref_running_sum, rtol=2e-3, atol=2e-3)
        print(f"PASS softmax-init br={br} bc={bc} compile={compile_s:.3f}s launch={launch_s:.3f}s")
        return

    prev_qk = rng.uniform(-4.0, 4.0, size=(br, bc)).astype(np.float32)
    qk = rng.uniform(-4.0, 4.0, size=(br, bc)).astype(np.float32)
    _, running_max, running_sum = softmax_init_reference(prev_qk, scale=scale)
    ref_p_nz, ref_running_max, ref_running_sum, ref_exp_scale = softmax_update_reference(
        qk,
        running_max,
        running_sum,
        scale=scale,
    )

    qk_t = _to_device(torch, qk)
    p_nz_t = _to_device(torch, np.zeros((br, bc), dtype=np.float16))
    running_max_t = _to_device(torch, running_max)
    running_sum_t = _to_device(torch, running_sum)
    exp_scale_t = _to_device(torch, np.zeros((br, 1), dtype=np.float32))

    t0 = time.perf_counter()
    compiled[1, stream](
        qk_t.data_ptr(),
        p_nz_t.data_ptr(),
        running_max_t.data_ptr(),
        running_sum_t.data_ptr(),
        exp_scale_t.data_ptr(),
        float(scale),
    )
    torch.npu.synchronize()
    launch_s = time.perf_counter() - t0

    got_p_nz = p_nz_t.cpu().numpy()
    got_running_max = running_max_t.cpu().numpy()
    got_running_sum = running_sum_t.cpu().numpy()
    got_exp_scale = exp_scale_t.cpu().numpy()

    _assert_close(
        "update.p_nz",
        got_p_nz.astype(np.float32),
        ref_p_nz.astype(np.float32),
        rtol=2e-3,
        atol=2e-3,
    )
    _assert_close("update.running_max", got_running_max, ref_running_max, rtol=1e-6, atol=1e-6)
    _assert_close("update.running_sum", got_running_sum, ref_running_sum, rtol=2e-3, atol=2e-3)
    _assert_close("update.exp_scale", got_exp_scale, ref_exp_scale, rtol=2e-3, atol=2e-3)
    print(f"PASS softmax-update br={br} bc={bc} compile={compile_s:.3f}s launch={launch_s:.3f}s")


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--emit-mlir",
        action="store_true",
        help="print the probe MLIR and exit",
    )
    parser.add_argument(
        "--init",
        action="store_true",
        help="emit init probe MLIR when used with --emit-mlir; otherwise only run init validation",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="emit update probe MLIR when used with --emit-mlir; otherwise only run update validation",
    )
    parser.add_argument("--br", type=int, default=32)
    parser.add_argument("--bc", type=int, default=256)
    parser.add_argument("--head-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260605)
    parser.add_argument("-o", "--output", default="-", help="output MLIR path, or '-' for stdout")
    return parser


def main(argv=None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    run_init = args.init or not args.update
    run_update = args.update or not args.init

    if args.emit_mlir:
        if run_init:
            mlir_text = emit_softmax_mlir(init=True, br=args.br, bc=args.bc)
            if args.output == "-":
                print(mlir_text)
            else:
                Path(args.output).write_text(mlir_text, encoding="utf-8")
        if run_update:
            mlir_text = emit_softmax_mlir(init=False, br=args.br, bc=args.bc)
            if args.output == "-":
                print(mlir_text)
            else:
                suffix = ".update" if run_init and args.output != "-" else ""
                out_path = Path(args.output + suffix) if suffix else Path(args.output)
                out_path.write_text(mlir_text, encoding="utf-8")
        return 0

    if run_init:
        run_demo(
            init=True,
            br=args.br,
            bc=args.bc,
            head_size=args.head_size,
            seed=args.seed,
        )
    if run_update:
        run_demo(
            init=False,
            br=args.br,
            bc=args.bc,
            head_size=args.head_size,
            seed=args.seed + 1,
        )
    print("All requested softmax validation cases passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
