#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import numpy as np

from common import assert_close, auto_main
from ptodsl import pto


M = 1
M_PADDED = 16
K = 256
N = 20
N_STORAGE = 32
SCALE_K = K // 32

L1_A_DATA_ADDR = 0
L1_A_SCALE_ADDR = 1024
L1_B_DATA_ADDR = 1152
L1_B_SCALE_ADDR = 9344
L1_BIAS_ADDR = 9600

L0_BASE_ADDR = 0
L0_SCALE_LHS_ADDR = 0
L0_SCALE_RHS_ADDR = 256

E4M3_BITS = np.asarray([0x00, 0x30, 0x38, 0x40, 0xB0, 0xB8, 0xC0], dtype=np.uint8)
E5M2_BITS = np.asarray([0x00, 0x38, 0x3C, 0x40, 0xB8, 0xBC, 0xC0], dtype=np.uint8)


def _decode_e4m3fn(values):
    values = np.asarray(values, dtype=np.uint8)
    sign = np.where((values & 0x80) != 0, -1.0, 1.0).astype(np.float32)
    exponent = (values >> 3) & 0x0F
    mantissa = values & 0x07
    out = np.zeros(values.shape, dtype=np.float32)
    subnormal = exponent == 0
    normal = (exponent != 0) & (exponent != 0x0F)
    out[subnormal] = mantissa[subnormal].astype(np.float32) * (2.0 ** -9)
    out[normal] = (1.0 + mantissa[normal].astype(np.float32) / 8.0) * np.exp2(
        exponent[normal].astype(np.int32) - 7
    )
    out[exponent == 0x0F] = np.nan
    return sign * out


def _decode_e5m2(values):
    values = np.asarray(values, dtype=np.uint8)
    sign = np.where((values & 0x80) != 0, -1.0, 1.0).astype(np.float32)
    exponent = (values >> 2) & 0x1F
    mantissa = values & 0x03
    out = np.zeros(values.shape, dtype=np.float32)
    subnormal = exponent == 0
    normal = (exponent != 0) & (exponent != 0x1F)
    out[subnormal] = mantissa[subnormal].astype(np.float32) * (2.0 ** -16)
    out[normal] = (1.0 + mantissa[normal].astype(np.float32) / 4.0) * np.exp2(
        exponent[normal].astype(np.int32) - 15
    )
    out[exponent == 0x1F] = np.nan
    return sign * out


def _pack_left_scale(scale):
    packed = np.zeros((M_PADDED, SCALE_K), dtype=np.uint8)
    packed[:M, :SCALE_K] = scale
    return packed


def _pack_right_scale(scale):
    padded = np.zeros((SCALE_K, N_STORAGE), dtype=np.uint8)
    padded[:SCALE_K, :N] = scale
    packed = padded.reshape((SCALE_K // 2, 2, N_STORAGE // 16, 16)).transpose(2, 0, 3, 1)
    return packed.reshape(SCALE_K, N_STORAGE)


def _build_fp8_payload(seed, *, variant):
    rng = np.random.default_rng(seed)

    a_raw = rng.choice(E4M3_BITS, size=(M_PADDED, K)).astype(np.uint8)
    b_raw = rng.choice(E5M2_BITS, size=(K, N_STORAGE)).astype(np.uint8)
    a_scale_logical = rng.integers(127, 130, size=(M, SCALE_K), dtype=np.uint8)
    b_scale_logical = rng.integers(127, 130, size=(SCALE_K, N), dtype=np.uint8)

    a_scale_raw = _pack_left_scale(a_scale_logical)
    b_scale_raw = _pack_right_scale(b_scale_logical)

    bias_raw = None
    if variant == "bias":
        bias_raw = np.zeros((M, N_STORAGE), dtype=np.float32)
        bias_raw[:, :N] = rng.uniform(-2.0, 2.0, size=(M, N)).astype(np.float32)

    a = _decode_e4m3fn(a_raw[:M, :K])
    b = _decode_e5m2(b_raw[:K, :N])
    a_scale = np.exp2(a_scale_logical.astype(np.int16) - 127).astype(np.float32)
    b_scale = np.exp2(b_scale_logical.astype(np.int16) - 127).astype(np.float32)

    a_real = a * a_scale[:, np.arange(K) // 32]
    b_real = b * b_scale[np.arange(K) // 32, :]
    golden_valid = (a_real @ b_real).astype(np.float32)

    if variant == "acc":
        golden_valid = golden_valid * 2.0
    elif variant == "bias":
        golden_valid = golden_valid + bias_raw[:, :N]

    golden = np.zeros((M, N_STORAGE), dtype=np.float32)
    golden[:, :N] = golden_valid

    inputs = [a_raw, b_raw, a_scale_raw, b_scale_raw]
    if bias_raw is not None:
        inputs.append(bias_raw)
    return inputs, golden


def _alloc_common_tiles():
    lhs_tile = pto.alloc_tile(
        shape=[M, K],
        dtype=pto.f8e4m3,
        memory_space=pto.MemorySpace.LEFT,
        addr=L0_BASE_ADDR,
        valid_shape=[M, K],
        blayout="ColMajor",
        slayout="RowMajor",
    )
    lhs_scale_tile = pto.alloc_tile(
        shape=[M, SCALE_K],
        dtype=pto.f16,
        memory_space=pto.MemorySpace.SCALING,
        addr=L0_SCALE_LHS_ADDR,
        valid_shape=[M, SCALE_K],
        blayout="RowMajor",
        slayout="RowMajor",
        fractal_size=32,
    )
    rhs_tile = pto.alloc_tile(
        shape=[K, N_STORAGE],
        dtype=pto.f8e5m2,
        memory_space=pto.MemorySpace.RIGHT,
        addr=L0_BASE_ADDR,
        valid_shape=[K, N],
        blayout="RowMajor",
        slayout="ColMajor",
    )
    rhs_scale_tile = pto.alloc_tile(
        shape=[SCALE_K, N_STORAGE],
        dtype=pto.f16,
        memory_space=pto.MemorySpace.SCALING,
        addr=L0_SCALE_RHS_ADDR,
        valid_shape=[SCALE_K, N],
        blayout="ColMajor",
        slayout="ColMajor",
        fractal_size=32,
    )
    dst_tile = pto.alloc_tile(
        shape=[M, N_STORAGE],
        dtype=pto.f32,
        memory_space=pto.MemorySpace.ACC,
        addr=L0_BASE_ADDR,
        valid_shape=[M, N],
        blayout="ColMajor",
        slayout="RowMajor",
    )
    return lhs_tile, lhs_scale_tile, rhs_tile, rhs_scale_tile, dst_tile


def _alloc_bias_tile():
    return pto.alloc_tile(
        shape=[M, N_STORAGE],
        dtype=pto.f32,
        memory_space=pto.MemorySpace.BIAS,
        addr=L0_BASE_ADDR,
        valid_shape=[M, N],
        blayout="RowMajor",
        slayout="NoneBox",
    )


def _stage_fp8_tiles(a_ptr, b_ptr, a_scale_ptr, b_scale_ptr, lhs_tile, rhs_tile, bias_ptr=None, bias_tile=None):
    a_l1_ptr = pto.castptr(pto.ui64(L1_A_DATA_ADDR), pto.ptr(pto.f8e4m3, "mat"))
    a_scale_l1_ptr = pto.castptr(pto.ui64(L1_A_SCALE_ADDR), pto.ptr(pto.f8e4m3, "mat"))
    b_l1_ptr = pto.castptr(pto.ui64(L1_B_DATA_ADDR), pto.ptr(pto.f8e5m2, "mat"))
    b_scale_l1_ptr = pto.castptr(pto.ui64(L1_B_SCALE_ADDR), pto.ptr(pto.f8e5m2, "mat"))

    pto.mte_gm_l1(a_ptr, a_l1_ptr, 1024, nburst=(1, 0, 0))
    pto.mte_gm_l1(a_scale_ptr, a_scale_l1_ptr, 64, nburst=(2, 0, 0))
    pto.mte_gm_l1_frac(
        b_ptr,
        b_l1_ptr,
        pto.FractalMode.ND2NZ,
        shape=(K, N),
        src_layout=(N_STORAGE,),
        dst_group=(1, 1, K, 0),
        ctrl=(0, False),
    )
    pto.mte_gm_l1(b_scale_ptr, b_scale_l1_ptr, 256, nburst=(1, 0, 0))

    bias_l1_ptr = None
    if bias_ptr is not None and bias_tile is not None:
        bias_l1_ptr = pto.castptr(pto.ui64(L1_BIAS_ADDR), pto.ptr(pto.f32, "mat"))
        pto.mte_gm_l1(bias_ptr, bias_l1_ptr, 128, nburst=(1, 0, 0))

    pto.set_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=0)
    pto.wait_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=0)

    pto.mte_l1_l0a(a_l1_ptr, lhs_tile.as_ptr(), M, K)
    pto.mte_l1_l0b(b_l1_ptr, rhs_tile.as_ptr(), K, N, transpose=True)
    pto.mte_l1_l0a_mx(a_scale_l1_ptr, lhs_tile.as_ptr(), M, K)
    pto.mte_l1_l0b_mx(b_scale_l1_ptr, rhs_tile.as_ptr(), K, N)
    if bias_l1_ptr is not None and bias_tile is not None:
        pto.mte_l1_bt(bias_l1_ptr, bias_tile.as_ptr(), 128, nburst=(1, 0, 0))

    pto.set_flag(pto.Pipe.MTE1, pto.Pipe.M, event_id=0)
    pto.wait_flag(pto.Pipe.MTE1, pto.Pipe.M, event_id=0)


def _writeback_output(dst_tile, out_ptr):
    pto.set_flag(pto.Pipe.M, pto.Pipe.FIX, event_id=1)
    pto.wait_flag(pto.Pipe.M, pto.Pipe.FIX, event_id=1)
    pto.mte_l0c_gm(
        dst_tile.as_ptr(),
        out_ptr,
        M,
        N_STORAGE,
        16,
        N_STORAGE,
        0,
        0,
        layout="nz2nd",
    )
    pto.pipe_barrier(pto.Pipe.ALL)


@pto.jit(
    name="gemv_mx_fp8_pipeline_kernel",
    kernel_kind="cube",
    target="a5",
    mode="explicit",
    insert_sync=False,
)
def gemv_mx_fp8_pipeline_kernel(
    a_ptr: pto.ptr(pto.f8e4m3, "gm"),
    b_ptr: pto.ptr(pto.f8e5m2, "gm"),
    a_scale_ptr: pto.ptr(pto.f8e4m3, "gm"),
    b_scale_ptr: pto.ptr(pto.f8e5m2, "gm"),
    out_ptr: pto.ptr(pto.f32, "gm"),
):
    lhs_tile, lhs_scale_tile, rhs_tile, rhs_scale_tile, dst_tile = _alloc_common_tiles()
    _stage_fp8_tiles(a_ptr, b_ptr, a_scale_ptr, b_scale_ptr, lhs_tile, rhs_tile)
    pto.tile.gemv_mx(lhs_tile, lhs_scale_tile, rhs_tile, rhs_scale_tile, dst_tile)
    _writeback_output(dst_tile, out_ptr)


@pto.jit(
    name="gemv_mx_acc_fp8_pipeline_kernel",
    kernel_kind="cube",
    target="a5",
    mode="explicit",
    insert_sync=False,
)
def gemv_mx_acc_fp8_pipeline_kernel(
    a_ptr: pto.ptr(pto.f8e4m3, "gm"),
    b_ptr: pto.ptr(pto.f8e5m2, "gm"),
    a_scale_ptr: pto.ptr(pto.f8e4m3, "gm"),
    b_scale_ptr: pto.ptr(pto.f8e5m2, "gm"),
    out_ptr: pto.ptr(pto.f32, "gm"),
):
    lhs_tile, lhs_scale_tile, rhs_tile, rhs_scale_tile, dst_tile = _alloc_common_tiles()
    _stage_fp8_tiles(a_ptr, b_ptr, a_scale_ptr, b_scale_ptr, lhs_tile, rhs_tile)
    pto.tile.gemv_mx(lhs_tile, lhs_scale_tile, rhs_tile, rhs_scale_tile, dst_tile)
    pto.tile.gemv_mx_acc(dst_tile, lhs_tile, lhs_scale_tile, rhs_tile, rhs_scale_tile, dst_tile)
    _writeback_output(dst_tile, out_ptr)


@pto.jit(
    name="gemv_mx_bias_fp8_pipeline_kernel",
    kernel_kind="cube",
    target="a5",
    mode="explicit",
    insert_sync=False,
)
def gemv_mx_bias_fp8_pipeline_kernel(
    a_ptr: pto.ptr(pto.f8e4m3, "gm"),
    b_ptr: pto.ptr(pto.f8e5m2, "gm"),
    a_scale_ptr: pto.ptr(pto.f8e4m3, "gm"),
    b_scale_ptr: pto.ptr(pto.f8e5m2, "gm"),
    bias_ptr: pto.ptr(pto.f32, "gm"),
    out_ptr: pto.ptr(pto.f32, "gm"),
):
    lhs_tile, lhs_scale_tile, rhs_tile, rhs_scale_tile, dst_tile = _alloc_common_tiles()
    bias_tile = _alloc_bias_tile()
    _stage_fp8_tiles(
        a_ptr,
        b_ptr,
        a_scale_ptr,
        b_scale_ptr,
        lhs_tile,
        rhs_tile,
        bias_ptr=bias_ptr,
        bias_tile=bias_tile,
    )
    pto.tile.gemv_mx_bias(lhs_tile, lhs_scale_tile, rhs_tile, rhs_scale_tile, bias_tile, dst_tile)
    _writeback_output(dst_tile, out_ptr)


def _make_case(name, kernel, *, seed, variant):
    def make_case():
        inputs, golden = _build_fp8_payload(seed, variant=variant)
        out = np.zeros((M, N_STORAGE), dtype=np.float32)
        return [*inputs, out], golden

    def check(device_inputs, golden):
        actual = device_inputs[-1].cpu().numpy()
        assert_close(actual, golden, rtol=1e-3, atol=1e-3)

    return {
        "name": name,
        "kernel": kernel,
        "make_case": make_case,
        "check": check,
    }


CASES = [
    _make_case(
        "gemv_mx_fp8_e4m3_e5m2",
        gemv_mx_fp8_pipeline_kernel,
        seed=19,
        variant="plain",
    ),
    _make_case(
        "gemv_mx_acc_fp8_e4m3_e5m2_double",
        gemv_mx_acc_fp8_pipeline_kernel,
        seed=23,
        variant="acc",
    ),
    _make_case(
        "gemv_mx_bias_fp8_e4m3_e5m2",
        gemv_mx_bias_fp8_pipeline_kernel,
        seed=29,
        variant="bias",
    ),
]


auto_main(globals())
