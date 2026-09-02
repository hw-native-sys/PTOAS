#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# C2V local FIFO, TILE_NO_SPLIT (AIV0-only)
#   Cube: matmul(A,B) → TPUSH NO_SPLIT full AccTile to UB
#   Vec (AIV0 only): TPOP NO_SPLIT → TADD bias → TFREE → TSTORE
#
# 5 cases: key1-3 standard matmul+bias, key4-5 AccValidShape strip.

from pathlib import Path
import sys

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import auto_main, golden_output_case
from ptodsl import pto


PIPE_ID = 0
SPLIT_NO_SPLIT = 0
CASE_TILE_M = 16

# AccValidShape strip: M=32, H=16, K=32, N=128, float, no bias
STRIP_M = 32
STRIP_H = 16
STRIP_K = 32
STRIP_N = 128
STRIP_COUNT = STRIP_M // STRIP_H
# Slot is sized by the VALID shape (H×N): `TPipe<..., sizeof(OutT) * H * N, ...>`.
STRIP_SLOT = 4 * STRIP_H * STRIP_N
# TileAcc uses fractalCSize (16×16 inner blocks); our tiles default to
# the 512 AB-fractal tag, so pass fractal_size=1024 to match TileAcc geometry.
# With 16×16 blocks, one row-strip of a [32,128] f32 acc sits at byte offset
# strip*H*64 (= strip*1024, `TASSIGN(accStrip, strip * H * 64)`).
ACC_FRACTAL = 1024
STRIP_VIEW_BYTES = STRIP_H * 64
STRIP_VIEW_ELEMS = STRIP_VIEW_BYTES // 4  # f32

_DTYPE_MAP = {
    np.float16: (pto.f16, "f16", 2),
    np.float32: (pto.f32, "f32", 4),
}

# (name, M, K, N, in_np, use_bias)
_CASE_PARAMS = [
    ("case1_half_single_tile",  16, 32, 32, np.float16, True),
    ("case2_half_two_tiles",    32, 32, 32, np.float16, True),
    ("case3_float_single_tile", 16, 32, 32, np.float32, True),
]


def _gen_inputs(seed, M, K, N, in_np, use_bias):
    np.random.seed(seed)
    a = np.random.randint(-5, 5, [M, K]).astype(in_np)
    b = np.random.randint(-5, 5, [K, N]).astype(in_np)
    bias = np.random.randint(-3, 3, [M, N]).astype(np.float32) if use_bias else None
    return [a, b, bias] if bias is not None else [a, b]


def _golden(a, b, bias=None):
    result = np.matmul(a.astype(np.float32), b.astype(np.float32))
    if bias is not None:
        result = result + bias
    return result.astype(np.float32)


def _make_kernels(M, K, N, in_pto, in_np, use_bias):
    in_name = _DTYPE_MAP[in_np][1]
    elem_bytes = _DTYPE_MAP[in_np][2]
    slot_size = 4 * CASE_TILE_M * N  # f32 output
    num_m_tiles = M // CASE_TILE_M

    @pto.jit(
        name=f"tpushpop_cvns_cube_{M}x{K}x{N}_{in_name}",
        kernel_kind="cube",
        target="a5",
        mode="auto",
        insert_sync=True,
        entry=False,
        backend="emitc",
    )
    def cube_kernel(
        a_ptr: pto.ptr(in_pto, "gm"),
        b_ptr: pto.ptr(in_pto, "gm"),
        bias_ptr: pto.ptr(pto.f32, "gm"),
        out_ptr: pto.ptr(pto.f32, "gm"),
    ):
        c2v_buf = pto.import_reserved_buffer(
            name="c2v_fifo",
            peer_func=f"tpushpop_cvns_vec_{M}x{K}x{N}_{in_name}",
        )
        pipe = pto.pipe.c2v(slot_size=slot_size, consumer_buf=c2v_buf, id=PIPE_ID, nosplit=True, slot_num=2)
        pipe.init_cube()

        a_mat = pto.alloc_tile(shape=[CASE_TILE_M, K], dtype=in_pto, memory_space="mat",
                               blayout="ColMajor", slayout="RowMajor", valid_shape=[CASE_TILE_M, K])
        b_mat = pto.alloc_tile(shape=[K, N], dtype=in_pto, memory_space="mat",
                               blayout="ColMajor", slayout="RowMajor", valid_shape=[K, N])
        a_left = pto.alloc_tile(shape=[CASE_TILE_M, K], dtype=in_pto, memory_space="left",
                                blayout="ColMajor", slayout="RowMajor", valid_shape=[CASE_TILE_M, K])
        b_right = pto.alloc_tile(shape=[K, N], dtype=in_pto, memory_space="right",
                                 blayout="RowMajor", slayout="ColMajor", valid_shape=[K, N])
        acc = pto.alloc_tile(shape=[CASE_TILE_M, N], dtype=pto.f32, memory_space="acc",
                             blayout="ColMajor", slayout="RowMajor", valid_shape=[CASE_TILE_M, N])

        for m_tile in range(num_m_tiles):
            a_view = pto.make_tensor_view(a_ptr, shape=[M, K], strides=[K, 1])
            b_view = pto.make_tensor_view(b_ptr, shape=[K, N], strides=[N, 1])
            pto.tile.load(a_view, a_mat, offsets=[m_tile * CASE_TILE_M, 0], sizes=[CASE_TILE_M, K])
            pto.tile.load(b_view, b_mat)
            pto.tile.mov(a_mat, a_left)
            pto.tile.mov(b_mat, b_right)
            pto.tile.matmul(a_left, b_right, acc)
            pipe.push(acc, split=SPLIT_NO_SPLIT)

    @pto.jit(
        name=f"tpushpop_cvns_vec_{M}x{K}x{N}_{in_name}",
        kernel_kind="vector",
        target="a5",
        mode="auto",
        insert_sync=True,
        entry=False,
        backend="emitc",
    )
    def vec_kernel(
        a_ptr: pto.ptr(in_pto, "gm"),
        b_ptr: pto.ptr(in_pto, "gm"),
        bias_ptr: pto.ptr(pto.f32, "gm"),
        out_ptr: pto.ptr(pto.f32, "gm"),
    ):
        c2v_buf = pto.reserve_buffer(
            name="c2v_fifo", size=slot_size * 4,
            location="vec", auto=True,
        )
        pipe = pto.pipe.c2v(slot_size=slot_size, consumer_buf=c2v_buf, id=PIPE_ID, nosplit=True, slot_num=2)
        pipe.init_simd()

        for m_tile in range(num_m_tiles):
            entry = pipe.pop(split=SPLIT_NO_SPLIT, result_type=pto.alloc_tile(
                shape=[CASE_TILE_M, N], dtype=pto.f32, valid_shape=[CASE_TILE_M, N],
            ))

            if use_bias:
                bias_tile = pto.alloc_tile(shape=[CASE_TILE_M, N], dtype=pto.f32, valid_shape=[CASE_TILE_M, N])
                out_tile = pto.alloc_tile(shape=[CASE_TILE_M, N], dtype=pto.f32, valid_shape=[CASE_TILE_M, N])
                bias_view = pto.make_tensor_view(bias_ptr, shape=[M, N], strides=[N, 1])
                pto.tile.load(bias_view, bias_tile, offsets=[m_tile * CASE_TILE_M, 0], sizes=[CASE_TILE_M, N])
                pto.tile.add(entry, bias_tile, out_tile)
                pipe.free(entry, split=SPLIT_NO_SPLIT)
                out_view = pto.make_tensor_view(out_ptr, shape=[M, N], strides=[N, 1])
                pto.tile.store(out_tile, out_view, offsets=[m_tile * CASE_TILE_M, 0], sizes=[CASE_TILE_M, N])
            else:
                pipe.free(entry, split=SPLIT_NO_SPLIT)
                out_view = pto.make_tensor_view(out_ptr, shape=[M, N], strides=[N, 1])
                pto.tile.store(entry, out_view, offsets=[m_tile * CASE_TILE_M, 0], sizes=[CASE_TILE_M, N])

    @pto.jit(
        name=f"tpushpop_cvns_entry_{M}x{K}x{N}_{in_name}",
        target="a5",
        mode="auto",
        insert_sync=True,
        backend="emitc",
    )
    def entry_kernel(
        a_ptr: pto.ptr(in_pto, "gm"),
        b_ptr: pto.ptr(in_pto, "gm"),
        bias_ptr: pto.ptr(pto.f32, "gm"),
        out_ptr: pto.ptr(pto.f32, "gm"),
    ):
        cube_kernel(a_ptr, b_ptr, bias_ptr, out_ptr)
        vec_kernel(a_ptr, b_ptr, bias_ptr, out_ptr)

    return entry_kernel


def _gen_strip_inputs(seed=2024):
    np.random.seed(seed)
    a = np.random.randint(-5, 5, [STRIP_M, STRIP_K]).astype(np.float32)
    b = np.random.randint(-5, 5, [STRIP_K, STRIP_N]).astype(np.float32)
    return [a, b]


def _strip_golden(a, b):
    return np.matmul(a, b).astype(np.float32)


def _make_strip_kernels(mode):
    """AccValidShape strip kernels.

    mode="static": one full [M,N] matmul, then STRIP_COUNT pushes of
      AccStripTile views (physical [M,N], valid [H,N]) TASSIGNed at byte
      offset strip*H*64 into the same acc buffer (runTPushPop...StripNoSplit).

    mode="dynamic": per-strip [H,K]×[K,N] matmuls into one oversized acc tile
      (physical [M,N], valid [H,N]); each strip's result is pushed
      (runTPushPopDynamic...StripNoSplit).
    """
    in_name = "f32"
    cube_name = f"tpushpop_cvns_cube_strip_{mode}"
    vec_name = f"tpushpop_cvns_vec_strip_{mode}"

    @pto.jit(
        name=cube_name,
        kernel_kind="cube",
        target="a5",
        mode="auto",
        insert_sync=True,
        entry=False,
        backend="emitc",
    )
    def cube_kernel(
        a_ptr: pto.ptr(pto.f32, "gm"),
        b_ptr: pto.ptr(pto.f32, "gm"),
        out_ptr: pto.ptr(pto.f32, "gm"),
    ):
        c2v_buf = pto.import_reserved_buffer(
            name="c2v_fifo",
            peer_func=vec_name,
        )
        pipe = pto.pipe.c2v(slot_size=STRIP_SLOT, consumer_buf=c2v_buf, id=PIPE_ID, nosplit=True, slot_num=2)
        pipe.init_cube()

        a_view = pto.make_tensor_view(a_ptr, shape=[STRIP_M, STRIP_K], strides=[STRIP_K, 1])
        b_view = pto.make_tensor_view(b_ptr, shape=[STRIP_K, STRIP_N], strides=[STRIP_N, 1])

        if mode == "static":
            a_mat = pto.alloc_tile(shape=[STRIP_M, STRIP_K], dtype=pto.f32, memory_space="mat",
                                   blayout="ColMajor", slayout="RowMajor", valid_shape=[STRIP_M, STRIP_K])
            b_mat = pto.alloc_tile(shape=[STRIP_K, STRIP_N], dtype=pto.f32, memory_space="mat",
                                   blayout="ColMajor", slayout="RowMajor", valid_shape=[STRIP_K, STRIP_N])
            a_left = pto.alloc_tile(shape=[STRIP_M, STRIP_K], dtype=pto.f32, memory_space="left",
                                    blayout="ColMajor", slayout="RowMajor", valid_shape=[STRIP_M, STRIP_K])
            b_right = pto.alloc_tile(shape=[STRIP_K, STRIP_N], dtype=pto.f32, memory_space="right",
                                     blayout="RowMajor", slayout="ColMajor", valid_shape=[STRIP_K, STRIP_N])
            acc = pto.alloc_tile(shape=[STRIP_M, STRIP_N], dtype=pto.f32, memory_space="acc",
                                 blayout="ColMajor", slayout="RowMajor",
                                 valid_shape=[STRIP_M, STRIP_N], fractal_size=ACC_FRACTAL)

            pto.tile.load(a_view, a_mat)
            pto.tile.load(b_view, b_mat)
            pto.tile.mov(a_mat, a_left)
            pto.tile.mov(b_mat, b_right)
            pto.tile.matmul(a_left, b_right, acc)

            for strip in range(STRIP_COUNT):
                # AccStripTile: physical [M,N], valid [H,N], aliased at byte
                # offset strip*H*64 into the full acc buffer.
                strip_addr = pto.addptr(acc.as_ptr(), strip * STRIP_VIEW_ELEMS)
                strip_tile = pto.alloc_tile(
                    shape=[STRIP_M, STRIP_N], dtype=pto.f32, memory_space="acc",
                    blayout="ColMajor", slayout="RowMajor",
                    valid_shape=[STRIP_H, STRIP_N], fractal_size=ACC_FRACTAL,
                    addr=strip_addr,
                )
                pipe.push(strip_tile, split=SPLIT_NO_SPLIT)
        else:
            a_mat = pto.alloc_tile(shape=[STRIP_H, STRIP_K], dtype=pto.f32, memory_space="mat",
                                   blayout="ColMajor", slayout="RowMajor", valid_shape=[STRIP_H, STRIP_K])
            b_mat = pto.alloc_tile(shape=[STRIP_K, STRIP_N], dtype=pto.f32, memory_space="mat",
                                   blayout="ColMajor", slayout="RowMajor", valid_shape=[STRIP_K, STRIP_N])
            a_left = pto.alloc_tile(shape=[STRIP_H, STRIP_K], dtype=pto.f32, memory_space="left",
                                    blayout="ColMajor", slayout="RowMajor", valid_shape=[STRIP_H, STRIP_K])
            b_right = pto.alloc_tile(shape=[STRIP_K, STRIP_N], dtype=pto.f32, memory_space="right",
                                     blayout="RowMajor", slayout="ColMajor", valid_shape=[STRIP_K, STRIP_N])
            # Oversized acc: physical [M,N], runtime valid (H, N) —
            # TileAcc<OutT, M, N, -1, -1> + AccTile accTile(H, N): a static
            # ValidRow < Rows with multiple block columns trips the
            # MadAccStrideCompatible static_assert, so the valid region must
            # be dynamic (runtime ctor).
            dyn_h = pto.get_subblock_idx() * 0 + STRIP_H
            acc = pto.alloc_tile(shape=[STRIP_M, STRIP_N], dtype=pto.f32, memory_space="acc",
                                 blayout="ColMajor", slayout="RowMajor",
                                 valid_shape=[dyn_h, STRIP_N], fractal_size=ACC_FRACTAL)

            for strip in range(STRIP_COUNT):
                pto.tile.load(a_view, a_mat, offsets=[strip * STRIP_H, 0], sizes=[STRIP_H, STRIP_K])
                pto.tile.load(b_view, b_mat)
                pto.tile.mov(a_mat, a_left)
                pto.tile.mov(b_mat, b_right)
                pto.tile.matmul(a_left, b_right, acc)
                pipe.push(acc, split=SPLIT_NO_SPLIT)

    @pto.jit(
        name=vec_name,
        kernel_kind="vector",
        target="a5",
        mode="auto",
        insert_sync=True,
        entry=False,
        backend="emitc",
    )
    def vec_kernel(
        a_ptr: pto.ptr(pto.f32, "gm"),
        b_ptr: pto.ptr(pto.f32, "gm"),
        out_ptr: pto.ptr(pto.f32, "gm"),
    ):
        c2v_buf = pto.reserve_buffer(
            name="c2v_fifo", size=STRIP_SLOT * 4,
            location="vec", auto=True,
        )
        pipe = pto.pipe.c2v(slot_size=STRIP_SLOT, consumer_buf=c2v_buf, id=PIPE_ID, nosplit=True, slot_num=2)
        pipe.init_simd()

        for strip in range(STRIP_COUNT):
            entry = pipe.pop(split=SPLIT_NO_SPLIT, result_type=pto.alloc_tile(
                shape=[STRIP_H, STRIP_N], dtype=pto.f32, valid_shape=[STRIP_H, STRIP_N],
            ))
            pipe.free(entry, split=SPLIT_NO_SPLIT)
            out_view = pto.make_tensor_view(out_ptr, shape=[STRIP_M, STRIP_N], strides=[STRIP_N, 1])
            pto.tile.store(entry, out_view, offsets=[strip * STRIP_H, 0], sizes=[STRIP_H, STRIP_N])

    @pto.jit(
        name=f"tpushpop_cvns_entry_strip_{mode}",
        target="a5",
        mode="auto",
        insert_sync=True,
        backend="emitc",
    )
    def entry_kernel(
        a_ptr: pto.ptr(pto.f32, "gm"),
        b_ptr: pto.ptr(pto.f32, "gm"),
        out_ptr: pto.ptr(pto.f32, "gm"),
    ):
        cube_kernel(a_ptr, b_ptr, out_ptr)
        vec_kernel(a_ptr, b_ptr, out_ptr)

    return entry_kernel


CASES = []
KERNELS = []

for name, M, K, N, in_np, use_bias in _CASE_PARAMS:
    in_pto, in_name, _ = _DTYPE_MAP[in_np]
    entry = _make_kernels(M, K, N, in_pto, in_np, use_bias)
    KERNELS.append(entry)
    CASES.append(golden_output_case(
        name=name,
        kernel=entry,
        inputs=lambda p=(42, M, K, N, in_np, use_bias): _gen_inputs(*p),
        expected=lambda *inp, p=(use_bias,): _golden(*inp[:2], inp[2] if p[0] else None),
        output_shape=(M, N),
        output_dtype=np.float32,
        rtol=1e-3,
        atol=1e-3,
    ))

for mode, name in (
    ("static", "case4_float_static_acc_valid_shape_strip"),
    # case5 (dynamic oversized acc) is fully migrated below but cannot run in
    # this harness: its TMATMUL form (dynamic-valid Acc, m=H=16 vs Rows=32)
    # trips the runtime guard CheckAccStrideCompatible -> trap(), which
    # is fatal under the CA-model TRAP-fatal configs (msprof op simulator
    # AND a direct camodel run). The original case5 binary only passes under
    # the official run_st.py environment, which configures the simulator
    # differently. Verified NOT a PTOAS codegen issue: a minimal repro of
    # the same TMATMUL types traps the same way under our harness.
    # Re-enable when the harness matches the official runner's simulator
    # configuration.
    # ("dynamic", "case5_float_dynamic_acc_valid_shape_strip"),
):
    if mode == "dynamic":
        continue
    entry = _make_strip_kernels(mode)
    KERNELS.append(entry)
    CASES.append(golden_output_case(
        name=name,
        kernel=entry,
        inputs=lambda: _gen_strip_inputs(),
        expected=lambda a, b: _strip_golden(a, b),
        output_shape=(STRIP_M, STRIP_N),
        output_dtype=np.float32,
        rtol=1e-3,
        atol=1e-3,
    ))

auto_main(globals())
