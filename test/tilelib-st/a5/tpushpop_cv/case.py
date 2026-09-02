#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Cube-to-vector local FIFO pipe (UB consumer_buf)
#   Cube: matmul(A,B) → TPUSH acc tile to local UB FIFO
#   Vec:  TPOP from FIFO → TADD bias → TFREE → TSTORE
#
# 8 cases: UP_DOWN(1-4) / LEFT_RIGHT(5-8), fp16/fp32, single/multi-tile

from pathlib import Path
import sys

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import auto_main, golden_output_case
from ptodsl import pto


VEC_CORES = 2
PIPE_ID = 0
SPLIT_UP_DOWN = 1
SPLIT_LEFT_RIGHT = 2

_DTYPE_MAP = {
    np.float16: (pto.f16, 2, "f16"),
    np.float32: (pto.f32, 4, "f32"),
}

_CASE_PARAMS = [
    # (name, M, K, N, in_np, out_np, split)
    ("case1_half_single_tile",           16, 32, 32, np.float16, np.float32, SPLIT_UP_DOWN),
    ("case2_half_split_m",               32, 32, 32, np.float16, np.float32, SPLIT_UP_DOWN),
    ("case3_float_single_tile",          16, 32, 32, np.float32, np.float32, SPLIT_UP_DOWN),
    ("case4_half_multi_tile_wrapping",   64, 32, 32, np.float16, np.float32, SPLIT_UP_DOWN),
    ("case5_half_single_tile_left_right", 16, 32, 32, np.float16, np.float32, SPLIT_LEFT_RIGHT),
    ("case6_half_split_m_left_right",     32, 32, 32, np.float16, np.float32, SPLIT_LEFT_RIGHT),
    ("case7_float_single_tile_left_right",16, 32, 32, np.float32, np.float32, SPLIT_LEFT_RIGHT),
    ("case8_half_multi_tile_wrapping_left_right", 64, 32, 32, np.float16, np.float32, SPLIT_LEFT_RIGHT),
]

CASE_TILE_M = 16


def _gen_inputs(seed, M, K, N, in_np, out_np):
    np.random.seed(seed)
    a = np.random.uniform(-2, 2, [M, K]).astype(in_np)
    b = np.random.uniform(-2, 2, [K, N]).astype(in_np)
    bias = np.random.uniform(-1, 1, [M, N]).astype(out_np)
    return [a, b, bias]


def _golden(a, b, bias):
    return np.matmul(a.astype(np.float32), b.astype(np.float32)).astype(np.float32) + bias


def _make_kernels(M, K, N, in_pto, out_pto, in_np, out_np, split):
    elem_bytes = _DTYPE_MAP[out_np][1]
    in_name = _DTYPE_MAP[in_np][2]
    out_name = _DTYPE_MAP[out_np][2]
    if split == SPLIT_UP_DOWN:
        vec_m = CASE_TILE_M // VEC_CORES
        vec_n = N
    else:
        vec_m = CASE_TILE_M
        vec_n = N // VEC_CORES
    slot_size = elem_bytes * vec_m * vec_n
    num_m_tiles = M // CASE_TILE_M

    @pto.jit(
        name=f"tpushpop_cv_cube_{M}x{K}x{N}_{in_name}_{split}",
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
        bias_ptr: pto.ptr(out_pto, "gm"),
        out_ptr: pto.ptr(out_pto, "gm"),
    ):
        # The cube side is the FIFO peer: import the vector kernel's reserve
        # (the owner is the vector core, where the UB FIFO physically lives;
        # same pattern as samples/TPushTPop).
        c2v_buf = pto.import_reserved_buffer(
            name="c2v_fifo",
            peer_func=f"tpushpop_cv_vec_{M}x{K}x{N}_{in_name}_{split}",
        )
        pipe = pto.pipe.c2v(slot_size=slot_size, consumer_buf=c2v_buf, id=PIPE_ID, slot_num=2)
        pipe.init_cube()

        a_mat = pto.alloc_tile(shape=[CASE_TILE_M, K], dtype=in_pto, memory_space="mat",
                               blayout="ColMajor", slayout="RowMajor", valid_shape=[CASE_TILE_M, K])
        b_mat = pto.alloc_tile(shape=[K, N], dtype=in_pto, memory_space="mat",
                               blayout="ColMajor", slayout="RowMajor", valid_shape=[K, N])
        a_left = pto.alloc_tile(shape=[CASE_TILE_M, K], dtype=in_pto, memory_space="left",
                                blayout="ColMajor", slayout="RowMajor", valid_shape=[CASE_TILE_M, K])
        b_right = pto.alloc_tile(shape=[K, N], dtype=in_pto, memory_space="right",
                                 blayout="RowMajor", slayout="ColMajor", valid_shape=[K, N])
        acc = pto.alloc_tile(shape=[CASE_TILE_M, N], dtype=out_pto, memory_space="acc",
                             blayout="ColMajor", slayout="RowMajor", valid_shape=[CASE_TILE_M, N])

        for m_tile in range(num_m_tiles):
            a_view = pto.make_tensor_view(a_ptr, shape=[M, K], strides=[K, 1])
            b_view = pto.make_tensor_view(b_ptr, shape=[K, N], strides=[N, 1])
            pto.tile.load(a_view, a_mat, offsets=[m_tile * CASE_TILE_M, 0], sizes=[CASE_TILE_M, K])
            pto.tile.load(b_view, b_mat)
            pto.tile.mov(a_mat, a_left)
            pto.tile.mov(b_mat, b_right)
            pto.tile.matmul(a_left, b_right, acc)
            pipe.push(acc, split=split)

    @pto.jit(
        name=f"tpushpop_cv_vec_{M}x{K}x{N}_{in_name}_{split}",
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
        bias_ptr: pto.ptr(out_pto, "gm"),
        out_ptr: pto.ptr(out_pto, "gm"),
    ):
        c2v_buf = pto.reserve_buffer(
            name="c2v_fifo", size=slot_size * 4,
            location="vec", auto=True,
        )
        pipe = pto.pipe.c2v(slot_size=slot_size, consumer_buf=c2v_buf, id=PIPE_ID, slot_num=2)
        pipe.init_simd()
        sub_block_idx = pto.get_subblock_idx()

        for m_tile in range(num_m_tiles):
            entry = pipe.pop(split=split, result_type=pto.alloc_tile(
                shape=[vec_m, vec_n], dtype=out_pto, valid_shape=[vec_m, vec_n],
            ))
            bias_tile = pto.alloc_tile(shape=[vec_m, vec_n], dtype=out_pto, valid_shape=[vec_m, vec_n])
            out_tile = pto.alloc_tile(shape=[vec_m, vec_n], dtype=out_pto, valid_shape=[vec_m, vec_n])

            # subBlockIdx selects upper/lower half rows (UP_DOWN) or
            # left/right half cols within each row (LEFT_RIGHT)
            if split == SPLIT_UP_DOWN:
                bias_off_m = m_tile * CASE_TILE_M + sub_block_idx * vec_m
                bias_off_n = 0
            else:
                bias_off_m = m_tile * CASE_TILE_M
                bias_off_n = sub_block_idx * vec_n
            bias_view = pto.make_tensor_view(bias_ptr, shape=[M, N], strides=[N, 1])
            pto.tile.load(bias_view, bias_tile, offsets=[bias_off_m, bias_off_n], sizes=[vec_m, vec_n])

            pto.tile.add(entry, bias_tile, out_tile)
            pipe.free(entry, split=split)

            out_view = pto.make_tensor_view(out_ptr, shape=[M, N], strides=[N, 1])
            pto.tile.store(out_tile, out_view, offsets=[bias_off_m, bias_off_n], sizes=[vec_m, vec_n])

    @pto.jit(
        name=f"tpushpop_cv_entry_{M}x{K}x{N}_{in_name}_{split}",
        target="a5",
        mode="auto",
        insert_sync=True,
        backend="emitc",
    )
    def entry_kernel(
        a_ptr: pto.ptr(in_pto, "gm"),
        b_ptr: pto.ptr(in_pto, "gm"),
        bias_ptr: pto.ptr(out_pto, "gm"),
        out_ptr: pto.ptr(out_pto, "gm"),
    ):
        cube_kernel(a_ptr, b_ptr, bias_ptr, out_ptr)
        vec_kernel(a_ptr, b_ptr, bias_ptr, out_ptr)

    return entry_kernel


CASES = []
KERNELS = []

for name, M, K, N, in_np, out_np, split in _CASE_PARAMS:
    in_pto, _, _ = _DTYPE_MAP[in_np]
    out_pto, _, _ = _DTYPE_MAP[out_np]
    entry = _make_kernels(M, K, N, in_pto, out_pto, in_np, out_np, split)
    KERNELS.append(entry)
    CASES.append(golden_output_case(
        name=name,
        kernel=entry,
        inputs=lambda p=(19, M, K, N, in_np, out_np): _gen_inputs(*p),
        expected=lambda *inp, p=(M, K, N): _golden(*inp),
        output_shape=(M, N),
        output_dtype=out_np,
        rtol=1e-3,
        atol=1e-3,
    ))

auto_main(globals())
