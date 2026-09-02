#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Bidirectional local FIFO (C2V UB + V2C L1)
#   Flow: C = A + B; E = matmul(C, D); golden = E - F
#   Vec Phase 1 (V2C): TADD(A,B) → TMOV NZ → TPUSH to L1
#   Cube: TPOP from L1 → TLOAD D → TMATMUL → TPUSH to UB
#   Vec Phase 2 (C2V): TPOP from UB → TSUB(E,F) → TFREE → TSTORE
#
# 2 cases: UP_DOWN(1) / LEFT_RIGHT(2), 128x64x128 f32

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
M = 128
K = 64
N = 128

_CASE_PARAMS = [
    ("case1_float_dir_both",            SPLIT_UP_DOWN),
    ("case2_float_dir_both_left_right", SPLIT_LEFT_RIGHT),
]


def _gen_inputs(seed):
    np.random.seed(seed)
    a = np.random.uniform(-2, 2, [M, K]).astype(np.float32)
    b = np.random.uniform(-2, 2, [M, K]).astype(np.float32)
    d = np.random.uniform(-2, 2, [K, N]).astype(np.float32)
    f = np.random.uniform(-2, 2, [M, N]).astype(np.float32)
    return [a, b, d, f]


def _golden(a, b, d, f):
    c = (a + b).astype(np.float32)
    e = np.matmul(c, d).astype(np.float32)
    return (e - f).astype(np.float32)


def _make_kernels(split):
    if split == SPLIT_UP_DOWN:
        vec_m = M // VEC_CORES
        vec_k = K
        vec_n = N
    else:
        vec_m = M
        vec_k = K // VEC_CORES
        vec_n = N // VEC_CORES
    slot_size_v2c = 4 * M * K
    slot_size_c2v = 4 * M * N
    # The bidirectional pipe shares one unified slot_size (M*K*4) across
    # entries in BOTH directions; FIFO_DEPTH = 2.
    # UP_DOWN dual-split writes only [vec_m, N] per vec core per slot.
    FIFO_DEPTH = 2
    fifo_extent = slot_size_v2c * FIFO_DEPTH

    @pto.jit(
        name=f"tpushpop_dirboth_vec_{split}",
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
        d_ptr: pto.ptr(pto.f32, "gm"),
        f_ptr: pto.ptr(pto.f32, "gm"),
        out_ptr: pto.ptr(pto.f32, "gm"),
    ):
        c2v_buf = pto.reserve_buffer(name="c2v_fifo", size=fifo_extent, location="vec", auto=True)
        v2c_buf = pto.import_reserved_buffer(name="v2c_fifo", peer_func=f"tpushpop_dirboth_cube_{split}")
        pipe = pto.pipe.bidirectional(
            slot_size=slot_size_v2c,
            c2v_consumer_buf=c2v_buf,
            v2c_consumer_buf=v2c_buf,
            id=PIPE_ID,
            slot_num=FIFO_DEPTH,
        )
        pipe.init_simd()
        c2v = pipe.c2v
        v2c = pipe.v2c
        sub_block_idx = pto.get_subblock_idx()
        # subBlockIdx selects row half (UP_DOWN) or col half (LEFT_RIGHT)
        # for both the A/B loads and the F/out accesses
        if split == SPLIT_UP_DOWN:
            off_m = sub_block_idx * vec_m
            off_n = 0
        else:
            off_m = 0
            off_n = sub_block_idx * vec_k

        # Phase 1: V2C — TADD(A,B) → TMOV NZ → TPUSH
        tile_a = pto.alloc_tile(shape=[vec_m, vec_k], dtype=pto.f32, valid_shape=[vec_m, vec_k])
        tile_b = pto.alloc_tile(shape=[vec_m, vec_k], dtype=pto.f32, valid_shape=[vec_m, vec_k])
        tile_c = pto.alloc_tile(shape=[vec_m, vec_k], dtype=pto.f32, valid_shape=[vec_m, vec_k])
        tile_c_nz = pto.alloc_tile(shape=[vec_m, vec_k], dtype=pto.f32,
                                   blayout="ColMajor", slayout="RowMajor", valid_shape=[vec_m, vec_k])

        a_view = pto.make_tensor_view(a_ptr, shape=[M, K], strides=[K, 1])
        b_view = pto.make_tensor_view(b_ptr, shape=[M, K], strides=[K, 1])
        pto.tile.load(a_view, tile_a, offsets=[off_m, off_n], sizes=[vec_m, vec_k])
        pto.tile.load(b_view, tile_b, offsets=[off_m, off_n], sizes=[vec_m, vec_k])
        pto.tile.add(tile_a, tile_b, tile_c)
        pto.tile.mov(tile_c, tile_c_nz)
        v2c.push(tile_c_nz, split=split)

        # Phase 2: C2V — TPOP → TSUB(E,F) → TFREE → TSTORE
        tile_e = pto.alloc_tile(shape=[vec_m, vec_n], dtype=pto.f32, valid_shape=[vec_m, vec_n])
        tile_f = pto.alloc_tile(shape=[vec_m, vec_n], dtype=pto.f32, valid_shape=[vec_m, vec_n])
        tile_g = pto.alloc_tile(shape=[vec_m, vec_n], dtype=pto.f32, valid_shape=[vec_m, vec_n])

        entry = c2v.pop(split=split, result_type=tile_e)
        f_view = pto.make_tensor_view(f_ptr, shape=[M, N], strides=[N, 1])
        if split == SPLIT_UP_DOWN:
            f_out_off = [sub_block_idx * vec_m, 0]
        else:
            f_out_off = [0, sub_block_idx * vec_n]
        pto.tile.load(f_view, tile_f, offsets=f_out_off, sizes=[vec_m, vec_n])
        pto.tile.sub(entry, tile_f, tile_g)
        c2v.free(entry, split=split)
        out_view = pto.make_tensor_view(out_ptr, shape=[M, N], strides=[N, 1])
        pto.tile.store(tile_g, out_view, offsets=f_out_off, sizes=[vec_m, vec_n])

    @pto.jit(
        name=f"tpushpop_dirboth_cube_{split}",
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
        d_ptr: pto.ptr(pto.f32, "gm"),
        f_ptr: pto.ptr(pto.f32, "gm"),
        out_ptr: pto.ptr(pto.f32, "gm"),
    ):
        c2v_buf = pto.import_reserved_buffer(name="c2v_fifo", peer_func=f"tpushpop_dirboth_vec_{split}")
        v2c_buf = pto.reserve_buffer(name="v2c_fifo", size=fifo_extent, location="mat", auto=True)
        pipe = pto.pipe.bidirectional(
            slot_size=slot_size_v2c,
            c2v_consumer_buf=c2v_buf,
            v2c_consumer_buf=v2c_buf,
            id=PIPE_ID,
            slot_num=FIFO_DEPTH,
        )
        pipe.init_cube()
        c2v = pipe.c2v
        v2c = pipe.v2c

        pop_tile = pto.alloc_tile(shape=[M, K], dtype=pto.f32, memory_space="mat",
                                   blayout="ColMajor", slayout="RowMajor", valid_shape=[M, K])
        d_mat = pto.alloc_tile(shape=[K, N], dtype=pto.f32, memory_space="mat",
                               blayout="ColMajor", slayout="RowMajor", valid_shape=[K, N])
        a_left = pto.alloc_tile(shape=[M, K], dtype=pto.f32, memory_space="left",
                                blayout="ColMajor", slayout="RowMajor", valid_shape=[M, K])
        b_right = pto.alloc_tile(shape=[K, N], dtype=pto.f32, memory_space="right",
                                 blayout="RowMajor", slayout="ColMajor", valid_shape=[K, N])
        acc = pto.alloc_tile(shape=[M, N], dtype=pto.f32, memory_space="acc",
                             blayout="ColMajor", slayout="RowMajor", valid_shape=[M, N])

        entry = v2c.pop(split=split, result_type=pop_tile)
        d_view = pto.make_tensor_view(d_ptr, shape=[K, N], strides=[N, 1])
        pto.tile.load(d_view, d_mat)
        pto.tile.mov(entry, a_left)
        pto.tile.mov(d_mat, b_right)
        v2c.free(entry, split=split)
        pto.tile.matmul(a_left, b_right, acc)
        c2v.push(acc, split=split)

    @pto.jit(
        name=f"tpushpop_dirboth_entry_{split}",
        target="a5",
        mode="auto",
        insert_sync=True,
        backend="emitc",
    )
    def entry_kernel(
        a_ptr: pto.ptr(pto.f32, "gm"),
        b_ptr: pto.ptr(pto.f32, "gm"),
        d_ptr: pto.ptr(pto.f32, "gm"),
        f_ptr: pto.ptr(pto.f32, "gm"),
        out_ptr: pto.ptr(pto.f32, "gm"),
    ):
        vec_kernel(a_ptr, b_ptr, d_ptr, f_ptr, out_ptr)
        cube_kernel(a_ptr, b_ptr, d_ptr, f_ptr, out_ptr)

    return entry_kernel


CASES = []
KERNELS = []

for name, split in _CASE_PARAMS:
    entry = _make_kernels(split)
    KERNELS.append(entry)
    CASES.append(golden_output_case(
        name=name,
        kernel=entry,
        inputs=lambda: _gen_inputs(19),
        expected=lambda *inp: _golden(*inp),
        output_shape=(M, N),
        output_dtype=np.float32,
        rtol=1e-3,
        atol=1e-3,
    ))

auto_main(globals())
