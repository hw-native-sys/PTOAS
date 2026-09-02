#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Vector-to-cube local FIFO pipe (L1 consumer_buf)
#   Vec:  TDEQUANT(quantB, scale, offset) → TMOV NZ → TPUSH to L1 FIFO
#   Cube: TPOP from L1 → K-tiling: TMATMUL/TMATMUL_ACC → TSTORE
#
# 13 cases: UP_DOWN(1-6,13) / LEFT_RIGHT(7-12), int8/int16 K-tiling, float subblockid

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
TILE_K = 64
M = 16

_QUANT_MAP = {
    np.int8:  (pto.i8,  "i8"),
    np.int16: (pto.i16, "i16"),
}

# (name, K, N, quant_np, split)
_CASE_PARAMS = [
    ("case1_int8_single_k_tile",           64,  32, np.int8,  SPLIT_UP_DOWN),
    ("case2_int8_two_k_tiles",             128, 32, np.int8,  SPLIT_UP_DOWN),
    ("case3_int8_four_k_tiles",            256, 32, np.int8,  SPLIT_UP_DOWN),
    ("case4_int16_single_k_tile",          64,  32, np.int16, SPLIT_UP_DOWN),
    ("case5_int16_two_k_tiles",            128, 32, np.int16, SPLIT_UP_DOWN),
    ("case6_int16_four_k_tiles",           256, 32, np.int16, SPLIT_UP_DOWN),
    ("case7_int8_single_k_tile_left_right", 64, 64, np.int8,  SPLIT_LEFT_RIGHT),
    ("case8_int8_two_k_tiles_left_right",  128, 64, np.int8,  SPLIT_LEFT_RIGHT),
    ("case9_int8_four_k_tiles_left_right", 256, 64, np.int8,  SPLIT_LEFT_RIGHT),
    ("case10_int16_single_k_tile_left_right", 64, 32, np.int16, SPLIT_LEFT_RIGHT),
    ("case11_int16_two_k_tiles_left_right", 128, 32, np.int16, SPLIT_LEFT_RIGHT),
    ("case12_int16_four_k_tiles_left_right", 256, 32, np.int16, SPLIT_LEFT_RIGHT),
]

# case13: subblockid, float, M=128, K=64, N=64
_SUBBLOCK_PARAMS = ("case13_float_subblock_id_up_down", 128, 64, 64, np.float32)


def _gen_inputs(seed, M, K, N, quant_np):
    np.random.seed(seed)
    a = np.random.uniform(-2, 2, [M, K]).astype(np.float32)
    scale = np.random.uniform(0.01, 0.1, [K]).astype(np.float32)
    offset = np.random.uniform(-1, 1, [K]).astype(np.float32)
    b_float = np.random.uniform(-2, 2, [K, N]).astype(np.float32)
    if quant_np == np.int8:
        b_quant = np.clip(np.round(b_float / scale[:, None] + offset[:, None]), -128, 127).astype(np.int8)
    else:
        b_quant = np.clip(np.round(b_float / scale[:, None] + offset[:, None]), -32768, 32767).astype(np.int16)
    return [a, b_quant, scale, offset]


def _golden(a, b_quant, scale, offset):
    b_dequant = (b_quant.astype(np.float32) - offset[:, None]) * scale[:, None]
    return np.matmul(a.astype(np.float32), b_dequant).astype(np.float32)


def _gen_subblock_inputs(seed, M, K, N):
    np.random.seed(seed)
    src0 = np.random.uniform(-2, 2, [M, K]).astype(np.float32)
    src1 = np.random.uniform(-2, 2, [M, K]).astype(np.float32)
    src2 = np.random.uniform(-2, 2, [K, N]).astype(np.float32)
    return [src0, src1, src2]


def _golden_subblock(src0, src1, src2):
    # Explicit swapped sub-block ID: each vec core pushes into its peer's row
    # window, so the assembled [M, K] result has its row halves exchanged
    # (TPUSH(..., 1 - subBlockIdx)).
    res = (src0 + src1).astype(np.float32)
    out = np.matmul(res, src2).astype(np.float32)
    half = out.shape[0] // 2
    return np.concatenate([out[half:], out[:half]], axis=0)


def _make_kernels(K, N, quant_pto, quant_np, split):
    quant_name = _QUANT_MAP[quant_np][1]
    num_k_tiles = K // TILE_K
    if split == SPLIT_UP_DOWN:
        prod_k = TILE_K // VEC_CORES
        prod_n = N
    else:
        prod_k = TILE_K
        prod_n = N // VEC_CORES
    slot_size = 4 * TILE_K * N  # f32

    @pto.jit(
        name=f"tpushpop_vc_vec_{K}x{N}_{quant_name}_{split}",
        kernel_kind="vector",
        target="a5",
        mode="auto",
        insert_sync=True,
        entry=False,
        backend="emitc",
    )
    def vec_kernel(
        a_ptr: pto.ptr(pto.f32, "gm"),
        quant_b_ptr: pto.ptr(quant_pto, "gm"),
        scale_ptr: pto.ptr(pto.f32, "gm"),
        offset_ptr: pto.ptr(pto.f32, "gm"),
        out_ptr: pto.ptr(pto.f32, "gm"),
    ):
        v2c_buf = pto.import_reserved_buffer(
            name="v2c_fifo",
            peer_func=f"tpushpop_vc_cube_{K}x{N}_{quant_name}_{split}",
        )
        pipe = pto.pipe.v2c(slot_size=slot_size, consumer_buf=v2c_buf, id=PIPE_ID, slot_num=2)
        pipe.init_simd()
        sub_block_idx = pto.get_subblock_idx()

        quant_tile = pto.alloc_tile(shape=[prod_k, prod_n], dtype=quant_pto, valid_shape=[prod_k, prod_n])
        dequant_tile = pto.alloc_tile(shape=[prod_k, prod_n], dtype=pto.f32, valid_shape=[prod_k, prod_n])
        dequant_nz = pto.alloc_tile(shape=[prod_k, prod_n], dtype=pto.f32,
                                    blayout="ColMajor", slayout="RowMajor", valid_shape=[prod_k, prod_n])
        scale_tile = pto.alloc_tile(shape=[prod_k, 8], dtype=pto.f32, valid_shape=[prod_k, 1])
        offset_tile = pto.alloc_tile(shape=[prod_k, 8], dtype=pto.f32, valid_shape=[prod_k, 1])

        for k_tile in range(num_k_tiles):
            # subBlockIdx selects K-half rows (UP_DOWN) or N-half cols
            # within each row (LEFT_RIGHT)
            if split == SPLIT_UP_DOWN:
                qb_off_m = k_tile * TILE_K + sub_block_idx * prod_k
                qb_off_n = 0
            else:
                qb_off_m = k_tile * TILE_K
                qb_off_n = sub_block_idx * prod_n
            quant_view = pto.make_tensor_view(quant_b_ptr, shape=[K, N], strides=[N, 1])
            pto.tile.load(quant_view, quant_tile, offsets=[qb_off_m, qb_off_n], sizes=[prod_k, prod_n])

            scale_view = pto.make_tensor_view(scale_ptr, shape=[K, 1], strides=[1, 1])
            pto.tile.load(scale_view, scale_tile, offsets=[qb_off_m, 0], sizes=[prod_k, 1])

            offset_view = pto.make_tensor_view(offset_ptr, shape=[K, 1], strides=[1, 1])
            pto.tile.load(offset_view, offset_tile, offsets=[qb_off_m, 0], sizes=[prod_k, 1])

            pto.tile.dequant(quant_tile, scale_tile, offset_tile, dequant_tile)
            pto.tile.mov(dequant_tile, dequant_nz)
            pipe.push(dequant_nz, split=split)

    @pto.jit(
        name=f"tpushpop_vc_cube_{K}x{N}_{quant_name}_{split}",
        kernel_kind="cube",
        target="a5",
        mode="auto",
        insert_sync=True,
        entry=False,
        backend="emitc",
    )
    def cube_kernel(
        a_ptr: pto.ptr(pto.f32, "gm"),
        quant_b_ptr: pto.ptr(quant_pto, "gm"),
        scale_ptr: pto.ptr(pto.f32, "gm"),
        offset_ptr: pto.ptr(pto.f32, "gm"),
        out_ptr: pto.ptr(pto.f32, "gm"),
    ):
        v2c_buf = pto.reserve_buffer(
            name="v2c_fifo", size=slot_size * 4,
            location="mat", auto=True,
        )
        pipe = pto.pipe.v2c(slot_size=slot_size, consumer_buf=v2c_buf, id=PIPE_ID, slot_num=2)
        pipe.init_cube()

        a_mat = pto.alloc_tile(shape=[M, TILE_K], dtype=pto.f32, memory_space="mat",
                               blayout="ColMajor", slayout="RowMajor", valid_shape=[M, TILE_K])
        b_mat = pto.alloc_tile(shape=[TILE_K, N], dtype=pto.f32, memory_space="mat",
                               blayout="ColMajor", slayout="RowMajor", valid_shape=[TILE_K, N])
        a_left = pto.alloc_tile(shape=[M, TILE_K], dtype=pto.f32, memory_space="left",
                                blayout="ColMajor", slayout="RowMajor", valid_shape=[M, TILE_K])
        b_right = pto.alloc_tile(shape=[TILE_K, N], dtype=pto.f32, memory_space="right",
                                 blayout="RowMajor", slayout="ColMajor", valid_shape=[TILE_K, N])
        acc = pto.alloc_tile(shape=[M, N], dtype=pto.f32, memory_space="acc",
                             blayout="ColMajor", slayout="RowMajor", valid_shape=[M, N])

        for k_tile in range(num_k_tiles):
            a_view = pto.make_tensor_view(a_ptr, shape=[M, K], strides=[K, 1])
            pto.tile.load(a_view, a_mat, offsets=[0, k_tile * TILE_K], sizes=[M, TILE_K])

            pop_tile = pto.alloc_tile(shape=[TILE_K, N], dtype=pto.f32, memory_space="mat",
                                      blayout="ColMajor", slayout="RowMajor", valid_shape=[TILE_K, N])
            entry = pipe.pop(split=split, result_type=pop_tile)

            pto.tile.mov(a_mat, a_left)
            pto.tile.mov(entry, b_right)
            pipe.free(entry, split=split)

            if k_tile == 0:
                pto.tile.matmul(a_left, b_right, acc)
            else:
                pto.tile.matmul_acc(acc, a_left, b_right, acc)

        out_view = pto.make_tensor_view(out_ptr, shape=[M, N], strides=[N, 1])
        pto.tile.store(acc, out_view)

    @pto.jit(
        name=f"tpushpop_vc_entry_{K}x{N}_{quant_name}_{split}",
        target="a5",
        mode="auto",
        insert_sync=True,
        backend="emitc",
    )
    def entry_kernel(
        a_ptr: pto.ptr(pto.f32, "gm"),
        quant_b_ptr: pto.ptr(quant_pto, "gm"),
        scale_ptr: pto.ptr(pto.f32, "gm"),
        offset_ptr: pto.ptr(pto.f32, "gm"),
        out_ptr: pto.ptr(pto.f32, "gm"),
    ):
        cube_kernel(a_ptr, quant_b_ptr, scale_ptr, offset_ptr, out_ptr)
        vec_kernel(a_ptr, quant_b_ptr, scale_ptr, offset_ptr, out_ptr)

    return entry_kernel


def _make_subblock_kernels(M, K, N):
    vec_rows = M // VEC_CORES
    slot_size = 4 * M * K

    @pto.jit(
        name=f"tpushpop_vc_vec_subblock_{M}x{K}x{N}",
        kernel_kind="vector",
        target="a5",
        mode="auto",
        insert_sync=True,
        entry=False,
        backend="emitc",
    )
    def vec_kernel(
        src0_ptr: pto.ptr(pto.f32, "gm"),
        src1_ptr: pto.ptr(pto.f32, "gm"),
        src2_ptr: pto.ptr(pto.f32, "gm"),
        out_ptr: pto.ptr(pto.f32, "gm"),
    ):
        v2c_buf = pto.import_reserved_buffer(
            name="v2c_fifo",
            peer_func=f"tpushpop_vc_cube_subblock_{M}x{K}x{N}",
        )
        pipe = pto.pipe.v2c(slot_size=slot_size, consumer_buf=v2c_buf, id=PIPE_ID, slot_num=2)
        pipe.init_simd()
        sub_block_idx = pto.get_subblock_idx()

        src0_tile = pto.alloc_tile(shape=[vec_rows, K], dtype=pto.f32, valid_shape=[vec_rows, K])
        src1_tile = pto.alloc_tile(shape=[vec_rows, K], dtype=pto.f32, valid_shape=[vec_rows, K])
        res_tile = pto.alloc_tile(shape=[vec_rows, K], dtype=pto.f32, valid_shape=[vec_rows, K])
        res_nz = pto.alloc_tile(shape=[vec_rows, K], dtype=pto.f32,
                                blayout="ColMajor", slayout="RowMajor", valid_shape=[vec_rows, K])

        src0_view = pto.make_tensor_view(src0_ptr, shape=[M, K], strides=[K, 1])
        src1_view = pto.make_tensor_view(src1_ptr, shape=[M, K], strides=[K, 1])
        off = sub_block_idx * vec_rows
        pto.tile.load(src0_view, src0_tile, offsets=[off, 0], sizes=[vec_rows, K])
        pto.tile.load(src1_view, src1_tile, offsets=[off, 0], sizes=[vec_rows, K])
        pto.tile.add(src0_tile, src1_tile, res_tile)
        pto.tile.mov(res_tile, res_nz)
        pipe.push(res_nz, split=SPLIT_UP_DOWN, sub_block_id=1 - sub_block_idx)

    @pto.jit(
        name=f"tpushpop_vc_cube_subblock_{M}x{K}x{N}",
        kernel_kind="cube",
        target="a5",
        mode="auto",
        insert_sync=True,
        entry=False,
        backend="emitc",
    )
    def cube_kernel(
        src0_ptr: pto.ptr(pto.f32, "gm"),
        src1_ptr: pto.ptr(pto.f32, "gm"),
        src2_ptr: pto.ptr(pto.f32, "gm"),
        out_ptr: pto.ptr(pto.f32, "gm"),
    ):
        v2c_buf = pto.reserve_buffer(
            name="v2c_fifo", size=slot_size * 4,
            location="mat", auto=True,
        )
        pipe = pto.pipe.v2c(slot_size=slot_size, consumer_buf=v2c_buf, id=PIPE_ID, slot_num=2)
        pipe.init_cube()

        pop_tile = pto.alloc_tile(shape=[M, K], dtype=pto.f32, memory_space="mat",
                                  blayout="ColMajor", slayout="RowMajor", valid_shape=[M, K])
        b_mat = pto.alloc_tile(shape=[K, N], dtype=pto.f32, memory_space="mat",
                               blayout="ColMajor", slayout="RowMajor", valid_shape=[K, N])
        a_left = pto.alloc_tile(shape=[M, K], dtype=pto.f32, memory_space="left",
                                blayout="ColMajor", slayout="RowMajor", valid_shape=[M, K])
        b_right = pto.alloc_tile(shape=[K, N], dtype=pto.f32, memory_space="right",
                                 blayout="RowMajor", slayout="ColMajor", valid_shape=[K, N])
        acc = pto.alloc_tile(shape=[M, N], dtype=pto.f32, memory_space="acc",
                             blayout="ColMajor", slayout="RowMajor", valid_shape=[M, N])

        entry = pipe.pop(split=SPLIT_UP_DOWN, result_type=pop_tile)
        src2_view = pto.make_tensor_view(src2_ptr, shape=[K, N], strides=[N, 1])
        pto.tile.load(src2_view, b_mat)

        pto.tile.mov(entry, a_left)
        pto.tile.mov(b_mat, b_right)
        pipe.free(entry, split=SPLIT_UP_DOWN)

        pto.tile.matmul(a_left, b_right, acc)
        out_view = pto.make_tensor_view(out_ptr, shape=[M, N], strides=[N, 1])
        pto.tile.store(acc, out_view)

    @pto.jit(
        name=f"tpushpop_vc_entry_subblock_{M}x{K}x{N}",
        target="a5",
        mode="auto",
        insert_sync=True,
        backend="emitc",
    )
    def entry_kernel(
        src0_ptr: pto.ptr(pto.f32, "gm"),
        src1_ptr: pto.ptr(pto.f32, "gm"),
        src2_ptr: pto.ptr(pto.f32, "gm"),
        out_ptr: pto.ptr(pto.f32, "gm"),
    ):
        vec_kernel(src0_ptr, src1_ptr, src2_ptr, out_ptr)
        cube_kernel(src0_ptr, src1_ptr, src2_ptr, out_ptr)

    return entry_kernel


CASES = []
KERNELS = []

for name, K, N, quant_np, split in _CASE_PARAMS:
    quant_pto, quant_name = _QUANT_MAP[quant_np]
    entry = _make_kernels(K, N, quant_pto, quant_np, split)
    KERNELS.append(entry)
    CASES.append(golden_output_case(
        name=name,
        kernel=entry,
        inputs=lambda p=(19, M, K, N, quant_np): _gen_inputs(*p),
        expected=lambda *inp: _golden(*inp),
        output_shape=(M, N),
        output_dtype=np.float32,
        rtol=1e-3,
        atol=1e-3,
    ))

# case13: subblockid
sub_name, sub_M, sub_K, sub_N, sub_np = _SUBBLOCK_PARAMS
sub_entry = _make_subblock_kernels(sub_M, sub_K, sub_N)
KERNELS.append(sub_entry)
CASES.append(golden_output_case(
    name=sub_name,
    kernel=sub_entry,
    inputs=lambda: _gen_subblock_inputs(19, sub_M, sub_K, sub_N),
    expected=lambda *inp: _golden_subblock(*inp),
    output_shape=(sub_M, sub_N),
    output_dtype=sub_np,
    rtol=1e-3,
    atol=1e-3,
))

auto_main(globals())
