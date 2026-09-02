#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# V2C local FIFO, TILE_NO_SPLIT (AIV0-only)
#   Vec (AIV0 only): TDEQUANT → TMOV NZ → TPUSH NO_SPLIT to L1
#   Cube: TPOP NO_SPLIT → K-tiling: TMATMUL/TMATMUL_ACC → TSTORE
#
# 6 cases: int8/int16, K=64/128/256 (1/2/4 K-tiles)

from pathlib import Path
import sys

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import auto_main, golden_output_case
from ptodsl import pto


PIPE_ID = 0
SPLIT_NO_SPLIT = 0
TILE_K = 64
M = 16

_QUANT_MAP = {
    np.int8:  (pto.i8,  "i8"),
    np.int16: (pto.i16, "i16"),
}

_CASE_PARAMS = [
    ("case1_int8_single_k_tile",   64,  32, np.int8),
    ("case2_int8_two_k_tiles",     128, 32, np.int8),
    ("case3_int8_four_k_tiles",    256, 32, np.int8),
    ("case4_int16_single_k_tile",  64,  32, np.int16),
    ("case5_int16_two_k_tiles",    128, 32, np.int16),
    ("case6_int16_four_k_tiles",   256, 32, np.int16),
]


def _gen_inputs(seed, K, N, quant_np):
    np.random.seed(seed)
    a = np.random.uniform(-1.0, 1.0, [M, K]).astype(np.float32)
    if quant_np == np.int8:
        b_quant = np.random.randint(-64, 64, [K, N]).astype(np.int8)
    else:
        b_quant = np.random.randint(-512, 512, [K, N]).astype(np.int16)
    scale = np.random.uniform(0.01, 0.1, [K]).astype(np.float32)
    offset = np.random.uniform(-1.0, 1.0, [K]).astype(np.float32)
    return [a, b_quant, scale, offset]


def _golden(a, b_quant, scale, offset):
    b_dequant = (b_quant.astype(np.float64) - offset[:, None].astype(np.float64)) * scale[:, None].astype(np.float64)
    return np.matmul(a.astype(np.float64), b_dequant).astype(np.float32)


def _make_kernels(K, N, quant_pto, quant_np):
    quant_name = _QUANT_MAP[quant_np][1]
    num_k_tiles = K // TILE_K
    slot_size = 4 * TILE_K * N

    @pto.jit(
        name=f"tpushpop_vcns_vec_{K}x{N}_{quant_name}",
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
            peer_func=f"tpushpop_vcns_cube_{K}x{N}_{quant_name}",
        )
        pipe = pto.pipe.v2c(slot_size=slot_size, consumer_buf=v2c_buf, id=PIPE_ID, nosplit=True, slot_num=2)
        pipe.init_simd()

        quant_tile = pto.alloc_tile(shape=[TILE_K, N], dtype=quant_pto, valid_shape=[TILE_K, N])
        dequant_tile = pto.alloc_tile(shape=[TILE_K, N], dtype=pto.f32, valid_shape=[TILE_K, N])
        dequant_nz = pto.alloc_tile(shape=[TILE_K, N], dtype=pto.f32,
                                    blayout="ColMajor", slayout="RowMajor", valid_shape=[TILE_K, N])
        scale_tile = pto.alloc_tile(shape=[TILE_K, 8], dtype=pto.f32, valid_shape=[TILE_K, 1])
        offset_tile = pto.alloc_tile(shape=[TILE_K, 8], dtype=pto.f32, valid_shape=[TILE_K, 1])

        for k_tile in range(num_k_tiles):
            quant_view = pto.make_tensor_view(quant_b_ptr, shape=[K, N], strides=[N, 1])
            pto.tile.load(quant_view, quant_tile, offsets=[k_tile * TILE_K, 0], sizes=[TILE_K, N])

            scale_view = pto.make_tensor_view(scale_ptr, shape=[K, 1], strides=[1, 1])
            pto.tile.load(scale_view, scale_tile, offsets=[k_tile * TILE_K, 0], sizes=[TILE_K, 1])

            offset_view = pto.make_tensor_view(offset_ptr, shape=[K, 1], strides=[1, 1])
            pto.tile.load(offset_view, offset_tile, offsets=[k_tile * TILE_K, 0], sizes=[TILE_K, 1])

            pto.tile.dequant(quant_tile, scale_tile, offset_tile, dequant_tile)
            pto.tile.mov(dequant_tile, dequant_nz)
            pipe.push(dequant_nz, split=SPLIT_NO_SPLIT)

    @pto.jit(
        name=f"tpushpop_vcns_cube_{K}x{N}_{quant_name}",
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
        pipe = pto.pipe.v2c(slot_size=slot_size, consumer_buf=v2c_buf, id=PIPE_ID, nosplit=True, slot_num=2)
        pipe.init_cube()

        a_mat = pto.alloc_tile(shape=[M, TILE_K], dtype=pto.f32, memory_space="mat",
                               blayout="ColMajor", slayout="RowMajor", valid_shape=[M, TILE_K])
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
            entry = pipe.pop(split=SPLIT_NO_SPLIT, result_type=pop_tile)

            pto.tile.mov(a_mat, a_left)
            pto.tile.mov(entry, b_right)
            pipe.free(entry, split=SPLIT_NO_SPLIT)

            if k_tile == 0:
                pto.tile.matmul(a_left, b_right, acc)
            else:
                pto.tile.matmul_acc(acc, a_left, b_right, acc)

        out_view = pto.make_tensor_view(out_ptr, shape=[M, N], strides=[N, 1])
        pto.tile.store(acc, out_view)

    @pto.jit(
        name=f"tpushpop_vcns_entry_{K}x{N}_{quant_name}",
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


CASES = []
KERNELS = []

for name, K, N, quant_np in _CASE_PARAMS:
    quant_pto, quant_name = _QUANT_MAP[quant_np]
    entry = _make_kernels(K, N, quant_pto, quant_np)
    KERNELS.append(entry)
    CASES.append(golden_output_case(
        name=name,
        kernel=entry,
        inputs=lambda p=(42, K, N, quant_np): _gen_inputs(*p),
        expected=lambda *inp: _golden(*inp),
        output_shape=(M, N),
        output_dtype=np.float32,
        rtol=1e-3,
        atol=1e-3,
    ))

auto_main(globals())
