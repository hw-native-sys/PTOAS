#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Subtile test case1_half_128x512
#   M=128, K=128, N=128, RepeatN=4, InT=fp16, OutT=fp32
#   GM-staged cube-to-vector pipe, slot=[128, 512] f32
#   Cube: matmul → TALLOC → TSTORE [M,N] into [M,FULL_N] slot → TPUSH
#   Vec:  TPOP [M/2, FULL_N] GlobalTensor → subtile loop TLOAD [16,512] → TADDS(+3.14) → TSTORE

from pathlib import Path
import sys

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import auto_main
from ptodsl import pto


# ── Case parameters ──────────────────────────────────────────────────────

M = 128
K = 128
N = 128
REPEAT_N = 4
FULL_N = N * REPEAT_N  # 512
VEC_CORES = 2
VEC_M = 16
VEC_LOAD_TIMES = M // (VEC_CORES * VEC_M)  # 4

SLOT_SIZE = M * FULL_N * 4  # f32 = 4 bytes
PIPE_ID = 0
SPLIT_UP_DOWN = 1

np.random.seed(19)


def _gen_inputs():
    a = np.random.uniform(-1, 1, [M, K]).astype(np.float16)
    b = np.random.uniform(-1, 1, [K, N]).astype(np.float16)
    return [a, b]


def _golden(a, b):
    matmul = np.matmul(a.astype(np.float32), b.astype(np.float32)).astype(np.float32)
    result = np.tile(matmul, [1, REPEAT_N]) + np.float32(3.14)
    return result


# ── Cube kernel ──────────────────────────────────────────────────────────

@pto.jit(
    name="tpushpop_subtile_cube",
    kernel_kind="cube",
    target="a5",
    mode="auto",
    insert_sync=True,
    entry=False,
    backend="emitc",
)
def _cube_kernel(
    a_ptr: pto.ptr(pto.f16, "gm"),
    b_ptr: pto.ptr(pto.f16, "gm"),
    gm_slot_ptr: pto.ptr(pto.f32, "gm"),
):
    slot_view = pto.make_tensor_view(gm_slot_ptr, shape=[M, FULL_N], strides=[FULL_N, 1])
    pipe = pto.pipe.c2v(slot_size=SLOT_SIZE, gm_slot_tensor=slot_view, id=PIPE_ID, slot_num=2)
    pipe.init_cube()

    a_mat = pto.alloc_tile(
        shape=[M, K], dtype=pto.f16, memory_space="mat",
        blayout="ColMajor", slayout="RowMajor",
        valid_shape=[M, K],
    )
    b_mat = pto.alloc_tile(
        shape=[K, N], dtype=pto.f16, memory_space="mat",
        blayout="ColMajor", slayout="RowMajor",
        valid_shape=[K, N],
    )
    a_left = pto.alloc_tile(
        shape=[M, K], dtype=pto.f16, memory_space="left",
        blayout="ColMajor", slayout="RowMajor",
        valid_shape=[M, K],
    )
    b_right = pto.alloc_tile(
        shape=[K, N], dtype=pto.f16, memory_space="right",
        blayout="RowMajor", slayout="ColMajor",
        valid_shape=[K, N],
    )
    acc = pto.alloc_tile(
        shape=[M, N], dtype=pto.f32, memory_space="acc",
        blayout="ColMajor", slayout="RowMajor",
        valid_shape=[M, N],
    )

    a_view = pto.make_tensor_view(a_ptr, shape=[M, K], strides=[K, 1])
    b_view = pto.make_tensor_view(b_ptr, shape=[K, N], strides=[N, 1])
    pto.tile.load(a_view, a_mat)
    pto.tile.load(b_view, b_mat)
    pto.tile.mov(a_mat, a_left)
    pto.tile.mov(b_mat, b_right)
    pto.tile.matmul(a_left, b_right, acc)

    # Allocate GM slot entry, store [M, N] slices into [M, FULL_N] slot
    push_entry = pipe.alloc(split=SPLIT_UP_DOWN)
    for n_tile in range(REPEAT_N):
        store_view = pto.make_tensor_view(
            gm_slot_ptr, shape=[M, N], strides=[FULL_N, 1],
        )
        pto.tile.store(acc, store_view, offsets=[0, n_tile * N], sizes=[M, N])

    pipe.push(push_entry, split=SPLIT_UP_DOWN)


# ── Vector kernel ────────────────────────────────────────────────────────

@pto.jit(
    name="tpushpop_subtile_vec",
    kernel_kind="vector",
    target="a5",
    mode="auto",
    insert_sync=True,
    entry=False,
    backend="emitc",
)
def _vec_kernel(
    a_ptr: pto.ptr(pto.f16, "gm"),
    b_ptr: pto.ptr(pto.f16, "gm"),
    out_ptr: pto.ptr(pto.f32, "gm"),
    gm_slot_ptr: pto.ptr(pto.f32, "gm"),
):
    # The GM FIFO slot is a standalone fifoMem buffer shared by both cores.
    slot_view = pto.make_tensor_view(gm_slot_ptr, shape=[M, FULL_N], strides=[FULL_N, 1])
    pipe = pto.pipe.c2v(slot_size=SLOT_SIZE, gm_slot_tensor=slot_view, id=PIPE_ID, slot_num=2)
    pipe.init_simd()

    # PopGlobal = GlobalTensor<OutT, [1,1,1, M/VEC_CORES, FULL_N], stride FULL_N>:
    # each core pops its OWN [64, 512] sub-range view (UP_DOWN subAIVOffset uses
    # this shape). Popping the full [128, 512] slot view makes AIV1's offset
    # span the whole slot and read past the end.
    cons_rows = M // VEC_CORES
    pop_type = pto.make_tensor_view(gm_slot_ptr, shape=[cons_rows, FULL_N],
                                    strides=[FULL_N, 1])
    pop_entry = pipe.pop(split=SPLIT_UP_DOWN, result_type=pop_type)
    # vecBaseRow = (M / VEC_CORES) * subBlockIdx — AIV1 writes rows
    # [64,128) of out, AIV0 writes rows [0,64)
    vec_base_row = (M // VEC_CORES) * pto.get_subblock_idx()

    for row_slice in range(VEC_LOAD_TIMES):
        vec_tile = pto.alloc_tile(
            shape=[VEC_M, FULL_N], dtype=pto.f32,
            valid_shape=[VEC_M, FULL_N],
        )
        dst_tile = pto.alloc_tile(
            shape=[VEC_M, FULL_N], dtype=pto.f32,
            valid_shape=[VEC_M, FULL_N],
        )

        pto.tile.load(pop_entry, vec_tile,
                       offsets=[row_slice * VEC_M, 0],
                       sizes=[VEC_M, FULL_N])
        pto.tile.adds(vec_tile, 3.14, dst_tile)

        out_view = pto.make_tensor_view(out_ptr, shape=[M, FULL_N], strides=[FULL_N, 1])
        pto.tile.store(dst_tile, out_view,
                        offsets=[vec_base_row + row_slice * VEC_M, 0],
                        sizes=[VEC_M, FULL_N])

    pipe.free(pop_entry, split=SPLIT_UP_DOWN)


# ── Entry kernel ─────────────────────────────────────────────────────────

@pto.jit(
    name="tpushpop_subtile_entry",
    target="a5",
    mode="auto",
    insert_sync=True,
    backend="emitc",
)
def _entry_kernel(
    a_ptr: pto.ptr(pto.f16, "gm"),
    b_ptr: pto.ptr(pto.f16, "gm"),
    out_ptr: pto.ptr(pto.f32, "gm"),
    gm_slot_ptr: pto.ptr(pto.f32, "gm"),
):
    _cube_kernel(a_ptr, b_ptr, gm_slot_ptr)
    _vec_kernel(a_ptr, b_ptr, out_ptr, gm_slot_ptr)


# ── Case registration ────────────────────────────────────────────────────

def _make_case():
    # Generate the random pair ONCE: make_case/golden must see the same data
    # (_gen_inputs() advances the global RNG on each call).
    a, b = _gen_inputs()
    return ([a, b, np.zeros((M, FULL_N), dtype=np.float32),
             np.zeros(SLOT_SIZE // 4, dtype=np.float32)],
            _golden(a, b))


CASES = [
    {
        "name": "case1_half_128x512",
        "kernel": _entry_kernel,
        "make_case": _make_case,
        "check": lambda dev_inputs, golden: np.testing.assert_allclose(
            dev_inputs[2].cpu().numpy(), golden, rtol=1e-3, atol=1e-3,
        ),
    },
]

KERNELS = [_entry_kernel]

auto_main(globals())
