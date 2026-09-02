#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Sub-block dispatch contract for the TPUSH/TPOP two- vs three-argument
# overloads.
#
# 7 cases:
#   1-4 V2C (L1 FIFO): implicit vs explicit subBlockId × NO_SPLIT vs UP_DOWN
#   5-7 C2V_GM (GM FIFO): implicit vs explicit subBlockId × NO_SPLIT vs UP_DOWN
#
# Explicit-ID cases hand each core its PEER's id (1 - get_subblockid()), so a
# correct implementation produces half-swapped output; one that silently reads
# the hardware id does not.

from pathlib import Path
import sys

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import auto_main, golden_output_case
from ptodsl import pto


VEC_CORES = 2
PIPE_ID = 0
SPLIT_NO_SPLIT = 0
SPLIT_UP_DOWN = 1


# ── V2C: each vector core loads its K-slice of B, NZ-converts, pushes into L1
# FIFO; cube pops the whole [K, N], multiplies with A, stores A x B. Golden
# pins which vector core wrote which row window of the FIFO slot.
def _make_v2c_kernels(M, K, N, split, explicit_id):
    is_no_split = split == SPLIT_NO_SPLIT
    prod_k = K if is_no_split else K // VEC_CORES
    slot_size = 4 * K * N  # f32
    kernel_tag = f"{M}x{K}x{N}_s{split}_{'expl' if explicit_id else 'impl'}"

    @pto.jit(
        name=f"tpushpop_sbd_vec_{kernel_tag}",
        kernel_kind="vector",
        target="a5",
        mode="auto",
        insert_sync=True,
        entry=False,
        backend="emitc",
    )
    def vec_kernel(
        src_b_ptr: pto.ptr(pto.f32, "gm"),
    ):
        v2c_buf = pto.import_reserved_buffer(
            name="v2c_fifo",
            peer_func=f"tpushpop_sbd_cube_{kernel_tag}",
        )
        pipe = pto.pipe.v2c(slot_size=slot_size, consumer_buf=v2c_buf, id=PIPE_ID, slot_num=2)
        pipe.init_simd()
        sub_block_idx = pto.get_subblock_idx()

        def _emit_producer_body():
            b_tile = pto.alloc_tile(shape=[prod_k, N], dtype=pto.f32, valid_shape=[prod_k, N])
            b_nz = pto.alloc_tile(shape=[prod_k, N], dtype=pto.f32,
                                  blayout="ColMajor", slayout="RowMajor", valid_shape=[prod_k, N])
            b_view = pto.make_tensor_view(src_b_ptr, shape=[K, N], strides=[N, 1])
            pto.tile.load(b_view, b_tile, offsets=[sub_block_idx * prod_k, 0], sizes=[prod_k, N])
            pto.tile.mov(b_tile, b_nz)
            if explicit_id:
                # Hand each core its peer's id: the caller supplied value
                # must be the one the FIFO address arithmetic uses.
                pipe.push(b_nz, split=split, sub_block_id=1 - sub_block_idx)
            else:
                pipe.push(b_nz, split=split)

        # TILE_NO_SPLIT only signals sub-block 0, so AIV1 must stay out.
        if is_no_split:
            with pto.if_(sub_block_idx == 0) as br:
                with br.then_:
                    _emit_producer_body()
        else:
            _emit_producer_body()


    @pto.jit(
        name=f"tpushpop_sbd_cube_{kernel_tag}",
        kernel_kind="cube",
        target="a5",
        mode="auto",
        insert_sync=True,
        entry=False,
        backend="emitc",
    )
    def cube_kernel(
        out_ptr: pto.ptr(pto.f32, "gm"),
        src_a_ptr: pto.ptr(pto.f32, "gm"),
        src_b_ptr: pto.ptr(pto.f32, "gm"),
    ):
        v2c_buf = pto.reserve_buffer(
            name="v2c_fifo", size=slot_size * 4,
            location="mat", auto=True,
        )
        pipe = pto.pipe.v2c(slot_size=slot_size, consumer_buf=v2c_buf, id=PIPE_ID, slot_num=2)
        pipe.init_cube()

        a_mat = pto.alloc_tile(shape=[M, K], dtype=pto.f32, memory_space="mat",
                               blayout="ColMajor", slayout="RowMajor", valid_shape=[M, K])
        pop_tile = pto.alloc_tile(shape=[K, N], dtype=pto.f32, memory_space="mat",
                                  blayout="ColMajor", slayout="RowMajor", valid_shape=[K, N])
        a_left = pto.alloc_tile(shape=[M, K], dtype=pto.f32, memory_space="left",
                                blayout="ColMajor", slayout="RowMajor", valid_shape=[M, K])
        b_right = pto.alloc_tile(shape=[K, N], dtype=pto.f32, memory_space="right",
                                 blayout="RowMajor", slayout="ColMajor", valid_shape=[K, N])
        acc = pto.alloc_tile(shape=[M, N], dtype=pto.f32, memory_space="acc",
                             blayout="ColMajor", slayout="RowMajor", valid_shape=[M, N])

        a_view = pto.make_tensor_view(src_a_ptr, shape=[M, K], strides=[K, 1])
        pto.tile.load(a_view, a_mat)

        # The Mat consumer path takes the whole slot whatever the sub-block ID
        # is; the explicit overload must behave exactly like the implicit one.
        if explicit_id:
            entry = pipe.pop(split=split, result_type=pop_tile, sub_block_id=0)
        else:
            entry = pipe.pop(split=split, result_type=pop_tile)

        pto.tile.mov(a_mat, a_left)
        pto.tile.mov(entry, b_right)
        pipe.free(entry, split=split)
        pto.tile.matmul(a_left, b_right, acc)

        out_view = pto.make_tensor_view(out_ptr, shape=[M, N], strides=[N, 1])
        out_part = pto.partition_view(out_view, offsets=[0, 0], sizes=[M, N])
        pto.tile.store(acc, out_part)

    @pto.jit(
        name=f"tpushpop_sbd_entry_{kernel_tag}",
        target="a5",
        mode="auto",
        insert_sync=True,
        entry=True,
        backend="emitc",
    )
    def entry_kernel(
        src_a_ptr: pto.ptr(pto.f32, "gm"),
        src_b_ptr: pto.ptr(pto.f32, "gm"),
        out_ptr: pto.ptr(pto.f32, "gm"),
    ):
        cube_kernel(out_ptr, src_a_ptr, src_b_ptr)
        vec_kernel(src_b_ptr)

    return entry_kernel


def _golden_v2c(a, b, split, explicit_id):
    # Implicit id: each core pushes its own K-half in order → B reassembled.
    # Explicit swapped id: core 0 writes the lower half with id 1 (offset by
    # prod_k) and core 1 writes the upper half with id 0 → halves swap.
    # TILE_NO_SPLIT ignores the ID entirely (single lane): the contract is
    # "passing a non-zero ID must not move the payload either".
    if not explicit_id or split == SPLIT_NO_SPLIT:
        return np.matmul(a, b)
    K = b.shape[0]
    half = K // 2
    b_swapped = np.concatenate([b[half:], b[:half]], axis=0)
    return np.matmul(a, b_swapped)


# ── C2V_GM: cube computes A x B, pushes whole [M, N] acc into GM FIFO; each
# vector core pops its own M-sub-range and stores to its own row window.
def _make_c2v_gm_kernels(M, K, N, split, explicit_id, fifo_elems):
    is_no_split = split == SPLIT_NO_SPLIT
    cons_m = M if is_no_split else M // VEC_CORES
    slot_size = 4 * M * N  # f32
    kernel_tag = f"{M}x{K}x{N}_s{split}_{'expl' if explicit_id else 'impl'}"

    @pto.jit(
        name=f"tpushpop_sbd_gm_cube_{kernel_tag}",
        kernel_kind="cube",
        target="a5",
        mode="auto",
        insert_sync=True,
        entry=False,
        backend="emitc",
    )
    def cube_kernel(
        src_a_ptr: pto.ptr(pto.f32, "gm"),
        src_b_ptr: pto.ptr(pto.f32, "gm"),
        fifo_ptr: pto.ptr(pto.f32, "gm"),
    ):
        fifo_view = pto.make_tensor_view(fifo_ptr, shape=[M, N], strides=[N, 1])
        pipe = pto.pipe.c2v(slot_size=slot_size, gm_slot_tensor=fifo_view, id=PIPE_ID, slot_num=2)
        pipe.init_cube()

        a_mat = pto.alloc_tile(shape=[M, K], dtype=pto.f32, memory_space="mat",
                               blayout="ColMajor", slayout="RowMajor", valid_shape=[M, K])
        b_mat = pto.alloc_tile(shape=[K, N], dtype=pto.f32, memory_space="mat",
                               blayout="ColMajor", slayout="RowMajor", valid_shape=[K, N])
        a_left = pto.alloc_tile(shape=[M, K], dtype=pto.f32, memory_space="left",
                                blayout="ColMajor", slayout="RowMajor", valid_shape=[M, K])
        b_right = pto.alloc_tile(shape=[K, N], dtype=pto.f32, memory_space="right",
                                 blayout="RowMajor", slayout="ColMajor", valid_shape=[K, N])
        acc = pto.alloc_tile(shape=[M, N], dtype=pto.f32, memory_space="acc",
                             blayout="ColMajor", slayout="RowMajor", valid_shape=[M, N])

        a_view = pto.make_tensor_view(src_a_ptr, shape=[M, K], strides=[K, 1])
        b_view = pto.make_tensor_view(src_b_ptr, shape=[K, N], strides=[N, 1])
        pto.tile.load(a_view, a_mat)
        pto.tile.load(b_view, b_mat)
        pto.tile.mov(a_mat, a_left)
        pto.tile.mov(b_mat, b_right)
        pto.tile.matmul(a_left, b_right, acc)

        # The Acc producer path never looks at the sub-block ID.
        pipe.push(acc, split=split)

    @pto.jit(
        name=f"tpushpop_sbd_gm_vec_{kernel_tag}",
        kernel_kind="vector",
        target="a5",
        mode="auto",
        insert_sync=True,
        entry=False,
        backend="emitc",
    )
    def vec_kernel(
        out_ptr: pto.ptr(pto.f32, "gm"),
        fifo_ptr: pto.ptr(pto.f32, "gm"),
    ):
        fifo_view = pto.make_tensor_view(fifo_ptr, shape=[M, N], strides=[N, 1])
        pipe = pto.pipe.c2v(slot_size=slot_size, gm_slot_tensor=fifo_view, id=PIPE_ID, slot_num=2)
        pipe.init_simd()
        sub_block_idx = pto.get_subblock_idx()

        def _emit_consumer_body():
            # TPOP<MatPipe, VecTileCons, Split>(mPipe, vecTile): the consumer
            # is a UB tile of the SUB-RANGE shape [cons_m, N]; TPOP assigns its
            # UB address and TLOADs the correctly-offset GM sub-range into it.
            result_tile = pto.alloc_tile(shape=[cons_m, N], dtype=pto.f32,
                                         valid_shape=[cons_m, N])
            if explicit_id:
                # Hand each core its peer's id: the caller supplied value
                # must be the one the FIFO sub-range arithmetic uses.
                entry = pipe.pop(split=split, result_type=result_tile,
                                 sub_block_id=1 - sub_block_idx)
            else:
                entry = pipe.pop(split=split, result_type=result_tile)

            out_view = pto.make_tensor_view(out_ptr, shape=[M, N], strides=[N, 1])
            out_part = pto.partition_view(out_view, offsets=[sub_block_idx * cons_m, 0],
                                          sizes=[cons_m, N])
            pto.tile.store(entry, out_part)

        if is_no_split:
            with pto.if_(sub_block_idx == 0) as br:
                with br.then_:
                    _emit_consumer_body()
        else:
            _emit_consumer_body()

    @pto.jit(
        name=f"tpushpop_sbd_gm_entry_{kernel_tag}",
        target="a5",
        mode="auto",
        insert_sync=True,
        entry=True,
        backend="emitc",
    )
    def entry_kernel(
        src_a_ptr: pto.ptr(pto.f32, "gm"),
        src_b_ptr: pto.ptr(pto.f32, "gm"),
        fifo_ptr: pto.ptr(pto.f32, "gm"),
        out_ptr: pto.ptr(pto.f32, "gm"),
    ):
        cube_kernel(src_a_ptr, src_b_ptr, fifo_ptr)
        vec_kernel(out_ptr, fifo_ptr)

    return entry_kernel


def _golden_c2v_gm(a, b, split, explicit_id):
    out = np.matmul(a, b)
    if not explicit_id or split == SPLIT_NO_SPLIT:
        return out
    # Explicit swapped id: each core reads its peer's M-sub-range → halves swap.
    M = out.shape[0]
    half = M // 2
    return np.concatenate([out[half:], out[:half]], axis=0)


# ── Case registry ──────────────────────────────────────────────────────────

_V2C_CASES = [
    # (name, M, K, N, split, explicit_id)
    ("case1_v2c_nosplit_implicit_id",        16, 64, 32, SPLIT_NO_SPLIT, False),
    ("case2_v2c_nosplit_explicit_id",        16, 64, 32, SPLIT_NO_SPLIT, True),
    ("case3_v2c_split_implicit_id",          16, 64, 32, SPLIT_UP_DOWN,  False),
    ("case4_v2c_split_explicit_swapped_id",  16, 64, 32, SPLIT_UP_DOWN,  True),
]

_C2V_GM_CASES = [
    ("case5_c2v_gm_nosplit_implicit_id",        32, 32, 64, SPLIT_NO_SPLIT, False),
    ("case6_c2v_gm_split_implicit_id",          32, 32, 64, SPLIT_UP_DOWN,  False),
    ("case7_c2v_gm_split_explicit_swapped_id",  32, 32, 64, SPLIT_UP_DOWN,  True),
]

CASES = []
KERNELS = []

for name, M, K, N, split, explicit_id in _V2C_CASES:
    kernel = _make_v2c_kernels(M, K, N, split, explicit_id)
    KERNELS.append(kernel)

    def _gen(seed=0, M=M, K=K, N=N):
        np.random.seed(seed)
        return [np.random.uniform(-2, 2, [M, K]).astype(np.float32),
                np.random.uniform(-2, 2, [K, N]).astype(np.float32)]

    CASES.append(golden_output_case(
        name,
        kernel,
        inputs=_gen,
        expected=lambda *inp, p=(split, explicit_id): _golden_v2c(inp[0], inp[1], *p),
    ))

for name, M, K, N, split, explicit_id in _C2V_GM_CASES:
    fifo_elems = M * N
    kernel = _make_c2v_gm_kernels(M, K, N, split, explicit_id, fifo_elems)
    KERNELS.append(kernel)

    def _gen(seed=0, M=M, K=K, N=N):
        np.random.seed(seed)
        return [np.random.uniform(-2, 2, [M, K]).astype(np.float32),
                np.random.uniform(-2, 2, [K, N]).astype(np.float32),
                np.zeros([M * N], dtype=np.float32)]

    CASES.append(golden_output_case(
        name,
        kernel,
        inputs=_gen,
        expected=lambda *inp, p=(split, explicit_id): _golden_c2v_gm(inp[0], inp[1], *p),
    ))

auto_main(globals())
