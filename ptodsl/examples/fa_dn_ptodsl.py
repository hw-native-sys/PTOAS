# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""
PTODSL structural port of the A5 Flash Attention DN kernel in ``fa_dn.cpp``.

This file is intentionally source-locked to:

- ``ptodsl/examples/fa_dn.cpp``
- ``ptodsl/docs/user_guide/14-fa-dn-cpp-pseudocode.md``

The goal here is to preserve the same control skeleton as the C++ kernel:

1. ``compute_qk``  : cube Q @ K_t^T
2. ``compute_p``   : vector online softmax + P materialization
3. ``compute_pv``  : cube P_t @ V_t
4. ``compute_gu``  : vector running-output update

and the same top-level prologue / steady-state / epilogue scheduling in
``runTFA``.

This file currently prioritizes stage boundaries, loop nesting, and state
ownership over PTODSL compile completeness. In particular, the stage helper
bodies are written as structure-aligned scaffolding for the C++ kernel:

- ``compute_qk(tile_id, sub_tile_id, ...)``
- ``compute_p(tile_id, row_slice, ...)``
- ``compute_pv(tile_id, sub_tile_id, ...)``
- ``compute_gu(tile_id, ...)``

The intended next step is to refine the per-slice data motion inside those
helpers without changing the surrounding control skeleton.
"""

import argparse
from dataclasses import dataclass

from ptodsl import pto, scalar
try:
    from .fa_dn_matmul import AccMode, pto_macro_matmul
except ImportError:
    from fa_dn_matmul import AccMode, pto_macro_matmul
try:
    from .fa_dn_softmax import pto_macro_fa_softmax_dn
except ImportError:
    from fa_dn_softmax import pto_macro_fa_softmax_dn
try:
    from .tsync_custom_helper import (
        SyncOpType,
        TSync_Custom,
    )
except ImportError:
    from tsync_custom_helper import (
        SyncOpType,
        TSync_Custom,
    )


BUF0_QK_READY = 0
BUF1_SM_READY = 2
UPDATE_READY = 4
SS_BUF_READY = 6
RUNNING_O_AVALIABLE = 8

kFaCvFifoSize = 8
kFaCvFifoConsSyncPeriod = kFaCvFifoSize // 2
kFaCubeS1 = 128
kFaTileS1 = 256
# ``fa_dn.cpp`` guards the current UB->L1 path with
# ``qkPreloadNum <= pMatTNBuffers (2)``.
# Keep the PTODSL demo default at 2 so ``emit_fa_dn_mlir()`` does not trip the
# structural assert by default.
kFaQkPreload = 2
kFaLaunchCoreCount = 28
kFaProfileBytesPerBlock = 1024 * 3
kFaCvCommSlotBytes = 512
VEC_CORES = 2


def _visible_tiles_for_block(logical_block_idx: int, CUBE_S0: int, Tile_S1: int) -> int:
    return 1 + (logical_block_idx * CUBE_S0) // Tile_S1


def _reduce_tile(rows: int):
    return pto.alloc_tile(
        shape=[rows, 1],
        dtype=pto.f32,
        valid_shape=[rows, 1],
        blayout="ColMajor",
    )


def _steady_tile_end(num_tiles_s1: int, qkPreloadNum: int) -> int:
    return max(num_tiles_s1 - qkPreloadNum, 0)


def should_notify_consumption(sync_iter: int, fifo_size: int, sync_period: int) -> int:
    assert fifo_size >= 1, "CV FIFO size must be >= 1"
    period = sync_period if sync_period > 0 else 1
    assert period >= 1, "CV FIFO consume sync period must be >= 1"
    return ((sync_iter + 1) % period) == 0


def should_wait_consumption(sync_iter: int, fifo_size: int, sync_period: int) -> int:
    assert fifo_size >= 1, "CV FIFO size must be >= 1"
    period = sync_period if sync_period > 0 else 1
    assert period >= 1, "CV FIFO consume sync period must be >= 1"
    if sync_iter < fifo_size:
        return False
    return (sync_iter % period) == 0


def _is_runtime_scalar_value(value) -> bool:
    return not isinstance(value, (bool, int, float)) and hasattr(value, "type")


def _scalar_min_or_static(lhs, rhs):
    if _is_runtime_scalar_value(lhs) or _is_runtime_scalar_value(rhs):
        return scalar.min(lhs, rhs)
    return min(lhs, rhs)


def _scalar_max_or_static(lhs, rhs):
    if _is_runtime_scalar_value(lhs) or _is_runtime_scalar_value(rhs):
        return scalar.max(lhs, rhs)
    return max(lhs, rhs)


def allocate_cube_tile_buffers(qMatTile, kMatTile, pMatTile, vMatTile):
    # Structural mirror of ``fa_dn.cpp``: MAT-stage L1 tile groups are allocated
    # first as one bundle, then L0C accumulators are handled separately below.
    _ = (qMatTile, kMatTile, pMatTile, vMatTile)


@dataclass(frozen=True)
class PreBTExtOpReadyHook:
    def __call__(self):
        pto.wait_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=1)


@dataclass(frozen=True)
class PreATExtOpReadyHook:
    def __call__(self):
        pto.wait_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=0)


@dataclass(frozen=True)
class PReadyHook:
    sync: object
    enable: object

    def __call__(self):
        with pto.if_(self.enable) as enable_br:
            with enable_br.then_:
                self.sync.wait()


@dataclass(frozen=True)
class QReadyHook:
    enable: object

    def __call__(self):
        with pto.if_(self.enable) as enable_br:
            with enable_br.then_:
                pto.wait_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=1)


@dataclass(frozen=True)
class Sm2PvFreeHook:
    sync: object
    enable: object

    def __call__(self):
        with pto.if_(self.enable) as enable_br:
            with enable_br.then_:
                self.sync.free()


@pto.tileop
def qk_matmul_stage(
    qMatTile: pto.Tile,
    kMatTile: pto.Tile,
    qL0ATile: pto.Tile,
    kL0BTile: pto.Tile,
    qkAccTile: pto.Tile,
    qkVecTileSub: pto.Tile,
):
    rows = qMatTile.valid_shape[0]
    head = qMatTile.valid_shape[1]
    cols = kMatTile.valid_shape[0]

    pto.mte_l1_l0a(qMatTile.as_ptr(), qL0ATile.as_ptr(), rows, head)
    pto.mte_l1_l0b(kMatTile.as_ptr(), kL0BTile.as_ptr(), head, cols, transpose=True)
    pto.mad(qL0ATile.as_ptr(), kL0BTile.as_ptr(), qkAccTile.as_ptr(), rows, cols, head)
    pto.mte_l0c_ub(qkAccTile.as_ptr(), qkVecTileSub.as_ptr(), rows, cols, cols, cols, 0)


@pto.tileop
def pv_matmul_stage(
    pMatTileSub: pto.Tile,
    vMatTile: pto.Tile,
    pL0ATile: pto.Tile,
    vL0BTile: pto.Tile,
    pvAccCurrTile: pto.Tile,
    pvVecTileSub: pto.Tile,
):
    rows = pMatTileSub.valid_shape[0]
    kv = pMatTileSub.valid_shape[1]
    head = vMatTile.valid_shape[1]

    pto.mte_l1_l0a(pMatTileSub.as_ptr(), pL0ATile.as_ptr(), rows, kv)
    pto.mte_l1_l0b(vMatTile.as_ptr(), vL0BTile.as_ptr(), kv, head)
    pto.mad(pL0ATile.as_ptr(), vL0BTile.as_ptr(), pvAccCurrTile.as_ptr(), rows, head, kv)
    pto.mte_l0c_ub(pvAccCurrTile.as_ptr(), pvVecTileSub.as_ptr(), rows, head, head, head, 0)

@pto.simt
def gu_update_stage(
    runningOTile: pto.Tile,
    pvVecTile: pto.Tile,
    l1_exp_max_ififo: pto.Tile,
    tmpOTile: pto.Tile,
):
    rows = runningOTile.shape[0]
    cols = runningOTile.shape[1]
    for row in range(rows):
        row_exp = scalar.load(l1_exp_max_ififo[row, 0])
        for col in range(cols):
            accum = scalar.load(runningOTile[row, col])
            pv_val = scalar.load(pvVecTile[row, col])
            scalar.store(accum * row_exp + pv_val, tmpOTile[row, col])
    pto.tile.mov(tmpOTile, runningOTile)


@pto.simt
def gu_last_stage(
    runningOTile: pto.Tile,
    pvVecTile: pto.Tile,
    l1_exp_max_ififo: pto.Tile,
    l2_global_sum: pto.Tile,
    tmpOTile: pto.Tile,
):
    rows = runningOTile.shape[0]
    cols = runningOTile.shape[1]
    for row in range(rows):
        row_exp = scalar.load(l1_exp_max_ififo[row, 0])
        row_sum = scalar.load(l2_global_sum[row, 0])
        for col in range(cols):
            accum = scalar.load(runningOTile[row, col])
            pv_val = scalar.load(pvVecTile[row, col])
            scalar.store((accum * row_exp + pv_val) / row_sum, tmpOTile[row, col])
    pto.tile.mov(tmpOTile, runningOTile)


@pto.simt
def gu_single_last_stage(
    runningOTile: pto.Tile,
    l2_global_sum: pto.Tile,
):
    rows = runningOTile.shape[0]
    cols = runningOTile.shape[1]
    for row in range(rows):
        row_sum = scalar.load(l2_global_sum[row, 0])
        for col in range(cols):
            scalar.store(scalar.load(runningOTile[row, col]) / row_sum, runningOTile[row, col])


def compute_qk(
    *,
    S0: int,
    HEAD_SIZE: int,
    S1: int,
    Cube_S0: int,
    Cube_S1: int,
    Tile_S1: int,
    QKP_CV_FIFO: int,
    CV_FIFO_CONS_SYNC_PERIOD: int,
    INTERMEDIATE_CHECK: bool,
    CAUSAL_MASK: bool,
    SRC_VEC_TN_BUFFERS: int,
    TileMatQData,
    TileMatKData,
    TileQKData,
    TileQKVecData,
    TSyncQK2SM,
    tile_id: int,
    sub_tile_id: int,
    ub_buf_idx: int,
    q,
    k,
    qk_tile_fifo,
    qMatTile,
    kMatTile,
    qkAccTile,
    qkVecTile,
    qkMatTileEventId,
    accTileEvtID,
    qk2smSync,
    blk_idx: int,
):
    kTileFactor = Tile_S1 // Cube_S1
    Cube_HEAD = HEAD_SIZE
    assert QKP_CV_FIFO >= 1, "QKP_CV_FIFO must be >= 1"
    assert Tile_S1 % Cube_S1 == 0, "TILE_S1 must be divisible by CUBE_S1"
    
    s0_index = blk_idx * Cube_S0
    s1_index = tile_id * Tile_S1 + sub_tile_id * Cube_S1
    sync_iter = tile_id
    should_wait_consume = should_wait_consumption(
        sync_iter, QKP_CV_FIFO, CV_FIFO_CONS_SYNC_PERIOD
    )

    GlobalDataQ = lambda ptr: pto.make_tensor_view(
        ptr,
        shape=[Cube_S0, HEAD_SIZE],
        strides=[HEAD_SIZE, 1],
        layout="DN",
    )
    GlobalDataK = lambda ptr: pto.make_tensor_view(
        ptr,
        shape=[Cube_S1, HEAD_SIZE],
        strides=[HEAD_SIZE, 1],
    )
    TileDataF_Sub = lambda addr: pto.alloc_tile(
        shape=[Tile_S1, Cube_S0 // VEC_CORES // kTileFactor],
        dtype=pto.f32,
        valid_shape=[Tile_S1, Cube_S0 // VEC_CORES // kTileFactor],
        blayout="RowMajor",
        addr=addr,
    )

    def emit_qk_main_path():
        qGlobal = GlobalDataQ(q)
        kGlobal = GlobalDataK(pto.addptr(k, s1_index * HEAD_SIZE))

        pto.wait_flag(pto.Pipe.MTE1, pto.Pipe.MTE2, event_id=qkMatTileEventId)

        should_load_q = (tile_id == 0) & (sub_tile_id == 0)
        with pto.if_(should_load_q) as should_load_q_br:
            with should_load_q_br.then_:
                qGlobalPart = pto.partition_view(
                    qGlobal,
                    offsets=[0, 0],
                    sizes=[Cube_S0, HEAD_SIZE],
                )
                pto.tile.load(qGlobalPart, qMatTile)
                pto.set_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=1)

        kGlobalPart = pto.partition_view(
            kGlobal,
            offsets=[0, 0],
            sizes=[Cube_S1, HEAD_SIZE],
        )
        pto.tile.load(kGlobalPart, kMatTile)

        pto.set_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=0)
        qReadyHook = QReadyHook((tile_id == 0) & (sub_tile_id == 0))
        preATExtOpReadyHook = PreATExtOpReadyHook()

        pto.wait_flag(pto.Pipe.FIX, pto.Pipe.M, event_id=accTileEvtID)

        pto_macro_matmul(
            Cube_M=Cube_S1,
            Tile_K=Cube_HEAD,
            Cube_N=Cube_S0,
            L1LoadBFirst=True,
            aMatTile=kMatTile,
            bMatTile=qMatTile,
            cAccTile=qkAccTile,
            accMode=AccMode.Init,
            preATExtOpHook=preATExtOpReadyHook,
            preBTExtOpHook=qReadyHook,
            postTExtOpHook=lambda: None,
        )

        pto.set_flag(pto.Pipe.MTE1, pto.Pipe.MTE2, event_id=qkMatTileEventId)
        pto.set_flag(pto.Pipe.M, pto.Pipe.FIX, event_id=0)
        pto.wait_flag(pto.Pipe.M, pto.Pipe.FIX, event_id=0)

        should_allocate_qk2sm = (sub_tile_id == 0) & should_wait_consume
        with pto.if_(should_allocate_qk2sm) as should_allocate_qk2sm_br:
            with should_allocate_qk2sm_br.then_:
                qk2smSync.allocate()

        if INTERMEDIATE_CHECK:
            buf_idx = tile_id % QKP_CV_FIFO
            base_elems = buf_idx * kTileFactor * Cube_S0 * Cube_S1 + sub_tile_id * Cube_S0 * Cube_S1
            GlobalDataQK = lambda ptr: pto.make_tensor_view(
                ptr,
                shape=[Cube_S1, Cube_S0],
                strides=[Cube_S0, 1],
            )
            qkGlobalTile = GlobalDataQK(pto.addptr(qk_tile_fifo, base_elems))
            qkGlobalTilePart = pto.partition_view(
                qkGlobalTile,
                offsets=[0, 0],
                sizes=[Cube_S1, Cube_S0],
            )
            pto.tile.store(qkAccTile, qkGlobalTilePart)
            pto.set_flag(pto.Pipe.FIX, pto.Pipe.M, event_id=accTileEvtID)
            pto.wait_flag(pto.Pipe.FIX, pto.Pipe.M, event_id=accTileEvtID)

        Vec_S0 = Cube_S0 // VEC_CORES // kTileFactor
        qkVecTileSubDN = TileDataF_Sub(pto.addptr(qkVecTile.as_ptr(), sub_tile_id * Cube_S1))
        pto.tile.mov(qkAccTile, qkVecTileSubDN, mode="split_n")

        pto.set_flag(pto.Pipe.FIX, pto.Pipe.M, event_id=accTileEvtID)

        with pto.if_(sub_tile_id == kTileFactor - 1) as sub_tile_is_last_br:
            with sub_tile_is_last_br.then_:
                qk2smSync.record()

    if CAUSAL_MASK:
        with pto.if_(s1_index > s0_index) as s1_gt_s0_br:
            with s1_gt_s0_br.then_:
                should_allocate_qk2sm = (sub_tile_id == 0) & should_wait_consume
                with pto.if_(should_allocate_qk2sm) as should_allocate_qk2sm_br:
                    with should_allocate_qk2sm_br.then_:
                        qk2smSync.allocate()
                with pto.if_(sub_tile_id == kTileFactor - 1) as sub_tile_is_last_br:
                    with sub_tile_is_last_br.then_:
                        qk2smSync.record()
            with s1_gt_s0_br.else_:
                emit_qk_main_path()
    else:
        emit_qk_main_path()


def compute_p(
    *,
    S0: int,
    HEAD_SIZE: int,
    S1: int,
    Cube_S0: int,
    Cube_S1: int,
    Tile_S1: int,
    QKP_CV_FIFO: int,
    CV_FIFO_CONS_SYNC_PERIOD: int,
    INTERMEDIATE_CHECK: bool,
    CAUSAL_MASK: bool,
    PMAT_TN_BUFFERS: int,
    TileDataF_T,
    TileDataH_T,
    TileDataH_NZ_T,
    ReduceTileF_T,
    TileMatPData,
    TSyncQK2SM,
    TSyncSM2PV,
    tile_id: int,
    row_slice: int,
    qk_tile_fifo,
    p_tile_fifo,
    exp_max_ififo,
    qkVecTile,
    x_expT,
    m1_local_max,
    l1_local_sum,
    m2_global_max,
    l2_global_sum,
    l1_exp_max_ififo,
    pMatTile,
    nzConvBuffer,
    pTileEventId,
    qk2smSync,
    sm2pvSync,
    logical_block_idx: int,
    num_tiles: int,
):
    kTileFactor = Tile_S1 // Cube_S1
    Vec_S0 = Cube_S0 // VEC_CORES // kTileFactor
    initFlag = tile_id == 0
    assert QKP_CV_FIFO >= 1, "QKP_CV_FIFO must be >= 1"
    assert Tile_S1 % Cube_S1 == 0, "TILE_S1 must be divisible by CUBE_S1"
    assert Cube_S0 % (VEC_CORES * kTileFactor) == 0, (
        "Vec rows must divide evenly across tile slices"
    )

    subblock_base_rows = (Cube_S0 // VEC_CORES) * pto.get_subblock_idx()
    row_offset = subblock_base_rows + row_slice * Vec_S0
    s0_index = logical_block_idx * Cube_S0 + row_offset
    s1_index = tile_id * Tile_S1
    sync_iter = tile_id
    last_tile = tile_id == num_tiles - 1
    should_notify_consume = should_notify_consumption(
        sync_iter, QKP_CV_FIFO, CV_FIFO_CONS_SYNC_PERIOD
    )

    pto.wait_flag(pto.Pipe.V, pto.Pipe.MTE2, event_id=pTileEventId)
    with pto.if_(row_slice == 0) as row_slice_is_zero_br:
        with row_slice_is_zero_br.then_:
            qk2smSync.wait()

    buf_idx = tile_id % QKP_CV_FIFO
    base_elems = buf_idx * kTileFactor * Cube_S0 * Cube_S1

    pto.set_flag(pto.Pipe.MTE2, pto.Pipe.V, event_id=0)
    pto.wait_flag(pto.Pipe.MTE2, pto.Pipe.V, event_id=0)

    ReduceSliceTile = lambda addr: pto.alloc_tile(
        shape=[1, Vec_S0],
        dtype=pto.f32,
        valid_shape=[1, Vec_S0],
        blayout="RowMajor",
        addr=addr,
    )
    reduce_slice_rows = row_slice * Vec_S0
    reduce_row_byte_offset = reduce_slice_rows * pto.bytewidth(pto.f32)
    # Use alloc_tile(addr) directly may cause bugs here
    m1_local_max_slice = ReduceSliceTile(pto.addptr(m1_local_max.as_ptr(), reduce_slice_rows))
    l1_local_sum_slice = ReduceSliceTile(pto.addptr(l1_local_sum.as_ptr(), reduce_slice_rows))
    m2_global_max_slice = ReduceSliceTile(pto.addptr(m2_global_max.as_ptr(), reduce_slice_rows))
    l2_global_sum_slice = ReduceSliceTile(pto.addptr(l2_global_sum.as_ptr(), reduce_slice_rows))
    l1_exp_max_slice = ReduceSliceTile(pto.addptr(l1_exp_max_ififo.as_ptr(), reduce_slice_rows))
    triu = pto.alloc_tile(shape=[1, 16], dtype=pto.f16, valid_shape=[1, 1])

    row_slice_base = row_slice * Vec_S0
    row_slice_limit = row_slice_base + Vec_S0

    pto.wait_flag(pto.Pipe.MTE3, pto.Pipe.V, event_id=pTileEventId)
    with pto.if_(initFlag) as init_flag_br:
        with init_flag_br.then_:
            pto_macro_fa_softmax_dn(
                x_expT,
                qkVecTile,
                m1_local_max_slice,
                l1_local_sum_slice,
                m2_global_max_slice,
                l2_global_sum_slice,
                l1_exp_max_slice,
                triu,
                nzConvBuffer,
                s0_index,
                s1_index,
                tile_id,
                sync_iter,
                last_tile,
                pto.const(0, dtype=pto.i32),
                pto.const(Vec_S0, dtype=pto.i32),
                pto.const(Tile_S1, dtype=pto.i32),
                init=True,
                head_size=HEAD_SIZE,
                causal_mask=CAUSAL_MASK,
            )
        with init_flag_br.else_:
            pto_macro_fa_softmax_dn(
                x_expT,
                qkVecTile,
                m1_local_max_slice,
                l1_local_sum_slice,
                m2_global_max_slice,
                l2_global_sum_slice,
                l1_exp_max_slice,
                triu,
                nzConvBuffer,
                s0_index,
                s1_index,
                tile_id,
                sync_iter,
                last_tile,
                pto.const(0, dtype=pto.i32),
                pto.const(Vec_S0, dtype=pto.i32),
                pto.const(Tile_S1, dtype=pto.i32),
                init=False,
                head_size=HEAD_SIZE,
                causal_mask=CAUSAL_MASK,
            )

    should_free_qk2sm = (row_slice == kTileFactor - 1) & should_notify_consume
    with pto.if_(should_free_qk2sm) as should_free_qk2sm_br:
        with should_free_qk2sm_br.then_:
            qk2smSync.free()

    pto.set_flag(pto.Pipe.V, pto.Pipe.MTE2, event_id=pTileEventId)
    pto.set_flag(pto.Pipe.V, pto.Pipe.MTE3, event_id=0)
    pto.wait_flag(pto.Pipe.V, pto.Pipe.MTE3, event_id=0)

    should_alloc_sm2pv = (row_slice == 0) & (tile_id >= PMAT_TN_BUFFERS)
    with pto.if_(should_alloc_sm2pv) as should_alloc_sm2pv_br:
        with should_alloc_sm2pv_br.then_:
            sm2pvSync.allocate()

    GlobalPTileHalfSub = lambda ptr: pto.make_tensor_view(
        ptr,
        shape=[Cube_S1, Vec_S0],
        strides=[Cube_S0, 1],
    )
    TileDataH_Sub = lambda addr: pto.alloc_tile(
        shape=[Tile_S1, Vec_S0],
        dtype=pto.f16,
        valid_shape=[Cube_S1, Vec_S0],
        blayout="RowMajor",
        addr=addr,
    )
    p_ptr = pto.addptr(p_tile_fifo, base_elems + row_offset)

    with pto.for_(0, kTileFactor, step=1) as sub_col:
        if INTERMEDIATE_CHECK:
            NzBufRows = Cube_S1 + 1
            col_byte_offset = sub_col * NzBufRows * Vec_S0 * pto.bytewidth(pto.f16)
            p_ptr_sub = pto.addptr(p_ptr, sub_col * Cube_S1 * Cube_S0)
            pTileHalfSub = GlobalPTileHalfSub(p_ptr_sub)
            xExpSub = TileDataH_Sub(pto.addptr(nzConvBuffer.as_ptr(), sub_col * NzBufRows * Vec_S0))
            pTileHalfSubPart = pto.partition_view(
                pTileHalfSub,
                offsets=[0, 0],
                sizes=[Cube_S1, Vec_S0],
            )
            pto.tile.store(xExpSub, pTileHalfSubPart)
            pto.set_flag(pto.Pipe.MTE3, pto.Pipe.V, event_id=0)
            pto.wait_flag(pto.Pipe.MTE3, pto.Pipe.V, event_id=0)
            _ = col_byte_offset

        # Softmax vsstb already filled nzConvBuffer (NZ+1); skip ND->NZ TMOV before TINSERT.
        pto.set_flag(pto.Pipe.V, pto.Pipe.MTE3, event_id=0)
        pto.wait_flag(pto.Pipe.V, pto.Pipe.MTE3, event_id=0)

        row_offset = sub_col * Cube_S1
        col_offset = Vec_S0 * pto.get_subblock_idx()
        pto.tile.insert(nzConvBuffer, pMatTile, row_offset, col_offset)

    if INTERMEDIATE_CHECK:
        should_dump_pmax = row_slice == kTileFactor - 1
        with pto.if_(should_dump_pmax) as should_dump_pmax_br:
            with should_dump_pmax_br.then_:
                SubblockRows = Cube_S0 // VEC_CORES
                GlobalPMaxFloatSub = lambda ptr: pto.make_tensor_view(
                    ptr,
                    shape=[1, SubblockRows],
                    strides=[Cube_S0, 1],
                )
                ExpMaxSub = lambda addr: pto.alloc_tile(
                    shape=[1, SubblockRows],
                    dtype=pto.f32,
                    valid_shape=[1, SubblockRows],
                    blayout="RowMajor",
                    addr=addr,
                )
                base_elems_pmax = buf_idx * Cube_S0 + subblock_base_rows
                p_ptr_fp32 = pto.addptr(exp_max_ififo, base_elems_pmax)
                pMaxGlobal = GlobalPMaxFloatSub(p_ptr_fp32)
                l1_exp_max_rowmajor = ExpMaxSub(l1_exp_max_ififo.as_ptr())
                l1_exp_max_rowmajor = pto.tile.reshape(l1_exp_max_ififo, shape=[1, SubblockRows], blayout="RowMajor")
                pMaxGlobalPart = pto.partition_view(
                    pMaxGlobal,
                    offsets=[0, 0],
                    sizes=[1, SubblockRows],
                )
                pto.tile.store(l1_exp_max_rowmajor, pMaxGlobalPart)
        pto.set_flag(pto.Pipe.MTE3, pto.Pipe.V, event_id=0)
        pto.wait_flag(pto.Pipe.MTE3, pto.Pipe.V, event_id=0)

    should_record_sm2pv = row_slice == kTileFactor - 1
    with pto.if_(should_record_sm2pv) as should_record_sm2pv_br:
        with should_record_sm2pv_br.then_:
            sm2pvSync.record()

    pto.set_flag(pto.Pipe.MTE3, pto.Pipe.V, event_id=pTileEventId)



def compute_pv(
    *,
    S0: int,
    HEAD_SIZE: int,
    S1: int,
    Cube_S0: int,
    Cube_S1: int,
    Tile_S1: int,
    QKP_CV_FIFO: int,
    PV_CV_FIFO: int,
    CV_FIFO_CONS_SYNC_PERIOD: int,
    INTERMEDIATE_CHECK: bool,
    CAUSAL_MASK: bool,
    OUT_O_TILE_NBUFFERS: int,
    TileMatPData,
    TileMatVData,
    TilePVData,
    TileOutT,
    TSyncSM2PV,
    TSyncPV2GU,
    tile_id: int,
    sub_tile_id: int,
    pv_ub_buf_idx: int,
    p_tile_fifo,
    v,
    pv_tile_fifo,
    pv_pend_tile_fifo,
    pMatTile,
    vMatTile,
    pvAccTile,
    pvAccPendTile,
    runningOTile,
    pvPendTile,
    pvVecTile,
    svMatTileEventId,
    accTileEvtID,
    sm2pvSync,
    pv2guSync,
    blk_idx: int,
):
    Cube_HEAD = HEAD_SIZE
    assert QKP_CV_FIFO >= 1, "QKP_CV_FIFO must be >= 1"
    assert Tile_S1 % Cube_S1 == 0, "TILE_S1 must be divisible by CUBE_S1"
    kTileFactor = Tile_S1 // Cube_S1
    s0_index = blk_idx * Cube_S0
    s1_index = tile_id * Tile_S1 + sub_tile_id * Cube_S1
    sync_iter = tile_id
    should_wait_consume = should_wait_consumption(
        sync_iter, QKP_CV_FIFO, CV_FIFO_CONS_SYNC_PERIOD
    )
    next_will_be_skipped = ((s1_index + Cube_S1) > s0_index) & CAUSAL_MASK

    def emit_pv_main_path():
        GlobalVT = lambda ptr: pto.make_tensor_view(ptr, shape=[Cube_S1, HEAD_SIZE], strides=[HEAD_SIZE, 1])

        pto.wait_flag(pto.Pipe.MTE1, pto.Pipe.MTE2, event_id=svMatTileEventId)

        vLoad = GlobalVT(pto.addptr(v, s1_index * HEAD_SIZE))
        pto.tile.load(vLoad, vMatTile)

        pReadyHook = PReadyHook(sm2pvSync, sub_tile_id == 0)

        pto.set_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=1)

        with pto.if_(sub_tile_id == 0) as sub_tile_is_zero_br:
            with sub_tile_is_zero_br.then_:
                pto.wait_flag(pto.Pipe.FIX, pto.Pipe.M, event_id=accTileEvtID)

        preBTExtOpReadyHook = PreBTExtOpReadyHook()
        sm2pvFreeHook = Sm2PvFreeHook(sm2pvSync, sub_tile_id == kTileFactor - 1)
        with pto.if_(sub_tile_id == 0) as sub_tile_is_zero_br:
            with sub_tile_is_zero_br.then_:
                pto_macro_matmul(
                    Cube_M=Cube_S0,
                    Tile_K=Cube_S1,
                    Cube_N=Cube_HEAD,
                    aMatTile=pMatTile,
                    bMatTile=vMatTile,
                    cAccTile=pvAccTile,
                    accMode=AccMode.Init,
                    preATExtOpHook=pReadyHook,
                    preBTExtOpHook=preBTExtOpReadyHook,
                    postTExtOpHook=sm2pvFreeHook,
                )
            with sub_tile_is_zero_br.else_:
                pto_macro_matmul(
                    Cube_M=Cube_S0,
                    Tile_K=Cube_S1,
                    Cube_N=Cube_HEAD,
                    aMatTile=pMatTile,
                    bMatTile=vMatTile,
                    cAccTile=pvAccTile,
                    accMode=AccMode.Acc,
                    preATExtOpHook=pReadyHook,
                    preBTExtOpHook=preBTExtOpReadyHook,
                    postTExtOpHook=sm2pvFreeHook,
                )
        pto.set_flag(pto.Pipe.MTE1, pto.Pipe.MTE2, event_id=svMatTileEventId)

        should_finalize_pv = (sub_tile_id == kTileFactor - 1) | next_will_be_skipped
        with pto.if_(should_finalize_pv) as should_finalize_pv_br:
            with should_finalize_pv_br.then_:
                pto.set_flag(pto.Pipe.M, pto.Pipe.FIX, event_id=0)
                pto.wait_flag(pto.Pipe.M, pto.Pipe.FIX, event_id=0)

                with pto.if_(should_wait_consume) as should_wait_consume_br:
                    with should_wait_consume_br.then_:
                        pv2guSync.allocate()

                with pto.if_(tile_id == 0) as tile_is_zero_br:
                    with tile_is_zero_br.then_:
                        pto.wait_intra_flag(pto.Pipe.FIX, RUNNING_O_AVALIABLE)
                        pto.wait_intra_flag(pto.Pipe.FIX, RUNNING_O_AVALIABLE + 16)
                        pto.tile.mov(pvAccTile, runningOTile)
                    with tile_is_zero_br.else_:
                        pto.tile.mov(pvAccTile, pvVecTile[pv_ub_buf_idx])

                if INTERMEDIATE_CHECK:
                    GlobalDataPV = lambda ptr: pto.make_tensor_view(
                        ptr,
                        shape=[Cube_S0, HEAD_SIZE],
                        strides=[HEAD_SIZE, 1],
                    )
                    buf_idx_pv = tile_id % PV_CV_FIFO
                    base_elems_pv = buf_idx_pv * Cube_S0 * HEAD_SIZE
                    pvGlobalTile = GlobalDataPV(pto.addptr(pv_tile_fifo, base_elems_pv))
                    pvGlobalTilePart = pto.partition_view(
                        pvGlobalTile,
                        offsets=[0, 0],
                        sizes=[Cube_S0, HEAD_SIZE],
                    )
                    pto.tile.store(pvAccTile, pvGlobalTilePart)

                pto.set_flag(pto.Pipe.FIX, pto.Pipe.M, event_id=accTileEvtID)
                pv2guSync.record()

    if CAUSAL_MASK:
        with pto.if_(s1_index > s0_index) as s1_gt_s0_br:
            with s1_gt_s0_br.then_:
                should_notify_consume = should_notify_consumption(
                    sync_iter, QKP_CV_FIFO, CV_FIFO_CONS_SYNC_PERIOD
                )
                with pto.if_(sub_tile_id == 0) as sub_tile_is_zero_br:
                    with sub_tile_is_zero_br.then_:
                        sm2pvSync.wait()
                        pto.wait_flag(pto.Pipe.V, pto.Pipe.M, event_id=1)
                should_free_sm2pv = (sub_tile_id == kTileFactor - 1) & should_notify_consume
                with pto.if_(should_free_sm2pv) as should_free_sm2pv_br:
                    with should_free_sm2pv_br.then_:
                        sm2pvSync.free()
            with s1_gt_s0_br.else_:
                emit_pv_main_path()
    else:
        emit_pv_main_path()

def compute_gu(
    *,
    S0: int,
    HEAD_SIZE: int,
    S1: int,
    Cube_S0: int,
    Tile_S1: int,
    PV_CV_FIFO: int,
    CV_FIFO_CONS_SYNC_PERIOD: int,
    INTERMEDIATE_CHECK: bool,
    CAUSAL_MASK: bool,
    SRC_VEC_TN_BUFFERS: int,
    OUT_O_TILE_NBUFFERS: int,
    TileOutT,
    ReduceTileF_T,
    TSyncPV2GU,
    tile_id: int,
    num_tiles: int,
    pv_tile_fifo,
    pv_pend_tile_fifo,
    o_out,
    o_parts_out,
    runningOTile,
    pvVecTile,
    pvPendTile,
    l1_exp_max_ififo,
    l2_global_sum,
    guEventId,
    pv2guSync,
    tmpOTile,
):
    Vec_S0 = Cube_S0 // VEC_CORES

    GlobalDataPV_VEC = lambda ptr: pto.make_tensor_view(
        ptr,
        shape=[Vec_S0, HEAD_SIZE],
        strides=[HEAD_SIZE, 1],
    )

    should_notify_consume = should_notify_consumption(
        tile_id, PV_CV_FIFO, CV_FIFO_CONS_SYNC_PERIOD
    )
    buf_idx = tile_id % PV_CV_FIFO
    base_elems = buf_idx * Cube_S0 * HEAD_SIZE

    subblock_base_rows = (Cube_S0 // VEC_CORES) * pto.get_subblock_idx()
    pv_out_ptr = pto.addptr(pv_tile_fifo, base_elems + subblock_base_rows * HEAD_SIZE)
    pvGlobalVec = GlobalDataPV_VEC(pv_out_ptr)

    pv2guSync.wait()
    pto.wait_flag(pto.Pipe.V, pto.Pipe.MTE2, event_id=guEventId)

    with pto.if_(tile_id > 0) as tile_gt_zero_br:
        with tile_gt_zero_br.then_:
            with pto.if_(tile_id < num_tiles - 1) as not_last_br:
                with not_last_br.then_:
                    gu_update_stage(runningOTile, pvVecTile, l1_exp_max_ififo, tmpOTile)
                with not_last_br.else_:
                    gu_last_stage(
                        runningOTile, pvVecTile, l1_exp_max_ififo, l2_global_sum, tmpOTile
                    )
        with tile_gt_zero_br.else_:
            if CAUSAL_MASK:
                with pto.if_(tile_id == num_tiles - 1) as single_last_br:
                    with single_last_br.then_:
                        gu_single_last_stage(runningOTile, l2_global_sum)

    pto.set_flag(pto.Pipe.V, pto.Pipe.MTE2, event_id=guEventId)
    with pto.if_(should_notify_consume) as should_notify_consume_br:
        with should_notify_consume_br.then_:
            pv2guSync.free()

    with pto.if_(tile_id == num_tiles - 1) as tile_is_last_br:
        with tile_is_last_br.then_:
            pto.set_flag(pto.Pipe.V, pto.Pipe.MTE3, event_id=0)
            pto.wait_flag(pto.Pipe.V, pto.Pipe.MTE3, event_id=0)
            GlobalOutT = lambda ptr: pto.make_tensor_view(
                ptr,
                shape=[Vec_S0, HEAD_SIZE],
                strides=[HEAD_SIZE, 1],
            )
            outGlobal = GlobalOutT(pto.addptr(o_out, subblock_base_rows * HEAD_SIZE))
            outPart = pto.partition_view(
                outGlobal,
                offsets=[0, 0],
                sizes=[Vec_S0, HEAD_SIZE],
            )
            pto.tile.store(runningOTile, outPart)
            pto.set_intra_flag(pto.Pipe.MTE3, RUNNING_O_AVALIABLE)


@pto.jit(target="a5", mode="explicit")
def runTFA(
    ffts_addr: pto.ptr(pto.ui64, "gm"),
    q: pto.ptr(pto.f16, "gm"),
    k: pto.ptr(pto.f16, "gm"),
    v: pto.ptr(pto.f16, "gm"),
    p_tile_fifo: pto.ptr(pto.f16, "gm"),
    exp_max_ififo: pto.ptr(pto.f32, "gm"),
    o_out: pto.ptr(pto.f32, "gm"),
    o_parts_out: pto.ptr(pto.f32, "gm"),
    qk_tile_fifo: pto.ptr(pto.f32, "gm"),
    pv_tile_fifo: pto.ptr(pto.f32, "gm"),
    pv_pend_tile_fifo: pto.ptr(pto.f32, "gm"),
    cv_comm_buf: pto.ptr(pto.ui8, "gm"),
    profile_buf: pto.ptr(pto.ui8, "gm"),
    *,
    S0: pto.const_expr = 128,
    HEAD_SIZE: pto.const_expr = 128,
    S1: pto.const_expr = 1024,
    CUBE_S0: pto.const_expr = 128,
    CUBE_S1: pto.const_expr = kFaCubeS1,
    TILE_S1: pto.const_expr = kFaTileS1,
    QK_PRELOAD: pto.const_expr = kFaQkPreload,
    CV_FIFO_SIZE: pto.const_expr = kFaCvFifoSize,
    INTERMEDIATE_CHECK: pto.const_expr = False,
    CAUSAL_MASK: pto.const_expr = False,
    CV_FIFO_CONS_SYNC_PERIOD: pto.const_expr = kFaCvFifoConsSyncPeriod,
):
    # S0 (rows total), Cube_S0 (per-block rows), S1 (cols), HEAD_SIZE (inner)
    Cube_S0 = CUBE_S0
    logical_block_count = S0 // Cube_S0
    launch_block_count = (
        logical_block_count
        if logical_block_count < kFaLaunchCoreCount
        else kFaLaunchCoreCount
    )
    Cube_S1 = CUBE_S1
    Tile_S1 = TILE_S1
    assert Tile_S1 % Cube_S1 == 0, "TILE_S1 must be divisible by CUBE_S1"
    kTileFactor = Tile_S1 // Cube_S1
    Cube_HEAD = HEAD_SIZE
    Vec_S0 = Cube_S0 // VEC_CORES // kTileFactor
    VecGuRows = Cube_S0 // VEC_CORES
    assert Cube_S0 % (VEC_CORES * kTileFactor) == 0, (
        "Vec rows must divide evenly across tile slices"
    )


    # ------------------------------------------------------------------------------
    # Tuning knobs (pipeline)
    #
    # qkPreloadNum controls how many (QK -> P) tiles we warm up before entering the steady-state loop.
    # - Larger preload improves overlap (Cube/VEC concurrency) for long S1.
    # - Larger preload increases FIFO footprint (qkGlobalTensorNBuffers / pvGlobalTensorNBuffers /
    # guGlobalTensorNBuffers).
    #
    # Buffer counts for optional double-buffering (default 1)
    # - srcVecTNBuffers/xexpVecTNBuffers: Vec ping-pong for QK load and x_exp output
    # - *MatTNBuffers: L1 ping-pong for Cube stage (K/P/V)
    # Keep these small (1-2) unless you have measured stall bubbles that require deeper buffering.
    # ------------------------------------------------------------------------------
    qkPreloadNum = QK_PRELOAD
    srcVecTNBuffers = 2
    xexpVecTNBuffers = 2
    outOTileNBuffers = 2
    qMatTNBuffers = 1
    kMatTNBuffers = 2
    pMatTNBuffers = 2
    vMatTNBuffers = 2
    qkp_tile_fifo_size = CV_FIFO_SIZE
    pv_tile_fifo_size = CV_FIFO_SIZE
    assert qkPreloadNum >= 1, "qkPreloadNum must be >= 1"
    assert CV_FIFO_CONS_SYNC_PERIOD >= 1, "CV_FIFO_CONS_SYNC_PERIOD must be >= 1"
    assert (qkPreloadNum > 1) or (kTileFactor == 1), (
        "qkPreloadNum must be > 1 unless kTileFactor == 1"
    )
    assert qkPreloadNum <= pMatTNBuffers, (
        "USE_UB_TO_L1_PATH requires qkPreloadNum <= pMatTNBuffers (2) to avoid buffer races. "
        "Use --qk-preload 2 when running with UB mode enabled."
    )

    # Define tile types for first QK matmul / second PV matmul.
    TileMatQData = lambda: pto.alloc_tile(shape=[Cube_S0, Cube_HEAD], dtype=pto.f16, memory_space=pto.MemorySpace.MAT, valid_shape=[Cube_S0, Cube_HEAD], blayout="ColMajor", slayout="RowMajor")
    TileMatKData = lambda: pto.alloc_tile(shape=[Cube_S1, Cube_HEAD], dtype=pto.f16, memory_space=pto.MemorySpace.MAT, valid_shape=[Cube_S1, Cube_HEAD], blayout="ColMajor", slayout="RowMajor")
    TileQKData = lambda: pto.alloc_tile(shape=[Cube_S0, Cube_S1], dtype=pto.f32, memory_space=pto.MemorySpace.ACC, valid_shape=[Cube_S0, Cube_S1])
    TileMatPData = lambda: pto.alloc_tile(shape=[Cube_S0, Tile_S1], dtype=pto.f32, memory_space=pto.MemorySpace.MAT, valid_shape=[Cube_S0, Tile_S1], blayout="ColMajor", slayout="RowMajor")
    TileMatVData = lambda: pto.alloc_tile(shape=[Cube_S1, Cube_HEAD], dtype=pto.f16, memory_space=pto.MemorySpace.MAT, valid_shape=[Cube_S1, Cube_HEAD], blayout="ColMajor", slayout="RowMajor")
    TilePVData = lambda: pto.alloc_tile(shape=[Cube_S0, Cube_HEAD], dtype=pto.f32, memory_space=pto.MemorySpace.ACC, valid_shape=[Cube_S0, Cube_HEAD])

    qMatTile = [TileMatQData() for _ in range(qMatTNBuffers)]
    kMatTile = [TileMatKData() for _ in range(kMatTNBuffers)]
    qkAccTile = TileQKData()
    pMatTile = [TileMatPData() for _ in range(pMatTNBuffers)]
    vMatTile = [TileMatVData() for _ in range(vMatTNBuffers)]
    pvAccPendTile = TilePVData()
    pvAccCurrTile = TilePVData()

    # allocate_cube_tile_buffers(qMatTile, kMatTile, pMatTile, vMatTile)

    # TASSIGN(pvAccPendTile, 0x20000u);
    # TASSIGN(pvAccCurrTile, 0x30000u);

    # Define tile types for FA softmax P computation. UB offsets for softmax tiles
    # Define per-tile vector tiles sized to Cube_S1
    # DN layout version
    TileDataF_T = lambda: pto.alloc_tile(shape=[Cube_S0, Tile_S1], dtype=pto.f32, valid_shape=[Cube_S0, Tile_S1], blayout="RowMajor")
    TileDataH_T = lambda: pto.alloc_tile(shape=[Cube_S0, Tile_S1], dtype=pto.f16, valid_shape=[Cube_S0, Tile_S1], blayout="RowMajor")
    SubblockRows = Cube_S0 // VEC_CORES
    # Reduce tiles cover one vector core's rows (Cube_S0 / VEC_CORES); slices are extracted per row_slice
    ReduceTileF_T = lambda: pto.alloc_tile(shape=[1, SubblockRows], dtype=pto.f32, valid_shape=[1, SubblockRows], blayout="RowMajor")

    NzBufRows = Cube_S1 + 1
    TileDataH_NZ_T = lambda: pto.alloc_tile(shape=[NzBufRows, Vec_S0], dtype=pto.f16, valid_shape=[NzBufRows, Vec_S0], blayout="ColMajor", slayout="RowMajor")

    qkVecTile = [TileDataF_T() for _ in range(srcVecTNBuffers)]
    m1_local_max = ReduceTileF_T()
    l1_local_sum = ReduceTileF_T()
    m2_global_max = ReduceTileF_T()
    l2_global_sum = ReduceTileF_T()
    l1_exp_max_ififo = [ReduceTileF_T() for _ in range(qkp_tile_fifo_size)]
    x_expT = [TileDataH_T() for _ in range(xexpVecTNBuffers)]
    nzConvBuffer = [TileDataH_NZ_T() for _ in range(xexpVecTNBuffers)]

    TileOutGuT = lambda: pto.alloc_tile(shape=[VecGuRows, Cube_HEAD], dtype=pto.f32, valid_shape=[VecGuRows, Cube_HEAD], blayout="RowMajor")
    pvVecTile = [TileOutGuT() for _ in range(outOTileNBuffers)]
    runningOTile = TileOutGuT()

    # allocate_vec_tile_buffers(qkVecTile, m1_local_max, l1_local_sum, m2_global_max, l2_global_sum, l1_exp_max_ififo, x_expT, pvVecTile, runningOTile)

    # TASSIGN(nzConvBuffer[0], nzBufOffset);
    # TASSIGN(nzConvBuffer[1], nzBufOffset - nzBufSize);

    p_fifo_block_stride = qkp_tile_fifo_size * Cube_S0 * Tile_S1
    p_max_fifo_block_stride = qkp_tile_fifo_size * Cube_S0
    qk_fifo_block_stride = p_fifo_block_stride
    pv_fifo_block_stride = pv_tile_fifo_size * Cube_S0 * Cube_HEAD

    # QK uses L0C->UB (TMOV); Vec must wait PIPE_V, not PIPE_MTE2 (GM path).
    qk2smSync = TSync_Custom(SyncOpType.TMOV_C2UB, SyncOpType.TLOAD, BUF0_QK_READY)
    sm2pvSync = TSync_Custom(SyncOpType.TINSERT_V2L1, SyncOpType.TLOAD, BUF1_SM_READY)
    pv2guSync = TSync_Custom(SyncOpType.TMOV_C2UB, SyncOpType.TLOAD, UPDATE_READY)
    use_cv_comm = (not INTERMEDIATE_CHECK) and (launch_block_count >= kFaLaunchCoreCount)
    pvAccTileEvtID = 2
    physical_block_idx = 0

    # if DAV_CUBE:
    pto.set_flag(pto.Pipe.M, pto.Pipe.MTE1, event_id=0)
    pto.set_flag(pto.Pipe.M, pto.Pipe.MTE1, event_id=1)
    pto.set_flag(pto.Pipe.MTE1, pto.Pipe.MTE2, event_id=0)
    pto.set_flag(pto.Pipe.MTE1, pto.Pipe.MTE2, event_id=1)
    pto.set_flag(pto.Pipe.MTE1, pto.Pipe.MTE2, event_id=2)
    pto.set_flag(pto.Pipe.MTE1, pto.Pipe.MTE2, event_id=3)
    pto.set_flag(pto.Pipe.FIX, pto.Pipe.M, event_id=0)
    pto.set_flag(pto.Pipe.FIX, pto.Pipe.M, event_id=1)
    pto.set_flag(pto.Pipe.FIX, pto.Pipe.M, event_id=2)
    # if DAV_VEC:
    pto.set_flag(pto.Pipe.V, pto.Pipe.MTE2, event_id=0)
    pto.set_flag(pto.Pipe.V, pto.Pipe.MTE2, event_id=1)
    pto.set_flag(pto.Pipe.MTE3, pto.Pipe.V, event_id=0)
    pto.set_flag(pto.Pipe.MTE3, pto.Pipe.V, event_id=1)

    physical_comm_slot = (
        {"op": "TSYNC_CVID", "physical_block_idx": physical_block_idx, "cv_comm_buf": cv_comm_buf}
        if use_cv_comm
        else physical_block_idx
    )

    with pto.for_(
        physical_block_idx,
        logical_block_count,
        step=launch_block_count,
    ) as logical_block_idx:
        # const uint64_t tStart = get_sys_cnt();
        # assign_running_acc_tile(qkAccTile, 0);

        block_offset_rows = logical_block_idx * Cube_S0
        comm_slot = physical_comm_slot if use_cv_comm else logical_block_idx

        # __gm__ uint64_t *profile_entry = nullptr;
        # if (profile_buf != nullptr) {
        #     std::size_t profile_block_base = static_cast<std::size_t>(logical_block_idx) * kFaProfileBytesPerBlock;
        #     std::size_t profile_offset = profile_block_base;
        #     if constexpr (DAV_VEC) {
        #         profile_offset +=
        #             (static_cast<std::size_t>(get_subblockid()) + 1U) * 1024U; // vec subblock 0/1 use 2nd/3rd KB
        #     }
        #     profile_entry = reinterpret_cast<__gm__ uint64_t *>(profile_buf + profile_offset);
        #     profile_entry[0] = tStart;
        # }

        q_block = pto.addptr(q, block_offset_rows * Cube_HEAD)
        p_tile_fifo_block = pto.addptr(p_tile_fifo, comm_slot * p_fifo_block_stride)
        exp_max_ififo_block = pto.addptr(exp_max_ififo, comm_slot * p_max_fifo_block_stride)
        o_out_block = pto.addptr(o_out, block_offset_rows * Cube_HEAD)
        o_parts_block = pto.addptr(o_parts_out, block_offset_rows * Cube_HEAD)
        qk_tile_fifo_block = pto.addptr(qk_tile_fifo, comm_slot * qk_fifo_block_stride)
        pv_tile_fifo_block = pto.addptr(pv_tile_fifo, comm_slot * pv_fifo_block_stride)
        pv_pend_tile_fifo_block = pto.addptr(pv_pend_tile_fifo, comm_slot * pv_fifo_block_stride)

        num_tiles_s1 = S1 // Tile_S1
        if CAUSAL_MASK:
            num_tiles_s1 = 1 + ((logical_block_idx * Cube_S0) // Tile_S1)

        p_gu_src_pingpong_id = 0  # shared ping-pong for softmax vec tiles, pv output tiles, and GU input tiles
        k_src_pingpong_id = 0  # separate ping-pong for K tiles
        pv_src_pingpong_id = 0  # separate ping-pong for P V tiles

        qkAccTileEvtID = 0
        has_next_logical_block = (logical_block_idx + launch_block_count) < logical_block_count

        # QK and P pre-computation (tile_id based)
        preload_limit = _scalar_min_or_static(qkPreloadNum, num_tiles_s1)
        with pto.for_(0, preload_limit, step=1) as preload_tile:
            # if DAV_CUBE:
            with pto.for_(0, kTileFactor, step=1) as sub_tile:
                # ``fa_dn.cpp`` rotates the running ACC tile here via
                # ``assign_running_acc_tile(qkAccTile)`` before each QK.
                # qkAccTileEvtID = assign_running_acc_tile(qkAccTile);
                tile_buf_idx = preload_tile % srcVecTNBuffers
                compute_qk(
                    S0=S0,
                    HEAD_SIZE=Cube_HEAD,
                    S1=S1,
                    Cube_S0=Cube_S0,
                    Cube_S1=Cube_S1,
                    Tile_S1=Tile_S1,
                    QKP_CV_FIFO=qkp_tile_fifo_size,
                    CV_FIFO_CONS_SYNC_PERIOD=CV_FIFO_CONS_SYNC_PERIOD,
                    INTERMEDIATE_CHECK=INTERMEDIATE_CHECK,
                    CAUSAL_MASK=CAUSAL_MASK,
                    SRC_VEC_TN_BUFFERS=srcVecTNBuffers,
                    TileMatQData=TileMatQData,
                    TileMatKData=TileMatKData,
                    TileQKData=TileQKData,
                    TileQKVecData=TileDataF_T,
                    TSyncQK2SM=qk2smSync,
                    tile_id=preload_tile,
                    sub_tile_id=sub_tile,
                    ub_buf_idx=tile_buf_idx,
                    q=q_block,
                    k=k,
                    qk_tile_fifo=qk_tile_fifo_block,
                    qMatTile=qMatTile[0],
                    kMatTile=kMatTile[k_src_pingpong_id % kMatTNBuffers],
                    qkAccTile=qkAccTile,
                    qkVecTile=qkVecTile[tile_buf_idx],
                    qkMatTileEventId=k_src_pingpong_id % kMatTNBuffers,
                    accTileEvtID=qkAccTileEvtID,
                    qk2smSync=qk2smSync,
                    blk_idx=logical_block_idx,
                )
                k_src_pingpong_id += 1

            # if DAV_VEC:
            with pto.for_(0, kTileFactor, step=1) as row_slice:
                tile_buf_idx = preload_tile % srcVecTNBuffers
                compute_p(
                    S0=S0,
                    HEAD_SIZE=Cube_HEAD,
                    S1=S1,
                    Cube_S0=Cube_S0,
                    Cube_S1=Cube_S1,
                    Tile_S1=Tile_S1,
                    QKP_CV_FIFO=qkp_tile_fifo_size,
                    CV_FIFO_CONS_SYNC_PERIOD=CV_FIFO_CONS_SYNC_PERIOD,
                    INTERMEDIATE_CHECK=INTERMEDIATE_CHECK,
                    CAUSAL_MASK=CAUSAL_MASK,
                    PMAT_TN_BUFFERS=pMatTNBuffers,
                    TileDataF_T=TileDataF_T,
                    TileDataH_T=TileDataH_T,
                    TileDataH_NZ_T=TileDataH_NZ_T,
                    ReduceTileF_T=ReduceTileF_T,
                    TileMatPData=TileMatPData,
                    TSyncQK2SM=qk2smSync,
                    TSyncSM2PV=sm2pvSync,
                    tile_id=preload_tile,
                    row_slice=row_slice,
                    qk_tile_fifo=qk_tile_fifo_block,
                    p_tile_fifo=p_tile_fifo_block,
                    exp_max_ififo=exp_max_ififo_block,
                    qkVecTile=qkVecTile[tile_buf_idx],
                    x_expT=x_expT[p_gu_src_pingpong_id % xexpVecTNBuffers],
                    m1_local_max=m1_local_max,
                    l1_local_sum=l1_local_sum,
                    m2_global_max=m2_global_max,
                    l2_global_sum=l2_global_sum,
                    l1_exp_max_ififo=l1_exp_max_ififo[preload_tile % qkp_tile_fifo_size],
                    pMatTile=pMatTile[preload_tile % pMatTNBuffers],
                    nzConvBuffer=nzConvBuffer[p_gu_src_pingpong_id % xexpVecTNBuffers],
                    pTileEventId=p_gu_src_pingpong_id % xexpVecTNBuffers,
                    qk2smSync=qk2smSync,
                    sm2pvSync=sm2pvSync,
                    blk_idx=logical_block_idx,
                    num_tiles=num_tiles_s1,
                )
                p_gu_src_pingpong_id += 1

        steady_tile_end = _scalar_max_or_static(num_tiles_s1 - qkPreloadNum, 0)
        with pto.for_(0, steady_tile_end, step=1) as tile_id:
            next_qk_tile = tile_id + qkPreloadNum
            # qkAccTileEvtID = assign_running_acc_tile(qkAccTile);
            with pto.for_(0, kTileFactor, step=1) as sub_tile:

                # if DAV_CUBE:
                tile_buf_idx = next_qk_tile % srcVecTNBuffers
                compute_qk(
                    S0=S0,
                    HEAD_SIZE=Cube_HEAD,
                    S1=S1,
                    Cube_S0=Cube_S0,
                    Cube_S1=Cube_S1,
                    Tile_S1=Tile_S1,
                    QKP_CV_FIFO=qkp_tile_fifo_size,
                    CV_FIFO_CONS_SYNC_PERIOD=CV_FIFO_CONS_SYNC_PERIOD,
                    INTERMEDIATE_CHECK=INTERMEDIATE_CHECK,
                    CAUSAL_MASK=CAUSAL_MASK,
                    SRC_VEC_TN_BUFFERS=srcVecTNBuffers,
                    TileMatQData=TileMatQData,
                    TileMatKData=TileMatKData,
                    TileQKData=TileQKData,
                    TileQKVecData=TileDataF_T,
                    TSyncQK2SM=qk2smSync,
                    tile_id=next_qk_tile,
                    sub_tile_id=sub_tile,
                    ub_buf_idx=tile_buf_idx,
                    q=q_block,
                    k=k,
                    qk_tile_fifo=qk_tile_fifo_block,
                    qMatTile=qMatTile[0],
                    kMatTile=kMatTile[k_src_pingpong_id % kMatTNBuffers],
                    qkAccTile=qkAccTile,
                    qkVecTile=qkVecTile[tile_buf_idx],
                    qkMatTileEventId=k_src_pingpong_id % kMatTNBuffers,
                    accTileEvtID=qkAccTileEvtID,
                    qk2smSync=qk2smSync,
                    blk_idx=logical_block_idx,
                )
                k_src_pingpong_id += 1

                # if DAV_VEC:
                tile_buf_idx = next_qk_tile % srcVecTNBuffers
                compute_p(
                    S0=S0,
                    HEAD_SIZE=Cube_HEAD,
                    S1=S1,
                    Cube_S0=Cube_S0,
                    Cube_S1=Cube_S1,
                    Tile_S1=Tile_S1,
                    QKP_CV_FIFO=qkp_tile_fifo_size,
                    CV_FIFO_CONS_SYNC_PERIOD=CV_FIFO_CONS_SYNC_PERIOD,
                    INTERMEDIATE_CHECK=INTERMEDIATE_CHECK,
                    CAUSAL_MASK=CAUSAL_MASK,
                    PMAT_TN_BUFFERS=pMatTNBuffers,
                    TileDataF_T=TileDataF_T,
                    TileDataH_T=TileDataH_T,
                    TileDataH_NZ_T=TileDataH_NZ_T,
                    ReduceTileF_T=ReduceTileF_T,
                    TileMatPData=TileMatPData,
                    TSyncQK2SM=qk2smSync,
                    TSyncSM2PV=sm2pvSync,
                    tile_id=next_qk_tile,
                    row_slice=sub_tile,
                    qk_tile_fifo=qk_tile_fifo_block,
                    p_tile_fifo=p_tile_fifo_block,
                    exp_max_ififo=exp_max_ififo_block,
                    qkVecTile=qkVecTile[tile_buf_idx],
                    x_expT=x_expT[p_gu_src_pingpong_id % xexpVecTNBuffers],
                    m1_local_max=m1_local_max,
                    l1_local_sum=l1_local_sum,
                    m2_global_max=m2_global_max,
                    l2_global_sum=l2_global_sum,
                    l1_exp_max_ififo=l1_exp_max_ififo[next_qk_tile % qkp_tile_fifo_size],
                    pMatTile=pMatTile[next_qk_tile % pMatTNBuffers],
                    nzConvBuffer=nzConvBuffer[p_gu_src_pingpong_id % xexpVecTNBuffers],
                    pTileEventId=p_gu_src_pingpong_id % xexpVecTNBuffers,
                    qk2smSync=qk2smSync,
                    sm2pvSync=sm2pvSync,
                    blk_idx=logical_block_idx,
                    num_tiles=num_tiles_s1,
                )
                p_gu_src_pingpong_id += 1

                # if DAV_CUBE:
                pvPendTile = pvVecTile[(tile_id + 1) % outOTileNBuffers]
                compute_pv(
                    S0=S0,
                    HEAD_SIZE=Cube_HEAD,
                    S1=S1,
                    Cube_S0=Cube_S0,
                    Cube_S1=Cube_S1,
                    Tile_S1=Tile_S1,
                    QKP_CV_FIFO=qkp_tile_fifo_size,
                    PV_CV_FIFO=pv_tile_fifo_size,
                    CV_FIFO_CONS_SYNC_PERIOD=CV_FIFO_CONS_SYNC_PERIOD,
                    INTERMEDIATE_CHECK=INTERMEDIATE_CHECK,
                    CAUSAL_MASK=CAUSAL_MASK,
                    OUT_O_TILE_NBUFFERS=outOTileNBuffers,
                    TileMatPData=TileMatPData,
                    TileMatVData=TileMatVData,
                    TilePVData=TilePVData,
                    TileOutT=TileOutGuT,
                    TSyncSM2PV=sm2pvSync,
                    TSyncPV2GU=pv2guSync,
                    tile_id=tile_id,
                    sub_tile_id=sub_tile,
                    pv_ub_buf_idx=tile_id % outOTileNBuffers,
                    p_tile_fifo=p_tile_fifo_block,
                    v=v,
                    pv_tile_fifo=pv_tile_fifo_block,
                    pv_pend_tile_fifo=pv_pend_tile_fifo_block,
                    pMatTile=pMatTile[pv_src_pingpong_id % pMatTNBuffers],
                    vMatTile=vMatTile[pv_src_pingpong_id % vMatTNBuffers],
                    pvAccTile=pvAccCurrTile,
                    pvAccPendTile=pvAccPendTile,
                    runningOTile=runningOTile,
                    pvPendTile=pvPendTile,
                    pvVecTile=pvVecTile,
                    svMatTileEventId=pv_src_pingpong_id % vMatTNBuffers,
                    accTileEvtID=pvAccTileEvtID,
                    sm2pvSync=sm2pvSync,
                    pv2guSync=pv2guSync,
                    blk_idx=logical_block_idx,
                )
                pv_src_pingpong_id += 1

            # if DAV_VEC:
            pvPendTile = pvVecTile[(tile_id + 1) % outOTileNBuffers]
            compute_gu(
                S0=S0,
                HEAD_SIZE=Cube_HEAD,
                S1=S1,
                Cube_S0=Cube_S0,
                Tile_S1=Tile_S1,
                PV_CV_FIFO=pv_tile_fifo_size,
                CV_FIFO_CONS_SYNC_PERIOD=CV_FIFO_CONS_SYNC_PERIOD,
                INTERMEDIATE_CHECK=INTERMEDIATE_CHECK,
                CAUSAL_MASK=CAUSAL_MASK,
                SRC_VEC_TN_BUFFERS=srcVecTNBuffers,
                OUT_O_TILE_NBUFFERS=outOTileNBuffers,
                TileOutT=TileOutGuT,
                ReduceTileF_T=ReduceTileF_T,
                TSyncPV2GU=pv2guSync,
                tile_id=tile_id,
                num_tiles=num_tiles_s1,
                pv_tile_fifo=pv_tile_fifo_block,
                pv_pend_tile_fifo=pv_pend_tile_fifo_block,
                o_out=o_out_block,
                o_parts_out=o_parts_block,
                runningOTile=runningOTile,
                pvVecTile=pvVecTile[tile_id % outOTileNBuffers],
                pvPendTile=pvPendTile,
                l1_exp_max_ififo=l1_exp_max_ififo[tile_id % qkp_tile_fifo_size],
                l2_global_sum=l2_global_sum,
                guEventId=tile_id % outOTileNBuffers,
                pv2guSync=pv2guSync,
                tmpOTile=tmpOTile,
            )
            p_gu_src_pingpong_id += 1

        with pto.for_(steady_tile_end, num_tiles_s1, step=1) as tile_id:
            with pto.for_(0, kTileFactor, step=1) as sub_tile:

                # if DAV_CUBE:
                pvPendTile = pvVecTile[(tile_id + 1) % outOTileNBuffers]
                compute_pv(
                    S0=S0,
                    HEAD_SIZE=Cube_HEAD,
                    S1=S1,
                    Cube_S0=Cube_S0,
                    Cube_S1=Cube_S1,
                    Tile_S1=Tile_S1,
                    QKP_CV_FIFO=qkp_tile_fifo_size,
                    PV_CV_FIFO=pv_tile_fifo_size,
                    CV_FIFO_CONS_SYNC_PERIOD=CV_FIFO_CONS_SYNC_PERIOD,
                    INTERMEDIATE_CHECK=INTERMEDIATE_CHECK,
                    CAUSAL_MASK=CAUSAL_MASK,
                    OUT_O_TILE_NBUFFERS=outOTileNBuffers,
                    TileMatPData=TileMatPData,
                    TileMatVData=TileMatVData,
                    TilePVData=TilePVData,
                    TileOutT=TileOutGuT,
                    TSyncSM2PV=sm2pvSync,
                    TSyncPV2GU=pv2guSync,
                    tile_id=tile_id,
                    sub_tile_id=sub_tile,
                    pv_ub_buf_idx=tile_id % outOTileNBuffers,
                    p_tile_fifo=p_tile_fifo_block,
                    v=v,
                    pv_tile_fifo=pv_tile_fifo_block,
                    pv_pend_tile_fifo=pv_pend_tile_fifo_block,
                    pMatTile=pMatTile[pv_src_pingpong_id % pMatTNBuffers],
                    vMatTile=vMatTile[pv_src_pingpong_id % vMatTNBuffers],
                    pvAccTile=pvAccCurrTile,
                    pvAccPendTile=pvAccPendTile,
                    runningOTile=runningOTile,
                    pvPendTile=pvPendTile,
                    pvVecTile=pvVecTile,
                    svMatTileEventId=pv_src_pingpong_id % vMatTNBuffers,
                    accTileEvtID=pvAccTileEvtID,
                    sm2pvSync=sm2pvSync,
                    pv2guSync=pv2guSync,
                    blk_idx=logical_block_idx,
                )
                pv_src_pingpong_id += 1

            # if DAV_VEC:
            pvPendTile = pvVecTile[(tile_id + 1) % outOTileNBuffers]
            compute_gu(
                S0=S0,
                HEAD_SIZE=Cube_HEAD,
                S1=S1,
                Cube_S0=Cube_S0,
                Tile_S1=Tile_S1,
                PV_CV_FIFO=pv_tile_fifo_size,
                CV_FIFO_CONS_SYNC_PERIOD=CV_FIFO_CONS_SYNC_PERIOD,
                INTERMEDIATE_CHECK=INTERMEDIATE_CHECK,
                CAUSAL_MASK=CAUSAL_MASK,
                SRC_VEC_TN_BUFFERS=srcVecTNBuffers,
                OUT_O_TILE_NBUFFERS=outOTileNBuffers,
                TileOutT=TileOutGuT,
                ReduceTileF_T=ReduceTileF_T,
                TSyncPV2GU=pv2guSync,
                tile_id=tile_id,
                num_tiles=num_tiles_s1,
                pv_tile_fifo=pv_tile_fifo_block,
                pv_pend_tile_fifo=pv_pend_tile_fifo_block,
                o_out=o_out_block,
                o_parts_out=o_parts_block,
                runningOTile=runningOTile,
                pvVecTile=pvVecTile[tile_id % outOTileNBuffers],
                pvPendTile=pvPendTile,
                l1_exp_max_ififo=l1_exp_max_ififo[tile_id % qkp_tile_fifo_size],
                l2_global_sum=l2_global_sum,
                guEventId=tile_id % outOTileNBuffers,
                pv2guSync=pv2guSync,
                tmpOTile=tmpOTile,
            )
            p_gu_src_pingpong_id += 1

        # int pending_qk_sm_consumed = 0;
        # int pending_update_consumed = 0;
        # if constexpr (CAUSAL_MASK) {
        #     pending_qk_sm_consumed = pending_consumption_events(num_tiles_s1, static_cast<int>(qkp_tile_fifo_size),
        #                                                         CV_FIFO_CONS_SYNC_PERIOD);
        #     pending_update_consumed = pending_consumption_events(num_tiles_s1, static_cast<int>(qkp_tile_fifo_size),
        #                                                          CV_FIFO_CONS_SYNC_PERIOD);
        # } else {
        #     constexpr int num_tiles_s1_const = S1 / Tile_S1;
        #     pending_qk_sm_consumed =
        #         pending_consumption_events_const<num_tiles_s1_const, qkp_tile_fifo_size, CV_FIFO_CONS_SYNC_PERIOD>();
        #     pending_update_consumed =
        #         pending_consumption_events_const<num_tiles_s1_const, qkp_tile_fifo_size, CV_FIFO_CONS_SYNC_PERIOD>();
        # }
        # const int pending_sv_consumed = pending_qk_sm_consumed; // same schedule and FIFO settings
        
        # if constexpr (DAV_CUBE) {
        #     for (int i = 0; i < pending_qk_sm_consumed; ++i)
        #         qk2smSync.allocate();
        #     for (int i = 0; i < pending_update_consumed; ++i)
        #         pv2guSync.allocate();
        # }

        # if constexpr (DAV_VEC) {
        #     for (int i = 0; i < pending_sv_consumed; ++i) {
        #         sm2pvSync.allocate();
        #     }
        # }

        # const uint64_t tEnd = get_sys_cnt();
        # if (profile_entry != nullptr) {
        #     profile_entry[1] = tEnd;
        # }

    # if DAV_CUBE:
    pto.wait_flag(pto.Pipe.M, pto.Pipe.MTE1, event_id=0)
    pto.wait_flag(pto.Pipe.M, pto.Pipe.MTE1, event_id=1)
    pto.wait_flag(pto.Pipe.MTE1, pto.Pipe.MTE2, event_id=0)
    pto.wait_flag(pto.Pipe.MTE1, pto.Pipe.MTE2, event_id=1)
    pto.wait_flag(pto.Pipe.MTE1, pto.Pipe.MTE2, event_id=2)
    pto.wait_flag(pto.Pipe.MTE1, pto.Pipe.MTE2, event_id=3)
    pto.wait_flag(pto.Pipe.FIX, pto.Pipe.M, event_id=0)
    pto.wait_flag(pto.Pipe.FIX, pto.Pipe.M, event_id=1)
    pto.wait_flag(pto.Pipe.FIX, pto.Pipe.M, event_id=2)
    _ = (RUNNING_O_AVALIABLE, RUNNING_O_AVALIABLE + 16)
    # if DAV_VEC:
    pto.wait_flag(pto.Pipe.V, pto.Pipe.MTE2, event_id=0)
    pto.wait_flag(pto.Pipe.V, pto.Pipe.MTE2, event_id=1)
    pto.wait_flag(pto.Pipe.MTE3, pto.Pipe.V, event_id=0)
    pto.wait_flag(pto.Pipe.MTE3, pto.Pipe.V, event_id=1)

fa_dn_ptodsl = runTFA


def emit_fa_dn_mlir(
    *,
    s0: int = 128,
    head_dim: int = 128,
    s1: int = 1024,
    q_rows: int = 128,
    cube_s1: int = kFaCubeS1,
    tile_s1: int = kFaTileS1,
    qk_preload: int = kFaQkPreload,
    cv_fifo_size: int = kFaCvFifoSize,
    intermediate_check: bool = False,
    causal: bool = False,
    cv_fifo_cons_sync_period: int = kFaCvFifoConsSyncPeriod,
) -> str:
    compiled = runTFA.compile(
        S0=s0,
        HEAD_SIZE=head_dim,
        S1=s1,
        CUBE_S0=q_rows,
        CUBE_S1=cube_s1,
        TILE_S1=tile_s1,
        QK_PRELOAD=qk_preload,
        CV_FIFO_SIZE=cv_fifo_size,
        INTERMEDIATE_CHECK=intermediate_check,
        CAUSAL_MASK=causal,
        CV_FIFO_CONS_SYNC_PERIOD=cv_fifo_cons_sync_period,
    )
    return compiled.mlir_text()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emit-mlir", action="store_true", help="print compiled MLIR and exit")
    parser.add_argument("--s0", type=int, default=128)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--s1", type=int, default=1024)
    parser.add_argument("--q-rows", type=int, default=128)
    parser.add_argument("--cube-s1", type=int, default=kFaCubeS1)
    parser.add_argument("--tile-s1", type=int, default=kFaTileS1)
    parser.add_argument("--qk-preload", type=int, default=kFaQkPreload)
    parser.add_argument("--cv-fifo-size", type=int, default=kFaCvFifoSize)
    parser.add_argument("--intermediate-check", action="store_true")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--cv-fifo-cons-sync-period", type=int, default=kFaCvFifoConsSyncPeriod)
    args = parser.parse_args(argv)

    if args.emit_mlir:
        print(
            emit_fa_dn_mlir(
                s0=args.s0,
                head_dim=args.head_dim,
                s1=args.s1,
                q_rows=args.q_rows,
                cube_s1=args.cube_s1,
                tile_s1=args.tile_s1,
                qk_preload=args.qk_preload,
                cv_fifo_size=args.cv_fifo_size,
                intermediate_check=args.intermediate_check,
                causal=args.causal,
                cv_fifo_cons_sync_period=args.cv_fifo_cons_sync_period,
            )
        )
        return 0

    raise SystemExit("fa_dn_ptodsl.py currently supports --emit-mlir only")


if __name__ == "__main__":
    raise SystemExit(main())
