# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
CV-split PTODSL port of the hw-native Python Flash Attention example.

This variant preserves the legacy ``cube kernel`` + ``vector kernel`` split and
adapts the cross-kernel communication to the current A5 PTODSL local-pipe
surface. Unlike the previous draft, the Cube and Vector pieces are authored as
``@pto.jit(entry=False)`` kernel modules, and a single outer ``@pto.jit``
entry owns the host-visible ABI.
"""

import argparse
import os
from pathlib import Path
import time

import numpy as np

from ptodsl import pto
try:
    from .gu import fa_gu_init_vpto, fa_gu_update_vpto
    from .softmax import fa_softmax_init_vpto, fa_softmax_update_vpto
except ImportError:
    from gu import fa_gu_init_vpto, fa_gu_update_vpto
    from softmax import fa_softmax_init_vpto, fa_softmax_update_vpto


S0 = 128
S0_HALF = S0 // 2
HEAD = 128
CUBE_S1 = 128
VEC_CORES = 2
SLOT_NUM = 8

DEFAULT_S1_TILE = int(os.environ.get("FA_S1_TILE", "256"))
if DEFAULT_S1_TILE not in (256, 512):
    raise ValueError(f"FA_S1_TILE must be 256 or 512, got {DEFAULT_S1_TILE}")
if DEFAULT_S1_TILE % CUBE_S1 != 0:
    raise ValueError(f"FA_S1_TILE={DEFAULT_S1_TILE} must be a multiple of CUBE_S1={CUBE_S1}")

DEFAULT_Q_ROWS = int(os.environ.get("FA_Q_ROWS", "128"))
if DEFAULT_Q_ROWS % S0 != 0:
    raise ValueError(f"FA_Q_ROWS={DEFAULT_Q_ROWS} must be a multiple of S0={S0}")

DEFAULT_QK_PRELOAD = int(os.environ.get("FA_QK_PRELOAD", os.environ.get("FA_DSL_QK_PRELOAD", "3")))
if DEFAULT_QK_PRELOAD not in (3, 4):
    raise ValueError(f"FA_QK_PRELOAD must be 3 or 4, got {DEFAULT_QK_PRELOAD}")

DEFAULT_EXP_RING = int(os.environ.get("FA_EXP_RING", os.environ.get("FA_DSL_EXP_RING", str(DEFAULT_QK_PRELOAD))))
if DEFAULT_EXP_RING != DEFAULT_QK_PRELOAD:
    raise ValueError(
        f"FA_EXP_RING must currently equal FA_QK_PRELOAD ({DEFAULT_QK_PRELOAD}), got {DEFAULT_EXP_RING}"
    )

QK_C2V_PIPE_ID = 0
P_V2C_PIPE_ID = 1
PV_C2V_PIPE_ID = 2
SPLIT_UP_DOWN = 1

_DEVICE = "npu:0"
NEG_INF_F32 = -3.4028235e38

_ENTRY_SYMBOL = "hw_native_flash_attention_cv_split"
_CUBE_SYMBOL = "hw_native_flash_attention_cv_split_cube"
_VECTOR_SYMBOL = "hw_native_flash_attention_cv_split_vector"


def _gm_slot_layout(*, head_dim: int, s1_tile: int) -> tuple[int, int, int, int, int, int, int, int]:
    SLOT_SIZE_QK = S0 * s1_tile * 4
    SLOT_SIZE_PV = S0 * head_dim * 4
    SLOT_SIZE_P = S0 * s1_tile * 2
    GM_BYTES_PER_BLOCK = (SLOT_SIZE_QK + SLOT_SIZE_PV + SLOT_SIZE_P) * SLOT_NUM
    GM_ELEMS_PER_BLOCK = GM_BYTES_PER_BLOCK // 4
    GM_QK_OFF_F32 = 0
    GM_PV_OFF_F32 = (SLOT_SIZE_QK * SLOT_NUM) // 4
    GM_P_OFF_F32 = GM_PV_OFF_F32 + (SLOT_SIZE_PV * SLOT_NUM) // 4
    return (
        SLOT_SIZE_QK,
        SLOT_SIZE_PV,
        SLOT_SIZE_P,
        GM_BYTES_PER_BLOCK,
        GM_ELEMS_PER_BLOCK,
        GM_QK_OFF_F32,
        GM_PV_OFF_F32,
        GM_P_OFF_F32,
    )


def _validate_specialization(*, head_dim: int, s1_tile: int, qk_preload: int, causal: bool, q_rows: int) -> None:
    if head_dim != HEAD:
        raise ValueError(f"cv-split flash attention currently requires head_dim={HEAD}, got {head_dim}")
    if s1_tile not in (256, 512):
        raise ValueError(f"s1_tile must be 256 or 512, got {s1_tile}")
    if s1_tile % CUBE_S1 != 0:
        raise ValueError(f"s1_tile={s1_tile} must be a multiple of CUBE_S1={CUBE_S1}")
    if qk_preload not in (3, 4):
        raise ValueError(f"qk_preload must be 3 or 4, got {qk_preload}")
    if causal:
        raise ValueError("hw-native flash attention cv-split port is non-causal; causal=True is not supported yet")
    if q_rows % S0 != 0:
        raise ValueError(f"q_rows={q_rows} must be a multiple of S0={S0}")

# -------------------------------------------------------------------------
# Helper: even share of NUM_Q_BLOCKS across this core grid.
# The C++ kernel uses one Q-row block per AIC core (block_idx -> Q rows);
# in DSL we let the launcher choose blockDim and split inside.
# -------------------------------------------------------------------------
def compute_qb_range(total_q_blocks):
    block_num = pto.index_cast(pto.get_block_num())
    bid = pto.index_cast(pto.get_block_idx())
    floor_div = total_q_blocks // block_num
    extra = total_q_blocks % block_num
    fat_start = bid * (floor_div + 1)
    thin_start = extra * (floor_div + 1) + (bid - extra) * floor_div
    qb_start = pto.select(bid < extra, fat_start, thin_start)
    per_core = pto.select(bid < extra, floor_div + 1, floor_div)
    return bid, qb_start, qb_start + per_core


def _specialized_symbol(base: str, *, head_dim: int, s1_tile: int, qk_preload: int, q_rows: int) -> str:
    return f"{base}_h{head_dim}_s1t{s1_tile}_qp{qk_preload}_qr{q_rows}"


def _validate_runtime_problem(*, q_rows: int, s1: int, s1_tile: int, qk_preload: int) -> None:
    if q_rows <= 0:
        raise ValueError(f"q_rows must be positive, got {q_rows}")
    if s1 <= 0:
        raise ValueError(f"s1 must be positive, got {s1}")
    if s1 % s1_tile != 0:
        raise ValueError(f"s1={s1} must be a multiple of s1_tile={s1_tile}")
    if s1 // s1_tile < qk_preload:
        raise ValueError(
            f"s1={s1} provides only {s1 // s1_tile} logical S1 tiles, "
            f"but qk_preload={qk_preload} requires at least that many"
        )


def _reference_flash_attention(q: np.ndarray, k_tokens: np.ndarray, v_tokens: np.ndarray) -> np.ndarray:
    q_f32 = q.astype(np.float32, copy=False)
    k_f32 = k_tokens.astype(np.float32, copy=False)
    v_f32 = v_tokens.astype(np.float32, copy=False)
    scale = 1.0 / np.sqrt(q_f32.shape[1])

    scores = q_f32 @ k_f32.T
    scores *= scale
    row_max = np.max(scores, axis=1, keepdims=True)
    probs = np.exp(scores - row_max, dtype=np.float32)
    probs /= np.sum(probs, axis=1, keepdims=True, dtype=np.float32)
    return probs @ v_f32


def _init_runtime():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "hw_native_flash_attention_cv_split.py launch requires a Python environment with torch installed"
        ) from exc
    try:
        import torch_npu
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "hw_native_flash_attention_cv_split.py launch requires a Python environment with torch_npu installed"
        ) from exc

    torch.npu.config.allow_internal_format = False
    torch_npu.npu.set_compile_mode(jit_compile=False)
    torch.npu.set_device(_DEVICE)
    return torch


def _current_stream(torch):
    return torch.npu.current_stream()._as_parameter_  # noqa: SLF001


def _build_flash_attention_entry(
    *,
    head_dim: int = HEAD,
    s1_tile: int = DEFAULT_S1_TILE,
    qk_preload: int = DEFAULT_QK_PRELOAD,
    causal: bool = False,
    q_rows: int = DEFAULT_Q_ROWS,
):
    _validate_specialization(
        head_dim=head_dim,
        s1_tile=s1_tile,
        qk_preload=qk_preload,
        causal=causal,
        q_rows=q_rows,
    )

    cube_symbol = _specialized_symbol(
        _CUBE_SYMBOL,
        head_dim=head_dim,
        s1_tile=s1_tile,
        qk_preload=qk_preload,
        q_rows=q_rows,
    )
    vector_symbol = _specialized_symbol(
        _VECTOR_SYMBOL,
        head_dim=head_dim,
        s1_tile=s1_tile,
        qk_preload=qk_preload,
        q_rows=q_rows,
    )
    entry_symbol = _specialized_symbol(
        _ENTRY_SYMBOL,
        head_dim=head_dim,
        s1_tile=s1_tile,
        qk_preload=qk_preload,
        q_rows=q_rows,
    )

    tile_factor = s1_tile // CUBE_S1
    vec_gu_rows = S0 // VEC_CORES
    vec_s0 = S0 // VEC_CORES // tile_factor
    total_q_blocks = q_rows // S0
    scale_value = 1.0 / (head_dim ** 0.5)
    exp_ring = DEFAULT_EXP_RING if qk_preload == DEFAULT_QK_PRELOAD else qk_preload
    qk_c2v_pipe_id = int(QK_C2V_PIPE_ID)
    p_v2c_pipe_id = int(P_V2C_PIPE_ID)
    pv_c2v_pipe_id = int(PV_C2V_PIPE_ID)
    split_up_down = int(SPLIT_UP_DOWN)
    (
        SLOT_SIZE_QK,
        SLOT_SIZE_PV,
        SLOT_SIZE_P,
        _GM_BYTES_PER_BLOCK,
        GM_ELEMS_PER_BLOCK,
        GM_QK_OFF_F32,
        GM_PV_OFF_F32,
        GM_P_OFF_F32,
    ) = _gm_slot_layout(head_dim=head_dim, s1_tile=s1_tile)

    # =========================================================================
    # Cube kernel
    # =========================================================================
    @pto.jit(
        name=cube_symbol,
        target="a5",
        entry=False,
        kernel_kind="cube",
        mode="auto",
        backend="emitc",
        insert_sync=True,
    )
    def cube_kernel(
        gm_slot_buffer: pto.ptr(pto.f32, "gm"),
        gm_slot_buffer_fp16: pto.ptr(pto.f16, "gm"),
        gm_q: pto.ptr(pto.f16, "gm"),
        gm_k: pto.ptr(pto.f16, "gm"),
        gm_v: pto.ptr(pto.f16, "gm"),
        s0_i64: pto.i64,
        s1_i64: pto.i64,
    ):
        s0 = pto.index_cast(s0_i64)
        s1 = pto.index_cast(s1_i64)
        num_tiles_s1 = s1 // s1_tile
        steady_tiles = num_tiles_s1 - qk_preload

        bid, qb_start, qb_end = compute_qb_range(total_q_blocks)

        gm_blk = pto.addptr(gm_slot_buffer, bid * GM_ELEMS_PER_BLOCK)
        gm_qk = pto.addptr(gm_blk, GM_QK_OFF_F32)
        gm_pv = pto.addptr(gm_blk, GM_PV_OFF_F32)
        # The P slot is fp16-typed, so address it via the fp16-cast slot buffer.
        # GM_P_OFF_F32 is in fp32 elements; double for fp16 element stride.
        gm_blk_fp16 = pto.addptr(gm_slot_buffer_fp16, bid * (2 * GM_ELEMS_PER_BLOCK))
        gm_p = pto.addptr(gm_blk_fp16, 2 * GM_P_OFF_F32)

        # ---- QK pipe (cube producer): l2g2l GM-staged slot ----
        qk_slot_view = pto.make_tensor_view(gm_qk, shape=[S0, s1_tile], strides=[s1_tile, 1])
        qk_pipe = pto.pipe.c2v(
            slot_size=SLOT_SIZE_QK,
            gm_slot_tensor=qk_slot_view,
            id=qk_c2v_pipe_id,
        )

        # ---- PV pipe (cube producer): l2g2l GM-staged slot ----
        pv_slot_view = pto.make_tensor_view(gm_pv, shape=[S0, head_dim], strides=[head_dim, 1])
        pv_pipe = pto.pipe.c2v(
            slot_size=SLOT_SIZE_PV,
            gm_slot_tensor=pv_slot_view,
            id=pv_c2v_pipe_id,
        )

        # ---- P pipe (cube consumer of vec output): l2g2l GM-staged slot ----
        p_slot_view_cube = pto.make_tensor_view(gm_p, shape=[S0, s1_tile], strides=[s1_tile, 1])
        p_pipe = pto.pipe.v2c(
            slot_size=SLOT_SIZE_P,
            gm_slot_tensor=p_slot_view_cube,
            id=p_v2c_pipe_id,
        )
        qk_pipe.init_cube()
        pv_pipe.init_cube()
        p_pipe.init_cube()

        # ---- Allocate cube tiles. Match the manual kernel's ping-pong for
        # K/P/V MAT tiles where L1 capacity allows it. RIGHT is single-buffered
        # because two 128x128 RIGHT tiles for both QK and PV overflow L0B.
        q_mat = pto.alloc_tile(
            shape=[S0, head_dim],
            dtype=pto.f16,
            memory_space="MAT",
            blayout="ColMajor",
            slayout="RowMajor",
        )
        q_left = pto.alloc_tile(
            shape=[S0, head_dim],
            dtype=pto.f16,
            memory_space="LEFT",
            blayout="ColMajor",
            slayout="RowMajor",
        )
        k_mat_a = pto.alloc_tile(
            shape=[head_dim, CUBE_S1],
            dtype=pto.f16,
            memory_space="MAT",
            blayout="RowMajor",
            slayout="ColMajor",
        )
        k_mat_b = pto.alloc_tile(
            shape=[head_dim, CUBE_S1],
            dtype=pto.f16,
            memory_space="MAT",
            blayout="RowMajor",
            slayout="ColMajor",
        )
        k_right_a = pto.alloc_tile(
            shape=[head_dim, CUBE_S1],
            dtype=pto.f16,
            memory_space="RIGHT",
            slayout="ColMajor",
        )
        qk_acc_a = pto.alloc_tile(
            shape=[S0, CUBE_S1],
            dtype=pto.f32,
            memory_space="ACC",
            blayout="ColMajor",
            slayout="RowMajor",
        )
        p_recv_a = pto.alloc_tile(
            shape=[S0, CUBE_S1],
            dtype=pto.f16,
            memory_space="MAT",
            blayout="ColMajor",
            slayout="RowMajor",
        )
        p_left_a = pto.alloc_tile(
            shape=[S0, CUBE_S1],
            dtype=pto.f16,
            memory_space="LEFT",
            blayout="ColMajor",
            slayout="RowMajor",
        )
        v_mat_a = pto.alloc_tile(
            shape=[CUBE_S1, head_dim],
            dtype=pto.f16,
            memory_space="MAT",
            blayout="ColMajor",
            slayout="RowMajor",
        )
        v_right_a = pto.alloc_tile(
            shape=[CUBE_S1, head_dim],
            dtype=pto.f16,
            memory_space="RIGHT",
            slayout="ColMajor",
        )
        pv_acc_a = pto.alloc_tile(
            shape=[S0, head_dim],
            dtype=pto.f32,
            memory_space="ACC",
            blayout="ColMajor",
            slayout="RowMajor",
        )
        k_mat = [k_mat_a, k_mat_b]
        k_right = [k_right_a, k_right_a]
        qk_acc = [qk_acc_a, qk_acc_a]
        p_recv = [p_recv_a, p_recv_a]
        p_left = [p_left_a, p_left_a]
        v_mat = [v_mat_a, v_mat_a]
        v_right = [v_right_a, v_right_a]
        pv_acc = [pv_acc_a, pv_acc_a]

        tv_q = pto.make_tensor_view(gm_q, shape=[s0, head_dim], strides=[head_dim, 1])
        tv_k = pto.make_tensor_view(gm_k, shape=[head_dim, s1], strides=[1, head_dim], layout="DN")
        tv_v = pto.make_tensor_view(gm_v, shape=[s1, head_dim], strides=[head_dim, 1])

        # Closures over the shared tile state. The steady state overlaps PV for
        # the current S1 tile with QK for the next S1 tile at CUBE_S1 granularity.
        def compute_qk_subtile(qk_entry, s1_tile_idx, sub, b):
            pto.tile.load(
                tv_k,
                k_mat[b],
                offsets=[0, s1_tile_idx * s1_tile + sub * CUBE_S1],
                sizes=[head_dim, CUBE_S1],
            )
            pto.tile.mov(k_mat[b], k_right[b])
            pto.tile.matmul(q_left, k_right[b], qk_acc[b])
            pto.tile.store(
                qk_acc[b],
                qk_entry,
                offsets=[0, sub * CUBE_S1],
                sizes=[S0, CUBE_S1],
            )

        def compute_qk_tile(s1_tile_idx, b):
            qk_entry = qk_pipe.alloc(split=split_up_down)
            for sub in pto.static_range(tile_factor):
                compute_qk_subtile(qk_entry, s1_tile_idx, sub, b)
            qk_pipe.push(qk_entry, split=split_up_down)

        def accumulate_pv_subtile(p_entry, t_idx, sub, b):
            pto.tile.load(
                p_entry,
                p_recv[b],
                offsets=[0, sub * CUBE_S1],
                sizes=[S0, CUBE_S1],
            )
            pto.tile.mov(p_recv[b], p_left[b])
            pto.tile.load(
                tv_v,
                v_mat[b],
                offsets=[t_idx * s1_tile + sub * CUBE_S1, 0],
                sizes=[CUBE_S1, head_dim],
            )
            pto.tile.mov(v_mat[b], v_right[b])
            if pto.const_expr(sub == 0):
                pto.tile.matmul(p_left[b], v_right[b], pv_acc[b])
            else:
                pto.tile.matmul_acc(pv_acc[b], p_left[b], v_right[b], pv_acc[b])

        def push_pv(p_entry, b):
            p_pipe.free(p_entry, split=split_up_down)
            pv_entry = pv_pipe.alloc(split=split_up_down)
            pto.tile.store(
                pv_acc[b],
                pv_entry,
                offsets=[0, 0],
                sizes=[S0, head_dim],
            )
            pv_pipe.push(pv_entry, split=split_up_down)

        def accumulate_pv_tile(t_idx, b):
            p_entry = p_pipe.pop(split=split_up_down)
            for sub in pto.static_range(tile_factor):
                accumulate_pv_subtile(p_entry, t_idx, sub, b)
            push_pv(p_entry, b)

        def process_interleaved_qk_pv(next_idx, current_idx, b):
            p_entry = p_pipe.pop(split=split_up_down)
            for sub in pto.static_range(tile_factor):
                accumulate_pv_subtile(p_entry, current_idx, sub, b)
                if pto.const_expr(sub == 0):
                    qk_entry = qk_pipe.alloc(split=split_up_down)
                if pto.const_expr(sub == tile_factor - 1):
                    push_pv(p_entry, b)
                compute_qk_subtile(qk_entry, next_idx, sub, b)
                if pto.const_expr(sub == tile_factor - 1):
                    qk_pipe.push(qk_entry, split=split_up_down)

        # ---- Q-block loop ----
        for qb in range(qb_start, qb_end, 1):
            pto.tile.load(
                tv_q,
                q_mat,
                offsets=[qb * S0, 0],
                sizes=[S0, head_dim],
            )
            pto.tile.mov(q_mat, q_left)

            # ---- prologue: emit QK[0..QK_PRELOAD-1] -------------------------
            # V loading is now inline in emit_pv (per-sub-tile), so no preload.
            for kp in pto.static_range(qk_preload):
                compute_qk_tile(kp, kp % 2)

            # ---- steady state ------------------------------------------------
            # Match the 140tflops schedule: consume current P/PV and emit the
            # next QK slot at CUBE_S1 sub-tile granularity.
            for tile_id in range(0, steady_tiles, 1):
                next_tile = tile_id + qk_preload
                process_interleaved_qk_pv(next_tile, tile_id, 0)

            # ---- epilogue: drain the last QK_PRELOAD PVs -------------------
            for k in pto.static_range(qk_preload):
                b = 0
                t_idx = steady_tiles + k
                accumulate_pv_tile(t_idx, b)

    @pto.jit(
        name=vector_symbol,
        target="a5",
        entry=False,
        kernel_kind="vector",
        mode="auto",
        backend="emitc",
        insert_sync=True,
    )
    def flash_attention_vector_kernel(
        gm_slot_buffer: pto.ptr(pto.f32, "gm"),
        gm_slot_buffer_fp16: pto.ptr(pto.f16, "gm"),
        gm_o: pto.ptr(pto.f32, "gm"),
        s0: pto.i64,
        s1: pto.i64,
    ):
        s0_index = pto.index_cast(s0)
        s1_index = pto.index_cast(s1)
        num_tiles_s1 = s1_index // s1_tile
        steady_tiles = num_tiles_s1 - qk_preload

        bid, qb_start, qb_end = compute_qb_range(total_q_blocks)

        gm_blk = pto.addptr(gm_slot_buffer, bid * GM_ELEMS_PER_BLOCK)
        gm_qk = pto.addptr(gm_blk, GM_QK_OFF_F32)
        gm_pv = pto.addptr(gm_blk, GM_PV_OFF_F32)
        gm_blk_fp16 = pto.addptr(gm_slot_buffer_fp16, bid * (2 * GM_ELEMS_PER_BLOCK))
        gm_p = pto.addptr(gm_blk_fp16, 2 * GM_P_OFF_F32)

        # ---- QK pipe (vec consumer): l2g2l GM-staged slot ----
        # Vec sees one slot as [VecGuRows, S1_TILE] -- SPLIT_UP_DOWN halves
        # the row count when crossing into the subblock; per row_slice we
        # tload a [Vec_S0, S1_TILE] partition.
        qk_slot_view = pto.make_tensor_view(gm_qk, shape=[vec_gu_rows, s1_tile], strides=[s1_tile, 1])
        qk_pipe = pto.pipe.c2v(
            slot_size=SLOT_SIZE_QK,
            gm_slot_tensor=qk_slot_view,
            id=qk_c2v_pipe_id,
        )
        # ---- PV pipe (vec consumer): l2g2l GM-staged slot ----
        pv_slot_view = pto.make_tensor_view(gm_pv, shape=[vec_gu_rows, head_dim], strides=[head_dim, 1])
        pv_pipe = pto.pipe.c2v(
            slot_size=SLOT_SIZE_PV,
            gm_slot_tensor=pv_slot_view,
            id=pv_c2v_pipe_id,
        )

        # ---- P pipe (vec producer): l2g2l GM-staged slot ----
        p_slot_view = pto.make_tensor_view(gm_p, shape=[vec_gu_rows, s1_tile], strides=[s1_tile, 1])
        p_pipe = pto.pipe.v2c(
            slot_size=SLOT_SIZE_P,
            gm_slot_tensor=p_slot_view,
            id=p_v2c_pipe_id,
        )

        qk_pipe.init_simd()
        p_pipe.init_simd()
        pv_pipe.init_simd()

        # ---- Vec tile allocations.
        # Per-slice working tiles are reused across the row_slice loop (each
        # iter overwrites the previous), so a single allocation per type is
        # enough. Reduce/state tiles are per-row_slice arrays because each
        # row_slice tracks its own running_max/running_sum independently.
        qk_vec = pto.alloc_tile(shape=[vec_s0, s1_tile], dtype=pto.f32)
        p_nz = pto.alloc_tile(shape=[vec_s0, s1_tile], dtype=pto.f16)
        pv_vec = [pto.alloc_tile(shape=[vec_s0, head_dim], dtype=pto.f32) for _ in range(tile_factor)]
        o_tile = [pto.alloc_tile(shape=[vec_s0, head_dim], dtype=pto.f32) for _ in range(tile_factor)]

        running_max = [
            pto.alloc_tile(shape=[vec_s0, 1], dtype=pto.f32, valid_shape=[vec_s0, 1], blayout="ColMajor")
            for _ in range(tile_factor)
        ]
        running_sum = [
            pto.alloc_tile(shape=[vec_s0, 1], dtype=pto.f32, valid_shape=[vec_s0, 1], blayout="ColMajor")
            for _ in range(tile_factor)
        ]
        # The shorter 140tflops-style preload only needs one exp_max slot per
        # preloaded logical S1 tile.
        exp_max_ring = [
            [
                pto.alloc_tile(shape=[vec_s0, 1], dtype=pto.f32, valid_shape=[vec_s0, 1], blayout="ColMajor")
                for _ in range(tile_factor)
            ]
            for _ in range(exp_ring)
        ]

        softmax_scale = scale_value
        sb_idx = pto.index_cast(pto.get_subblock_idx())
        row_off_sb = sb_idx * S0_HALF
        tv_o = pto.make_tensor_view(gm_o, shape=[s0_index, head_dim], strides=[head_dim, 1])

        # ---- apply_softmax(exp_max_slots, is_init): one streaming softmax ------
        # Pop the wide QK slot (full subblock) and talloc one wide P slot;
        # iterate TILE_FACTOR row_slices, doing per-slice softmax math on
        # [Vec_S0, S1_TILE] tiles and per-slice reduce state. After all
        # row_slices, push the wide P slot.
        def apply_softmax(exp_max_slots, is_init):
            qk_entry = qk_pipe.pop(split=split_up_down)
            p_entry = p_pipe.alloc(split=split_up_down)
            for row_slice in pto.static_range(tile_factor):
                row_off = row_slice * vec_s0
                pto.tile.load(
                    qk_entry,
                    qk_vec,
                    offsets=[row_off, 0],
                    sizes=[vec_s0, s1_tile],
                )
                qk = qk_vec
                rmax = running_max[row_slice]
                rsum = running_sum[row_slice]
                exp_slot = exp_max_slots[row_slice]
                if pto.const_expr(is_init):
                    fa_softmax_init_vpto(
                        qk,
                        p_nz,
                        rmax,
                        rsum,
                        softmax_scale,
                    )
                else:
                    fa_softmax_update_vpto(
                        qk,
                        p_nz,
                        rmax,
                        rsum,
                        exp_slot,
                        softmax_scale,
                    )
                pto.tile.store(
                    p_nz,
                    p_entry,
                    offsets=[row_off, 0],
                    sizes=[vec_s0, s1_tile],
                )
            p_pipe.push(p_entry, split=split_up_down)
            qk_pipe.free(qk_entry, split=split_up_down)

        # ---- update_output(exp_max_slots, is_init): rescale + add running O ------
        # GU also runs per-row_slice: each row_slice owns its own o_tile and
        # pv_vec, indexed by the same exp_max_slots used during softmax.
        def update_output(exp_max_slots, is_init):
            pv_entry = pv_pipe.pop(split=split_up_down, result_type=pv_pipe.entry_type)
            for row_slice in pto.static_range(tile_factor):
                row_off = row_slice * vec_s0
                pto.tile.load(
                    pv_entry,
                    pv_vec[row_slice],
                    offsets=[row_off, 0],
                    sizes=[vec_s0, head_dim],
                )
                if pto.const_expr(is_init):
                    fa_gu_init_vpto(pv_vec[row_slice], o_tile[row_slice])
                else:
                    fa_gu_update_vpto(o_tile[row_slice], pv_vec[row_slice], exp_max_slots[row_slice])
            pv_pipe.free(pv_entry, split=split_up_down)

        def apply_softmax_for_tile(tile_id):
            mod = tile_id % exp_ring
            if mod == 0:
                apply_softmax(exp_max_ring[0], is_init=False)
            elif mod == 1:
                apply_softmax(exp_max_ring[1], is_init=False)
            elif mod == 2:
                apply_softmax(exp_max_ring[2], is_init=False)
            elif pto.const_expr(exp_ring > 3):
                apply_softmax(exp_max_ring[3], is_init=False)

        def update_output_for_tile(tile_id):
            mod = tile_id % exp_ring
            if mod == 0:
                update_output(exp_max_ring[0], is_init=False)
            elif mod == 1:
                update_output(exp_max_ring[1], is_init=False)
            elif mod == 2:
                update_output(exp_max_ring[2], is_init=False)
            elif pto.const_expr(exp_ring > 3):
                update_output(exp_max_ring[3], is_init=False)

        def finish_output_for_tile(tile_id):
            if tile_id == 0:
                update_output(exp_max_ring[0], is_init=True)
            else:
                update_output_for_tile(tile_id)

        for qb in range(qb_start, qb_end, 1):
            # ---- vec prologue: softmax(0..QK_PRELOAD-1) --------------------
            for kp in pto.static_range(qk_preload):
                apply_softmax(exp_max_ring[kp], is_init=(kp == 0))

            # ---- vec steady state. Match the 140tflops order: drain the
            # current PV/GU tile before producing the future P tile.
            if steady_tiles > 0:
                update_output(exp_max_ring[0], is_init=True)
                apply_softmax(exp_max_ring[qk_preload % exp_ring], is_init=False)
                for tile_id in range(1, steady_tiles, 1):
                    next_tile = tile_id + qk_preload
                    update_output_for_tile(tile_id)
                    apply_softmax_for_tile(next_tile)

            # ---- vec epilogue: drain last QK_PRELOAD gus -------------------
            for k in pto.static_range(qk_preload):
                tile_id = steady_tiles + k
                finish_output_for_tile(tile_id)

            # Final divide + GM store, one row_slice at a time.
            for row_slice in pto.static_range(tile_factor):
                row_off = row_off_sb + row_slice * vec_s0
                pto.tile.rowexpanddiv(o_tile[row_slice], running_sum[row_slice], o_tile[row_slice])
                pto.tile.store(
                    o_tile[row_slice],
                    tv_o,
                    offsets=[qb * S0 + row_off, 0],
                    sizes=[vec_s0, head_dim],
                )
    @pto.jit(name=entry_symbol, target="a5", mode="explicit", backend="emitc", insert_sync=True)
    def flash_attention_entry(
        gm_slot_buffer: pto.ptr(pto.f32, "gm"),
        gm_slot_buffer_fp16: pto.ptr(pto.f16, "gm"),
        gm_q: pto.ptr(pto.f16, "gm"),
        gm_k: pto.ptr(pto.f16, "gm"),
        gm_v: pto.ptr(pto.f16, "gm"),
        gm_o: pto.ptr(pto.f32, "gm"),
        s0: pto.i64,
        s1: pto.i64,
    ):
        cube_kernel(gm_slot_buffer, gm_slot_buffer_fp16, gm_q, gm_k, gm_v, s0, s1)
        flash_attention_vector_kernel(gm_slot_buffer, gm_slot_buffer_fp16, gm_o, s0, s1)

    return flash_attention_entry


def emit_flash_attention_mlir(
    *,
    head_dim: int = HEAD,
    s1_tile: int = DEFAULT_S1_TILE,
    qk_preload: int = DEFAULT_QK_PRELOAD,
    causal: bool = False,
    q_rows: int = DEFAULT_Q_ROWS,
) -> str:
    entry_kernel = _build_flash_attention_entry(
        head_dim=head_dim,
        s1_tile=s1_tile,
        qk_preload=qk_preload,
        causal=causal,
        q_rows=q_rows,
    )
    return entry_kernel.compile().mlir_text()


def compile_flash_attention_kernel(
    *,
    head_dim: int = HEAD,
    s1_tile: int = DEFAULT_S1_TILE,
    qk_preload: int = DEFAULT_QK_PRELOAD,
    causal: bool = False,
    q_rows: int = DEFAULT_Q_ROWS,
):
    entry_kernel = _build_flash_attention_entry(
        head_dim=head_dim,
        s1_tile=s1_tile,
        qk_preload=qk_preload,
        causal=causal,
        q_rows=q_rows,
    )
    return entry_kernel.compile()


def run_demo(
    *,
    head_dim: int = HEAD,
    s1_tile: int = DEFAULT_S1_TILE,
    qk_preload: int = DEFAULT_QK_PRELOAD,
    causal: bool = False,
    q_rows: int = DEFAULT_Q_ROWS,
    s1: int = DEFAULT_S1_TILE * DEFAULT_QK_PRELOAD,
    seed: int = 20260601,
) -> None:
    _validate_specialization(
        head_dim=head_dim,
        s1_tile=s1_tile,
        qk_preload=qk_preload,
        causal=causal,
        q_rows=q_rows,
    )
    _validate_runtime_problem(
        q_rows=q_rows,
        s1=s1,
        s1_tile=s1_tile,
        qk_preload=qk_preload,
    )

    torch = _init_runtime()
    rng = np.random.RandomState(seed)

    host_q = rng.randn(q_rows, head_dim).astype(np.float16)
    host_k_tokens = rng.randn(s1, head_dim).astype(np.float16)
    host_v = rng.randn(s1, head_dim).astype(np.float16)
    host_k = host_k_tokens
    host_ref = _reference_flash_attention(host_q, host_k_tokens, host_v)

    q_t = torch.from_numpy(host_q).to(_DEVICE)
    k_t = torch.from_numpy(host_k).to(_DEVICE)
    v_t = torch.from_numpy(host_v).to(_DEVICE)
    o_t = torch.empty((q_rows, head_dim), dtype=torch.float32, device=_DEVICE)
    total_q_blocks = q_rows // S0
    (
        _SLOT_SIZE_QK,
        _SLOT_SIZE_PV,
        _SLOT_SIZE_P,
        _GM_BYTES_PER_BLOCK,
        GM_ELEMS_PER_BLOCK,
        _GM_QK_OFF_F32,
        _GM_PV_OFF_F32,
        _GM_P_OFF_F32,
    ) = _gm_slot_layout(head_dim=head_dim, s1_tile=s1_tile)
    gm_slot_buffer_t = torch.empty(total_q_blocks * GM_ELEMS_PER_BLOCK, dtype=torch.float32, device=_DEVICE)
    gm_slot_buffer_fp16_t = torch.empty(total_q_blocks * (2 * GM_ELEMS_PER_BLOCK), dtype=torch.float16, device=_DEVICE)
    stream = _current_stream(torch)

    t0 = time.perf_counter()
    compiled = compile_flash_attention_kernel(
        head_dim=head_dim,
        s1_tile=s1_tile,
        qk_preload=qk_preload,
        causal=causal,
        q_rows=q_rows,
    )
    compile_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    compiled[1, stream](
        gm_slot_buffer_t.data_ptr(),
        gm_slot_buffer_fp16_t.data_ptr(),
        q_t.data_ptr(),
        k_t.data_ptr(),
        v_t.data_ptr(),
        o_t.data_ptr(),
        q_rows,
        s1,
    )
    torch.npu.synchronize()
    launch_s = time.perf_counter() - t0

    host_out = o_t.cpu().numpy()
    np.testing.assert_allclose(host_out, host_ref, rtol=6e-2, atol=6e-2)
    print(
        f"PASS hw-native-fa-cv-split q_rows={q_rows} s1={s1} head={head_dim} "
        f"s1_tile={s1_tile} qk_preload={qk_preload} "
        f"compile={compile_s:.3f}s launch={launch_s:.3f}s"
    )


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Emit or launch the CV-split local-pipe PTODSL hw-native FlashAttention port."
    )
    parser.add_argument(
        "--emit-mlir",
        action="store_true",
        help="print compiled MLIR and exit",
    )
    parser.add_argument("--head-dim", type=int, default=HEAD)
    parser.add_argument("--s1-tile", type=int, default=DEFAULT_S1_TILE)
    parser.add_argument("--qk-preload", type=int, default=DEFAULT_QK_PRELOAD)
    parser.add_argument("--q-rows", type=int, default=DEFAULT_Q_ROWS)
    parser.add_argument(
        "--s1",
        type=int,
        default=DEFAULT_S1_TILE * DEFAULT_QK_PRELOAD,
        help="runtime S1 length; must be a multiple of --s1-tile and provide at least --qk-preload logical tiles",
    )
    parser.add_argument("--seed", type=int, default=20260601)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("-o", "--output", default="-", help="output MLIR path, or '-' for stdout")
    return parser


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.emit_mlir:
        mlir_text = emit_flash_attention_mlir(
            head_dim=args.head_dim,
            s1_tile=args.s1_tile,
            qk_preload=args.qk_preload,
            causal=args.causal,
            q_rows=args.q_rows,
        )
        if args.output == "-":
            print(mlir_text)
            return 0
        Path(args.output).write_text(mlir_text, encoding="utf-8")
        return 0

    run_demo(
        head_dim=args.head_dim,
        s1_tile=args.s1_tile,
        qk_preload=args.qk_preload,
        causal=args.causal,
        q_rows=args.q_rows,
        s1=args.s1,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
