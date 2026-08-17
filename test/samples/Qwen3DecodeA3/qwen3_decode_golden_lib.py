#!/usr/bin/python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import numpy as np
from dataclasses import replace

from validation_runtime import (
    bf16_to_float32,
    float32_to_bf16,
    load_case_meta,
    load_integer_assignments,
    load_strided_2d,
    rng,
    store_strided_2d,
    write_buffers,
    write_golden,
)

BATCH = 16
AIV_BATCH = 8
MAX_SEQ = 4096
NUM_HEADS = 64
NUM_KV_HEADS = 8
HEAD_DIM = 128
HALF_DIM = HEAD_DIM // 2
HIDDEN = NUM_HEADS * HEAD_DIM
INTERMEDIATE = 25600
KV_HIDDEN = NUM_KV_HEADS * HEAD_DIM
Q_PER_KV = NUM_HEADS // NUM_KV_HEADS
Q_HEAD_BATCH = 8
Q_HEAD_PAD = 16
TOTAL_Q_GROUPS = NUM_KV_HEADS
PROJ_BLOCKS_PER_LAUNCH = 2
Q_PROJ_WINDOW = Q_HEAD_BATCH * HEAD_DIM
SEQ_TILE = 256
RMSNORM_K_CHUNK = 512
Q_PROJ_K_CHUNK = 128
Q_OUT_CHUNK = 256
KV_PROJ_K_CHUNK = 128
KV_OUT_CHUNK = 256
K_CHUNK = 128
MLP_OUT_CHUNK = 256
MLP_SPMD_INNER = 2
MLP_GROUP_CHUNK = MLP_SPMD_INNER * MLP_OUT_CHUNK
DOWN_K_CHUNK = 128
DOWN_N_CHUNK = 256
HIDDEN_BLOCKS = HIDDEN // K_CHUNK
MAX_CTX_BLOCKS = (MAX_SEQ + SEQ_TILE - 1) // SEQ_TILE
EPS = np.float32(1e-6)
HIDDEN_INV = np.float32(1.0 / HIDDEN)
ATTN_SCALE = np.float32(1.0 / np.sqrt(HEAD_DIM))
NEG_INF = np.finfo(np.float32).min


def make_fp32(generator, count: int, *, scale: float = 0.05, positive: bool = False) -> np.ndarray:
    if positive:
        return generator.uniform(0.25, 1.5, size=count).astype(np.float32)
    return generator.uniform(-scale, scale, size=count).astype(np.float32)


def make_bf16(generator, count: int, *, scale: float = 0.05, positive: bool = False) -> np.ndarray:
    return float32_to_bf16(make_fp32(generator, count, scale=scale, positive=positive))


def make_padded_rows_bf16(
    generator,
    count: int,
    *,
    cols: int,
    rows_per_group: int,
    active_rows: int,
    scale: float = 0.05,
    positive: bool = False,
) -> np.ndarray:
    out = make_bf16(generator, count, scale=scale, positive=positive)
    rows = out.size // cols
    for row in range(rows):
        if row % rows_per_group >= active_rows:
            start = row * cols
            out[start:start + cols] = 0
    return out


def _flat_output(meta, name: str):
    return np.zeros(meta.elem_counts[name], dtype=meta.np_types[name])


def _select_aiv_subblock(ints, *, default: int = 0) -> int:
    if len(ints) < 2:
        return default
    subblock = int(ints[1])
    max_subblock = max(BATCH // AIV_BATCH - 1, 0)
    return min(max(subblock, 0), max_subblock)


def _rope_table_halves(pos: int):
    phase = np.linspace(0.0, 1.0, HALF_DIM, dtype=np.float32)
    pos_scale = np.float32(pos + 1)
    cos_lo = (np.float32(1.0) - np.float32(0.05) * phase * pos_scale / np.float32(MAX_SEQ)).astype(np.float32)
    cos_hi = (np.float32(0.97) - np.float32(0.04) * phase * pos_scale / np.float32(MAX_SEQ)).astype(np.float32)
    sin_lo = (np.float32(0.01) + np.float32(0.02) * phase * pos_scale / np.float32(MAX_SEQ)).astype(np.float32)
    sin_hi = (np.float32(0.015) + np.float32(0.02) * phase * pos_scale / np.float32(MAX_SEQ)).astype(np.float32)
    return cos_lo, sin_lo, cos_hi, sin_hi


def _store_group_scores(buffer, values, *, group_index: int, sb: int, rows_per_group: int, cols: int):
    offset = (group_index * MAX_CTX_BLOCKS * rows_per_group + sb * rows_per_group) * cols
    return store_strided_2d(buffer, values, offset=offset, row_stride=cols)


def build_rmsnorm(meta, generator, ints):
    del ints
    buffers = {
        "v1": make_bf16(generator, meta.elem_counts["v1"], scale=0.05),
        "v2": _flat_output(meta, "v2"),
        "v3": make_fp32(generator, meta.elem_counts["v3"], scale=0.05),
    }
    sq_sum = np.zeros((BATCH, 1), dtype=np.float32)
    for kb in range(HIDDEN // RMSNORM_K_CHUNK):
        k0 = kb * RMSNORM_K_CHUNK
        x_chunk = bf16_to_float32(
            load_strided_2d(buffers["v1"], offset=k0, rows=BATCH, cols=RMSNORM_K_CHUNK, row_stride=HIDDEN)
        )
        sq_sum += np.sum(x_chunk * x_chunk, axis=1, keepdims=True)
    inv_rms = np.reciprocal(np.sqrt(sq_sum * HIDDEN_INV + EPS))
    output = np.array(buffers["v2"], copy=True)
    for kb in range(HIDDEN // RMSNORM_K_CHUNK):
        k0 = kb * RMSNORM_K_CHUNK
        x_chunk = bf16_to_float32(
            load_strided_2d(buffers["v1"], offset=k0, rows=BATCH, cols=RMSNORM_K_CHUNK, row_stride=HIDDEN)
        )
        gamma = load_strided_2d(buffers["v3"], offset=k0, rows=1, cols=RMSNORM_K_CHUNK, row_stride=HIDDEN).astype(np.float32)
        normed = x_chunk * inv_rms * gamma
        output = store_strided_2d(output, float32_to_bf16(normed), offset=k0, row_stride=HIDDEN)
    return buffers, {"v2": output}


def build_q_proj(meta, generator, ints):
    out_block = ints[0]
    q0 = out_block * Q_OUT_CHUNK
    buffers = {
        "v1": make_bf16(generator, meta.elem_counts["v1"], scale=0.05),
        "v2": make_bf16(generator, meta.elem_counts["v2"], scale=0.05),
        "v3": _flat_output(meta, "v3"),
    }
    acc = np.zeros((BATCH, Q_OUT_CHUNK), dtype=np.float32)
    for kb in range(HIDDEN // Q_PROJ_K_CHUNK):
        k0 = kb * Q_PROJ_K_CHUNK
        normed = bf16_to_float32(load_strided_2d(buffers["v1"], offset=k0, rows=BATCH, cols=Q_PROJ_K_CHUNK, row_stride=HIDDEN))
        w_chunk = bf16_to_float32(
            load_strided_2d(buffers["v2"], offset=k0 * HIDDEN + q0, rows=Q_PROJ_K_CHUNK, cols=Q_OUT_CHUNK, row_stride=HIDDEN)
        )
        acc += normed @ w_chunk
    output = np.array(buffers["v3"], copy=True)
    output = store_strided_2d(output, acc, offset=q0, row_stride=HIDDEN)
    return buffers, {"v3": output}


def build_kv_proj(meta, generator, ints):
    out_block = ints[0]
    kv0 = out_block * KV_OUT_CHUNK
    buffers = {
        "v1": make_bf16(generator, meta.elem_counts["v1"], scale=0.05),
        "v2": make_bf16(generator, meta.elem_counts["v2"], scale=0.05),
        "v3": make_bf16(generator, meta.elem_counts["v3"], scale=0.05),
        "v4": _flat_output(meta, "v4"),
        "v5": _flat_output(meta, "v5"),
    }
    k_acc = np.zeros((BATCH, KV_OUT_CHUNK), dtype=np.float32)
    v_acc = np.zeros((BATCH, KV_OUT_CHUNK), dtype=np.float32)
    for kb in range(HIDDEN // KV_PROJ_K_CHUNK):
        k0 = kb * KV_PROJ_K_CHUNK
        normed = bf16_to_float32(load_strided_2d(buffers["v1"], offset=k0, rows=BATCH, cols=KV_PROJ_K_CHUNK, row_stride=HIDDEN))
        wk_chunk = bf16_to_float32(
            load_strided_2d(buffers["v2"], offset=k0 * KV_HIDDEN + kv0, rows=KV_PROJ_K_CHUNK, cols=KV_OUT_CHUNK, row_stride=KV_HIDDEN)
        )
        wv_chunk = bf16_to_float32(
            load_strided_2d(buffers["v3"], offset=k0 * KV_HIDDEN + kv0, rows=KV_PROJ_K_CHUNK, cols=KV_OUT_CHUNK, row_stride=KV_HIDDEN)
        )
        k_acc += normed @ wk_chunk
        v_acc += normed @ wv_chunk
    out_k = np.array(buffers["v4"], copy=True)
    out_v = np.array(buffers["v5"], copy=True)
    out_k = store_strided_2d(out_k, k_acc, offset=kv0, row_stride=KV_HIDDEN)
    out_v = store_strided_2d(out_v, v_acc, offset=kv0, row_stride=KV_HIDDEN)
    return buffers, {"v4": out_k, "v5": out_v}


def build_rope_kv_cache(meta, generator, ints):
    batch_index, pos = ints[:2]
    cos_lo, sin_lo, cos_hi, sin_hi = _rope_table_halves(pos)
    buffers = {
        "v1": _flat_output(meta, "v1"),
        "v2": _flat_output(meta, "v2"),
        "v3": _flat_output(meta, "v3"),
        "v4": make_fp32(generator, meta.elem_counts["v4"], positive=True),
        "v5": np.asarray(cos_lo, dtype=np.float32),
        "v6": np.asarray(sin_lo, dtype=np.float32),
        "v7": np.asarray(cos_hi, dtype=np.float32),
        "v8": np.asarray(sin_hi, dtype=np.float32),
        "v9": make_fp32(generator, meta.elem_counts["v9"], positive=True),
        "v10": make_fp32(generator, meta.elem_counts["v10"], positive=True),
    }
    out_q = np.array(buffers["v1"], copy=True)
    out_k = np.array(buffers["v2"], copy=True)
    out_v = np.array(buffers["v3"], copy=True)
    k_proj = load_strided_2d(
        buffers["v4"],
        offset=batch_index * KV_HIDDEN,
        rows=1,
        cols=KV_HIDDEN,
        row_stride=KV_HIDDEN,
    ).astype(np.float32)
    cos_lo = np.asarray(buffers["v5"][:HALF_DIM], dtype=np.float32).reshape(1, HALF_DIM)
    sin_lo = np.asarray(buffers["v6"][:HALF_DIM], dtype=np.float32).reshape(1, HALF_DIM)
    cos_hi = np.asarray(buffers["v7"][:HALF_DIM], dtype=np.float32).reshape(1, HALF_DIM)
    sin_hi = np.asarray(buffers["v8"][:HALF_DIM], dtype=np.float32).reshape(1, HALF_DIM)
    v_proj = load_strided_2d(
        buffers["v9"],
        offset=batch_index * KV_HIDDEN,
        rows=1,
        cols=KV_HIDDEN,
        row_stride=KV_HIDDEN,
    ).astype(np.float32)
    q_proj = load_strided_2d(
        buffers["v10"],
        offset=batch_index * HIDDEN,
        rows=1,
        cols=HIDDEN,
        row_stride=HIDDEN,
    ).astype(np.float32)
    for kvh in range(NUM_KV_HEADS):
        kv_col = kvh * HEAD_DIM
        k_lo = k_proj[:, kv_col:kv_col + HALF_DIM]
        k_hi = k_proj[:, kv_col + HALF_DIM:kv_col + HEAD_DIM]
        rot_lo = k_lo * cos_lo - k_hi * sin_lo
        rot_hi = k_hi * cos_hi + k_lo * sin_hi
        cache_row = batch_index * NUM_KV_HEADS * MAX_SEQ + kvh * MAX_SEQ + pos
        out_k = store_strided_2d(out_k, float32_to_bf16(rot_lo), offset=cache_row * HEAD_DIM, row_stride=HEAD_DIM)
        out_k = store_strided_2d(out_k, float32_to_bf16(rot_hi), offset=cache_row * HEAD_DIM + HALF_DIM, row_stride=HEAD_DIM)
        v_chunk = v_proj[:, kv_col:kv_col + HEAD_DIM]
        out_v = store_strided_2d(out_v, float32_to_bf16(v_chunk), offset=cache_row * HEAD_DIM, row_stride=HEAD_DIM)

        q_col = kvh * Q_PROJ_WINDOW
        q_block = q_proj[:, q_col:q_col + Q_PROJ_WINDOW].reshape(Q_HEAD_BATCH, HEAD_DIM)
        q_lo = q_block[:, :HALF_DIM]
        q_hi = q_block[:, HALF_DIM:]
        q_rot_lo = q_lo * cos_lo - q_hi * sin_lo
        q_rot_hi = q_hi * cos_hi + q_lo * sin_hi
        q_row = batch_index * TOTAL_Q_GROUPS * Q_HEAD_PAD + kvh * Q_HEAD_PAD
        out_q = store_strided_2d(out_q, float32_to_bf16(q_rot_lo), offset=q_row * HEAD_DIM, row_stride=HEAD_DIM)
        out_q = store_strided_2d(out_q, float32_to_bf16(q_rot_hi), offset=q_row * HEAD_DIM + HALF_DIM, row_stride=HEAD_DIM)
    return buffers, {"v1": out_q, "v2": out_k, "v3": out_v}


def build_qk_matmul(meta, generator, ints):
    batch_index, ctx_blocks, pair_index = ints[:3]
    gi0 = pair_index * 2
    gi1 = gi0 + 1
    buffers = {
        "v1": make_padded_rows_bf16(
            generator,
            meta.elem_counts["v1"],
            cols=HEAD_DIM,
            rows_per_group=Q_HEAD_PAD,
            active_rows=Q_HEAD_BATCH,
            scale=0.05,
        ),
        "v2": _flat_output(meta, "v2"),
        "v3": make_bf16(generator, meta.elem_counts["v3"], scale=0.05),
    }
    output = np.array(buffers["v2"], copy=True)
    for gi in (gi0, gi1):
        q_offset = (batch_index * TOTAL_Q_GROUPS * Q_HEAD_PAD + gi * Q_HEAD_PAD) * HEAD_DIM
        q_padded = bf16_to_float32(load_strided_2d(buffers["v1"], offset=q_offset, rows=Q_HEAD_PAD, cols=HEAD_DIM, row_stride=HEAD_DIM))
        for sb in range(ctx_blocks):
            cache_offset = (batch_index * NUM_KV_HEADS * MAX_SEQ + gi * MAX_SEQ + sb * SEQ_TILE) * HEAD_DIM
            k_tile = bf16_to_float32(load_strided_2d(buffers["v3"], offset=cache_offset, rows=SEQ_TILE, cols=HEAD_DIM, row_stride=HEAD_DIM))
            raw_scores = q_padded @ k_tile.T
            output = _store_group_scores(output, raw_scores, group_index=gi, sb=sb, rows_per_group=Q_HEAD_PAD, cols=SEQ_TILE)
    return buffers, {"v2": output}


def build_softmax(meta, generator, ints):
    ctx_blocks, ctx_len, pair_index = ints[:3]
    gi0 = pair_index * 2
    gi1 = gi0 + 1
    buffers = {
        "v1": _flat_output(meta, "v1"),
        "v2": _flat_output(meta, "v2"),
        "v3": _flat_output(meta, "v3"),
        "v4": make_fp32(generator, meta.elem_counts["v4"], scale=0.05),
    }
    out_li = np.array(buffers["v1"], copy=True)
    out_mi = np.array(buffers["v2"], copy=True)
    out_exp = np.array(buffers["v3"], copy=True)
    for gi in (gi0, gi1):
        for sb in range(ctx_blocks):
            valid_len = min(SEQ_TILE, max(ctx_len - sb * SEQ_TILE, 0))
            scores = np.full((Q_HEAD_BATCH, SEQ_TILE), NEG_INF, dtype=np.float32)
            if valid_len > 0:
                in_offset = (gi * MAX_CTX_BLOCKS * Q_HEAD_PAD + sb * Q_HEAD_PAD) * SEQ_TILE
                scores_valid = load_strided_2d(
                    buffers["v4"],
                    offset=in_offset,
                    rows=Q_HEAD_BATCH,
                    cols=valid_len,
                    row_stride=SEQ_TILE,
                ).astype(np.float32)
                scores[:, :valid_len] = scores_valid[:, :valid_len]
            scores *= ATTN_SCALE
            cur_mi = np.max(scores, axis=1, keepdims=True)
            exp_scores = np.exp(scores - cur_mi)
            exp_scores_bf16 = float32_to_bf16(exp_scores)
            cur_li = np.sum(bf16_to_float32(exp_scores_bf16), axis=1, keepdims=True)
            out_exp = _store_group_scores(out_exp, exp_scores_bf16, group_index=gi, sb=sb, rows_per_group=Q_HEAD_PAD, cols=SEQ_TILE)
            base = gi * MAX_CTX_BLOCKS * Q_HEAD_BATCH + sb * Q_HEAD_BATCH
            out_mi[base:base + Q_HEAD_BATCH] = cur_mi.reshape(-1)
            out_li[base:base + Q_HEAD_BATCH] = cur_li.reshape(-1)
    return buffers, {"v1": out_li, "v2": out_mi, "v3": out_exp}


def build_sv_matmul(meta, generator, ints):
    ctx_blocks, batch_index, pair_index = ints[:3]
    gi0 = pair_index * 2
    gi1 = gi0 + 1
    buffers = {
        "v1": _flat_output(meta, "v1"),
        "v2": make_padded_rows_bf16(
            generator,
            meta.elem_counts["v2"],
            cols=SEQ_TILE,
            rows_per_group=Q_HEAD_PAD,
            active_rows=Q_HEAD_BATCH,
            scale=0.05,
            positive=True,
        ),
        "v3": make_bf16(generator, meta.elem_counts["v3"], scale=0.05),
    }
    output = np.array(buffers["v1"], copy=True)
    for gi in (gi0, gi1):
        for sb in range(ctx_blocks):
            exp_offset = (gi * MAX_CTX_BLOCKS * Q_HEAD_PAD + sb * Q_HEAD_PAD) * SEQ_TILE
            exp_tile = bf16_to_float32(
                load_strided_2d(buffers["v2"], offset=exp_offset, rows=Q_HEAD_PAD, cols=SEQ_TILE, row_stride=SEQ_TILE)
            )
            cache_offset = (batch_index * NUM_KV_HEADS * MAX_SEQ + gi * MAX_SEQ + sb * SEQ_TILE) * HEAD_DIM
            v_tile = bf16_to_float32(load_strided_2d(buffers["v3"], offset=cache_offset, rows=SEQ_TILE, cols=HEAD_DIM, row_stride=HEAD_DIM))
            oi_tmp = exp_tile @ v_tile
            output = _store_group_scores(output, oi_tmp, group_index=gi, sb=sb, rows_per_group=Q_HEAD_PAD, cols=HEAD_DIM)
    return buffers, {"v1": output}


def build_online_softmax(meta, generator, ints):
    ctx_blocks, pair_index = ints[:2]
    gi0 = pair_index * 2
    gi1 = gi0 + 1
    buffers = {
        "v1": make_fp32(generator, meta.elem_counts["v1"], scale=0.05),
        "v2": make_fp32(generator, meta.elem_counts["v2"], scale=0.05),
        "v3": make_fp32(generator, meta.elem_counts["v3"], scale=0.05, positive=True),
        "v4": _flat_output(meta, "v4"),
    }
    output = np.array(buffers["v4"], copy=True)
    for gi in (gi0, gi1):
        base = gi * MAX_CTX_BLOCKS * Q_HEAD_BATCH
        oi = load_strided_2d(
            buffers["v1"],
            offset=(gi * MAX_CTX_BLOCKS * Q_HEAD_PAD) * HEAD_DIM,
            rows=Q_HEAD_BATCH,
            cols=HEAD_DIM,
            row_stride=HEAD_DIM,
        ).astype(np.float32)
        mi = np.asarray(buffers["v2"][base:base + Q_HEAD_BATCH], dtype=np.float32).reshape(Q_HEAD_BATCH, 1)
        li = np.asarray(buffers["v3"][base:base + Q_HEAD_BATCH], dtype=np.float32).reshape(Q_HEAD_BATCH, 1)
        for sb in range(1, ctx_blocks):
            oi_tmp = load_strided_2d(
                buffers["v1"],
                offset=(gi * MAX_CTX_BLOCKS * Q_HEAD_PAD + sb * Q_HEAD_PAD) * HEAD_DIM,
                rows=Q_HEAD_BATCH,
                cols=HEAD_DIM,
                row_stride=HEAD_DIM,
            ).astype(np.float32)
            cur_mi = np.asarray(
                buffers["v2"][base + sb * Q_HEAD_BATCH:base + (sb + 1) * Q_HEAD_BATCH],
                dtype=np.float32,
            ).reshape(Q_HEAD_BATCH, 1)
            cur_li = np.asarray(
                buffers["v3"][base + sb * Q_HEAD_BATCH:base + (sb + 1) * Q_HEAD_BATCH],
                dtype=np.float32,
            ).reshape(Q_HEAD_BATCH, 1)
            mi_new = np.maximum(mi, cur_mi)
            alpha = np.exp(mi - mi_new)
            beta = np.exp(cur_mi - mi_new)
            li = alpha * li + beta * cur_li
            oi = oi * alpha + oi_tmp * beta
            mi = mi_new
        ctx = oi / li
        output = store_strided_2d(
            output,
            float32_to_bf16(ctx.reshape(1, Q_HEAD_BATCH * HEAD_DIM)),
            offset=gi * Q_HEAD_BATCH * HEAD_DIM,
            row_stride=HIDDEN,
        )
    return buffers, {"v4": output}


def build_out_proj_residual(meta, generator, ints):
    out_block = ints[0]
    row_base = _select_aiv_subblock(ints) * AIV_BATCH
    buffers = {
        "v1": _flat_output(meta, "v1"),
        "v2": make_bf16(generator, meta.elem_counts["v2"], scale=0.05),
        "v3": make_bf16(generator, meta.elem_counts["v3"], scale=0.05),
        "v4": make_bf16(generator, meta.elem_counts["v4"], scale=0.05),
        "v5": make_fp32(generator, meta.elem_counts["v5"], scale=0.05),
    }
    output = np.array(buffers["v1"], copy=True)
    for local_block in range(PROJ_BLOCKS_PER_LAUNCH):
        o0 = (out_block + local_block) * Q_OUT_CHUNK
        acc = np.zeros((AIV_BATCH, Q_OUT_CHUNK), dtype=np.float32)
        for kb in range(HIDDEN // K_CHUNK):
            k0 = kb * K_CHUNK
            attn_chunk = bf16_to_float32(
                load_strided_2d(
                    buffers["v3"],
                    offset=row_base * HIDDEN + k0,
                    rows=AIV_BATCH,
                    cols=K_CHUNK,
                    row_stride=HIDDEN,
                )
            )
            w_chunk = bf16_to_float32(
                load_strided_2d(buffers["v4"], offset=k0 * HIDDEN + o0, rows=K_CHUNK, cols=Q_OUT_CHUNK, row_stride=HIDDEN)
            )
            acc += attn_chunk @ w_chunk
        resid = bf16_to_float32(
            load_strided_2d(
                buffers["v2"],
                offset=row_base * HIDDEN + o0,
                rows=AIV_BATCH,
                cols=Q_OUT_CHUNK,
                row_stride=HIDDEN,
            )
        )
        output = store_strided_2d(output, acc + resid, offset=row_base * HIDDEN + o0, row_stride=HIDDEN)
    return buffers, {"v1": output}


def build_post_rmsnorm(meta, generator, ints):
    del ints
    buffers = {
        "v1": make_fp32(generator, meta.elem_counts["v1"], scale=0.05),
        "v2": _flat_output(meta, "v2"),
        "v3": make_fp32(generator, meta.elem_counts["v3"], scale=0.05),
    }
    sq_sum = np.zeros((BATCH, 1), dtype=np.float32)
    for kb in range(HIDDEN_BLOCKS):
        k0 = kb * K_CHUNK
        resid_chunk = load_strided_2d(buffers["v1"], offset=k0, rows=BATCH, cols=K_CHUNK, row_stride=HIDDEN).astype(np.float32)
        sq_sum += np.sum(resid_chunk * resid_chunk, axis=1, keepdims=True)
    inv_rms = np.reciprocal(np.sqrt(sq_sum * HIDDEN_INV + EPS))
    output = np.array(buffers["v2"], copy=True)
    for kb in range(HIDDEN_BLOCKS):
        k0 = kb * K_CHUNK
        resid_chunk = load_strided_2d(buffers["v1"], offset=k0, rows=BATCH, cols=K_CHUNK, row_stride=HIDDEN).astype(np.float32)
        gamma = load_strided_2d(buffers["v3"], offset=k0, rows=1, cols=K_CHUNK, row_stride=HIDDEN).astype(np.float32)
        post_normed = resid_chunk * inv_rms * gamma
        output = store_strided_2d(output, float32_to_bf16(post_normed), offset=k0, row_stride=HIDDEN)
    return buffers, {"v2": output}


def _build_group_proj(meta, generator, ints):
    block_base, local_block = ints[:2]
    global_o0 = (block_base + local_block) * MLP_OUT_CHUNK
    local_o0 = local_block * MLP_OUT_CHUNK
    buffers = {
        "v1": make_bf16(generator, meta.elem_counts["v1"], scale=0.05),
        "v2": make_bf16(generator, meta.elem_counts["v2"], scale=0.05),
        "v3": _flat_output(meta, "v3"),
    }
    acc = np.zeros((BATCH, MLP_OUT_CHUNK), dtype=np.float32)
    for kb in range(HIDDEN_BLOCKS):
        k0 = kb * K_CHUNK
        post_chunk = bf16_to_float32(load_strided_2d(buffers["v1"], offset=k0, rows=BATCH, cols=K_CHUNK, row_stride=HIDDEN))
        w_chunk = bf16_to_float32(
            load_strided_2d(buffers["v2"], offset=k0 * INTERMEDIATE + global_o0, rows=K_CHUNK, cols=MLP_OUT_CHUNK, row_stride=INTERMEDIATE)
        )
        acc += post_chunk @ w_chunk
    output = np.array(buffers["v3"], copy=True)
    output = store_strided_2d(output, acc, offset=local_o0, row_stride=MLP_GROUP_CHUNK)
    return buffers, {"v3": output}


def build_gate_proj(meta, generator, ints):
    return _build_group_proj(meta, generator, ints)


def build_up_proj(meta, generator, ints):
    return _build_group_proj(meta, generator, ints)


def build_silu(meta, generator, ints):
    block_base, local_block = ints[:2]
    global_o0 = (block_base + local_block) * MLP_OUT_CHUNK
    local_o0 = local_block * MLP_OUT_CHUNK
    buffers = {
        "v1": make_fp32(generator, meta.elem_counts["v1"], scale=0.05),
        "v2": make_fp32(generator, meta.elem_counts["v2"], scale=0.05),
        "v3": _flat_output(meta, "v3"),
    }
    gate = load_strided_2d(buffers["v1"], offset=local_o0, rows=BATCH, cols=MLP_OUT_CHUNK, row_stride=MLP_GROUP_CHUNK).astype(np.float32)
    up = load_strided_2d(buffers["v2"], offset=local_o0, rows=BATCH, cols=MLP_OUT_CHUNK, row_stride=MLP_GROUP_CHUNK).astype(np.float32)
    sigmoid = np.reciprocal(np.float32(1.0) + np.exp(-gate))
    mlp_chunk = gate * sigmoid * up
    output = np.array(buffers["v3"], copy=True)
    output = store_strided_2d(output, float32_to_bf16(mlp_chunk), offset=global_o0, row_stride=INTERMEDIATE)
    return buffers, {"v3": output}


def build_down_proj_residual(meta, generator, ints):
    out_block = ints[0]
    row_base = _select_aiv_subblock(ints) * AIV_BATCH
    buffers = {
        "v1": _flat_output(meta, "v1"),
        "v2": make_fp32(generator, meta.elem_counts["v2"], scale=0.05),
        "v3": make_bf16(generator, meta.elem_counts["v3"], scale=0.05),
        "v4": make_bf16(generator, meta.elem_counts["v4"], scale=0.05),
        "v5": make_fp32(generator, meta.elem_counts["v5"], scale=0.05),
    }
    output = np.array(buffers["v1"], copy=True)
    for local_block in range(PROJ_BLOCKS_PER_LAUNCH):
        d0 = (out_block + local_block) * DOWN_N_CHUNK
        acc = np.zeros((AIV_BATCH, DOWN_N_CHUNK), dtype=np.float32)
        for kb in range(INTERMEDIATE // DOWN_K_CHUNK):
            k0 = kb * DOWN_K_CHUNK
            mlp_chunk = bf16_to_float32(
                load_strided_2d(
                    buffers["v3"],
                    offset=row_base * INTERMEDIATE + k0,
                    rows=AIV_BATCH,
                    cols=DOWN_K_CHUNK,
                    row_stride=INTERMEDIATE,
                )
            )
            w_chunk = bf16_to_float32(
                load_strided_2d(buffers["v4"], offset=k0 * HIDDEN + d0, rows=DOWN_K_CHUNK, cols=DOWN_N_CHUNK, row_stride=HIDDEN)
            )
            acc += mlp_chunk @ w_chunk
        resid = load_strided_2d(
            buffers["v2"],
            offset=row_base * HIDDEN + d0,
            rows=AIV_BATCH,
            cols=DOWN_N_CHUNK,
            row_stride=HIDDEN,
        ).astype(np.float32)
        output = store_strided_2d(
            output,
            float32_to_bf16(acc + resid),
            offset=row_base * HIDDEN + d0,
            row_stride=HIDDEN,
        )
    return buffers, {"v1": output}


BUILDERS = {
    "rmsnorm": build_rmsnorm,
    "q_proj": build_q_proj,
    "kv_proj": build_kv_proj,
    "rope_kv_cache": build_rope_kv_cache,
    "qk_matmul": build_qk_matmul,
    "softmax": build_softmax,
    "sv_matmul": build_sv_matmul,
    "online_softmax": build_online_softmax,
    "out_proj_residual": build_out_proj_residual,
    "post_rmsnorm": build_post_rmsnorm,
    "gate_proj": build_gate_proj,
    "up_proj": build_up_proj,
    "silu": build_silu,
    "down_proj_residual": build_down_proj_residual,
}


def _build_legacy_name_map(meta):
    ordered = list(meta.read_order)
    return {f'v{idx}': name for idx, name in enumerate(ordered, start=1)}


def _with_legacy_meta_aliases(meta, legacy_to_actual):
    elem_counts = dict(meta.elem_counts)
    np_types = dict(meta.np_types)
    for legacy, actual in legacy_to_actual.items():
        if legacy in elem_counts:
            continue
        if actual not in elem_counts or actual not in np_types:
            continue
        elem_counts[legacy] = elem_counts[actual]
        np_types[legacy] = np_types[actual]
    return replace(meta, elem_counts=elem_counts, np_types=np_types)


def _rewrite_legacy_buffer_names(entries, legacy_to_actual):
    rewritten = {}
    for name, value in entries.items():
        rewritten[legacy_to_actual.get(name, name)] = value
    return rewritten


def run_case(case_name: str):
    meta = load_case_meta()
    legacy_to_actual = _build_legacy_name_map(meta)
    meta = _with_legacy_meta_aliases(meta, legacy_to_actual)
    generator = rng()
    ints = load_integer_assignments()
    buffers, golden = BUILDERS[case_name](meta, generator, ints)
    buffers = _rewrite_legacy_buffer_names(buffers, legacy_to_actual)
    golden = _rewrite_legacy_buffer_names(golden, legacy_to_actual)
    write_buffers(meta, buffers)
    write_golden(meta, golden)
