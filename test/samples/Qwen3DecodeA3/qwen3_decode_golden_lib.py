#!/usr/bin/python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import numpy as np

from validation_runtime import (
    bf16_to_float32,
    float32_to_bf16,
    load_case_meta,
    load_int32_assignments,
    load_strided_2d,
    rng,
    store_strided_2d,
    write_buffers,
    write_golden,
)

BATCH_TILE = 16
HIDDEN = 8192
KV_HIDDEN = 1024
MAX_SEQ = 4096
NUM_KV_HEADS = 8
HEAD_DIM = 128
HALF_DIM = 64
Q_PER_KV = 8
Q_HEAD_BATCH = 8
Q_HEAD_PAD = 16
SEQ_TILE = 64
SCOPE1_K_CHUNK = 512
Q_OUT_CHUNK = 64
KV_OUT_CHUNK = 64
K_CHUNK = 128
MLP_OUT_CHUNK = 256
SCOPE1_HIDDEN_BLOCKS = HIDDEN // SCOPE1_K_CHUNK
HIDDEN_BLOCKS = HIDDEN // K_CHUNK
KV_OUT_BLOCKS = KV_HIDDEN // KV_OUT_CHUNK
Q_OUT_BLOCKS = HIDDEN // Q_OUT_CHUNK
MLP_OUT_BLOCKS = 25600 // MLP_OUT_CHUNK
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


def round_fp32_to_bf16_fp32(values: np.ndarray) -> np.ndarray:
    return bf16_to_float32(float32_to_bf16(values))


def store_flat(buffer, values, *, offset: int):
    flat = np.asarray(buffer).reshape(-1)
    arr = np.asarray(values).reshape(-1)
    end = offset + arr.size
    if end > flat.size:
        raise ValueError(f'flat store out of bounds: [{offset}:{end}] > {flat.size}')
    flat[offset:end] = arr
    return flat


def make_padded_rows_bf16(generator, count: int, *, cols: int, rows_per_group: int, active_rows: int, scale: float = 0.05, positive: bool = False) -> np.ndarray:
    out = make_bf16(generator, count, scale=scale, positive=positive)
    rows = out.size // cols
    for row in range(rows):
        if row % rows_per_group >= active_rows:
            start = row * cols
            out[start:start + cols] = 0
    return out


def build_case_0(meta, generator, ints):
    b0 = ints[0]
    buffers = {
        'v1': make_bf16(generator, meta.elem_counts['v1'], scale=0.05),
        'v2': make_fp32(generator, meta.elem_counts['v2'], scale=0.05),
        'v3': np.zeros(meta.elem_counts['v3'], dtype=meta.np_types['v3']),
    }
    sq_sum = np.zeros((BATCH_TILE, 1), dtype=np.float32)
    for kb in range(SCOPE1_HIDDEN_BLOCKS):
        k0 = kb * SCOPE1_K_CHUNK
        x_chunk = bf16_to_float32(
            load_strided_2d(buffers['v1'], offset=b0 * HIDDEN + k0, rows=BATCH_TILE, cols=SCOPE1_K_CHUNK, row_stride=HIDDEN)
        )
        sq_sum += np.sum(x_chunk * x_chunk, axis=1, keepdims=True)
    inv_rms = np.reciprocal(np.sqrt(sq_sum * HIDDEN_INV + EPS))
    output = np.array(buffers['v3'], copy=True)
    for kb in range(SCOPE1_HIDDEN_BLOCKS):
        k0 = kb * SCOPE1_K_CHUNK
        x_chunk = bf16_to_float32(
            load_strided_2d(buffers['v1'], offset=b0 * HIDDEN + k0, rows=BATCH_TILE, cols=SCOPE1_K_CHUNK, row_stride=HIDDEN)
        )
        gamma = load_strided_2d(buffers['v2'], offset=k0, rows=1, cols=SCOPE1_K_CHUNK, row_stride=HIDDEN).astype(np.float32)
        normed = x_chunk * inv_rms * gamma
        output = store_strided_2d(output, float32_to_bf16(normed), offset=k0, row_stride=HIDDEN)
    return buffers, {'v3': output}


def build_case_1(meta, generator, ints):
    b0, ob = ints[:2]
    buffers = {
        'v1': make_bf16(generator, meta.elem_counts['v1'], scale=0.05),
        'v2': np.zeros(meta.elem_counts['v2'], dtype=meta.np_types['v2']),
        'v3': make_bf16(generator, meta.elem_counts['v3'], scale=0.05),
    }
    output = np.array(buffers['v2'], copy=True)
    for ob_ci in range(4):
        block = ob * 4 + ob_ci
        if block >= Q_OUT_BLOCKS:
            continue
        q0 = block * Q_OUT_CHUNK
        acc = np.zeros((BATCH_TILE, Q_OUT_CHUNK), dtype=np.float32)
        for kb in range(SCOPE1_HIDDEN_BLOCKS):
            k0 = kb * SCOPE1_K_CHUNK
            normed = bf16_to_float32(load_strided_2d(buffers['v1'], offset=k0, rows=BATCH_TILE, cols=SCOPE1_K_CHUNK, row_stride=HIDDEN))
            w_chunk = bf16_to_float32(
                load_strided_2d(buffers['v3'], offset=k0 * HIDDEN + q0, rows=SCOPE1_K_CHUNK, cols=Q_OUT_CHUNK, row_stride=HIDDEN)
            )
            acc += normed @ w_chunk
        output = store_strided_2d(output, acc, offset=b0 * HIDDEN + q0, row_stride=HIDDEN)
    return buffers, {'v2': output}


def build_case_2(meta, generator, ints):
    b0, ob = ints[:2]
    buffers = {
        'v1': np.zeros(meta.elem_counts['v1'], dtype=meta.np_types['v1']),
        'v2': make_bf16(generator, meta.elem_counts['v2'], scale=0.05),
        'v3': np.zeros(meta.elem_counts['v3'], dtype=meta.np_types['v3']),
        'v4': make_bf16(generator, meta.elem_counts['v4'], scale=0.05),
        'v5': make_bf16(generator, meta.elem_counts['v5'], scale=0.05),
    }
    k_proj = np.array(buffers['v1'], copy=True)
    v_proj = np.array(buffers['v3'], copy=True)
    for ob_ci in range(4):
        block = ob * 4 + ob_ci
        if block >= KV_OUT_BLOCKS:
            continue
        kv0 = block * KV_OUT_CHUNK
        k_acc = np.zeros((BATCH_TILE, KV_OUT_CHUNK), dtype=np.float32)
        v_acc = np.zeros((BATCH_TILE, KV_OUT_CHUNK), dtype=np.float32)
        for kb in range(SCOPE1_HIDDEN_BLOCKS):
            k0 = kb * SCOPE1_K_CHUNK
            normed = bf16_to_float32(load_strided_2d(buffers['v2'], offset=k0, rows=BATCH_TILE, cols=SCOPE1_K_CHUNK, row_stride=HIDDEN))
            wk_chunk = bf16_to_float32(
                load_strided_2d(buffers['v4'], offset=k0 * KV_HIDDEN + kv0, rows=SCOPE1_K_CHUNK, cols=KV_OUT_CHUNK, row_stride=KV_HIDDEN)
            )
            wv_chunk = bf16_to_float32(
                load_strided_2d(buffers['v5'], offset=k0 * KV_HIDDEN + kv0, rows=SCOPE1_K_CHUNK, cols=KV_OUT_CHUNK, row_stride=KV_HIDDEN)
            )
            k_acc += normed @ wk_chunk
            v_acc += normed @ wv_chunk
        k_proj = store_strided_2d(k_proj, k_acc, offset=b0 * KV_HIDDEN + kv0, row_stride=KV_HIDDEN)
        v_proj = store_strided_2d(v_proj, v_acc, offset=b0 * KV_HIDDEN + kv0, row_stride=KV_HIDDEN)
    return buffers, {'v1': k_proj, 'v3': v_proj}


def build_case_3(meta, generator, ints):
    del ints
    buffers = {
        'v1': make_bf16(generator, meta.elem_counts['v1'], scale=0.05),
    }
    output = np.array(buffers['v1'], copy=True)
    rows = output.size // HEAD_DIM
    for row in range(rows):
        if row % Q_HEAD_PAD >= Q_HEAD_BATCH:
            start = row * HEAD_DIM
            output[start:start + HEAD_DIM] = 0
    return buffers, {'v1': output}


def build_case_4(meta, generator, ints):
    b, group_base, pos = ints[:3]
    buffers = {
        'v1': make_padded_rows_bf16(generator, meta.elem_counts['v1'], cols=HEAD_DIM, rows_per_group=Q_HEAD_PAD, active_rows=Q_HEAD_BATCH, scale=0.05),
        'v2': make_fp32(generator, meta.elem_counts['v2'], scale=0.05),
        'v3': make_fp32(generator, meta.elem_counts['v3'], scale=0.05),
        'v4': make_bf16(generator, meta.elem_counts['v4'], scale=0.05),
        'v5': make_fp32(generator, meta.elem_counts['v5'], scale=0.05),
        'v6': make_fp32(generator, meta.elem_counts['v6'], scale=0.05),
        'v7': make_fp32(generator, meta.elem_counts['v7'], scale=0.05),
        'v8': make_fp32(generator, meta.elem_counts['v8'], scale=0.05),
        'v9': make_bf16(generator, meta.elem_counts['v9'], scale=0.05),
        'v10': make_fp32(generator, meta.elem_counts['v10'], scale=0.05),
    }
    out_q = np.array(buffers['v1'], copy=True)
    out_k = np.array(buffers['v4'], copy=True)
    out_v = np.array(buffers['v9'], copy=True)
    cos_lo = np.asarray(buffers['v2'][:HALF_DIM], dtype=np.float32)
    cos_hi = np.asarray(buffers['v3'][:HALF_DIM], dtype=np.float32)
    sin_lo = np.asarray(buffers['v7'][:HALF_DIM], dtype=np.float32)
    sin_hi = np.asarray(buffers['v8'][:HALF_DIM], dtype=np.float32)
    k_proj = load_strided_2d(buffers['v5'], offset=b * KV_HIDDEN, rows=1, cols=KV_HIDDEN, row_stride=KV_HIDDEN).astype(np.float32)
    q_proj = load_strided_2d(buffers['v6'], offset=b * HIDDEN, rows=1, cols=HIDDEN, row_stride=HIDDEN).astype(np.float32)
    v_proj = load_strided_2d(buffers['v10'], offset=b * KV_HIDDEN, rows=1, cols=KV_HIDDEN, row_stride=KV_HIDDEN).astype(np.float32)
    for ki in range(NUM_KV_HEADS):
        kvh = group_base * NUM_KV_HEADS + ki
        if kvh >= NUM_KV_HEADS:
            continue
        kv_col = kvh * HEAD_DIM
        k_lo = k_proj[0, kv_col:kv_col + HALF_DIM]
        k_hi = k_proj[0, kv_col + HALF_DIM:kv_col + HEAD_DIM]
        rot_lo = k_lo * cos_lo - k_hi * sin_lo
        rot_hi = k_hi * cos_hi + k_lo * sin_hi
        cache_row = b * NUM_KV_HEADS * MAX_SEQ + kvh * MAX_SEQ + pos
        out_k = store_strided_2d(out_k, float32_to_bf16(rot_lo.reshape(1, HALF_DIM)), offset=cache_row * HEAD_DIM, row_stride=HEAD_DIM)
        out_k = store_strided_2d(out_k, float32_to_bf16(rot_hi.reshape(1, HALF_DIM)), offset=cache_row * HEAD_DIM + HALF_DIM, row_stride=HEAD_DIM)
        v_chunk = v_proj[0, kv_col:kv_col + HEAD_DIM]
        out_v = store_strided_2d(out_v, float32_to_bf16(v_chunk.reshape(1, HEAD_DIM)), offset=cache_row * HEAD_DIM, row_stride=HEAD_DIM)
        q_base = kvh * Q_PER_KV
        for qi in range(Q_HEAD_BATCH):
            q_col = (q_base + qi) * HEAD_DIM
            q_lo = q_proj[0, q_col:q_col + HALF_DIM]
            q_hi = q_proj[0, q_col + HALF_DIM:q_col + HEAD_DIM]
            q_rot_lo = q_lo * cos_lo - q_hi * sin_lo
            q_rot_hi = q_hi * cos_hi + q_lo * sin_hi
            row = b * NUM_KV_HEADS * Q_HEAD_PAD + kvh * Q_HEAD_PAD + qi
            out_q = store_strided_2d(out_q, float32_to_bf16(q_rot_lo.reshape(1, HALF_DIM)), offset=row * HEAD_DIM, row_stride=HEAD_DIM)
            out_q = store_strided_2d(out_q, float32_to_bf16(q_rot_hi.reshape(1, HALF_DIM)), offset=row * HEAD_DIM + HALF_DIM, row_stride=HEAD_DIM)
    return buffers, {'v1': out_q, 'v4': out_k, 'v9': out_v}


def build_case_5(meta, generator, ints):
    b, ctx_blocks, kvh, sb_group = ints[:4]
    buffers = {
        'v1': np.zeros(meta.elem_counts['v1'], dtype=meta.np_types['v1']),
        'v2': make_bf16(generator, meta.elem_counts['v2'], scale=0.05),
        'v3': make_padded_rows_bf16(generator, meta.elem_counts['v3'], cols=HEAD_DIM, rows_per_group=Q_HEAD_PAD, active_rows=Q_HEAD_BATCH, scale=0.05),
    }
    output = np.array(buffers['v1'], copy=True)
    q_padded = bf16_to_float32(load_strided_2d(buffers['v3'], offset=0, rows=Q_HEAD_PAD, cols=HEAD_DIM, row_stride=HEAD_DIM))
    for local_sb in range(SEQ_TILE):
        sb = sb_group * SEQ_TILE + local_sb
        if sb >= ctx_blocks:
            continue
        cache_offset = (b * NUM_KV_HEADS * MAX_SEQ + kvh * MAX_SEQ + sb * SEQ_TILE) * HEAD_DIM
        k_tile = bf16_to_float32(load_strided_2d(buffers['v2'], offset=cache_offset, rows=SEQ_TILE, cols=HEAD_DIM, row_stride=HEAD_DIM))
        raw_scores = q_padded @ k_tile.T
        output = store_strided_2d(output, raw_scores, offset=sb * Q_HEAD_PAD * SEQ_TILE, row_stride=SEQ_TILE)
    return buffers, {'v1': output}


def build_case_6(meta, generator, ints):
    ctx_blocks, ctx_len, sb_group = ints[:3]
    buffers = {
        'v1': np.zeros(meta.elem_counts['v1'], dtype=meta.np_types['v1']),
        'v2': np.zeros(meta.elem_counts['v2'], dtype=meta.np_types['v2']),
        'v3': np.zeros(meta.elem_counts['v3'], dtype=meta.np_types['v3']),
        'v4': make_fp32(generator, meta.elem_counts['v4'], scale=0.05),
    }
    out_li = np.array(buffers['v1'], copy=True)
    out_mi = np.array(buffers['v2'], copy=True)
    out_exp = np.array(buffers['v3'], copy=True)
    for local_sb in range(SEQ_TILE):
        sb = sb_group * SEQ_TILE + local_sb
        if sb >= ctx_blocks:
            continue
        valid_len = min(SEQ_TILE, max(ctx_len - sb * SEQ_TILE, 0))
        scores_valid = load_strided_2d(buffers['v4'], offset=sb * Q_HEAD_PAD * SEQ_TILE, rows=Q_HEAD_BATCH, cols=SEQ_TILE, row_stride=SEQ_TILE).astype(np.float32)
        scores = np.full((Q_HEAD_BATCH, SEQ_TILE), NEG_INF, dtype=np.float32)
        if valid_len > 0:
            scores[:, :valid_len] = scores_valid[:, :valid_len]
        scores = scores * ATTN_SCALE
        cur_mi = np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores - cur_mi)
        exp_scores_bf16 = float32_to_bf16(exp_scores)
        cur_li = np.sum(bf16_to_float32(exp_scores_bf16), axis=1, keepdims=True)
        out_exp = store_strided_2d(out_exp, exp_scores_bf16, offset=sb * Q_HEAD_PAD * SEQ_TILE, row_stride=SEQ_TILE)
        out_mi[sb * Q_HEAD_BATCH:(sb + 1) * Q_HEAD_BATCH] = cur_mi.reshape(-1)
        out_li[sb * Q_HEAD_BATCH:(sb + 1) * Q_HEAD_BATCH] = cur_li.reshape(-1)
    return buffers, {'v1': out_li, 'v2': out_mi, 'v3': out_exp}


def build_case_7(meta, generator, ints):
    b, ctx_blocks, kvh, sb_group = ints[:4]
    buffers = {
        'v1': make_padded_rows_bf16(generator, meta.elem_counts['v1'], cols=SEQ_TILE, rows_per_group=Q_HEAD_PAD, active_rows=Q_HEAD_BATCH, scale=0.05, positive=True),
        'v2': np.zeros(meta.elem_counts['v2'], dtype=meta.np_types['v2']),
        'v3': make_bf16(generator, meta.elem_counts['v3'], scale=0.05),
    }
    output = np.array(buffers['v2'], copy=True)
    for local_sb in range(SEQ_TILE):
        sb = sb_group * SEQ_TILE + local_sb
        if sb >= ctx_blocks:
            continue
        exp_tile = bf16_to_float32(load_strided_2d(buffers['v1'], offset=sb * Q_HEAD_PAD * SEQ_TILE, rows=Q_HEAD_PAD, cols=SEQ_TILE, row_stride=SEQ_TILE))
        cache_offset = (b * NUM_KV_HEADS * MAX_SEQ + kvh * MAX_SEQ + sb * SEQ_TILE) * HEAD_DIM
        v_tile = bf16_to_float32(load_strided_2d(buffers['v3'], offset=cache_offset, rows=SEQ_TILE, cols=HEAD_DIM, row_stride=HEAD_DIM))
        oi_tmp = exp_tile @ v_tile
        output = store_strided_2d(output, oi_tmp, offset=sb * Q_HEAD_PAD * HEAD_DIM, row_stride=HEAD_DIM)
    return buffers, {'v2': output}


def build_case_8(meta, generator, ints):
    ctx_blocks, q_base = ints[:2]
    buffers = {
        'v1': make_fp32(generator, meta.elem_counts['v1'], scale=0.05, positive=True),
        'v2': make_fp32(generator, meta.elem_counts['v2'], scale=0.05),
        'v3': make_fp32(generator, meta.elem_counts['v3'], scale=0.05),
        'v4': np.zeros(meta.elem_counts['v4'], dtype=meta.np_types['v4']),
    }
    oi = load_strided_2d(buffers['v3'], offset=0, rows=Q_HEAD_BATCH, cols=HEAD_DIM, row_stride=HEAD_DIM).astype(np.float32)
    mi = np.asarray(buffers['v2'][:Q_HEAD_BATCH], dtype=np.float32).reshape(Q_HEAD_BATCH, 1)
    li = np.asarray(buffers['v1'][:Q_HEAD_BATCH], dtype=np.float32).reshape(Q_HEAD_BATCH, 1)
    for sb in range(1, ctx_blocks):
        oi_tmp = load_strided_2d(buffers['v3'], offset=sb * Q_HEAD_PAD * HEAD_DIM, rows=Q_HEAD_BATCH, cols=HEAD_DIM, row_stride=HEAD_DIM).astype(np.float32)
        cur_mi = np.asarray(buffers['v2'][sb * Q_HEAD_BATCH:(sb + 1) * Q_HEAD_BATCH], dtype=np.float32).reshape(Q_HEAD_BATCH, 1)
        cur_li = np.asarray(buffers['v1'][sb * Q_HEAD_BATCH:(sb + 1) * Q_HEAD_BATCH], dtype=np.float32).reshape(Q_HEAD_BATCH, 1)
        mi_new = np.maximum(mi, cur_mi)
        alpha = np.exp(mi - mi_new)
        beta = np.exp(cur_mi - mi_new)
        li = alpha * li + beta * cur_li
        oi = oi * alpha + oi_tmp * beta
        mi = mi_new
    ctx = oi / li
    output = np.array(buffers['v4'], copy=True)
    output = store_strided_2d(output, float32_to_bf16(ctx.reshape(1, Q_HEAD_BATCH * HEAD_DIM)), offset=q_base * HEAD_DIM, row_stride=HIDDEN)
    return buffers, {'v4': output}


def build_case_9(meta, generator, ints):
    b0, o0 = ints[:2]
    buffers = {
        'v1': make_bf16(generator, meta.elem_counts['v1'], scale=0.05),
        'v2': make_bf16(generator, meta.elem_counts['v2'], scale=0.05),
        'v3': np.zeros(meta.elem_counts['v3'], dtype=meta.np_types['v3']),
    }
    acc = np.zeros((BATCH_TILE, Q_OUT_CHUNK), dtype=np.float32)
    for kb in range(HIDDEN_BLOCKS):
        k0 = kb * K_CHUNK
        attn_chunk = bf16_to_float32(load_strided_2d(buffers['v1'], offset=b0 * HIDDEN + k0, rows=BATCH_TILE, cols=K_CHUNK, row_stride=HIDDEN))
        w_chunk = bf16_to_float32(load_strided_2d(buffers['v2'], offset=k0 * HIDDEN + o0, rows=K_CHUNK, cols=Q_OUT_CHUNK, row_stride=HIDDEN))
        acc += attn_chunk @ w_chunk
    output = np.array(buffers['v3'], copy=True)
    output = store_strided_2d(output, acc, offset=0, row_stride=Q_OUT_CHUNK)
    return buffers, {'v3': output}


def build_case_10(meta, generator, ints):
    b0, o0 = ints[:2]
    buffers = {
        'v1': make_bf16(generator, meta.elem_counts['v1'], scale=0.05),
        'v2': make_fp32(generator, meta.elem_counts['v2'], scale=0.05),
        'v3': np.zeros(meta.elem_counts['v3'], dtype=meta.np_types['v3']),
    }
    resid = bf16_to_float32(load_strided_2d(buffers['v1'], offset=b0 * HIDDEN + o0, rows=BATCH_TILE, cols=Q_OUT_CHUNK, row_stride=HIDDEN))
    o_acc = load_strided_2d(buffers['v2'], offset=0, rows=BATCH_TILE, cols=Q_OUT_CHUNK, row_stride=Q_OUT_CHUNK).astype(np.float32)
    output = np.array(buffers['v3'], copy=True)
    output = store_strided_2d(output, o_acc + resid, offset=o0, row_stride=HIDDEN)
    return buffers, {'v3': output}


def build_case_11(meta, generator, ints):
    del ints
    buffers = {
        'v1': np.zeros(meta.elem_counts['v1'], dtype=meta.np_types['v1']),
        'v2': make_fp32(generator, meta.elem_counts['v2'], scale=0.05),
        'v3': make_fp32(generator, meta.elem_counts['v3'], scale=0.05),
    }
    sq_sum = np.zeros((BATCH_TILE, 1), dtype=np.float32)
    for kb in range(HIDDEN_BLOCKS):
        k0 = kb * K_CHUNK
        resid_chunk = load_strided_2d(buffers['v3'], offset=k0, rows=BATCH_TILE, cols=K_CHUNK, row_stride=HIDDEN).astype(np.float32)
        sq_sum += np.sum(resid_chunk * resid_chunk, axis=1, keepdims=True)
    inv_rms = np.reciprocal(np.sqrt(sq_sum * HIDDEN_INV + EPS))
    output = np.array(buffers['v1'], copy=True)
    for kb in range(HIDDEN_BLOCKS):
        k0 = kb * K_CHUNK
        resid_chunk = load_strided_2d(buffers['v3'], offset=k0, rows=BATCH_TILE, cols=K_CHUNK, row_stride=HIDDEN).astype(np.float32)
        gamma = load_strided_2d(buffers['v2'], offset=k0, rows=1, cols=K_CHUNK, row_stride=HIDDEN).astype(np.float32)
        post_normed = resid_chunk * inv_rms * gamma
        output = store_strided_2d(output, float32_to_bf16(post_normed), offset=k0, row_stride=HIDDEN)
    return buffers, {'v1': output}


def build_case_12(meta, generator, ints):
    o0 = ints[0]
    buffers = {
        'v1': make_bf16(generator, meta.elem_counts['v1'], scale=0.05),
        'v2': make_bf16(generator, meta.elem_counts['v2'], scale=0.05),
        'v3': np.zeros(meta.elem_counts['v3'], dtype=meta.np_types['v3']),
    }
    acc = np.zeros((BATCH_TILE, MLP_OUT_CHUNK), dtype=np.float32)
    for kb in range(HIDDEN_BLOCKS):
        k0 = kb * K_CHUNK
        post_chunk = bf16_to_float32(load_strided_2d(buffers['v1'], offset=k0, rows=BATCH_TILE, cols=K_CHUNK, row_stride=HIDDEN))
        w_chunk = bf16_to_float32(load_strided_2d(buffers['v2'], offset=k0 * 25600 + o0, rows=K_CHUNK, cols=MLP_OUT_CHUNK, row_stride=25600))
        acc += post_chunk @ w_chunk
    output = np.array(buffers['v3'], copy=True)
    output = store_strided_2d(output, acc, offset=0, row_stride=MLP_OUT_CHUNK)
    return buffers, {'v3': output}


def build_case_13(meta, generator, ints):
    return build_case_12(meta, generator, ints)


def build_case_14(meta, generator, ints):
    del ints
    buffers = {
        'v1': make_fp32(generator, meta.elem_counts['v1'], scale=0.05),
        'v2': make_fp32(generator, meta.elem_counts['v2'], scale=0.05),
        'v3': np.zeros(meta.elem_counts['v3'], dtype=meta.np_types['v3']),
    }
    gate = load_strided_2d(buffers['v1'], offset=0, rows=BATCH_TILE, cols=MLP_OUT_CHUNK, row_stride=MLP_OUT_CHUNK).astype(np.float32)
    up = load_strided_2d(buffers['v2'], offset=0, rows=BATCH_TILE, cols=MLP_OUT_CHUNK, row_stride=MLP_OUT_CHUNK).astype(np.float32)
    sigmoid = np.reciprocal(1.0 + np.exp(-gate))
    mlp_chunk = gate * sigmoid * up
    output = np.array(buffers['v3'], copy=True)
    output = store_strided_2d(output, float32_to_bf16(mlp_chunk), offset=0, row_stride=25600)
    return buffers, {'v3': output}


def build_case_15(meta, generator, ints):
    d0 = ints[0]
    buffers = {
        'v1': make_bf16(generator, meta.elem_counts['v1'], scale=0.05),
        'v2': make_bf16(generator, meta.elem_counts['v2'], scale=0.05),
        'v3': np.zeros(meta.elem_counts['v3'], dtype=meta.np_types['v3']),
    }
    acc = np.zeros((BATCH_TILE, K_CHUNK), dtype=np.float32)
    for ob in range(MLP_OUT_BLOCKS):
        o0 = ob * MLP_OUT_CHUNK
        mlp_chunk = bf16_to_float32(load_strided_2d(buffers['v1'], offset=o0, rows=BATCH_TILE, cols=MLP_OUT_CHUNK, row_stride=25600))
        w_chunk = bf16_to_float32(load_strided_2d(buffers['v2'], offset=o0 * HIDDEN + d0, rows=MLP_OUT_CHUNK, cols=K_CHUNK, row_stride=HIDDEN))
        acc += mlp_chunk @ w_chunk
    output = np.array(buffers['v3'], copy=True)
    output = store_strided_2d(output, acc, offset=0, row_stride=K_CHUNK)
    return buffers, {'v3': output}


def build_case_16(meta, generator, ints):
    b0, d0 = ints[:2]
    buffers = {
        'v1': make_fp32(generator, meta.elem_counts['v1'], scale=0.05),
        'v2': np.zeros(meta.elem_counts['v2'], dtype=meta.np_types['v2']),
        'v3': make_fp32(generator, meta.elem_counts['v3'], scale=0.05),
    }
    down_acc = load_strided_2d(buffers['v1'], offset=0, rows=BATCH_TILE, cols=K_CHUNK, row_stride=K_CHUNK).astype(np.float32)
    resid = load_strided_2d(buffers['v3'], offset=d0, rows=BATCH_TILE, cols=K_CHUNK, row_stride=HIDDEN).astype(np.float32)
    output = np.array(buffers['v2'], copy=True)
    output = store_strided_2d(output, float32_to_bf16(down_acc + resid), offset=b0 * HIDDEN + d0, row_stride=HIDDEN)
    return buffers, {'v2': output}


BUILDERS = {
    'qwen3_decode_incore_0': build_case_0,
    'qwen3_decode_incore_1': build_case_1,
    'qwen3_decode_incore_2': build_case_2,
    'qwen3_decode_incore_3': build_case_3,
    'qwen3_decode_incore_4': build_case_4,
    'qwen3_decode_incore_5': build_case_5,
    'qwen3_decode_incore_6': build_case_6,
    'qwen3_decode_incore_7': build_case_7,
    'qwen3_decode_incore_8': build_case_8,
    'qwen3_decode_incore_9': build_case_9,
    'qwen3_decode_incore_10': build_case_10,
    'qwen3_decode_incore_11': build_case_11,
    'qwen3_decode_incore_12': build_case_12,
    'qwen3_decode_incore_13': build_case_13,
    'qwen3_decode_incore_14': build_case_14,
    'qwen3_decode_incore_15': build_case_15,
    'qwen3_decode_incore_16': build_case_16,
}


def run_case(case_name: str):
    meta = load_case_meta()
    generator = rng()
    ints = load_int32_assignments()
    buffers, golden = BUILDERS[case_name](meta, generator, ints)
    write_buffers(meta, buffers)
    write_golden(meta, golden)

