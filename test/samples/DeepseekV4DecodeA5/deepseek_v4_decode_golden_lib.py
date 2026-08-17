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
    load_integer_assignments,
    single_output,
    write_buffers,
    write_golden,
)

RMS_ROWS = 8
RMS_HIDDEN = 4096
RMS_CHUNK = 128
RMS_EPS = np.float32(1e-6)
KV_ROWS = 16
KV_HIDDEN = 128


def _case_bias(case_name: str) -> np.float32:
    total = 0
    for idx, ch in enumerate(case_name):
        total += (idx + 1) * ord(ch)
    return np.float32((total % 97) / 256.0)


def _make_float_payload(count: int, *, bias: np.float32) -> np.ndarray:
    if count <= 0:
        return np.empty((0,), dtype=np.float32)
    base = np.arange(count, dtype=np.float32)
    payload = ((base % 257.0) - 128.0) / 64.0
    payload += bias
    return payload.astype(np.float32, copy=False)


def _buffer_values(meta, name: str, case_name: str):
    count = int(meta.elem_counts[name])
    dtype = np.dtype(meta.np_types[name])
    if name in meta.outputs:
        return np.zeros((count,), dtype=dtype)

    if dtype == np.dtype(np.uint16):
        return float32_to_bf16(_make_float_payload(count, bias=_case_bias(case_name)))

    if np.issubdtype(dtype, np.floating):
        return _make_float_payload(count, bias=_case_bias(case_name)).astype(dtype, copy=False)

    if np.issubdtype(dtype, np.bool_):
        return np.zeros((count,), dtype=dtype)

    if np.issubdtype(dtype, np.integer):
        return np.zeros((count,), dtype=dtype)

    raise TypeError(f'unsupported dtype for {name}: {dtype}')


def _require_min_count(meta, name: str, expected: int) -> None:
    actual = int(meta.elem_counts[name])
    if actual < expected:
        raise ValueError(f'{name}: expected at least {expected} elements, got {actual}')


def _build_kv_hadamard(meta, buffers, ints):
    del ints
    out_name = single_output(meta)
    if len(meta.inputs) != 2:
        raise ValueError(f'kv_hadamard: expected 2 inputs, got {meta.inputs}')
    lhs_name, rhs_name = meta.inputs
    _require_min_count(meta, lhs_name, KV_ROWS * KV_HIDDEN)
    _require_min_count(meta, rhs_name, KV_HIDDEN * KV_HIDDEN)
    _require_min_count(meta, out_name, KV_ROWS * KV_HIDDEN)

    lhs = bf16_to_float32(buffers[lhs_name][:KV_ROWS * KV_HIDDEN]).reshape(KV_ROWS, KV_HIDDEN)
    rhs = bf16_to_float32(buffers[rhs_name][:KV_HIDDEN * KV_HIDDEN]).reshape(KV_HIDDEN, KV_HIDDEN)
    golden = np.zeros(meta.elem_counts[out_name], dtype=meta.np_types[out_name])
    golden[:KV_ROWS * KV_HIDDEN] = (lhs @ rhs).reshape(-1)
    return {out_name: golden}


def _build_rms_norm(meta, buffers, ints):
    out_name = single_output(meta)
    if len(meta.inputs) != 2:
        raise ValueError(f'rms_norm: expected 2 inputs, got {meta.inputs}')
    if len(ints) < 2:
        raise ValueError(f'rms_norm: expected block_idx/block_num, got {ints}')
    block_idx, block_num = ints[:2]
    if block_idx != 0 or block_num <= 0:
        raise ValueError(f'rms_norm: expected block_idx=0 and block_num>0, got {ints[:2]}')

    x_name, weight_name = meta.inputs
    row_offset = block_idx * RMS_ROWS * RMS_HIDDEN
    required_rows = row_offset + RMS_ROWS * RMS_HIDDEN
    _require_min_count(meta, x_name, required_rows)
    _require_min_count(meta, out_name, required_rows)
    _require_min_count(meta, weight_name, RMS_HIDDEN)

    x = bf16_to_float32(buffers[x_name][row_offset:required_rows]).reshape(RMS_ROWS, RMS_HIDDEN)
    weight = bf16_to_float32(buffers[weight_name][:RMS_HIDDEN]).reshape(1, RMS_HIDDEN)
    sq_sum = np.zeros((RMS_ROWS, 1), dtype=np.float32)
    for col in range(0, RMS_HIDDEN, RMS_CHUNK):
        chunk = x[:, col:col + RMS_CHUNK]
        sq_sum += np.sum(chunk * chunk, axis=1, keepdims=True, dtype=np.float32)
    inv_rms = np.reciprocal(np.sqrt(sq_sum * np.float32(1.0 / RMS_HIDDEN) + RMS_EPS))

    golden = np.zeros(meta.elem_counts[out_name], dtype=meta.np_types[out_name])
    for col in range(0, RMS_HIDDEN, RMS_CHUNK):
        normalized = x[:, col:col + RMS_CHUNK] * inv_rms * weight[:, col:col + RMS_CHUNK]
        for row in range(RMS_ROWS):
            start = row_offset + row * RMS_HIDDEN + col
            golden[start:start + RMS_CHUNK] = float32_to_bf16(normalized[row])
    return {out_name: golden}


CORRECTNESS_BUILDERS = {
    'kv_hadamard': _build_kv_hadamard,
    'rms_norm': _build_rms_norm,
}


def run_case(case_name: str):
    meta = load_case_meta()
    buffers = {
        name: _buffer_values(meta, name, case_name)
        for name in meta.read_order
    }
    write_buffers(meta, buffers)
    builder = CORRECTNESS_BUILDERS.get(case_name)
    if builder is not None:
        golden = builder(meta, buffers, load_integer_assignments())
        write_golden(meta, golden)
