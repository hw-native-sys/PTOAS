# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

import os
import sys
import math
import numpy as np
import ml_dtypes
from st_common import (
    FP4_E1M2,
    FP4_E2M1,
    make_fp4_random,
    zeros_fp4,
    pack_two_fp4 as mx_pack_two_fp4,
)

fp8_e4m3fn = ml_dtypes.float8_e4m3fn
fp8_e5m2 = ml_dtypes.float8_e5m2
fp4_e1m2x2 = FP4_E1M2
fp4_e2m1x2 = FP4_E2M1

np.random.seed(19)


def pack_two_fp4(matrix):
    row, col = matrix.shape
    flat = matrix.flatten()
    high = flat[::2].view(np.uint8)
    low = flat[1::2].view(np.uint8)
    low_bits = (low & 0x0F) << 4
    high_bits = high & 0x0F
    combined = low_bits | high_bits
    return combined.reshape(row, col // 2)


def ceil_align(num, align):
    return (num + align - 1) // align * align


def ceil_div(num, div):
    return (num + div - 1) // div


def ub_bias_cols(n_padded):
    return ceil_align(n_padded, 64)


def pack_mx_lhs_fp8_fractal(matrix):
    m, k = matrix.shape
    if m >= 32 and m % 32 == 0:
        packed = matrix.reshape(m // 32, 32, k // 32, 32).transpose(2, 0, 1, 3)
    else:
        packed = matrix.reshape(m, k // 32, 32).transpose(1, 0, 2)
    return np.ascontiguousarray(packed)


def pack_mx_lhs_fp8_fractal_chunks(matrix, chunk_physical_rows):
    # Split kernels pass each M chunk as an independent GM base. Each chunk must
    # be packed with its own 16-row physical extent, not the full padded M extent.
    packed_chunks = []
    row = 0
    for rows in chunk_physical_rows:
        chunk = matrix[row:row + rows, :]
        if chunk.shape[0] != rows:
            raise ValueError(f"invalid split_m_physical_rows {chunk_physical_rows} for M={matrix.shape[0]}")
        packed_chunks.append(pack_mx_lhs_fp8_fractal(chunk).reshape(-1))
        row += rows
    if row != matrix.shape[0]:
        raise ValueError(f"invalid split_m_physical_rows {chunk_physical_rows} for M={matrix.shape[0]}")
    return np.ascontiguousarray(np.concatenate(packed_chunks))


def pack_mx_rhs_fp8_fractal(matrix):
    k, n = matrix.shape
    packed = matrix.reshape(k // 16, 16, n // 32, 32).transpose(2, 0, 1, 3)
    return np.ascontiguousarray(packed)


def convert_scale_a_format(scale, block_size=16, c0_size_mx=2):
    m, k = scale.shape
    pad_m = (block_size - m % block_size) % block_size
    pad_k = (c0_size_mx - k % c0_size_mx) % c0_size_mx
    if pad_m > 0 or pad_k > 0:
        padded = np.pad(scale, ((0, pad_m), (0, pad_k)), mode='constant', constant_values=0)
    else:
        padded = scale
    m_padded = m + pad_m
    k_padded = k + pad_k
    result = padded.reshape((int(m_padded / block_size), block_size, int(k_padded / c0_size_mx), c0_size_mx))
    result = result.transpose(0, 2, 1, 3)
    result = result.reshape(result.shape[0] * result.shape[1], result.shape[2] * result.shape[3])
    return result


def convert_scale_b_format(scale, block_size=16, c0_size_mx=2, n_pad_to=None):
    k, n = scale.shape
    # RHS MX scale is packed in 16-column groups even when logical N is not
    # 16-aligned, so pad the physical column extent before reshaping.
    target_n = n if n_pad_to is None else max(n, n_pad_to)
    target_n = ceil_align(target_n, block_size)
    pad_n = target_n - n
    pad_k = (c0_size_mx - k % c0_size_mx) % c0_size_mx
    if pad_n > 0 or pad_k > 0:
        padded = np.pad(scale, ((0, pad_k), (0, pad_n)), mode='constant', constant_values=0)
    else:
        padded = scale
    k_padded, n_padded = padded.shape
    result = padded.reshape((int(k_padded / c0_size_mx), c0_size_mx, int(n_padded / 16), 16)).transpose(2, 0, 3, 1)
    result = result.reshape(result.shape[1] * result.shape[3], result.shape[0] * result.shape[2])
    return result


def gen_golden(case):
    atype = case["atype"]
    btype = case["btype"]
    m, k, n = case["m"], case["k"], case["n"]
    m_padded = case["m_padded"]
    n_padded = case["n_padded"]
    is_bias = case["is_bias"]
    is_fp4 = case["is_fp4"]
    split_m_physical_rows = case.get("split_m_physical_rows")

    k_aligned = ceil_align(k, 64)

    if atype == fp4_e2m1x2:
        x1 = make_fp4_random(atype, [m, k])
    elif atype == fp4_e1m2x2:
        x1 = make_fp4_random(atype, [m, k])
    else:
        x1 = np.random.randint(-10, 10, [m, k]).astype(atype)

    if btype == fp4_e2m1x2:
        x2 = make_fp4_random(btype, [k, n])
    elif btype == fp4_e1m2x2:
        x2 = make_fp4_random(btype, [k, n])
    else:
        x2 = np.random.randint(-10, 10, [k, n]).astype(btype)

    if is_fp4:
        x1_padded = zeros_fp4(atype, [m_padded, k_aligned])
        x1_padded[:m, :k] = x1
        x2_padded = zeros_fp4(btype, [k_aligned, n_padded])
        x2_padded[:k, :n] = x2
        x1_bin = mx_pack_two_fp4(x1_padded, atype)
        x2_bin = mx_pack_two_fp4(x2_padded, btype)
    else:
        x1_padded = np.zeros([m_padded, k_aligned], dtype=atype)
        x1_padded[:m, :k] = x1
        x2_padded = np.zeros([k_aligned, n_padded], dtype=btype)
        x2_padded[:k, :n] = x2
        if split_m_physical_rows is not None:
            x1_bin = pack_mx_lhs_fp8_fractal_chunks(x1_padded, split_m_physical_rows)
        else:
            x1_bin = pack_mx_lhs_fp8_fractal(x1_padded)
        x2_bin = pack_mx_rhs_fp8_fractal(x2_padded)

    x1_scale = np.random.randint(127, 130, [m, ceil_div(k_aligned, 32)]).astype(np.uint8)
    x2_scale = np.random.randint(127, 130, [ceil_div(k_aligned, 32), n]).astype(np.uint8)

    x1_mx = 2 ** (x1_scale.astype(np.float64) - 127)
    x2_mx = 2 ** (x2_scale.astype(np.float64) - 127)

    x1_full = np.zeros([m, k_aligned], dtype=np.float64)
    x2_full = np.zeros([k_aligned, n], dtype=np.float64)

    for i in range(k):
        x1_full[:, i] = x1[:, i] * x1_mx[:, i // 32]
        x2_full[i, :] = x2[i, :] * x2_mx[i // 32, :]

    x1_float = x1_full[:, :k]
    x2_float = x2_full[:k, :]

    x1_scale_gm = convert_scale_a_format(x1_scale, 16, 2)
    x2_scale_gm = convert_scale_b_format(x2_scale, 16, 2, n_pad_to=n_padded)

    if is_bias:
        bias = np.random.randint(1, 10, [n]).astype(np.float32)
        golden_valid = np.matmul(x1_float, x2_float).astype(np.float32) + bias
        golden = np.zeros([m_padded, n_padded], dtype=np.float32)
        golden[:m, :n] = golden_valid
        bias_padded = np.zeros([ub_bias_cols(n_padded)], dtype=np.float32)
        bias_padded[:n] = bias
        bias = bias_padded
    else:
        golden_valid = np.matmul(x1_float, x2_float).astype(np.float32)
        golden = np.zeros([m_padded, n_padded], dtype=np.float32)
        golden[:m, :n] = golden_valid

    return x1_bin, x2_bin, x1_scale_gm, x2_scale_gm, bias if is_bias else None, golden


from cases import CASES
from st_common import setup_case_rng, save_case_data

for case in CASES:
    setup_case_rng(case)
    case_dir = case["name"]
    if not os.path.exists(case_dir):
        os.makedirs(case_dir)

    x1, x2, x1_scale, x2_scale, bias, golden = gen_golden(case)

    save_dict = {"input1": x1, "input2": x2, "scale1": x1_scale, "scale2": x2_scale, "golden": golden}
    if bias is not None:
        save_dict["bias"] = bias

    save_case_data(case_dir, save_dict)
    print(f"[INFO] gen_data: {case_dir} m={case['m']} k={case['k']} n={case['n']} is_bias={case['is_bias']} is_fp4={case['is_fp4']}")
