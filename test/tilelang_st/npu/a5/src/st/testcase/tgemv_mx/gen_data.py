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


def convert_scale_b_format(scale, block_size=16, c0_size_mx=2):
    k, n = scale.shape
    pad_n = (block_size - n % block_size) % block_size
    pad_k = (c0_size_mx - k % c0_size_mx) % c0_size_mx
    if pad_n > 0 or pad_k > 0:
        padded = np.pad(scale, ((0, pad_k), (0, pad_n)), mode='constant', constant_values=0)
    else:
        padded = scale
    k_padded, n_padded = padded.shape
    result = padded.reshape((int(k_padded / c0_size_mx), c0_size_mx, int(n_padded / 16), 16)).transpose(2, 0, 3, 1)
    result = result.reshape(result.shape[1] * result.shape[3], result.shape[0] * result.shape[2])
    return result


def convert_scale_a_nd(scale):
    return scale.copy()


def convert_scale_a_row_major_padded(scale, block_size=16, c0_size_mx=2):
    m, k = scale.shape
    pad_m = (block_size - m % block_size) % block_size
    pad_k = (c0_size_mx - k % c0_size_mx) % c0_size_mx
    if pad_m > 0 or pad_k > 0:
        return np.pad(scale, ((0, pad_m), (0, pad_k)), mode='constant', constant_values=0)
    return scale.copy()


def convert_scale_b_nd(scale):
    return scale.reshape((scale.shape[0] // 2, 2, scale.shape[1])).transpose(0, 2, 1).copy()


def convert_scale_b_nd_padded(scale, block_size=16, c0_size_mx=2):
    k, n = scale.shape
    pad_n = (block_size - n % block_size) % block_size
    pad_k = (c0_size_mx - k % c0_size_mx) % c0_size_mx
    if pad_n > 0 or pad_k > 0:
        scale = np.pad(scale, ((0, pad_k), (0, pad_n)), mode='constant', constant_values=0)
    return scale.reshape((scale.shape[0] // c0_size_mx, c0_size_mx, scale.shape[1])).transpose(0, 2, 1).copy()


def convert_scale_b_raw(scale):
    return scale.copy()


def convert_scale_b_pair_groups(scale):
    # Raw micro-op tgemv_mx consumes right MX scale as linear bytes, but the
    # effective logical order groups K/32 rows in pairs before flattening.
    return scale.reshape((scale.shape[0] // 2, 2, scale.shape[1])).transpose(0, 2, 1).reshape(scale.shape[0], scale.shape[1]).copy()


def convert_scale_b_gemv_micro(scale, block_size=16, c0_size_mx=2):
    # Raw micro-op mte_l1_l0b_mx for the 1x128x62 GEMV case does not consume
    # the right MX scale in the same flattened order as tile-op style packing.
    # The closest simulator-observed contract is:
    #   1. pair adjacent K/32 scale groups,
    #   2. within each N16 block, emit the second group of each pair first,
    #   3. group two adjacent N16 blocks into one N32 super-block before
    #      advancing to the next pair of N16 blocks.
    k, n = scale.shape
    pad_n = (block_size - n % block_size) % block_size
    pad_k = (c0_size_mx - k % c0_size_mx) % c0_size_mx
    if pad_n > 0 or pad_k > 0:
        padded = np.pad(scale, ((0, pad_k), (0, pad_n)), mode='constant', constant_values=0)
    else:
        padded = scale
    k_padded, n_padded = padded.shape
    k_pairs = k_padded // c0_size_mx
    n_blocks = n_padded // block_size
    paired = padded.reshape(k_pairs, c0_size_mx, n_blocks, block_size)
    lane_major = paired.transpose(1, 2, 0, 3)[::-1]
    if n_blocks % 2 != 0:
        return lane_major.reshape(k_padded, n_padded).copy()
    return lane_major.reshape(c0_size_mx, n_blocks // 2, 2, k_pairs, block_size) \
        .transpose(1, 0, 2, 3, 4).reshape(k_padded, n_padded).copy()


def convert_scale_b_nn(scale, block_size=16, c0_size_mx=2):
    k, n = scale.shape
    pad_n = (block_size - n % block_size) % block_size
    pad_k = (c0_size_mx - k % c0_size_mx) % c0_size_mx
    if pad_n > 0 or pad_k > 0:
        padded = np.pad(scale, ((0, pad_k), (0, pad_n)), mode='constant', constant_values=0)
    else:
        padded = scale
    k_padded, n_padded = padded.shape
    return padded.reshape(1, n_padded // 16, k_padded // 2, 16, 2).copy()


def gen_golden(case):
    atype = case["atype"]
    btype = case["btype"]
    m, k, n = case["m"], case["k"], case["n"]
    m_padded = case["m_padded"]
    n_storage = case["n_storage"]
    n_padded = case["n_padded"]
    is_bias = case["is_bias"]
    is_fp4 = case["is_fp4"]

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
        x2_padded = zeros_fp4(btype, [k_aligned, n_storage])
        x2_padded[:k, :n] = x2
        x1_bin = mx_pack_two_fp4(x1_padded, atype)
        x2_bin = mx_pack_two_fp4(x2_padded, btype)
    else:
        x1_padded = np.zeros([m_padded, k_aligned], dtype=atype)
        x1_padded[:m, :k] = x1
        x2_padded = np.zeros([k_aligned, n_storage], dtype=btype)
        x2_padded[:k, :n] = x2
        x1_bin = x1_padded
        x2_bin = x2_padded

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

    if case["name"] in ("gemv_mx_fp8_e4m3_e5m2_1x256x20", "gemv_mx_bias_fp4_e1m2_1x2048x64"):
        x1_scale_gm = convert_scale_a_row_major_padded(x1_scale, 16, 2)
    else:
        x1_scale_gm = convert_scale_a_format(x1_scale, 16, 2)
    if case["name"] == "gemv_mx_fp4_e1m2_1x128x62":
        x2_scale_gm = convert_scale_b_nd_padded(x2_scale)
    else:
        x2_scale_gm = convert_scale_b_format(x2_scale, 16, 2)

    if is_bias:
        bias = np.random.randint(1, 10, [n]).astype(np.float32)
        golden_valid = np.matmul(x1_float, x2_float).astype(np.float32) + bias
        golden = np.zeros([m_padded, n_padded], dtype=np.float32)
        golden[:m, :n] = golden_valid
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
