#!/usr/bin/python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

"""Generate input and golden data for tstore_acc2gm ST cases.

For each case:
  - x1_gm[M, K]: left input matrix (loaded via TLOAD.MAT)
  - x2_gm[K, N]: right input matrix (loaded via TLOAD.MAT)
  - golden[M, N]: expected result = matmul(x1, x2) cast to dst_dtype
  - quant_vector[N]: per-column quantization vector (only for quant_mode=2)

For NZ2DN/NZ2NZ layout cases, the golden is reshaped to match the
destination layout format:
  - NZ2DN: col-major (DN) layout — golden is transposed [N, M]
  - NZ2NZ: fractal (NZ) layout — golden is reshaped into NZ blocks

Ref: pto-isa tstore_acc2gm gen_data.py patterns.
"""

import numpy as np
from cases import CASES
from st_common import setup_case_rng, save_case_data


def bf16_to_uint16(arr):
    """Convert float32 array to bfloat16 stored as uint16."""
    # bfloat16 is the upper 16 bits of float32
    f32_view = arr.view(np.uint32)
    bf16_uint16 = (f32_view >> 16).astype(np.uint16)
    return bf16_uint16


def uint16_to_bf16_as_f32(arr):
    """Convert bfloat16 (stored as uint16) back to float32 for computation."""
    uint32_view = arr.astype(np.uint32) << 16
    return uint32_view.view(np.float32)


def make_vector_quant_params(scales):
    """Encode qf322* pre_vec parameters as uint32 transport words.

    The hardware format stores one uint64 parameter per column. The lower
    32 bits contain the IEEE f32 scale bits; the upper 32 bits are zero.
    Host transport uses uint32[2 * N], matching the pto-isa tileop cases.
    """
    encoded = scales.reshape(-1).astype(np.float32).view(np.uint32).astype(np.uint64)
    return encoded.view(np.uint32)


def extract_vector_quant_scales(quant):
    """Decode qf322* pre_vec M1 scales from uint64 encoded parameters."""
    values = quant.reshape(-1).astype(np.uint64)
    decoded = np.empty(values.shape, dtype=np.float32)
    for idx, value in enumerate(values):
        raw = int(value)
        m1_bits = (raw >> 13) & 0x7FFFF
        sign_bit = (m1_bits >> 18) & 0x1
        exponent = (m1_bits >> 10) & 0xFF
        mantissa = m1_bits & 0x3FF
        decoded[idx] = ((-1) ** sign_bit) * (1 + mantissa / 1024.0) * (2 ** (exponent - 127))
    return decoded.reshape(1, -1)


for case in CASES:
    setup_case_rng(case)

    M = case["M"]
    N = case["N"]
    K = case["K"]
    quant_mode = case["quant_mode"]
    name = case["name"]

    src_dtype_raw = case.get("src_dtype_raw", None)
    dst_dtype_raw = case.get("dst_dtype_raw", None)
    scaling_dtype_raw = case.get("scaling_dtype_raw", None)

    dst_layout = case.get("dst_layout", "nz2nd")  # default NZ2ND (row-major)

    # Generate input matrices as float32, then compute golden from the exact
    # precision that is written to GM and consumed by the kernel.
    x1_f32 = np.random.uniform(-1, 1, size=(M, K)).astype(np.float32)
    x2_f32 = np.random.uniform(-1, 1, size=(K, N)).astype(np.float32)

    # Prepare input data in source dtype
    if src_dtype_raw == "bf16":
        x1_bin = bf16_to_uint16(x1_f32)
        x2_bin = bf16_to_uint16(x2_f32)
        x1_compute = uint16_to_bf16_as_f32(x1_bin)
        x2_compute = uint16_to_bf16_as_f32(x2_bin)
    elif case["src_dtype"] == np.float16:
        x1_bin = x1_f32.astype(np.float16)
        x2_bin = x2_f32.astype(np.float16)
        x1_compute = x1_bin.astype(np.float32)
        x2_compute = x2_bin.astype(np.float32)
    elif case["src_dtype"] == np.int8:
        x1_bin = np.random.randint(-5, 5, size=(M, K)).astype(np.int8)
        x2_bin = np.random.randint(-5, 5, size=(K, N)).astype(np.int8)
        x1_compute = x1_bin.astype(np.float32)
        x2_compute = x2_bin.astype(np.float32)
    else:
        x1_bin = x1_f32
        x2_bin = x2_f32
        x1_compute = x1_f32
        x2_compute = x2_f32

    golden_f32 = np.matmul(x1_compute, x2_compute)

    # Prepare golden in destination dtype
    if dst_dtype_raw == "bf16":
        golden_bin = bf16_to_uint16(golden_f32)
    elif case["dst_dtype"] == np.float32:
        golden_bin = golden_f32.astype(np.float32)
    elif case["dst_dtype"] == np.float16:
        golden_bin = golden_f32.astype(np.float16)
    elif case["dst_dtype"] == np.int32:
        golden_bin = golden_f32.astype(np.int32)
    else:
        golden_bin = golden_f32

    # For NZ2DN layout: golden is stored in col-major (DN) format.
    # The device writes [N, M] col-major to GM; golden should match that layout.
    # For comparison, we store golden as [M, N] row-major regardless,
    # and compare.py reshapes/compares accordingly.
    # For NZ2NZ layout: golden is stored in fractal NZ format in GM.
    # The golden is reshaped into NZ blocks for comparison.

    data_dict = {"x1_gm": x1_bin, "x2_gm": x2_bin, "golden": golden_bin}

    # For vector quant (TSTORE_FP), generate per-column quantization vector
    if quant_mode == 2:
        # Use non-trivial vector scaling values while keeping results within
        # f16/bf16 range.
        fp_quant_elems = 64
        quant_vector_f32 = (
            (np.arange(fp_quant_elems, dtype=np.float32) % np.float32(4.0)) + np.float32(1.0)
        ).reshape(1, fp_quant_elems)
        quant_matrix_f32 = np.repeat(quant_vector_f32, M, axis=0)
        if scaling_dtype_raw == "bf16":
            quant_bin = bf16_to_uint16(quant_vector_f32)
            quant_compute = uint16_to_bf16_as_f32(quant_bin)
        elif case.get("scaling_dtype") == np.float16:
            quant_bin = quant_matrix_f32.astype(np.float16)
            quant_compute = quant_bin.astype(np.float32)
        elif case.get("scaling_dtype") == np.uint32:
            quant_bin = make_vector_quant_params(quant_vector_f32)
            quant_compute = extract_vector_quant_scales(quant_bin.view(np.uint64))[:, :N]
        else:
            quant_bin = bf16_to_uint16(quant_vector_f32)
            quant_compute = uint16_to_bf16_as_f32(quant_bin)
        data_dict["quant_vector"] = quant_bin
        # Golden with quantization: result * quant_vector
        golden_quant = golden_f32 * quant_compute
        if dst_dtype_raw == "bf16":
            golden_bin = bf16_to_uint16(golden_quant)
        elif case["dst_dtype"] == np.float16:
            golden_bin = golden_quant.astype(np.float16)
        data_dict["golden"] = golden_bin

    save_case_data(name, data_dict)
    print(f"[INFO] gen_data: {name} M={M} N={N} K={K} quant_mode={quant_mode}")
