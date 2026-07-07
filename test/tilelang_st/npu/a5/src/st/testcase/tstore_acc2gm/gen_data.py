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
    """Convert float32 array to rounded bfloat16 stored as uint16."""
    f32_view = arr.view(np.uint32)
    rounding_bias = np.uint32(0x7FFF) + ((f32_view >> 16) & np.uint32(1))
    bf16_uint16 = ((f32_view + rounding_bias) >> 16).astype(np.uint16)
    return bf16_uint16


def uint16_to_bf16_as_f32(arr):
    """Convert bfloat16 (stored as uint16) back to float32 for computation."""
    uint32_view = arr.astype(np.uint32) << 16
    return uint32_view.view(np.float32)


for case in CASES:
    setup_case_rng(case)

    M = case["M"]
    N = case["N"]
    K = case["K"]
    quant_mode = case["quant_mode"]
    name = case["name"]

    src_dtype_raw = case.get("src_dtype_raw", None)
    dst_dtype_raw = case.get("dst_dtype_raw", None)
    dst_layout = case.get("dst_layout", "nz2nd")  # default NZ2ND (row-major)

    # Generate input matrices as float32 for computation
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
        x1_compute = x1_bin.astype(np.float32)
        x2_compute = x2_bin.astype(np.float32)

    # Hardware accumulates the rounded source dtype values.
    golden_f32 = np.matmul(x1_compute, x2_compute).astype(np.float32)

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
        # Use non-trivial quantization values to actually test scaling functionality.
        # Values near 1.0 keep results within dtype range while being distinct from 1.0.
        quant_vector_f32 = np.random.uniform(0.5, 1.5, size=(1, N)).astype(np.float32)
        quant_vector_gm = quant_vector_f32.view(np.uint32).astype(np.uint64)
        quant_bits = np.bitwise_and(quant_vector_f32.view(np.uint32).copy(), 0xFFFFE000)
        quant_compute = quant_bits.view(np.float32)
        data_dict["quant_vector"] = quant_vector_gm
        golden_quant = golden_f32 * quant_compute
        if dst_dtype_raw == "bf16":
            golden_bin = bf16_to_uint16(golden_quant)
        elif case["dst_dtype"] == np.float16:
            golden_bin = golden_quant.astype(np.float16)
        data_dict["golden"] = golden_bin

    save_case_data(name, data_dict)
    print(f"[INFO] gen_data: {name} M={M} N={N} K={K} quant_mode={quant_mode}")
