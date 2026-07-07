#!/usr/bin/python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

"""Generate input and golden data for tload_mat ST cases.

Pipeline: TLOAD.MAT -> TMATMUL -> TSTORE.ACC
Golden = matmul(x1, x2) cast to f32 (ACC output dtype).
The test verifies that TLOAD.MAT correctly loads data from GM to L1 MAT buffer
through the matmul pipeline.
"""

import numpy as np
from cases import CASES
from st_common import setup_case_rng, save_case_data


def bf16_to_uint16(arr):
    """Convert float32 array to bfloat16 stored as uint16."""
    f32_view = arr.view(np.uint32)
    return (f32_view >> 16).astype(np.uint16)


def bf16_to_float32(arr):
    """Round float32 values to bfloat16 precision and return float32 values."""
    u16 = bf16_to_uint16(arr)
    return (u16.astype(np.uint32) << 16).view(np.float32)


for case in CASES:
    setup_case_rng(case)

    M = case["M"]
    N = case["N"]
    K = case["K"]
    name = case["name"]
    layout = case["layout"]

    # Generate input matrices as float32 for computation
    x1_f32 = np.random.uniform(-1, 1, size=(M, K)).astype(np.float32)
    x2_f32 = np.random.uniform(-1, 1, size=(K, N)).astype(np.float32)

    dtype_raw = case.get("dtype_raw", None)
    if dtype_raw == "bf16":
        x1_compute = bf16_to_float32(x1_f32)
        x2_compute = bf16_to_float32(x2_f32)
    elif case["dtype"] == np.float16:
        x1_compute = x1_f32.astype(np.float16).astype(np.float32)
        x2_compute = x2_f32.astype(np.float16).astype(np.float32)
    else:
        x1_compute = x1_f32
        x2_compute = x2_f32

    # Golden = matmul result in f32 (ACC output is always f32 for float matmul)
    golden_f32 = np.matmul(x1_compute, x2_compute)

    # For DN2NZ cases, x1 is stored in DN (col-major/transposed) format
    # in GM. x2 stays ND so the case isolates the DN load path under matmul.
    if layout == "dn2nz" and case["dtype"] != np.float32:
        x1_gm_f32 = x1_f32.T  # [K, M] transposed for DN format
        x2_gm_f32 = x2_f32
    else:
        x1_gm_f32 = x1_f32
        x2_gm_f32 = x2_f32

    # Prepare input data in source dtype (using DN-layout data if needed)
    if dtype_raw == "bf16":
        x1_bin = bf16_to_uint16(x1_gm_f32)
        x2_bin = bf16_to_uint16(x2_gm_f32)
    elif case["dtype"] == np.float16:
        x1_bin = x1_gm_f32.astype(np.float16)
        x2_bin = x2_gm_f32.astype(np.float16)
    elif case["dtype"] == np.float32:
        x1_bin = x1_gm_f32
        x2_bin = x2_gm_f32
    else:
        x1_bin = x1_gm_f32
        x2_bin = x2_gm_f32

    # Golden is always f32 (ACC output)
    golden_bin = golden_f32

    save_case_data(name, {"x1_gm": x1_bin, "x2_gm": x2_bin, "golden": golden_bin})
    print(f"[INFO] gen_data: {name} M={M} N={N} K={K} layout={layout}")
