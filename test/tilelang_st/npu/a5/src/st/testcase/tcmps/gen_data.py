#!/usr/bin/python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

import numpy as np
from cases import CASES
from st_common import validate_cases, setup_case_rng, save_case_data

# Scalar value for comparison (matches the scalar passed in launch.cpp)
SCALAR = 5.0

validate_cases(CASES)

for case in CASES:
    setup_case_rng(case)

    dtype = case["dtype"]
    out_dtype = case["out_dtype"]
    shape = case["shape"]
    valid_shape = case["valid_shape"]

    # Generate random input matching testcase/tcmps pattern
    if np.issubdtype(dtype, np.floating):
        input1 = np.random.randint(-5, 5, size=shape).astype(dtype)
    else:
        input1 = np.random.randint(1, 10, size=shape).astype(dtype)

    vr, vc = valid_shape
    if np.issubdtype(dtype, np.floating):
        scalar_val = dtype(SCALAR)
    else:
        scalar_val = dtype(int(SCALAR))

    # Compute element-wise comparison result (0 or 1 per element)
    # Using "lt" mode to match the template
    cmp_result = (input1[:vr, :vc] < scalar_val).astype(np.uint8, copy=False)

    # The 2D tcmps templates store each row at the destination tile's physical
    # row stride, with one predicate bit per compared element. Pack the valid
    # bits of each row into ceil(valid_cols / 8) bytes; the compare side gathers
    # those bytes from the physical destination rows.
    row_bytes = (vc + 7) // 8
    golden = np.zeros(vr * row_bytes, dtype=np.uint8)
    for row in range(vr):
        for col in range(vc):
            if cmp_result[row, col]:
                golden[row * row_bytes + col // 8] |= 1 << (col % 8)

    save_case_data(case["name"], {"input1": input1, "golden": golden})
    print(f"[INFO] gen_data: {case['name']} shape={shape} valid_shape={valid_shape} dtype={dtype.__name__} out_dtype={out_dtype.__name__} scalar={SCALAR}")
