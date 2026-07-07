#!/usr/bin/python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

"""Single source of truth for tload_mat ST test cases.

End-to-end cube pipeline: TLOAD.MAT (GM -> MAT) + TMATMUL + TSTORE.ACC (ACC -> GM).
Tests TLOAD.MAT with different dtype and layout conversions.
The golden result is the identity matmul (since load+matmul+store should preserve data).

Ref: pto-isa tload_mix test cases covering ND2NZ and DN2NZ with multiple dtypes.
"""

import numpy as np

CASES = [
    # ND2NZ layout (row-major GM source -> NZ MAT dest)
    {
        "name": "f16_nd2nz",
        "dtype": np.float16,
        "layout": "nd2nz",
        "M": 16,
        "N": 32,
        "K": 16,
        "eps": 1e-3,
    },
    {
        "name": "bf16_nd2nz",
        "dtype": None,  # bf16 stored as uint16
        "dtype_raw": "bf16",
        "layout": "nd2nz",
        "M": 16,
        "N": 32,
        "K": 16,
        "eps": 1e-3,
    },
    {
        "name": "f32_nd2nz",
        "dtype": np.float32,
        "layout": "nd2nz",
        "M": 16,
        "N": 32,
        "K": 16,
        "eps": 1e-3,
    },
    # DN2NZ layout (col-major GM source -> NZ MAT dest)
    {
        "name": "f16_dn2nz",
        "dtype": np.float16,
        "layout": "dn2nz",
        "M": 16,
        "N": 32,
        "K": 16,
        "eps": 1e-3,
    },
    {
        "name": "bf16_dn2nz",
        "dtype": None,
        "dtype_raw": "bf16",
        "layout": "dn2nz",
        "M": 16,
        "N": 32,
        "K": 16,
        "eps": 1e-3,
    },
    {
        "name": "f32_dn2nz",
        "dtype": np.float32,
        "layout": "dn2nz",
        "M": 16,
        "N": 32,
        "K": 16,
        "eps": 1e-3,
    },
]
