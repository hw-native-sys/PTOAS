#!/usr/bin/python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

"""Single source of truth for tstore_acc2gm ST test cases.

End-to-end cube pipeline: TLOAD.MAT -> TMATMUL -> TSTORE.ACC / TSTORE_FP.

Each case defines:
  - name:        case identifier, used as subdirectory name and by main.cpp kCases[].
  - src_dtype:   source (input) numpy dtype for TLOAD.MAT.
  - acc_dtype:   accumulator numpy dtype (f32 for float matmul, i32 for integer).
  - dst_dtype:   destination numpy dtype for TSTORE.ACC / TSTORE_FP output.
  - M, N, K:     matmul dimensions.
  - quant_mode:  0 = no-quant (TSTORE.ACC), 2 = vector quant (TSTORE_FP).
  - scaling_dtype: numpy dtype for scaling buffer (None for quant_mode=0).
  - eps:         tolerance for numpy.allclose (atol and rtol).

Ref: pto-isa tstore_acc2gm test cases covering no-quant and vector quant modes.
"""

import numpy as np

CASES = [
    # No-quant TSTORE.ACC cases (NZ2ND layout)
    {
        "name": "f16_f32_f32_nz2nd",
        "src_dtype": np.float16,
        "acc_dtype": np.float32,
        "dst_dtype": np.float32,
        "M": 16,
        "N": 32,
        "K": 16,
        "quant_mode": 0,
        "scaling_dtype": None,
        "eps": 1e-3,
    },
    {
        "name": "f16_f32_f16_nz2nd",
        "src_dtype": np.float16,
        "acc_dtype": np.float32,
        "dst_dtype": np.float16,
        "M": 16,
        "N": 32,
        "K": 16,
        "quant_mode": 0,
        "scaling_dtype": None,
        "eps": 1e-3,
    },
    {
        "name": "bf16_f32_f32_nz2nd",
        "src_dtype": None,  # bfloat16 not directly available in numpy; use uint16 as storage
        "acc_dtype": np.float32,
        "dst_dtype": np.float32,
        "M": 16,
        "N": 32,
        "K": 16,
        "quant_mode": 0,
        "scaling_dtype": None,
        "eps": 1e-3,
        "src_dtype_raw": "bf16",
        "dst_dtype_raw": "f32",
    },
    {
        "name": "bf16_f32_bf16_nz2nd",
        "src_dtype": None,
        "acc_dtype": np.float32,
        "dst_dtype": None,
        "M": 16,
        "N": 32,
        "K": 16,
        "quant_mode": 0,
        "scaling_dtype": None,
        "eps": 2e-2,
        "src_dtype_raw": "bf16",
        "dst_dtype_raw": "bf16",
    },
    {
        "name": "i8_i32_i32_nz2nd",
        "src_dtype": np.int8,
        "acc_dtype": np.int32,
        "dst_dtype": np.int32,
        "M": 16,
        "N": 32,
        "K": 16,
        "quant_mode": 0,
        "scaling_dtype": None,
        "eps": 0,
    },
    # No-quant TSTORE.ACC cases (NZ2DN layout — col-major GM dest)
    {
        "name": "f16_f32_f32_nz2dn",
        "src_dtype": np.float16,
        "acc_dtype": np.float32,
        "dst_dtype": np.float32,
        "M": 16,
        "N": 32,
        "K": 16,
        "quant_mode": 0,
        "scaling_dtype": None,
        "eps": 1e-3,
        "dst_layout": "nz2dn",
    },
    {
        "name": "bf16_f32_bf16_nz2dn",
        "src_dtype": None,
        "acc_dtype": np.float32,
        "dst_dtype": None,
        "M": 16,
        "N": 32,
        "K": 16,
        "quant_mode": 0,
        "scaling_dtype": None,
        "eps": 2e-2,
        "src_dtype_raw": "bf16",
        "dst_dtype_raw": "bf16",
        "dst_layout": "nz2dn",
    },
    # No-quant TSTORE.ACC cases (NZ2NZ layout — fractal GM dest)
    {
        "name": "f16_f32_f32_nz2nz",
        "src_dtype": np.float16,
        "acc_dtype": np.float32,
        "dst_dtype": np.float32,
        "M": 16,
        "N": 32,
        "K": 16,
        "quant_mode": 0,
        "scaling_dtype": None,
        "eps": 1e-3,
        "dst_layout": "nz2nz",
    },
    {
        "name": "bf16_f32_bf16_nz2nz",
        "src_dtype": None,
        "acc_dtype": np.float32,
        "dst_dtype": None,
        "M": 16,
        "N": 32,
        "K": 16,
        "quant_mode": 0,
        "scaling_dtype": None,
        "eps": 2e-2,
        "src_dtype_raw": "bf16",
        "dst_dtype_raw": "bf16",
        "dst_layout": "nz2nz",
    },
    # Vector quant TSTORE_FP cases (NZ2ND layout)
    {
        "name": "f16_f32_f16_vec",
        "src_dtype": np.float16,
        "acc_dtype": np.float32,
        "dst_dtype": np.float16,
        "M": 16,
        "N": 64,
        "K": 16,
        "quant_mode": 2,
        "scaling_dtype": np.uint32,
        "eps": 1e-3,
    },
    {
        "name": "bf16_f32_bf16_vec",
        "src_dtype": None,
        "acc_dtype": np.float32,
        "dst_dtype": None,
        "M": 16,
        "N": 64,
        "K": 16,
        "quant_mode": 2,
        "scaling_dtype": np.uint32,
        "eps": 2e-2,
        "src_dtype_raw": "bf16",
        "dst_dtype_raw": "bf16",
        "scaling_dtype_raw": "f32",
    },
]
