# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

"""Single source of truth for tmatmul_mx ST test cases."""

import numpy as np
import ml_dtypes
from mx_fp4 import FP4_E1M2, FP4_E2M1

fp8_e4m3fn = ml_dtypes.float8_e4m3fn
fp8_e5m2 = ml_dtypes.float8_e5m2
fp4_e1m2x2 = FP4_E1M2
fp4_e2m1x2 = FP4_E2M1

CASES = [
    {"name": "fp8_e5m2_128x64x64", "atype": fp8_e5m2, "btype": fp8_e5m2, "m": 128, "k": 64, "n": 64, "m_padded": 128, "n_padded": 64, "is_bias": False, "is_fp4": False, "eps": 1e-3},
    {"name": "fp8_e4m3_127x72x64", "atype": fp8_e4m3fn, "btype": fp8_e4m3fn, "m": 127, "k": 72, "n": 64, "m_padded": 128, "n_padded": 64, "is_bias": False, "is_fp4": False, "eps": 1e-3},
    {"name": "fp8_e4m3_e5m2_128x110x63", "atype": fp8_e4m3fn, "btype": fp8_e5m2, "m": 128, "k": 110, "n": 63, "m_padded": 128, "n_padded": 64, "is_bias": False, "is_fp4": False, "eps": 1e-3},
    {"name": "fp4_e2m1_128x64x64", "atype": fp4_e2m1x2, "btype": fp4_e2m1x2, "m": 128, "k": 64, "n": 64, "m_padded": 128, "n_padded": 64, "is_bias": False, "is_fp4": True, "eps": 1e-3},
    {"name": "fp4_e1m2_e2m1_117x64x60", "atype": fp4_e1m2x2, "btype": fp4_e2m1x2, "m": 117, "k": 64, "n": 60, "m_padded": 128, "n_padded": 64, "is_bias": False, "is_fp4": True, "eps": 1e-3},
    {"name": "fp4_e2m1_e1m2_128x118x64", "atype": fp4_e2m1x2, "btype": fp4_e1m2x2, "m": 128, "k": 118, "n": 64, "m_padded": 128, "n_padded": 64, "is_bias": False, "is_fp4": True, "eps": 1e-3},
    {"name": "fp4_e2m1_e1m2_115x64x30", "atype": fp4_e2m1x2, "btype": fp4_e1m2x2, "m": 115, "k": 64, "n": 30, "m_padded": 128, "n_padded": 64, "is_bias": False, "is_fp4": True, "eps": 1e-3},
    {"name": "fp8_e4m3_16x32x16", "atype": fp8_e4m3fn, "btype": fp8_e4m3fn, "m": 16, "k": 32, "n": 16, "m_padded": 16, "n_padded": 16, "is_bias": False, "is_fp4": False, "eps": 1e-3},
    {"name": "fp8_e4m3_e5m2_10x50x54", "atype": fp8_e4m3fn, "btype": fp8_e5m2, "m": 10, "k": 50, "n": 54, "m_padded": 16, "n_padded": 64, "is_bias": False, "is_fp4": False, "eps": 1e-3},
    {"name": "fp4_e2m1_4x30x8", "atype": fp4_e2m1x2, "btype": fp4_e2m1x2, "m": 4, "k": 30, "n": 8, "m_padded": 16, "n_padded": 64, "is_bias": False, "is_fp4": True, "eps": 1e-3},
]
