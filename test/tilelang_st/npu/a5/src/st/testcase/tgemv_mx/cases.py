# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

"""Single source of truth for tgemv_mx ST test cases."""

import numpy as np
import ml_dtypes
from st_common import FP4_E1M2, FP4_E2M1

fp8_e4m3fn = ml_dtypes.float8_e4m3fn
fp8_e5m2 = ml_dtypes.float8_e5m2
fp4_e1m2x2 = FP4_E1M2
fp4_e2m1x2 = FP4_E2M1

CASES = [
    {"name": "gemv_mx_fp4_e1m2_1x128x62", "atype": fp4_e1m2x2, "btype": fp4_e1m2x2, "m": 1, "k": 128, "n": 62, "m_padded": 16, "n_storage": 64, "n_padded": 64, "is_bias": False, "is_fp4": True, "is_split_k": False, "eps": 1e-3},
    {"name": "gemv_mx_fp8_e4m3_e5m2_1x256x20", "atype": fp8_e4m3fn, "btype": fp8_e5m2, "m": 1, "k": 256, "n": 20, "m_padded": 16, "n_storage": 32, "n_padded": 32, "is_bias": False, "is_fp4": False, "is_split_k": False, "eps": 1e-3},
    {"name": "gemv_mx_bias_fp4_e1m2_1x64x62", "atype": fp4_e1m2x2, "btype": fp4_e1m2x2, "m": 1, "k": 64, "n": 62, "m_padded": 16, "n_storage": 64, "n_padded": 64, "is_bias": True, "is_fp4": True, "is_split_k": False, "eps": 1e-3},
    {"name": "gemv_mx_bias_fp4_e1m2_1x2048x64", "atype": fp4_e1m2x2, "btype": fp4_e1m2x2, "m": 1, "k": 2048, "n": 64, "m_padded": 16, "n_storage": 64, "n_padded": 64, "is_bias": True, "is_fp4": True, "is_split_k": True, "eps": 1e-3},
]
