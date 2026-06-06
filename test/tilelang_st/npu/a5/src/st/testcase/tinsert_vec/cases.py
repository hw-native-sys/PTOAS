# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

"""Test cases for pto.tinsert Vec->Vec ND vkernel ST."""

import numpy as np


CASES = [
    {
        "name": "vec2vec_nd_f16_16x16_into_32x32_idx00",
        "kernel": "TINSERT_vec2vec_nd_f16_16x16_into_32x32_idx00",
        "dtype": np.float16,
        "src_shape": (16, 16),
        "dst_shape": (32, 32),
        "index_row": 0,
        "index_col": 0,
        "has_output": True,
        "eps": 1e-2,
    },
    {
        "name": "vec2vec_nd_f16_16x16_into_32x32_idx816",
        "kernel": "TINSERT_vec2vec_nd_f16_16x16_into_32x32_idx816",
        "dtype": np.float16,
        "src_shape": (16, 16),
        "dst_shape": (32, 32),
        "index_row": 8,
        "index_col": 16,
        "has_output": True,
        "eps": 1e-2,
    },
    {
        "name": "vec2vec_nd_f32_16x16_into_32x32_idx00",
        "kernel": "TINSERT_vec2vec_nd_f32_16x16_into_32x32_idx00",
        "dtype": np.float32,
        "src_shape": (16, 16),
        "dst_shape": (32, 32),
        "index_row": 0,
        "index_col": 0,
        "has_output": True,
        "eps": 1e-6,
    },
]