#!/usr/bin/python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

"""Single source of truth for tcmps ST test cases."""

import numpy as np

CASES = [
    # float32 cases matching testcase/tcmps
    {
        "name": "f32_1x64",
        "dtype": np.float32,
        "out_dtype": np.uint8,
        "shape": (1, 64),
        "valid_shape": (1, 64),
        "eps": 0,
    },
    {
        "name": "f32_4x64",
        "dtype": np.float32,
        "out_dtype": np.uint8,
        "shape": (4, 64),
        "valid_shape": (4, 64),
        "eps": 0,
    },
    {
        "name": "f32_8x64",
        "dtype": np.float32,
        "out_dtype": np.uint8,
        "shape": (8, 64),
        "valid_shape": (8, 64),
        "eps": 0,
    },
    {
        "name": "f32_32x64",
        "dtype": np.float32,
        "out_dtype": np.uint8,
        "shape": (32, 64),
        "valid_shape": (32, 64),
        "eps": 0,
    },
    {
        "name": "f32_128x128",
        "dtype": np.float32,
        "out_dtype": np.uint8,
        "shape": (128, 128),
        "valid_shape": (128, 128),
        "eps": 0,
    },
    # int32 cases matching testcase/tcmps
    {
        "name": "i32_16x32",
        "dtype": np.int32,
        "out_dtype": np.uint8,
        "shape": (16, 32),
        "valid_shape": (16, 32),
        "eps": 0,
    },
    {
        "name": "i32_32x32",
        "dtype": np.int32,
        "out_dtype": np.uint8,
        "shape": (32, 32),
        "valid_shape": (32, 32),
        "eps": 0,
    },
    {
        "name": "i32_32x64_valid32x64",
        "dtype": np.int32,
        "out_dtype": np.uint8,
        "shape": (64, 64),
        "valid_shape": (32, 64),
        "eps": 0,
    },
    # Non-aligned cases
    {
        "name": "f32_7x448",
        "dtype": np.float32,
        "out_dtype": np.uint8,
        "shape": (7, 448),
        "valid_shape": (7, 448),
        "eps": 0,
    },
    {
        "name": "f32_256x16",
        "dtype": np.float32,
        "out_dtype": np.uint8,
        "shape": (256, 16),
        "valid_shape": (256, 16),
        "dst_shape": (256, 32),
        "dst_valid_shape": (256, 32),
        "eps": 0,
    },
    {
        "name": "i32_31x128",
        "dtype": np.int32,
        "out_dtype": np.uint8,
        "shape": (31, 128),
        "valid_shape": (31, 128),
        "eps": 0,
    },
    # 16B cases (f16, i16)
    {
        "name": "f16_32x128",
        "dtype": np.float16,
        "out_dtype": np.uint8,
        "shape": (32, 128),
        "valid_shape": (32, 128),
        "eps": 0,
    },
    {
        "name": "i16_32x128",
        "dtype": np.int16,
        "out_dtype": np.uint8,
        "shape": (32, 128),
        "valid_shape": (32, 128),
        "eps": 0,
    },
]
