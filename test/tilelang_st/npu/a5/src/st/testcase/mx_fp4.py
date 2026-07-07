#!/usr/bin/python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Local FP4 helpers for TileLang MX ST data generation."""

import numpy as np
import ml_dtypes


FP4_E1M2 = "f4e1m2x2"
FP4_E2M1 = ml_dtypes.float4_e2m1fn


def is_fp4_e1m2(dtype):
    return dtype == FP4_E1M2


def is_fp4_e2m1(dtype):
    return dtype == FP4_E2M1


def make_fp4_random(dtype, shape):
    if is_fp4_e2m1(dtype):
        return np.random.randint(-6, 6, shape).astype(FP4_E2M1)
    if is_fp4_e1m2(dtype):
        return np.random.randint(-1, 2, shape).astype(np.float32)
    raise TypeError(f"unsupported fp4 dtype: {dtype!r}")


def zeros_fp4(dtype, shape):
    if is_fp4_e2m1(dtype):
        return np.zeros(shape, dtype=FP4_E2M1)
    if is_fp4_e1m2(dtype):
        return np.zeros(shape, dtype=np.float32)
    raise TypeError(f"unsupported fp4 dtype: {dtype!r}")


def _e1m2_nibbles(values):
    values = np.asarray(values, dtype=np.float32)
    nibbles = np.zeros(values.shape, dtype=np.uint8)
    nibbles[values > 0] = 0x4
    nibbles[values < 0] = 0xC
    return nibbles


def fp4_nibbles(values, dtype):
    if is_fp4_e2m1(dtype):
        return np.asarray(values).reshape(-1).view(np.uint8).reshape(np.asarray(values).shape) & 0x0F
    if is_fp4_e1m2(dtype):
        return _e1m2_nibbles(values)
    raise TypeError(f"unsupported fp4 dtype: {dtype!r}")


def pack_two_fp4(matrix, dtype):
    row, col = matrix.shape
    flat = fp4_nibbles(matrix, dtype).reshape(-1)
    high = flat[::2] & 0x0F
    low = (flat[1::2] & 0x0F) << 4
    return (low | high).reshape(row, col // 2)
