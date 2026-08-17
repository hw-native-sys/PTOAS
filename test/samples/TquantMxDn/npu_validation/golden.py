#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Standalone board golden for f32 MXFP8 DN quantization and DN-to-ZZ."""

import numpy as np

M, N, GROUP = 64, 64, 32


def fp8_bytes(values):
    from ml_dtypes import float8_e4m3fn
    return np.clip(values, -448, 448).astype(float8_e4m3fn).view(np.uint8)


def exp_scaling(max_values):
    bits = np.asarray(max_values, dtype=np.float32).view(np.uint32)
    exponent = ((bits & np.uint32(0x7F800000)) >> np.uint32(23)).astype(np.int32)
    exp = np.clip(exponent - 8, 0, 254).astype(np.uint8)
    scale_exp = np.clip(254 - exp.astype(np.int32), 0, 255).astype(np.uint32)
    scaling = (scale_exp << 23).view(np.float32)
    low = exponent <= 8
    exp[low] = 0
    scaling[low] = np.float32(2.0 ** -127)
    return exp, scaling


def dn_to_zz(exp):
    return exp.reshape(exp.shape[0] // 2, 2, exp.shape[1] // 16, 16).transpose(2, 0, 3, 1).reshape(-1)


def main():
    np.random.seed(41)
    src = np.random.uniform(-10, 10, size=(M, N)).astype(np.float32)
    group_max = np.max(np.abs(src.reshape(M // GROUP, GROUP, N)), axis=1)
    exp, scaling = exp_scaling(group_max)
    scaled = src * np.repeat(scaling, GROUP, axis=0)
    src.tofile("input.bin")
    fp8_bytes(scaled).tofile("dst.bin")
    exp.tofile("exp.bin")
    group_max.tofile("max.bin")
    scaling.tofile("scaling.bin")
    dn_to_zz(exp).tofile("exp_zz.bin")


if __name__ == "__main__":
    main()
