# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Require finite, exactly matching FP32 values in every output tile."""

import sys

import numpy as np


def main():
    golden = np.fromfile("golden_v3.bin", dtype=np.float32)
    output = np.fromfile("v3.bin", dtype=np.float32)
    if golden.size != 128 * 32 * 32 or output.shape != golden.shape:
        raise ValueError("UF loop output must contain 128 complete 32x32 tiles")
    if not np.isfinite(output).all() or not np.allclose(output, golden, rtol=0, atol=0):
        raise ValueError("UF loop FP32 output differs from the exact reference")
    print("[INFO] compare passed: 128 tiles, rtol=0, atol=0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
