#!/usr/bin/python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Generate exactly representable signed inputs for issue #1745."""
import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--products", type=int, choices=(2, 80), default=2)
    args = parser.parse_args()
    rng = np.random.default_rng(1745)
    a = rng.integers(-32, 33, size=(80, 64)).astype(np.float32) / 32
    b = rng.integers(-32, 33, size=(64, 32)).astype(np.float32) / 32
    # Inputs are exactly representable in TF32; double precision matmul avoids
    # depending on the CPU BLAS accumulation implementation for the oracle.
    expected = (args.products * (a.astype(np.float64) @ b.astype(np.float64))).astype(np.float32)
    np.full((80, 32), np.nan, dtype=np.float32).tofile("v1.bin")
    a.tofile("v2.bin")
    b.tofile("v3.bin")
    expected.tofile("golden_v1.bin")


if __name__ == "__main__":
    main()
