#!/usr/bin/python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Compare all output elements against an independent CPU oracle."""
import numpy as np


def main():
    expected = np.fromfile("golden_v1.bin", dtype=np.float32)
    actual = np.fromfile("v1.bin", dtype=np.float32)
    if expected.size != 2560 or actual.size != expected.size:
        raise ValueError("Expected exactly 80x32 output elements")
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-5)
    print(f"PASS: {actual.size} elements, max_abs_error={np.max(np.abs(actual - expected))}")


if __name__ == "__main__":
    main()
