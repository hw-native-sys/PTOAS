#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the license.

import sys

import numpy as np

N = 256


def check(name: str, actual: np.ndarray, expected: np.ndarray) -> bool:
    mismatches = np.flatnonzero(actual.view(np.uint32) != expected.view(np.uint32))
    print(f"{name} bit mismatches: {mismatches.size} / {expected.size}")
    if mismatches.size:
        idx = int(mismatches[0])
        print(
            f"  first idx={idx}: actual={actual[idx]!r} "
            f"(0x{actual.view(np.uint32)[idx]:08x}), expected={expected[idx]!r} "
            f"(0x{expected.view(np.uint32)[idx]:08x})"
        )
        print(f"  max abs error: {np.max(np.abs(actual - expected))!r}")
        return False
    return True


def main() -> None:
    actual = np.fromfile("out.bin", dtype=np.float32)
    expected = np.fromfile("golden_out.bin", dtype=np.float32)
    if actual.size != 3 * N or expected.size != 3 * N:
        print(f"[ERROR] expected {3 * N} values, got actual={actual.size}, golden={expected.size}")
        sys.exit(2)
    results = [
        check("separate mulf+addf", actual[:N], expected[:N]),
        check("explicit pto.fma", actual[N : 2 * N], expected[N : 2 * N]),
        check("scalar arith.divf", actual[2 * N :], expected[2 * N :]),
    ]
    if not all(results):
        sys.exit(2)
    print("[INFO] all FP32 precision checks passed")


if __name__ == "__main__":
    main()
