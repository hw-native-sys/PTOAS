#!/usr/bin/python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import os
import sys
import numpy as np


CASES = [
    ("fp8_load_16x64", (16, 64)),
    ("fp8_load_32x64", (32, 64)),
    ("fp8_load_128x64", (128, 64)),
]


def compare_case(name, shape):
    golden = np.fromfile(os.path.join(name, "golden.bin"), dtype=np.uint8).reshape(shape)
    output = np.fromfile(os.path.join(name, "output.bin"), dtype=np.uint8).reshape(shape)
    if np.array_equal(golden, output):
        print(f"[INFO] {name}: compare passed")
        return True

    mismatch = golden != output
    count = int(np.count_nonzero(mismatch))
    print(f"[ERROR] {name}: {count}/{golden.size} byte mismatches")
    for row, col in np.argwhere(mismatch)[:5]:
        print(f"  [{row},{col}]: expected=0x{golden[row, col]:02x}, actual=0x{output[row, col]:02x}")
    return False


def main():
    all_passed = True
    for name, shape in CASES:
        if not compare_case(name, shape):
            all_passed = False
    if not all_passed:
        sys.exit(2)
    print("[INFO] all cases passed")


if __name__ == "__main__":
    main()
