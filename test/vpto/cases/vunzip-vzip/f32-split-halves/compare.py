#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Bit-exact split check for the f32 vunzip split-halves case.

v2.bin is the low half and v3.bin the high half, each compared as raw uint16
against the independent python extraction.
"""

import sys

import numpy as np

DTYPE = np.uint16
OUTPUTS = (2, 3)


def main() -> None:
    failed = False
    for index in OUTPUTS:
        golden = np.fromfile(f"golden_v{index}.bin", dtype=DTYPE)
        output = np.fromfile(f"v{index}.bin", dtype=DTYPE)
        if golden.shape == output.shape and np.array_equal(golden, output):
            print(f"[INFO] v{index} passed (bit-exact, n={golden.size})")
            continue
        failed = True
        if golden.shape != output.shape:
            print(f"[ERROR] v{index}: shape {output.shape}, expected {golden.shape}")
            continue
        bad = np.flatnonzero(golden != output)
        first = int(bad[0])
        print(f"[ERROR] v{index}: {bad.size} mismatches, first_index={first} "
              f"golden=0x{int(golden[first]):04X} output=0x{int(output[first]):04X}")
    if failed:
        sys.exit(2)
    print("[INFO] f32 split-halves check passed")


if __name__ == "__main__":
    main()
