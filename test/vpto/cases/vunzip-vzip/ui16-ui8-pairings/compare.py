#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Bit-exact check for the ui16 -> ui8 vunzip/vzip pairing case."""

import sys

import numpy as np

DTYPE = np.uint16


def main() -> None:
    golden = np.fromfile("golden_v2.bin", dtype=DTYPE)
    output = np.fromfile("v2.bin", dtype=DTYPE)
    if golden.shape != output.shape:
        print(f"[ERROR] shape mismatch golden={golden.shape} output={output.shape}")
        sys.exit(2)
    if np.array_equal(golden, output):
        print(f"[INFO] compare passed (bit-exact, n={golden.size})")
        return
    bad = np.flatnonzero(golden != output)
    first = int(bad[0])
    print(f"[ERROR] compare failed n_bad={bad.size} first_index={first} "
          f"golden=0x{int(golden[first]):04X} output=0x{int(output[first]):04X}")
    sys.exit(2)


if __name__ == "__main__":
    main()
