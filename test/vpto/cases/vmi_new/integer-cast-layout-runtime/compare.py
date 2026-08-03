#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import sys

import numpy as np


OUTPUT_DTYPES = {
    2: np.uint16, 3: np.uint32, 5: np.int16, 6: np.int32,
    8: np.uint32, 9: np.uint8, 11: np.int32, 12: np.int8,
    14: np.uint16, 15: np.uint8, 17: np.int16, 18: np.int8,
    21: np.uint16, 22: np.uint32, 23: np.uint32,
}


def main() -> None:
    failed = False
    for index, dtype in OUTPUT_DTYPES.items():
        golden = np.fromfile(f"golden_v{index}.bin", dtype=dtype)
        output = np.fromfile(f"v{index}.bin", dtype=dtype)
        if golden.shape == output.shape and np.array_equal(golden, output):
            continue
        failed = True
        if golden.shape != output.shape:
            print(f"[ERROR] v{index}: shape {output.shape}, expected {golden.shape}")
            continue
        mismatches = np.flatnonzero(golden != output)
        shown = mismatches[:8]
        details = ", ".join(
            f"{int(i)}:{golden[i]}!={output[i]}" for i in shown)
        print(f"[ERROR] v{index}: {mismatches.size} mismatches ({details})")
    if failed:
        sys.exit(2)
    print("[INFO] all integer cast outputs match exactly")


if __name__ == "__main__":
    main()
