#!/usr/bin/env python3
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


def main() -> None:
    strict = os.getenv("COMPARE_STRICT", "1") != "0"
    golden = np.fromfile("golden_dst.bin", dtype=np.int32)
    output = np.fromfile("dst.bin", dtype=np.int32)
    ok = golden.shape == output.shape and np.array_equal(golden, output)
    if not ok:
        diff = np.nonzero(golden != output)[0]
        idx = int(diff[0]) if diff.size else 0
        print(
            f"[ERROR] mismatch at idx={idx}, golden={int(golden[idx])}, "
            f"out={int(output[idx])}"
        )
        if strict:
            sys.exit(2)
    print("[INFO] compare passed" if ok else "[WARN] compare failed (non-gating)")


if __name__ == "__main__":
    main()
