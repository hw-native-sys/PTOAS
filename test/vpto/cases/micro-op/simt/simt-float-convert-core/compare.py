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


def main():
    strict = os.getenv("COMPARE_STRICT", "1") != "0"
    golden = np.fromfile("golden_v1.bin", dtype=np.int32)
    out = np.fromfile("v1.bin", dtype=np.int32)
    written = np.concatenate(
        (
            np.arange(0, 18, dtype=np.intp),
            np.arange(32, 96, dtype=np.intp),
            np.arange(96, 128, dtype=np.intp),
            np.arange(128, 224, dtype=np.intp),
            np.arange(224, 672, dtype=np.intp),
        )
    )
    ok = golden.shape == out.shape and np.array_equal(golden[written], out[written])
    if not ok:
        idxs = written[golden[written] != out[written]]
        idx = int(idxs[0]) if idxs.size else 0
        print(
            f"[ERROR] mismatch at idx={idx}, golden={int(golden[idx])}, out={int(out[idx])}"
        )
        if strict:
            sys.exit(2)
    print("[INFO] compare passed" if ok else "[WARN] compare failed (non-gating)")


if __name__ == "__main__":
    main()
