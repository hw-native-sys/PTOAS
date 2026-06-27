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


def compare_bin(golden_path, output_path, dtype, eps):
    if not os.path.exists(golden_path):
        print(f"[ERROR] Golden missing: {golden_path}")
        return False
    if not os.path.exists(output_path):
        print(f"[ERROR] Output missing: {output_path}")
        return False

    golden = np.fromfile(golden_path, dtype=dtype)
    output = np.fromfile(output_path, dtype=dtype)
    if golden.shape != output.shape:
        print(f"[ERROR] Shape mismatch: {golden.shape} vs {output.shape}")
        return False
    if not np.allclose(golden, output, atol=eps, rtol=eps, equal_nan=True):
        diff = np.abs(golden.astype(np.float64) - output.astype(np.float64))
        idx = int(np.argmax(diff))
        print(f"[ERROR] Mismatch: idx={idx} golden={golden[idx]} out={output[idx]}")
        return False
    return True


def main():
    strict = os.getenv("COMPARE_STRICT", "1") != "0"
    ok = compare_bin("golden_v2.bin", "v2.bin", np.float32, 1e-6)
    if not ok:
        if strict:
            print("[ERROR] compare failed")
            sys.exit(2)
        print("[WARN] compare failed (non-gating)")
        return
    print("[INFO] compare passed")


if __name__ == "__main__":
    main()
