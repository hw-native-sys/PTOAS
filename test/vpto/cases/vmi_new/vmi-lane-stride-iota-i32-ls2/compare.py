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


def main() -> None:
    golden = np.fromfile("golden_v2.bin", dtype=np.int32)
    output = np.fromfile("v2.bin", dtype=np.int32)
    if golden.shape != output.shape or not np.array_equal(golden, output):
        diff = np.nonzero(golden != output)[0]
        for idx in diff[:8]:
            g = int(golden[idx])
            o = int(output[idx])
            print(f"[ERROR] idx={int(idx)} golden={g} output={o}")
        print(f"[ERROR] total_diff={int(diff.size)}/{int(golden.size)}")
        neg = np.sum(output < 0)
        if neg > 0:
            print(f"[HINT]  {int(neg)} output values are negative — arithmetic-shift bug?")
        sys.exit(2)
    print("[INFO] compare passed")


if __name__ == "__main__":
    main()
