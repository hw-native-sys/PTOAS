#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Output check for the cast-spine case.

The si32 -> si16 -> si32 round trip and +1 are exact for the chosen inputs.
"""

import sys

import numpy as np

DTYPE = np.int32


def main() -> None:
    golden = np.fromfile("golden_v2.bin", dtype=DTYPE)
    output = np.fromfile("v2.bin", dtype=DTYPE)
    if golden.shape != output.shape:
        print(f"[ERROR] shape mismatch golden={golden.shape} output={output.shape}")
        sys.exit(2)
    if not np.array_equal(golden, output):
        bad = np.nonzero(golden != output)[0]
        idx = int(bad[0]) if bad.size else -1
        print(f"[ERROR] compare failed n_mismatch={bad.size} idx={idx} "
              f"golden={golden[idx] if idx >= 0 else 'n/a'} "
              f"output={output[idx] if idx >= 0 else 'n/a'}")
        sys.exit(2)
    print(f"[INFO] compare passed (bit-exact, n={golden.size})")


if __name__ == "__main__":
    main()
