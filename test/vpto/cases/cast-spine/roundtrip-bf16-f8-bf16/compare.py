#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Output check for the cast-spine case.

v2.bin is compared as raw bf16 bit patterns; a bit-exact match is expected
because the golden chain is lossless for the chosen inputs.
"""

import sys

import numpy as np

DTYPE = np.uint16


ATOL = 1.0e-3


def main() -> None:
    golden = np.fromfile("golden_v2.bin", dtype=DTYPE)
    output = np.fromfile("v2.bin", dtype=DTYPE)
    if golden.shape != output.shape:
        print(f"[ERROR] shape mismatch golden={golden.shape} output={output.shape}")
        sys.exit(2)
    if np.array_equal(golden, output):
        print(f"[INFO] compare passed (bit-exact, n={golden.size})")
        return
    delta = np.abs(golden.astype(np.float64) - output.astype(np.float64))
    maxdiff = float(delta.max())
    bad = np.nonzero(delta > ATOL)[0]
    if bad.size:
        idx = int(bad[0])
        print(f"[ERROR] compare failed n_bad={bad.size} idx={idx} "
              f"golden={golden[idx]} output={output[idx]} maxdiff={maxdiff}")
        sys.exit(2)
    print(f"[WARN] compare passed with tolerance atol={ATOL} maxdiff={maxdiff}")


if __name__ == "__main__":
    main()
