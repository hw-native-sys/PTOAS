#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Output check for quant-dequant-bf16-f8-bf16.

v2.bin is the raw f8E4M3FN byte stream (bit-exact expected) and v3.bin the
dequantised bf16 bit pattern (bit-exact expected): both chains are lossless for
the chosen input set.
"""

import sys

import numpy as np


def main() -> None:
    gold_q = np.fromfile("golden_v2.bin", dtype=np.uint8)
    out_q = np.fromfile("v2.bin", dtype=np.uint8)
    gold_y = np.fromfile("golden_v3.bin", dtype=np.uint16)
    out_y = np.fromfile("v3.bin", dtype=np.uint16)

    if gold_q.shape != out_q.shape or not np.array_equal(gold_q, out_q):
        bad = np.nonzero(gold_q != out_q)[0] if gold_q.shape == out_q.shape else []
        idx = int(bad[0]) if bad.size else -1
        print(f"[ERROR] f8 compare failed n_mismatch={bad.size} idx={idx} "
              f"golden={gold_q[idx] if idx >= 0 else 'n/a'} "
              f"output={out_q[idx] if idx >= 0 else 'n/a'}")
        sys.exit(2)

    if gold_y.shape != out_y.shape or not np.array_equal(gold_y, out_y):
        bad = np.nonzero(gold_y != out_y)[0] if gold_y.shape == out_y.shape else []
        idx = int(bad[0]) if bad.size else -1
        print(f"[ERROR] bf16 compare failed n_mismatch={bad.size} idx={idx} "
              f"golden=0x{int(gold_y[idx]):04x} output=0x{int(out_y[idx]):04x}"
              if idx >= 0 else "[ERROR] bf16 compare failed shape mismatch")
        sys.exit(2)

    print(f"[INFO] compare passed (bit-exact f8 n={gold_q.size}, bf16 n={gold_y.size})")


if __name__ == "__main__":
    main()
