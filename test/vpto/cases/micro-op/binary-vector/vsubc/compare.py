#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# case: micro-op/binary-vector/vsubc
# family: binary-vector
# target_ops: pto.vsubc
# scenarios: core-u32-unsigned, full-mask, borrow-boundary

import os
import sys
import numpy as np


REPEAT_BYTES = 256
LOGICAL_ELEMS = 64
SRC_ELEM_BYTES = 4


def _ceil_div(x, y):
    return (x + y - 1) // y


def _packed_pred_storage_bytes(logical_elems, src_elem_bytes):
    repeat_elems = REPEAT_BYTES // src_elem_bytes
    repeat_times = _ceil_div(logical_elems, repeat_elems) + 1
    loop_count = repeat_times // 2
    return loop_count * 16


def compare_result():
    golden = np.fromfile("golden_v3.bin", dtype=np.uint32, count=64)
    output = np.fromfile("v3.bin", dtype=np.uint32, count=64)
    return golden.shape == output.shape and np.array_equal(golden, output)


def compare_borrow():
    prefix_bytes = _packed_pred_storage_bytes(LOGICAL_ELEMS, SRC_ELEM_BYTES)
    golden = np.fromfile("golden_v4.bin", dtype=np.uint8)
    output = np.fromfile("v4.bin", dtype=np.uint8)
    if golden.size < prefix_bytes or output.size < prefix_bytes:
        return False
    return np.array_equal(golden[:prefix_bytes], output[:prefix_bytes])


def main():
    strict = os.getenv("COMPARE_STRICT", "1") != "0"
    ok = compare_result() and compare_borrow()
    if not ok:
        if strict:
            print("[ERROR] compare failed")
            sys.exit(2)
        print("[WARN] compare failed (non-gating)")
        return
    print("[INFO] compare passed")


if __name__ == "__main__":
    main()
