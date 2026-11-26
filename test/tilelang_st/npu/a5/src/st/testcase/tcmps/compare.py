#!/usr/bin/python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

import os
import sys
import numpy as np

from cases import CASES
from st_common import style_fail, style_pass, validate_cases


def main():
    validate_cases(CASES)
    case_filter = sys.argv[1] if len(sys.argv) > 1 else None

    all_passed = True
    for case in CASES:
        if case_filter is not None and case["name"] != case_filter:
            continue

        case_dir = case["name"]
        dst_rows, physical_cols = case.get("dst_shape", case["shape"])
        vr, vc = case["valid_shape"]

        # The 2D tcmps templates write each row at the destination tile's
        # physical row stride. Gather the packed predicate bytes of the valid
        # rows and compare them with the row-packed golden.
        row_bytes = (vc + 7) // 8
        golden = np.fromfile(os.path.join(case_dir, "golden.bin"), dtype=np.uint8)
        if golden.size != vr * row_bytes:
            print(style_fail(f"[ERROR] {case['name']}: unexpected golden size {golden.size}"))
            all_passed = False
            continue

        output = np.fromfile(os.path.join(case_dir, "output.bin"), dtype=np.uint8)
        needed = dst_rows * physical_cols
        if output.size != needed:
            print(style_fail(f"[ERROR] {case['name']}: output has {output.size} bytes, expected {needed}"))
            all_passed = False
            continue
        packed_output = output.reshape(dst_rows, physical_cols)[:vr, :row_bytes].reshape(-1)

        # Compare byte-by-byte
        ok = np.array_equal(golden, packed_output)
        if not ok:
            # Find first mismatch for debugging
            diff_mask = golden != packed_output
            diff_indices = np.where(diff_mask)[0]
            if len(diff_indices) > 0:
                diff_idx = diff_indices[0]
                max_diff = int(np.max(np.abs(golden.astype(int) - packed_output.astype(int))))
                print(style_fail(f"[ERROR] Mismatch: max diff={max_diff} at byte idx={diff_idx} "
                                 f"(golden=0x{golden[diff_idx]:02x}, output=0x{packed_output[diff_idx]:02x})"))
            else:
                print(style_fail(f"[ERROR] Mismatch: shapes differ golden={golden.shape} output={packed_output.shape}"))

        if ok:
            print(style_pass(f"[INFO] {case['name']}: compare passed"))
        else:
            print(style_fail(f"[ERROR] {case['name']}: compare failed"))
            all_passed = False

    if not all_passed:
        sys.exit(2)
    print(style_pass("[INFO] all cases passed"))


if __name__ == "__main__":
    main()
