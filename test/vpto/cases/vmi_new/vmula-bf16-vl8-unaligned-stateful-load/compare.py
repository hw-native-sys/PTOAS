#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# vmula bf16 output is bit-compared as uint16 bf16 bit patterns: the DUT
# rounds once to bf16 (RNE), so the simulator/NPU result must match the
# golden bits exactly.  A mismatch here means the vldas+vldus stateful load
# (#1374) read wrong lanes from the misaligned address.

import sys

import numpy as np


def main() -> None:
    golden = np.fromfile("golden_v4.bin", dtype=np.uint16)
    output = np.fromfile("v4.bin", dtype=np.uint16)
    if golden.shape != output.shape or not np.array_equal(golden, output):
        common_size = min(golden.size, output.size)
        diff = np.nonzero(golden[:common_size] != output[:common_size])[0]
        idx = int(diff[0]) if diff.size else common_size
        golden_value = int(golden[idx]) if idx < golden.size else "n/a"
        output_value = int(output[idx]) if idx < output.size else "n/a"
        print(f"[ERROR] compare failed idx={idx} golden={golden_value} output={output_value}")
        sys.exit(2)
    print("[INFO] compare passed")


if __name__ == "__main__":
    main()
