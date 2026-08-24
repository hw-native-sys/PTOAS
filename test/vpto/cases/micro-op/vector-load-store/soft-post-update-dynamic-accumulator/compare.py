#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import numpy as np


def main() -> None:
    golden = np.fromfile("golden_output.bin", dtype=np.float32)
    output = np.fromfile("output.bin", dtype=np.float32)
    if golden.shape != output.shape:
        print(
            f"[ERROR] dynamic accumulator shape mismatch: "
            f"golden={golden.shape}, output={output.shape}"
        )
        raise SystemExit(2)
    if not np.array_equal(golden, output):
        mismatch = np.flatnonzero(golden != output)
        index = int(mismatch[0])
        print(
            f"[ERROR] dynamic accumulator mismatch: idx={index}, "
            f"golden={golden[index]}, output={output[index]}"
        )
        raise SystemExit(2)
    print("[INFO] dynamic accumulator soft-postupdate compare passed")


if __name__ == "__main__":
    main()
