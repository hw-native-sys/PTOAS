#!/usr/bin/python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import os
import numpy as np


CASES = [
    ("fp8_load_16x64", (16, 64)),
    ("fp8_load_32x64", (32, 64)),
    ("fp8_load_128x64", (128, 64)),
]


def main():
    rng = np.random.default_rng(20260707)
    for name, shape in CASES:
        case_dir = name
        os.makedirs(case_dir, exist_ok=True)
        data = rng.integers(0, 256, size=shape, dtype=np.uint8)
        data.tofile(os.path.join(case_dir, "input.bin"))
        data.tofile(os.path.join(case_dir, "golden.bin"))
        print(f"[INFO] Generated {name}: shape={shape}, dtype=uint8")


if __name__ == "__main__":
    main()
