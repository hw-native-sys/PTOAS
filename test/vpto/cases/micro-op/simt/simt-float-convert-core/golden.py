#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import argparse
from pathlib import Path

import numpy as np

ELEMS = 1024


def generate(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    v1 = np.full(ELEMS, -1, dtype=np.int32)
    golden_v1 = np.zeros(ELEMS, dtype=np.int32)
    golden_v1[:18] = np.array(
        [3, 2, 2, 1, 2, 2, 0, 16, 16, 1, 2, 2, 2, 6, 6, 6, -4, 4],
        dtype=np.int32,
    )
    values = np.arange(-16, 16, dtype=np.float16)
    three = np.float16(3)
    golden_v1[32:64] = np.minimum(values, three).view(np.uint16).astype(np.int32)
    golden_v1[64:96] = np.maximum(values, three).view(np.uint16).astype(np.int32)
    golden_v1[96:128] = np.abs(np.arange(-16, 16, dtype=np.float16)).view(np.uint16)
    signed_tid = np.arange(-16, 16, dtype=np.int32)
    as_f32 = signed_tid.astype(np.float32)
    golden_v1[128:160] = as_f32.view(np.int32)
    golden_v1[160:192] = as_f32.astype(np.int32)
    golden_v1[192:224] = as_f32.astype(np.float16).view(np.uint16).astype(np.int32)
    rounding_results = [2, -2, 3, -3, 2, -3, 3, -2, 2, -2]
    for index, value in enumerate(rounding_results):
        start = 224 + index * 32
        golden_v1[start : start + 32] = value
    golden_v1[544:576] = np.iinfo(np.int32).max
    golden_v1[576:608] = np.iinfo(np.int32).min
    golden_v1[608:640] = 0x7BFF
    golden_v1[640:672] = 0x7C00
    v1.tofile(output_dir / "v1.bin")
    golden_v1.tofile(output_dir / "golden_v1.bin")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    generate(args.output_dir)


if __name__ == "__main__":
    main()
