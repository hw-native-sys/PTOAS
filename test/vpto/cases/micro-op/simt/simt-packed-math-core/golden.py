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

ELEMS = 32


def pack_u16_pair(lo: int, hi: int) -> np.uint32:
    return np.uint32((hi << 16) | lo)


def generate(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    initial = np.zeros(ELEMS, dtype=np.uint32)
    golden = np.zeros(ELEMS, dtype=np.uint32)

    golden[0] = pack_u16_pair(0x3C00, 0x3C00)   # shuffle: f16x2 (1, 1)
    golden[1] = pack_u16_pair(0x4000, 0x4000)   # sqrt: f16x2 (2, 2)
    golden[2] = pack_u16_pair(0x3C00, 0x4000)   # abs: f16x2 (1, 2)
    golden[3] = pack_u16_pair(0x3C00, 0x3C00)   # exp(0): f16x2 (1, 1)
    golden[4] = pack_u16_pair(0x0000, 0x0000)   # log(1): f16x2 (0, 0)
    golden[5] = pack_u16_pair(0x3C00, 0x3C00)   # min: f16x2 (1, 1)
    golden[6] = pack_u16_pair(0x4400, 0x4400)   # pow(2, 2): f16x2 (4, 4)
    golden[7] = pack_u16_pair(0x4200, 0x4200)   # fma: f16x2 (3, 3)
    golden[8] = pack_u16_pair(0x3C00, 0x4000)   # f32x2 -> f16x2 (1, 2)
    golden[9] = pack_u16_pair(0x3F80, 0x4000)   # abs: bf16x2 (1, 2)
    golden[10] = pack_u16_pair(0x4000, 0x4000)  # max: bf16x2 (2, 2)
    golden[11] = pack_u16_pair(0x4040, 0x4040)  # fma: bf16x2 (3, 3)
    golden[12] = pack_u16_pair(0x3F80, 0x4000)  # f32x2 -> bf16x2 (1, 2)
    golden[13] = np.uint32(0x4038)              # f32x2 -> f8e4m3x2 (1, 2)
    golden[14] = np.uint32(0x403C)              # f32x2 -> f8e5m2x2 (1, 2)
    golden[15] = np.uint32(0x1008)              # f32x2 -> hif8x2 (1, 2)
    golden[16] = np.uint32(0x3F800000)          # f16x2 -> f32x2 lane 0
    golden[17] = np.uint32(0x40000000)          # f16x2 -> f32x2 lane 1
    golden[18] = np.uint32(0x3F800000)          # bf16x2 -> f32x2 lane 0
    golden[19] = np.uint32(0x40000000)          # bf16x2 -> f32x2 lane 1
    golden[20] = np.uint32(0x3F800000)          # f8e4m3x2 -> f32x2 lane 0
    golden[21] = np.uint32(0x40000000)          # f8e4m3x2 -> f32x2 lane 1
    golden[22] = np.uint32(0x3F800000)          # f8e5m2x2 -> f32x2 lane 0
    golden[23] = np.uint32(0x40000000)          # f8e5m2x2 -> f32x2 lane 1
    golden[24] = np.uint32(0x3F800000)          # hif8x2 -> f32x2 lane 0
    golden[25] = np.uint32(0x40000000)          # hif8x2 -> f32x2 lane 1

    initial.tofile(output_dir / "v1.bin")
    golden.tofile(output_dir / "golden_v1.bin")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    generate(args.output_dir)


if __name__ == "__main__":
    main()
