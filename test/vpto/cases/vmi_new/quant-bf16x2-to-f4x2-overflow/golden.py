#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Reference for bf16x2 -> f4E1M2x2 with out-of-range inputs.
# This provisional oracle maps to the nearest entry in the assumed finite E1M2
# table, including the endpoints for out-of-range values. The actual hardware
# encoding behavior, E1M2 table, rounding interpretation, and nibble order must
# be calibrated on the first A5 NPU run.

import argparse
from pathlib import Path

import numpy as np

SRC_ELEMS = 128          # bf16 input lanes
DST_ELEMS = 64           # f4x2 output bytes

E1M2_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
PAIR_HI_FIRST = True

# Overflow (|v| > 1.75), exact max, in-range, and underflow inputs.
VALUES = np.array([100.0, -100.0, 5.0, -5.0, 1.9, -1.9, 1.75, -1.75,
                   1.0, -1.0, 0.5, -0.5, 0.0, 1e-5, 3.0, -3.0],
                  dtype=np.float32)


def f32_to_bf16_bits(values: np.ndarray) -> np.ndarray:
    f32 = values.astype(np.float32)
    bits = f32.view(np.uint32)
    lsb = (bits >> 16) & 1
    rounding = 0x7FFF + lsb
    bf16 = (bits + rounding) >> 16
    return bf16.astype(np.uint16)


def bf16_to_f32(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << 16).view(np.float32)


def quantize_nearest_table(values: np.ndarray) -> np.ndarray:
    """Map each value to the nearest entry in the provisional E1M2 table."""
    sign = (values < 0).astype(np.int32)
    mag = np.abs(values)
    table = np.array(E1M2_VALUES, dtype=np.float32)
    idx = np.argmin(np.abs(table[None, :] - mag[:, None]), axis=1)
    return ((sign << 3) | idx).astype(np.uint8)


def generate(output_dir: Path) -> None:
    repeats = (SRC_ELEMS + len(VALUES) - 1) // len(VALUES)
    src_f32 = np.tile(VALUES, repeats)[:SRC_ELEMS].astype(np.float32)
    src_bf16 = f32_to_bf16_bits(src_f32)

    f4 = quantize_nearest_table(bf16_to_f32(src_bf16)).astype(np.int32)
    even = f4[0::2]
    odd = f4[1::2]
    if PAIR_HI_FIRST:
        dst = ((odd << 4) | even).astype(np.uint8)
    else:
        dst = ((even << 4) | odd).astype(np.uint8)
    assert dst.size == DST_ELEMS

    src_buf = src_bf16.astype("<u2")
    dst_buf = np.full(DST_ELEMS, 0xA5, dtype=np.uint8)

    output_dir.mkdir(parents=True, exist_ok=True)
    src_buf.tofile(output_dir / "v1.bin")
    dst_buf.tofile(output_dir / "v2.bin")
    dst.tofile(output_dir / "golden_v2.bin")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    generate(args.output_dir)


if __name__ == "__main__":
    main()
