#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Reference for bf16x2 -> f4E2M1x2 (second FP4 format).
# E2M1 magnitude table {0, 0.5, 1, 1.5, 2, 3, 4, 6} for codes 0..7; code bit3
# is sign. Input values are exactly representable so no rounding ambiguity.
# Calibration note: the E2M1 table, rounding interpretation, and low/high
# nibble order below are provisional oracles. Confirm them independently on the
# first A5 NPU run before treating a strict golden mismatch as a compiler bug.

import argparse
from pathlib import Path

import numpy as np

SRC_ELEMS = 128          # bf16 input lanes
DST_ELEMS = 64           # f4x2 output bytes

# Magnitude values for E2M1 code 0..7 (code bit3 = sign).
E2M1_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
PAIR_HI_FIRST = True

# Exactly representable E2M1 values (both signs, spanning the range).
VALUES = np.array([0.0, 0.5, -0.5, 1.0, -1.0, 1.5, -1.5, 2.0, -2.0, 3.0, 4.0, 6.0],
                  dtype=np.float32)


def f32_to_bf16_bits(values: np.ndarray) -> np.ndarray:
    """Round f32 to nearest bf16 and return its 16-bit pattern."""
    f32 = values.astype(np.float32)
    bits = f32.view(np.uint32)
    lsb = (bits >> 16) & 1
    rounding = 0x7FFF + lsb
    bf16 = (bits + rounding) >> 16
    return bf16.astype(np.uint16)


def bf16_to_f32(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << 16).view(np.float32)


def quantize_f4e2m1(values: np.ndarray) -> np.ndarray:
    """Nearest f4E2M1 code (0..15) for each f32 value."""
    sign = (values < 0).astype(np.int32)
    mag = np.abs(values)
    table = np.array(E2M1_VALUES, dtype=np.float32)
    idx = np.argmin(np.abs(table[None, :] - mag[:, None]), axis=1)
    return ((sign << 3) | idx).astype(np.uint8)


def generate(output_dir: Path) -> None:
    repeats = (SRC_ELEMS + len(VALUES) - 1) // len(VALUES)
    src_f32 = np.tile(VALUES, repeats)[:SRC_ELEMS].astype(np.float32)
    src_bf16 = f32_to_bf16_bits(src_f32)

    f4 = quantize_f4e2m1(bf16_to_f32(src_bf16)).astype(np.int32)
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
