#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Reference for f4E1M2x2 -> bf16x2 (contiguous path).
# Each f4E1M2x2 byte packs two 4-bit FP4 E1M2 codes: low nibble = even bf16
# lane, high nibble = odd bf16 lane (mirrors the quant case PAIR_HI_FIRST).
# Each code dequantizes to one bf16 value; the pair becomes one bf16x2.
#
# Calibration note: the E1M2 table and nibble order below are provisional
# oracles mirroring quant-bf16x2-to-f4x2-contiguous. Confirm them on the first
# A5 NPU run before treating a strict golden mismatch as a compiler regression.

import argparse
from pathlib import Path

import numpy as np

SRC_ELEMS = 256          # f4x2 input bytes
DST_ELEMS = 512          # bf16 output lanes

# Magnitude values for E1M2 code 0..7 (code bit3 = sign). IEEE-style table.
E1M2_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
PAIR_HI_FIRST = True     # True: byte = (f4(odd) << 4) | f4(even); False: reversed

# Input f4 codes (all 16 codes, values exactly representable in the table).
CODES = np.array(
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], dtype=np.uint8
)


def f32_to_bf16_bits(values: np.ndarray) -> np.ndarray:
    """Round f32 to nearest bf16 and return its 16-bit pattern."""
    f32 = values.astype(np.float32)
    bits = f32.view(np.uint32)
    # round-to-nearest-even on the low 16 bits
    lsb = (bits >> 16) & 1
    rounding = 0x7FFF + lsb
    bf16 = (bits + rounding) >> 16
    return bf16.astype(np.uint16)


def dequantize_f4e1m2(codes: np.ndarray) -> np.ndarray:
    """f4E1M2 code (0..15) -> f32 value."""
    sign = ((codes >> 3) & 1).astype(np.int32)
    mag = np.array(E1M2_VALUES, dtype=np.float32)[(codes & 0x7).astype(np.int32)]
    values = np.where(sign == 1, -mag, mag).astype(np.float32)
    return values


def generate(output_dir: Path) -> None:
    # SRC_ELEMS f4x2 bytes pack 2 * SRC_ELEMS f4 codes.
    num_codes = SRC_ELEMS * 2
    repeats = (num_codes + len(CODES) - 1) // len(CODES)
    codes = np.tile(CODES, repeats)[:num_codes].astype(np.uint8)

    # Pack into f4x2 bytes: low nibble = even lane code, high nibble = odd.
    even_codes = codes[0::2]
    odd_codes = codes[1::2]
    assert even_codes.size == odd_codes.size == SRC_ELEMS
    if PAIR_HI_FIRST:
        src_bytes = ((odd_codes.astype(np.uint8) << 4) | even_codes).astype(np.uint8)
    else:
        src_bytes = ((even_codes.astype(np.uint8) << 4) | odd_codes).astype(np.uint8)

    # Dequantize: bf16[2j] = low nibble, bf16[2j+1] = high nibble.
    vals_even = dequantize_f4e1m2(even_codes)
    vals_odd = dequantize_f4e1m2(odd_codes)
    out_f32 = np.empty(DST_ELEMS, dtype=np.float32)
    out_f32[0::2] = vals_even
    out_f32[1::2] = vals_odd
    dst_bf16 = f32_to_bf16_bits(out_f32)

    src_buf = src_bytes.astype(np.uint8)
    dst_buf = np.full(DST_ELEMS, 0xA5, dtype=np.uint16)

    output_dir.mkdir(parents=True, exist_ok=True)
    src_buf.tofile(output_dir / "v1.bin")
    dst_buf.tofile(output_dir / "v2.bin")
    dst_bf16.tofile(output_dir / "golden_v2.bin")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    generate(args.output_dir)


if __name__ == "__main__":
    main()
