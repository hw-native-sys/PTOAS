#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Reference for bf16x2 -> f4E1M2x2 rounding {R,A,Z} on f4-boundary values.
# Three chunks of 128 bf16 are quantized with RNE / RTA / RTZ respectively
# (the same input set, so the output regions differ only by rounding mode).
# Calibration note: the E1M2 table, R/A/Z interpretation, tie behavior, and
# low/high nibble order below are provisional oracles. Confirm them on the first
# A5 NPU run before treating a strict golden mismatch as a compiler regression.

import argparse
from pathlib import Path

import numpy as np

SRC_ELEMS = 384          # bf16 input lanes (3 chunks of 128)
DST_ELEMS = 192          # f4x2 output bytes (3 regions of 64)

# Magnitude values for E1M2 code 0..7 (code bit3 = sign). IEEE-style table.
E1M2_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
PAIR_HI_FIRST = True

# f4-boundary values (exactly halfway between two representable E1M2 values),
# so RNE / RTA / RTZ exercise tie-breaking differences.
VALUES = np.array([0.125, 0.375, 0.625, 1.125, 1.375, 1.625, -0.125, -0.625],
                  dtype=np.float32)
CHUNK = 128


def f32_to_bf16_bits(values: np.ndarray) -> np.ndarray:
    f32 = values.astype(np.float32)
    bits = f32.view(np.uint32)
    lsb = (bits >> 16) & 1
    rounding = 0x7FFF + lsb
    bf16 = (bits + rounding) >> 16
    return bf16.astype(np.uint16)


def bf16_to_f32(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << 16).view(np.float32)


def quantize_f4e1m2(values: np.ndarray, mode: str) -> np.ndarray:
    """Nearest f4E1M2 code; ties (halfway) resolved by rounding mode."""
    table = np.array(E1M2_VALUES, dtype=np.float32)
    out = np.zeros(len(values), dtype=np.int32)
    for i, v in enumerate(values):
        sign = 1 if v < 0 else 0
        mag = abs(float(v))
        d = np.abs(table - mag)
        cands = np.where(np.isclose(d, d.min()))[0]
        if len(cands) == 1:
            idx = int(cands[0])
        else:
            lo, hi = int(cands[0]), int(cands[1])
            if mode == "rne":
                idx = lo if lo % 2 == 0 else hi
            elif mode == "rta":
                idx = hi
            elif mode == "rtz":
                idx = lo
            else:
                raise ValueError(mode)
        out[i] = (sign << 3) | idx
    return out.astype(np.uint8)


def pack_pairs(f4: np.ndarray) -> np.ndarray:
    f4 = f4.astype(np.int32)
    even = f4[0::2]
    odd = f4[1::2]
    if PAIR_HI_FIRST:
        return ((odd << 4) | even).astype(np.uint8)
    return ((even << 4) | odd).astype(np.uint8)


def generate(output_dir: Path) -> None:
    repeats = (SRC_ELEMS + len(VALUES) - 1) // len(VALUES)
    src_f32 = np.tile(VALUES, repeats)[:SRC_ELEMS].astype(np.float32)
    src_bf16 = f32_to_bf16_bits(src_f32)

    modes = ["rne", "rta", "rtz"]
    dst = np.concatenate([
        pack_pairs(quantize_f4e1m2(bf16_to_f32(src_bf16)[c * CHUNK:(c + 1) * CHUNK], m))
        for c, m in enumerate(modes)
    ])
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
