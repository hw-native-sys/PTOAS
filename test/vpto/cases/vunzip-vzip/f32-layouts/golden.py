#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Deterministic input and identity golden for the f32 vunzip/vzip layout case.

Every region is a bit-exact round trip, so the oracle for the device output is
the input bit pattern itself.  The words are chosen to be human checkable and
to exercise sign, exponent and mantissa edge values; the split is validated at
the bit level instead of by a float tolerance.
"""

import argparse
from pathlib import Path

import numpy as np

ELEMS = 768

# 0x3F800000 = 1.0f, 0x0000FFFF, 0xFFFFFFFF = NaN, 0x80000000 = -0.0f,
# 0x7F800000 = +inf, 0xFF800000 = -inf, 0x4B000001, 0x00000001 (denormal),
# 0x3E800000 = 0.25f, 0xC0490FDB = -pi, 0x3F000000 = 0.5f, 0x41000000 = 8.0f.
WORDS = np.array(
    [
        0x3F800000,
        0x0000FFFF,
        0xFFFFFFFF,
        0x80000000,
        0x7F800000,
        0xFF800000,
        0x4B000001,
        0x00000001,
        0x3E800000,
        0xC0490FDB,
        0x3F000000,
        0x41000000,
    ],
    dtype=np.uint32,
)


def generate(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    words = np.resize(WORDS, ELEMS).astype(np.uint32)
    words.tofile(output_dir / "v1.bin")
    np.zeros(ELEMS, dtype=np.uint32).tofile(output_dir / "v2.bin")
    words.tofile(output_dir / "golden_v2.bin")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    generate(args.output_dir)


if __name__ == "__main__":
    main()
