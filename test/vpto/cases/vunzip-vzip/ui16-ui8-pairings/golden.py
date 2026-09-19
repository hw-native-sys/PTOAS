#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Deterministic input and identity golden for the ui16 -> ui8 pairing case.

Both regions are bit-exact round trips, so the oracle for the 16-bit device
output is the 16-bit input word itself.  The pattern mixes the sign/width edge
values and both byte halves so a wrong half or a byte swap is visible.
"""

import argparse
from pathlib import Path

import numpy as np

ELEMS = 384

WORDS = np.array(
    [
        0x0000,
        0x0001,
        0x00FF,
        0x0100,
        0x1234,
        0x7FFF,
        0x8000,
        0xABCD,
        0xFF00,
        0x00FF,
        0xFFFE,
        0xFFFF,
        0x0101,
        0x1010,
        0x55AA,
        0xAA55,
    ],
    dtype=np.uint16,
)


def generate(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    words = np.resize(WORDS, ELEMS).astype(np.uint16)
    words.tofile(output_dir / "v1.bin")
    np.zeros(ELEMS, dtype=np.uint16).tofile(output_dir / "v2.bin")
    words.tofile(output_dir / "golden_v2.bin")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    generate(args.output_dir)


if __name__ == "__main__":
    main()
