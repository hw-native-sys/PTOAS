#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Golden for the short f32 -> packed f8e4m3 narrow stores with sentinel gaps."""

import argparse
from pathlib import Path

import numpy as np

ELEMS = 256
SEED = 23
SENTINEL = np.uint8(0xA5)

# Values that are exactly representable in f8e4m3fn, with their bit patterns, so
# the narrowing is exact and the comparison can be exact.
F8_VALUES = (
    (0.0, 0x00),
    (0.5, 0x30),
    (1.0, 0x38),
    (2.0, 0x40),
    (4.0, 0x48),
    (-1.0, 0xB8),
    (-2.0, 0xC0),
    (-4.0, 0xC8),
)
# (n, source element offset, destination element offset); every offset is 32B
# aligned for its element type (see the kernel's bound reads and stores).
SHAPES = ((1, 0, 0), (2, 32, 32), (4, 64, 64), (8, 96, 96))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    del args.seed  # the pattern is deterministic; the seed keeps the CLI uniform

    values = np.array([value for value, _ in F8_VALUES], dtype=np.float32)
    patterns = np.array([pattern for _, pattern in F8_VALUES], dtype=np.uint8)
    repeats = (ELEMS + values.size - 1) // values.size
    src = np.tile(values, repeats)[:ELEMS]
    dst = np.full(ELEMS, SENTINEL, dtype=np.uint8)
    golden = dst.copy()
    for n, src_offset, dst_offset in SHAPES:
        for k in range(n):
            value = np.float32(src[src_offset + k])
            matches = np.nonzero(values == value)[0]
            if matches.size != 1:
                raise ValueError(f"value {value} is not an exact f8e4m3fn pattern")
            golden[dst_offset + k] = patterns[int(matches[0])]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    src.tofile(args.output_dir / "v1.bin")
    dst.tofile(args.output_dir / "v2.bin")
    golden.tofile(args.output_dir / "golden_v2.bin")


if __name__ == "__main__":
    main()
