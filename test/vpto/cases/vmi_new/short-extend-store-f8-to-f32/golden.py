#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Golden for the short packed f8e4m3 -> f32 widen stores with sentinel gaps."""

import argparse
from pathlib import Path

import numpy as np

ELEMS = 256
SEED = 23
SENTINEL = np.float32(-123.25)

# f8e4m3fn bit patterns for values that are exactly representable in f32, so the
# widening is exact and the comparison can be exact.
F8_VALUES = {
    0x00: 0.0,
    0x38: 1.0,
    0xB8: -1.0,
    0x30: 0.5,
    0x40: 2.0,
    0xC0: -2.0,
    0x48: 4.0,
    0xC8: -4.0,
}
# (n, source element offset, destination element offset); every offset is 32B
# aligned for its element type (see the kernel's bound reads and stores).
SHAPES = ((1, 0, 0), (2, 32, 32), (4, 64, 64), (8, 96, 96))


def decode(byte: int) -> np.float32:
    if byte not in F8_VALUES:
        raise ValueError(f"unexpected f8e4m3fn byte 0x{byte:02x} in the source")
    return np.float32(F8_VALUES[byte])


def generate(output_dir: Path, seed: int) -> None:
    del seed  # the pattern is deterministic; the seed keeps the CLI uniform
    pattern = np.array(sorted(F8_VALUES), dtype=np.uint8)
    src = np.tile(pattern, (ELEMS + pattern.size - 1) // pattern.size)[:ELEMS]
    dst = np.full(ELEMS, SENTINEL, dtype=np.float32)
    golden = dst.copy()
    for n, src_offset, dst_offset in SHAPES:
        for k in range(n):
            golden[dst_offset + k] = decode(int(src[src_offset + k]))

    output_dir.mkdir(parents=True, exist_ok=True)
    src.tofile(output_dir / "v1.bin")
    dst.tofile(output_dir / "v2.bin")
    golden.tofile(output_dir / "golden_v2.bin")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    generate(args.output_dir, args.seed)


if __name__ == "__main__":
    main()
