#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Golden for the short packed si8 -> si32 widen stores with sentinel gaps."""

import argparse
from pathlib import Path

import numpy as np

ELEMS = 256
SEED = 23
SENTINEL = np.int32(-123)
PATTERN = np.array([1, 2, 3, 4, -1, -2, -3, -4], dtype=np.int8)
# (n, source element offset, destination element offset)
SHAPES = ((1, 0, 0), (2, 32, 32), (4, 64, 64), (8, 96, 96))


def generate(output_dir: Path, seed: int) -> None:
    del seed  # the pattern is deterministic; the seed keeps the CLI uniform
    src = np.tile(PATTERN, (ELEMS + PATTERN.size - 1) // PATTERN.size)[:ELEMS]
    dst = np.full(ELEMS, SENTINEL, dtype=np.int32)
    golden = dst.copy()
    for n, src_offset, dst_offset in SHAPES:
        golden[dst_offset:dst_offset + n] = src[src_offset:src_offset + n]

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
