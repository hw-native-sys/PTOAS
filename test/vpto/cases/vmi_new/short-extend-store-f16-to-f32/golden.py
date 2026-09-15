#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Golden for the short dense f16 -> f32 widen stores with sentinel gaps."""

import argparse
from pathlib import Path

import numpy as np

ELEMS = 256
SEED = 23
SENTINEL = np.float32(-123.25)
# (n, source element offset, destination element offset)
SHAPES = ((1, 0, 0), (2, 16, 32), (4, 32, 64), (8, 48, 96))


def generate(output_dir: Path, seed: int) -> None:
    # f16 -> f32 widening is exact and every value is exactly representable in
    # f16, so the comparison can be exact.
    src = (np.arange(ELEMS, dtype=np.float32) / 2.0 + 0.5).astype(np.float16)
    dst = np.full(ELEMS, SENTINEL, dtype=np.float32)
    golden = dst.copy()
    for n, src_offset, dst_offset in SHAPES:
        golden[dst_offset:dst_offset + n] = src[src_offset:src_offset + n].astype(np.float32)

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
