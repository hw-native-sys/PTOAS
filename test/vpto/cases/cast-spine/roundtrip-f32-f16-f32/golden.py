#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Golden data for roundtrip-f32-f16-f32.

Chain:  f32 --vcvt(SAT)--> f16 --vcvt--> f32 --*1.5--> f32

The input set is exact in f16 (multiples of 0.25 with a small magnitude) and the
product by 1.5 stays exact in f32, so the oracle is bit-exact.
"""

import argparse
from pathlib import Path

import numpy as np

ELEMS = 8192
SCALE = np.float32(1.5)
EXACT_VALUES = np.array(
    [0.0, 0.25, -0.25, 0.5, -0.5, 0.75, 1.0, -1.0, 1.5, -1.5, 2.0, -2.0,
     2.5, -2.5, 3.0, -3.0, 4.0, -4.0],
    dtype=np.float32,
)


def generate(output_dir: Path) -> None:
    rng = np.random.default_rng(1337)
    src = EXACT_VALUES[rng.integers(0, EXACT_VALUES.size, ELEMS)].astype(np.float32)
    restored = src.astype(np.float16).astype(np.float32)
    golden = restored * SCALE

    output_dir.mkdir(parents=True, exist_ok=True)
    src.tofile(output_dir / "v1.bin")
    np.zeros(ELEMS, dtype=np.float32).tofile(output_dir / "v2.bin")
    golden.astype(np.float32).tofile(output_dir / "golden_v2.bin")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    generate(args.output_dir)


if __name__ == "__main__":
    main()
