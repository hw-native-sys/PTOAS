#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Golden data for roundtrip-si32-si16-si32.

Chain:  si32 --vcvt(SAT)--> si16 --vcvt--> si32 --+1--> si32

The input set stays inside the si16 range, so the narrowing/widening round trip
is lossless and the oracle is bit-exact.
"""

import argparse
from pathlib import Path

import numpy as np

ELEMS = 4096


def generate(output_dir: Path) -> None:
    rng = np.random.default_rng(1337)
    src = rng.integers(-1000, 1001, ELEMS).astype(np.int32)
    golden = src.astype(np.int16).astype(np.int32) + np.int32(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    src.tofile(output_dir / "v1.bin")
    np.zeros(ELEMS, dtype=np.int32).tofile(output_dir / "v2.bin")
    golden.astype(np.int32).tofile(output_dir / "golden_v2.bin")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    generate(args.output_dir)


if __name__ == "__main__":
    main()
