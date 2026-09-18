#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Golden for vmi-lane-stride-iota-i16-ls2:
#   vci(0, lane_stride=2, VL=64 i16) → logical [0..63].
#   PK_B32 store selects every 2nd physical lane → 64 i16 values [0..63].
#
# Regression guard: without the vbitcast fix, physical lanes 64-127 carry
# values that map to negative i16 after arithmetic >>1.

import argparse
from pathlib import Path

import numpy as np


def generate(output_dir: Path) -> None:
    sentinel = np.int16(-1)
    out = np.full(64, sentinel, dtype=np.int16)
    golden = np.arange(64, dtype=np.int16)

    assert np.all(golden >= 0), "golden contains negative values — bug in generator"

    output_dir.mkdir(parents=True, exist_ok=True)
    out.tofile(output_dir / "v1.bin")
    golden.tofile(output_dir / "golden_v1.bin")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    generate(args.output_dir)


if __name__ == "__main__":
    main()
