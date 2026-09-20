#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Golden for vmi-lane-stride-iota-f32-ls2:
#   v1.bin: src input – 32 f32 zeros (vload seeds lane_stride=2).
#   v2.bin: dst sentinel → overwritten by kernel.
#   golden_v2.bin: vci(0.0, lane_stride=2, VL=32 f32) → logical [0.0..31.0].
#   PK_B64 store selects every 2nd physical lane → 32 f32 values [0.0..31.0].

import argparse
from pathlib import Path

import numpy as np


def generate(output_dir: Path) -> None:
    src = np.zeros(32, dtype=np.float32)
    dst_sentinel = np.full(32, -1.0, dtype=np.float32)
    golden = np.arange(32, dtype=np.float32)

    output_dir.mkdir(parents=True, exist_ok=True)
    src.tofile(output_dir / "v1.bin")
    dst_sentinel.tofile(output_dir / "v2.bin")
    golden.tofile(output_dir / "golden_v2.bin")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    generate(args.output_dir)


if __name__ == "__main__":
    main()
