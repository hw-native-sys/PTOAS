#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import argparse
from pathlib import Path

import numpy as np


def generate(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    src = np.array(
        [0x01020304, -0x1234567, 0x11223344, -7, 99, 100, 101, 102],
        dtype=np.int32,
    )
    dst = np.full(8, np.int32(0x5A5A5A5A), dtype=np.int32)
    golden_dst = dst.copy()
    golden_dst[3] = src[1]
    src.tofile(output_dir / "src.bin")
    dst.tofile(output_dir / "dst.bin")
    golden_dst.tofile(output_dir / "golden_dst.bin")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    generate(args.output_dir)


if __name__ == "__main__":
    main()
