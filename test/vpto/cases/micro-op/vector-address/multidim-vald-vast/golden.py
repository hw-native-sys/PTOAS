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


SOURCE_BYTES = 4096
OUTPUT_BYTES = 8192
SEED = 43


def generate(output_dir: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    values = rng.uniform(-32.0, 32.0, size=(SOURCE_BYTES // 4,)).astype(np.float32)
    data = values.view(np.uint8)
    golden = np.zeros((OUTPUT_BYTES,), dtype=np.uint8)

    golden[0:1024] = data[0:1024]
    golden[1024:2048] = data[0:1024]

    x2_base = 2048
    golden[x2_base : x2_base + 512] = data[0:512]
    golden[x2_base + 1024 : x2_base + 1280] = data[1024:1280]
    golden[x2_base + 2048 : x2_base + 2304] = data[2052:2308]

    output_dir.mkdir(parents=True, exist_ok=True)
    data.tofile(output_dir / "v1.bin")
    np.zeros((OUTPUT_BYTES,), dtype=np.uint8).tofile(output_dir / "v2.bin")
    golden.tofile(output_dir / "golden_v2.bin")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate inputs/golden for VPTO combined vector-address memory case."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    generate(args.output_dir, args.seed)


if __name__ == "__main__":
    main()
