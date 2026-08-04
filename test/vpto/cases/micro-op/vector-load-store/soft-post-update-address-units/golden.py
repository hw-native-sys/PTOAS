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


ELEMENTS = 1024
SEED = 29


def generate(output_dir: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    data = rng.uniform(-16.0, 16.0, size=(ELEMENTS,)).astype(np.float32)
    output = np.zeros((ELEMENTS,), dtype=np.float32)
    spr_input = (np.arange(ELEMENTS, dtype=np.uint32) * 17) + 3
    spr_output = np.zeros((ELEMENTS,), dtype=np.uint32)

    golden_spr = spr_input.copy()
    # The initial base is element 4 and the signed immediate is -1 word. The
    # post-update recurrence therefore stores cleared AR values at [3, 7).
    golden_spr[3:7] = 0

    output_dir.mkdir(parents=True, exist_ok=True)
    data.tofile(output_dir / "input.bin")
    output.tofile(output_dir / "output.bin")
    data.tofile(output_dir / "golden_output.bin")
    spr_input.tofile(output_dir / "input_spr.bin")
    spr_output.tofile(output_dir / "output_spr.bin")
    golden_spr.tofile(output_dir / "golden_output_spr.bin")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate data for soft post-update address-unit validation."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    generate(args.output_dir, args.seed)


if __name__ == "__main__":
    main()
