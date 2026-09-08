#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# case: kernels/tdump-vec
# family: kernels
# target_ops: pto.mte_gm_ub, pto.tdump
# scenarios: tdump-vec-f32-4x16, header-plus-row-major-payload

import argparse
from pathlib import Path

import numpy as np


ROWS = 4
COLS = 16
SEED = 19

# pto.tdump v1 header layout (16 x u32, little-endian).
MAGIC = 0x50544450  # "PTD0"
VERSION = 1
ELEM_SIZE = 4
ELEM_TYPE_CODE = 3  # f32
NDIM = 2
DATA_OFFSET = 64


def generate(output_dir: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    payload = rng.uniform(-1.0, 1.0, size=(ROWS, COLS)).astype(np.float32)

    payload.tofile(output_dir / "v1.bin")

    header = np.array(
        [MAGIC, VERSION, ELEM_SIZE, ELEM_TYPE_CODE, NDIM,
         ROWS, COLS, ROWS, COLS, DATA_OFFSET, 0, 0, 0, 0, 0, 0],
        dtype="<u4",
    )
    golden = np.concatenate(
        [header.view(np.uint8), payload.reshape(-1).view(np.uint8)]
    )
    golden.tofile(output_dir / "golden_v2.bin")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=".", type=Path)
    parser.add_argument("--seed", default=SEED, type=int)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    generate(args.output_dir, args.seed)
    print(f"[INFO] golden generated in {args.output_dir}")


if __name__ == "__main__":
    main()
