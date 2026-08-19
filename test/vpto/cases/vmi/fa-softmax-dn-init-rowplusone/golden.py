#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
# This is the RowPlusOne variant of fa-softmax-dn-init. The only kernel
# difference is nz_buf uses CompactMode::RowPlusOne (UB virtualRow=129 with a
# +1 padding band that never leaves UB). The GM nz_out is still standard
# flattened NZ, so this golden is byte-for-byte identical to the plain-NZ
# case (same nd_to_nz). See kernel.pto header for details.

import argparse
from pathlib import Path

import numpy as np

try:
    from ml_dtypes import bfloat16 as bf16_dtype
except ImportError:  # numpy >= 2.1 ships bfloat16
    bf16_dtype = np.bfloat16

# Must match kernel.pto: scores [128,64] f32, x_exp [128,64] bf16,
# global_max/global_sum [1,64] f32, scale = 1/sqrt(64) = 0.125.
ROWS = 128
COLS = 64
SCALE = np.float32(1.0 / np.sqrt(np.float64(COLS)))  # 0.125


def nd_to_nz(data, rows, cols, c0=16, n0=16):
    """Convert ND (row-major) layout to NZ fractal layout.

    Mirrors pto-isa tests/npu/a5/src/st/testcase/tmov_nd2nz/gen_data.py::nd_to_nz
    (verified-correct golden). NZ layout: [c1, n1, n0, c0] where
    c1 = cols/c0, n1 = rows/n0. For bf16 (2B): c0 = CUBE_BLOCK_SIZE/(FRACTAL_NZ_ROW*sizeof) = 512/(16*2) = 16.
    """
    c1 = cols // c0
    n1 = rows // n0
    return data.reshape(n1, n0, c1, c0).transpose(2, 0, 1, 3).reshape(-1)


def generate(output_dir: Path) -> None:
    rng = np.random.RandomState(20260721)
    # Modest-range scores so exp doesn't overflow; softmax is column-wise
    # (axis over the 128 rows, per the 64 columns).
    scores = rng.uniform(-2.0, 2.0, size=(ROWS, COLS)).astype(np.float32)

    # global_max[j] = max_i scores[i,j], then * SCALE (matches tcolmax + tmuls)
    gmax = np.max(scores, axis=0, keepdims=True).astype(np.float32)
    gmax = (gmax * SCALE).astype(np.float32)

    # x = (scores - gmax) * SCALE  ; exp(x)
    shifted = (scores - gmax).astype(np.float32)
    scaled = (shifted * SCALE).astype(np.float32)
    ex = np.exp(scaled).astype(np.float32)

    # global_sum[j] = sum_i ex[i,j]
    gsum = np.sum(ex, axis=0, keepdims=True).astype(np.float32)

    # x_exp output: cast ex to bf16 (matches tcvt f32->bf16, CAST_ROUND)
    x_exp = ex.astype(bf16_dtype)

    # nz_out: NZ fractal rearrange of x_exp (matches pto.tmov ND->NZ).
    # Bit-exact rearrange of the bf16 x_exp bytes, compared against nd_to_nz.
    golden_nz = nd_to_nz(x_exp, ROWS, COLS, c0=16, n0=16)

    output_dir.mkdir(parents=True, exist_ok=True)
    scores.tofile(output_dir / "v1.bin")
    x_exp.tofile(output_dir / "golden_v2.bin")
    gmax.tofile(output_dir / "golden_v3.bin")
    gsum.tofile(output_dir / "golden_v4.bin")
    golden_nz.tofile(output_dir / "golden_v5.bin")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    generate(args.output_dir)


if __name__ == "__main__":
    main()
