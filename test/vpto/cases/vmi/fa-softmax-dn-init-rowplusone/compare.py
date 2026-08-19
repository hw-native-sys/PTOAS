#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import sys

import numpy as np

try:
    from ml_dtypes import bfloat16 as bf16_dtype
except ImportError:  # numpy >= 2.1 ships bfloat16
    bf16_dtype = np.bfloat16


def _check(name: str, out_path: str, gold_path: str, atol: float, rtol: float) -> bool:
    out = np.fromfile(out_path, dtype=bf16_dtype if name == "x_exp" else np.float32)
    gold = np.fromfile(gold_path, dtype=bf16_dtype if name == "x_exp" else np.float32)
    if gold.shape != out.shape:
        print(f"[ERROR] {name}: shape {out.shape} != golden {gold.shape}")
        return False
    if not np.allclose(gold, out, atol=atol, rtol=rtol):
        diff = np.nonzero(~np.isclose(gold, out, atol=atol, rtol=rtol))[0]
        idx = int(diff[0]) if diff.size else -1
        print(f"[ERROR] {name} compare failed idx={idx} golden={gold[idx] if idx >= 0 else 'n/a'} output={out[idx] if idx >= 0 else 'n/a'}")
        return False
    print(f"[INFO] {name} compare passed")
    return True


def main() -> None:
    ok = True
    # x_exp (bf16): bf16 cast tolerance
    ok = _check("x_exp", "v2.bin", "golden_v2.bin", atol=2e-2, rtol=2e-2) and ok
    # global_max / global_sum (f32 reductions): tighter
    ok = _check("global_max", "v3.bin", "golden_v3.bin", atol=1e-4, rtol=1e-4) and ok
    ok = _check("global_sum", "v4.bin", "golden_v4.bin", atol=1e-3, rtol=1e-3) and ok
    # nz_out: NZ fractal rearrange of x_exp. Pure byte rearrange of bf16 x_exp,
    # so bit-exact (atol=0, rtol=0). This directly verifies the pto.tmov ND->NZ
    # rearrange against the pto-isa-verified nd_to_nz golden (mirrors
    # pto-isa tests/.../tmov_nd2nz, case_half_128x64_repeat1).
    ok = _check("nz", "v5.bin", "golden_v5.bin", atol=0.0, rtol=0.0) and ok
    if not ok:
        sys.exit(2)
    print("[INFO] compare passed")


if __name__ == "__main__":
    main()
