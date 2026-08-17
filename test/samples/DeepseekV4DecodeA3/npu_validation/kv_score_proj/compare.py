#!/usr/bin/python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import os
import sys

import numpy as np

from validation_runtime import load_case_meta


VALID_ROWS = 8
COLS = 1024
VALID_ELEMS = VALID_ROWS * COLS
EPS = 0.001


def _compare_valid_prefix(golden_path: str, output_path: str) -> bool:
    if not os.path.exists(output_path):
        print(f"[ERROR] Output missing: {output_path}")
        return False
    if not os.path.exists(golden_path):
        print(f"[ERROR] Golden missing: {golden_path}")
        return False

    golden = np.fromfile(golden_path, dtype=np.float32)
    output = np.fromfile(output_path, dtype=np.float32)
    if golden.size < VALID_ELEMS or output.size < VALID_ELEMS:
        print(
            f"[ERROR] Shape mismatch: {golden_path} {golden.shape} vs {output_path} {output.shape}, "
            f"need at least {VALID_ELEMS} elements"
        )
        return False

    golden_valid = golden[:VALID_ELEMS]
    output_valid = output[:VALID_ELEMS]
    if np.allclose(golden_valid, output_valid, atol=EPS, rtol=EPS, equal_nan=True):
        return True

    diff = np.abs(golden_valid.astype(np.float64) - output_valid.astype(np.float64))
    close = np.isclose(golden_valid, output_valid, atol=EPS, rtol=EPS, equal_nan=True)
    bad = np.nonzero(~close)[0]
    diff_for_report = np.where(np.isfinite(diff), diff, np.inf)
    idx = int(bad[np.argmax(diff_for_report[bad])]) if bad.size else 0
    row, col = divmod(idx, COLS)
    print(
        f"[ERROR] Mismatch(valid rows): {golden_path} vs {output_path}, "
        f"max diff={float(diff_for_report[idx])} at idx={idx}, row={row}, col={col} "
        f"(golden={float(golden_valid[idx])}, out={float(output_valid[idx])}, dtype=float32)"
    )
    return False


def main():
    meta = load_case_meta()
    ok = True
    for name in meta.outputs:
        if meta.np_types[name] != np.dtype(np.float32):
            print(f"[ERROR] Unsupported output dtype for {name}: {meta.np_types[name]}")
            ok = False
            continue
        ok = _compare_valid_prefix(f"golden_{name}.bin", f"{name}.bin") and ok

    if not ok:
        if os.getenv("COMPARE_STRICT", "1") != "0":
            print("[ERROR] compare failed")
            sys.exit(2)
        print("[WARN] compare failed (non-gating)")
        return False
    print("[INFO] compare passed")
    return True


if __name__ == "__main__":
    main()
