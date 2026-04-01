#!/usr/bin/python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

import os
import sys
import numpy as np


def compare_bin(golden_path, output_path, dtype, eps):
    if not os.path.exists(output_path):
        print(f"[ERROR] Output missing: {output_path}")
        return False
    if not os.path.exists(golden_path):
        print(f"[ERROR] Golden missing: {golden_path}")
        return False
    dtype_np = np.dtype(dtype)
    golden = np.fromfile(golden_path, dtype=dtype_np)
    output = np.fromfile(output_path, dtype=dtype_np)
    if golden.shape != output.shape:
        print(f"[ERROR] Shape mismatch: {golden_path} {golden.shape} vs {output_path} {output.shape}")
        return False
    if not np.allclose(golden, output, atol=eps, rtol=eps, equal_nan=True):
        if golden.size:
            if np.issubdtype(dtype_np, np.floating):
                g = golden.astype(np.float64, copy=False)
                o = output.astype(np.float64, copy=False)
            elif np.issubdtype(dtype_np, np.integer) or np.issubdtype(dtype_np, np.unsignedinteger):
                g = golden.astype(np.int64, copy=False)
                o = output.astype(np.int64, copy=False)
            else:
                g = golden.astype(np.float64, copy=False)
                o = output.astype(np.float64, copy=False)
            abs_diff = np.abs(g - o)
            idx = int(np.argmax(abs_diff))
            diff = float(abs_diff[idx])
            print(
                f"[ERROR] Mismatch: {golden_path} vs {output_path}, max diff={diff} at idx={idx} "
                f"(golden={g[idx]}, out={o[idx]}, dtype={dtype_np})"
            )
        else:
            print(f"[ERROR] Mismatch: {golden_path} vs {output_path}, empty buffers, dtype={dtype_np}")
        return False
    return True


def compare_bin_prefix(golden_path, output_path, dtype, eps, count):
    if not os.path.exists(output_path):
        print(f"[ERROR] Output missing: {output_path}")
        return False
    if not os.path.exists(golden_path):
        print(f"[ERROR] Golden missing: {golden_path}")
        return False
    try:
        count = int(count)
    except Exception:
        print(f"[ERROR] Invalid prefix count: {count}")
        return False
    if count <= 0:
        print(f"[ERROR] Invalid prefix count: {count}")
        return False

    dtype_np = np.dtype(dtype)
    golden = np.fromfile(golden_path, dtype=dtype_np, count=count)
    output = np.fromfile(output_path, dtype=dtype_np, count=count)

    if golden.size != count or output.size != count:
        print(
            f"[ERROR] Prefix read too small: need={count} elems, "
            f"golden={golden.size}, out={output.size}"
        )
        return False

    if not np.allclose(golden, output, atol=eps, rtol=eps, equal_nan=True):
        if golden.size:
            if np.issubdtype(dtype_np, np.floating):
                g = golden.astype(np.float64, copy=False)
                o = output.astype(np.float64, copy=False)
            elif np.issubdtype(dtype_np, np.integer) or np.issubdtype(dtype_np, np.unsignedinteger):
                g = golden.astype(np.int64, copy=False)
                o = output.astype(np.int64, copy=False)
            else:
                g = golden.astype(np.float64, copy=False)
                o = output.astype(np.float64, copy=False)
            abs_diff = np.abs(g - o)
            idx = int(np.argmax(abs_diff))
            diff = float(abs_diff[idx])
            print(
                f"[ERROR] Mismatch (prefix): {golden_path} vs {output_path}, max diff={diff} at idx={idx} "
                f"(golden={g[idx]}, out={o[idx]}, dtype={dtype_np}, count={count})"
            )
        else:
            print(f"[ERROR] Mismatch (prefix): {golden_path} vs {output_path}, empty buffers, dtype={dtype_np}")
        return False
    return True


def compare_bin_at_indices(golden_path, output_path, dtype, eps, indices_path, indices_dtype):
    if not os.path.exists(output_path):
        print(f"[ERROR] Output missing: {output_path}")
        return False
    if not os.path.exists(golden_path):
        print(f"[ERROR] Golden missing: {golden_path}")
        return False
    if not os.path.exists(indices_path):
        print(f"[ERROR] Indices missing: {indices_path}")
        return False

    dtype_np = np.dtype(dtype)
    indices_dtype_np = np.dtype(indices_dtype)
    golden = np.fromfile(golden_path, dtype=dtype_np)
    output = np.fromfile(output_path, dtype=dtype_np)
    indices = np.fromfile(indices_path, dtype=indices_dtype_np)

    if golden.shape != output.shape:
        print(f"[ERROR] Shape mismatch: {golden_path} {golden.shape} vs {output_path} {output.shape}")
        return False

    if indices.size == 0:
        return True

    indices_i64 = indices.astype(np.int64, copy=False)
    if np.any(indices_i64 < 0) or np.any(indices_i64 >= golden.size):
        bad = int(np.nonzero((indices_i64 < 0) | (indices_i64 >= golden.size))[0][0])
        print(
            f"[ERROR] Indexed compare out of range at idx={bad} "
            f"(value={indices_i64[bad]}, limit={golden.size})"
        )
        return False

    golden_sel = golden[indices_i64]
    output_sel = output[indices_i64]
    if not np.allclose(golden_sel, output_sel, atol=eps, rtol=eps, equal_nan=True):
        if golden_sel.size:
            if np.issubdtype(dtype_np, np.floating):
                g = golden_sel.astype(np.float64, copy=False)
                o = output_sel.astype(np.float64, copy=False)
            elif np.issubdtype(dtype_np, np.integer) or np.issubdtype(dtype_np, np.unsignedinteger):
                g = golden_sel.astype(np.int64, copy=False)
                o = output_sel.astype(np.int64, copy=False)
            else:
                g = golden_sel.astype(np.float64, copy=False)
                o = output_sel.astype(np.float64, copy=False)
            abs_diff = np.abs(g - o)
            pos = int(np.argmax(abs_diff))
            diff = float(abs_diff[pos])
            src_idx = int(indices_i64[pos])
            print(
                f"[ERROR] Mismatch (indexed): {golden_path} vs {output_path}, max diff={diff} at output_idx={src_idx} "
                f"(golden={g[pos]}, out={o[pos]}, dtype={dtype_np})"
            )
        else:
            print(f"[ERROR] Mismatch (indexed): {golden_path} vs {output_path}, empty indexed selection, dtype={dtype_np}")
        return False
    return True


def compare_packed_pred_mask(golden_path, output_path, rows, cols):
    """
    Compare outputs of pto.tcmp / pto.tcmps.

    These ops produce a *packed predicate mask* and do not define every byte in
    the logical u8 tile buffer. In practice, only the first N bytes of each row
    are meaningful (packed as 64-bit chunks). Ignore the rest to avoid flaky
    compares caused by undefined bytes.
    """
    if not os.path.exists(output_path):
        print(f"[ERROR] Output missing: {output_path}")
        return False
    if not os.path.exists(golden_path):
        print(f"[ERROR] Golden missing: {golden_path}")
        return False
    try:
        rows = int(rows)
        cols = int(cols)
    except Exception:
        print(f"[ERROR] Invalid rows/cols for packed mask compare: rows={rows} cols={cols}")
        return False
    if rows <= 0 or cols <= 0:
        print(f"[ERROR] Invalid rows/cols for packed mask compare: rows={rows} cols={cols}")
        return False

    golden = np.fromfile(golden_path, dtype=np.uint8)
    output = np.fromfile(output_path, dtype=np.uint8)

    need = rows * cols
    if golden.size < need or output.size < need:
        print(
            f"[ERROR] Packed mask buffer too small: need={need} bytes, "
            f"golden={golden.size}, out={output.size}"
        )
        return False

    golden = golden[:need].reshape(rows, cols)
    output = output[:need].reshape(rows, cols)

    # Packed mask layout: 1 predicate bit per element, packed into 64-bit words
    # per row (so 8 bytes per 64 columns). For cols <= 64 we still use one word.
    row_bytes = ((cols + 63) // 64) * 8
    row_bytes = min(row_bytes, cols)

    golden_sel = golden[:, :row_bytes].reshape(-1)
    output_sel = output[:, :row_bytes].reshape(-1)

    if not np.array_equal(golden_sel, output_sel):
        diff = np.nonzero(golden_sel != output_sel)[0]
        idx = int(diff[0]) if diff.size else 0
        print(
            f"[ERROR] Mismatch (packed mask): {golden_path} vs {output_path}, first diff at idx={idx} "
            f"(golden={int(golden_sel[idx])}, out={int(output_sel[idx])}, rows={rows}, cols={cols}, row_bytes={row_bytes})"
        )
        return False
    return True


def main():
    strict = os.getenv("COMPARE_STRICT", "1") != "0"
@COMPARES@
    if not ok:
        if strict:
            print("[ERROR] compare failed")
            sys.exit(2)
        print("[WARN] compare failed (non-gating)")
        return
    print("[INFO] compare passed")


if __name__ == "__main__":
    main()
