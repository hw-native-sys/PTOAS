#!/usr/bin/python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Compare the axis1 MXFP8 plus ND-to-ZZ sample outputs.

The tquant.mx kernel has four outputs with different dtypes (fp8 dst, e8m0 exp,
f32 max, f32 scaling), so we cannot use the single-dtype compare_outputs helper.
Instead we compare each output with its own dtype and tolerance.
"""

from pathlib import Path
import sys

import numpy as np

for search_root in (Path(__file__).resolve().parent, Path(__file__).resolve().parents[1]):
    if (search_root / "validation_runtime.py").is_file():
        sys.path.insert(0, str(search_root))
        break

from validation_runtime import compare_file, finalize_compare, load_case_meta


M = 16
K = 64
GROUP_SIZE = 32
GROUP_COUNT = (M * K) // GROUP_SIZE


def compare_file_prefix(golden_path, output_path, dtype, logical_count, atol):
    if not Path(output_path).exists():
        print(f"[ERROR] Output missing: {output_path}")
        return False
    if not Path(golden_path).exists():
        print(f"[ERROR] Golden missing: {golden_path}")
        return False
    dtype = np.dtype(dtype)
    golden = np.fromfile(golden_path, dtype=dtype)
    output = np.fromfile(output_path, dtype=dtype)
    if golden.shape != output.shape:
        print(f"[ERROR] Shape mismatch: {golden_path} {golden.shape} vs {output_path} {output.shape}")
        return False
    if golden.size < logical_count:
        print(f"[ERROR] Buffer too small for logical compare: need {logical_count}, got {golden.size}")
        return False
    golden = golden[:logical_count]
    output = output[:logical_count]
    if np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, np.unsignedinteger):
        if atol == 0.0:
            ok = np.array_equal(golden, output)
        else:
            ok = np.allclose(golden, output, atol=atol, rtol=atol)
    else:
        ok = np.allclose(golden, output, atol=atol, rtol=atol, equal_nan=True)
    if not ok:
        diff = np.abs(golden.astype(np.float64) - output.astype(np.float64))
        index = int(np.argmax(diff))
        print(
            f"[ERROR] Logical mismatch: {golden_path} vs {output_path}, max diff={float(diff[index])} "
            f"at idx={index} (golden={golden[index]}, out={output[index]}, dtype={golden.dtype})"
        )
        return False
    return True


def main():
    meta = load_case_meta()
    output_names = meta.outputs

    # Outputs are ordered by tstore appearance: dst, exp, max, scaling, exp_zz.
    dst_name = output_names[0] if len(output_names) > 0 else "v2"
    exp_name = output_names[1] if len(output_names) > 1 else "v3"
    max_name = output_names[2] if len(output_names) > 2 else "v4"
    scaling_name = output_names[3] if len(output_names) > 3 else "v5"
    exp_zz_name = output_names[4] if len(output_names) > 4 else "v6"

    ok = True
    # dst: fp8 e4m3fn packed as int8 — exact byte match.
    ok = compare_file(f"golden_{dst_name}.bin", f"{dst_name}.bin", np.int8, atol=0.0) and ok
    # exp/max/scaling contain the 32 canonical [M, N/32] group values.
    ok = compare_file_prefix(f"golden_{exp_name}.bin", f"{exp_name}.bin", np.uint8, GROUP_COUNT, atol=0.0) and ok
    ok = compare_file_prefix(f"golden_{max_name}.bin", f"{max_name}.bin", np.float32, GROUP_COUNT, atol=1e-5) and ok
    ok = compare_file_prefix(
        f"golden_{scaling_name}.bin", f"{scaling_name}.bin", np.float32, GROUP_COUNT, atol=1e-5
    ) and ok
    ok = compare_file(
        f"golden_{exp_zz_name}.bin", f"{exp_zz_name}.bin", np.uint8, atol=0.0
    ) and ok

    finalize_compare(ok)


if __name__ == "__main__":
    main()
