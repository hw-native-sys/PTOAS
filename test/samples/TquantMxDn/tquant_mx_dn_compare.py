#!/usr/bin/python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from pathlib import Path
import sys
import numpy as np

for search_root in (Path(__file__).resolve().parent, Path(__file__).resolve().parents[1]):
    if (search_root / "validation_runtime.py").is_file():
        sys.path.insert(0, str(search_root))
        break

from validation_runtime import compare_file, finalize_compare, load_case_meta

GROUP_COUNT = 2 * 64


def compare_prefix(golden_path, output_path, dtype, count, atol):
    golden = np.fromfile(golden_path, dtype=dtype)
    output = np.fromfile(output_path, dtype=dtype)
    if golden.size < count or output.size < count:
        return False
    return np.allclose(golden[:count], output[:count], atol=atol, rtol=atol, equal_nan=True)


def main():
    meta = load_case_meta()
    names = meta.outputs
    dst, exp, max_name, scaling, exp_zz = (names + ["v2", "v3", "v4", "v5", "v6"])[:5]
    ok = compare_file(f"golden_{dst}.bin", f"{dst}.bin", np.int8, atol=0.0)
    ok = compare_prefix(f"golden_{exp}.bin", f"{exp}.bin", np.uint8, GROUP_COUNT, 0.0) and ok
    ok = compare_prefix(f"golden_{max_name}.bin", f"{max_name}.bin", np.float32, GROUP_COUNT, 1e-5) and ok
    ok = compare_prefix(f"golden_{scaling}.bin", f"{scaling}.bin", np.float32, GROUP_COUNT, 1e-5) and ok
    ok = compare_file(f"golden_{exp_zz}.bin", f"{exp_zz}.bin", np.uint8, atol=0.0) and ok
    finalize_compare(ok)


if __name__ == "__main__":
    main()
