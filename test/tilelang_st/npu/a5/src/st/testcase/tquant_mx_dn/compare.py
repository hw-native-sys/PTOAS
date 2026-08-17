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
from ml_dtypes import bfloat16

from cases import CASES
from st_common import result_cmp, style_fail, style_pass


def read_float(path, dtype):
    data = np.fromfile(path, dtype=dtype)
    return data.astype(np.float32) if dtype == bfloat16 else data


def main():
    selected = sys.argv[1] if len(sys.argv) > 1 else None
    passed = True
    for case in CASES:
        if selected and selected != case["name"]:
            continue
        root = case["name"]
        src_dtype = bfloat16 if case["src_type"] == "bf16" else np.float32
        checks = [
            ("dst", np.uint8, 0.0),
            ("exp", np.uint8, 0.0),
            ("max", src_dtype, 1e-5),
            ("scaling", src_dtype, 1e-5),
            ("exp_zz", np.uint8, 0.0),
        ]
        case_ok = True
        for name, dtype, atol in checks:
            golden = read_float(os.path.join(root, f"golden_{name}.bin"), dtype)
            output = read_float(os.path.join(root, f"output_{name}.bin"), dtype)
            if not np.allclose(golden, output, atol=atol, rtol=atol, equal_nan=True):
                case_ok = False
                print(style_fail(f"[ERROR] {case['name']} {name} mismatch"))
        if case_ok:
            print(style_pass(f"[INFO] {case['name']}: compare passed"))
        passed &= case_ok
    if not passed:
        raise SystemExit(2)
    print(style_pass("[INFO] all MX TQUANT cases passed"))


if __name__ == "__main__":
    main()
