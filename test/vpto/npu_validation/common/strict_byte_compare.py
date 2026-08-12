#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from pathlib import Path

import numpy as np


def compare_bytes(case_name: str) -> None:
    golden = np.fromfile(Path("golden_output.bin"), dtype=np.uint8)
    output = np.fromfile(Path("output.bin"), dtype=np.uint8)
    if golden.shape == output.shape and np.array_equal(golden, output):
        print(f"[INFO] {case_name} compare passed")
        return
    mismatch = np.flatnonzero(golden != output)
    idx = int(mismatch[0]) if mismatch.size else 0
    raise SystemExit(
        f"[ERROR] {case_name} mismatch: shape={output.shape}, idx={idx}, "
        f"golden={int(golden[idx]) if golden.size else 'n/a'}, "
        f"output={int(output[idx]) if output.size else 'n/a'}"
    )
