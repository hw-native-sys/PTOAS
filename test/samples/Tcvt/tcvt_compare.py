#!/usr/bin/python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""
Output comparator for the tcvt_kernel_2d sample (f32 -> f16).

Compares the NPU-produced f16 output to the precomputed golden_<dst>.bin.
A tolerance of 1e-3 is used to accommodate the limited precision of float16
(~3 decimal digits).
"""

import numpy as np
from pathlib import Path
import sys

for search_root in (Path(__file__).resolve().parent, Path(__file__).resolve().parents[1]):
    if (search_root / 'validation_runtime.py').is_file():
        sys.path.insert(0, str(search_root))
        break

from validation_runtime import compare_outputs

if __name__ == '__main__':
    # f32 -> f16 via CAST_RINT (round-half-to-even) matches numpy's default
    # dtype cast exactly, so the golden is bitwise-identical to the NPU output.
    # Use atol=0 to catch any 1-ULP rounding regression.
    compare_outputs(np.float16, atol=0)
