#!/usr/bin/python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""
Golden generator for the tcvt_kernel_2d sample (f32 -> f16).

Reads the NPU test-harness main.cpp to discover buffer names and element
counts, then:
  - Writes deterministic f32 input to <src>.bin
  - Computes the expected f16 output via numpy (round-half-to-even, matching
    the default CAST_RINT rounding mode used by pto.tcvt)
  - Writes the golden to golden_<dst>.bin
"""

import numpy as np
from pathlib import Path
import sys

for search_root in (Path(__file__).resolve().parent, Path(__file__).resolve().parents[1]):
    if (search_root / 'validation_runtime.py').is_file():
        sys.path.insert(0, str(search_root))
        break

from validation_runtime import (
    default_buffers,
    float_values,
    load_case_meta,
    rng,
    single_output,
    write_buffers,
    write_golden,
)


def main():
    meta = load_case_meta()
    [src_name] = meta.inputs

    generator = rng()
    # Use the 'signed_small' range so all values survive f32->f16 without
    # overflow (f16 max ≈ 65504; values in [-1.5, 1.5] are well within range).
    src = float_values(generator, meta.elem_counts[src_name], style='signed_small')

    buffers = default_buffers(meta)
    buffers[src_name] = src
    write_buffers(meta, buffers)

    # f32 -> f16: numpy uses round-half-to-even by default, matching CAST_RINT.
    # The golden is bitwise-identical to the NPU output, so compare.py uses atol=0.
    golden_f16 = np.asarray(src, dtype=np.float16)
    write_golden(meta, {single_output(meta): golden_f16})


if __name__ == '__main__':
    main()
