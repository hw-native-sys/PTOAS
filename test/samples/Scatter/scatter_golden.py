#!/usr/bin/python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
import numpy as np
from pathlib import Path
import sys

for search_root in (Path(__file__).resolve().parent, Path(__file__).resolve().parents[1]):
    if (search_root / 'validation_runtime.py').is_file():
        sys.path.insert(0, str(search_root))
        break

from validation_runtime import default_buffers, float_values, load_case_meta, rng, single_output, write_buffers, write_golden


def main():
    meta = load_case_meta()
    src_name, idx_name = meta.inputs
    generator = rng()
    src_dtype = meta.np_types[src_name]
    n_src = meta.elem_counts[src_name]
    n_idx = meta.elem_counts[idx_name]
    # Infer 2D shape: idx count == src count == rows * cols
    cols = 1
    for c in (64, 32, 16):
        if n_src % c == 0:
            cols = c
            break
    rows = n_src // cols
    src = float_values(generator, n_src, style='signed').astype(src_dtype, copy=False)
    src_2d = src.reshape(rows, cols)
    row_dest = (np.arange(rows, dtype=np.int64) * 17) % rows
    idx = np.repeat(row_dest[:, None], cols, axis=1)
    out = np.zeros((rows, cols), dtype=src_dtype)
    out[row_dest, :] = src_2d
    buffers = default_buffers(meta)
    buffers[src_name] = src
    buffers[idx_name] = idx.astype(meta.np_types[idx_name], copy=False).reshape(-1)[:n_idx]
    write_buffers(meta, buffers)
    write_golden(meta, {single_output(meta): out.reshape(-1)})


if __name__ == '__main__':
    main()
