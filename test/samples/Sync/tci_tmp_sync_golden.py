#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Independent integer golden for repeated explicit-tmp TCI/store pairs."""

from pathlib import Path
import sys

import numpy as np

for search_root in (Path(__file__).resolve().parent, Path(__file__).resolve().parents[1]):
    if (search_root / "validation_runtime.py").is_file():
        sys.path.insert(0, str(search_root))
        break

from validation_runtime import default_buffers, load_case_meta, write_buffers, write_golden


ROWS = 64
COLS = 32


def main():
    """Initialize output sentinels and check all rows in both directions."""
    meta = load_case_meta()
    if len(meta.outputs) != 2:
        raise ValueError(f"expected descending and ascending outputs, got {meta.outputs}")
    descending_name, ascending_name = meta.outputs
    buffers = default_buffers(meta)
    for name in meta.outputs:
        if meta.elem_counts[name] != ROWS * COLS:
            raise ValueError(f"expected {ROWS * COLS} elements for {name}, got {meta.elem_counts[name]}")
        buffers[name] = np.full(ROWS * COLS, -12345, dtype=np.int32)
    write_buffers(meta, buffers)
    indices = np.arange(COLS, dtype=np.int32)
    write_golden(meta, {
        descending_name: np.tile(31 - indices, ROWS),
        ascending_name: np.tile(10 + indices, ROWS),
    })


if __name__ == "__main__":
    main()
