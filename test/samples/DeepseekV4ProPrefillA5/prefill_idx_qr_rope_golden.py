#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Independent interleaved rotary reference for one token's 64 query heads."""

import numpy as np

from validation_runtime import (
    default_buffers, float32_to_bf16, load_case_meta, load_int32_assignments,
    write_buffers, write_golden,
)


def main():
    meta = load_case_meta()
    buffers = default_buffers(meta)
    generator = np.random.RandomState(19)
    for name in ("v1", "v2", "v4"):
        buffers[name] = generator.random_sample(meta.elem_counts[name]).astype(np.float32)
    block_index, _ = load_int32_assignments()
    row_start = block_index * 64
    rows = np.arange(row_start, row_start + 64)
    query = buffers["v4"][rows[:, None] * 128 + np.arange(64, 128)]
    cos = np.repeat(buffers["v1"][block_index * 32:(block_index + 1) * 32], 2)
    sin = np.repeat(buffers["v2"][block_index * 32:(block_index + 1) * 32], 2)
    swapped = query[:, np.arange(64) ^ 1]
    signs = np.tile(np.array([-1, 1], dtype=np.float32), 32)
    rotated = query * cos + swapped * (sin * signs)
    expected = np.zeros(meta.elem_counts["v3"], dtype=np.uint16)
    expected[row_start * 64:(row_start + 64) * 64] = float32_to_bf16(rotated).reshape(-1)
    write_buffers(meta, buffers)
    write_golden(meta, {"v3": expected})


if __name__ == "__main__":
    main()
