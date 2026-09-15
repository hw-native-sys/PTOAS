#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Independent dequantization, per-head RMS normalization, and rotary reference."""

from pathlib import Path
import sys

import numpy as np

for root in (Path(__file__).resolve().parent, Path(__file__).resolve().parent.parent):
    if (root / "validation_runtime.py").is_file():
        sys.path.insert(0, str(root))
        break

from validation_runtime import (
    default_buffers, float32_to_bf16, load_case_meta, load_int32_assignments,
    load_scalar_assignments, write_buffers, write_golden,
)


def reference(buffers, token_count, block_index):
    output = np.zeros(buffers["v1"].size, dtype=np.float32)
    for start in range(0, token_count, 8):
        rows = np.arange(start, start + 8)
        rope_positions = rows[:, None] * 64 + np.arange(64)
        cos = buffers["v3"][rope_positions]
        sin = buffers["v4"][rope_positions]
        indices = buffers["v5"][rope_positions]
        for head in range(block_index * 4, block_index * 4 + 4):
            columns = head * 512 + np.arange(512)
            positions = rows[:, None] * 32768 + columns
            dequantized = buffers["v6"][positions].astype(np.float32) * buffers["v2"][rows, None]
            dequantized *= buffers["v7"][columns]
            variance = np.mean(dequantized * dequantized, axis=1, keepdims=True)
            normalized = dequantized / np.sqrt(variance + np.float32(1e-6))
            output[positions[:, :448]] = normalized[:, :448]
            rope = normalized[:, 448:]
            output[positions[:, 448:]] = rope * cos + np.take_along_axis(rope, indices, axis=1) * sin
    return float32_to_bf16(output)


def main():
    meta = load_case_meta()
    buffers = default_buffers(meta)
    # Preserve the original seed and inputs so the historical failure stays covered.
    generator = np.random.RandomState(19)
    for name in ("v2", "v3", "v4", "v7"):
        buffers[name] = generator.random_sample(meta.elem_counts[name]).astype(np.float32)
    for name in ("v5", "v6"):
        buffers[name] = (np.arange(meta.elem_counts[name]) % 8).astype(np.int32)
    scalars = load_scalar_assignments("int64_t")
    block_scalars = load_int32_assignments()
    if len(scalars) != 4 or len(block_scalars) != 2:
        raise ValueError("Unexpected qproj host scalar signature")
    expected = reference(buffers, scalars[1], block_scalars[0])
    write_buffers(meta, buffers)
    write_golden(meta, {"v1": expected})


if __name__ == "__main__":
    main()
