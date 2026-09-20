#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Independent dequantization/RMS/RoPE reference for the Pro 128-head ABI."""

import numpy as np

from validation_runtime import (
    default_buffers, float32_to_bf16, load_case_meta, load_int32_assignments,
    load_scalar_assignments, write_buffers, write_golden,
)


def main():
    meta = load_case_meta()
    buffers = default_buffers(meta)
    generator = np.random.RandomState(19)
    for name in ("v2", "v3", "v4", "v7"):
        buffers[name] = generator.random_sample(meta.elem_counts[name]).astype(np.float32)
    for name in ("v5", "v6"):
        buffers[name] = (np.arange(meta.elem_counts[name]) % 8).astype(np.int32)
    _, _, token_count = load_scalar_assignments("int64_t")
    block_index, _ = load_int32_assignments()
    if token_count <= 0 or token_count % 8:
        raise ValueError("Pro qproj requires at least one complete eight-token tile")
    expected = np.zeros(meta.elem_counts["v1"], dtype=np.float32)
    for start in range(0, token_count, 8):
        rows = np.arange(start, start + 8)
        rope_positions = rows[:, None] * 64 + np.arange(64)
        for head in range(block_index * 4, block_index * 4 + 4):
            columns = head * 512 + np.arange(512)
            positions = rows[:, None] * 65536 + columns
            values = buffers["v6"][positions].astype(np.float32) * buffers["v2"][rows, None]
            values *= buffers["v7"][columns]
            variance = np.mean(values * values, axis=1, keepdims=True)
            normalized = values / np.sqrt(variance + np.float32(1e-6))
            rope = normalized[:, 448:]
            swapped = np.take_along_axis(rope, buffers["v5"][rope_positions], axis=1)
            expected[positions[:, :448]] = normalized[:, :448]
            expected[positions[:, 448:]] = (
                rope * buffers["v3"][rope_positions] + swapped * buffers["v4"][rope_positions]
            )
    write_buffers(meta, buffers)
    write_golden(meta, {"v1": float32_to_bf16(expected)})


if __name__ == "__main__":
    main()
