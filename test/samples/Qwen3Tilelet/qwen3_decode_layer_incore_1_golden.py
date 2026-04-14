#!/usr/bin/python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import numpy as np

from validation_runtime import (
    bf16_to_float32,
    float32_to_bf16,
    load_case_meta,
    load_int32_assignments,
    load_strided_2d,
    rng,
    store_strided_2d,
    write_buffers,
    write_golden,
)


def make_fp32(generator, count: int, *, scale: float = 0.05, positive: bool = False) -> np.ndarray:
    if positive:
        return generator.uniform(0.5, 1.5, size=count).astype(np.float32)
    return generator.uniform(-scale, scale, size=count).astype(np.float32)


def make_bf16(generator, count: int, *, scale: float = 0.05) -> np.ndarray:
    return float32_to_bf16(make_fp32(generator, count, scale=scale))


def round_fp32_to_bf16_fp32(values: np.ndarray) -> np.ndarray:
    return bf16_to_float32(float32_to_bf16(values))


def main():
    rows = 16
    meta = load_case_meta()
    generator = rng()
    b0, ob = load_int32_assignments()[:2]

    buffers = {
        "v1": make_bf16(generator, meta.elem_counts["v1"], scale=0.05),
        "v2": make_fp32(generator, meta.elem_counts["v2"], positive=True),
        "v3": make_fp32(generator, meta.elem_counts["v3"], positive=True),
        "v4": np.zeros(meta.elem_counts["v4"], dtype=meta.np_types["v4"]),
        "v5": make_bf16(generator, meta.elem_counts["v5"], scale=0.05),
    }

    inv_rms = np.asarray(buffers["v3"], dtype=np.float32).reshape(rows, 1)
    output = np.zeros_like(buffers["v4"])

    for ob_ci in range(4):
        q0 = (ob * 4 + ob_ci) * 64
        acc = np.zeros((rows, 64), dtype=np.float32)
        for kb in range(40):
            k0 = kb * 128
            x_chunk = bf16_to_float32(
                load_strided_2d(buffers["v1"], offset=b0 * 5120 + k0, rows=rows, cols=128, row_stride=5120)
            )
            gamma = load_strided_2d(buffers["v2"], offset=k0, rows=1, cols=128, row_stride=5120).astype(np.float32)
            normed = round_fp32_to_bf16_fp32(x_chunk * inv_rms * gamma)
            w_chunk = bf16_to_float32(
                load_strided_2d(buffers["v5"], offset=k0 * 5120 + q0, rows=128, cols=64, row_stride=5120)
            )
            acc += normed @ w_chunk
        output = store_strided_2d(output, float32_to_bf16(acc), offset=b0 * 5120 + q0, row_stride=5120)

    write_buffers(meta, buffers)
    write_golden(meta, {"v4": output})


if __name__ == "__main__":
    main()
