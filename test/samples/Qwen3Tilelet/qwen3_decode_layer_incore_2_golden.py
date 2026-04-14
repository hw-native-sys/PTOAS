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
        "v5": np.zeros(meta.elem_counts["v5"], dtype=meta.np_types["v5"]),
        "v6": make_bf16(generator, meta.elem_counts["v6"], scale=0.05),
        "v7": make_bf16(generator, meta.elem_counts["v7"], scale=0.05),
    }

    inv_rms = np.asarray(buffers["v3"], dtype=np.float32).reshape(rows, 1)
    k_proj = np.zeros_like(buffers["v4"])
    v_proj = np.zeros_like(buffers["v5"])

    for ob_ci in range(8):
        kv0 = (ob * 8 + ob_ci) * 64
        k_acc = np.zeros((rows, 64), dtype=np.float32)
        v_acc = np.zeros((rows, 64), dtype=np.float32)
        for kb in range(40):
            k0 = kb * 128
            x_chunk = bf16_to_float32(
                load_strided_2d(buffers["v1"], offset=b0 * 5120 + k0, rows=rows, cols=128, row_stride=5120)
            )
            gamma = load_strided_2d(buffers["v2"], offset=k0, rows=1, cols=128, row_stride=5120).astype(np.float32)
            normed = round_fp32_to_bf16_fp32(x_chunk * inv_rms * gamma)
            wk_chunk = bf16_to_float32(
                load_strided_2d(buffers["v6"], offset=k0 * 1024 + kv0, rows=128, cols=64, row_stride=1024)
            )
            wv_chunk = bf16_to_float32(
                load_strided_2d(buffers["v7"], offset=k0 * 1024 + kv0, rows=128, cols=64, row_stride=1024)
            )
            k_acc += normed @ wk_chunk
            v_acc += normed @ wv_chunk
        k_proj = store_strided_2d(k_proj, float32_to_bf16(k_acc), offset=b0 * 1024 + kv0, row_stride=1024)
        v_proj = store_strided_2d(v_proj, float32_to_bf16(v_acc), offset=b0 * 1024 + kv0, row_stride=1024)

    write_buffers(meta, buffers)
    write_golden(meta, {"v4": k_proj, "v5": v_proj})


if __name__ == "__main__":
    main()
