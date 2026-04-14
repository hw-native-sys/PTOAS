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


def make_fp32(generator, count: int, *, scale: float = 0.05) -> np.ndarray:
    return generator.uniform(-scale, scale, size=count).astype(np.float32)


def make_bf16(generator, count: int, *, scale: float = 0.05) -> np.ndarray:
    return float32_to_bf16(make_fp32(generator, count, scale=scale))


def main():
    rows = 16
    meta = load_case_meta()
    generator = rng()
    b0, ob = load_int32_assignments()[:2]

    buffers = {
        "v1": make_fp32(generator, meta.elem_counts["v1"], scale=0.05),
        "v2": make_bf16(generator, meta.elem_counts["v2"], scale=0.05),
        "v3": np.zeros(meta.elem_counts["v3"], dtype=meta.np_types["v3"]),
        "v4": make_bf16(generator, meta.elem_counts["v4"], scale=0.05),
    }

    output = np.zeros_like(buffers["v3"])

    for ob_ci in range(8):
        o0 = (ob * 8 + ob_ci) * 64
        acc = np.zeros((rows, 64), dtype=np.float32)
        for kb in range(40):
            k0 = kb * 128
            attn_chunk = load_strided_2d(buffers["v1"], offset=b0 * 5120 + k0, rows=rows, cols=128, row_stride=5120)
            attn_chunk = bf16_to_float32(float32_to_bf16(attn_chunk))
            w_chunk = bf16_to_float32(
                load_strided_2d(buffers["v4"], offset=k0 * 5120 + o0, rows=128, cols=64, row_stride=5120)
            )
            acc += attn_chunk @ w_chunk
        resid = bf16_to_float32(
            load_strided_2d(buffers["v2"], offset=b0 * 5120 + o0, rows=rows, cols=64, row_stride=5120)
        )
        output = store_strided_2d(output, acc + resid, offset=o0, row_stride=5120)

    write_buffers(meta, buffers)
    write_golden(meta, {"v3": output})


if __name__ == "__main__":
    main()
