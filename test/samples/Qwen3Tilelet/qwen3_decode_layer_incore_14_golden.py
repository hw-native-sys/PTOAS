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


def make_fp32(generator, count: int, *, scale: float = 0.01) -> np.ndarray:
    return generator.uniform(-scale, scale, size=count).astype(np.float32)


def make_bf16(generator, count: int, *, scale: float = 0.01) -> np.ndarray:
    return float32_to_bf16(make_fp32(generator, count, scale=scale))


def main():
    rows = 16
    meta = load_case_meta()
    generator = rng()
    dob, o0 = load_int32_assignments()[:2]

    buffers = {
        "v1": make_fp32(generator, meta.elem_counts["v1"], scale=0.01),
        "v2": make_bf16(generator, meta.elem_counts["v2"], scale=0.01),
        "v3": make_bf16(generator, meta.elem_counts["v3"], scale=0.01),
    }

    output = np.array(buffers["v1"], copy=True)
    mlp_chunk = bf16_to_float32(load_strided_2d(buffers["v2"], offset=0, rows=rows, cols=64, row_stride=64))

    for dob_ci in range(4):
        d0 = (dob * 4 + dob_ci) * 128
        down_prev = load_strided_2d(output, offset=d0, rows=rows, cols=128, row_stride=5120).astype(np.float32)
        w_down = bf16_to_float32(
            load_strided_2d(buffers["v3"], offset=o0 * 5120 + d0, rows=64, cols=128, row_stride=5120)
        )
        output = store_strided_2d(output, down_prev + mlp_chunk @ w_down, offset=d0, row_stride=5120)

    write_buffers(meta, buffers)
    write_golden(meta, {"v1": output})


if __name__ == "__main__":
    main()
