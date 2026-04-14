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
    o0 = load_int32_assignments()[0]

    buffers = {
        "v1": make_fp32(generator, meta.elem_counts["v1"], scale=0.01),
        "v2": make_bf16(generator, meta.elem_counts["v2"], scale=0.01),
        "v3": make_fp32(generator, meta.elem_counts["v3"], scale=0.01),
        "v4": make_bf16(generator, meta.elem_counts["v4"], scale=0.01),
        "v5": make_bf16(generator, meta.elem_counts["v5"], scale=0.01),
        "v6": np.zeros(meta.elem_counts["v6"], dtype=meta.np_types["v6"]),
    }

    gate_acc = np.zeros((rows, 64), dtype=np.float32)
    up_acc = np.zeros((rows, 64), dtype=np.float32)

    for kb in range(40):
        k0 = kb * 128
        post_chunk = bf16_to_float32(
            load_strided_2d(buffers["v2"], offset=k0, rows=rows, cols=128, row_stride=5120)
        )
        w_gate = bf16_to_float32(
            load_strided_2d(buffers["v4"], offset=k0 * 25600 + o0, rows=128, cols=64, row_stride=25600)
        )
        w_up = bf16_to_float32(
            load_strided_2d(buffers["v5"], offset=k0 * 25600 + o0, rows=128, cols=64, row_stride=25600)
        )
        gate_acc += post_chunk @ w_gate
        up_acc += post_chunk @ w_up

    sigmoid = np.reciprocal(1.0 + np.exp(-gate_acc))
    mlp_chunk = gate_acc * sigmoid * up_acc
    output = float32_to_bf16(mlp_chunk)

    write_buffers(meta, buffers)
    write_golden(meta, {"v6": output})


if __name__ == "__main__":
    main()
