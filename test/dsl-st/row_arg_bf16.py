#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See the License in the root of the software repository for the full text of the License.
# See LICENSE in the root of the software repository for the full text of the License.

"""Exercise BF16 row indices across full vectors, ties, and a masked tail."""

import numpy as np

from common import auto_main, golden_output_case
from ptodsl import pto


ROWS = 8
COLS = 133
PHYSICAL_COLS = 144


def _kernel(name, reduce, index_dtype):
    @pto.jit(name=name, kernel_kind="vector", target="a5", backend="vpto")
    def kernel(src_ptr: pto.ptr(pto.bf16, "gm"), dst_ptr: pto.ptr(index_dtype, "gm")):
        src_view = pto.make_tensor_view(src_ptr, shape=[ROWS, COLS], strides=[COLS, 1])
        dst_view = pto.make_tensor_view(dst_ptr, shape=[ROWS, 1], strides=[1, 1])
        src = pto.alloc_tile(
            shape=[ROWS, PHYSICAL_COLS], dtype=pto.bf16,
            valid_shape=[ROWS, COLS], blayout="RowMajor",
        )
        dst = pto.alloc_tile(
            shape=[ROWS, 8], dtype=index_dtype,
            valid_shape=[ROWS, 1], blayout="RowMajor",
        )
        pto.tile.load(src_view, src)
        reduce(src, dst)
        pto.tile.store(dst, dst_view)

    return kernel


def _inputs():
    values = np.zeros((ROWS, COLS), dtype=np.float32)
    values[0, [63, 65]] = 7
    values[1, [64, 128]] = -7
    values[2, -1] = 6
    values[3, -1] = -6
    values[4] = -2
    values[4, 127] = -1
    values[5] = 2
    values[5, 127] = 1
    values[6, 64] = 4
    values[7, 64] = -4
    return [(values.view(np.uint32) >> 16).astype(np.uint16)]


def _case(op_name, reduce, golden_reduce, index_name, index_dtype, numpy_dtype):
    name = f"{op_name}_bf16_to_{index_name}"

    def expected(bits):
        values = (bits.astype(np.uint32) << 16).view(np.float32)
        return golden_reduce(values, axis=1).astype(numpy_dtype)

    return golden_output_case(
        name, _kernel(f"{name}_kernel", reduce, index_dtype),
        inputs=_inputs, expected=expected, rtol=0, atol=0,
    )


CASES = [
    _case(op_name, reduce, golden_reduce, index_name, index_dtype, numpy_dtype)
    for op_name, reduce, golden_reduce in (
        ("rowargmax", pto.tile.rowargmax, np.argmax),
        ("rowargmin", pto.tile.rowargmin, np.argmin),
    )
    for index_name, index_dtype, numpy_dtype in (
        ("i32", pto.i32, np.int32),
        ("ui32", pto.ui32, np.uint32),
    )
]

auto_main(globals())
