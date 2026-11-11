#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Check A5 row and column arg reductions with both index and value outputs."""

import numpy as np

from common import auto_main
from ptodsl import pto

ROWS = 8
COLS = 16


@pto.jit(name="arg_reduction_dual_output", kernel_kind="vector", target="a5", backend="vpto")
def arg_reduction_dual_output(
    src_ptr: pto.ptr(pto.f32, "gm"),
    row_max_idx_ptr: pto.ptr(pto.i32, "gm"),
    row_max_val_ptr: pto.ptr(pto.f32, "gm"),
    row_min_idx_ptr: pto.ptr(pto.i32, "gm"),
    row_min_val_ptr: pto.ptr(pto.f32, "gm"),
    col_max_idx_ptr: pto.ptr(pto.i32, "gm"),
    col_max_val_ptr: pto.ptr(pto.f32, "gm"),
    col_min_idx_ptr: pto.ptr(pto.i32, "gm"),
    col_min_val_ptr: pto.ptr(pto.f32, "gm"),
):
    src_view = pto.make_tensor_view(src_ptr, shape=[ROWS, COLS], strides=[COLS, 1])
    src = pto.alloc_tile(
        shape=[ROWS, COLS], dtype=pto.f32,
        valid_shape=[ROWS, COLS], blayout="RowMajor",
    )
    pto.tile.load(src_view, src)

    row_idx_max = pto.alloc_tile(
        shape=[ROWS, COLS], dtype=pto.i32,
        valid_shape=[ROWS, 1], blayout="RowMajor",
    )
    row_val_max = pto.alloc_tile(
        shape=[ROWS, COLS], dtype=pto.f32,
        valid_shape=[ROWS, 1], blayout="RowMajor",
    )
    row_idx_min = pto.alloc_tile(
        shape=[ROWS, COLS], dtype=pto.i32,
        valid_shape=[ROWS, 1], blayout="RowMajor",
    )
    row_val_min = pto.alloc_tile(
        shape=[ROWS, COLS], dtype=pto.f32,
        valid_shape=[ROWS, 1], blayout="RowMajor",
    )
    col_idx_max = pto.alloc_tile(
        shape=[ROWS, COLS], dtype=pto.i32,
        valid_shape=[1, COLS], blayout="RowMajor",
    )
    col_val_max = pto.alloc_tile(
        shape=[ROWS, COLS], dtype=pto.f32,
        valid_shape=[1, COLS], blayout="RowMajor",
    )
    col_idx_min = pto.alloc_tile(
        shape=[ROWS, COLS], dtype=pto.i32,
        valid_shape=[1, COLS], blayout="RowMajor",
    )
    col_val_min = pto.alloc_tile(
        shape=[ROWS, COLS], dtype=pto.f32,
        valid_shape=[1, COLS], blayout="RowMajor",
    )

    pto.tile.rowargmax(src, row_idx_max, row_val_max)
    pto.tile.rowargmin(src, row_idx_min, row_val_min)
    pto.tile.colargmax(src, col_idx_max, col_val_max)
    pto.tile.colargmin(src, col_idx_min, col_val_min)

    row_idx_max_view = pto.make_tensor_view(row_max_idx_ptr, shape=[ROWS, 1], strides=[1, 1])
    row_val_max_view = pto.make_tensor_view(row_max_val_ptr, shape=[ROWS, 1], strides=[1, 1])
    row_idx_min_view = pto.make_tensor_view(row_min_idx_ptr, shape=[ROWS, 1], strides=[1, 1])
    row_val_min_view = pto.make_tensor_view(row_min_val_ptr, shape=[ROWS, 1], strides=[1, 1])
    col_idx_max_view = pto.make_tensor_view(col_max_idx_ptr, shape=[1, COLS], strides=[COLS, 1])
    col_val_max_view = pto.make_tensor_view(col_max_val_ptr, shape=[1, COLS], strides=[COLS, 1])
    col_idx_min_view = pto.make_tensor_view(col_min_idx_ptr, shape=[1, COLS], strides=[COLS, 1])
    col_val_min_view = pto.make_tensor_view(col_min_val_ptr, shape=[1, COLS], strides=[COLS, 1])
    pto.tile.store(row_idx_max, row_idx_max_view)
    pto.tile.store(row_val_max, row_val_max_view)
    pto.tile.store(row_idx_min, row_idx_min_view)
    pto.tile.store(row_val_min, row_val_min_view)
    pto.tile.store(col_idx_max, col_idx_max_view)
    pto.tile.store(col_val_max, col_val_max_view)
    pto.tile.store(col_idx_min, col_idx_min_view)
    pto.tile.store(col_val_min, col_val_min_view)


def make_case():
    src = np.arange(ROWS * COLS, dtype=np.float32).reshape(ROWS, COLS)
    expected = (
        np.argmax(src, axis=1).astype(np.int32).reshape(ROWS, 1),
        np.max(src, axis=1).reshape(ROWS, 1),
        np.argmin(src, axis=1).astype(np.int32).reshape(ROWS, 1),
        np.min(src, axis=1).reshape(ROWS, 1),
        np.argmax(src, axis=0).astype(np.int32).reshape(1, COLS),
        np.max(src, axis=0).reshape(1, COLS),
        np.argmin(src, axis=0).astype(np.int32).reshape(1, COLS),
        np.min(src, axis=0).reshape(1, COLS),
    )
    outputs = [np.zeros_like(value) for value in expected]
    return [src, *outputs], expected


def check_case(device_inputs, expected):
    for output, golden in zip(device_inputs[1:], expected):
        np.testing.assert_array_equal(output.cpu().numpy(), golden)


CASES = [
    {
        "name": "arg_reduction_dual_output",
        "kernel": arg_reduction_dual_output,
        "make_case": make_case,
        "check": check_case,
    },
]

auto_main(globals())
