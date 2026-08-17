#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""PTODSL ST coverage for A5 int32 column-expand soft division."""

from pathlib import Path
import sys

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import auto_main, golden_output_case
from ptodsl import pto


ROWS = 8
COLS = 64


@pto.jit(name="tcolexpanddiv_i32_soft_8x64", target="a5")
def _kernel(
    lhs_ptr: pto.ptr(pto.i32, "gm"),
    rhs_ptr: pto.ptr(pto.i32, "gm"),
    dst_ptr: pto.ptr(pto.i32, "gm"),
):
    lhs_view = pto.make_tensor_view(lhs_ptr, shape=[ROWS, COLS], strides=[COLS, 1])
    rhs_view = pto.make_tensor_view(rhs_ptr, shape=[1, COLS], strides=[COLS, 1])
    dst_view = pto.make_tensor_view(dst_ptr, shape=[ROWS, COLS], strides=[COLS, 1])
    lhs = pto.alloc_tile(shape=[ROWS, COLS], dtype=pto.i32)
    rhs = pto.alloc_tile(shape=[1, COLS], dtype=pto.i32)
    dst = pto.alloc_tile(shape=[ROWS, COLS], dtype=pto.i32)
    pto.tile.load(lhs_view, lhs)
    pto.tile.load(rhs_view, rhs)
    pto.tile.colexpanddiv(lhs, rhs, dst)
    pto.tile.store(dst, dst_view)


def _signed_i32_divide(lhs, rhs):
    lhs64 = lhs.astype(np.int64)
    rhs64 = rhs.astype(np.int64)
    quotient = np.abs(lhs64) // np.abs(rhs64)
    quotient = np.where((lhs64 < 0) ^ (rhs64 < 0), -quotient, quotient)
    return ((quotient & np.int64(0xFFFFFFFF)).astype(np.uint32).view(np.int32))


def _make_inputs():
    values = np.array(
        [0, 1, -1, 7, -7, 2**24 + 1, -(2**24 + 1), 2**31 - 1, -2**31,
         123456789, -123456789, 0x40000001, -0x40000001],
        dtype=np.int32,
    )
    lhs = np.resize(values, (ROWS, COLS))
    rhs_values = np.array([1, -1, 2, -2, 3, 7, -7, 0x40000001, -0x40000001], dtype=np.int32)
    rhs = np.resize(rhs_values, (1, COLS))
    return [lhs, rhs]


def _make_expected(lhs, rhs):
    return _signed_i32_divide(lhs, np.broadcast_to(rhs, lhs.shape))


CASES = [
    golden_output_case(
        "tcolexpanddiv_i32_soft_8x64",
        _kernel,
        inputs=_make_inputs,
        expected=_make_expected,
        rtol=0,
        atol=0,
    ),
]


auto_main(globals())
