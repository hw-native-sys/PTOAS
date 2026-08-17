#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""PTODSL ST coverage for both A5 int32 ``tdivs`` operand orders."""

from pathlib import Path
import sys

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import auto_main, golden_output_case
from ptodsl import pto


ROWS = 1
COLS = 64


@pto.jit(name="tdivs_i32_soft_tile_scalar_1x64", target="a5")
def _tile_scalar_kernel(
    src_ptr: pto.ptr(pto.i32, "gm"),
    scalar: pto.i32,
    dst_ptr: pto.ptr(pto.i32, "gm"),
):
    src_view = pto.make_tensor_view(src_ptr, shape=[ROWS, COLS], strides=[COLS, 1])
    dst_view = pto.make_tensor_view(dst_ptr, shape=[ROWS, COLS], strides=[COLS, 1])
    src = pto.alloc_tile(shape=[ROWS, COLS], dtype=pto.i32)
    dst = pto.alloc_tile(shape=[ROWS, COLS], dtype=pto.i32)
    pto.tile.load(src_view, src)
    pto.tile.divs(src, scalar, dst)
    pto.tile.store(dst, dst_view)


@pto.jit(name="tdivs_i32_soft_scalar_tile_1x64", target="a5")
def _scalar_tile_kernel(
    scalar: pto.i32,
    src_ptr: pto.ptr(pto.i32, "gm"),
    dst_ptr: pto.ptr(pto.i32, "gm"),
):
    src_view = pto.make_tensor_view(src_ptr, shape=[ROWS, COLS], strides=[COLS, 1])
    dst_view = pto.make_tensor_view(dst_ptr, shape=[ROWS, COLS], strides=[COLS, 1])
    src = pto.alloc_tile(shape=[ROWS, COLS], dtype=pto.i32)
    dst = pto.alloc_tile(shape=[ROWS, COLS], dtype=pto.i32)
    pto.tile.load(src_view, src)
    pto.tile.divs(scalar, src, dst)
    pto.tile.store(dst, dst_view)


def _signed_i32_divide(lhs, rhs):
    lhs64 = lhs.astype(np.int64)
    rhs64 = rhs.astype(np.int64)
    quotient = np.abs(lhs64) // np.abs(rhs64)
    quotient = np.where((lhs64 < 0) ^ (rhs64 < 0), -quotient, quotient)
    return ((quotient & np.int64(0xFFFFFFFF)).astype(np.uint32).view(np.int32))


def _values():
    return np.array(
        [1, -1, 2, -2, 3, 7, -7, 2**24 + 1, -(2**24 + 1), 2**31 - 1,
         -2**31, 123456789, -123456789, 0x40000001, -0x40000001],
        dtype=np.int32,
    )


def _make_tile_scalar_inputs():
    src = np.resize(_values(), (ROWS, COLS))
    scalar = np.array(-7, dtype=np.int32)
    return [src, scalar]


def _make_scalar_tile_inputs():
    src = np.resize(_values(), (ROWS, COLS))
    scalar = np.array(-2**31, dtype=np.int32)
    return [scalar, src]


CASES = [
    golden_output_case(
        "tdivs_i32_soft_tile_scalar_1x64",
        _tile_scalar_kernel,
        inputs=_make_tile_scalar_inputs,
        expected=_signed_i32_divide,
        rtol=0,
        atol=0,
    ),
    golden_output_case(
        "tdivs_i32_soft_scalar_tile_1x64",
        _scalar_tile_kernel,
        inputs=_make_scalar_tile_inputs,
        expected=_signed_i32_divide,
        rtol=0,
        atol=0,
    ),
]


auto_main(globals())
