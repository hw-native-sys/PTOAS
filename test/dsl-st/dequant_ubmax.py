#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You should not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See the License in the root directory of this software repository for the full text of the License.

"""TDequant UB-pressure end-to-end ST (issue #1689).

i16 src 992x32 and i8 src 1128x32 (with f32 dst and per-row scale/offset) pack
the A5 vec space tightly enough that the memory planner used to let dst reuse
the address of the src tile it was still reading: src is i8/i16 and dst is f32,
so a dst row writes 128 bytes over 64/32-byte src rows and corrupts every row
from row 1 on. The planner now forbids the alias; these cases lock the golden
results at the pressure shapes (issue #1689, dequant_i16_ubmax992x32 /
dequant_i8_ubmax1133x32).

The i8 case uses 1128 rows, a multiple of 8, so all four tile sizes are
multiples of 256 bytes. Together they occupy 252672 bytes
(36096 * 3 + 144384), leaving 1280 bytes of A5 vec capacity while still
reproducing the original unsafe reuse.
"""

import numpy as np

from common import auto_main, golden_output_case
from ptodsl import pto


COLS = 32
SEED = 0x1689


def _make_dequant_kernel(name, src_dtype, rows):
    @pto.jit(name=name, target="a5")
    def _kernel(
        src_ptr: pto.ptr(src_dtype, "gm"),
        scale_ptr: pto.ptr(pto.f32, "gm"),
        off_ptr: pto.ptr(pto.f32, "gm"),
        dst_ptr: pto.ptr(pto.f32, "gm"),
    ):
        src_view = pto.make_tensor_view(src_ptr, shape=[rows, COLS], strides=[COLS, 1])
        scale_view = pto.make_tensor_view(scale_ptr, shape=[rows, 1], strides=[1, 1])
        off_view = pto.make_tensor_view(off_ptr, shape=[rows, 1], strides=[1, 1])
        dst_view = pto.make_tensor_view(dst_ptr, shape=[rows, COLS], strides=[COLS, 1])

        src_tile = pto.alloc_tile(shape=[rows, COLS], dtype=src_dtype)
        # 32-byte physical rows with a one-lane valid column keep every per-row
        # parameter load 32B-aligned while preserving the issue #1689 pressure.
        scale_tile = pto.alloc_tile(shape=[rows, 8], valid_shape=[rows, 1], dtype=pto.f32)
        off_tile = pto.alloc_tile(shape=[rows, 8], valid_shape=[rows, 1], dtype=pto.f32)
        dst_tile = pto.alloc_tile(shape=[rows, COLS], dtype=pto.f32)

        pto.tile.load(src_view, src_tile)
        pto.tile.load(scale_view, scale_tile)
        pto.tile.load(off_view, off_tile)
        pto.tile.dequant(src_tile, scale_tile, off_tile, dst_tile)
        pto.tile.store(dst_tile, dst_view)

    return _kernel


def _make_case(case_name, src_np_dtype, rows):
    kernel = _make_dequant_kernel(
        "tdequant_" + case_name,
        pto.i16 if src_np_dtype is np.int16 else pto.i8,
        rows,
    )

    def _make_inputs():
        rng = np.random.RandomState(SEED)
        src = rng.uniform(-100, 100, size=(rows, COLS)).astype(src_np_dtype)
        scale = rng.uniform(0.5, 2.0, size=(rows, 1)).astype(np.float32)
        offset = rng.uniform(-10, 10, size=(rows, 1)).astype(np.float32)
        return [src, scale, offset]

    def _make_expected(src, scale, offset):
        value = src.astype(np.float32)
        return ((value - offset) * scale).astype(np.float32)

    return golden_output_case(
        case_name,
        kernel,
        inputs=_make_inputs,
        expected=_make_expected,
        rtol=1e-3,
        atol=1e-3,
    )


CASES = [
    _make_case("i16_ubmax_992x32", np.int16, 992),
    _make_case("i8_ubmax_1128x32", np.int8, 1128),
]


auto_main(globals())
