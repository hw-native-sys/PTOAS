#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See the License in the root of the software repository for the full text of the License.

"""Issue #1599: A5 row max/min over 8-bit tiles through the PTODSL tile API.

Exercises the widened 8-bit row-extreme lowering on shapes that cover the
masked-tail, odd-tail split, full-vector-register, and multi-iteration paths.
min/max of widened 8-bit inputs is exact, so the golden compare is bitwise.
"""

import numpy as np

from common import auto_main, golden_output_case
from ptodsl import pto


def _row_extreme_kernel(name, maximum, dtype, rows, cols, cols_phys=None):
    reduce_entry = pto.tile.rowmax if maximum else pto.tile.rowmin
    # Row-major none-box tiles must keep 32-byte aligned physical rows, so
    # odd logical tails ride on the next aligned physical width.
    phys_cols = cols if cols_phys is None else cols_phys

    @pto.jit(name=name, kernel_kind="vector", target="a5", backend="vpto")
    def kernel(src_ptr: pto.ptr(dtype, "gm"), out_ptr: pto.ptr(dtype, "gm")):
        src_view = pto.make_tensor_view(src_ptr, shape=[rows, cols], strides=[cols, 1])
        dst_view = pto.make_tensor_view(out_ptr, shape=[rows, 1], strides=[1, 1])
        src = pto.alloc_tile(shape=[rows, phys_cols], dtype=dtype, valid_shape=[rows, cols], blayout="RowMajor")
        dst = pto.alloc_tile(shape=[rows, phys_cols], dtype=dtype, valid_shape=[rows, 1], blayout="RowMajor")
        pto.tile.load(src_view, src)
        reduce_entry(src, dst)
        pto.tile.store(dst, dst_view)

    return kernel


def _extreme_case(name, *, maximum, dtype, np_dtype, low, high, rows, cols, cols_phys=None):
    kernel = _row_extreme_kernel(f"{name}_kernel", maximum, dtype, rows, cols, cols_phys=cols_phys)

    def make_inputs(np_dtype=np_dtype, low=low, high=high, rows=rows, cols=cols):
        rng = np.random.default_rng(1599)
        return [rng.integers(low, high, size=(rows, cols), dtype=np_dtype)]

    def reduce_golden(src):
        return src.max(axis=1) if maximum else src.min(axis=1)

    return golden_output_case(
        name,
        kernel,
        inputs=make_inputs,
        expected=reduce_golden,
        rtol=0.0,
        atol=0.0,
    )


CASES = [
    # Tail smaller than one 256-lane i8 vector register (32 valid lanes).
    _extreme_case(
        "rowmax_i8_tail32", maximum=True, dtype=pto.i8, np_dtype=np.int8,
        low=-128, high=128, rows=8, cols=32),
    # Odd tail splits into ceil(33/2)=17 even and 16 odd widened lanes; the
    # physical width stays 32-byte aligned at 64.
    _extreme_case(
        "rowmin_i8_odd_tail33", maximum=False, dtype=pto.i8, np_dtype=np.int8,
        low=-128, high=128, rows=8, cols=33, cols_phys=64),
    # Exactly one full vector register per row.
    _extreme_case(
        "rowmax_ui8_full256", maximum=True, dtype=pto.ui8, np_dtype=np.uint8,
        low=0, high=256, rows=8, cols=256),
    # Two iterations: one full register plus a 44-lane tail on an aligned tile.
    _extreme_case(
        "rowmin_ui8_multi_iter300", maximum=False, dtype=pto.ui8,
        np_dtype=np.uint8, low=0, high=256, rows=8, cols=300, cols_phys=320),
    # Single logical column: the odd half reduces with an empty mask.
    _extreme_case(
        "rowmax_i8_single_col", maximum=True, dtype=pto.i8, np_dtype=np.int8,
        low=-128, high=128, rows=3, cols=1, cols_phys=32),
]

auto_main(globals())
