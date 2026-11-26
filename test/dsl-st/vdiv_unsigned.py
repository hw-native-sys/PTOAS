#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Full-range unsigned A5 vdiv, including high-bit divisors and partial masks."""

import numpy as np

from common import auto_main, golden_output_case
from ptodsl import pto


def _build_kernel(dtype, cols, sparse, name):
    @pto.jit(name=name, kernel_kind="vector", target="a5", mode="explicit", insert_sync=False)
    def kernel(lhs_ptr: pto.ptr(dtype, "gm"), rhs_ptr: pto.ptr(dtype, "gm"), out_ptr: pto.ptr(dtype, "gm")):
        lhs_view = pto.make_tensor_view(lhs_ptr, shape=[1, cols], strides=[cols, 1])
        rhs_view = pto.make_tensor_view(rhs_ptr, shape=[1, cols], strides=[cols, 1])
        out_view = pto.make_tensor_view(out_ptr, shape=[1, cols], strides=[cols, 1])
        lhs = pto.alloc_tile(shape=[1, cols], dtype=dtype, addr=0,
                             valid_shape=[1, cols], blayout="RowMajor")
        rhs = pto.alloc_tile(shape=[1, cols], dtype=dtype, addr=2048,
                             valid_shape=[1, cols], blayout="RowMajor")
        dst = pto.alloc_tile(shape=[1, cols], dtype=dtype, addr=4096,
                             valid_shape=[1, cols], blayout="RowMajor")
        pto.tile.load(lhs_view, lhs)
        pto.tile.load(rhs_view, rhs)
        pto.set_flag("MTE2", "V", event_id=0)
        pto.wait_flag("MTE2", "V", event_id=0)
        with pto.tileop():
            full, _ = pto.make_mask(dtype, cols)
            x = pto.vlds(lhs[0, 0:])
            y = pto.vlds(rhs[0, 0:])
            mask = full
            if sparse:
                mask = pto.vcmps(pto.vands(x, dtype(1), full), dtype(0), full, pto.CmpMode.NE)
            quotient = pto.vdiv(x, y, mask)
            pto.vsts(quotient, dst[0, 0:], full)
        pto.set_flag("V", "MTE3", event_id=0)
        pto.wait_flag("V", "MTE3", event_id=0)
        pto.tile.store(dst, out_view)

    return kernel


def _inputs(bits, cols):
    dtype = np.uint16 if bits == 16 else np.uint32
    top = 1 << (bits - 1)
    x_edges = [0, 1, top - 1, top, top + 1, 2 * top - 2, 2 * top - 1]
    y_edges = [1, 2, 3, 7, top - 1, top, top + 1, 2 * top - 1]
    x, y = np.meshgrid(x_edges, y_edges)
    rng = np.random.default_rng(1607)
    lhs = rng.integers(0, 2 * top, size=cols, dtype=dtype)
    rhs = rng.integers(1, 2 * top, size=cols, dtype=dtype)
    lhs[:x.size] = x.ravel()
    rhs[:y.size] = y.ravel()
    if bits == 32:
        # The rounded reciprocal for this divisor can overshoot. Check
        # both sides of the first two quotient boundaries.
        divisor = 45_210_183
        boundary_lhs = [divisor - 2, divisor - 1, divisor, divisor + 1,
                        2 * divisor - 1, 2 * divisor, 2 * divisor + 1]
        end = x.size + len(boundary_lhs)
        lhs[x.size:end] = boundary_lhs
        rhs[x.size:end] = divisor
    return [lhs.reshape(1, cols), rhs.reshape(1, cols)]


def _case(bits, sparse):
    dtype = pto.ui16 if bits == 16 else pto.ui32
    cols = 2048 // bits
    name = f"vdiv_ui{bits}_{'sparse' if sparse else 'full'}"

    def expected(lhs, rhs):
        quotient = lhs // rhs
        return np.where((lhs & 1) != 0, quotient, 0) if sparse else quotient

    return golden_output_case(
        name, _build_kernel(dtype, cols, sparse, name),
        inputs=lambda: _inputs(bits, cols), expected=expected, rtol=0.0, atol=0.0,
    )


CASES = [_case(bits, sparse) for bits in (16, 32) for sparse in (False, True)]

auto_main(globals())
