#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""
A5 signed i16 vector division simulator ST (issue 1241).

Exercises ``pto.vdiv`` on ``!pto.vreg<128xi16>`` with a full ``b16`` mask. On A5
the 16-bit integer HiVM VDIV intrinsic is not supported, so the backend
materializes the Software Library implementation (u32-domain f32 reciprocal
plus exact remainder correction) and the kernel must terminate with C-style
truncating signed division results.
"""

import numpy as np

from common import auto_main, golden_output_case
from ptodsl import pto


COLS = 128
# Nonzero denominators covering powers of two, small values, and non-power
# magnitudes.  |numerator| < 32768 avoids the INT16_MIN / -1 overflow pair.
DENOMINATORS = np.array(
    [14, -14, 3, -3, 133, -133, 1, -1, 128, -128, 7, -7, 1024, -1024, 4096, -4096],
    dtype=np.int16,
)


@pto.jit(
    name="vdiv_i16_kernel",
    kernel_kind="vector",
    target="a5",
    mode="explicit",
    insert_sync=False,
)
def vdiv_i16_kernel(
    lhs_ptr: pto.ptr(pto.i16, "gm"),
    rhs_ptr: pto.ptr(pto.i16, "gm"),
    out_ptr: pto.ptr(pto.i16, "gm"),
):
    total = COLS
    offsets = [0, 0, 0, 0, 0]
    lhs_view = pto.make_tensor_view(
        lhs_ptr,
        shape=[1, 1, 1, 1, COLS],
        strides=[total, total, total, total, 1],
    )
    rhs_view = pto.make_tensor_view(
        rhs_ptr,
        shape=[1, 1, 1, 1, COLS],
        strides=[total, total, total, total, 1],
    )
    out_view = pto.make_tensor_view(
        out_ptr,
        shape=[1, 1, 1, 1, COLS],
        strides=[total, total, total, total, 1],
    )
    lhs_part = pto.partition_view(
        lhs_view, offsets=offsets, sizes=[1, 1, 1, 1, COLS]
    )
    rhs_part = pto.partition_view(
        rhs_view, offsets=offsets, sizes=[1, 1, 1, 1, COLS]
    )
    out_part = pto.partition_view(
        out_view, offsets=offsets, sizes=[1, 1, 1, 1, COLS]
    )

    lhs_tile = pto.alloc_tile(
        shape=[1, COLS],
        dtype=pto.i16,
        addr=0,
        valid_shape=[1, COLS],
        blayout="RowMajor",
    )
    rhs_tile = pto.alloc_tile(
        shape=[1, COLS],
        dtype=pto.i16,
        addr=2048,
        valid_shape=[1, COLS],
        blayout="RowMajor",
    )
    dst_tile = pto.alloc_tile(
        shape=[1, COLS],
        dtype=pto.i16,
        addr=4096,
        valid_shape=[1, COLS],
        blayout="RowMajor",
    )

    pto.tile.load(lhs_part, lhs_tile)
    pto.tile.load(rhs_part, rhs_tile)
    pto.set_flag("MTE2", "V", event_id=0)
    pto.wait_flag("MTE2", "V", event_id=0)

    with pto.tileop():
        mask16 = pto.pset_b16(pto.MaskPattern.ALL)
        lhs_v = pto.vlds(lhs_tile[0, 0:])
        rhs_v = pto.vlds(rhs_tile[0, 0:])
        quotient = pto.vdiv(lhs_v, rhs_v, mask16)
        pto.vsts(quotient, dst_tile.as_ptr(), 0, mask16, dist="NORM_B16")

    pto.set_flag("V", "MTE3", event_id=0)
    pto.wait_flag("V", "MTE3", event_id=0)
    pto.tile.store(dst_tile, out_part)


def make_inputs():
    rng = np.random.RandomState(0x1241)
    x = rng.randint(-30000, 30000, size=(1, COLS)).astype(np.int16)
    y = np.resize(DENOMINATORS, (1, COLS)).astype(np.int16)
    return [x, y]


def make_expected(x, y):
    x32 = np.asarray(x, dtype=np.int32)
    y32 = np.asarray(y, dtype=np.int32)
    # C-style signed division truncating toward zero.  numpy ``//`` floors on
    # negatives, so compute the magnitude first (floor == trunc on positives)
    # and attach the sign afterwards.
    qabs = np.abs(x32) // np.abs(y32)
    q = np.where((x32 < 0) ^ (y32 < 0), -qabs, qabs)
    return q.astype(np.int16)


CASES = [
    golden_output_case(
        "vdiv_i16_soft_128_full_mask",
        vdiv_i16_kernel,
        inputs=make_inputs,
        expected=make_expected,
        rtol=0.0,
        atol=0.0,
    ),
]


auto_main(globals())