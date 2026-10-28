#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""pto.vcvt with a partial predicate on a 64-lane LOGICAL widening.

A LOGICAL 64-lane f16 -> f32 widening is carried by a 128-lane physical f16
vector whose lanes are lane-strided, so the masked conversion has to convert
both halves of the carrier, not just its even lanes.  The kernel loads 64 f16
values, builds a 32-of-64 predicate with pto.vmi.create_mask, converts the
whole logical vector to f32 under that predicate, then stores and copies out
all 64 f32 lanes.  The inactive half must be exactly zero on every one of the
64 *logical* lanes; a spread that only keeps the even carrier half active
leaves every odd logical lane zero even where the predicate is set.

64 lanes is the smallest shape where the logical vector is narrower than the
physical carrier the lane-strided vlds materializes, so it pins the odd-half
conversion that the 128-lane shape (which widens through the same-width
carrier pair) does not exercise.  The source lanes are all nonzero and the
golden compares with atol=0/rtol=0, so a dropped odd logical lane fails
instead of being absorbed by a tolerance.
"""

import numpy as np

from common import auto_main, golden_output_case
from ptodsl import pto

LANES = 64
ACTIVE = LANES // 2
SRC_UB_BYTES = LANES * 2
DST_UB = 1024
DST_UB_BYTES = LANES * 4


@pto.jit(
    name="masked_vcvt_zero_inactive_l64_kernel",
    kernel_kind="vector",
    target="a5",
    backend="vpto",
    mode="explicit",
)
def masked_vcvt_zero_inactive_l64_kernel(
    src_gm: pto.ptr(pto.f16, 'gm'),
    out_gm: pto.ptr(pto.f32, 'gm'),
):
    src_ub = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.f16, 'ub'))
    dst_ub = pto.castptr(pto.const(DST_UB, dtype=pto.i64), pto.ptr(pto.f32, 'ub'))
    pto.mte_gm_ub(
        src_gm, src_ub, 0, SRC_UB_BYTES, nburst=(1, SRC_UB_BYTES, SRC_UB_BYTES),
    )
    pto.set_flag('MTE2', 'V', event_id=0)
    pto.wait_flag('MTE2', 'V', event_id=0)
    with pto.vecscope():
        mask = pto.vmi.create_mask(ACTIVE, size=LANES)
        src = pto.vmi.vload(src_ub, 0, size=LANES)
        conv = pto.vmi.vcvt(src, pto.f32, mask)
        # Deliberately unmasked: every logical lane of the converted vector must
        # leave with its post-conversion value, so the odd logical lanes expose a
        # missing conversion instead of hiding behind the store predicate.
        pto.vmi.vstore(conv, dst_ub, 0)
    pto.set_flag('V', 'MTE3', event_id=0)
    pto.wait_flag('V', 'MTE3', event_id=0)
    pto.mte_ub_gm(
        dst_ub, out_gm, DST_UB_BYTES, nburst=(1, DST_UB_BYTES, DST_UB_BYTES), l2_cache='naci',
    )


def make_inputs():
    # Active lanes are negative, inactive lanes far positive: a merge or a
    # pass-through result cannot land accidentally close to the golden zero.
    active = np.linspace(-17.0, -1.5, ACTIVE, dtype=np.float16)
    inactive = np.linspace(1000.0, 2000.0, LANES - ACTIVE, dtype=np.float16)
    return [np.concatenate([active, inactive])]


def make_expected(src):
    # f16 -> f32 widening is exact, and the predicate clears the upper half.
    expected = np.zeros(LANES, dtype=np.float32)
    expected[:ACTIVE] = src[:ACTIVE].astype(np.float32)
    return expected


CASES = [
    golden_output_case(
        "masked_vcvt_zero_inactive_l64_a32",
        masked_vcvt_zero_inactive_l64_kernel,
        inputs=make_inputs,
        expected=make_expected,
        rtol=0.0,
        atol=0.0,
    ),
]


auto_main(globals())
