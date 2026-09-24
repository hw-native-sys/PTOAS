# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Register-level residual correction following the unscaled DivPrecisionImpl.

The native vdiv_0ulp_ftz_true wrapper is described in PTOAS issue #1312.
Keep native special results and use fused vmula, not separately rounded mul/add.
"""

from ptodsl import pto


def div_vf32_soft(lhs, rhs, mask):
    """Preserve native special results and apply the reference f32 correction."""
    # Keep the caller's predicate for the whole correction pipeline.  Using an
    # all-lanes mask here makes inactive lanes participate in the candidate
    # search; that is observable for sparse predicates because the temporary
    # vector values can feed back through the masked select operations on A5.
    # The final select still gives inactive lanes the required deterministic
    # zero value.
    compute_mask = mask
    original = pto.vdiv(lhs, rhs, compute_mask)
    bits = pto.vbitcast(original, pto.ui32)
    tagged = pto.vor(bits, pto.vbr(pto.ui32(0x80000000)), compute_mask)
    special = pto.vcmps(tagged, pto.ui32(0xff800000), compute_mask, pto.CmpMode.GE)
    zero = pto.vcmps(original, pto.f32(0.0), compute_mask, pto.CmpMode.EQ)
    special = pto.por(special, zero, compute_mask)
    negative_lhs = pto.vmuls(lhs, pto.f32(-1.0), compute_mask)
    previous = pto.vbitcast(
        pto.vadds(pto.vbitcast(original, pto.si32), pto.si32(-1), compute_mask), pto.f32
    )
    following = pto.vbitcast(
        pto.vadds(pto.vbitcast(original, pto.si32), pto.si32(1), compute_mask), pto.f32
    )
    one = pto.vbr(pto.ui32(1))
    odd_z = pto.vand(bits, one, compute_mask)
    odd_z = pto.vcmp(odd_z, one, compute_mask, pto.CmpMode.EQ)
    # The quotient error is |rhs * q - lhs|.  Keep this fused so the
    # residual comparison is not affected by an intermediate rounding.
    residual = pto.vabs(pto.vmula(negative_lhs, rhs, original, compute_mask), compute_mask)
    residual_previous = pto.vabs(pto.vmula(negative_lhs, rhs, previous, compute_mask), compute_mask)
    residual_following = pto.vabs(pto.vmula(negative_lhs, rhs, following, compute_mask), compute_mask)
    previous_better = pto.vcmp(residual_previous, residual, compute_mask, pto.CmpMode.LT)
    previous_tie = pto.vcmp(residual, residual_previous, compute_mask, pto.CmpMode.EQ)
    keep_previous = pto.por(
        previous_better, pto.pand(previous_tie, odd_z, compute_mask), compute_mask
    )
    best = pto.vsel(previous, original, keep_previous)
    residual = pto.vsel(residual_previous, residual, keep_previous)
    following_better = pto.vcmp(residual_following, residual, compute_mask, pto.CmpMode.LT)
    following_tie = pto.vcmp(residual_following, residual, compute_mask, pto.CmpMode.EQ)
    best_bits = pto.vbitcast(best, pto.ui32)
    odd_best = pto.vcmp(pto.vand(best_bits, one, compute_mask), one, compute_mask, pto.CmpMode.EQ)
    improve = pto.por(
        following_better, pto.pand(following_tie, odd_best, compute_mask), compute_mask
    )
    best = pto.vsel(following, best, improve)
    result = pto.vsel(original, best, special)
    return pto.vsel(result, pto.vbr(pto.f32(0.0)), mask)
