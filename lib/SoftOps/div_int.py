# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""A5 signed integer division soft implementation."""

from ptodsl import pto


def div_i32_soft(vec, scalar_vec, mask):
    """Compute signed i32 division without relying on integer ``vdiv``.

    The quotient magnitude is estimated from an f32 reciprocal, widened to
    i64, and corrected against the exact remainder. The result is exact for
    every nonzero int32 divisor; division by zero is intentionally undefined.
    """
    result_dtype = pto.si32 if str(vec.type).endswith("xsi32>") else pto.i32
    zero = result_dtype(0)
    neg_one = result_dtype(-1)
    fp32_one = pto.f32(1.0)
    false_mask = pto.pset_b32(pto.PAT.ALLF)

    zero_mask = pto.vcmps(scalar_vec, zero, mask, pto.CmpMode.EQ)
    active_mask = pto.pnot(zero_mask, mask)

    abs_x = pto.vbitcast(pto.vabs(vec, active_mask), pto.ui32)
    abs_y = pto.vbitcast(pto.vabs(scalar_vec, active_mask), pto.ui32)
    x_xor_y = pto.vxor(vec, scalar_vec, active_mask)
    positive = pto.vcmps(x_xor_y, zero, active_mask, pto.CmpMode.GE)

    y_float = pto.vcvt(
        pto.vbitcast(abs_y, pto.i32), pto.f32, active_mask,
        rnd=pto.VcvtRoundMode.R,
    )
    y_recip = pto.vdiv(pto.vbr(fp32_one), y_float, active_mask)
    reciprocal_bits = pto.vadds(
        pto.vbitcast(y_recip, pto.ui32), pto.ui32(0x0FFFFFFE), active_mask
    )

    low_mask, high_mask = pto.pintlv_b32(active_mask, false_mask)
    lower_bits, higher_bits = pto.vintlv(reciprocal_bits, pto.vbr(pto.ui32(0)))
    lower_i64 = pto.vcvt(
        pto.vbitcast(lower_bits, pto.f32), pto.i64, low_mask,
        rnd=pto.VcvtRoundMode.F, sat=pto.VcvtSatMode.NOSAT,
        part=pto.VcvtPartMode.EVEN,
    )
    higher_i64 = pto.vcvt(
        pto.vbitcast(higher_bits, pto.f32), pto.i64, high_mask,
        rnd=pto.VcvtRoundMode.F, sat=pto.VcvtSatMode.NOSAT,
        part=pto.VcvtPartMode.EVEN,
    )
    reciprocal, _ = pto.vdintlv(
        pto.vbitcast(lower_i64, pto.ui32),
        pto.vbitcast(higher_i64, pto.ui32),
    )
    active_mask, _ = pto.pdintlv_b32(low_mask, high_mask)

    reciprocal_negative = pto.vcmps(
        pto.vbitcast(reciprocal_bits, pto.f32), pto.f32(0.0), active_mask, pto.CmpMode.LT
    )
    reciprocal = pto.vsel(pto.vbr(pto.ui32(0)), reciprocal, reciprocal_negative)

    correction = pto.vmul(reciprocal, abs_y, active_mask)
    correction = pto.vbitcast(
        pto.vneg(pto.vbitcast(correction, pto.i32), active_mask), pto.ui32
    )
    _, correction_high = pto.vmull(reciprocal, correction, active_mask)
    reciprocal = pto.vadd(reciprocal, correction_high, active_mask)

    _, quotient = pto.vmull(abs_x, reciprocal, active_mask)
    unsigned_sign_bit = pto.vbr(pto.ui32(0x80000000))
    for _ in range(4):
        product_low, product_high = pto.vmull(quotient, abs_y, active_mask)
        product_high_nonzero = pto.vcmp(
            product_high, pto.vbr(pto.ui32(0)), active_mask, pto.CmpMode.NE
        )
        product_low_ordered = pto.vbitcast(
            pto.vxor(product_low, unsigned_sign_bit, active_mask), pto.i32
        )
        dividend_ordered = pto.vbitcast(
            pto.vxor(abs_x, unsigned_sign_bit, active_mask), pto.i32
        )
        low_too_high = pto.vcmp(
            product_low_ordered, dividend_ordered, active_mask, pto.CmpMode.GT
        )
        too_high = pto.por(product_high_nonzero, low_too_high, active_mask)
        quotient = pto.vsel(
            pto.vsubs(quotient, pto.ui32(1), active_mask), quotient, too_high
        )

    product = pto.vmul(quotient, abs_y, active_mask)
    remainder = pto.vsub(abs_x, product, active_mask)
    for _ in range(2):
        ge = pto.vcmp(remainder, abs_y, active_mask, pto.CmpMode.GE)
        remainder = pto.vsel(pto.vsub(remainder, abs_y, active_mask), remainder, ge)
        quotient = pto.vsel(pto.vadds(quotient, pto.ui32(1), active_mask), quotient, ge)

    signed_bits = pto.vbitcast(quotient, result_dtype)
    negative_quotient = pto.vneg(signed_bits, active_mask)
    signed_quotient = pto.vsel(signed_bits, negative_quotient, positive)
    return pto.vsel(pto.vbr(neg_one), signed_quotient, zero_mask)
