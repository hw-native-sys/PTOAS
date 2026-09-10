# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""A5 signed integer division soft implementation."""

from dataclasses import dataclass

from ptodsl import pto


@dataclass
class _I16WidenedOperands:
    """Carry the widened i16 division operands between helper stages."""

    vx_lower_u32: object
    vx_higher_u32: object
    vy_lower_u32: object
    vy_higher_u32: object
    active_low: object
    active_high: object


def _estimate_i32_reciprocal(abs_y, active_mask, false_mask):
    """Estimate the reciprocal multiplier used by the i32 correction loop."""
    fp32_one = pto.f32(1.0)
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
    return pto.vadd(reciprocal, correction_high, active_mask), active_mask


def div_i32_soft(vec, scalar_vec, mask):
    """Compute signed i32 division without relying on integer ``vdiv``.

    The quotient magnitude is estimated from an f32 reciprocal, widened to
    i64, and corrected against the exact remainder. The result is exact for
    every nonzero int32 divisor; division by zero is intentionally undefined.
    """
    result_dtype = pto.si32 if str(vec.type).endswith("xsi32>") else pto.i32
    zero = result_dtype(0)
    neg_one = result_dtype(-1)
    false_mask = pto.pset_b32(pto.PAT.ALLF)

    zero_mask = pto.vcmps(scalar_vec, zero, mask, pto.CmpMode.EQ)
    active_mask = pto.pnot(zero_mask, mask)

    abs_x = pto.vbitcast(pto.vabs(vec, active_mask), pto.ui32)
    abs_y = pto.vbitcast(pto.vabs(scalar_vec, active_mask), pto.ui32)
    x_xor_y = pto.vxor(vec, scalar_vec, active_mask)
    positive = pto.vcmps(x_xor_y, zero, active_mask, pto.CmpMode.GE)

    reciprocal, active_mask = _estimate_i32_reciprocal(abs_y, active_mask, false_mask)

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


def _widen_i16_operands(abs_x, abs_y, active_mask):
    """Widen i16 magnitudes into interleaved u32 halves."""
    full_b32 = pto.pset_b32(pto.PAT.ALL)
    false_mask = pto.pset_b16(pto.PAT.ALLF)
    low_mask, high_mask = pto.pintlv_b16(active_mask, false_mask)
    zero_u16 = pto.vbr(pto.ui16(0))
    vy_lower, vy_higher = pto.vintlv(abs_y, zero_u16)
    vx_lower, vx_higher = pto.vintlv(abs_x, zero_u16)
    vy_lower = pto.vcvt(vy_lower, pto.ui32, low_mask, part=pto.VcvtPartMode.EVEN)
    vy_higher = pto.vcvt(vy_higher, pto.ui32, high_mask, part=pto.VcvtPartMode.EVEN)
    vx_lower = pto.vcvt(vx_lower, pto.ui32, low_mask, part=pto.VcvtPartMode.EVEN)
    vx_higher = pto.vcvt(vx_higher, pto.ui32, high_mask, part=pto.VcvtPartMode.EVEN)
    active_low = pto.vcmps(vy_lower, pto.ui32(0), full_b32, pto.CmpMode.NE)
    active_high = pto.vcmps(vy_higher, pto.ui32(0), full_b32, pto.CmpMode.NE)
    return _I16WidenedOperands(
        vx_lower, vx_higher, vy_lower, vy_higher, active_low, active_high
    )


def _estimate_i16_quotient(widened):
    """Estimate each i16 quotient magnitude inside the widened u32 halves.

    Each divisor half is converted to f32, its reciprocal is scaled by 65536
    so the subsequent u32 multiply lands in the correct fixed-point domain,
    and the two halves are packed back into one vector through ``vdintlv``.
    """
    fp32_one = pto.f32(1.0)
    fp32_65536 = pto.f32(65536.0)
    lower_f32 = pto.vcvt(
        pto.vbitcast(widened.vy_lower_u32, pto.i32), pto.f32, widened.active_low,
        rnd=pto.VcvtRoundMode.F,
    )
    higher_f32 = pto.vcvt(
        pto.vbitcast(widened.vy_higher_u32, pto.i32), pto.f32, widened.active_high,
        rnd=pto.VcvtRoundMode.F,
    )
    lower_recip = pto.vdiv(pto.vbr(fp32_one), lower_f32, widened.active_low)
    higher_recip = pto.vdiv(pto.vbr(fp32_one), higher_f32, widened.active_high)
    lower_scaled = pto.vmul(lower_recip, pto.vbr(fp32_65536), widened.active_low)
    higher_scaled = pto.vmul(higher_recip, pto.vbr(fp32_65536), widened.active_high)
    lower_v = pto.vbitcast(
        pto.vcvt(
            lower_scaled, pto.i32, widened.active_low,
            rnd=pto.VcvtRoundMode.F, sat=pto.VcvtSatMode.NOSAT,
        ),
        pto.ui32,
    )
    higher_v = pto.vbitcast(
        pto.vcvt(
            higher_scaled, pto.i32, widened.active_high,
            rnd=pto.VcvtRoundMode.F, sat=pto.VcvtSatMode.NOSAT,
        ),
        pto.ui32,
    )
    q_lower = pto.vmul(lower_v, widened.vx_lower_u32, widened.active_low)
    q_higher = pto.vmul(higher_v, widened.vx_higher_u32, widened.active_high)
    _, q_tmp = pto.vdintlv(
        pto.vbitcast(q_lower, pto.ui16),
        pto.vbitcast(q_higher, pto.ui16),
    )
    return q_tmp


def div_i16_soft(vec, scalar_vec, mask):
    """Compute signed i16 division without relying on integer ``vdiv``.

    Each 128-lane i16 operand is widened into two 64-lane u32 halves through
    b16 interleave plus a 2:1 ``vcvt`` (the VPTO vcvt contract conserves total
    vector bits, so a straight i16->f32 conversion at the same lane count is
    not expressible). The quotient magnitude is estimated from a 65536-scaled
    f32 reciprocal in the u32 domain, narrowed back to u16 through ``vdintlv``,
    and corrected against the exact remainder. The result is exact for every
    nonzero int16 divisor; division by zero is intentionally undefined.
    """
    result_dtype = pto.si16 if str(vec.type).endswith("xsi16>") else pto.i16
    zero = result_dtype(0)
    neg_one = result_dtype(-1)
    ui16_one = pto.ui16(1)
    full_b16 = pto.pset_b16(pto.PAT.ALL)

    zero_mask = pto.vcmps(scalar_vec, zero, mask, pto.CmpMode.EQ)
    active_mask = pto.pnot(zero_mask, mask)

    abs_x = pto.vbitcast(pto.vabs(vec, active_mask), pto.ui16)
    abs_y = pto.vbitcast(pto.vabs(scalar_vec, active_mask), pto.ui16)
    x_xor_y = pto.vxor(vec, scalar_vec, active_mask)
    positive = pto.vcmps(x_xor_y, zero, active_mask, pto.CmpMode.GE)

    widened = _widen_i16_operands(abs_x, abs_y, active_mask)
    q_tmp = _estimate_i16_quotient(widened)

    yq = pto.vmul(q_tmp, abs_y, active_mask)
    remainder = pto.vsub(abs_x, yq, active_mask)
    for _ in range(2):
        ge = pto.vcmp(remainder, abs_y, active_mask, pto.CmpMode.GE)
        remainder = pto.vsel(
            pto.vsub(remainder, abs_y, active_mask), remainder, ge
        )
        q_tmp = pto.vsel(
            pto.vadds(q_tmp, ui16_one, active_mask), q_tmp, ge
        )

    signed_bits = pto.vbitcast(q_tmp, result_dtype)
    negative_quotient = pto.vneg(signed_bits, active_mask)
    signed_quotient = pto.vsel(signed_bits, negative_quotient, positive)
    return pto.vsel(pto.vbr(neg_one), signed_quotient, zero_mask)
