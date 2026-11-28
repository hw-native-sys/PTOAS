# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Signed vector division following the dav-3510 zeroing implementation."""

from ptodsl import pto

from .div_uint import _div_ui16_quotient, _refine_quotient


def _signed_zero_quotient(lhs, dtype, bits, mask, sign_mask):
    """Select the signed endpoint for a zero divisor."""
    maximum = (1 << (bits - 1)) - 1
    minimum = -(1 << (bits - 1))
    positive = pto.vdup(dtype(maximum), mask)
    negative = pto.vdup(dtype(minimum), mask)
    is_negative = pto.vcmps(lhs, dtype(0), sign_mask, pto.CmpMode.LT)
    return pto.vsel(negative, positive, is_negative)


def _convert_i32_reciprocal(reciprocal_bits, active_mask, fill_mask, fill_vector):
    """Convert through f32-to-i64 to preserve the unsigned 32-bit magnitude."""
    low_mask, high_mask = pto.pintlv_b32(active_mask, fill_mask)
    lower_bits, higher_bits = pto.vintlv(reciprocal_bits, fill_vector)
    lower_i64 = pto.vcvt(
        pto.vbitcast(lower_bits, pto.f32), pto.i64, low_mask,
        rnd=pto.VcvtRoundMode.F, sat=pto.VcvtSatMode.NOSAT, part=pto.VcvtPartMode.EVEN,
    )
    higher_i64 = pto.vcvt(
        pto.vbitcast(higher_bits, pto.f32), pto.i64, high_mask,
        rnd=pto.VcvtRoundMode.F, sat=pto.VcvtSatMode.NOSAT, part=pto.VcvtPartMode.EVEN,
    )
    reciprocal, _ = pto.vdintlv(
        pto.vbitcast(lower_i64, pto.ui32), pto.vbitcast(higher_i64, pto.ui32)
    )
    active_mask, _ = pto.pdintlv_b32(low_mask, high_mask)
    return reciprocal, active_mask


def _estimate_i32_reciprocal(abs_y, one, active_mask, original_mask):
    """Convert the f32 reciprocal estimate to an unsigned fixed-point multiplier."""
    y_float = pto.vcvt(
        pto.vbitcast(abs_y, pto.i32), pto.f32, active_mask, rnd=pto.VcvtRoundMode.R
    )
    y_recip = pto.vdiv(one, y_float, active_mask)
    reciprocal_bits = pto.vadds(
        pto.vbitcast(y_recip, pto.ui32), pto.ui32(0x0FFFFFFE), active_mask
    )
    # The odd lanes of each interleaved half are unused by PART_EVEN.
    reciprocal, active_mask = _convert_i32_reciprocal(
        reciprocal_bits, active_mask, original_mask, reciprocal_bits
    )
    reciprocal_zero = pto.vdup(pto.ui32(0), active_mask)
    reciprocal_negative = pto.vcmps(
        pto.vbitcast(reciprocal_bits, pto.f32), pto.f32(0.0), active_mask, pto.CmpMode.LT
    )
    return pto.vsel(reciprocal_zero, reciprocal, reciprocal_negative), active_mask


def _refine_i32_reciprocal(abs_y, reciprocal, mask):
    """Refine the reciprocal with one fixed-point Newton step."""
    correction = pto.vmul(reciprocal, abs_y, mask)
    correction = pto.vbitcast(pto.vneg(pto.vbitcast(correction, pto.i32), mask), pto.ui32)
    _, correction_high = pto.vmull(reciprocal, correction, mask)
    return pto.vadd(reciprocal, correction_high, mask)


def div_i32_soft(lhs, rhs, mask):
    """Compute signed 32-bit division following the dav-3510 vector sequence."""
    dtype = pto.si32 if str(lhs.type).endswith("xsi32>") else pto.i32
    zero = dtype(0)
    one = pto.vdup(pto.f32(1.0), mask)
    zero_mask = pto.vcmps(rhs, zero, mask, pto.CmpMode.EQ)
    zero_quotient = _signed_zero_quotient(lhs, dtype, 32, mask, zero_mask)
    active_mask = pto.pnot(zero_mask, mask)

    abs_x = pto.vbitcast(pto.vabs(lhs, active_mask), pto.ui32)
    abs_y = pto.vbitcast(pto.vabs(rhs, active_mask), pto.ui32)
    signs = pto.vxor(lhs, rhs, active_mask)
    nonnegative = pto.vcmps(signs, zero, active_mask, pto.CmpMode.GE)

    reciprocal, active_mask = _estimate_i32_reciprocal(abs_y, one, active_mask, mask)
    reciprocal = _refine_i32_reciprocal(abs_y, reciprocal, active_mask)
    _, quotient = pto.vmull(abs_x, reciprocal, active_mask)
    quotient = _refine_quotient(abs_x, abs_y, quotient, active_mask, pto.ui32(1))
    signed_quotient = pto.vbitcast(quotient, dtype)
    negative_quotient = pto.vneg(signed_quotient, active_mask)
    signed_quotient = pto.vsel(signed_quotient, negative_quotient, nonnegative)
    return pto.vsel(zero_quotient, signed_quotient, zero_mask)


def div_i16_soft(lhs, rhs, mask):
    """Compute signed 16-bit division using the dav-3510 unsigned magnitude path."""
    dtype = pto.si16 if str(lhs.type).endswith("xsi16>") else pto.i16
    zero = dtype(0)
    zero_mask = pto.vcmps(rhs, zero, mask, pto.CmpMode.EQ)
    zero_quotient = _signed_zero_quotient(lhs, dtype, 16, mask, mask)
    active_mask = pto.pnot(zero_mask, mask)

    abs_x = pto.vbitcast(pto.vabs(lhs, active_mask), pto.ui16)
    abs_y = pto.vbitcast(pto.vabs(rhs, active_mask), pto.ui16)
    signs = pto.vxor(lhs, rhs, active_mask)
    nonnegative = pto.vcmps(signs, zero, active_mask, pto.CmpMode.GE)

    quotient, _ = _div_ui16_quotient(abs_x, abs_y, active_mask)
    signed_quotient = pto.vbitcast(quotient, dtype)
    negative_quotient = pto.vneg(signed_quotient, active_mask)
    signed_quotient = pto.vsel(signed_quotient, negative_quotient, nonnegative)
    return pto.vsel(zero_quotient, signed_quotient, zero_mask)
