# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""A5 unsigned integer division for tile and register operations."""

from ptodsl import pto


def div_uint_soft(lhs, rhs, mask, dtype):
    """Restoring division, with one quotient bit resolved per iteration.

    A zero divisor is undefined; this loop sets every quotient bit in that lane.
    """
    bits = {"ui8": 8, "ui16": 16, "ui32": 32}[str(dtype)]
    scalar_dtype = {8: pto.ui8, 16: pto.ui16, 32: pto.ui32}[bits]
    zero = pto.vbr(scalar_dtype(0))
    one = pto.vbr(scalar_dtype(1))
    remainder = zero
    quotient = zero

    for bit in range(bits - 1, -1, -1):
        carry = pto.vcmps(remainder, scalar_dtype(1 << (bits - 1)), mask, pto.CmpMode.GE)
        next_bit = pto.vands(pto.vshrs(lhs, bit, mask), scalar_dtype(1), mask)
        remainder = pto.vor(pto.vshls(remainder, 1, mask), next_bit, mask)
        ge = pto.por(carry, pto.vcmp(remainder, rhs, mask, pto.CmpMode.GE), mask)
        remainder = pto.vsel(pto.vsub(remainder, rhs, mask), remainder, ge)
        quotient_bit = pto.vshls(one, bit, mask)
        quotient = pto.vsel(pto.vor(quotient, quotient_bit, mask), quotient, ge)
    return quotient


def _refine_quotient(lhs, rhs, quotient, mask, one):
    """Correct a reciprocal-based quotient by comparing the integer remainder."""
    product = pto.vmul(quotient, rhs, mask)
    remainder = pto.vsub(lhs, product, mask)
    correction = pto.vcmp(remainder, rhs, mask, pto.CmpMode.GE)
    # dav-3510 uses predicated VMOV to update selected lanes in place.
    # VSEL implements the same merge by selecting the old value for inactive lanes.
    remainder = pto.vsel(pto.vsub(remainder, rhs, correction), remainder, correction)
    quotient = pto.vsel(pto.vadds(quotient, one, correction), quotient, correction)
    correction = pto.vcmp(remainder, rhs, mask, pto.CmpMode.GE)
    return pto.vsel(pto.vadds(quotient, one, correction), quotient, correction)


def _estimate_u16_quotient(lhs, rhs, full_b32, one, zero):
    """Follow the dav-3510 u16 reciprocal estimate using both register halves."""
    divisor_low, divisor_high = pto.vintlv(rhs, zero)
    divisor_low = pto.vcvt(pto.vbitcast(divisor_low, pto.i32), pto.f32, full_b32, rnd=pto.VcvtRoundMode.F)
    divisor_high = pto.vcvt(pto.vbitcast(divisor_high, pto.i32), pto.f32, full_b32, rnd=pto.VcvtRoundMode.F)
    reciprocal_low = pto.vdiv(one, divisor_low, full_b32)
    reciprocal_high = pto.vdiv(one, divisor_high, full_b32)
    scale = pto.vbitcast(pto.vdup(pto.ui32(0x47800000), full_b32), pto.f32)
    scaled_low = pto.vmul(reciprocal_low, scale, full_b32)
    scaled_high = pto.vmul(reciprocal_high, scale, full_b32)
    fixed_low = pto.vcvt(scaled_low, pto.i32, full_b32, rnd=pto.VcvtRoundMode.F, sat=pto.VcvtSatMode.NOSAT)
    fixed_high = pto.vcvt(scaled_high, pto.i32, full_b32, rnd=pto.VcvtRoundMode.F, sat=pto.VcvtSatMode.NOSAT)
    dividend_low, dividend_high = pto.vintlv(lhs, zero)
    product_low = pto.vmul(pto.vbitcast(fixed_low, pto.ui32), pto.vbitcast(dividend_low, pto.ui32), full_b32)
    product_high = pto.vmul(pto.vbitcast(fixed_high, pto.ui32), pto.vbitcast(dividend_high, pto.ui32), full_b32)
    _, quotient = pto.vdintlv(pto.vbitcast(product_low, pto.ui16), pto.vbitcast(product_high, pto.ui16))
    return quotient


def _div_ui16_quotient(vec, divisor, mask):
    """Compute the dav-3510 u16 quotient before zeroing inactive lanes."""
    zero_mask = pto.vcmps(divisor, pto.ui16(0), mask, pto.CmpMode.EQ)
    zero_quotient = pto.vdup(pto.ui16(0xFFFF), mask)
    active = pto.pnot(zero_mask, mask)
    full_b16 = pto.pset_b16(pto.PAT.ALL)
    # PAT_ALL sets every physical predicate bit. View the same predicate as b32.
    full_b32 = pto.pbitcast(full_b16, pto.mask_type("b32"))
    one = pto.vdup(pto.f32(1.0), full_b32)
    zero = pto.vdup(pto.ui16(0), full_b16)

    quotient = _estimate_u16_quotient(vec, divisor, full_b32, one, zero)
    quotient = _refine_quotient(vec, divisor, quotient, active, pto.ui16(1))
    return pto.vsel(zero_quotient, quotient, zero_mask), zero


def div_ui16_soft(vec, divisor, mask):
    """Divide u16 register values with the dav-3510 reciprocal sequence."""
    quotient, zero = _div_ui16_quotient(vec, divisor, mask)
    # The register estimate runs on every lane. The PTO result zeroes inactive lanes.
    return pto.vsel(quotient, zero, mask)


def _estimate_u32_reciprocal(divisor, all_lanes, one, zero):
    """Convert both unsigned halves through i64 for the dav-3510 u32 estimate."""
    zero_f32 = pto.vdup(pto.f32(0.0), all_lanes)
    divisor_low, divisor_high = pto.vintlv(divisor, zero)
    divisor_low = pto.vcvt(
        pto.vbitcast(divisor_low, pto.i64), pto.f32, all_lanes,
        rnd=pto.VcvtRoundMode.F, part=pto.VcvtPartMode.EVEN,
    )
    divisor_high = pto.vcvt(
        pto.vbitcast(divisor_high, pto.i64), pto.f32, all_lanes,
        rnd=pto.VcvtRoundMode.F, part=pto.VcvtPartMode.EVEN,
    )
    divisor_f32, _ = pto.vdintlv(divisor_low, divisor_high)
    reciprocal = pto.vdiv(one, divisor_f32, all_lanes)
    scale = pto.vdup(pto.f32(2.0**32 - 2.0**11), all_lanes)
    scaled = pto.vmul(reciprocal, scale, all_lanes)
    scaled_low, scaled_high = pto.vintlv(scaled, zero_f32)
    fixed_low = pto.vcvt(
        scaled_low, pto.i64, all_lanes,
        rnd=pto.VcvtRoundMode.F, sat=pto.VcvtSatMode.NOSAT, part=pto.VcvtPartMode.EVEN,
    )
    fixed_high = pto.vcvt(
        scaled_high, pto.i64, all_lanes,
        rnd=pto.VcvtRoundMode.F, sat=pto.VcvtSatMode.NOSAT, part=pto.VcvtPartMode.EVEN,
    )
    multiplier, _ = pto.vdintlv(pto.vbitcast(fixed_low, pto.ui32), pto.vbitcast(fixed_high, pto.ui32))
    return multiplier


def div_ui32_soft(vec, divisor, mask):
    """Divide u32 register values with the dav-3510 reciprocal sequence."""
    zero_mask = pto.vcmps(divisor, pto.ui32(0), mask, pto.CmpMode.EQ)
    zero_quotient = pto.vdup(pto.ui32(0xFFFFFFFF), mask)
    active = pto.pnot(zero_mask, mask)
    all_lanes = pto.pset_b32(pto.PAT.ALL)
    one = pto.vdup(pto.f32(1.0), all_lanes)
    zero = pto.vdup(pto.ui32(0), all_lanes)

    multiplier = _estimate_u32_reciprocal(divisor, all_lanes, one, zero)
    error = pto.vmul(multiplier, divisor, all_lanes)
    error = pto.vbitcast(pto.vneg(pto.vbitcast(error, pto.i32), all_lanes), pto.ui32)
    _, correction = pto.vmull(multiplier, error, all_lanes)
    multiplier = pto.vadd(multiplier, correction, all_lanes)
    _, quotient = pto.vmull(vec, multiplier, all_lanes)

    quotient = _refine_quotient(vec, divisor, quotient, active, pto.ui32(1))
    quotient = pto.vsel(zero_quotient, quotient, zero_mask)
    return pto.vsel(quotient, zero, mask)
