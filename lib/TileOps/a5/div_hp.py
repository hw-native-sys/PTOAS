# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Shared IEEE 754 high-precision division algorithms for pto.tdiv and pto.tdivs

This module provides inline_proc functions that implement IEEE 754 compliant
division with improved accuracy for:
- Precision-sensitive values (1/7, 7/3, etc.)
- Subnormal numbers (denormals)
- Overflow/underflow boundary cases
- NaN propagation

Reference: pto-isa/include/pto/npu/a5/custom/Div754.hpp
"""

from dataclasses import dataclass

from ptodsl import pto


@dataclass
class _F32NormalizedState:
    """Normalized f32 division operands and masks shared by the fix-up phases."""

    src0_norm: object
    src1_norm: object
    src0_abs: object
    src1_abs: object
    src0_all_u32: object
    src1_all_u32: object
    mask_src0_subnormal: object
    mask_src0_normal: object
    mask_src1_subnormal: object
    mask_src1_normal: object
    mask_valid: object


@dataclass
class _F16NormalizedState:
    """Normalized f16 division operands and masks shared by the fix-up phases."""

    src0_norm: object
    src1_norm: object
    src0_abs: object
    src1_abs: object
    src0_all_u16: object
    src1_all_u16: object
    mask_src0_subnormal: object
    mask_src0_normal: object
    mask_src1_subnormal: object
    mask_src1_normal: object
    mask_valid: object


def _vmula(addend, lhs, rhs, mask):
    return pto.vadd(addend, pto.vmul(lhs, rhs, mask), mask)


def _div_three_candidate_search_f32(lhs, rhs, mask):
    """Three-candidate search core algorithm for IEEE 754 division accuracy improvement.

    Corresponds to DivPrecisionImpl in pto-isa/include/pto/npu/a5/custom/Div754.hpp:16-62

    Algorithm: Computes three candidates (z, z-1, z+1) and selects the one with smallest
    residual |lhs - z*rhs|, improving accuracy for values like 1/7 that have infinite
    binary representation.
    """

    # IEEE 754 Float32 bit patterns (corresponds to Div754.hpp:18-19)
    inf_bound_u32 = pto.ui32(0x7f800000)  # Infinity bound: sign=0, exp=255, mant=0
    sign_bit_u32 = pto.ui32(0x80000000)   # Sign bit mask: bit31=1, others=0
    zero_f32 = pto.f32(0.0)
    one_f32 = pto.f32(1.0)
    neg_one_f32 = pto.f32(-1.0)

    z = pto.vdiv(lhs, rhs, mask)
    z_init = z

    z_u32 = pto.vbitcast(z, pto.ui32)
    z_or_sign = pto.vor(z_u32, pto.vbr(sign_bit_u32), mask)
    is_inf_nan = pto.vcmp(z_or_sign, pto.vbr(inf_bound_u32), mask, pto.CmpMode.GE)

    is_zero = pto.vcmp(z, pto.vbr(zero_f32), mask, pto.CmpMode.EQ)

    special_mask = pto.por(is_inf_nan, is_zero, mask)

    y = pto.vmuls(rhs, neg_one_f32, mask)
    r = _vmula(lhs, z, y, mask)

    z_pre = pto.vadds(z, neg_one_f32, mask)
    z_next = pto.vadds(z, one_f32, mask)

    r_pre = _vmula(lhs, z_pre, y, mask)
    r_next = _vmula(lhs, z_next, y, mask)

    r_abs = pto.vabs(r, mask)
    r_pre_abs = pto.vabs(r_pre, mask)
    r_next_abs = pto.vabs(r_next, mask)

    better_pre = pto.vcmp(r_pre_abs, r_abs, mask, pto.CmpMode.LT)
    z_best = pto.vsel(z_pre, z, better_pre)
    r_best_abs = pto.vsel(r_pre_abs, r_abs, better_pre)

    better_next = pto.vcmp(r_next_abs, r_best_abs, mask, pto.CmpMode.LT)
    z_best = pto.vsel(z_next, z_best, better_next)

    divided = pto.vsel(z_init, z_best, special_mask)

    return divided


def _div_f32_normalize(src0, src1, mask):
    """Classify f32 specials/subnormals and re-bias the operand exponents.

    Produces the exponent-normalized operand pair together with the
    subnormal/normal masks and validity mask consumed by the later fix-up
    phases. Mirrors Div754.hpp:65-178.
    """
    f32_inf = pto.ui32(0x7f800000)            # +Infinity: sign=0, exp=255, mant=0
    sign_extractor = pto.ui32(0x80000000)     # Sign bit mask (bit31)
    exponent_extractor = pto.ui32(0x807FFFFF) # Clear exponent bits [30:23]
    exponent_normalizer = pto.ui32(0x3F800000) # Bias 127: 1.0f reference
    subnormal_threshold = pto.ui32(0x007FFFFF) # Max subnormal: (1-2^-23)*2^-126
    nan_value = pto.ui32(0x7fc00000)          # Quiet NaN: exp=255, mant=0x400000
    min_denormal = pto.ui32(0x1)              # Smallest positive: 2^-149

    # Subnormal normalization factors (corresponds to Div754.hpp:86-89)
    normalize_scale_enlarge = pto.f32(8388608.0)           # 2^23: shifts subnormals to normal range
    normalize_scale_reduce = pto.f32(1.1920928955078125e-07) # 2^-23: inverse for result compensation
    src0_abs = pto.vabs(src0, mask)
    src1_abs = pto.vabs(src1, mask)

    src0_abs_u32 = pto.vbitcast(src0_abs, pto.ui32)
    src1_abs_u32 = pto.vbitcast(src1_abs, pto.ui32)

    mask_inf_src0 = pto.vcmp(src0_abs_u32, pto.vbr(f32_inf), mask, pto.CmpMode.EQ)
    mask_inf_src1 = pto.vcmp(src1_abs_u32, pto.vbr(f32_inf), mask, pto.CmpMode.EQ)
    mask_invalid = pto.por(mask_inf_src0, mask_inf_src1, mask)

    mask_zero_src0 = pto.vcmp(src0_abs_u32, pto.vbr(pto.ui32(0)), mask, pto.CmpMode.EQ)
    mask_invalid = pto.por(mask_invalid, mask_zero_src0, mask)
    mask_zero_src1 = pto.vcmp(src1_abs_u32, pto.vbr(pto.ui32(0)), mask, pto.CmpMode.EQ)
    mask_invalid = pto.por(mask_invalid, mask_zero_src1, mask)

    mask_valid = pto.pnot(mask_invalid, mask)

    # Detect subnormal numbers (denormals)
    # NOTE: Uses EQ/LT comparison matching pto-isa Div754.hpp asymmetry:
    # - src0: EQ comparison (Div754.hpp:159) - detects exact max subnormal
    # - src1: LT comparison (Div754.hpp:166) - covers entire subnormal range
    mask_src0_subnormal = pto.vcmp(src0_abs_u32, pto.vbr(subnormal_threshold), mask, pto.CmpMode.EQ)
    mask_src0_normal = pto.pnot(mask_src0_subnormal, mask)
    src0_subnormal = pto.vmuls(src0, normalize_scale_enlarge, mask_src0_subnormal)

    mask_src1_subnormal = pto.vcmp(src1_abs_u32, pto.vbr(subnormal_threshold), mask, pto.CmpMode.LT)
    mask_src1_normal = pto.pnot(mask_src1_subnormal, mask)
    src1_subnormal = pto.vmuls(src1, normalize_scale_enlarge, mask_src1_subnormal)

    src0_all = pto.vsel(src0, src0_subnormal, mask_src0_normal)
    src1_all = pto.vsel(src1, src1_subnormal, mask_src1_normal)

    src0_all_u32 = pto.vbitcast(src0_all, pto.ui32)
    src1_all_u32 = pto.vbitcast(src1_all, pto.ui32)

    src0_norm_u32 = pto.vand(src0_all_u32, pto.vbr(exponent_extractor), mask_valid)
    src1_norm_u32 = pto.vand(src1_all_u32, pto.vbr(exponent_extractor), mask_valid)

    src0_norm_u32 = pto.vadd(src0_norm_u32, pto.vbr(exponent_normalizer), mask_valid)
    src1_norm_u32 = pto.vadd(src1_norm_u32, pto.vbr(exponent_normalizer), mask_valid)

    src0_norm_f32 = pto.vbitcast(src0_norm_u32, pto.f32)
    src1_norm_f32 = pto.vbitcast(src1_norm_u32, pto.f32)
    src0_norm = pto.vsel(src0_norm_f32, src0_all, mask_valid)
    src1_norm = pto.vsel(src1_norm_f32, src1_all, mask_valid)
    return _F32NormalizedState(
        src0_norm, src1_norm, src0_abs, src1_abs, src0_all_u32,
        src1_all_u32, mask_src0_subnormal, mask_src0_normal,
        mask_src1_subnormal, mask_src1_normal, mask_valid,
    )


def _div_f32_underflow(dst_u32, dst_sign, scale, state, mask):
    """Replace f32 denormal underflow results in the u32 domain.

    When the normalized exponent difference lands below the f32 denormal
    range the quotient bits are substituted by a denormal or zero pattern
    carrying the sign. Mirrors Div754.hpp:228-258.
    """
    src0_norm = state.src0_norm
    src1_norm = state.src1_norm
    mask_valid = state.mask_valid

    neg23 = pto.si32(-23)
    min_denormal = pto.ui32(0x1)
    mask_underflow1 = pto.vcmp(scale, pto.vbr(neg23), mask, pto.CmpMode.EQ)
    mask_underflow1 = pto.pand(mask_underflow1, mask_valid, mask)

    z1_u32 = pto.vadd(dst_sign, pto.vbr(min_denormal), mask_underflow1)
    z2_u32 = pto.vadd(dst_sign, pto.vbr(pto.ui32(0)), mask_underflow1)

    src0_norm_abs = pto.vabs(src0_norm, mask_valid)
    src1_norm_abs = pto.vabs(src1_norm, mask_valid)
    mask_norm = pto.vcmp(src0_norm_abs, src1_norm_abs, mask_valid, pto.CmpMode.LE)

    z1_sel = pto.vsel(z2_u32, z1_u32, mask_norm)
    dst_u32_temp = pto.vsel(z1_sel, dst_u32, mask_underflow1)

    mask_underflow1_not = pto.pnot(mask_underflow1, mask)
    mask_valid_temp = pto.pand(mask_underflow1_not, mask_valid, mask)

    mask_underflow2 = pto.vcmp(scale, pto.vbr(neg23), mask, pto.CmpMode.LT)
    mask_underflow2 = pto.pand(mask_underflow2, mask_valid_temp, mask)

    z1_u32 = pto.vadd(dst_sign, pto.vbr(pto.ui32(0)), mask_underflow2)
    dst_u32_temp = pto.vsel(z1_u32, dst_u32_temp, mask_underflow2)

    mask_underflow2_not = pto.pnot(mask_underflow2, mask)
    mask_valid_temp = pto.pand(mask_underflow2_not, mask_valid_temp, mask)
    return dst_u32_temp, mask_valid_temp


def _div_f32_boundary(underflow_result, dst_sign, scale, state, mask):
    """Clamp f32 overflow, rescale by the exponent and propagate NaNs.

    Operates on the lane mask left by the underflow phase, then applies the
    exponent scale factor (or the shifted 1.0 reference for negative)
    before the final NaN overwrite. Mirrors Div754.hpp:260-288.
    """
    dst_u32_temp, mask_valid_temp = underflow_result
    src0_abs = state.src0_abs
    src1_abs = state.src1_abs

    f32_inf = pto.ui32(0x7f800000)
    nan_value = pto.ui32(0x7fc00000)
    max_exp = pto.si32(255)
    mask_overflow1 = pto.vcmp(scale, pto.vbr(max_exp), mask, pto.CmpMode.EQ)
    mask_overflow1 = pto.pand(mask_overflow1, mask_valid_temp, mask)

    scale_adj = pto.vadds(scale, pto.si32(-1), mask_overflow1)
    scale = pto.vsel(scale_adj, scale, mask_overflow1)

    dst_f32_temp = pto.vbitcast(dst_u32_temp, pto.f32)
    z1_f32 = pto.vmuls(dst_f32_temp, pto.f32(2.0), mask_overflow1)
    dst_f32_temp = pto.vsel(z1_f32, dst_f32_temp, mask_overflow1)

    mask_overflow2 = pto.vcmp(scale, pto.vbr(max_exp), mask, pto.CmpMode.GT)
    mask_overflow2 = pto.pand(mask_overflow2, mask_valid_temp, mask)

    z1_u32 = pto.vadd(dst_sign, pto.vbr(f32_inf), mask_overflow2)
    dst_u32_temp = pto.vbitcast(dst_f32_temp, pto.ui32)
    dst_u32_temp = pto.vsel(z1_u32, dst_u32_temp, mask_overflow2)

    mask_overflow2_not = pto.pnot(mask_overflow2, mask)
    mask_valid_final = pto.pand(mask_overflow2_not, mask_valid_temp, mask)

    zero_exp = pto.si32(0)
    mask_pos_exp = pto.vcmp(scale, pto.vbr(zero_exp), mask_valid_final, pto.CmpMode.GT)

    scale_u32 = pto.vbitcast(scale, pto.ui32)
    exp_shifted = pto.vshls(scale_u32, pto.i16(23), mask_pos_exp)
    exp_factor_f32 = pto.vbitcast(exp_shifted, pto.f32)

    dst_f32_temp = pto.vbitcast(dst_u32_temp, pto.f32)
    z1_f32 = pto.vmul(dst_f32_temp, exp_factor_f32, mask_pos_exp)
    dst_f32_temp = pto.vsel(z1_f32, dst_f32_temp, mask_pos_exp)

    mask_pos_exp_not = pto.pnot(mask_pos_exp, mask_valid_final)

    # Handle negative exponent (underflow scenarios)
    # Corresponds to Div754.hpp:275
    # Value 0x00400000 = Float32 with exp=0, mantissa bit22=1 (used for shift calculation)
    four_million = pto.i32(4194304)  # Normal float 1.0 in bit representation for exponent manipulation
    scale_abs = pto.vabs(scale, mask_pos_exp_not)

    shr_base_vec = pto.vdup(four_million, mask_pos_exp_not)
    shr_base_i32 = pto.vbitcast(shr_base_vec, pto.si32)
    shr_factor_i32 = pto.vshr(shr_base_i32, scale_abs, mask_pos_exp_not)
    shr_factor_f32 = pto.vbitcast(shr_factor_i32, pto.f32)

    z1_f32 = pto.vmul(dst_f32_temp, shr_factor_f32, mask_pos_exp_not)
    dst_f32_temp = pto.vsel(z1_f32, dst_f32_temp, mask_pos_exp_not)

    mask_nan_src0 = pto.vcmp(src0_abs, src0_abs, mask, pto.CmpMode.NE)
    mask_nan_src1 = pto.vcmp(src1_abs, src1_abs, mask, pto.CmpMode.NE)
    mask_nan = pto.por(mask_nan_src0, mask_nan_src1, mask)

    nan_vec = pto.vbr(nan_value)
    nan_f32_vec = pto.vbitcast(nan_vec, pto.f32)
    dst_final = pto.vsel(nan_f32_vec, dst_f32_temp, mask_nan)
    return dst_final


def _div_ieee754_f32_impl(src0, src1, mask):
    """Complete IEEE 754 float32 high-precision division with subnormal and overflow handling.

    Corresponds to DivIEEE754FloatImpl in pto-isa/include/pto/npu/a5/custom/Div754.hpp:65-288

    Key improvements over pto-isa:
    - Subnormal detection uses LT (line 94) instead of EQ (Div754.hpp:159)
      Rationale: Covers entire subnormal range [2^-149, 2^-126), not just max subnormal
    """
    state = _div_f32_normalize(src0, src1, mask)

    f32_inf = pto.ui32(0x7f800000)
    sign_extractor = pto.ui32(0x80000000)
    normalize_scale_reduce = pto.f32(1.1920928955078125e-07)
    normalize_scale_enlarge = pto.f32(8388608.0)
    dst = _div_three_candidate_search_f32(state.src0_norm, state.src1_norm, state.mask_valid)

    mask0 = pto.pand(state.mask_src0_subnormal, state.mask_src1_normal, mask)
    z1 = pto.vmuls(dst, normalize_scale_reduce, mask0)
    dst = pto.vsel(z1, dst, mask0)

    mask0 = pto.pand(state.mask_src0_normal, state.mask_src1_subnormal, mask)
    z1 = pto.vmuls(dst, normalize_scale_enlarge, mask0)
    dst = pto.vsel(z1, dst, mask0)

    dst_u32 = pto.vbitcast(dst, pto.ui32)
    dst_sign = pto.vand(dst_u32, pto.vbr(sign_extractor), mask)

    src0_exponent = pto.vand(state.src0_all_u32, pto.vbr(f32_inf), mask)
    src1_exponent = pto.vand(state.src1_all_u32, pto.vbr(f32_inf), mask)

    src0_exp_shifted = pto.vshrs(src0_exponent, pto.i16(23), mask)
    src1_exp_shifted = pto.vshrs(src1_exponent, pto.i16(23), mask)

    src0_exp_i32 = pto.vbitcast(src0_exp_shifted, pto.si32)
    src1_exp_i32 = pto.vbitcast(src1_exp_shifted, pto.si32)

    scale = pto.vsub(src0_exp_i32, src1_exp_i32, mask)
    scale = pto.vadds(scale, pto.si32(127), mask)

    underflow_result = _div_f32_underflow(
        dst_u32, dst_sign, scale, state, mask,
    )
    return _div_f32_boundary(
        underflow_result, dst_sign, scale, state, mask,
    )


def _div_f16_normalize(src0, src1, mask):
    """Classify f16 specials/subnormals and re-bias the operand exponents.

    Symmetric subnormal handling (both operands use LT) matching
    Div754.hpp:291-401. Returns the normalized operands plus the masks
    consumed by the fix-up phases.
    """
    f16_inf = pto.ui16(0x7C00)              # +Infinity: sign=0, exp=31, mant=0
    exponent_extractor = pto.ui16(0x83FF)   # Clear exponent bits [14:10]
    exponent_normalizer = pto.ui16(0x3C00)  # 1.0f16 reference (bias=15)
    sign_extractor = pto.ui16(0x8000)       # Sign bit mask (bit15)
    subnormal_threshold = pto.ui16(0x03FF)  # Max subnormal: (1-2^-10)*2^-14
    nan_value = pto.ui16(0x7E00)            # Quiet NaN: exp=31, mant=0x200
    min_denormal = pto.ui16(0x1)            # Smallest positive: 2^-24

    # Subnormal normalization factors (corresponds to Div754.hpp:306-309)
    normalize_scale_enlarge = pto.f16(1024.0)             # 2^10: shifts subnormals to normal range
    normalize_scale_reduce = pto.f16(0.0009765625)        # 2^-10: inverse for result compensation

    src0_abs = pto.vabs(src0, mask)
    src1_abs = pto.vabs(src1, mask)

    src0_abs_u16 = pto.vbitcast(src0_abs, pto.ui16)
    src1_abs_u16 = pto.vbitcast(src1_abs, pto.ui16)

    # Detect Infinity values
    mask_inf_src0 = pto.vcmp(src0_abs_u16, pto.vbr(f16_inf), mask, pto.CmpMode.EQ)
    mask_inf_src1 = pto.vcmp(src1_abs_u16, pto.vbr(f16_inf), mask, pto.CmpMode.EQ)
    mask_invalid = pto.por(mask_inf_src0, mask_inf_src1, mask)

    # Detect Zero values
    mask_zero_src0 = pto.vcmp(src0_abs_u16, pto.vbr(pto.ui16(0)), mask, pto.CmpMode.EQ)
    mask_invalid = pto.por(mask_invalid, mask_zero_src0, mask)
    mask_zero_src1 = pto.vcmp(src1_abs_u16, pto.vbr(pto.ui16(0)), mask, pto.CmpMode.EQ)
    mask_invalid = pto.por(mask_invalid, mask_zero_src1, mask)

    mask_valid = pto.pnot(mask_invalid, mask)

    # Detect subnormal numbers (denormals)
    # NOTE: F16 uses LT for BOTH src0 and src1 (symmetric detection)
    # Different from F32's asymmetric EQ/LT pattern
    mask_src0_subnormal = pto.vcmp(src0_abs_u16, pto.vbr(subnormal_threshold), mask, pto.CmpMode.LT)
    mask_src0_normal = pto.pnot(mask_src0_subnormal, mask)
    src0_subnormal = pto.vmuls(src0, normalize_scale_enlarge, mask_src0_subnormal)

    mask_src1_subnormal = pto.vcmp(src1_abs_u16, pto.vbr(subnormal_threshold), mask, pto.CmpMode.LT)
    mask_src1_normal = pto.pnot(mask_src1_subnormal, mask)
    src1_subnormal = pto.vmuls(src1, normalize_scale_enlarge, mask_src1_subnormal)

    # Merge normalized subnormals with normal values
    src0_all = pto.vsel(src0, src0_subnormal, mask_src0_normal)
    src1_all = pto.vsel(src1, src1_subnormal, mask_src1_normal)

    src0_all_u16 = pto.vbitcast(src0_all, pto.ui16)
    src1_all_u16 = pto.vbitcast(src1_all, pto.ui16)

    # Standardize exponent bits (corresponds to Div754.hpp:391-401)
    src0_norm_u16 = pto.vand(src0_all_u16, pto.vbr(exponent_extractor), mask_valid)
    src1_norm_u16 = pto.vand(src1_all_u16, pto.vbr(exponent_extractor), mask_valid)

    src0_norm_u16 = pto.vadd(src0_norm_u16, pto.vbr(exponent_normalizer), mask_valid)
    src1_norm_u16 = pto.vadd(src1_norm_u16, pto.vbr(exponent_normalizer), mask_valid)

    src0_norm_f16 = pto.vbitcast(src0_norm_u16, pto.f16)
    src1_norm_f16 = pto.vbitcast(src1_norm_u16, pto.f16)
    src0_norm = pto.vsel(src0_norm_f16, src0_all, mask_valid)
    src1_norm = pto.vsel(src1_norm_f16, src1_all, mask_valid)
    return _F16NormalizedState(
        src0_norm, src1_norm, src0_abs, src1_abs, src0_all_u16,
        src1_all_u16, mask_src0_subnormal, mask_src0_normal,
        mask_src1_subnormal, mask_src1_normal, mask_valid,
    )


def _div_f16_underflow(dst_u16, dst_sign, scale, state, mask):
    """Replace f16 denormal underflow results in the u16 domain.

    Equivalent of the f32 underflow phase with f16 bias (-9 boundary).
    Mirrors Div754.hpp:443-463.
    """
    src0_norm = state.src0_norm
    src1_norm = state.src1_norm
    mask_valid = state.mask_valid

    neg9 = pto.si16(-9)
    min_denormal = pto.ui16(0x1)
    src0_norm_abs = pto.vabs(src0_norm, mask_valid)
    src1_norm_abs = pto.vabs(src1_norm, mask_valid)
    mask_norm = pto.vcmp(src0_norm_abs, src1_norm_abs, mask_valid, pto.CmpMode.LE)
    mask_underflow1 = pto.vcmp(scale, pto.vbr(neg9), mask, pto.CmpMode.EQ)
    mask_underflow1 = pto.pand(mask_underflow1, mask_valid, mask)

    z1_u16 = pto.vadd(dst_sign, pto.vbr(min_denormal), mask_underflow1)
    z2_u16 = pto.vadd(dst_sign, pto.vbr(pto.ui16(0)), mask_underflow1)

    z1_sel = pto.vsel(z2_u16, z1_u16, mask_norm)
    dst_u16_temp = pto.vsel(z1_sel, dst_u16, mask_underflow1)

    mask_underflow1_not = pto.pnot(mask_underflow1, mask)
    mask_valid_temp = pto.pand(mask_underflow1_not, mask_valid, mask)

    # Underflow handling: scale < -9 (corresponds to Div754.hpp:456-463)
    mask_underflow2 = pto.vcmp(scale, pto.vbr(neg9), mask, pto.CmpMode.LT)
    mask_underflow2 = pto.pand(mask_underflow2, mask_valid_temp, mask)

    z1_u16 = pto.vadd(dst_sign, pto.vbr(pto.ui16(0)), mask_underflow2)
    dst_u16_temp = pto.vsel(z1_u16, dst_u16_temp, mask_underflow2)

    mask_underflow2_not = pto.pnot(mask_underflow2, mask)
    mask_valid_temp = pto.pand(mask_underflow2_not, mask_valid_temp, mask)
    return dst_u16_temp, mask_valid_temp


def _div_f16_boundary(underflow_result, dst_sign, scale, state, mask):
    """Clamp f16 overflow, rescale by the exponent and propagate NaNs.

    f16 counterpart of the f32 boundary phase with bias-15 exponent
    handling. Mirrors Div754.hpp:465-501.
    """
    dst_u16_temp, mask_valid_temp = underflow_result
    src0_abs = state.src0_abs
    src1_abs = state.src1_abs

    f16_inf = pto.ui16(0x7C00)
    nan_value = pto.ui16(0x7E00)
    max_exp = pto.si16(31)
    mask_overflow1 = pto.vcmp(scale, pto.vbr(max_exp), mask, pto.CmpMode.EQ)
    mask_overflow1 = pto.pand(mask_overflow1, mask_valid_temp, mask)

    scale_adj = pto.vadds(scale, pto.si16(-1), mask_overflow1)
    scale = pto.vsel(scale_adj, scale, mask_overflow1)

    dst_f16_temp = pto.vbitcast(dst_u16_temp, pto.f16)
    z1_f16 = pto.vmuls(dst_f16_temp, pto.f16(2.0), mask_overflow1)
    dst_f16_temp = pto.vsel(z1_f16, dst_f16_temp, mask_overflow1)

    # Overflow handling: scale > 31 (corresponds to Div754.hpp:474-480)
    mask_overflow2 = pto.vcmp(scale, pto.vbr(max_exp), mask, pto.CmpMode.GT)
    mask_overflow2 = pto.pand(mask_overflow2, mask_valid_temp, mask)

    z1_u16 = pto.vadd(dst_sign, pto.vbr(f16_inf), mask_overflow2)
    dst_u16_temp = pto.vbitcast(dst_f16_temp, pto.ui16)
    dst_u16_temp = pto.vsel(z1_u16, dst_u16_temp, mask_overflow2)

    mask_overflow2_not = pto.pnot(mask_overflow2, mask)
    mask_valid_final = pto.pand(mask_overflow2_not, mask_valid_temp, mask)

    # Positive exponent handling (corresponds to Div754.hpp:482-486)
    zero_exp = pto.si16(0)
    mask_pos_exp = pto.vcmp(scale, pto.vbr(zero_exp), mask_valid_final, pto.CmpMode.GT)

    scale_u16 = pto.vbitcast(scale, pto.ui16)
    exp_shifted = pto.vshls(scale_u16, pto.i16(10), mask_pos_exp)
    exp_factor_f16 = pto.vbitcast(exp_shifted, pto.f16)

    dst_f16_temp = pto.vbitcast(dst_u16_temp, pto.f16)
    z1_f16 = pto.vmul(dst_f16_temp, exp_factor_f16, mask_pos_exp)
    dst_f16_temp = pto.vsel(z1_f16, dst_f16_temp, mask_pos_exp)

    # Negative exponent handling (corresponds to Div754.hpp:488-493)
    mask_pos_exp_not = pto.pnot(mask_pos_exp, mask_valid_final)

    # Value 0x0200 = Float16 with exp=0, mantissa bit9=1 (used for shift calculation)
    shr_base = pto.i16(512)  # 0x0200
    scale_abs = pto.vabs(scale, mask_pos_exp_not)

    shr_base_vec = pto.vdup(shr_base, mask_pos_exp_not)
    shr_base_i16 = pto.vbitcast(shr_base_vec, pto.si16)
    shr_factor_i16 = pto.vshr(shr_base_i16, scale_abs, mask_pos_exp_not)
    shr_factor_f16 = pto.vbitcast(shr_factor_i16, pto.f16)

    z1_f16 = pto.vmul(dst_f16_temp, shr_factor_f16, mask_pos_exp_not)
    dst_f16_temp = pto.vsel(z1_f16, dst_f16_temp, mask_pos_exp_not)

    # NaN propagation (corresponds to Div754.hpp:495-501)
    mask_nan_src0 = pto.vcmp(src0_abs, src0_abs, mask, pto.CmpMode.NE)
    mask_nan_src1 = pto.vcmp(src1_abs, src1_abs, mask, pto.CmpMode.NE)
    mask_nan = pto.por(mask_nan_src0, mask_nan_src1, mask)

    nan_vec = pto.vbr(nan_value)
    nan_f16_vec = pto.vbitcast(nan_vec, pto.f16)
    dst_final = pto.vsel(nan_f16_vec, dst_f16_temp, mask_nan)
    return dst_final


def _div_ieee754_f16_impl(src0, src1, mask):
    """Complete IEEE 754 float16 high-precision division with subnormal handling.

    Follows pto-isa Div754.hpp:291-502 (DivIEEE754HalfImpl).

    Key differences from F32 implementation:
    - Uses LT for both src0/src1 subnormal detection (symmetric, not EQ/LT like F32)
    - Normalization factor: 2^10 (not 2^23 for F32)
    - Exponent bias: 15 (not 127 for F32)
    - Exponent shift: 10 bits (not 23 for F32)
    - Direct vdiv call (no three-candidate search)
    """
    state = _div_f16_normalize(src0, src1, mask)

    f16_inf = pto.ui16(0x7C00)
    sign_extractor = pto.ui16(0x8000)
    normalize_scale_reduce = pto.f16(0.0009765625)
    normalize_scale_enlarge = pto.f16(1024.0)
    dst = pto.vdiv(state.src0_norm, state.src1_norm, mask)

    # Subnormal dividend, normal divisor: scale down result
    # Corresponds to Div754.hpp:408-412
    mask0 = pto.pand(state.mask_src0_subnormal, state.mask_src1_normal, mask)
    z1 = pto.vmuls(dst, normalize_scale_reduce, mask0)
    dst = pto.vsel(z1, dst, mask0)

    # Normal dividend, subnormal divisor: scale up result
    # Corresponds to Div754.hpp:414-419
    mask0 = pto.pand(state.mask_src0_normal, state.mask_src1_subnormal, mask)
    z1 = pto.vmuls(dst, normalize_scale_enlarge, mask0)
    dst = pto.vsel(z1, dst, mask0)

    # Preserve sign for overflow/underflow handling
    dst_u16 = pto.vbitcast(dst, pto.ui16)
    dst_sign = pto.vand(dst_u16, pto.vbr(sign_extractor), mask)

    # Extract exponent bits (corresponds to Div754.hpp:428-439)
    src0_exponent = pto.vand(state.src0_all_u16, pto.vbr(f16_inf), mask)
    src1_exponent = pto.vand(state.src1_all_u16, pto.vbr(f16_inf), mask)

    src0_exp_shifted = pto.vshrs(src0_exponent, pto.i16(10), mask)
    src1_exp_shifted = pto.vshrs(src1_exponent, pto.i16(10), mask)

    src0_exp_i16 = pto.vbitcast(src0_exp_shifted, pto.si16)
    src1_exp_i16 = pto.vbitcast(src1_exp_shifted, pto.si16)

    scale = pto.vsub(src0_exp_i16, src1_exp_i16, mask)
    scale = pto.vadds(scale, pto.si16(15), mask)

    underflow_result = _div_f16_underflow(
        dst_u16, dst_sign, scale, state, mask,
    )
    return _div_f16_boundary(
        underflow_result, dst_sign, scale, state, mask,
    )
