# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Correctly rounded scalar f32 division using PR #1254 integer long division."""

from ptodsl import pto
from ptodsl._surface_values import wrap_surface_value


def _select(condition, yes, no):
    # Generic select returns signless integers; preserve helper signedness.
    return pto.bitcast(pto.select(condition, yes, no), yes.type)


def _u32(value):
    return wrap_surface_value(pto.ui32(value))


def _i32(value):
    return wrap_surface_value(pto.i32(value))


def _clz_fraction(value):
    count = _u32(0)
    for check, shift in ((16, 16), (24, 8), (28, 4), (30, 2), (31, 1)):
        empty = (value >> _u32(check)) == _u32(0)
        count = _select(empty, count + _u32(shift), count)
        value = _select(empty, value << _u32(shift), value)
    return count


def _decode(bits):
    exponent = (bits >> _u32(23)) & _u32(255)
    fraction = bits & _u32(0x7fffff)
    shift = _clz_fraction(fraction) - _u32(8)
    subnormal = (exponent == _u32(0)) & (fraction != _u32(0))
    mantissa = _select(subnormal, fraction << shift, fraction | _u32(0x800000))
    power = _select(subnormal, _i32(-149) - pto.bitcast(shift, pto.i32),
                       pto.bitcast(exponent, pto.i32) - _i32(150))
    special = (exponent == _u32(255)) | ((bits & _u32(0x7fffffff)) == _u32(0))
    return mantissa, power, special


def _quotient(mantissa, divisor):
    digits = [_u32(0)]
    for shift in (23, 16, 9, 2):
        digits.append((mantissa >> _u32(shift)) & _u32(127))
    digits.extend([(mantissa & _u32(3)) << _u32(5), _u32(0), _u32(0), _u32(0)])
    remainder = _u32(0)
    quotient = _u32(0)
    for digit in digits:
        remainder = (remainder << _u32(7)) | digit
        part = pto.div(remainder, divisor)
        remainder = remainder - part * divisor
        quotient = (quotient << _u32(7)) | part
    return quotient, remainder


def _safe_shift(amount):
    return _select(amount > _u32(31), _u32(31), amount)


def _round_quotient(raw, remainder, shift):
    # Retain all discarded bits and the exact remainder for the tie decision.
    sig = raw >> _safe_shift(shift)
    guard = (raw >> _safe_shift(shift - _u32(1))) & _u32(1)
    mask = (_u32(1) << _safe_shift(shift - _u32(1))) - _u32(1)
    sticky = ((raw & mask) != _u32(0)) | (remainder != _u32(0))
    increment = (guard != _u32(0)) & (sticky | ((sig & _u32(1)) != _u32(0)))
    return sig + _select(increment, _u32(1), _u32(0))


def div_f32_soft(lhs, rhs):
    """RNE for finite nonzero f32 operands; native handling for special inputs."""
    a = pto.bitcast(lhs, pto.ui32)
    b = pto.bitcast(rhs, pto.ui32)
    ma, ea, special_a = _decode(a)
    mb, eb, special_b = _decode(b)
    less = ma < mb
    exponent = ea - eb - _select(less, _i32(1), _i32(0))
    ma = _select(less, ma << _u32(1), ma)
    raw, remainder = _quotient(ma, mb)
    sig = _round_quotient(raw, remainder, _u32(3))
    carry = sig == _u32(0x1000000)
    sig = _select(carry, sig >> _u32(1), sig)
    normal_exponent = exponent + _select(carry, _i32(1), _i32(0))
    normal = (pto.bitcast(normal_exponent + _i32(127), pto.ui32) << _u32(23))
    normal = normal | (sig & _u32(0x7fffff))
    normal = _select(normal_exponent >= _i32(128), _u32(0x7f800000), normal)
    # Use the original exponent: normal rounding must not double-round subnormals.
    shift = pto.bitcast(_i32(-123) - exponent, pto.ui32)
    subnormal = _round_quotient(raw, remainder, shift)
    subnormal = _select(exponent < _i32(-150), _u32(0), subnormal)
    bits = _select(exponent >= _i32(-126), normal, subnormal)
    bits = bits | ((a ^ b) & _u32(0x80000000))
    return _select(special_a | special_b, lhs / rhs, pto.bitcast(bits, pto.f32))
