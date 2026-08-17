# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""A5 SIMT scalar trigonometric software implementations."""

from ptodsl import pto
from ptodsl import scalar


def _reduce_angle(value):
    # Reduce to [-pi/4, pi/4] with a nearest-quadrant reduction.  The
    # polynomial below is much more accurate on this interval than a full
    # turn reduction, which matters for the Box-Muller angle range.
    quadrants = pto.round(value * pto.f32(0.6366197723675814))
    reduced = value - quadrants * pto.f32(1.5707963267948966)
    quadrant = quadrants - pto.f32(4.0) * pto.round(quadrants * pto.f32(0.25))
    quadrant = scalar.select(quadrant < pto.f32(0.0), quadrant + pto.f32(4.0), quadrant)
    return reduced, quadrant


def sin_f32_soft(value):
    """Approximate sin(value) for an f32 SIMT scalar."""
    x, quadrant = _reduce_angle(value)
    x2 = x * x
    # sin(x) = x * (1 + x^2 * P(x^2)), through the x^13 term.
    p = pto.f32(-1.0 / 6227020800.0)
    p = pto.fma(p, x2, pto.f32(1.0 / 39916800.0))
    p = pto.fma(p, x2, pto.f32(-1.0 / 362880.0))
    p = pto.fma(p, x2, pto.f32(1.0 / 5040.0))
    p = pto.fma(p, x2, pto.f32(-1.0 / 120.0))
    p = pto.fma(p, x2, pto.f32(1.0 / 6.0))
    sin_x = x - x * x2 * p
    cos_x = _cos_polynomial(x, x2)
    neg_sin_x = pto.f32(0.0) - sin_x
    neg_cos_x = pto.f32(0.0) - cos_x
    result = scalar.select(quadrant == pto.f32(1.0), cos_x, sin_x)
    result = scalar.select(quadrant == pto.f32(2.0), neg_sin_x, result)
    return scalar.select(quadrant == pto.f32(3.0), neg_cos_x, result)


def _cos_polynomial(x, x2):
    # cos(x) = 1 + x^2 * P(x^2), through the x^14 term.
    p = pto.f32(-1.0 / 87178291200.0)
    p = pto.fma(p, x2, pto.f32(1.0 / 479001600.0))
    p = pto.fma(p, x2, pto.f32(-1.0 / 3628800.0))
    p = pto.fma(p, x2, pto.f32(1.0 / 40320.0))
    p = pto.fma(p, x2, pto.f32(-1.0 / 720.0))
    p = pto.fma(p, x2, pto.f32(1.0 / 24.0))
    p = pto.fma(p, x2, pto.f32(-1.0 / 2.0))
    return pto.fma(p, x2, pto.f32(1.0))


def cos_f32_soft(value):
    """Approximate cos(value) for an f32 SIMT scalar."""
    x, quadrant = _reduce_angle(value)
    x2 = x * x
    cos_x = _cos_polynomial(x, x2)
    sin_x = x - x * x2 * _sin_polynomial(x2)
    neg_sin_x = pto.f32(0.0) - sin_x
    neg_cos_x = pto.f32(0.0) - cos_x
    result = scalar.select(quadrant == pto.f32(1.0), neg_sin_x, cos_x)
    result = scalar.select(quadrant == pto.f32(2.0), neg_cos_x, result)
    return scalar.select(quadrant == pto.f32(3.0), sin_x, result)


def _sin_polynomial(x2):
    p = pto.f32(-1.0 / 6227020800.0)
    p = pto.fma(p, x2, pto.f32(1.0 / 39916800.0))
    p = pto.fma(p, x2, pto.f32(-1.0 / 362880.0))
    p = pto.fma(p, x2, pto.f32(1.0 / 5040.0))
    p = pto.fma(p, x2, pto.f32(-1.0 / 120.0))
    return pto.fma(p, x2, pto.f32(1.0 / 6.0))
