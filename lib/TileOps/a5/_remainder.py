# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Shared PTODSL implementations for remainder/fmod-style TileOps."""

from ptodsl import pto
import ptodsl.tilelib as tilelib

from ._common import ub_row_major_constraints
from ._elementwise import (
    _elementwise_template_decorator,
    emit_binary_1d,
    emit_binary_2d,
    emit_scalar_binary_1d,
    emit_scalar_binary_2d,
    traversal_metadata,
)


# The signed and unsigned 16/32-bit integer types share one lowering path; the
# scalar operand no longer rejects unsigned element types during verification,
# so the scalar tables register ui16/ui32 as well. ``FMODS_DTYPES`` additionally
# covers bf16 because the A5 ``pto.tfmods`` verifier accepts it (the binary
# ``pto.tfmod`` verifier does not, so ``FMOD_DTYPES`` stays without bf16).
# ``REMS_DTYPES`` excludes bf16 because the A5 ``pto.trems`` verifier only
# accepts 16/32-bit integers and f16/f32. The bf16 HP helper implements only
# Z (fmod); adding bf16 remainder requires TFmodRemHPPost<false>'s sign fixup.
FMOD_DTYPES = [(dtype,) * 3 for dtype in ("f32", "f16", "i16", "ui16", "i32", "ui32")]
FMODS_DTYPES = [(dtype,) * 3 for dtype in ("f32", "f16", "i32", "i16", "ui16", "ui32", "bf16")]
REM_DTYPES = [(dtype,) * 4 for dtype in ("f32", "f16", "i16", "ui16", "i32", "ui32")]
REMS_DTYPES = [(dtype,) * 4 for dtype in ("f32", "f16", "i16", "i32", "ui16", "ui32")]


def _remainder(lhs, rhs, mask, *, round_mode, dtype):
    quotient = pto.vdiv(lhs, rhs, mask)
    if str(dtype) in {"f16", "bf16", "f32"}:
        quotient = pto.vtrc(quotient, mask, rnd=round_mode)
    product = pto.vmul(quotient, rhs, mask)
    result = pto.vsub(lhs, product, mask)
    # ``F`` implements the legacy floor-style remainder (sign follows the
    # divisor).  ``Z`` is fmod and must keep the dividend sign, so the
    # correction would be incorrect for negative operands.
    if str(dtype) == "f32" and str(round_mode).upper() != "Z":
        sign_diff_mask = pto.vcmps(
            pto.vmul(rhs, result, mask), pto.f32(0.0), mask, pto.CmpMode.LT
        )
        corrected = pto.vadd(result, rhs, sign_diff_mask)
        result = pto.vsel(corrected, result, sign_diff_mask)
    return result


_F32_SIGN_MASK = 0x80000000
_F32_SIGN_SCALE = -2.0
_F32_FMOD_MAX = 3.4028234663852886e38
_F32_FMOD_SUBNORMAL = 1.1754943508222875e-38
_F32_FMOD_TWO_NEG_24 = 2.0 ** -24
_F32_FMOD_NAN = float("nan")
_F32_TWO_POW_24 = 2.0 ** 24
_F32_TWO_POW_48 = 2.0 ** 48
_F32_TWO_POW_72 = 2.0 ** 72
_F32_TWO_POW_96 = 2.0 ** 96
_F32_TWO_POW_120 = 2.0 ** 120
_F32_FMOD_SCALE1 = (
    _F32_TWO_POW_24,
    _F32_TWO_POW_24,
    _F32_TWO_POW_48,
    _F32_TWO_POW_72,
    _F32_TWO_POW_96,
    _F32_TWO_POW_120,
    _F32_TWO_POW_120,
    _F32_TWO_POW_120,
    _F32_TWO_POW_120,
    _F32_TWO_POW_120,
    _F32_TWO_POW_120,
)
_F32_FMOD_SCALE2 = (
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    _F32_TWO_POW_24,
    _F32_TWO_POW_48,
    _F32_TWO_POW_72,
    _F32_TWO_POW_96,
    _F32_TWO_POW_120,
)


def _f32_sign_bit(value, mask):
    """Return 1.0 for a set sign bit and 0.0 otherwise (covers -0.0)."""
    bits = pto.vbitcast(value, pto.ui32)
    negative = pto.vcmp(
        bits,
        pto.vdup(pto.ui32(_F32_SIGN_MASK), mask),
        mask,
        pto.CmpMode.GE,
    )
    return pto.vsel(
        pto.vdup(pto.f32(1.0), mask),
        pto.vdup(pto.f32(0.0), mask),
        negative,
    )


def _f32_sign_multiplier(value, mask):
    sign = _f32_sign_bit(value, mask)
    return pto.vadds(
        pto.vmuls(sign, pto.f32(_F32_SIGN_SCALE), mask),
        pto.f32(1.0),
        mask,
    )


def _f32_solve_scale(dst, divisor, iteration, mask):
    """One scaling pass of the A5 high-precision remainder algorithm."""
    scale1 = pto.f32(_F32_FMOD_SCALE1[iteration - 1])
    scale2 = pto.f32(_F32_FMOD_SCALE2[iteration - 1])
    if iteration == 1:
        subnormal = pto.vcmps(
            divisor, pto.f32(_F32_FMOD_SUBNORMAL), mask, pto.CmpMode.LE
        )
        scaled_divisor = pto.vmuls(divisor, scale1, mask)
        divisor = pto.vsel(scaled_divisor, divisor, subnormal)
        scaled_dst = pto.vmuls(dst, scale1, mask)
        dst = pto.vsel(scaled_dst, dst, subnormal)
    else:
        divisor = pto.vmuls(divisor, scale1, mask)
        if iteration > 5:
            divisor = pto.vmuls(divisor, scale2, mask)

    quotient = pto.vdiv(dst, divisor, mask)
    quotient = pto.vtrc(quotient, mask, rnd="Z")
    if iteration != 1:
        divisor = pto.vmins(divisor, pto.f32(_F32_FMOD_MAX), mask)
    quotient = pto.vneg(quotient, mask)
    dst = pto.vmula(dst, quotient, divisor, mask)
    if iteration == 1:
        sign = _f32_sign_bit(dst, mask)
        dst = pto.vadd(dst, pto.vmul(sign, divisor, mask), mask)
    return dst


def _f32_restore_fmod_zero(dividend, divisor, result, mask):
    """Preserve signed zero only for a nonzero, non-NaN divisor."""
    zero = pto.f32(0.0)
    zero_mask = pto.pand(
        pto.vcmps(dividend, zero, mask, pto.CmpMode.EQ),
        pto.pand(
            pto.pnot(pto.vcmps(divisor, zero, mask, pto.CmpMode.EQ), mask),
            pto.vcmp(divisor, divisor, mask, pto.CmpMode.EQ),
            mask,
        ),
        mask,
    )
    return pto.vsel(dividend, result, zero_mask)


def _f32_fmod_high_precision(dividend, divisor, mask):
    """Compute fmod without the overflowing x/y quotient.

    This is the A5 ``TFmodRemHP`` scaling sequence: eleven passes scale the
    divisor up before dividing, so the quotient always fits in f32 even when
    bf16's full exponent range is used. This retains the upstream algorithm's
    f32 rounding limitations, rather than guaranteeing correctly rounded IEEE
    fmod for every finite input. Large dividends with non-power-of-two
    divisors can lose remainder accuracy during the scaled reductions.
    """
    result = pto.vabs(dividend, mask)
    abs_divisor = pto.vabs(divisor, mask)
    for iteration in range(len(_F32_FMOD_SCALE1), 0, -1):
        result = _f32_solve_scale(result, abs_divisor, iteration, mask)

    result = pto.vmul(result, _f32_sign_multiplier(dividend, mask), mask)

    subnormal = pto.vcmps(
        abs_divisor, pto.f32(_F32_FMOD_SUBNORMAL), mask, pto.CmpMode.LE
    )
    result = pto.vsel(
        pto.vmuls(result, pto.f32(_F32_FMOD_TWO_NEG_24), subnormal),
        result,
        subnormal,
    )

    pos_inf = pto.f32(float("inf"))
    neg_inf = pto.f32(float("-inf"))
    divisor_inf = pto.por(
        pto.vcmps(divisor, pos_inf, mask, pto.CmpMode.EQ),
        pto.vcmps(divisor, neg_inf, mask, pto.CmpMode.EQ),
        mask,
    )
    result = pto.vsel(dividend, result, divisor_inf)

    dividend_inf = pto.por(
        pto.vcmps(dividend, pos_inf, mask, pto.CmpMode.EQ),
        pto.vcmps(dividend, neg_inf, mask, pto.CmpMode.EQ),
        mask,
    )
    result = pto.vsel(
        pto.vdup(pto.f32(_F32_FMOD_NAN), mask),
        result,
        pto.pand(dividend_inf, divisor_inf, mask),
    )

    return _f32_restore_fmod_zero(dividend, divisor, result, mask)


def _bf16_scalar_remainder(value, scalar, mask, *, round_mode):
    """Widen bf16 to f32, run the high-precision remainder, narrow back.

    A5 bisheng rejects the bf16 HiVM divide intrinsic, and the naive
    ``x - trunc(x / y) * y`` widening overflows f32 for the upper bf16
    exponent range. The high-precision scaling sequence avoids that quotient
    overflow. ``vcvt`` is zero-inactive, so lanes outside the source mask
    contribute zero and the final masked store discards them.

    Only round_mode="Z" (fmod) is implemented. A floor-remainder variant needs
    the sign correction from the upstream TFmodRemHPPost<false> first.
    """

    full_mask_b16 = pto.pset_b16(pto.PAT.ALL)
    full_mask_f32 = pto.pset_b32(pto.PAT.ALL)
    scalar_vec = pto.vcvt(
        pto.vbr(scalar), pto.f32, full_mask_b16, part=pto.VcvtPartMode.EVEN
    )
    even = _f32_fmod_high_precision(
        pto.vcvt(value, pto.f32, mask, part=pto.VcvtPartMode.EVEN),
        scalar_vec,
        full_mask_f32,
    )
    odd = _f32_fmod_high_precision(
        pto.vcvt(value, pto.f32, mask, part=pto.VcvtPartMode.ODD),
        scalar_vec,
        full_mask_f32,
    )
    even_bf16 = pto.vcvt(
        even,
        pto.bf16,
        full_mask_f32,
        rnd=round_mode,
        sat=pto.VcvtSatMode.NOSAT,
        part=pto.VcvtPartMode.EVEN,
    )
    odd_bf16 = pto.vcvt(
        odd,
        pto.bf16,
        full_mask_f32,
        rnd=round_mode,
        sat=pto.VcvtSatMode.NOSAT,
        part=pto.VcvtPartMode.ODD,
    )
    return pto.vor(even_bf16, odd_bf16, full_mask_b16)


def _scalar_remainder(lhs, scalar, mask, *, round_mode, dtype):
    if str(dtype) == "bf16":
        return _bf16_scalar_remainder(
            lhs, scalar, mask, round_mode=round_mode
        )
    scalar_vec = pto.vbr(scalar)
    quotient = pto.vdiv(lhs, scalar_vec, mask)
    if str(dtype) in {"f16", "f32"}:
        quotient = pto.vtrc(quotient, mask, rnd=round_mode)
    product = pto.vmuls(quotient, scalar, mask)
    return pto.vsub(lhs, product, mask)


def register_binary_remainder(*, op, name, dtypes, round_mode, has_tmp=False,
                              traversal="2d", priority=None,
                              candidate_id=None):
    if traversal not in {"1d", "2d"}:
        raise ValueError(
            f"unsupported remainder traversal {traversal!r}; "
            "expected '1d' or '2d'"
        )
    loop_depth, priority, candidate_id = traversal_metadata(
        traversal,
        priority=priority,
        candidate_id=candidate_id,
    )
    constraints = ub_row_major_constraints("src0", "src1", "dst")
    if has_tmp:
        constraints = ub_row_major_constraints("src0", "src1", "tmp", "dst")
        if traversal == "1d":
            constraints.append(
                tilelib.require_elementwise_1d(
                    "src0",
                    "src1",
                    "tmp",
                    "dst",
                )
            )

        @_elementwise_template_decorator(
            op=op, name=name, dtypes=dtypes, constraints=constraints,
            priority=priority, candidate_id=candidate_id,
            loop_depth=loop_depth, tags=("elementwise", "remainder"),
        )
        def template(src0: pto.Tile, src1: pto.Tile, tmp: pto.Tile, dst: pto.Tile):
            _ = tmp
            _emit_binary(src0, src1, dst, round_mode, traversal)

        return template

    if traversal == "1d":
        constraints.append(
            tilelib.require_elementwise_1d("src0", "src1", "dst")
        )

    @_elementwise_template_decorator(
        op=op, name=name, dtypes=dtypes, constraints=constraints,
        priority=priority, candidate_id=candidate_id,
        loop_depth=loop_depth, tags=("elementwise", "remainder"),
    )
    def template(src0: pto.Tile, src1: pto.Tile, dst: pto.Tile):
        _emit_binary(src0, src1, dst, round_mode, traversal)

    return template


def _emit_binary(src0, src1, dst, round_mode, traversal):
    dtype = dst.dtype

    def remainder(lhs, rhs, mask):
        return _remainder(
            lhs,
            rhs,
            mask,
            round_mode=round_mode,
            dtype=dtype,
        )

    if traversal == "1d":
        emit_binary_1d(src0, src1, dst, remainder)
    else:
        emit_binary_2d(src0, src1, dst, remainder)


def register_scalar_remainder(*, op, name, dtypes, round_mode, has_tmp=False,
                              traversal="2d", priority=None,
                              candidate_id=None):
    if traversal not in {"1d", "2d"}:
        raise ValueError(
            f"unsupported remainder traversal {traversal!r}; "
            "expected '1d' or '2d'"
        )
    loop_depth, priority, candidate_id = traversal_metadata(
        traversal,
        priority=priority,
        candidate_id=candidate_id,
    )
    constraints = ub_row_major_constraints("src", "dst")
    if has_tmp:
        constraints = ub_row_major_constraints("src", "tmp", "dst")
        if traversal == "1d":
            constraints.append(
                tilelib.require_elementwise_1d("src", "tmp", "dst")
            )

        @_elementwise_template_decorator(
            op=op, name=name, dtypes=dtypes, constraints=constraints,
            priority=priority, candidate_id=candidate_id,
            loop_depth=loop_depth, tags=("elementwise", "scalar", "remainder"),
        )
        def template(src: pto.Tile, scalar, tmp: pto.Tile, dst: pto.Tile):
            _ = tmp
            _emit_scalar(src, scalar, dst, round_mode, traversal)

        return template

    if traversal == "1d":
        constraints.append(tilelib.require_elementwise_1d("src", "dst"))

    @_elementwise_template_decorator(
        op=op, name=name, dtypes=dtypes, constraints=constraints,
        priority=priority, candidate_id=candidate_id,
        loop_depth=loop_depth, tags=("elementwise", "scalar", "remainder"),
    )
    def template(src: pto.Tile, scalar, dst: pto.Tile):
        _emit_scalar(src, scalar, dst, round_mode, traversal)

    return template


def _emit_scalar(src, scalar, dst, round_mode, traversal):
    dtype = dst.dtype

    def remainder(value, scalar_value, mask):
        return _scalar_remainder(
            value,
            scalar_value,
            mask,
            round_mode=round_mode,
            dtype=dtype,
        )

    if traversal == "1d":
        emit_scalar_binary_1d(src, scalar, dst, remainder)
    else:
        emit_scalar_binary_2d(src, scalar, dst, remainder)


__all__ = [
    "FMOD_DTYPES",
    "FMODS_DTYPES",
    "REM_DTYPES",
    "REMS_DTYPES",
    "register_binary_remainder",
    "register_scalar_remainder",
]
