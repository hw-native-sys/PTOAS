# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from __future__ import annotations

from dataclasses import dataclass, field

from ptodsl import pto
import ptodsl.tilelib as tilelib


def _ceil_division(n, d):
    return (n + d - 1) // d


def _quant_tiles(operand_memory_spaces, **_):
    return all(space in {"ub", "vec"} for space in operand_memory_spaces)

@tilelib.tile_template(
    op="pto.tquant",
    target="a5",
    name="template_tquant_int8_sym",
    dtypes=[("f32", "f32", "i8")],
    iteration_axis="none",
    op_engine="vector",
    op_class="elementwise",
    constraints=[_quant_tiles],
    id=0,
    loop_depth=2,
    is_post_update=False,
    tags=("quant", "int8", "sym"),
)
def template_tquant_int8_sym(
    src: pto.Tile,
    scale: pto.Tile,
    dst: pto.Tile,
):
    """INT8_SYM: dst = round(src * scale) → i8."""
    valid_rows, valid_cols = dst.valid_shape
    lanes_f32 = pto.elements_per_vreg(pto.f32)
    scale_ptr = scale.as_ptr()

    for row in range(0, valid_rows, 1):
        remained = valid_cols
        remained_b16 = valid_cols * 2
        remained_b8 = valid_cols * 4
        for col in range(0, valid_cols, lanes_f32):
            mask, remained = pto.make_mask(pto.f32, remained)
            mask_b16, remained_b16 = pto.make_mask(pto.f16, remained_b16)
            mask_b8, remained_b8 = pto.make_mask(pto.i8, remained_b8)
            src_v = pto.vlds(src[row, col:])
            scale_v = pto.vlds(scale_ptr, row, dist="BRC_B32")
            mul_v = pto.vmul(src_v, scale_v, mask)
            s32_v = pto.vcvt(mul_v, pto.i32, mask, rnd=pto.VcvtRoundMode.R, sat=pto.VcvtSatMode.SAT)
            f32_v = pto.vcvt(s32_v, pto.f32, mask, rnd=pto.VcvtRoundMode.R)
            f16_v = pto.vcvt(f32_v, pto.f16, mask,
                             rnd=pto.VcvtRoundMode.R, sat=pto.VcvtSatMode.SAT,
                             part=pto.VcvtPartMode.EVEN)
            i8_v = pto.vcvt(f16_v, pto.i8, mask_b16,
                            rnd=pto.VcvtRoundMode.R, sat=pto.VcvtSatMode.SAT,
                            part=pto.VcvtPartMode.EVEN)
            pto.vsts(i8_v, dst[row, col:], mask, dist=pto.VStoreDist.PK4_B32)

@tilelib.tile_template(
    op="pto.tquant",
    target="a5",
    name="template_tquant_int8_asym",
    dtypes=[("f32", "f32", "f32", "ui8")],
    iteration_axis="none",
    op_engine="vector",
    op_class="elementwise",
    constraints=[_quant_tiles],
    id=1,
    loop_depth=2,
    is_post_update=False,
    tags=("quant", "int8", "asym"),
)
def template_tquant_int8_asym(
    src: pto.Tile,
    scale: pto.Tile,
    offset: pto.Tile,
    dst: pto.Tile,
):
    """INT8_ASYM: dst = round(src * scale + offset) → ui8."""
    valid_rows, valid_cols = dst.valid_shape
    lanes_f32 = pto.elements_per_vreg(pto.f32)
    scale_ptr = scale.as_ptr()
    offset_ptr = offset.as_ptr()

    for row in range(0, valid_rows, 1):
        remained = valid_cols
        remained_b16 = valid_cols * 2
        remained_b8 = valid_cols * 4
        for col in range(0, valid_cols, lanes_f32):
            mask, remained = pto.make_mask(pto.f32, remained)
            mask_b16, remained_b16 = pto.make_mask(pto.f16, remained_b16)
            mask_b8, remained_b8 = pto.make_mask(pto.ui8, remained_b8)
            src_v = pto.vlds(src[row, col:])
            scale_v = pto.vlds(scale_ptr, row, dist="BRC_B32")
            offset_v = pto.vlds(offset_ptr, row, dist="BRC_B32")
            acc_v = pto.vadd(
                pto.vmul(src_v, scale_v, mask),
                offset_v, mask,
            )
            s32_v = pto.vcvt(acc_v, pto.i32, mask, rnd=pto.VcvtRoundMode.R, sat=pto.VcvtSatMode.SAT)
            f32_v = pto.vcvt(s32_v, pto.f32, mask, rnd=pto.VcvtRoundMode.R)
            f16_v = pto.vcvt(f32_v, pto.f16, mask,
                             rnd=pto.VcvtRoundMode.R, sat=pto.VcvtSatMode.SAT,
                             part=pto.VcvtPartMode.EVEN)
            ui8_v = pto.vcvt(f16_v, pto.ui8, mask_b16,
                             rnd=pto.VcvtRoundMode.R, sat=pto.VcvtSatMode.SAT,
                             part=pto.VcvtPartMode.EVEN)
            pto.vsts(ui8_v, dst[row, col:], mask, dist=pto.VStoreDist.PK4_B32)


# ── MX 量化（ND/DN templates + helpers）──

def _mx_nd_2d_tiles(operand_memory_spaces, grp_axis=None, **_):
    """ND (flat + Exp2D auto-dispatch) template constraint. 拒绝 DN（grp_axis=0/axis0）。"""
    if not all(space in {"ub", "vec"} for space in operand_memory_spaces):
        return False
    if grp_axis is not None and str(grp_axis) in ("0", "axis0"):
        return False
    return True
_MX_TAGS = ("quant", "mx")


@dataclass(frozen=True)
class _AlgTag:
    """Base for algorithm tags (OcpF8E4M3Alg, etc.)."""
    name: str


OcpF8E4M3Alg = _AlgTag("OcpF8E4M3Alg")
OcpF4E2M1Alg = _AlgTag("OcpF4E2M1Alg")
NvF8E4M3Alg = _AlgTag("NvF8E4M3Alg")
NvF4E2M1Alg = _AlgTag("NvF4E2M1Alg")


@dataclass(frozen=True)
class _OcpMxFp8E4M3Spec:
    maxExp: int = 0x0400
    expNan: int = 0x00FF
    b16Nan: int = 0x7F81
    f32Emax: int = 8


@dataclass(frozen=True)
class _NvMxFp8E4M3Spec:
    descaleMultiplier: float = 1.0 / 448.0
    b16SpecialScaleBits: int = 0x7F81
    f32SpecialScaleBits: int = 0x7FC00000


_OCP_MXFP8_SPEC = _OcpMxFp8E4M3Spec()
_NV_MXFP8_SPEC = _NvMxFp8E4M3Spec()


@dataclass
class _F32OcpQuantCtx:
    vb32_exp_mask: object = None
    vb32_mantissa_mask: object = None
    vb32_b8_nan: object = None
    vb32_f32_nan: object = None
    vb32_b8_emax: object = None
    vb32_exp_max: object = None
    vb32_recip_min_scale: object = None
    vb32_zero: object = None
    kExpMask: int = 0x7F800000
    kMantissaMask: int = 0x007FFFFF
    kExpMax: int = 0xFE
    kF32Nan: int = 0x7FC00000
    kRecipMinScale: int = 0x7F000000
    kExpNan: int = field(default_factory=lambda: _OCP_MXFP8_SPEC.expNan)
    f32Emax: int = field(default_factory=lambda: _OCP_MXFP8_SPEC.f32Emax)
    shr: int = 23


@dataclass
class _F32NvQuantCtx:
    vb32_exp_mask: object = None
    vb32_mantissa_mask: object = None
    vb32_exp_nan: object = None
    vb32_exp_max: object = None
    vb32_special_scale: object = None
    vb32_min_rcp: object = None
    descaleMultiplier: float = field(default_factory=lambda: _NV_MXFP8_SPEC.descaleMultiplier)
    shr: int = 23
 
def _abs_reduce_max_naive(src_ptr, max_ptr, total_count, vl_count, lanes_f32, preg_lower32, preg_upper32, src_base=0, max_base=0):
    zero_v = pto.vbr(pto.f32(0))
    remained = total_count
    for i in range(vl_count):
        preg, remained = pto.make_mask(pto.f32, remained)
        v_in = pto.vlds(src_ptr, src_base + i * lanes_f32)
        v_abs = pto.vabs(v_in, preg)
        v_sel = pto.vsel(v_abs, zero_v, preg)
        v_max_0 = pto.vcmax(v_sel, preg_lower32)
        v_max_1 = pto.vcmax(v_sel, preg_upper32)
        pto.vsts(v_max_0, max_ptr, max_base + 2 * i, preg, dist=pto.VStoreDist._1PT_B32)
        pto.vsts(v_max_1, max_ptr, max_base + 2 * i + 1, preg, dist=pto.VStoreDist._1PT_B32)

def _abs_reduce_max_f32_opt(src_ptr, max_ptr, vl_count, lanes_f32, total_count):
    """Input must be multiple of 256 elements."""
    preg_lower8 = pto.pset_b32(pto.PAT.VL8)
    remained = total_count
    for i in range(vl_count // 4):
        preg_vl0, remained = pto.make_mask(pto.f32, remained)
        preg_vl1, remained = pto.make_mask(pto.f32, remained)
        preg_vl2, remained = pto.make_mask(pto.f32, remained)
        preg_vl3, remained = pto.make_mask(pto.f32, remained)
        off = i * 4 * lanes_f32
        v_in_1, v_in_2 = pto.vldsx2(src_ptr, off, "DINTLV_B32")
        v_in_3, v_in_4 = pto.vldsx2(src_ptr, off + 128, "DINTLV_B32")
        v_in_1 = pto.vabs(v_in_1, preg_vl0)
        v_in_3 = pto.vabs(v_in_3, preg_vl2)
        v_di_1, v_di_2 = pto.vdintlv(v_in_1, v_in_3)
        v_in_2 = pto.vabs(v_in_2, preg_vl1)
        v_in_4 = pto.vabs(v_in_4, preg_vl3)
        v_di_3, v_di_4 = pto.vdintlv(v_in_2, v_in_4)
        v_max_0 = pto.vmax(v_di_1, v_di_2, preg_vl0)
        v_max_1 = pto.vmax(v_di_3, v_di_4, preg_vl1)
        v_max = pto.vmax(v_max_0, v_max_1, preg_vl0)
        v_gp_max = pto.vcgmax(v_max, preg_vl0)
        pto.vsts(v_gp_max, max_ptr, i * 8, preg_lower8, dist=pto.VStoreDist.NORM_B32)

def _abs_reduce_max_f32_opt_largesizes(src_ptr, max_ptr, vl_count, lanes_f32, total_count):
    """Input must be multiple of 2K elements."""
    preg_all = pto.pset_b32(pto.PAT.ALL)
    remained = total_count
    for i in range(vl_count // 32):
        ureg_max = pto.init_align()
        for j in range(8):
            preg_vl0, remained = pto.make_mask(pto.f32, remained)
            preg_vl1, remained = pto.make_mask(pto.f32, remained)
            preg_vl2, remained = pto.make_mask(pto.f32, remained)
            preg_vl3, remained = pto.make_mask(pto.f32, remained)
            off = (i * 32 + j * 4) * lanes_f32
            v_in_1, v_in_2 = pto.vldsx2(src_ptr, off, "DINTLV_B32")
            v_in_1 = pto.vabs(v_in_1, preg_vl0)
            v_in_2 = pto.vabs(v_in_2, preg_vl1)
            v_in_3, v_in_4 = pto.vldsx2(src_ptr, off + 2 * lanes_f32, "DINTLV_B32")
            v_in_3 = pto.vabs(v_in_3, preg_vl2)
            v_in_4 = pto.vabs(v_in_4, preg_vl3)
            v_in_1 = pto.vmax(v_in_1, v_in_2, preg_vl0)
            v_in_3 = pto.vmax(v_in_3, v_in_4, preg_vl2)
            v_di_1, v_di_2 = pto.vdintlv(v_in_1, v_in_3)
            v_max = pto.vmax(v_di_1, v_di_2, preg_vl0)
            v_gp_max = pto.vcgmax(v_max, preg_all)
            ureg_max = pto.vstus(ureg_max, 8, v_gp_max, pto.addptr(max_ptr, 64 * i + 8 * j))
        pto.vstas(ureg_max, pto.addptr(max_ptr, 64 * i), 0)
 

def _init_f32_ocp_quant_ctx():
    ctx = _F32OcpQuantCtx()
    ctx.vb32_exp_mask = pto.vbr(pto.si32(ctx.kExpMask))
    ctx.vb32_mantissa_mask = pto.vbr(pto.si32(ctx.kMantissaMask))
    ctx.vb32_b8_nan = pto.vbr(pto.si32(ctx.kExpNan))
    ctx.vb32_f32_nan = pto.vbr(pto.si32(ctx.kF32Nan))
    ctx.vb32_exp_max = pto.vbr(pto.si32(ctx.kExpMax))
    ctx.vb32_b8_emax = pto.vbr(pto.si32(ctx.f32Emax))
    ctx.vb32_recip_min_scale = pto.vbr(pto.si32(ctx.kRecipMinScale))
    ctx.vb32_zero = pto.vbr(pto.si32(0))
    return ctx


def _compute_f32_ocp_exp_and_scaling(ctx, preg, max_ptr, off):
    v_max = pto.vlds(max_ptr, off)
    v_max_s32 = pto.vbitcast(v_max, pto.si32)
    v_exp = pto.vand(v_max_s32, ctx.vb32_exp_mask, preg)
    v_mant = pto.vand(v_max_s32, ctx.vb32_mantissa_mask, preg)
    v_exp = pto.vshrs(v_exp, ctx.shr, preg)
    v_shared_exp = pto.vsub(v_exp, ctx.vb32_b8_emax, preg)
    v_scaling = pto.vsub(ctx.vb32_exp_max, v_shared_exp, preg)
    v_scaling = pto.vshls(v_scaling, ctx.shr, preg)
    preg_min_scale = pto.vcmps(v_exp, pto.si32(ctx.f32Emax), preg, pto.CmpMode.LE)
    v_scaling = pto.vsel(ctx.vb32_recip_min_scale, v_scaling, preg_min_scale)
    v_shared_exp = pto.vsel(ctx.vb32_zero, v_shared_exp, preg_min_scale)
    preg_special = pto.vcmps(v_exp, pto.si32(ctx.kExpNan), preg, pto.CmpMode.EQ)
    preg_nan = pto.vcmps(v_mant, pto.si32(0), preg_special, pto.CmpMode.NE)
    v_scaling = pto.vsel(ctx.vb32_f32_nan, v_scaling, preg_nan)
    v_shared_exp = pto.vsel(ctx.vb32_b8_nan, v_shared_exp, preg_nan)
    return v_shared_exp, v_scaling


def _extract_f32_ocp_exp_and_scaling_core(max_ptr, exp_ptr, scaling_ptr, off, elem_count):
    ctx = _init_f32_ocp_quant_ctx()
    preg_b32, _ = pto.make_mask(pto.f32, elem_count)
    v_shared_exp, v_scaling = _compute_f32_ocp_exp_and_scaling(ctx, preg_b32, max_ptr, off)
    exp_ptr_si32 = pto.castptr(exp_ptr, pto.ptr(pto.si32, "ub"))
    scaling_ptr_si32 = pto.castptr(scaling_ptr, pto.ptr(pto.si32, "ub"))
    pto.vsts(v_shared_exp, exp_ptr_si32, off // 4, preg_b32, dist=pto.VStoreDist.PK4_B32)
    pto.vsts(v_scaling, scaling_ptr_si32, off, preg_b32, dist=_tquant_norm_dist(pto.f32))


def _store_scaling_unrolled(scaling_ptr, v_scaling, iter, lanes_f32, scaling_elem_count):
    v_scaling_0, v_scaling_1 = pto.vintlv(v_scaling, v_scaling)
    base_off0 = 2 * iter * lanes_f32
    base_off1 = base_off0 + lanes_f32
    rem0 = scaling_elem_count - base_off0 if scaling_elem_count > base_off0 else 0
    rem1 = scaling_elem_count - base_off1 if scaling_elem_count > base_off1 else 0
    preg_sc0, _ = pto.make_mask(pto.f32, rem0)
    pto.vsts(v_scaling_0, scaling_ptr, 2 * iter * lanes_f32, preg_sc0, dist=pto.VStoreDist.NORM_B32)
    if rem1 > 0:
        preg_sc1, _ = pto.make_mask(pto.f32, rem1)
        pto.vsts(v_scaling_1, scaling_ptr, 2 * iter * lanes_f32 + lanes_f32, preg_sc1, dist=pto.VStoreDist.NORM_B32)


def _extract_b8_exp_and_scaling_unrolled(max_ptr, exp_ptr, scaling_ptr, exp_loop_count, total_count, lanes_f32):
    ctx = _init_f32_ocp_quant_ctx()
    remained = total_count
    scaling_elem_count = total_count * 2
    for i in range(exp_loop_count):
        preg_b32, remained = pto.make_mask(pto.f32, remained)
        v_shared_exp, v_scaling = _compute_f32_ocp_exp_and_scaling(ctx, preg_b32, max_ptr, i * lanes_f32)
        pto.vsts(v_shared_exp, exp_ptr, i * lanes_f32, preg_b32, dist=pto.VStoreDist.PK4_B32)
        _store_scaling_unrolled(scaling_ptr, v_scaling, i, lanes_f32, scaling_elem_count)


def _extract_b8_exp_and_scaling_f32(max_ptr, exp_ptr, scaling_ptr, exp_loop_count, total_count, lanes_f32, unroll=False):
    if unroll:
        _extract_b8_exp_and_scaling_unrolled(max_ptr, exp_ptr, scaling_ptr, exp_loop_count, total_count, lanes_f32)
    else:
        for i in range(exp_loop_count):
            _extract_f32_ocp_exp_and_scaling_core(max_ptr, exp_ptr, scaling_ptr, i * lanes_f32, total_count)


def _init_f32_nv_quant_ctx():
    ctx = _F32NvQuantCtx()
    ctx.vb32_exp_mask = pto.vbr(pto.si32(0x7F800000))
    ctx.vb32_mantissa_mask = pto.vbr(pto.si32(0x007FFFFF))
    ctx.vb32_exp_nan = pto.vbr(pto.si32(0xFF))
    ctx.vb32_exp_max = pto.vbr(pto.si32(0xFE))
    ctx.vb32_special_scale = pto.vbr(pto.si32(_NV_MXFP8_SPEC.f32SpecialScaleBits))
    ctx.vb32_min_rcp = pto.vbr(pto.si32(0x00400000))
    return ctx


def _round_up_nv_shared_exponent(v_shared_exp, v_exponent, v_mantissa, preg_b32):
    v_shared_exp_inc = pto.vadds(v_exponent, 1, preg_b32)
    preg_mant_gt_zero = pto.vcmps(v_mantissa, pto.si32(0), preg_b32, pto.CmpMode.NE)
    preg_exp_gt_zero = pto.vcmps(v_exponent, pto.si32(0), preg_b32, pto.CmpMode.GT)
    preg_exp_lt_max = pto.vcmps(v_exponent, pto.si32(0xFE), preg_b32, pto.CmpMode.LT)
    preg_exp_eq_zero = pto.vcmps(v_exponent, pto.si32(0), preg_b32, pto.CmpMode.EQ)
    preg_round_normal = pto.pand(preg_mant_gt_zero, preg_exp_gt_zero, preg_b32)
    preg_round_normal = pto.pand(preg_round_normal, preg_exp_lt_max, preg_b32)
    preg_mant_gt_half_sub = pto.vcmps(v_mantissa, pto.si32(0x00400000), preg_b32, pto.CmpMode.GT)
    preg_round_subnormal = pto.pand(preg_mant_gt_half_sub, preg_exp_eq_zero, preg_b32)
    preg_round_up = pto.por(preg_round_normal, preg_round_subnormal, preg_b32)
    return pto.vsel(v_shared_exp_inc, v_exponent, preg_round_up)


def _compute_f32_nv_exp_and_scaling(ctx, preg, max_ptr, off):
    v_max = pto.vlds(max_ptr, off)
    v_max = pto.vmuls(v_max, pto.f32(ctx.descaleMultiplier), preg)
    v_max_s32 = pto.vbitcast(v_max, pto.si32)
    v_exp = pto.vand(v_max_s32, ctx.vb32_exp_mask, preg)
    v_mant = pto.vand(v_max_s32, ctx.vb32_mantissa_mask, preg)
    v_exp = pto.vshrs(v_exp, ctx.shr, preg)
    v_shared_exp = _round_up_nv_shared_exponent(v_exp, v_exp, v_mant, preg)
    v_scaling = pto.vsub(ctx.vb32_exp_max, v_shared_exp, preg)
    v_scaling = pto.vshls(v_scaling, ctx.shr, preg)
    preg_exp_ff = pto.vcmps(v_exp, pto.si32(0xFF), preg, pto.CmpMode.EQ)
    preg_mant_gt_zero = pto.vcmps(v_mant, pto.si32(0), preg, pto.CmpMode.NE)
    preg_mant_eq_zero = pto.pnot(preg_mant_gt_zero, preg)
    preg_inf = pto.pand(preg_exp_ff, preg_mant_eq_zero, preg)
    preg_nan = pto.pand(preg_exp_ff, preg_mant_gt_zero, preg)
    v_scaling = pto.vsel(ctx.vb32_min_rcp, v_scaling, preg_inf)
    v_shared_exp = pto.vsel(ctx.vb32_exp_max, v_shared_exp, preg_inf)
    v_scaling = pto.vsel(ctx.vb32_special_scale, v_scaling, preg_nan)
    v_shared_exp = pto.vsel(ctx.vb32_exp_nan, v_shared_exp, preg_nan)
    return v_shared_exp, v_scaling


def _extract_f32_nv_exp_and_scaling_core(max_ptr, exp_ptr, scaling_ptr, off, elem_count):
    ctx = _init_f32_nv_quant_ctx()
    preg_b32, _ = pto.make_mask(pto.f32, elem_count)
    v_shared_exp, v_scaling = _compute_f32_nv_exp_and_scaling(ctx, preg_b32, max_ptr, off)
    exp_ptr_si32 = pto.castptr(exp_ptr, pto.ptr(pto.si32, "ub"))
    scaling_ptr_si32 = pto.castptr(scaling_ptr, pto.ptr(pto.si32, "ub"))
    pto.vsts(v_shared_exp, exp_ptr_si32, off // 4, preg_b32, dist=pto.VStoreDist.PK4_B32)
    pto.vsts(v_scaling, scaling_ptr_si32, off, preg_b32, dist=_tquant_norm_dist(pto.f32))


def _extract_nv_exponent_and_scaling_f32(max_ptr, exp_ptr, scaling_ptr, exp_loop_count, total_count, lanes_f32, unroll=False):
    ctx = _init_f32_nv_quant_ctx()
    remained = total_count
    scaling_elem_count = total_count * 2
    for i in range(exp_loop_count):
        preg_b32, remained = pto.make_mask(pto.f32, remained)
        v_shared_exp, v_scaling = _compute_f32_nv_exp_and_scaling(ctx, preg_b32, max_ptr, i * lanes_f32)
        pto.vsts(v_shared_exp, exp_ptr, i * lanes_f32, preg_b32, dist=pto.VStoreDist.PK4_B32)
        if unroll:
            _store_scaling_unrolled(scaling_ptr, v_scaling, i, lanes_f32, scaling_elem_count)
        else:
            pto.vsts(v_scaling, scaling_ptr, i * lanes_f32, preg_b32, dist=pto.VStoreDist.NORM_B32)


def _extract_b8_exp_and_scaling_nv_f32(max_ptr, exp_ptr, scaling_ptr, exp_loop_count, total_count, lanes_f32, unroll=False):
    _extract_nv_exponent_and_scaling_f32(max_ptr, exp_ptr, scaling_ptr, exp_loop_count, total_count, lanes_f32, unroll=unroll)

def _init_f32_exp_scaling_ctx(scale_alg):
    ctx = {}
    ctx["vb32_exp_mask"] = pto.vbr(pto.si32(0x7F800000))
    ctx["vb32_mantissa_mask"] = pto.vbr(pto.si32(0x007FFFFF))
    ctx["vb32_b8_nan"] = pto.vbr(pto.si32(0xFF))
    ctx["vb32_zero"] = pto.vbr(pto.si32(0))
    if scale_alg == "nv":
        ctx["vb32_b8_exp_fe"] = pto.vbr(pto.si32(0xFE))
        ctx["vb32_f32_nan"] = pto.vbr(pto.si32(_NV_MXFP8_SPEC.f32SpecialScaleBits))
        ctx["vb32_f32_min_rcp"] = pto.vbr(pto.si32(0x00400000))
    else:
        ctx["vb32_f32_nan"] = pto.vbr(pto.si32(0x7FC00000))
        ctx["vb32_b8_emax"] = pto.vbr(pto.si32(_OCP_MXFP8_SPEC.f32Emax))
        ctx["vb32_exp_max"] = pto.vbr(pto.si32(0xFE))
        ctx["vb32_recip_min_scale"] = pto.vbr(pto.si32(0x7F000000))
    return ctx

def _compute_f32_exp_scaling_nv(ctx, v_max, preg_b32):
    v_max = pto.vmuls(v_max, pto.f32(_NV_MXFP8_SPEC.descaleMultiplier), preg_b32)
    v_max_s32 = pto.vbitcast(v_max, pto.si32)
    v_exp = pto.vand(v_max_s32, ctx["vb32_exp_mask"], preg_b32)
    v_mant = pto.vand(v_max_s32, ctx["vb32_mantissa_mask"], preg_b32)
    v_exp = pto.vshrs(v_exp, 23, preg_b32)
    v_shared_exp = v_exp
    v_shared_exp_inc = pto.vadds(v_shared_exp, 1, preg_b32)
    preg_mant_gt_zero = pto.vcmps(v_mant, pto.si32(0), preg_b32, pto.CmpMode.NE)
    preg_exp_gt_zero = pto.vcmps(v_exp, pto.si32(0), preg_b32, pto.CmpMode.GT)
    preg_exp_lt_max = pto.vcmps(v_exp, pto.si32(0xFE), preg_b32, pto.CmpMode.LT)
    preg_exp_eq_zero = pto.vcmps(v_exp, pto.si32(0), preg_b32, pto.CmpMode.EQ)
    preg_round_normal = pto.pand(preg_mant_gt_zero, preg_exp_gt_zero, preg_b32)
    preg_round_normal = pto.pand(preg_round_normal, preg_exp_lt_max, preg_b32)
    preg_mant_gt_half_sub = pto.vcmps(v_mant, pto.si32(0x00400000), preg_b32, pto.CmpMode.GT)
    preg_round_subnormal = pto.pand(preg_mant_gt_half_sub, preg_exp_eq_zero, preg_b32)
    preg_round_up = pto.por(preg_round_normal, preg_round_subnormal, preg_b32)
    v_shared_exp = pto.vsel(v_shared_exp_inc, v_shared_exp, preg_round_up)
    v_scaling = pto.vsub(ctx["vb32_b8_exp_fe"], v_shared_exp, preg_b32)
    v_scaling = pto.vshls(v_scaling, 23, preg_b32)
    preg_exp_ff = pto.vcmps(v_exp, pto.si32(0xFF), preg_b32, pto.CmpMode.EQ)
    preg_mant_eq_zero = pto.pnot(preg_mant_gt_zero, preg_b32)
    preg_inf = pto.pand(preg_exp_ff, preg_mant_eq_zero, preg_b32)
    preg_nan = pto.pand(preg_exp_ff, preg_mant_gt_zero, preg_b32)
    v_scaling = pto.vsel(ctx["vb32_f32_min_rcp"], v_scaling, preg_inf)
    v_shared_exp = pto.vsel(ctx["vb32_b8_exp_fe"], v_shared_exp, preg_inf)
    v_scaling = pto.vsel(ctx["vb32_f32_nan"], v_scaling, preg_nan)
    v_shared_exp = pto.vsel(ctx["vb32_b8_nan"], v_shared_exp, preg_nan)
    return v_shared_exp, v_scaling

def _compute_f32_exp_scaling_ocp(ctx, v_max, preg_b32):
    v_max_s32 = pto.vbitcast(v_max, pto.si32)
    v_exp = pto.vand(v_max_s32, ctx["vb32_exp_mask"], preg_b32)
    v_mant = pto.vand(v_max_s32, ctx["vb32_mantissa_mask"], preg_b32)
    v_exp = pto.vshrs(v_exp, 23, preg_b32)
    v_shared_exp = pto.vsub(v_exp, ctx["vb32_b8_emax"], preg_b32)
    v_scaling = pto.vsub(ctx["vb32_exp_max"], v_shared_exp, preg_b32)
    v_scaling = pto.vshls(v_scaling, 23, preg_b32)
    preg_min_scale = pto.vcmps(v_exp, pto.si32(_OCP_MXFP8_SPEC.f32Emax), preg_b32, pto.CmpMode.LE)
    v_scaling = pto.vsel(ctx["vb32_recip_min_scale"], v_scaling, preg_min_scale)
    v_shared_exp = pto.vsel(ctx["vb32_zero"], v_shared_exp, preg_min_scale)
    preg_special = pto.vcmps(v_exp, pto.si32(0xFF), preg_b32, pto.CmpMode.EQ)
    preg_nan = pto.vcmps(v_mant, pto.si32(0), preg_special, pto.CmpMode.NE)
    v_scaling = pto.vsel(ctx["vb32_f32_nan"], v_scaling, preg_nan)
    v_shared_exp = pto.vsel(ctx["vb32_b8_nan"], v_shared_exp, preg_nan)
    return v_shared_exp, v_scaling

def _extract_b8_exp_and_scaling_f32_2d_packed(max_ptr, exp_ptr, scaling_ptr, valid_rows, valid_groups_per_row, exp_col_stride, scale_alg="ocp"):
    ctx = _init_f32_exp_scaling_ctx(scale_alg)
    preg_one, _ = pto.make_mask(pto.f32, 1)
    scale_write_ptr = scaling_ptr
    ureg_scaling = pto.init_align()
    for row in range(valid_rows):
        max_read_ptr = pto.addptr(max_ptr, row * valid_groups_per_row)
        exp_write_ptr = pto.addptr(exp_ptr, row * exp_col_stride)
        ureg_exp = pto.init_align()
        for group in range(valid_groups_per_row):
            v_max = pto.vlds(pto.addptr(max_read_ptr, group), 0, dist="BRC_B32")
            if scale_alg == "nv":
                v_shared_exp, v_scaling = _compute_f32_exp_scaling_nv(ctx, v_max, preg_one)
            else:
                v_shared_exp, v_scaling = _compute_f32_exp_scaling_ocp(ctx, v_max, preg_one)
            v_shared_exp_u8 = pto.vbitcast(v_shared_exp, pto.ui8)
            v_scaling_f32 = pto.vbitcast(v_scaling, pto.f32)
            ureg_exp = pto.vstus(ureg_exp, 1, v_shared_exp_u8, exp_write_ptr)
            ureg_scaling = pto.vstus(ureg_scaling, 1, v_scaling_f32, scale_write_ptr)
            exp_write_ptr = pto.addptr(exp_write_ptr, 1)
            scale_write_ptr = pto.addptr(scale_write_ptr, 1)
        pto.vstas(ureg_exp, exp_write_ptr, 0)
    pto.vstas(ureg_scaling, scale_write_ptr, 0)


def _tquant_mxfp4_e2m1_b16_2d(src_ptr, exp_ptr, dst_ptr, max_ptr, scaling_ptr, valid_rows, valid_cols, src_cols, exp_col_stride, dtype, lanes_b16, scale_alg="ocp"):
    groups_per_row = _ceil_division(valid_cols, 32)
    _abs_reduce_max_b16_nd_2d_packed(src_ptr, max_ptr, valid_rows, valid_cols, src_cols, lanes_b16, dtype, scale_alg)
    pto.mem_bar(pto.BarrierType.VST_VLD)
    _extract_e2m1_exp_and_scaling_2d_packed(max_ptr, exp_ptr, scaling_ptr, valid_rows, groups_per_row, exp_col_stride, scale_alg, dtype)
    pto.mem_bar(pto.BarrierType.VST_VLD)
    _calc_quantized_fp4e2m1_values_2d_packed(src_ptr, scaling_ptr, dst_ptr, max_ptr, exp_ptr,
                                              valid_rows, valid_cols, src_cols, exp_col_stride, lanes_b16, dtype)

def _pad_max_write_to_alignment(v_max, ureg, write_ptr, groups_written, dtype):
    align_groups = _BLOCK_BYTE_SIZE // pto.bytewidth(dtype)
    padded_groups = _ceil_division(groups_written, align_groups) * align_groups
    pad_count = padded_groups - groups_written
    if pad_count > 0:
        ureg = pto.vstus(ureg, pad_count, v_max, write_ptr)
        write_ptr = pto.addptr(write_ptr, pad_count)
    pto.vstas(ureg, write_ptr, 0)


def _abs_reduce_max_b16_nd_2d(src_ptr, max_ptr, valid_rows, valid_cols, src_cols, lanes_b16, dtype=pto.bf16, scale_alg="ocp"):
    grp_size = 32
    elements_per_dintlv = 2 * lanes_b16
    grps_per_dintlv = elements_per_dintlv // grp_size
    groups_per_row = src_cols // grp_size
    elems_per_row = src_cols
    loop_num_per_row = _ceil_division(elems_per_row, elements_per_dintlv)
    ureg_max = pto.init_align()
    write_ptr = max_ptr
    for row in range(valid_rows):
        src_row_off = row * src_cols
        for i in range(loop_num_per_row):
            col_off = i * elements_per_dintlv
            remaining = elems_per_row - col_off if elems_per_row > col_off else 0
            if remaining > elements_per_dintlv:
                remaining = elements_per_dintlv
            if scale_alg == "nv" and dtype == pto.f16:
                v_max = _abs_reduce_max_b16_dintlv_window(src_ptr, src_row_off + col_off, remaining, lanes_b16, dtype, False)
            else:
                v_max = _abs_reduce_max_b16_dintlv_window(src_ptr, src_row_off + col_off, remaining, lanes_b16, dtype, True)
            grps_written = i * grps_per_dintlv
            grps_remaining = groups_per_row - grps_written if groups_per_row > grps_written else 0
            grps_this_iter = grps_per_dintlv if grps_remaining > grps_per_dintlv else grps_remaining
            v_max_bf16 = pto.vbitcast(v_max, pto.bf16)
            ureg_max = pto.vstus(ureg_max, grps_this_iter, v_max_bf16, write_ptr)
            write_ptr = pto.addptr(write_ptr, grps_this_iter)
    pto.vstas(ureg_max, write_ptr, 0)

def _abs_reduce_max_b16_nd_2d_packed(src_ptr, max_ptr, valid_rows, valid_cols, src_cols, lanes_b16, dtype=pto.bf16, scale_alg="ocp"):
    grp_size = 32
    elements_per_dintlv = 2 * lanes_b16
    grps_per_dintlv = elements_per_dintlv // grp_size
    groups_per_row = _ceil_division(valid_cols, grp_size)
    elems_per_row = groups_per_row * grp_size
    loop_num_per_row = _ceil_division(elems_per_row, elements_per_dintlv)
    write_ptr = max_ptr
    ureg_max = pto.init_align()
    if loop_num_per_row == 1:
        for row in range(valid_rows):
            src_row_off = row * src_cols
            if scale_alg == "nv" and dtype == pto.f16:
                v_max = _abs_reduce_max_b16_dintlv_window(src_ptr, src_row_off, elems_per_row, lanes_b16, dtype, False)
            else:
                v_max = _abs_reduce_max_b16_dintlv_window(src_ptr, src_row_off, elems_per_row, lanes_b16, dtype, True)
            v_max_bf16 = pto.vbitcast(v_max, pto.bf16)
            out_count = groups_per_row
            ureg_max = pto.vstus(ureg_max, out_count, v_max_bf16, write_ptr)
            write_ptr = pto.addptr(write_ptr, out_count)
        _pad_max_write_to_alignment(v_max_bf16, ureg_max, write_ptr, valid_rows * groups_per_row, dtype)
        return
    for row in range(valid_rows):
        src_row_off = row * src_cols
        for i in range(loop_num_per_row):
            col_off = i * elements_per_dintlv
            remaining = elems_per_row - col_off if elems_per_row > col_off else 0
            if remaining > elements_per_dintlv:
                remaining = elements_per_dintlv
            if scale_alg == "nv" and dtype == pto.f16:
                v_max = _abs_reduce_max_b16_dintlv_window(src_ptr, src_row_off + col_off, remaining, lanes_b16, dtype, False)
            else:
                v_max = _abs_reduce_max_b16_dintlv_window(src_ptr, src_row_off + col_off, remaining, lanes_b16, dtype, True)
            grps_written = i * grps_per_dintlv
            grps_remaining = groups_per_row - grps_written if groups_per_row > grps_written else 0
            grps_this_iter = grps_per_dintlv if grps_remaining > grps_per_dintlv else grps_remaining
            v_max_bf16 = pto.vbitcast(v_max, pto.bf16)
            ureg_max = pto.vstus(ureg_max, grps_this_iter, v_max_bf16, write_ptr)
            write_ptr = pto.addptr(write_ptr, grps_this_iter)
    pto.vstas(ureg_max, write_ptr, 0)

def _build_packed_scale_window(packed_scaling_ptr, scale_tmp_ptr, group_count):
    ureg = pto.init_align()
    write_ptr = scale_tmp_ptr
    for group in range(group_count):
        v_scale = pto.vlds(packed_scaling_ptr, group, dist="BRC_B16")
        ureg = pto.vstus(ureg, 1, v_scale, write_ptr)
        write_ptr = pto.addptr(write_ptr, 1)
    pto.vstas(ureg, write_ptr, 0)

def _calc_quantized_fp8_values_2d_packed(src_ptr, scaling_ptr, dst_ptr, max_tmp_ptr, exp_ptr,
                                          valid_rows, valid_cols, src_cols, exp_col_stride, lanes_b16, dtype=pto.bf16):
    k_group_size = 32
    k_groups_per_window = 8
    groups_per_row = _ceil_division(valid_cols, k_group_size)
    total_groups = valid_rows * groups_per_row
    windows_per_row = _ceil_division(groups_per_row, k_groups_per_window)
    for row in range(valid_rows):
        src_row = pto.addptr(src_ptr, row * src_cols)
        dst_row = pto.addptr(dst_ptr, row * src_cols)
        scale_row = pto.addptr(scaling_ptr, row * groups_per_row)
        scale_tmp_ptr = max_tmp_ptr
        if total_groups < k_groups_per_window:
            scale_tmp_ptr = pto.castptr(pto.addptr(exp_ptr, row * exp_col_stride + 16), pto.ptr(dtype, "ub"))
        for window in range(windows_per_row):
            group_base = window * k_groups_per_window
            groups_this_window = groups_per_row - group_base
            if groups_this_window > k_groups_per_window:
                groups_this_window = k_groups_per_window
            _build_packed_scale_window(pto.addptr(scale_row, group_base), scale_tmp_ptr, groups_this_window)
            pto.mem_bar(pto.BarrierType.VST_VLD)
            col_off = group_base * k_group_size
            remaining = groups_this_window * k_group_size
            _calc_quantized_fp8_values_b16_window(src_row, scale_tmp_ptr, pto.addptr(dst_row, col_off), 0, col_off, remaining, lanes_b16, dtype)

def _extract_mx_exp_and_scaling_2d_packed(max_ptr, exp_ptr, scaling_ptr, valid_rows, valid_groups_per_row, exp_col_stride, scale_alg, ocp_spec, nv_spec, dtype=pto.bf16):
    if scale_alg == "nv":
        _extract_nv_exp_and_scaling_b16_2d_packed(max_ptr, exp_ptr, scaling_ptr, valid_rows, valid_groups_per_row, exp_col_stride, nv_spec, dtype)
    else:
        _extract_mx_ocp_exp_and_scaling_2d_packed(max_ptr, exp_ptr, scaling_ptr, valid_rows, valid_groups_per_row, exp_col_stride, ocp_spec, dtype)

def _extract_b8_exp_and_scaling_2d_packed(max_ptr, exp_ptr, scaling_ptr, valid_rows, valid_groups_per_row, exp_col_stride, scale_alg, dtype=pto.bf16):
    _extract_mx_exp_and_scaling_2d_packed(max_ptr, exp_ptr, scaling_ptr, valid_rows, valid_groups_per_row, exp_col_stride, scale_alg, _OCP_MXFP8_SPEC, _NV_MXFP8_SPEC, dtype)


def _extract_e2m1_exp_and_scaling_2d_packed(max_ptr, exp_ptr, scaling_ptr, valid_rows, valid_groups_per_row, exp_col_stride, scale_alg, dtype=pto.bf16):
    _extract_mx_exp_and_scaling_2d_packed(max_ptr, exp_ptr, scaling_ptr, valid_rows, valid_groups_per_row, exp_col_stride, scale_alg, _OCP_MXFP4_SPEC, _NV_MXFP4_SPEC, dtype)

def _extract_mx_ocp_exp_and_scaling_2d_packed(max_ptr, exp_ptr, scaling_ptr, valid_rows, valid_groups_per_row, exp_col_stride, spec, dtype=pto.bf16):
    ctx = _init_ocp_exp_scaling_ctx(spec)
    preg_one_b16, _ = pto.make_mask(dtype, 1)
    scale_write_ptr = scaling_ptr
    ureg_scaling = pto.init_align()
    for row in range(valid_rows):
        max_read_ptr = pto.addptr(max_ptr, row * valid_groups_per_row)
        max_read_ptr_u16 = pto.castptr(max_read_ptr, pto.ptr(pto.ui16, "ub"))
        exp_write_ptr = pto.addptr(exp_ptr, row * exp_col_stride)
        ureg_exp = pto.init_align()
        for group in range(valid_groups_per_row):
            v_max_abs = pto.vlds(pto.addptr(max_read_ptr_u16, group), 0, dist="BRC_B16")
            v_exp_value, v_recip_scale = _compute_mx_ocp_exp_scaling(ctx, v_max_abs, preg_one_b16)
            v_exp_value_u8 = pto.vbitcast(v_exp_value, pto.ui8)
            v_recip_scale_u16 = pto.vbitcast(v_recip_scale, pto.ui16)
            ureg_exp = pto.vstus(ureg_exp, 1, v_exp_value_u8, exp_write_ptr)
            ureg_scaling = pto.vstus(ureg_scaling, 1, v_recip_scale_u16, scale_write_ptr)
            exp_write_ptr = pto.addptr(exp_write_ptr, 1)
            scale_write_ptr = pto.addptr(scale_write_ptr, 1)
        pto.vstas(ureg_exp, exp_write_ptr, 0)
    pto.vstas(ureg_scaling, scale_write_ptr, 0)


def _calc_quantized_fp8_values_f32(src_ptr, scaling_ptr, dst_ptr, vl_count, lanes_f32,
                                total_count, preg_lower32, preg_upper32):
    remained = total_count
    preg_all = pto.pset_b32(pto.PAT.ALL)
    for i in range(vl_count):
        preg, remained = pto.make_mask(pto.f32, remained)
        v_scaling_0 = pto.vlds(scaling_ptr, 2 * i, dist="BRC_B32")
        v_scaling_1 = pto.vlds(scaling_ptr, 2 * i + 1, dist="BRC_B32")
        v_in = pto.vlds(src_ptr, i * lanes_f32)
        v_out_1 = pto.vmul(v_in, v_scaling_0, preg_lower32)
        v_out_2 = pto.vmul(v_in, v_scaling_1, preg_upper32)
        v_out = pto.vor(v_out_1, v_out_2, preg_all)
        v_fp8 = pto.vcvt(v_out, pto.f8e4m3, preg,
                          rnd=pto.VcvtRoundMode.R, sat=pto.VcvtSatMode.SAT,
                          part=pto.VcvtPartMode.P0)
        pto.vsts(v_fp8, dst_ptr, i * lanes_f32, preg, dist=pto.VStoreDist.PK4_B32)


def _calc_quantized_fp8_values_unroll2(src_ptr, scaling_ptr, dst_ptr, vl_count, lanes_f32,
                                        total_count):
    preg_all = pto.pset_b32(pto.PAT.ALL)
    dst_ptr_u32 = pto.castptr(dst_ptr, pto.ptr(pto.ui32, "ub"))
    for i in range(vl_count // 2):
        v_scaling = pto.vlds(scaling_ptr, 8 * i, dist="E2B_B32")
        v_in_even, v_in_odd = pto.vldsx2(src_ptr, 2 * i * lanes_f32, "DINTLV_B32")
        v_out_1 = pto.vmul(v_in_even, v_scaling, preg_all)
        v_out_2 = pto.vmul(v_in_odd, v_scaling, preg_all)
        v_fp8_p0 = pto.vcvt(v_out_1, pto.f8e4m3, preg_all,
                            rnd=pto.VcvtRoundMode.R, sat=pto.VcvtSatMode.SAT,
                            part=pto.VcvtPartMode.P0)
        v_fp8_p1 = pto.vcvt(v_out_2, pto.f8e4m3, preg_all,
                            rnd=pto.VcvtRoundMode.R, sat=pto.VcvtSatMode.SAT,
                            part=pto.VcvtPartMode.P1)
        v_fp8_p0_u16 = pto.vbitcast(v_fp8_p0, pto.ui16)
        v_fp8_p1_u16 = pto.vbitcast(v_fp8_p1, pto.ui16)
        preg_all_b16 = pto.pset_b16(pto.PAT.ALL)
        v_fp8_u16 = pto.vor(v_fp8_p0_u16, v_fp8_p1_u16, preg_all_b16)
        v_fp8_u32 = pto.vbitcast(v_fp8_u16, pto.ui32)
        dst_ptr_u32 = pto.castptr(dst_ptr, pto.ptr(pto.ui32, "ub"))
        pto.vsts(v_fp8_u32, dst_ptr_u32, i * lanes_f32 // 2, preg_all, dist=pto.VStoreDist.PK_B32)

def _tquant_mxfp8_f32_quantize(src_ptr, exp_ptr, dst_ptr, max_ptr, scaling_ptr,
                         vl_count, exp_loop_count, num_groups, lanes_f32,
                         total_count, preg_lower32, preg_upper32, static_total, scale_alg="ocp"):
    """static_total = StaticRows * StaticCols."""
    can_unroll = (static_total > 1024) and (static_total % 256 == 0)
    if can_unroll and total_count % 256 == 0:
        if scale_alg == "nv":
            _extract_b8_exp_and_scaling_nv_f32(max_ptr, exp_ptr, scaling_ptr, exp_loop_count, num_groups, lanes_f32, unroll=True)
        else:
            _extract_b8_exp_and_scaling_f32(max_ptr, exp_ptr, scaling_ptr, exp_loop_count, num_groups, lanes_f32, unroll=True)
        pto.mem_bar(pto.BarrierType.VST_VLD)
        _calc_quantized_fp8_values_unroll2(src_ptr, scaling_ptr, dst_ptr, vl_count, lanes_f32,
                                    total_count)
    else:
        if scale_alg == "nv":
            _extract_b8_exp_and_scaling_nv_f32(max_ptr, exp_ptr, scaling_ptr, exp_loop_count, num_groups, lanes_f32, unroll=False)
        else:
            _extract_b8_exp_and_scaling_f32(max_ptr, exp_ptr, scaling_ptr, exp_loop_count, num_groups, lanes_f32, unroll=False)
        pto.mem_bar(pto.BarrierType.VST_VLD)
        _calc_quantized_fp8_values_f32(src_ptr, scaling_ptr, dst_ptr, vl_count, lanes_f32,
                                    total_count, preg_lower32, preg_upper32)


def _static_valid_dim(tile, index):
    static_valid_shape = getattr(tile, "_template_static_valid_shape", None)
    if static_valid_shape is None:
        static_valid_shape = getattr(tile, "static_valid_shape", None)
    if static_valid_shape is not None and static_valid_shape[index] is not None:
        return static_valid_shape[index]
    return tile.valid_shape[index]

@tilelib.tile_template(op="pto.tquant.mx", target="a5", name="template_tquant_mxfp8_f32",
    dtypes=[("f32", "ui8", "ui8", "f32", "f32")],
    iteration_axis="none", op_engine="vector", op_class="elementwise",
    constraints=[_mx_nd_2d_tiles], id=0, loop_depth=2, is_post_update=False, tags=_MX_TAGS + ("fp8", "f32"))
def template_tquant_mxfp8_f32(src: pto.Tile, dst: pto.Tile,
    exp_tile: pto.Tile, max_tile: pto.Tile, scaling_tile: pto.Tile):
    """MXFP8: f32 → fp8 E4M3.
    exp.rows > 1 时走 2D Packed，否则走 flat。"""
    scale_alg = pto.get_op_attr("quant_scale_alg", "ocp")
    src_ptr, dst_ptr, max_ptr, exp_ptr, scaling_ptr = _prepare_nd_ptrs(src, dst, max_tile, exp_tile, scaling_tile)
    if exp_tile.shape[0] > 1:
        valid_rows, valid_cols, src_cols, exp_col_stride = _prepare_2d_quant(src_ptr, src, exp_tile, pto.f32)
        lanes = pto.elements_per_vreg(pto.f32)
        k_group_size = 32
        groups_per_row = _ceil_division(valid_cols, k_group_size)
        vl_count_per_row = _ceil_division(valid_cols, lanes)
        preg_lower32 = pto.pset_b32(pto.PAT.VL32)
        preg_all = pto.pset_b32(pto.PAT.ALL)
        preg_upper32 = pto.pxor(preg_lower32, preg_all, preg_all)
        for row in range(valid_rows):
            _abs_reduce_max_naive(
                src_ptr, max_ptr,
                valid_cols, vl_count_per_row, lanes, preg_lower32, preg_upper32, row * src_cols, row * groups_per_row)
        pto.mem_bar(pto.BarrierType.VST_VLD)
        _extract_b8_exp_and_scaling_f32_2d_packed(
            max_ptr, exp_ptr, scaling_ptr, valid_rows, groups_per_row, exp_col_stride, scale_alg)
        pto.mem_bar(pto.BarrierType.VST_VLD)
        for row in range(valid_rows):
            _calc_quantized_fp8_values_f32(
                pto.addptr(src_ptr, row * src_cols), pto.addptr(scaling_ptr, row * groups_per_row),
                pto.addptr(dst_ptr, row * src_cols), vl_count_per_row, lanes, valid_cols, preg_lower32, preg_upper32)
        return

    valid_rows, valid_cols, static_cols, lanes, total_elems, vl_count, exp_loop_count, num_groups = _prepare_nd_quant(src_ptr, src, pto.f32)
    static_rows = src.shape[0]

    preg_all = pto.pset_b32(pto.PAT.ALL)
    preg_lower32 = pto.pset_b32(pto.PAT.VL32)
    preg_upper32 = pto.pxor(preg_lower32, preg_all, preg_all)

    # ── Stage 1: AbsReduceMax ──
    # Path selection: naive (≤1024), opt (multiple of 256), largesizes (multiple of 2048)
    if (valid_rows * valid_cols <= 1024):
        _abs_reduce_max_naive(src_ptr, max_ptr, total_elems, vl_count, lanes,
                              preg_lower32, preg_upper32)
    else:
        aligned_total = (total_elems // 256) * 256
        tail_total = total_elems - aligned_total
        if aligned_total > 0:
            aligned_vl_count = aligned_total // lanes
            if aligned_total % 2048 == 0:
                _abs_reduce_max_f32_opt_largesizes(
                    src_ptr, max_ptr, aligned_vl_count, lanes, aligned_total)
            else:
                _abs_reduce_max_f32_opt(
                    src_ptr, max_ptr, aligned_vl_count, lanes, aligned_total)
        if tail_total > 0:
            aligned_groups = aligned_total // 32
            tail_vl_count = _ceil_division(tail_total, lanes)
            _abs_reduce_max_naive(src_ptr, max_ptr, tail_total, tail_vl_count, lanes,
                                  preg_lower32, preg_upper32,
                                  src_base=aligned_total, max_base=aligned_groups)

    pto.mem_bar(pto.BarrierType.VST_VLD)

    # ── Stage 2+3: Quantize ──
    static_total = static_rows * static_cols
    _tquant_mxfp8_f32_quantize(src_ptr, exp_ptr, dst_ptr, max_ptr, scaling_ptr,
                         vl_count, exp_loop_count, num_groups, lanes,
                         total_elems, preg_lower32, preg_upper32, static_total, scale_alg=scale_alg)


@tilelib.tile_template(op="pto.tquant.mx", target="a5", name="template_tquant_mxfp8_b16",
    dtypes=[("bf16", "ui8", "ui8", "bf16", "bf16"), ("f16", "ui8", "ui8", "f16", "f16")],
    iteration_axis="none", op_engine="vector", op_class="elementwise",
    constraints=[_mx_nd_2d_tiles], id=1, loop_depth=2, is_post_update=False, tags=_MX_TAGS + ("fp8", "b16"))
def template_tquant_mxfp8_b16(src: pto.Tile, dst: pto.Tile, exp: pto.Tile, max_tile: pto.Tile, scaling: pto.Tile):
    """MXFP8: bf16/fp16 → fp8 E4M3.
    exp.rows > 1 时走 2D Packed（B16_2D overload 1），否则走 flat。"""
    scale_alg = pto.get_op_attr("quant_scale_alg", "ocp")
    dtype = pto.f16 if str(src.dtype) == "f16" else pto.bf16
    src_ptr, dst_ptr, max_ptr, exp_ptr, scaling_ptr = _prepare_nd_ptrs(src, dst, max_tile, exp, scaling)
    if exp.shape[0] > 1:
        valid_rows, valid_cols, src_cols, exp_col_stride = _prepare_2d_quant(src_ptr, src, exp, dtype)
        lanes = pto.elements_per_vreg(dtype)
        groups_per_row = _ceil_division(valid_cols, 32)
        _abs_reduce_max_b16_nd_2d_packed(src_ptr, max_ptr, valid_rows, valid_cols, src_cols, lanes, dtype, scale_alg)
        pto.mem_bar(pto.BarrierType.VST_VLD)
        _extract_b8_exp_and_scaling_2d_packed(max_ptr, exp_ptr, scaling_ptr, valid_rows, groups_per_row, exp_col_stride, scale_alg, dtype)
        pto.mem_bar(pto.BarrierType.VST_VLD)
        _calc_quantized_fp8_values_2d_packed(src_ptr, scaling_ptr, dst_ptr, max_ptr, exp_ptr,
                                              valid_rows, valid_cols, src_cols, exp_col_stride, lanes, dtype)
        return

    valid_rows, valid_cols, static_cols, lanes, total_elems, vl_count, exp_loop_count, num_groups = _prepare_nd_quant(src_ptr, src, dtype)

    if valid_cols == static_cols:
        _reduce_mx_b16_abs_max_flat(src_ptr, max_ptr, vl_count, total_elems, lanes, dtype, scale_alg)
        pto.mem_bar(pto.BarrierType.VST_VLD)
        if scale_alg == "nv":
            _extract_b8_exp_and_scaling_nv(max_ptr, exp_ptr, scaling_ptr, num_groups, dtype)
        else:
            _extract_b8_exp_and_scaling(max_ptr, exp_ptr, scaling_ptr, exp_loop_count, num_groups, lanes, dtype)
        pto.mem_bar(pto.BarrierType.VST_VLD)
        _calc_quantized_fp8_values_b16(src_ptr, scaling_ptr, dst_ptr, total_elems, lanes, dtype)
    else:
        _tquant_mxfp8_b16_2d(
            src_ptr, exp_ptr, dst_ptr, max_ptr, scaling_ptr, valid_rows, valid_cols, static_cols,
            lanes, dtype, scale_alg, exp_loop_count, num_groups)

@tilelib.tile_template(op="pto.tquant.mx", target="a5", name="template_tquant_mxfp4_e2m1_b16",
    dtypes=[("bf16", "f4e2m1x2", "ui8", "bf16", "bf16"), ("f16", "f4e2m1x2", "ui8", "f16", "f16")],
    iteration_axis="none", op_engine="vector", op_class="elementwise",
    constraints=[_mx_nd_2d_tiles], id=2, loop_depth=2, is_post_update=False, tags=_MX_TAGS + ("fp4", "b16"))
def template_tquant_mxfp4_e2m1_b16(src: pto.Tile, dst: pto.Tile, exp: pto.Tile, max_tile: pto.Tile, scaling: pto.Tile):
    """MXFP4: bf16/fp16 → fp4 E2M1.
    exp.rows > 1 时走 2D Packed，否则走 flat。"""
    scale_alg = pto.get_op_attr("quant_scale_alg", "ocp")
    dtype = src.dtype
    dtype = pto.f16 if str(dtype) == "f16" else pto.bf16
    src_ptr, dst_ptr, max_ptr, exp_ptr, scaling_ptr = _prepare_nd_ptrs(src, dst, max_tile, exp, scaling)
    if exp.shape[0] > 1:
        valid_rows, valid_cols, src_cols, exp_col_stride = _prepare_2d_quant(src_ptr, src, exp, dtype)
        lanes_b16 = pto.elements_per_vreg(dtype)
        _tquant_mxfp4_e2m1_b16_2d(
            src_ptr, exp_ptr, dst_ptr, max_ptr, scaling_ptr,
            valid_rows, valid_cols, src_cols, exp_col_stride, dtype, lanes_b16, scale_alg)
        return

    valid_rows, valid_cols, static_cols, lanes, total_elems, vl_count, exp_loop_count, num_groups = _prepare_nd_quant(src_ptr, src, dtype)

    _reduce_mx_b16_abs_max_flat(src_ptr, max_ptr, vl_count, total_elems, lanes, dtype, scale_alg)
    pto.mem_bar(pto.BarrierType.VST_VLD)
    if scale_alg == "nv":
        _extract_e2m1_exponent_and_scaling_nv(max_ptr, exp_ptr, scaling_ptr, num_groups, dtype)
    else:
        _extract_e2m1_exponent_and_scaling(max_ptr, exp_ptr, scaling_ptr, exp_loop_count, num_groups, lanes, dtype)
    pto.mem_bar(pto.BarrierType.VST_VLD)
    if dtype == pto.f16:
        _calc_quantized_fp4e2m1_values_half(src_ptr, scaling_ptr, dst_ptr, num_groups, lanes)
    else:
        _calc_quantized_fp4e2m1_values_bf16(src_ptr, scaling_ptr, dst_ptr, num_groups, lanes)


def _prepare_2d_quant(src_ptr, src, exp, dtype):
    """Common 2D template preamble: set_ctrl + dims + zero-pad + barrier.
    src_ptr must come from _prepare_nd_ptrs.
    Returns (valid_rows, valid_cols, src_cols, exp_col_stride)."""
    pto.set_ctrl(1 << 50)
    valid_rows = _static_valid_dim(src, 0)
    valid_cols = _static_valid_dim(src, 1)
    src_cols = src.shape[1]
    exp_col_stride = exp.shape[1]
    _zero_pad_source_tile(src_ptr, valid_rows, valid_cols, src_cols, dtype)
    pto.mem_bar(pto.BarrierType.VST_VLD)
    return valid_rows, valid_cols, src_cols, exp_col_stride

def _prepare_nd_quant(src_ptr, src, dtype):
    """Common ND template preamble: dims + set_ctrl + zero-pad + barrier.
    src_ptr must come from _prepare_nd_ptrs.
    Returns (valid_rows, valid_cols, static_cols, lanes, total_elems, vl_count, exp_loop_count, num_groups)."""
    valid_rows = _static_valid_dim(src, 0)
    valid_cols = _static_valid_dim(src, 1)
    static_cols = src.shape[1]
    lanes = pto.elements_per_vreg(dtype)
    total_elems = valid_rows * static_cols
    vl_count = _ceil_division(total_elems, lanes)
    num_groups = total_elems // 32
    exp_loop_count = _ceil_division(num_groups, lanes)
    pto.set_ctrl(1 << 50)
    _zero_pad_source_tile(src_ptr, valid_rows, valid_cols, static_cols, dtype)
    pto.mem_bar(pto.BarrierType.VST_VLD)
    return valid_rows, valid_cols, static_cols, lanes, total_elems, vl_count, exp_loop_count, num_groups

def _prepare_nd_ptrs(src, dst, max_tile, exp_tile, scaling_tile):
    """Common ND template pointer materialization.
    Returns (src_ptr, dst_ptr, max_ptr, exp_ptr, scaling_ptr)."""
    src_ptr = src.as_ptr()
    dst_ptr = dst.as_ptr()
    max_ptr = max_tile.as_ptr()
    exp_ptr = exp_tile.as_ptr()
    scaling_ptr = scaling_tile.as_ptr()
    return src_ptr, dst_ptr, max_ptr, exp_ptr, scaling_ptr

def _prepare_dn_quant(src, dtype):
    """Common DN template preamble: set_ctrl + dims + zero-pad + barrier.
    Returns (valid_rows, valid_cols, static_cols)."""
    pto.set_ctrl(1 << 50)
    valid_rows = _static_valid_dim(src, 0)
    valid_cols = _static_valid_dim(src, 1)
    static_cols = src.shape[1]
    src_ptr = src.as_ptr()
    _zero_pad_source_tile(src_ptr, valid_rows, valid_cols, static_cols, dtype)
    pto.mem_bar(pto.BarrierType.VST_VLD)
    return valid_rows, valid_cols, static_cols

def _mx_dn_tiles(operand_memory_spaces, grp_axis=None, **_):
    """DN template constraint。仅接受用户显式声明 DN（grp_axis=0/axis0）。"""
    if not all(space in {"ub", "vec"} for space in operand_memory_spaces):
        return False
    return grp_axis is not None and str(grp_axis) in ("0", "axis0")

@tilelib.tile_template(op="pto.tquant.mx", target="a5", name="template_tquant_mxfp8_dn",
    dtypes=[("bf16", "ui8", "ui8", "bf16", "bf16"), ("f16", "ui8", "ui8", "f16", "f16")],
    iteration_axis="none", op_engine="vector", op_class="elementwise",
    constraints=[_mx_dn_tiles], id=3, loop_depth=2, is_post_update=False, tags=_MX_TAGS + ("fp8", "dn"))
def template_tquant_mxfp8_dn(src: pto.Tile, dst: pto.Tile, exp: pto.Tile,
                              max_tile: pto.Tile, scaling: pto.Tile):
    """MXFP8 DN: bf16/fp16 → fp8."""
    dtype = src.dtype
    dtype = pto.f16 if str(dtype) == "f16" else pto.bf16
    scale_alg = pto.get_op_attr("quant_scale_alg", "ocp")
    interleave = pto.get_op_attr("interleave") == "true"
    valid_rows, valid_cols, static_cols = _prepare_dn_quant(src, dtype)
    dst_static_cols = dst.shape[1]
    src_ptr, dst_ptr, max_ptr, exp_ptr, scaling_ptr = _prepare_nd_ptrs(src, dst, max_tile, exp, scaling)
    exp_col_stride = exp.shape[1] if interleave else static_cols
    _tquant_mxfp8_dn_b16(src_ptr, dst_ptr, exp_ptr, max_ptr, scaling_ptr, dtype, valid_rows, valid_cols, static_cols, dst_static_cols, scale_alg, interleave, exp_col_stride)

@tilelib.tile_template(op="pto.tquant.mx", target="a5", name="template_tquant_mxfp4_e2m1_dn",
    dtypes=[("bf16", "f4e2m1x2", "ui8", "bf16", "bf16"), ("f16", "f4e2m1x2", "ui8", "f16", "f16")],
    iteration_axis="none", op_engine="vector", op_class="elementwise",
    constraints=[_mx_dn_tiles], id=4, loop_depth=2, is_post_update=False, tags=_MX_TAGS + ("fp4", "dn"))
def template_tquant_mxfp4_e2m1_dn(src: pto.Tile, dst: pto.Tile, exp: pto.Tile,
                              max_tile: pto.Tile, scaling: pto.Tile):
    """MXFP4 DN: bf16/fp16 → fp4."""
    dtype = src.dtype
    dtype = pto.f16 if str(dtype) == "f16" else pto.bf16
    scale_alg = pto.get_op_attr("quant_scale_alg", "ocp")
    interleave = pto.get_op_attr("interleave") == "true"
    valid_rows, valid_cols, static_cols = _prepare_dn_quant(src, dtype)
    dst_static_cols = dst.shape[1]
    src_ptr, dst_ptr, max_ptr, exp_ptr, scaling_ptr = _prepare_nd_ptrs(src, dst, max_tile, exp, scaling)
    exp_col_stride = exp.shape[1] if interleave else static_cols
    _tquant_mxfp4_e2m1_dn(src_ptr, dst_ptr, exp_ptr, max_ptr, scaling_ptr, dtype, valid_rows, valid_cols, static_cols, dst_static_cols, scale_alg, interleave, exp_col_stride)


def _tquant_pset_typed(dtype, pattern):
    sz = pto.bytewidth(dtype)
    if sz == 4:
        return pto.pset_b32(pattern)
    elif sz == 2:
        return pto.pset_b16(pattern)
    else:
        return pto.pset_b8(pattern)


def _tquant_zero_value(dtype):
    """Return a zero scalar matching sizeof(T), like (T)0."""
    sz = pto.bytewidth(dtype)
    if sz == 4:
        return pto.f32(0)
    elif sz == 2:
        return pto.bf16(0)
    else:
        return pto.ui8(0)


def _tquant_norm_dist(dtype):
    """Return NORM_B32/B16/B8 by sizeof(T), matching GetDistVst<T, DIST_NORM>."""
    sz = pto.bytewidth(dtype)
    if sz == 4:
        return pto.VStoreDist.NORM_B32
    elif sz == 2:
        return pto.VStoreDist.NORM_B16
    else:
        return pto.VStoreDist.NORM_B8


def _zero_pad_columns_vl_aligned(src_ptr, valid_rows, valid_cols, static_cols, dtype):
    elem_per_vl = pto.elements_per_vreg(dtype)
    rows_per_vl = elem_per_vl // static_cols
    pg_all = _tquant_pset_typed(dtype, pto.PAT.ALL)
    vc = valid_cols
    preg_valid, _ = pto.make_mask(dtype, vc)
    for r in range(1, rows_per_vl):
        range_start = r * static_cols
        range_end = range_start + valid_cols
        p_end, _ = pto.make_mask(dtype, range_end)
        p_start, _ = pto.make_mask(dtype, range_start)
        p_row = pto.pnot(p_start, p_end)
        preg_valid = pto.por(preg_valid, p_row, pg_all)
    v_zero = pto.vdup(_tquant_zero_value(dtype), pg_all)
    preg_pad = pto.pxor(pg_all, preg_valid, pg_all)
    total_elems = valid_rows * static_cols
    vl_count = _ceil_division(total_elems, elem_per_vl)
    for vi in range(vl_count):
        pto.vsts(v_zero, src_ptr, vi * elem_per_vl, preg_pad, dist=pto.VStoreDist.NORM_B16)

def _zero_pad_columns_unaligned(src_ptr, valid_rows, valid_cols, static_cols, dtype):
    elem_per_vl = pto.elements_per_vreg(dtype)
    pad_cols = static_cols - valid_cols
    pad_repeats = _ceil_division(pad_cols, elem_per_vl)
    pg_all = _tquant_pset_typed(dtype, pto.PAT.ALL)
    v_zero = pto.vdup(_tquant_zero_value(dtype), pg_all)
    for row in range(valid_rows):
        write_ptr = pto.addptr(src_ptr, row * static_cols + valid_cols)
        cols = pad_cols
        ureg = pto.init_align()
        for j in range(pad_repeats):
            sreg = elem_per_vl if cols > elem_per_vl else cols
            ureg = pto.vstus(ureg, sreg, v_zero, write_ptr)
            write_ptr = pto.addptr(write_ptr, sreg)
            cols -= elem_per_vl
        pto.vstas(ureg, write_ptr, 0)

def _zero_pad_source_tile(src_ptr, valid_rows, valid_cols, static_cols, dtype=pto.f32):
    """Skips float; dispatches VLAligned vs Unaligned."""
    if dtype == pto.f32:
        return
    if valid_cols >= static_cols:
        return
    elem_per_vl = pto.elements_per_vreg(dtype)
    if elem_per_vl % static_cols == 0:
        _zero_pad_columns_vl_aligned(src_ptr, valid_rows, valid_cols, static_cols, dtype)
    else:
        _zero_pad_columns_unaligned(src_ptr, valid_rows, valid_cols, static_cols, dtype)


def _update_dn_abs_max_4(v_0, v_1, v_2, v_3, v_max_0, v_max_1, v_max_2, v_max_3, preg, convert_fp16_to_bf16, is_f32, v_abs_mask=None):
    if convert_fp16_to_bf16:
        v_0b, v_1b = _fp16_to_bf16_preserve_special(v_0, v_1, preg, preg)
        v_2b, v_3b = _fp16_to_bf16_preserve_special(v_2, v_3, preg, preg)
        a_0 = pto.vand(pto.vbitcast(v_0b, pto.ui16), v_abs_mask, preg)
        a_1 = pto.vand(pto.vbitcast(v_1b, pto.ui16), v_abs_mask, preg)
        a_2 = pto.vand(pto.vbitcast(v_2b, pto.ui16), v_abs_mask, preg)
        a_3 = pto.vand(pto.vbitcast(v_3b, pto.ui16), v_abs_mask, preg)
    elif is_f32:
        a_0 = pto.vabs(v_0, preg)
        a_1 = pto.vabs(v_1, preg)
        a_2 = pto.vabs(v_2, preg)
        a_3 = pto.vabs(v_3, preg)
    else:
        a_0 = pto.vand(pto.vbitcast(v_0, pto.ui16), v_abs_mask, preg)
        a_1 = pto.vand(pto.vbitcast(v_1, pto.ui16), v_abs_mask, preg)
        a_2 = pto.vand(pto.vbitcast(v_2, pto.ui16), v_abs_mask, preg)
        a_3 = pto.vand(pto.vbitcast(v_3, pto.ui16), v_abs_mask, preg)
    return (pto.vmax(a_0, v_max_0, preg), pto.vmax(a_1, v_max_1, preg),
            pto.vmax(a_2, v_max_2, preg), pto.vmax(a_3, v_max_3, preg))


def _merge_and_store_dn_abs_max(v_max_0, v_max_1, v_max_2, v_max_3, max_ptr, offset, preg, dtype, norm_dist):
    v_max_0 = pto.vmax(v_max_0, v_max_1, preg)
    v_max_2 = pto.vmax(v_max_2, v_max_3, preg)
    v_max = pto.vmax(v_max_0, v_max_2, preg)
    if pto.bytewidth(dtype) != 4:
        v_max = pto.vbitcast(v_max, dtype)
    pto.vsts(v_max, max_ptr, offset, preg, dist=norm_dist)


def _abs_reduce_max_dn(src_ptr, max_ptr, valid_rows, valid_cols, static_cols, lanes, dtype=pto.f32, scale_alg="ocp"):
    grp_size = 32
    num_vls_per_row = _ceil_division(valid_cols, lanes)
    num_grps_per_col = _ceil_division(valid_rows, grp_size)
    num_vls_inner = 4
    inner_iters = _ceil_division(grp_size, num_vls_inner)
    preg_cols = valid_cols
    is_f32 = pto.bytewidth(dtype) == 4
    convert_fp16_to_bf16 = scale_alg == "ocp" and dtype == pto.f16
    norm_dist = _tquant_norm_dist(dtype)
    if is_f32:
        v_zero = pto.vbr(pto.f32(0))
        v_abs_mask = None
    else:
        v_abs_mask = pto.vbr(pto.ui16(_BF16_ABS_MASK))
        v_zero = pto.vbr(pto.ui16(0))
    for i in range(num_vls_per_row):
        vl_start = i * lanes
        preg, preg_cols = pto.make_mask(dtype, preg_cols)
        for j in range(num_grps_per_col):
            grp_start = j * grp_size * static_cols
            loop = pto.for_(0, inner_iters, step=1).carry(
                v0=v_zero, v1=v_zero, v2=v_zero, v3=v_zero)
            with loop:
                k = loop.iv
                inner_start = k * num_vls_inner * static_cols
                base_off = vl_start + grp_start + inner_start
                v_0 = pto.vlds(src_ptr, base_off)
                v_1 = pto.vlds(src_ptr, base_off + static_cols)
                v_2 = pto.vlds(src_ptr, base_off + 2 * static_cols)
                v_3 = pto.vlds(src_ptr, base_off + 3 * static_cols)
                v_max_0, v_max_1, v_max_2, v_max_3 = _update_dn_abs_max_4(
                    v_0, v_1, v_2, v_3, loop.v0, loop.v1, loop.v2, loop.v3, preg, convert_fp16_to_bf16, is_f32,
                    v_abs_mask)
                loop.update(v0=v_max_0, v1=v_max_1, v2=v_max_2, v3=v_max_3)
            v_max_0 = loop.final("v0")
            v_max_1 = loop.final("v1")
            v_max_2 = loop.final("v2")
            v_max_3 = loop.final("v3")
            _merge_and_store_dn_abs_max(
                v_max_0, v_max_1, v_max_2, v_max_3, max_ptr, j * static_cols + vl_start, preg, dtype, norm_dist)


def _calc_quantized_fp8_values_dn_float(src_ptr, scaling_ptr, dst_ptr, valid_rows, valid_cols, src_static_cols, dst_static_cols, lanes_f32):
    grp_size = 32
    num_vls_per_row = _ceil_division(valid_cols, lanes_f32)
    num_grps_per_col = _ceil_division(valid_rows, grp_size)
    preg_cols_b32 = valid_cols
    preg_cols_b8 = valid_cols * 4
    for i in range(num_vls_per_row):
        vl_start = i * lanes_f32
        preg_b32, preg_cols_b32 = pto.make_mask(pto.f32, preg_cols_b32)
        preg_b8, preg_cols_b8 = pto.make_mask(pto.ui8, preg_cols_b8)
        for j in range(num_grps_per_col):
            row_base = j * grp_size
            v_scaling = pto.vlds(scaling_ptr, vl_start + j * src_static_cols)
            with pto.for_(0, grp_size, step=1) as k:
                r = row_base + k
                src_off = r * src_static_cols + vl_start
                dst_off = r * dst_static_cols + vl_start
                v_input = pto.vlds(src_ptr, src_off)
                v_input = pto.vmul(v_input, v_scaling, preg_b32)
                v_fp8 = pto.vcvt(v_input, pto.f8e4m3, preg_b32,
                                 rnd=pto.VcvtRoundMode.R, sat=pto.VcvtSatMode.SAT,
                                 part=pto.VcvtPartMode.P0)
                pto.vsts(v_fp8, dst_ptr, dst_off, preg_b8,
                         dist=pto.VStoreDist.PK4_B32)


def _calc_quantized_fp8_values_dn_b16(src_ptr, scaling_ptr, dst_ptr, valid_rows, valid_cols, src_static_cols, dst_static_cols, lanes_b16, dtype=pto.bf16):
    grp_size = 32
    num_vls_per_row = _ceil_division(valid_cols, lanes_b16)
    num_grps_per_col = _ceil_division(valid_rows, grp_size)
    preg_cols_b16 = valid_cols
    preg_cols_b8 = valid_cols * 2
    preg_cols_b32 = _ceil_division(valid_cols, 2)
    preg_all_b16 = pto.pset_b16(pto.PAT.ALL)
    scaling_ptr_u16 = pto.castptr(scaling_ptr, pto.ptr(pto.ui16, "ub"))
    dst_ptr_u32 = pto.castptr(dst_ptr, pto.ptr(pto.ui32, "ub"))
    for i in range(num_vls_per_row):
        vl_start = i * lanes_b16
        preg_b16, preg_cols_b16 = pto.make_mask(dtype, preg_cols_b16)
        preg_b8, preg_cols_b8 = pto.make_mask(pto.ui8, preg_cols_b8)
        preg_b32, preg_cols_b32 = pto.make_mask(pto.f32, preg_cols_b32)
        for j in range(num_grps_per_col):
            row_base = j * grp_size
            v_scaling_u16 = pto.vlds(scaling_ptr_u16, vl_start + j * src_static_cols)
            v_scaling_bf16 = pto.vbitcast(v_scaling_u16, pto.bf16)
            v_scaling_f32_even = pto.vcvt(v_scaling_bf16, pto.f32, preg_b16, part=pto.VcvtPartMode.EVEN)
            v_scaling_f32_odd = pto.vcvt(v_scaling_bf16, pto.f32, preg_b16, part=pto.VcvtPartMode.ODD)
            with pto.for_(0, grp_size, step=1) as k:
                r = row_base + k
                src_off = r * src_static_cols + vl_start
                dst_off = r * dst_static_cols + vl_start
                v_input = pto.vlds(src_ptr, src_off)
                v_input_f32_even = pto.vcvt(v_input, pto.f32, preg_b16, part=pto.VcvtPartMode.EVEN)
                v_input_f32_odd = pto.vcvt(v_input, pto.f32, preg_b16, part=pto.VcvtPartMode.ODD)
                v_input_f32_even = pto.vmul(v_input_f32_even, v_scaling_f32_even, preg_b32)
                v_input_f32_odd = pto.vmul(v_input_f32_odd, v_scaling_f32_odd, preg_b32)
                v_fp8_p0 = pto.vcvt(v_input_f32_even, pto.f8e4m3, preg_b32,
                                    rnd=pto.VcvtRoundMode.R, sat=pto.VcvtSatMode.SAT, part=pto.VcvtPartMode.P0)
                v_fp8_p1 = pto.vcvt(v_input_f32_odd, pto.f8e4m3, preg_b32,
                                    rnd=pto.VcvtRoundMode.R, sat=pto.VcvtSatMode.SAT, part=pto.VcvtPartMode.P1)
                v_fp8_p0_u16 = pto.vbitcast(v_fp8_p0, pto.ui16)
                v_fp8_p1_u16 = pto.vbitcast(v_fp8_p1, pto.ui16)
                v_out_u16 = pto.vor(v_fp8_p0_u16, v_fp8_p1_u16, preg_all_b16)
                v_out_u32 = pto.vbitcast(v_out_u16, pto.ui32)
                pto.vsts(v_out_u32, dst_ptr_u32, dst_off // 4, preg_b32, dist=pto.VStoreDist.PK_B32)


def _extract_dn_linear_exponent_and_scaling(max_ptr, exp_ptr, scaling_ptr, valid_rows, valid_cols, static_cols, elements_per_vl, alg=OcpF8E4M3Alg, dtype=pto.f32):
    grp_size = 32
    vl_count = _ceil_division(valid_cols, elements_per_vl)
    group_count = _ceil_division(valid_rows, grp_size)
    is_f32 = pto.bytewidth(dtype) == 4
    ctx = None
    if not is_f32:
        ctx = _init_dn_b16_quant_ctx(alg, dtype)
    for group in range(group_count):
        max_row = pto.addptr(max_ptr, group * static_cols)
        exp_row = pto.addptr(exp_ptr, group * static_cols)
        scaling_row = pto.addptr(scaling_ptr, group * static_cols)
        for vl in range(vl_count):
            off = vl * elements_per_vl
            remaining = valid_cols - off if valid_cols > off else 0
            if remaining > elements_per_vl:
                remaining = elements_per_vl
            if is_f32:
                _extract_b8_exponent_and_scaling_vl(alg, max_row, exp_row, scaling_row, off, remaining, dtype)
            else:
                _extract_dn_b16_exponent_and_scaling_vl(ctx, max_row, exp_row, scaling_row, off, remaining, alg, dtype)


def _extract_b8_exponent_and_scaling_2d(max_ptr, exp_ptr, scaling_ptr, valid_rows, valid_cols, src_cols, lanes_b16, dtype=pto.bf16):
    elements_per_vl = lanes_b16
    groups_per_row = src_cols // 32
    valid_groups_per_row = _ceil_division(valid_cols, 32)
    loops_per_row = _ceil_division(valid_groups_per_row, elements_per_vl)
    for row in range(valid_rows):
        row_off = row * groups_per_row
        for i in range(loops_per_row):
            off = i * elements_per_vl
            rem = valid_groups_per_row - off if valid_groups_per_row > off else 0
            if rem > elements_per_vl:
                rem = elements_per_vl
            _extract_b8_exponent_and_scaling_vl(
                OcpF8E4M3Alg, pto.addptr(max_ptr, row_off), pto.addptr(exp_ptr, row_off),
                pto.addptr(scaling_ptr, row_off), off, rem, dtype)


def _store_dn_interleaved_exponent_b16(exp_ptr, v_exp0, v_exp1, dst_byte_offset, elem_count):
    preg, _ = pto.make_mask(pto.ui16, elem_count)
    v_exp1 = pto.vshls(v_exp1, 8, preg)
    v_exp0 = pto.vor(v_exp0, v_exp1, preg)
    exp_ptr_u16 = pto.castptr(exp_ptr, pto.ptr(pto.ui16, "ub"))
    pto.vsts(v_exp0, exp_ptr_u16, dst_byte_offset // 2, preg, dist=pto.VStoreDist.NORM_B16)


def _store_dn_interleaved_exponent_b32(exp_ptr, v_exp0, v_exp1, dst_byte_offset, elem_count):
    preg, _ = pto.make_mask(pto.f32, elem_count)
    v_exp1 = pto.vshls(v_exp1, 8, preg)
    v_exp0 = pto.vor(v_exp0, v_exp1, preg)
    exp_ptr_u16 = pto.castptr(exp_ptr, pto.ptr(pto.ui16, "ub"))
    pto.vsts(v_exp0, exp_ptr_u16, dst_byte_offset // 2, preg, dist=pto.VStoreDist.PK_B32)


def _extract_dn_interleaved_pair_vl(ctx, max_row0, max_row1, scaling_row0, scaling_row1, exp_ptr, off, dst_byte_offset, rem, alg, dtype=pto.bf16):
    if alg in (OcpF8E4M3Alg, OcpF4E2M1Alg):
        v_exp0, v_scaling0, preg0 = _compute_b16_ocp_exp_and_scaling_regs(ctx, max_row0, off, rem, dtype)
        v_exp1, v_scaling1, preg1 = _compute_b16_ocp_exp_and_scaling_regs(ctx, max_row1, off, rem, dtype)
        scaling_row0_u16 = pto.castptr(scaling_row0, pto.ptr(pto.ui16, "ub"))
        scaling_row1_u16 = pto.castptr(scaling_row1, pto.ptr(pto.ui16, "ub"))
        pto.vsts(v_scaling0, scaling_row0_u16, off, preg0, dist=pto.VStoreDist.NORM_B16)
        pto.vsts(v_scaling1, scaling_row1_u16, off, preg1, dist=pto.VStoreDist.NORM_B16)
        _store_dn_interleaved_exponent_b16(exp_ptr, v_exp0, v_exp1, dst_byte_offset, rem)
    else:
        v_exp0, v_scaling0, preg = _compute_b16_nv_exp_and_scaling_regs(ctx, max_row0, off, rem, dtype)
        v_exp1, v_scaling1, preg = _compute_b16_nv_exp_and_scaling_regs(ctx, max_row1, off, rem, dtype)
        scaling_row0_u16 = pto.castptr(scaling_row0, pto.ptr(pto.ui16, "ub"))
        scaling_row1_u16 = pto.castptr(scaling_row1, pto.ptr(pto.ui16, "ub"))
        pto.vsts(v_scaling0, scaling_row0_u16, off, preg, dist=pto.VStoreDist.PK_B32)
        pto.vsts(v_scaling1, scaling_row1_u16, off, preg, dist=pto.VStoreDist.PK_B32)
        _store_dn_interleaved_exponent_b32(exp_ptr, v_exp0, v_exp1, dst_byte_offset, rem)


def _extract_dn_interleaved_exponent_and_scaling(max_ptr, exp_ptr, exp_scratch_ptr, scaling_ptr, valid_rows, valid_cols, static_cols, exp_col_stride, elements_per_vl, alg, dtype):
    grp_size = 32
    scratch_row_stride = _ceil_division(static_cols, 32) * 32
    vl_count = _ceil_division(valid_cols, elements_per_vl)
    group_count = _ceil_division(valid_rows, grp_size)
    is_f32 = pto.bytewidth(dtype) == 4
    if is_f32:
        for group in range(group_count):
            max_row = pto.addptr(max_ptr, group * static_cols)
            exp_scratch_row = pto.addptr(exp_scratch_ptr, group * scratch_row_stride)
            scaling_row = pto.addptr(scaling_ptr, group * static_cols)
            for vl in range(vl_count):
                off = vl * elements_per_vl
                remaining = valid_cols - off if valid_cols > off else 0
                if remaining > elements_per_vl:
                    remaining = elements_per_vl
                _extract_b8_exponent_and_scaling_vl(alg, max_row, exp_scratch_row, scaling_row, off, remaining, dtype)
    else:
        ctx = _init_dn_b16_quant_ctx(alg, dtype)
        pair_count = group_count // 2
        for pair in range(pair_count):
            max_row0 = pto.addptr(max_ptr, pair * 2 * static_cols)
            max_row1 = pto.addptr(max_row0, static_cols)
            scaling_row0 = pto.addptr(scaling_ptr, pair * 2 * static_cols)
            scaling_row1 = pto.addptr(scaling_row0, static_cols)
            for vl in range(vl_count):
                off = vl * elements_per_vl
                remaining = valid_cols - off if valid_cols > off else 0
                if remaining > elements_per_vl:
                    remaining = elements_per_vl
                _extract_dn_interleaved_pair_vl(ctx, max_row0, max_row1, scaling_row0, scaling_row1, exp_ptr, off, pair * exp_col_stride + 2 * off, remaining, alg, dtype)


def _write_dn_interleaved_exponent_f32(exp_scratch, exp_ptr, valid_rows, valid_cols, static_cols, exp_col_stride):
    grp_size = 32
    bytes_per_input_vl = 128  # CCE_VL / 2
    scratch_row_stride = _ceil_division(static_cols, 32) * 32
    pair_count = valid_rows // (2 * grp_size)
    store_vl_count = _ceil_division(valid_cols, bytes_per_input_vl)
    for pair in range(pair_count):
        exp_scratch0 = pto.addptr(exp_scratch, pair * 2 * scratch_row_stride)
        exp_scratch1 = pto.addptr(exp_scratch0, scratch_row_stride)
        rem = valid_cols
        for vl in range(store_vl_count):
            off = vl * bytes_per_input_vl
            exp0 = pto.vlds(exp_scratch0, off, dist="UNPK_B8")
            exp1 = pto.vlds(exp_scratch1, off, dist="UNPK_B8")
            preg, _ = pto.make_mask(pto.ui16, rem)
            exp1_u16 = pto.vbitcast(exp1, pto.ui16)
            exp1_u16 = pto.vshls(exp1_u16, 8, preg)
            exp0_u16 = pto.vbitcast(exp0, pto.ui16)
            exp0_u16 = pto.vor(exp0_u16, exp1_u16, preg)
            dst_byte_offset = pair * exp_col_stride + 2 * off
            exp_ptr_u16 = pto.castptr(exp_ptr, pto.ptr(pto.ui16, "ub"))
            pto.vsts(exp0_u16, exp_ptr_u16, dst_byte_offset // 2, preg, dist=pto.VStoreDist.NORM_B16)


def _tquant_mxfp8_dn_f32(src_ptr, dst_ptr, exp_ptr, max_ptr, scaling_ptr, valid_rows, valid_cols, src_static_cols, dst_static_cols, scale_alg="ocp", interleave=False, exp_col_stride=None):
    lanes_f32 = pto.elements_per_vreg(pto.f32)
    alg = NvF8E4M3Alg if scale_alg == "nv" else OcpF8E4M3Alg

    _abs_reduce_max_dn(src_ptr, max_ptr, valid_rows, valid_cols, src_static_cols, lanes_f32, pto.f32, scale_alg)
    pto.mem_bar(pto.BarrierType.VST_VLD)
    if interleave:
        _extract_dn_interleaved_exponent_and_scaling(max_ptr, exp_ptr, dst_ptr, scaling_ptr, valid_rows, valid_cols, src_static_cols, exp_col_stride, lanes_f32, alg, pto.f32)
        pto.mem_bar(pto.BarrierType.VST_VLD)
        _write_dn_interleaved_exponent_f32(dst_ptr, exp_ptr, valid_rows, valid_cols, src_static_cols, exp_col_stride)
        pto.mem_bar(pto.BarrierType.VLD_VST)
    else:
        _extract_dn_linear_exponent_and_scaling(max_ptr, exp_ptr, scaling_ptr, valid_rows, valid_cols, src_static_cols, lanes_f32, alg)
        pto.mem_bar(pto.BarrierType.VST_VLD)
    _calc_quantized_fp8_values_dn_float(src_ptr, scaling_ptr, dst_ptr, valid_rows, valid_cols, src_static_cols, dst_static_cols, lanes_f32)


def _tquant_mxfp8_dn_b16(src_ptr, dst_ptr, exp_ptr, max_ptr, scaling_ptr, dtype, valid_rows, valid_cols, src_static_cols, dst_static_cols, scale_alg="ocp", interleave=False, exp_col_stride=None):
    alg = NvF8E4M3Alg if scale_alg == "nv" else OcpF8E4M3Alg
    lanes_b16 = pto.elements_per_vreg(dtype)
    extract_elements_per_vl = pto.elements_per_vreg(pto.f32) if scale_alg == "nv" else lanes_b16

    _abs_reduce_max_dn(src_ptr, max_ptr, valid_rows, valid_cols, src_static_cols, lanes_b16, dtype, scale_alg)
    pto.mem_bar(pto.BarrierType.VST_VLD)
    if interleave:
        _extract_dn_interleaved_exponent_and_scaling(max_ptr, exp_ptr, dst_ptr, scaling_ptr, valid_rows, valid_cols, src_static_cols, exp_col_stride, extract_elements_per_vl, alg, dtype)
    else:
        _extract_dn_linear_exponent_and_scaling(max_ptr, exp_ptr, scaling_ptr, valid_rows, valid_cols, src_static_cols, extract_elements_per_vl, alg, dtype)
    pto.mem_bar(pto.BarrierType.VST_VLD)
    _calc_quantized_fp8_values_dn_b16(src_ptr, scaling_ptr, dst_ptr, valid_rows, valid_cols, src_static_cols, dst_static_cols, lanes_b16, dtype)


def _extract_mx_ocp_exponent_and_scaling_vl(max_ptr, exp_ptr, scaling_ptr, off, rem, spec=None, dtype=pto.bf16):
    ctx = _init_b16_ocp_quant_ctx(spec)
    _compute_b16_ocp_exp_and_scaling(ctx, max_ptr, exp_ptr, scaling_ptr, off, rem, dtype)


def _extract_b16_nv_exponent_and_scaling_core(max_ptr, exp_ptr, scaling_ptr, off, rem, spec=None, dtype=pto.bf16):
    ctx = _init_b16_nv_quant_ctx(spec)
    _compute_b16_nv_exp_and_scaling(ctx, max_ptr, exp_ptr, scaling_ptr, off, rem, dtype)


def _init_dn_b16_quant_ctx(alg, dtype=pto.bf16):
    if alg in (OcpF8E4M3Alg, OcpF4E2M1Alg):
        spec = _OCP_MXFP8_SPEC if alg == OcpF8E4M3Alg else _OCP_MXFP4_SPEC
        return _init_b16_ocp_quant_ctx(spec)
    spec = _NV_MXFP8_SPEC if alg == NvF8E4M3Alg else _NV_MXFP4_SPEC
    return _init_b16_nv_quant_ctx(spec)


def _extract_dn_b16_exponent_and_scaling_vl(ctx, max_ptr, exp_ptr, scaling_ptr, off, rem, alg, dtype=pto.bf16):
    if alg in (OcpF8E4M3Alg, OcpF4E2M1Alg):
        _compute_b16_ocp_exp_and_scaling(ctx, max_ptr, exp_ptr, scaling_ptr, off, rem, dtype)
    else:
        _compute_b16_nv_exp_and_scaling(ctx, max_ptr, exp_ptr, scaling_ptr, off, rem, dtype)


def _extract_b8_exponent_and_scaling_vl(alg, max_ptr, exp_ptr, scaling_ptr, off, rem, dtype):
    if alg == OcpF8E4M3Alg:
        if dtype == pto.f32:
            _extract_f32_ocp_exp_and_scaling_core(max_ptr, exp_ptr, scaling_ptr, off, rem)
        else:
            _extract_mx_ocp_exponent_and_scaling_vl(max_ptr, exp_ptr, scaling_ptr, off, rem, _OCP_MXFP8_SPEC, dtype)
    elif alg == OcpF4E2M1Alg:
        _extract_mx_ocp_exponent_and_scaling_vl(max_ptr, exp_ptr, scaling_ptr, off, rem, _OCP_MXFP4_SPEC, dtype)
    elif alg == NvF8E4M3Alg:
        if dtype == pto.f32:
            _extract_f32_nv_exp_and_scaling_core(max_ptr, exp_ptr, scaling_ptr, off, rem)
        else:
            _extract_b16_nv_exponent_and_scaling_core(max_ptr, exp_ptr, scaling_ptr, off, rem, _NV_MXFP8_SPEC, dtype)
    elif alg == NvF4E2M1Alg:
        if dtype == pto.f32:
            _extract_f32_nv_exp_and_scaling_core(max_ptr, exp_ptr, scaling_ptr, off, rem)
        else:
            _extract_b16_nv_exponent_and_scaling_core(max_ptr, exp_ptr, scaling_ptr, off, rem, _NV_MXFP4_SPEC, dtype)


def _calc_quantized_fp4e2m1_values_dn_bf16(src_ptr, scaling_ptr, dst_ptr, valid_rows, valid_cols, src_static_cols, dst_static_cols, lanes_b16):
    grp_size = 32
    num_vls_per_row = _ceil_division(valid_cols, lanes_b16)
    num_grps_per_col = _ceil_division(valid_rows, grp_size)
    preg_cols_b16 = valid_cols
    for i in range(num_vls_per_row):
        vl_start = i * lanes_b16
        preg_b16, preg_cols_b16 = pto.make_mask(pto.bf16, preg_cols_b16)
        for j in range(num_grps_per_col):
            row_base = j * grp_size
            v_scaling = pto.vlds(scaling_ptr, vl_start + j * src_static_cols)
            with pto.for_(0, grp_size, step=1) as k:
                r = row_base + k
                off = r * src_static_cols + vl_start
                v_input = pto.vlds(src_ptr, off)
                v_scaled = pto.vmul(v_input, v_scaling, preg_b16)
                v_scaled_u16 = pto.vbitcast(v_scaled, pto.ui16)
                v_scaled_u16 = _saturate_bf16_nan_to_pos_inf(v_scaled_u16, preg_b16)
                v_scaled = pto.vbitcast(v_scaled_u16, pto.bf16)
                v_fp4 = pto.vcvt(v_scaled, pto.f4e2m1x2, preg_b16,
                                 rnd=pto.VcvtRoundMode.R, part=pto.VcvtPartMode.P0)
                pto.vsts(v_fp4, dst_ptr, off // 2, preg_b16,
                         dist=pto.VStoreDist.PK4_B32)


def _calc_quantized_fp4e2m1_values_dn_fp16_row(src_ptr, dst_ptr, v_sc_even, v_sc_odd, pack_index, r, vl_start, packed_bytes, src_static_cols, dst_static_cols, preg_b16, preg_b32):
    v_input = pto.vlds(src_ptr, r * src_static_cols + vl_start)
    v_f32_even = pto.vcvt(v_input, pto.f32, preg_b16, part=pto.VcvtPartMode.EVEN)
    v_f32_odd = pto.vcvt(v_input, pto.f32, preg_b16, part=pto.VcvtPartMode.ODD)
    v_f32_even = pto.vmul(v_f32_even, v_sc_even, preg_b32)
    v_f32_odd = pto.vmul(v_f32_odd, v_sc_odd, preg_b32)
    v_even_code = _calc_e2m1_signed_code_i32(v_f32_even, preg_b32)
    v_odd_code = _calc_e2m1_signed_code_i32(v_f32_odd, preg_b32)
    v_packed = _pack_e2m1_signed_code_bytes(v_even_code, v_odd_code, pack_index, preg_b32)
    dst_byte_offset = r * src_static_cols + vl_start
    preg_b8, _ = pto.make_mask(pto.ui8, packed_bytes)
    if (src_static_cols // 2) % 32 == 0:
        pto.vsts(v_packed, dst_ptr, dst_byte_offset // 2, preg_b8, dist=pto.VStoreDist.NORM_B8)
    else:
        scratch_ptr = pto.addptr(dst_ptr, src_static_cols // 2)
        pto.vsts(v_packed, scratch_ptr, 0, preg_b8, dist=pto.VStoreDist.NORM_B8)
        pto.mem_bar(pto.BarrierType.VST_VLD)
        v_packed_copy = pto.vlds(scratch_ptr, 0)
        dst_write_ptr_u8 = pto.castptr(pto.addptr(dst_ptr, dst_byte_offset // 2), pto.ptr(pto.ui8, "ub"))
        ureg_out = pto.init_align()
        ureg_out = pto.vstus(ureg_out, packed_bytes, v_packed_copy, dst_write_ptr_u8)
        pto.vstas(ureg_out, dst_write_ptr_u8, 0)


def _calc_quantized_fp4e2m1_values_dn_fp16(src_ptr, scaling_ptr, dst_ptr, valid_rows, valid_cols, src_static_cols, dst_static_cols, lanes_b16):
    grp_size = 32
    num_vls_per_row = _ceil_division(valid_cols, lanes_b16)
    num_grps_per_col = _ceil_division(valid_rows, grp_size)
    preg_cols_b16 = valid_cols
    preg_cols_b32 = valid_cols
    scaling_u16_ptr = pto.castptr(scaling_ptr, pto.ptr(pto.ui16, "ub"))
    pack_index = pto.vci(pto.i8(0), "ASC")
    pack_index_i16 = pto.vbitcast(pack_index, pto.si16)
    pack_index_i16 = pto.vmuls(pack_index_i16, pto.si16(4), pto.pset_b16(pto.PAT.ALL))
    pack_index = pto.vbitcast(pack_index_i16, pto.ui8)
    for i in range(num_vls_per_row):
        vl_start = i * lanes_b16
        packed_bytes = preg_cols_b32
        preg_b16, preg_cols_b16 = pto.make_mask(pto.f16, preg_cols_b16)
        preg_b32, preg_cols_b32 = pto.make_mask(pto.f32, preg_cols_b32)
        packed_bytes = packed_bytes - preg_cols_b32
        for j in range(num_grps_per_col):
            row_base = j * grp_size
            v_scaling_u16 = pto.vlds(scaling_u16_ptr, vl_start + j * src_static_cols)
            v_scaling_bf16 = pto.vbitcast(v_scaling_u16, pto.bf16)
            v_sc_even = pto.vcvt(v_scaling_bf16, pto.f32, preg_b16, part=pto.VcvtPartMode.EVEN)
            v_sc_odd = pto.vcvt(v_scaling_bf16, pto.f32, preg_b16, part=pto.VcvtPartMode.ODD)
            with pto.for_(0, grp_size, step=1) as k:
                r = row_base + k
                _calc_quantized_fp4e2m1_values_dn_fp16_row(src_ptr, dst_ptr, v_sc_even, v_sc_odd, pack_index, r, vl_start, packed_bytes, src_static_cols, dst_static_cols, preg_b16, preg_b32)


def _tquant_mxfp4_e2m1_dn(src_ptr, dst_ptr, exp_ptr, max_ptr, scaling_ptr, dtype, valid_rows, valid_cols, src_static_cols, dst_static_cols, scale_alg="ocp", interleave=False, exp_col_stride=None):
    alg = NvF4E2M1Alg if scale_alg == "nv" else OcpF4E2M1Alg
    lanes_b16 = pto.elements_per_vreg(dtype)
    extract_elements_per_vl = pto.elements_per_vreg(pto.f32) if scale_alg == "nv" else lanes_b16

    _abs_reduce_max_dn(src_ptr, max_ptr, valid_rows, valid_cols, src_static_cols, lanes_b16, dtype, scale_alg)
    pto.mem_bar(pto.BarrierType.VST_VLD)
    if interleave:
        _extract_dn_interleaved_exponent_and_scaling(max_ptr, exp_ptr, dst_ptr, scaling_ptr, valid_rows, valid_cols, src_static_cols, exp_col_stride, extract_elements_per_vl, alg, dtype)
    else:
        _extract_dn_linear_exponent_and_scaling(max_ptr, exp_ptr, scaling_ptr, valid_rows, valid_cols, src_static_cols, extract_elements_per_vl, alg, dtype)
    pto.mem_bar(pto.BarrierType.VST_VLD)
    if dtype == pto.f16:
        _calc_quantized_fp4e2m1_values_dn_fp16(src_ptr, scaling_ptr, dst_ptr, valid_rows, valid_cols, src_static_cols, dst_static_cols, lanes_b16)
    else:
        _calc_quantized_fp4e2m1_values_dn_bf16(src_ptr, scaling_ptr, dst_ptr, valid_rows, valid_cols, src_static_cols, dst_static_cols, lanes_b16)




@tilelib.tile_template(op="pto.tquant.mx", target="a5", name="template_tquant_mxfp8_f32_dn",
    dtypes=[("f32", "ui8", "ui8", "f32", "f32")],
    iteration_axis="none", op_engine="vector", op_class="elementwise",
    constraints=[_mx_dn_tiles], id=5, loop_depth=2, is_post_update=False, tags=_MX_TAGS + ("fp8", "f32", "dn"))
def template_tquant_mxfp8_f32_dn(src: pto.Tile, dst: pto.Tile, exp: pto.Tile,
                                 max_tile: pto.Tile, scaling: pto.Tile):
    """MXFP8 DN: f32 → fp8."""
    scale_alg = pto.get_op_attr("quant_scale_alg", "ocp")
    interleave = pto.get_op_attr("interleave") == "true"
    valid_rows, valid_cols, static_cols = _prepare_dn_quant(src, pto.f32)
    dst_static_cols = dst.shape[1]
    src_ptr, dst_ptr, max_ptr, exp_ptr, scaling_ptr = _prepare_nd_ptrs(src, dst, max_tile, exp, scaling)
    exp_col_stride = exp.shape[1] if interleave else static_cols
    _tquant_mxfp8_dn_f32(src_ptr, dst_ptr, exp_ptr, max_ptr, scaling_ptr, valid_rows, valid_cols, static_cols, dst_static_cols, scale_alg, interleave, exp_col_stride)


# ════════════════════════════════════════════════════════════════
#  B16 (BF16) MXFP8 helpers
# ════════════════════════════════════════════════════════════════

_BLOCK_BYTE_SIZE = 32
_BF16_ABS_MASK = 0x7FFF
_FP16_EXP_MASK = 0x7C00
_FP16_MANTISSA_MASK = 0x03FF
_FP16_INF_BITS = 0x7C00
_BF16_INF_BITS = 0x7F80
_BF16_NAN_BITS = 0x7FC0

def _fp16_to_bf16_preserve_special(src0, src1, preg_vl0, preg_vl1):
    """

    Converts fp16→bf16 via the vcvt(ROUND_Z) hardware instruction, then
    overrides Inf/NaN results with canonical bf16 sentinels (0x7F80/0x7FC0)
    so stage-2 can detect them via bf16 exp==0x7F80.
    """
    v_abs_mask = pto.vbr(pto.ui16(_BF16_ABS_MASK))
    v_fp16_exp_mask = pto.vbr(pto.ui16(_FP16_EXP_MASK))
    v_fp16_mant_mask = pto.vbr(pto.ui16(_FP16_MANTISSA_MASK))
    v_bf16_inf = pto.vbr(pto.ui16(_BF16_INF_BITS))
    v_bf16_nan = pto.vbr(pto.ui16(_BF16_NAN_BITS))
    src0_u16 = pto.vbitcast(src0, pto.ui16)
    src1_u16 = pto.vbitcast(src1, pto.ui16)
    v_abs_0 = pto.vand(src0_u16, v_abs_mask, preg_vl0)
    v_abs_1 = pto.vand(src1_u16, v_abs_mask, preg_vl1)
    v_exp_0 = pto.vand(v_abs_0, v_fp16_exp_mask, preg_vl0)
    v_exp_1 = pto.vand(v_abs_1, v_fp16_exp_mask, preg_vl1)
    v_mant_0 = pto.vand(v_abs_0, v_fp16_mant_mask, preg_vl0)
    v_mant_1 = pto.vand(v_abs_1, v_fp16_mant_mask, preg_vl1)
    preg_special_0 = pto.vcmps(v_exp_0, pto.ui16(_FP16_EXP_MASK), preg_vl0, pto.CmpMode.EQ)
    preg_special_1 = pto.vcmps(v_exp_1, pto.ui16(_FP16_EXP_MASK), preg_vl1, pto.CmpMode.EQ)
    preg_nan_0 = pto.vcmps(v_mant_0, pto.ui16(0), preg_special_0, pto.CmpMode.NE)
    preg_nan_1 = pto.vcmps(v_mant_1, pto.ui16(0), preg_special_1, pto.CmpMode.NE)
    preg_inf_0 = pto.vcmps(v_abs_0, pto.ui16(_FP16_INF_BITS), preg_vl0, pto.CmpMode.EQ)
    preg_inf_1 = pto.vcmps(v_abs_1, pto.ui16(_FP16_INF_BITS), preg_vl1, pto.CmpMode.EQ)
    v_bf16_0 = pto.vcvt(src0, pto.bf16, preg_vl0, rnd=pto.VcvtRoundMode.Z)
    v_bf16_1 = pto.vcvt(src1, pto.bf16, preg_vl1, rnd=pto.VcvtRoundMode.Z)
    v_bf16_0_u16 = pto.vbitcast(v_bf16_0, pto.ui16)
    v_bf16_1_u16 = pto.vbitcast(v_bf16_1, pto.ui16)
    v_bf16_0_u16 = pto.vsel(v_bf16_inf, v_bf16_0_u16, preg_inf_0)
    v_bf16_1_u16 = pto.vsel(v_bf16_inf, v_bf16_1_u16, preg_inf_1)
    v_bf16_0_u16 = pto.vsel(v_bf16_nan, v_bf16_0_u16, preg_nan_0)
    v_bf16_1_u16 = pto.vsel(v_bf16_nan, v_bf16_1_u16, preg_nan_1)
    return pto.vbitcast(v_bf16_0_u16, pto.bf16), pto.vbitcast(v_bf16_1_u16, pto.bf16)

def _apply_half_scaling_to_fp8_window(v_in_1, v_in_2, scaling_ptr, i,
                                       preg_b16_1, preg_b16_2,
                                       preg_f32_1_even, preg_f32_1_odd,
                                       preg_f32_2_even, preg_f32_2_odd):
    preg_all_b16 = pto.pset_b16(pto.PAT.ALL)
    scaling_ptr_u16 = pto.castptr(scaling_ptr, pto.ptr(pto.ui16, "ub"))
    v_scaling_u16 = pto.vlds(scaling_ptr_u16, 8 * i, dist="E2B_B16")
    v_scaling_bf16 = pto.vbitcast(v_scaling_u16, pto.bf16)
    v_scaling_f32 = pto.vcvt(v_scaling_bf16, pto.f32, preg_all_b16, part=pto.VcvtPartMode.EVEN)
    v_cvt_1 = pto.vcvt(v_in_1, pto.f32, preg_b16_1, part=pto.VcvtPartMode.EVEN)
    v_cvt_2 = pto.vcvt(v_in_1, pto.f32, preg_b16_1, part=pto.VcvtPartMode.ODD)
    v_cvt_3 = pto.vcvt(v_in_2, pto.f32, preg_b16_2, part=pto.VcvtPartMode.EVEN)
    v_cvt_4 = pto.vcvt(v_in_2, pto.f32, preg_b16_2, part=pto.VcvtPartMode.ODD)
    v_cvt_1 = pto.vmul(v_cvt_1, v_scaling_f32, preg_f32_1_even)
    v_cvt_2 = pto.vmul(v_cvt_2, v_scaling_f32, preg_f32_1_odd)
    v_cvt_3 = pto.vmul(v_cvt_3, v_scaling_f32, preg_f32_2_even)
    v_cvt_4 = pto.vmul(v_cvt_4, v_scaling_f32, preg_f32_2_odd)
    return v_cvt_1, v_cvt_2, v_cvt_3, v_cvt_4

def _abs_reduce_max_b16_dintlv_window(src_ptr, offset, remaining, lanes_b16, dtype, fp16_as_bf16_for_max=True):
    even_count = (remaining + 1) // 2
    odd_count = remaining // 2
    preg_vl0, _ = pto.make_mask(dtype, even_count)
    preg_vl1, _ = pto.make_mask(dtype, odd_count)
    v_in_1, v_in_2 = pto.vldsx2(src_ptr, offset, "DINTLV_B16")
    v_abs_mask = pto.vbr(pto.ui16(_BF16_ABS_MASK))
    is_fp16 = dtype == pto.f16
    if is_fp16 and fp16_as_bf16_for_max:
        v_bf16_1, v_bf16_2 = _fp16_to_bf16_preserve_special(v_in_1, v_in_2, preg_vl0, preg_vl1)
        v_abs_1 = pto.vand(pto.vbitcast(v_bf16_1, pto.ui16), v_abs_mask, preg_vl0)
        v_abs_2 = pto.vand(pto.vbitcast(v_bf16_2, pto.ui16), v_abs_mask, preg_vl1)
    else:
        v_abs_1 = pto.vand(pto.vbitcast(v_in_1, pto.ui16), v_abs_mask, preg_vl0)
        v_abs_2 = pto.vand(pto.vbitcast(v_in_2, pto.ui16), v_abs_mask, preg_vl1)
    v_abs_1 = pto.vmax(v_abs_1, v_abs_2, preg_vl0)
    v_max = pto.vcgmax(v_abs_1, preg_vl0)
    return v_max

def _abs_reduce_max_b16_nd(src_ptr, max_ptr, vl_count, total_count, lanes_b16, dtype, fp16_as_bf16_for_max=True):
    elements_per_dintlv = 2 * lanes_b16
    grps_per_dintlv = elements_per_dintlv // 32
    blks_per_vl = 256 // _BLOCK_BYTE_SIZE
    loop_num = _ceil_division(vl_count, 2)

    if loop_num == 1:
        remaining = min(total_count, elements_per_dintlv)
        out_count = _ceil_division(remaining, 32)
        preg_out, _ = pto.make_mask(dtype, out_count)
        v_max = _abs_reduce_max_b16_dintlv_window(src_ptr, 0, remaining, lanes_b16, dtype, fp16_as_bf16_for_max)
        pto.vsts(v_max, max_ptr, 0, preg_out, dist=pto.VStoreDist.NORM_B16)
        return

    ureg_max = pto.init_align()
    for i in range(loop_num):
        offset = i * elements_per_dintlv
        remaining = min(max(0, total_count - offset), elements_per_dintlv)
        v_max = _abs_reduce_max_b16_dintlv_window(src_ptr, offset, remaining, lanes_b16, dtype, fp16_as_bf16_for_max)
        ureg_max = pto.vstus(ureg_max, blks_per_vl, v_max, pto.addptr(max_ptr, i * grps_per_dintlv))
    pto.vstas(ureg_max, pto.addptr(max_ptr, loop_num * grps_per_dintlv), 0)

def _abs_reduce_max_b16_nd_largesizes(src_ptr, max_ptr, vl_count, total_count, lanes_b16, dtype, fp16_as_bf16_for_max=True):
    grp_size = 32
    grps_per_vl = lanes_b16 // grp_size
    num_vl_per_inner_loop = 2
    num_vl_per_outer_loop = 32
    grps_per_inner_loop = num_vl_per_inner_loop * grps_per_vl
    grps_per_outer_loop = num_vl_per_outer_loop * grps_per_vl
    blks_per_vl = 256 // _BLOCK_BYTE_SIZE
    v_abs_mask = pto.vbr(pto.ui16(_BF16_ABS_MASK))
    is_fp16 = dtype == pto.f16
    remained = total_count

    for i in range(vl_count // num_vl_per_outer_loop):
        ureg_max = pto.init_align()
        for j in range(num_vl_per_outer_loop // num_vl_per_inner_loop):
            preg_vl0, remained = pto.make_mask(dtype, remained)
            preg_vl1, remained = pto.make_mask(dtype, remained)
            offset = (i * num_vl_per_outer_loop + j * num_vl_per_inner_loop) * lanes_b16
            grp_offset = grps_per_outer_loop * i + grps_per_inner_loop * j
            v_in_1, v_in_2 = pto.vldsx2(src_ptr, offset, "DINTLV_B16")
            if is_fp16 and fp16_as_bf16_for_max:
                v_bf16_1, v_bf16_2 = _fp16_to_bf16_preserve_special(v_in_1, v_in_2, preg_vl0, preg_vl1)
                v_abs_1 = pto.vand(pto.vbitcast(v_bf16_1, pto.ui16), v_abs_mask, preg_vl0)
                v_abs_2 = pto.vand(pto.vbitcast(v_bf16_2, pto.ui16), v_abs_mask, preg_vl1)
            else:
                v_abs_1 = pto.vand(pto.vbitcast(v_in_1, pto.ui16), v_abs_mask, preg_vl0)
                v_abs_2 = pto.vand(pto.vbitcast(v_in_2, pto.ui16), v_abs_mask, preg_vl1)
            v_abs_1 = pto.vmax(v_abs_1, v_abs_2, preg_vl0)
            v_max = pto.vcgmax(v_abs_1, preg_vl0)
            ureg_max = pto.vstus(ureg_max, blks_per_vl, v_max, pto.addptr(max_ptr, grp_offset))
        pto.vstas(ureg_max, pto.addptr(max_ptr, grps_per_outer_loop * i), 0)

def _reduce_mx_b16_abs_max_flat(src_ptr, max_ptr, vl_count, total_count, lanes_b16, dtype, scale_alg):
    elements_per_large_loop = 32 * lanes_b16
    use_large = (total_count % elements_per_large_loop == 0)
    if scale_alg == "nv" and dtype == pto.f16:
        if use_large:
            _abs_reduce_max_b16_nd_largesizes(src_ptr, max_ptr, vl_count, total_count, lanes_b16, dtype, False)
        else:
            _abs_reduce_max_b16_nd(src_ptr, max_ptr, vl_count, total_count, lanes_b16, dtype, False)
    else:
        if use_large:
            _abs_reduce_max_b16_nd_largesizes(src_ptr, max_ptr, vl_count, total_count, lanes_b16, dtype, True)
        else:
            _abs_reduce_max_b16_nd(src_ptr, max_ptr, vl_count, total_count, lanes_b16, dtype, True)


@dataclass(frozen=True)
class _OcpMxFp4E2M1Spec:
    maxExp: int = 0x0100
    expNan: int = 0x00FF
    b16Nan: int = 0x7FC0


@dataclass(frozen=True)
class _NvMxFp4E2M1Spec:
    descaleMultiplier: float = 1.0 / 6.0
    b16SpecialScaleBits: int = 0x7FC0
    f32SpecialScaleBits: int = 0x7FC00000


_OCP_MXFP4_SPEC = _OcpMxFp4E2M1Spec()
_NV_MXFP4_SPEC = _NvMxFp4E2M1Spec()


@dataclass
class _B16OcpQuantCtx:
    vu16_max_exp_value: object = None
    vu16_scale_bias: object = None
    vu16_exp_nan: object = None
    vu16_nan: object = None
    vu16_exp_mask: object = None
    vu16_mantissa_mask: object = None
    preg_clamp: object = None
    preg_special: object = None
    preg_nan: object = None
    kExpMask: int = 0x7F80
    kMantissaMask: int = 0x007F
    kExpBias: int = 0x7F00
    kMaxExp: int = field(default_factory=lambda: _OCP_MXFP8_SPEC.maxExp)
    kExpNan: int = field(default_factory=lambda: _OCP_MXFP8_SPEC.expNan)
    kB16Nan: int = field(default_factory=lambda: _OCP_MXFP8_SPEC.b16Nan)
    expShift: int = 7


def _init_b16_ocp_quant_ctx(spec=None):
    ctx = _B16OcpQuantCtx()
    if spec is not None:
        ctx.kMaxExp = spec.maxExp
        ctx.kExpNan = spec.expNan
        ctx.kB16Nan = spec.b16Nan
    ctx.vu16_max_exp_value = pto.vbr(pto.ui16(ctx.kMaxExp))
    ctx.vu16_scale_bias = pto.vbr(pto.ui16(ctx.kExpBias))
    ctx.vu16_exp_nan = pto.vbr(pto.ui16(ctx.kExpNan))
    ctx.vu16_nan = pto.vbr(pto.ui16(ctx.kB16Nan))
    ctx.vu16_exp_mask = pto.vbr(pto.ui16(ctx.kExpMask))
    ctx.vu16_mantissa_mask = pto.vbr(pto.ui16(ctx.kMantissaMask))
    return ctx

def _init_ocp_exp_scaling_ctx(spec=None):
    ocp_spec = spec if spec is not None else _OCP_MXFP8_SPEC
    ctx = {}
    ctx["vu16_max_exp_value"] = pto.vbr(pto.ui16(ocp_spec.maxExp))
    ctx["vu16_scale_bias"] = pto.vbr(pto.ui16(0x7F00))
    ctx["vu16_exp_nan"] = pto.vbr(pto.ui16(ocp_spec.expNan))
    ctx["vu16_nan"] = pto.vbr(pto.ui16(ocp_spec.b16Nan))
    ctx["vu16_exp_mask"] = pto.vbr(pto.ui16(0x7F80))
    ctx["vu16_mantissa_mask"] = pto.vbr(pto.ui16(0x007F))
    return ctx

def _compute_mx_ocp_exp_scaling(ctx, v_max_abs, preg_one_b16):
    v_max_exp = pto.vand(v_max_abs, ctx["vu16_exp_mask"], preg_one_b16)
    v_mantissa = pto.vand(v_max_abs, ctx["vu16_mantissa_mask"], preg_one_b16)
    preg_special = pto.vcmps(v_max_exp, pto.ui16(0x7F80), preg_one_b16, pto.CmpMode.EQ)
    preg_nan = pto.vcmps(v_mantissa, pto.ui16(0), preg_special, pto.CmpMode.NE)
    preg_clamp = pto.vcmps(v_max_exp, pto.ui16(0x0400), preg_one_b16, pto.CmpMode.LE)
    v_max_exp = pto.vsel(ctx["vu16_max_exp_value"], v_max_exp, preg_clamp)
    v_shared_exp = pto.vsub(v_max_exp, ctx["vu16_max_exp_value"], preg_one_b16)
    v_exp_value = pto.vshrs(v_shared_exp, 7, preg_one_b16)
    v_exp_value = pto.vsel(ctx["vu16_exp_nan"], v_exp_value, preg_nan)
    v_recip_scale = pto.vsub(ctx["vu16_scale_bias"], v_shared_exp, preg_one_b16)
    v_recip_scale = pto.vsel(ctx["vu16_nan"], v_recip_scale, preg_nan)
    return v_exp_value, v_recip_scale


def _compute_b16_ocp_exp_and_scaling_regs(ctx, max_ptr, off, rem, dtype=pto.bf16):
    preg_b16, _ = pto.make_mask(dtype, rem)
    max_ptr_u16 = pto.castptr(max_ptr, pto.ptr(pto.ui16, "ub"))
    v_max_abs = pto.vlds(max_ptr_u16, off)
    v_max_exp = pto.vand(v_max_abs, ctx.vu16_exp_mask, preg_b16)
    v_mantissa = pto.vand(v_max_abs, ctx.vu16_mantissa_mask, preg_b16)
    ctx.preg_special = pto.vcmps(v_max_exp, pto.ui16(ctx.kExpMask), preg_b16, pto.CmpMode.EQ)
    ctx.preg_nan = pto.vcmps(v_mantissa, pto.ui16(0), ctx.preg_special, pto.CmpMode.NE)
    ctx.preg_clamp = pto.vcmps(v_max_exp, pto.ui16(ctx.kMaxExp), preg_b16, pto.CmpMode.LE)
    v_max_exp = pto.vsel(ctx.vu16_max_exp_value, v_max_exp, ctx.preg_clamp)
    v_shared_exp = pto.vsub(v_max_exp, ctx.vu16_max_exp_value, preg_b16)
    v_scale_value = pto.vshrs(v_shared_exp, ctx.expShift, preg_b16)
    v_scale_value = pto.vsel(ctx.vu16_exp_nan, v_scale_value, ctx.preg_nan)
    v_recip_scale = pto.vsub(ctx.vu16_scale_bias, v_shared_exp, preg_b16)
    v_recip_scale = pto.vsel(ctx.vu16_nan, v_recip_scale, ctx.preg_nan)
    return v_scale_value, v_recip_scale, preg_b16


def _compute_b16_ocp_exp_and_scaling(ctx, max_ptr, exp_ptr, scaling_ptr, off, rem, dtype=pto.bf16):
    v_scale_value, v_recip_scale, preg_b16 = _compute_b16_ocp_exp_and_scaling_regs(ctx, max_ptr, off, rem, dtype)
    exp_ptr_u16 = pto.castptr(exp_ptr, pto.ptr(pto.ui16, "ub"))
    scaling_ptr_u16 = pto.castptr(scaling_ptr, pto.ptr(pto.ui16, "ub"))
    pto.vsts(v_scale_value, exp_ptr_u16, off // pto.bytewidth(dtype), preg_b16, dist=pto.VStoreDist.PK_B16)
    pto.vsts(v_recip_scale, scaling_ptr_u16, off, preg_b16, dist=pto.VStoreDist.NORM_B16)

def _extract_e2m1_exponent_and_scaling(max_ptr, exp_ptr, scaling_ptr, exp_loop_count, num_groups, lanes_b16, dtype=pto.bf16):
    _extract_mx_ocp_exponent_and_scaling(max_ptr, exp_ptr, scaling_ptr, exp_loop_count, num_groups, lanes_b16, _OCP_MXFP4_SPEC, dtype)

def _extract_e2m1_exponent_and_scaling_nv(max_ptr, exp_ptr, scaling_ptr, num_groups, dtype=pto.bf16):
    _extract_nv_exponent_and_scaling_b16(max_ptr, exp_ptr, scaling_ptr, num_groups, _NV_MXFP4_SPEC, dtype)


def _extract_b8_exp_and_scaling(max_ptr, exp_ptr, scaling_ptr, exp_loop_count, num_groups, lanes_b16, dtype=pto.bf16):
    _extract_mx_ocp_exponent_and_scaling(max_ptr, exp_ptr, scaling_ptr, exp_loop_count, num_groups, lanes_b16, _OCP_MXFP8_SPEC, dtype)

def _extract_b8_exp_and_scaling_nv(max_ptr, exp_ptr, scaling_ptr, num_groups, dtype=pto.bf16):
    _extract_nv_exponent_and_scaling_b16(max_ptr, exp_ptr, scaling_ptr, num_groups, _NV_MXFP8_SPEC, dtype)

def _extract_mx_ocp_exponent_and_scaling(max_ptr, exp_ptr, scaling_ptr, exp_loop_count, total_count, lanes_b16, spec=None, dtype=pto.bf16):
    ctx = _init_b16_ocp_quant_ctx(spec)
    for i in range(exp_loop_count):
        off = i * lanes_b16
        rem = total_count - off if total_count > off else 0
        if rem > lanes_b16:
            rem = lanes_b16
        _compute_b16_ocp_exp_and_scaling(ctx, max_ptr, exp_ptr, scaling_ptr, off, rem, dtype)


@dataclass
class _B16NvQuantCtx:
    vb32_exp_mask: object = None
    vb32_mantissa_mask: object = None
    vb32_exp_max: object = None
    vb32_exp_nan: object = None
    vb32_special_scale: object = None
    vb32_min_rcp: object = None
    descaleMultiplier: float = field(default_factory=lambda: _NV_MXFP8_SPEC.descaleMultiplier)
    f32ExpShift: int = 23
    b16ExpShift: int = 7

def _init_b16_nv_quant_ctx(spec=None):
    ctx = _B16NvQuantCtx()
    if spec is not None:
        ctx.descaleMultiplier = spec.descaleMultiplier
        ctx._b16SpecialScaleBits = spec.b16SpecialScaleBits
    else:
        ctx._b16SpecialScaleBits = _NV_MXFP8_SPEC.b16SpecialScaleBits
    ctx.vb32_exp_mask = pto.vbr(pto.si32(0x7F800000))
    ctx.vb32_mantissa_mask = pto.vbr(pto.si32(0x007FFFFF))
    ctx.vb32_exp_nan = pto.vbr(pto.si32(0xFF))
    ctx.vb32_exp_max = pto.vbr(pto.si32(0xFE))
    ctx.vb32_special_scale = pto.vbr(pto.si32(ctx._b16SpecialScaleBits))
    ctx.vb32_min_rcp = pto.vbr(pto.si32(0x0040))
    return ctx


def _init_b16_nv_exp_scaling_ctx(spec=None):
    b16_special_scale_bits = spec.b16SpecialScaleBits if spec is not None else _NV_MXFP8_SPEC.b16SpecialScaleBits
    ctx = {}
    ctx["vb32_exp_mask"] = pto.vbr(pto.si32(0x7F800000))
    ctx["vb32_mantissa_mask"] = pto.vbr(pto.si32(0x007FFFFF))
    ctx["vb32_exp_nan"] = pto.vbr(pto.si32(0xFF))
    ctx["vb32_bf16_nan"] = pto.vbr(pto.si32(b16_special_scale_bits))
    ctx["vb32_bf16_min_rcp"] = pto.vbr(pto.si32(0x0040))
    ctx["vb32_exp_max"] = pto.vbr(pto.si32(0xFE))
    ctx["descale_multiplier"] = spec.descaleMultiplier if spec is not None else _NV_MXFP8_SPEC.descaleMultiplier
    return ctx

def _compute_b16_nv_exp_scaling(ctx, v_b16_max, preg_b16, preg_b32, dtype=pto.bf16):
    f32_exp_shift = 23
    bf16_exp_shift = 7
    v_f32_max = pto.vcvt(v_b16_max, pto.f32, preg_b16, part=pto.VcvtPartMode.EVEN)
    v_f32_max = pto.vmuls(v_f32_max, pto.f32(ctx["descale_multiplier"]), preg_b32)
    v_max_s32 = pto.vbitcast(v_f32_max, pto.si32)
    v_exp = pto.vand(v_max_s32, ctx["vb32_exp_mask"], preg_b32)
    v_mant = pto.vand(v_max_s32, ctx["vb32_mantissa_mask"], preg_b32)
    v_exp = pto.vshrs(v_exp, f32_exp_shift, preg_b32)
    v_shared_exp = v_exp
    v_shared_exp_inc = pto.vadds(v_shared_exp, 1, preg_b32)
    preg_mant_gt_zero = pto.vcmps(v_mant, pto.si32(0), preg_b32, pto.CmpMode.NE)
    preg_exp_gt_zero = pto.vcmps(v_exp, pto.si32(0), preg_b32, pto.CmpMode.GT)
    preg_exp_lt_max = pto.vcmps(v_exp, pto.si32(0xFE), preg_b32, pto.CmpMode.LT)
    preg_exp_eq_zero = pto.vcmps(v_exp, pto.si32(0), preg_b32, pto.CmpMode.EQ)
    preg_round_normal = pto.pand(preg_mant_gt_zero, preg_exp_gt_zero, preg_b32)
    preg_round_normal = pto.pand(preg_round_normal, preg_exp_lt_max, preg_b32)
    preg_mant_gt_half_subnormal = pto.vcmps(v_mant, pto.si32(0x00400000), preg_b32, pto.CmpMode.GT)
    preg_round_subnormal = pto.pand(preg_mant_gt_half_subnormal, preg_exp_eq_zero, preg_b32)
    preg_round_up = pto.por(preg_round_normal, preg_round_subnormal, preg_b32)
    v_shared_exp = pto.vsel(v_shared_exp_inc, v_shared_exp, preg_round_up)
    v_bf16_scale_bits = pto.vsub(ctx["vb32_exp_max"], v_shared_exp, preg_b32)
    v_bf16_scale_bits = pto.vshls(v_bf16_scale_bits, bf16_exp_shift, preg_b32)
    preg_exp_ff = pto.vcmps(v_exp, pto.si32(0xFF), preg_b32, pto.CmpMode.EQ)
    preg_mant_eq_zero = pto.pnot(preg_mant_gt_zero, preg_b32)
    preg_inf = pto.pand(preg_exp_ff, preg_mant_eq_zero, preg_b32)
    preg_nan = pto.pand(preg_exp_ff, preg_mant_gt_zero, preg_b32)
    v_bf16_scale_bits = pto.vsel(ctx["vb32_bf16_min_rcp"], v_bf16_scale_bits, preg_inf)
    v_shared_exp = pto.vsel(ctx["vb32_exp_max"], v_shared_exp, preg_inf)
    v_bf16_scale_bits = pto.vsel(ctx["vb32_bf16_nan"], v_bf16_scale_bits, preg_nan)
    v_shared_exp = pto.vsel(ctx["vb32_exp_nan"], v_shared_exp, preg_nan)
    return v_shared_exp, v_bf16_scale_bits


def _round_up_b16_nv_shared_exponent(v_shared_exp, v_exponent, v_mantissa, v_scaled_bits, preg_b32):
    v_shared_exp_inc = pto.vadds(v_shared_exp, 1, preg_b32)
    preg_mant_ne_zero = pto.vcmps(v_mantissa, pto.si32(0), preg_b32, pto.CmpMode.NE)
    preg_scaled_gt_half = pto.vcmps(v_scaled_bits, pto.si32(0x00400000), preg_b32, pto.CmpMode.GT)
    preg_round = pto.pand(preg_mant_ne_zero, preg_scaled_gt_half, preg_b32)
    preg_scaled_lt_max = pto.vcmps(v_scaled_bits, pto.si32(0x7F000000), preg_b32, pto.CmpMode.LT)
    preg_round = pto.pand(preg_round, preg_scaled_lt_max, preg_b32)
    v_shared_exp = pto.vsel(v_shared_exp_inc, v_shared_exp, preg_round)
    return v_shared_exp


def _compute_b16_nv_exp_and_scaling_regs(ctx, max_ptr, off, rem, dtype=pto.bf16):
    chunk_count_b16 = rem * 2
    chunk_count_b32 = rem
    preg_b16, _ = pto.make_mask(dtype, chunk_count_b16)
    preg_b32, _ = pto.make_mask(pto.f32, chunk_count_b32)
    v_b16_max = pto.vlds(max_ptr, off, dist="UNPK_B16")
    v_f32_max = pto.vcvt(v_b16_max, pto.f32, preg_b16, part=pto.VcvtPartMode.EVEN)
    v_f32_max = pto.vmuls(v_f32_max, pto.f32(ctx.descaleMultiplier), preg_b32)
    v_max_s32 = pto.vbitcast(v_f32_max, pto.si32)
    v_exponent = pto.vand(v_max_s32, ctx.vb32_exp_mask, preg_b32)
    v_mantissa = pto.vand(v_max_s32, ctx.vb32_mantissa_mask, preg_b32)
    v_exponent = pto.vshrs(v_exponent, ctx.f32ExpShift, preg_b32)
    v_shared_exp = _round_up_b16_nv_shared_exponent(v_exponent, v_exponent, v_mantissa, v_max_s32, preg_b32)
    v_bf16_scale_bits = pto.vsub(ctx.vb32_exp_max, v_shared_exp, preg_b32)
    v_bf16_scale_bits = pto.vshls(v_bf16_scale_bits, ctx.b16ExpShift, preg_b32)
    preg_inf = pto.vcmps(v_max_s32, pto.si32(0x7F800000), preg_b32, pto.CmpMode.EQ)
    v_bf16_scale_bits = pto.vsel(ctx.vb32_min_rcp, v_bf16_scale_bits, preg_inf)
    v_shared_exp = pto.vsel(ctx.vb32_exp_max, v_shared_exp, preg_inf)
    preg_nan = pto.vcmps(v_max_s32, pto.si32(0x7F800000), preg_b32, pto.CmpMode.GT)
    v_bf16_scale_bits = pto.vsel(ctx.vb32_special_scale, v_bf16_scale_bits, preg_nan)
    v_shared_exp = pto.vsel(ctx.vb32_exp_nan, v_shared_exp, preg_nan)
    return v_shared_exp, v_bf16_scale_bits, preg_b32

def _compute_b16_nv_exp_and_scaling(ctx, max_ptr, exp_ptr, scaling_ptr, off, rem, dtype=pto.bf16):
    v_shared_exp, v_bf16_scale_bits, preg_b32 = _compute_b16_nv_exp_and_scaling_regs(ctx, max_ptr, off, rem, dtype)
    exp_ptr_si32 = pto.castptr(exp_ptr, pto.ptr(pto.si32, "ub"))
    pto.vsts(v_shared_exp, exp_ptr_si32, off // 4, preg_b32, dist=pto.VStoreDist.PK4_B32)
    v_bf16_scale_bits_u16 = pto.vbitcast(v_bf16_scale_bits, pto.ui16)
    scaling_ptr_u16 = pto.castptr(scaling_ptr, pto.ptr(pto.ui16, "ub"))
    pto.vsts(v_bf16_scale_bits_u16, scaling_ptr_u16, off, preg_b32, dist=pto.VStoreDist.PK_B32)

def _extract_nv_exponent_and_scaling_b16(max_ptr, exp_ptr, scaling_ptr, total_count, spec=None, dtype=pto.bf16):
    ctx = _init_b16_nv_quant_ctx(spec)
    elements_per_chunk = pto.elements_per_vreg(pto.f32)
    loop_count = _ceil_division(total_count, elements_per_chunk)
    for i in range(loop_count):
        off = i * elements_per_chunk
        rem = total_count - off if total_count > off else 0
        if rem > elements_per_chunk:
            rem = elements_per_chunk
        _compute_b16_nv_exp_and_scaling(ctx, max_ptr, exp_ptr, scaling_ptr, off, rem, dtype)

def _calc_quantized_fp8_values_b16_window(src_ptr, scaling_ptr, dst_ptr, i, offset_b16, remaining, lanes_b16, dtype=pto.bf16):
    elements_per_vl_b8 = 256
    even_count = (remaining + 1) // 2
    odd_count = remaining // 2
    b8_count = remaining
    preg_b16_1, _ = pto.make_mask(dtype, even_count)
    preg_b16_2, _ = pto.make_mask(dtype, odd_count)
    preg_f32_1_even, _ = pto.make_mask(pto.f32, (even_count + 1) // 2)
    preg_f32_1_odd, _ = pto.make_mask(pto.f32, even_count // 2)
    preg_f32_2_even, _ = pto.make_mask(pto.f32, (odd_count + 1) // 2)
    preg_f32_2_odd, _ = pto.make_mask(pto.f32, odd_count // 2)
    preg_b8, _ = pto.make_mask(pto.ui8, b8_count)
    v_in_1, v_in_2 = pto.vldsx2(src_ptr, offset_b16, "DINTLV_B16")
    is_fp16 = dtype == pto.f16
    if is_fp16:
        v_cvt_1, v_cvt_2, v_cvt_3, v_cvt_4 = _apply_half_scaling_to_fp8_window(
            v_in_1, v_in_2, scaling_ptr, i,
            preg_b16_1, preg_b16_2,
            preg_f32_1_even, preg_f32_1_odd, preg_f32_2_even, preg_f32_2_odd)
    else:
        v_scaling = pto.vlds(scaling_ptr, 8 * i, dist="E2B_B16")
        v_out_1 = pto.vmul(v_in_1, v_scaling, preg_b16_1)
        v_out_2 = pto.vmul(v_in_2, v_scaling, preg_b16_2)
        v_cvt_1 = pto.vcvt(v_out_1, pto.f32, preg_b16_1, part=pto.VcvtPartMode.EVEN)
        v_cvt_2 = pto.vcvt(v_out_1, pto.f32, preg_b16_1, part=pto.VcvtPartMode.ODD)
        v_cvt_3 = pto.vcvt(v_out_2, pto.f32, preg_b16_2, part=pto.VcvtPartMode.EVEN)
        v_cvt_4 = pto.vcvt(v_out_2, pto.f32, preg_b16_2, part=pto.VcvtPartMode.ODD)
    v_fp8_p0 = pto.vcvt(v_cvt_1, pto.f8e4m3, preg_f32_1_even,
                        rnd=pto.VcvtRoundMode.R, sat=pto.VcvtSatMode.SAT, part=pto.VcvtPartMode.P0)
    v_fp8_p1 = pto.vcvt(v_cvt_3, pto.f8e4m3, preg_f32_2_even,
                        rnd=pto.VcvtRoundMode.R, sat=pto.VcvtSatMode.SAT, part=pto.VcvtPartMode.P1)
    v_fp8_p2 = pto.vcvt(v_cvt_2, pto.f8e4m3, preg_f32_1_odd,
                        rnd=pto.VcvtRoundMode.R, sat=pto.VcvtSatMode.SAT, part=pto.VcvtPartMode.P2)
    v_fp8_p3 = pto.vcvt(v_cvt_4, pto.f8e4m3, preg_f32_2_odd,
                        rnd=pto.VcvtRoundMode.R, sat=pto.VcvtSatMode.SAT, part=pto.VcvtPartMode.P3)
    v_fp8_p0_u16 = pto.vbitcast(v_fp8_p0, pto.ui16)
    v_fp8_p1_u16 = pto.vbitcast(v_fp8_p1, pto.ui16)
    v_fp8_p2_u16 = pto.vbitcast(v_fp8_p2, pto.ui16)
    v_fp8_p3_u16 = pto.vbitcast(v_fp8_p3, pto.ui16)
    preg_all_b16 = pto.pset_b16(pto.PAT.ALL)
    v_or1 = pto.vor(v_fp8_p0_u16, v_fp8_p1_u16, preg_all_b16)
    v_or2 = pto.vor(v_fp8_p2_u16, v_fp8_p3_u16, preg_all_b16)
    v_out_u16 = pto.vor(v_or1, v_or2, preg_all_b16)
    v_out_u8 = pto.vbitcast(v_out_u16, pto.ui8)
    pto.vsts(v_out_u8, dst_ptr, i * elements_per_vl_b8, preg_b8, dist=pto.VStoreDist.NORM_B8)

def _calc_quantized_fp8_values_b16(src_ptr, scaling_ptr, dst_ptr, total_count, lanes_b16, dtype=pto.bf16):
    elements_per_dintlv = 2 * lanes_b16
    vl_count = _ceil_division(total_count, lanes_b16)
    quant_iters = (vl_count + 1) // 2
    for i in range(quant_iters):
        offset_b16 = i * elements_per_dintlv
        remaining = total_count - offset_b16 if total_count > offset_b16 else 0
        if remaining > elements_per_dintlv:
            remaining = elements_per_dintlv
        _calc_quantized_fp8_values_b16_window(src_ptr, scaling_ptr, dst_ptr, i, offset_b16, remaining, lanes_b16, dtype)


def _calc_quantized_fp8_values_2d(src_ptr, scaling_ptr, dst_ptr, valid_rows, valid_cols, src_cols, lanes_b16, dtype=pto.bf16):
    elements_per_dintlv = 2 * lanes_b16
    groups_per_row = src_cols // 32
    loops_per_row = _ceil_division(valid_cols, elements_per_dintlv)
    for row in range(valid_rows):
        src_row_off = row * src_cols
        dst_row_off = row * src_cols
        scale_row_off = row * groups_per_row
        for i in range(loops_per_row):
            col_off = i * elements_per_dintlv
            remaining = valid_cols - col_off if valid_cols > col_off else 0
            if remaining > elements_per_dintlv:
                remaining = elements_per_dintlv
            _calc_quantized_fp8_values_b16_window(
                pto.addptr(src_ptr, src_row_off), pto.addptr(scaling_ptr, scale_row_off),
                pto.addptr(dst_ptr, dst_row_off), i, col_off, remaining, lanes_b16, dtype)


# ════════════════════════════════════════════════════════════════
#  MXFP4 E2M1 helpers
# ════════════════════════════════════════════════════════════════

_BF16_POSITIVE_INF_BITS = 0x7F80


def _saturate_bf16_nan_to_pos_inf(value, preg_b16):
    v_abs_mask = pto.vbr(pto.ui16(_BF16_ABS_MASK))
    v_inf = pto.vbr(pto.ui16(_BF16_POSITIVE_INF_BITS))
    v_abs = pto.vand(value, v_abs_mask, preg_b16)
    preg_nan = pto.vcmps(v_abs, pto.ui16(_BF16_POSITIVE_INF_BITS), preg_b16, pto.CmpMode.GT)
    return pto.vsel(v_inf, value, preg_nan)


def _calc_quantized_fp4e2m1_values_2d_packed(src_ptr, scaling_ptr, dst_ptr, max_tmp_ptr, exp_ptr,
                                              valid_rows, valid_cols, src_cols, exp_col_stride, lanes_b16, dtype=pto.bf16):
    k_group_size = 32
    k_groups_per_window = 8
    k_packed_bytes_per_group = k_group_size // 2
    groups_per_row = _ceil_division(valid_cols, k_group_size)
    windows_per_row = _ceil_division(groups_per_row, k_groups_per_window)
    for row in range(valid_rows):
        src_row = pto.addptr(src_ptr, row * src_cols)
        scale_row = pto.addptr(scaling_ptr, row * groups_per_row)
        dst_row = pto.addptr(dst_ptr, row * (src_cols // 2))
        for window in range(windows_per_row):
            group_base = window * k_groups_per_window
            groups_this_window = groups_per_row - group_base
            if groups_this_window > k_groups_per_window:
                groups_this_window = k_groups_per_window
            _build_packed_scale_window(pto.addptr(scale_row, group_base), max_tmp_ptr, groups_this_window)
            pto.mem_bar(pto.BarrierType.VST_VLD)
            if dtype == pto.f16:
                _calc_quantized_fp4e2m1_values_half(
                    pto.addptr(src_row, group_base * k_group_size), max_tmp_ptr,
                    pto.addptr(dst_row, group_base * k_packed_bytes_per_group), groups_this_window, lanes_b16)
            else:
                _calc_quantized_fp4e2m1_values_bf16(
                    pto.addptr(src_row, group_base * k_group_size), max_tmp_ptr,
                    pto.addptr(dst_row, group_base * k_packed_bytes_per_group), groups_this_window, lanes_b16)



def _calc_quantized_fp4e2m1_values_bf16_tail(src_tail_ptr, scaling_tail_ptr, dst_write_ptr, tail_groups, preg_b16_group, v_idx):
    k_group_size = 32
    k_packed_bytes_per_group = k_group_size // 2
    scaling_tail_ptr_u16 = pto.castptr(scaling_tail_ptr, pto.ptr(pto.ui16, "ub"))
    dst_write_ptr_u8 = pto.castptr(dst_write_ptr, pto.ptr(pto.ui8, "ub"))
    ureg_out = pto.init_align()
    for group in range(tail_groups):
        v_input = pto.vlds(src_tail_ptr, group * k_group_size)
        v_scale_u16 = pto.vlds(scaling_tail_ptr_u16, group, dist="BRC_B16")
        v_scale = pto.vbitcast(v_scale_u16, pto.bf16)
        v_scaled = pto.vmul(v_input, v_scale, preg_b16_group)
        v_scaled_u16 = pto.vbitcast(v_scaled, pto.ui16)
        v_scaled_u16 = _saturate_bf16_nan_to_pos_inf(v_scaled_u16, preg_b16_group)
        v_scaled = pto.vbitcast(v_scaled_u16, pto.bf16)
        v_output_p0 = pto.vcvt(v_scaled, pto.f4e2m1x2, preg_b16_group,
                               rnd=pto.VcvtRoundMode.R, part=pto.VcvtPartMode.P0)
        v_output_p0_u8 = pto.vbitcast(v_output_p0, pto.ui8)
        v_output = pto.vselr(v_output_p0_u8, v_idx)
        pto.mem_bar(pto.BarrierType.VST_VST)
        ureg_out = pto.vstus(ureg_out, k_packed_bytes_per_group, v_output, dst_write_ptr_u8)
        dst_write_ptr_u8 = pto.addptr(dst_write_ptr_u8, k_packed_bytes_per_group)
    pto.vstas(ureg_out, dst_write_ptr_u8, 0)


def _calc_quantized_fp4e2m1_values_bf16(src_ptr, scaling_ptr, dst_ptr, num_groups, lanes_b16):
    k_group_size = 32
    k_groups_per_window = 8
    k_elements_per_window = k_group_size * k_groups_per_window
    k_packed_bytes_per_window = k_elements_per_window // 2
    k_packed_bytes_per_half_window = k_packed_bytes_per_window // 2
    k_packed_bytes_per_group = k_group_size // 2

    preg_b16_window = pto.pset_b16(pto.PAT.ALL)
    preg_b16_group, _ = pto.make_mask(pto.bf16, k_group_size)

    v_idx = pto.vci(pto.i8(0), "ASC")
    v_idx_i16 = pto.vbitcast(v_idx, pto.si16)
    v_idx_i16 = pto.vmuls(v_idx_i16, pto.si16(4), pto.pset_b16(pto.PAT.ALL))
    v_idx = pto.vbitcast(v_idx_i16, pto.ui8)

    scaling_ptr_u16 = pto.castptr(scaling_ptr, pto.ptr(pto.ui16, "ub"))
    window_count = num_groups // k_groups_per_window
    for window in range(window_count):
        v_in_0, v_in_1 = pto.vldsx2(src_ptr, window * k_elements_per_window, "DINTLV_B16")
        v_scale_u16 = pto.vlds(scaling_ptr_u16, window * k_groups_per_window, dist="E2B_B16")
        v_scale = pto.vbitcast(v_scale_u16, pto.bf16)
        v_in_0 = pto.vmul(v_in_0, v_scale, preg_b16_window)
        v_in_1 = pto.vmul(v_in_1, v_scale, preg_b16_window)
        v_in_0_u16 = pto.vbitcast(v_in_0, pto.ui16)
        v_in_1_u16 = pto.vbitcast(v_in_1, pto.ui16)
        v_in_0_u16 = _saturate_bf16_nan_to_pos_inf(v_in_0_u16, preg_b16_window)
        v_in_1_u16 = _saturate_bf16_nan_to_pos_inf(v_in_1_u16, preg_b16_window)
        v_in_0 = pto.vbitcast(v_in_0_u16, pto.bf16)
        v_in_1 = pto.vbitcast(v_in_1_u16, pto.bf16)
        v_intlv_0, v_intlv_1 = pto.vintlv(v_in_0, v_in_1)
        v_output_0 = pto.vcvt(v_intlv_0, pto.f4e2m1x2, preg_b16_window,
                              rnd=pto.VcvtRoundMode.R, part=pto.VcvtPartMode.P0)
        v_output_1 = pto.vcvt(v_intlv_1, pto.f4e2m1x2, preg_b16_window,
                              rnd=pto.VcvtRoundMode.R, part=pto.VcvtPartMode.P0)
        pto.vsts(v_output_0, dst_ptr, window * k_packed_bytes_per_window, preg_b16_window,
                 dist=pto.VStoreDist.PK4_B32)
        pto.vsts(v_output_1, dst_ptr, window * k_packed_bytes_per_window + k_packed_bytes_per_half_window,
                 preg_b16_window, dist=pto.VStoreDist.PK4_B32)

    tail_groups = num_groups - window_count * k_groups_per_window
    if tail_groups == 0:
        return

    _calc_quantized_fp4e2m1_values_bf16_tail(
        pto.addptr(src_ptr, window_count * k_elements_per_window),
        pto.addptr(scaling_ptr, window_count * k_groups_per_window),
        pto.addptr(dst_ptr, window_count * k_packed_bytes_per_window),
        tail_groups, preg_b16_group, v_idx)


def _calc_quantized_fp4e2m1_values_half_tail(src_tail_ptr, scaling_tail_ptr, dst_write_ptr, tail_groups, pack_index):
    k_group_size = 32
    k_packed_bytes_per_group = k_group_size // 2
    preg_b16, _ = pto.make_mask(pto.f16, k_group_size)
    preg_f32, _ = pto.make_mask(pto.f32, k_packed_bytes_per_group)
    preg_all_b16 = pto.pset_b16(pto.PAT.ALL)
    scaling_tail_ptr_u16 = pto.castptr(scaling_tail_ptr, pto.ptr(pto.ui16, "ub"))
    dst_write_ptr_u8 = pto.castptr(dst_write_ptr, pto.ptr(pto.ui8, "ub"))
    ureg_out = pto.init_align()
    for group in range(tail_groups):
        v_input = pto.vlds(src_tail_ptr, group * k_group_size)
        v_scaling_u16 = pto.vlds(scaling_tail_ptr_u16, group, dist="BRC_B16")
        v_scaling_bf16 = pto.vbitcast(v_scaling_u16, pto.bf16)
        v_scaling_f32 = pto.vcvt(v_scaling_bf16, pto.f32, preg_all_b16, part=pto.VcvtPartMode.EVEN)
        v_even = pto.vcvt(v_input, pto.f32, preg_b16, part=pto.VcvtPartMode.EVEN)
        v_odd = pto.vcvt(v_input, pto.f32, preg_b16, part=pto.VcvtPartMode.ODD)
        v_even = pto.vmul(v_even, v_scaling_f32, preg_f32)
        v_odd = pto.vmul(v_odd, v_scaling_f32, preg_f32)
        v_even_code = _calc_e2m1_signed_code_i32(v_even, preg_f32)
        v_odd_code = _calc_e2m1_signed_code_i32(v_odd, preg_f32)
        v_output = _pack_e2m1_signed_code_bytes(v_even_code, v_odd_code, pack_index, preg_f32)
        v_output_u8 = pto.vbitcast(v_output, pto.ui8)
        pto.mem_bar(pto.BarrierType.VST_VST)
        ureg_out = pto.vstus(ureg_out, k_packed_bytes_per_group, v_output_u8, dst_write_ptr_u8)
        dst_write_ptr_u8 = pto.addptr(dst_write_ptr_u8, k_packed_bytes_per_group)
    pto.vstas(ureg_out, dst_write_ptr_u8, 0)


def _calc_quantized_fp4e2m1_values_half(src_ptr, scaling_ptr, dst_ptr, num_groups, lanes_b16):
    k_group_size = 32
    k_groups_per_window = 8
    k_elements_per_window = k_group_size * k_groups_per_window
    k_packed_bytes_per_window = k_elements_per_window // 2

    pack_index = pto.vci(pto.i8(0), "ASC")
    pack_index_i16 = pto.vbitcast(pack_index, pto.si16)
    pack_index_i16 = pto.vmuls(pack_index_i16, pto.si16(4), pto.pset_b16(pto.PAT.ALL))
    pack_index = pto.vbitcast(pack_index_i16, pto.ui8)

    window_count = num_groups // k_groups_per_window
    for window in range(window_count):
        _calc_quantized_fp4e2m1_values_half_window(src_ptr, scaling_ptr, dst_ptr, window, pack_index, lanes_b16)

    tail_groups = num_groups - window_count * k_groups_per_window
    if tail_groups == 0:
        return

    _calc_quantized_fp4e2m1_values_half_tail(
        pto.addptr(src_ptr, window_count * k_elements_per_window),
        pto.addptr(scaling_ptr, window_count * k_groups_per_window),
        pto.addptr(dst_ptr, window_count * k_packed_bytes_per_window),
        tail_groups, pack_index)


def _calc_quantized_fp4e2m1_values_half_window(src_ptr, scaling_ptr, dst_ptr, window, pack_index, lanes_b16):
    k_group_size = 32
    k_groups_per_window = 8
    k_elements_per_window = k_group_size * k_groups_per_window
    k_packed_bytes_per_window = k_elements_per_window // 2
    preg_b16, _ = pto.make_mask(pto.f16, lanes_b16)
    preg_f32, _ = pto.make_mask(pto.f32, lanes_b16 // 2)
    preg_b8, _ = pto.make_mask(pto.ui8, k_packed_bytes_per_window)
    preg_all_b16 = pto.pset_b16(pto.PAT.ALL)

    v_in_0, v_in_1 = pto.vldsx2(src_ptr, window * k_elements_per_window, "DINTLV_B16")
    scaling_ptr_u16 = pto.castptr(scaling_ptr, pto.ptr(pto.ui16, "ub"))
    v_scaling_u16 = pto.vlds(scaling_ptr_u16, window * k_groups_per_window, dist="E2B_B16")
    v_scaling_bf16 = pto.vbitcast(v_scaling_u16, pto.bf16)
    v_scaling_f32 = pto.vcvt(v_scaling_bf16, pto.f32, preg_all_b16, part=pto.VcvtPartMode.EVEN)

    v_mod_even = pto.vcvt(v_in_0, pto.f32, preg_b16, part=pto.VcvtPartMode.EVEN)
    v_mod_odd = pto.vcvt(v_in_1, pto.f32, preg_b16, part=pto.VcvtPartMode.EVEN)
    v_mod_even = pto.vmul(v_mod_even, v_scaling_f32, preg_f32)
    v_mod_odd = pto.vmul(v_mod_odd, v_scaling_f32, preg_f32)
    v_even_code = _calc_e2m1_signed_code_i32(v_mod_even, preg_f32)
    v_odd_code = _calc_e2m1_signed_code_i32(v_mod_odd, preg_f32)
    v_pair01 = _pack_e2m1_signed_code_bytes(v_even_code, v_odd_code, pack_index, preg_f32)

    v_mod_even = pto.vcvt(v_in_0, pto.f32, preg_b16, part=pto.VcvtPartMode.ODD)
    v_mod_odd = pto.vcvt(v_in_1, pto.f32, preg_b16, part=pto.VcvtPartMode.ODD)
    v_mod_even = pto.vmul(v_mod_even, v_scaling_f32, preg_f32)
    v_mod_odd = pto.vmul(v_mod_odd, v_scaling_f32, preg_f32)
    v_even_code = _calc_e2m1_signed_code_i32(v_mod_even, preg_f32)
    v_odd_code = _calc_e2m1_signed_code_i32(v_mod_odd, preg_f32)
    v_pair23 = _pack_e2m1_signed_code_bytes(v_even_code, v_odd_code, pack_index, preg_f32)

    v_output, v_scratch = pto.vintlv(v_pair01, v_pair23)
    v_output_u8 = pto.vbitcast(v_output, pto.ui8)
    dst_ptr_u8 = pto.castptr(dst_ptr, pto.ptr(pto.ui8, "ub"))
    pto.vsts(v_output_u8, dst_ptr_u8, window * k_packed_bytes_per_window, preg_b8,
             dist=pto.VStoreDist.NORM_B8)


_FP32_SIGN_BIT_OFFSET = 31
_FP32_SIGN_CLEAR_SHIFT = 1
_FP32_EXPONENT_OFFSET = 23
_FP32_EXPONENT_BIAS = 127
_FP32_POSITIVE_INF_BITS = 0x7F800000
_FP4_MAX_BIASED_EXPONENT_DELTA = 2
_FP4_MAGIC_ROUNDING_OFFSET = 23 - 1
_FP4_SIGN_BIT_MASK = 1 << 3
_FP4_MAX_MAGNITUDE_CODE = _FP4_SIGN_BIT_MASK - 1
_FP4_NEGATIVE_CODE_OFFSET = -_FP4_SIGN_BIT_MASK
_FP4_LOW_CODE_SHIFT = 32 - 4
_FP4_HIGH_CODE_SHIFT = 32 - 8


def _calc_e2m1_signed_code_i32(scaled, preg_f32):
    v_u32 = pto.vbitcast(scaled, pto.ui32)
    v_tmp = pto.vshrs(v_u32, _FP32_SIGN_BIT_OFFSET, preg_f32)
    preg_sign = pto.vcmps(v_tmp, pto.ui32(0), preg_f32, pto.CmpMode.NE)
    v_abs = pto.vshls(v_u32, _FP32_SIGN_CLEAR_SHIFT, preg_f32)
    v_abs = pto.vshrs(v_abs, _FP32_SIGN_CLEAR_SHIFT, preg_f32)
    preg_nan = pto.vcmps(v_abs, pto.ui32(_FP32_POSITIVE_INF_BITS), preg_f32, pto.CmpMode.GT)

    v_exp = pto.vshrs(v_abs, _FP32_EXPONENT_OFFSET, preg_f32)
    v_exp = pto.vbitcast(v_exp, pto.si32)
    v_exp = pto.vmaxs(v_exp, pto.si32(_FP32_EXPONENT_BIAS), preg_f32)
    v_exp = pto.vmins(v_exp, pto.si32(_FP32_EXPONENT_BIAS + _FP4_MAX_BIASED_EXPONENT_DELTA), preg_f32)

    v_tmp = pto.vadds(pto.vbitcast(v_exp, pto.si32), _FP4_MAGIC_ROUNDING_OFFSET, preg_f32)
    v_tmp = pto.vshls(v_tmp, _FP32_EXPONENT_OFFSET, preg_f32)
    v_scaled = pto.vadd(pto.vbitcast(v_abs, pto.f32), pto.vbitcast(v_tmp, pto.f32), preg_f32)
    v_abs = pto.vsub(pto.vbitcast(v_scaled, pto.ui32), pto.vbitcast(v_tmp, pto.ui32), preg_f32)

    v_exp = pto.vadds(pto.vbitcast(v_exp, pto.si32), -_FP32_EXPONENT_BIAS, preg_f32)
    v_exp = pto.vshls(v_exp, 1, preg_f32)
    v_abs = pto.vadd(pto.vbitcast(v_abs, pto.si32), v_exp, preg_f32)
    v_abs = pto.vmins(v_abs, pto.si32(_FP4_MAX_MAGNITUDE_CODE), preg_f32)

    v_signed = pto.vadds(v_abs, _FP4_NEGATIVE_CODE_OFFSET, preg_f32)
    v_signed = pto.vsel(v_signed, v_abs, preg_sign)
    v_signed = pto.vsel(v_abs, v_signed, preg_nan)

    return v_signed


def _pack_e2m1_signed_code_bytes(even_code, odd_code, pack_index, preg_f32):
    v_even_u32 = pto.vbitcast(even_code, pto.ui32)
    v_odd_u32 = pto.vbitcast(odd_code, pto.ui32)
    v_even_shifted = pto.vshls(v_even_u32, _FP4_LOW_CODE_SHIFT, preg_f32)
    v_even_shifted = pto.vshrs(v_even_shifted, _FP4_LOW_CODE_SHIFT, preg_f32)
    v_odd_shifted = pto.vshls(v_odd_u32, _FP4_LOW_CODE_SHIFT, preg_f32)
    v_odd_shifted = pto.vshrs(v_odd_shifted, _FP4_HIGH_CODE_SHIFT, preg_f32)
    v_packed = pto.vor(v_even_shifted, v_odd_shifted, preg_f32)
    v_packed_u8 = pto.vbitcast(v_packed, pto.ui8)
    return pto.vselr(v_packed_u8, pack_index)



def _extract_nv_exp_and_scaling_b16_2d_packed(max_ptr, exp_ptr, scaling_ptr, valid_rows, valid_groups_per_row, exp_col_stride, spec=None, dtype=pto.bf16):
    ctx = _init_b16_nv_exp_scaling_ctx(spec)
    preg_one_b16, _ = pto.make_mask(dtype, 1)
    preg_one_b32, _ = pto.make_mask(pto.f32, 1)
    scale_write_ptr = scaling_ptr
    ureg_scaling = pto.init_align()
    for row in range(valid_rows):
        max_read_ptr = pto.addptr(max_ptr, row * valid_groups_per_row)
        exp_write_ptr = pto.addptr(exp_ptr, row * exp_col_stride)
        ureg_exp = pto.init_align()
        for group in range(valid_groups_per_row):
            v_b16_max = pto.vlds(pto.addptr(max_read_ptr, group), 0, dist="BRC_B16")
            v_shared_exp, v_bf16_scale_bits = _compute_b16_nv_exp_scaling(ctx, v_b16_max, preg_one_b16, preg_one_b32, dtype)
            v_shared_exp_u8 = pto.vbitcast(v_shared_exp, pto.ui8)
            v_bf16_scale_bits_u16 = pto.vbitcast(v_bf16_scale_bits, pto.ui16)
            ureg_exp = pto.vstus(ureg_exp, 1, v_shared_exp_u8, exp_write_ptr)
            ureg_scaling = pto.vstus(ureg_scaling, 1, v_bf16_scale_bits_u16, scale_write_ptr)
            exp_write_ptr = pto.addptr(exp_write_ptr, 1)
            scale_write_ptr = pto.addptr(scale_write_ptr, 1)
        pto.vstas(ureg_exp, exp_write_ptr, 0)
    pto.vstas(ureg_scaling, scale_write_ptr, 0)

def _tquant_mxfp8_b16_2d(src_ptr, exp_ptr, dst_ptr, max_ptr, scaling_ptr, valid_rows, valid_cols, src_cols, lanes_b16, dtype, scale_alg="ocp", exp_loop_count=0, num_groups=0):
    if scale_alg == "nv":
        _abs_reduce_max_b16_nd_2d(src_ptr, max_ptr, valid_rows, valid_cols, src_cols, lanes_b16, dtype, scale_alg)
        pto.mem_bar(pto.BarrierType.VST_VLD)
        _extract_b8_exp_and_scaling_nv(max_ptr, exp_ptr, scaling_ptr, num_groups, dtype)
        pto.mem_bar(pto.BarrierType.VST_VLD)
        _calc_quantized_fp8_values_b16(src_ptr, scaling_ptr, dst_ptr, valid_rows * src_cols, lanes_b16, dtype)
    elif src_cols % 512 == 0:
        _abs_reduce_max_b16_nd_2d(src_ptr, max_ptr, valid_rows, valid_cols, src_cols, lanes_b16, dtype, scale_alg)
        pto.mem_bar(pto.BarrierType.VST_VLD)
        _extract_b8_exponent_and_scaling_2d(max_ptr, exp_ptr, scaling_ptr, valid_rows, valid_cols, src_cols, lanes_b16, dtype)
        pto.mem_bar(pto.BarrierType.VST_VLD)
        _calc_quantized_fp8_values_2d(src_ptr, scaling_ptr, dst_ptr, valid_rows, valid_cols, src_cols, lanes_b16, dtype)
    else:
        _abs_reduce_max_b16_nd_2d(src_ptr, max_ptr, valid_rows, valid_cols, src_cols, lanes_b16, dtype, scale_alg)
        pto.mem_bar(pto.BarrierType.VST_VLD)
        _extract_b8_exp_and_scaling(max_ptr, exp_ptr, scaling_ptr, exp_loop_count, num_groups, lanes_b16, dtype)
        pto.mem_bar(pto.BarrierType.VST_VLD)
        _calc_quantized_fp8_values_b16(src_ptr, scaling_ptr, dst_ptr, valid_rows * src_cols, lanes_b16, dtype)



