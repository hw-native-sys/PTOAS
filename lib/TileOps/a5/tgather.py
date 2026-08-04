# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""PTODSL TileLib templates for pto.tgather."""

from ptodsl import pto
import ptodsl.tilelib as tilelib
from ._common import NUMERIC_DTYPES
from ._common import element_store_dist
from ._row_arg import _scalar_literal


def _axis_is_row(axis_value, **_):
    return axis_value == "row"


def _axis_is_col(axis_value, **_):
    return axis_value == "col"


def _no_mask_pattern(mask_pattern=True):
    return mask_pattern is True


def _dst_shape_le_indices_shape(dst_valid_shape=(), indices_valid_shape=(), **_):
    return (
        len(dst_valid_shape) == 2
        and len(indices_valid_shape) == 2
        and dst_valid_shape[0] <= indices_valid_shape[0]
        and dst_valid_shape[1] <= indices_valid_shape[1]
    )


def gather_dtype_signatures(dtypes=NUMERIC_DTYPES):
    res = []
    for dtype in dtypes:
        for dtype_indices in ("i16", "ui16", "i32", "ui32"):
            res.append((dtype, dtype, dtype_indices))
    return res


@tilelib.tile_template(
    op="pto.tgather",
    target="a5",
    name="template_tgather",
    dtypes=gather_dtype_signatures(),
    iteration_axis="none",
    op_engine="vector",
    op_class="other",
    layouts=("row_major",),
    loop_depth=2,
    is_post_update=False,
    constraints=(_no_mask_pattern, _dst_shape_le_indices_shape),
    id=0,
)
def template_tgather(src: pto.Tile, dst: pto.Tile, indices: pto.Tile):
    dtype = dst.element_type
    dtype_indices = indices.element_type
    elem_bytes = pto.bytewidth(dtype)
    elem_bytes_s1 = pto.bytewidth(dtype_indices)
    lanes = pto.elements_per_vreg(dtype_indices)
    valid_rows, valid_cols = dst.valid_shape
    src_ptr = src.as_ptr()
    if pto.const_expr(elem_bytes == 2 and elem_bytes_s1 == 4):
        indices_type = pto.ui32
        mask_elem = indices_type
    elif pto.const_expr(elem_bytes == 1):
        indices_type = pto.ui16
        lanes = pto.elements_per_vreg(dtype) >> 1
        result_elem = pto.ui16 if pto.const_expr(str(dtype) in ("ui8",)) else pto.i16
        result_ty = pto.vreg_type(lanes, result_elem)
        mask_elem = result_elem
    elif pto.const_expr(elem_bytes == 4):
        indices_type = pto.ui32
        mask_elem = indices_type
    else:
        indices_type = pto.ui16
        mask_elem = indices_type
    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, lanes):
            indices_reg = pto.vlds(indices[row, col:])
            mask, remained = pto.make_mask(mask_elem, remained)
            if pto.const_expr(elem_bytes == 2 and elem_bytes_s1 == 4):
                dst_reg = pto.vgather2_bc(
                    src_ptr, pto.vbitcast(indices_reg, indices_type), mask
                )
            elif pto.const_expr(elem_bytes == 1):
                dst_reg = pto.vgather2(
                    src_ptr,
                    pto.vbitcast(indices_reg, indices_type),
                    mask,
                    result_vreg_type=result_ty,
                )
            else:
                dst_reg = pto.vgather2(
                    src_ptr, pto.vbitcast(indices_reg, indices_type), mask
                )
            if pto.const_expr(elem_bytes == 1):
                pto.vsts(dst_reg, dst[row, col:], mask, dist=pto.VStoreDist.PK_B16)
            else:
                pto.vsts(dst_reg, dst[row, col:], mask)


_GATHER_MASK_DTYPES = [(dtype, dtype) for dtype in NUMERIC_DTYPES]


_MASK_PATTERN_TO_INTERLEAVE = {
    "P1111": (),
    "P0101": (True,),
    "P1010": (False,),
    "P0001": (True, True),
    "P0010": (True, False),
    "P0100": (False, True),
    "P1000": (False, False),
}

_MASK_PATTERN_TO_STRIDE = {
    "P1111": (0, 1),
    "P0101": (0, 2),
    "P1010": (1, 2),
    "P0001": (0, 4),
    "P0010": (1, 4),
    "P0100": (2, 4),
    "P1000": (3, 4),
}


@tilelib.tile_template(
    op="pto.tgather",
    target="a5",
    name="template_tgather_mask_row",
    dtypes=_GATHER_MASK_DTYPES,
    iteration_axis="none",
    op_engine="vector",
    op_class="other",
    layouts=("row_major",),
    loop_depth=2,
    is_post_update=False,
    tags=("gather", "mask"),
    constraints=(_axis_is_row,),
    id=1,
)
def template_tgather_mask_row(src: pto.Tile, dst: pto.Tile):
    mask_pattern = pto.get_op_attr("mask_pattern", "P1111")
    interleave_args = _MASK_PATTERN_TO_INTERLEAVE[mask_pattern]
    dtype = dst.dtype
    valid_rows, valid_cols = dst.valid_shape
    lanes = pto.elements_per_vreg(dtype)
    times = 1 << len(interleave_args)
    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, lanes):
            if not interleave_args:
                src_reg = pto.vlds(src[row, col:])
                mask, remained = pto.make_mask(dtype, remained)
                pto.vsts(src_reg, dst[row, col:], mask)
            elif len(interleave_args) == 1:
                reg0 = pto.vlds(src[row, col * times :])
                reg1 = pto.vlds(src[row, col * times + lanes :])
                res0, res1 = pto.vdintlv(reg0, reg1)
                if interleave_args[0]:
                    dst_reg = res0
                else:
                    dst_reg = res1
                mask, remained = pto.make_mask(dtype, remained)
                pto.vsts(dst_reg, dst[row, col:], mask)
            else:
                reg0 = pto.vlds(src[row, col * times :])
                reg1 = pto.vlds(src[row, col * times + lanes :])
                reg2 = pto.vlds(src[row, col * times + lanes * 2 :])
                reg3 = pto.vlds(src[row, col * times + lanes * 3 :])
                r0_a, r0_b = pto.vdintlv(reg0, reg1)
                r1_a, r1_b = pto.vdintlv(reg2, reg3)
                if interleave_args[1]:
                    tmp0 = r0_a
                    tmp1 = r1_a
                else:
                    tmp0 = r0_b
                    tmp1 = r1_b
                final_a, final_b = pto.vdintlv(tmp0, tmp1)
                if interleave_args[0]:
                    dst_reg = final_a
                else:
                    dst_reg = final_b
                mask, remained = pto.make_mask(dtype, remained)
                pto.vsts(dst_reg, dst[row, col:], mask)


@tilelib.tile_template(
    op="pto.tgather",
    target="a5",
    name="template_tgather_mask_col",
    dtypes=_GATHER_MASK_DTYPES,
    iteration_axis="none",
    op_engine="vector",
    op_class="other",
    layouts=("row_major",),
    loop_depth=2,
    is_post_update=False,
    tags=("gather", "mask"),
    constraints=(_axis_is_col,),
    id=2,
)
def template_tgather_mask_col(src: pto.Tile, dst: pto.Tile):
    mask_pattern = pto.get_op_attr("mask_pattern", "P1111")
    start, stride = _MASK_PATTERN_TO_STRIDE[mask_pattern]
    dtype = dst.dtype
    valid_rows, valid_cols = dst.valid_shape
    lanes = pto.elements_per_vreg(dtype)
    for row in range(0, valid_rows, 1):
        remained = valid_cols
        row_src = row * stride + start
        for col in range(0, valid_cols, lanes):
            src_reg = pto.vlds(src[row_src, col:])
            mask, remained = pto.make_mask(dtype, remained)
            pto.vsts(src_reg, dst[row, col:], mask)


_CMP_GATHER_SRC_DTYPES = (
    "f32",
    "i32",
    "ui32",
    "i16",
    "ui16",
    "f16",
    "bf16",
    "i8",
    "ui8",
)
_CMP_GATHER_DST_DTYPES = ("i32", "ui32")
_CMP_GATHER_K_DTYPES = ("ui16", "ui32")


def _cmp_gather_dtype_signatures():
    res = []
    for src in _CMP_GATHER_SRC_DTYPES:
        res.append((src, "i32", "i32", "ui8", src))
    return res


def _has_cmp_mode(cmp_mode=None, **_):
    return cmp_mode is not None


@tilelib.tile_template(
    op="pto.tgather",
    target="a5",
    name="template_tgather_cmp",
    dtypes=_cmp_gather_dtype_signatures(),
    iteration_axis="none",
    op_engine="vector",
    op_class="other",
    layouts=("row_major",),
    loop_depth=2,
    is_post_update=False,
    tags=("gather", "cmp"),
    constraints=(_has_cmp_mode,),
    id=3,
)
def template_tgather_cmp(
    src: pto.Tile,
    dst: pto.Tile,
    cdst: pto.Tile,
    tmp: pto.Tile,
    k_value,
):
    cmp_mode = pto.get_op_attr("cmp_mode", "gt")
    offset = int(pto.get_op_attr("offset", 0))
    dtype = src.dtype
    elem_bytes = pto.bytewidth(dtype)
    src_valid_rows, src_valid_cols = src.valid_shape
    lanes_b32 = pto.elements_per_vreg(pto.i32)
    dst_ptr = dst.as_ptr()
    cdst_ptr = cdst.as_ptr()
    k_scalar = k_value
    with pto.vecscope():
        full_mask_b32 = pto.make_mask(pto.i32, pto.PAT.ALL)
        index = pto.vci(offset, "ASC")
        add_offset = pto.vbr(64)
        pto.sprclr("AR")

        dst_cols = dst.shape[1]

        if pto.const_expr(str(dtype) in ("f32",)):
            k_vec = pto.vbr(k_scalar)
            for row in range(0, src_valid_rows, 1):
                align = pto.init_align()
                remained = src_valid_cols
                row_dst_ptr = pto.addptr(dst_ptr, row * dst_cols * 4)
                for col in range(0, src_valid_cols, lanes_b32):
                    src_reg = pto.vlds(src[row, col:])
                    mask, remained = pto.make_mask(pto.i32, remained)
                    src_s32 = pto.vcvt(
                        src_reg, pto.i32, full_mask_b32, rnd=pto.VcvtRoundMode.R,
                        sat=pto.VcvtSatMode.NOSAT,
                    )
                    src_u32 = pto.vbitcast(src_s32, pto.ui32)
                    cmp_mask = pto.vcmp(src_u32, pto.vbitcast(k_vec, pto.ui32), mask, cmp_mode)
                    sqz = pto.vsqz(index, cmp_mask)
                    align = pto.vstur(align, sqz, row_dst_ptr, "POST_UPDATE")
                    index = pto.vadd(index, add_offset, full_mask_b32)
                pto.vstar(align, row_dst_ptr)
                pto.sprsts("AR", cdst_ptr, row * 4)
                pto.sprclr("AR")
                pto.mem_bar(pto.BarrierType.VST_VST)

        elif pto.const_expr(elem_bytes == 4):
            for row in range(0, src_valid_rows, 1):
                align = pto.init_align()
                remained = src_valid_cols
                row_dst_ptr = pto.addptr(dst_ptr, row * dst_cols * 4)
                for col in range(0, src_valid_cols, lanes_b32):
                    src_reg = pto.vlds(src[row, col:])
                    mask, remained = pto.make_mask(pto.i32, remained)
                    cmp_mask = pto.vcmps(src_reg, k_scalar, mask, cmp_mode)
                    sqz = pto.vsqz(index, cmp_mask)
                    align = pto.vstur(align, sqz, row_dst_ptr, "POST_UPDATE")
                    index = pto.vadd(index, add_offset, full_mask_b32)
                pto.vstar(align, row_dst_ptr)
                pto.sprsts("AR", cdst_ptr, row * 4)
                pto.sprclr("AR")
                pto.mem_bar(pto.BarrierType.VST_VST)

        elif pto.const_expr(str(dtype) in ("i16", "ui16", "f16", "bf16")):
            full_mask_b16 = pto.make_mask(dtype, pto.PAT.ALL)
            for row in range(0, src_valid_rows, 1):
                align = pto.init_align()
                remained = src_valid_cols
                row_dst_ptr = pto.addptr(dst_ptr, row * dst_cols * 4)
                for col in range(0, src_valid_cols, lanes_b32):
                    src_reg = pto.vlds(src[row, col:], dist="UNPK_B16")
                    src_f32 = pto.vcvt(
                        src_reg, pto.f32, full_mask_b16, part=pto.VcvtPartMode.EVEN
                    )
                    mask, remained = pto.make_mask(pto.i32, remained)
                    cmp_mask = pto.vcmps(src_f32, k_scalar, mask, cmp_mode)
                    sqz = pto.vsqz(index, cmp_mask)
                    align = pto.vstur(align, sqz, row_dst_ptr, "POST_UPDATE")
                    index = pto.vadd(index, add_offset, full_mask_b32)
                pto.vstar(align, row_dst_ptr)
                pto.sprsts("AR", cdst_ptr, row * 4)
                pto.sprclr("AR")
                pto.mem_bar(pto.BarrierType.VST_VST)

        elif pto.const_expr(elem_bytes == 1):
            full_mask_b8 = pto.make_mask(pto.i8, pto.PAT.ALL)
            v_zero = pto.vdup(pto.ui8(0), full_mask_b8)
            for row in range(0, src_valid_rows, 1):
                align = pto.init_align()
                remained = src_valid_cols
                row_dst_ptr = pto.addptr(dst_ptr, row * dst_cols * 4)
                for col in range(0, src_valid_cols, lanes_b32):
                    src_reg = pto.vlds(src[row, col:], dist="UNPK_B8")
                    src_u8 = pto.vbitcast(src_reg, pto.ui8)
                    intlv1, intlv2 = pto.vintlv(src_u8, v_zero)
                    src_i8_1 = pto.vbitcast(intlv1, pto.si8)
                    src_i8_2 = pto.vbitcast(intlv2, pto.si8)
                    score_u32_0 = pto.vcvt(
                        src_i8_1, pto.i32, full_mask_b8, part=pto.VcvtPartMode.P0
                    )
                    score_u32_1 = pto.vcvt(
                        src_i8_2, pto.i32, full_mask_b8, part=pto.VcvtPartMode.P0
                    )
                    mask0, remained = pto.make_mask(pto.i32, remained)
                    cmp_mask0 = pto.vcmps(
                        pto.vbitcast(score_u32_0, pto.ui32),
                        k_scalar,
                        mask0,
                        cmp_mode,
                    )
                    sqz0 = pto.vsqz(index, cmp_mask0)
                    align = pto.vstur(align, sqz0, row_dst_ptr, "POST_UPDATE")
                    index = pto.vadd(index, add_offset, full_mask_b32)
                    mask1, remained = pto.make_mask(pto.i32, remained)
                    cmp_mask1 = pto.vcmps(
                        pto.vbitcast(score_u32_1, pto.ui32),
                        k_scalar,
                        mask1,
                        cmp_mode,
                    )
                    sqz1 = pto.vsqz(index, cmp_mask1)
                    align = pto.vstur(align, sqz1, row_dst_ptr, "POST_UPDATE")
                    index = pto.vadd(index, add_offset, full_mask_b32)
                pto.vstar(align, row_dst_ptr)
                pto.sprsts("AR", cdst_ptr, row * 4)
                pto.sprclr("AR")
                pto.mem_bar(pto.BarrierType.VST_VST)
