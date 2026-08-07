# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""PTODSL TileLib templates for pto.tsel."""

from ptodsl import pto
from ptodsl._ast_rewrite import rewrite_jit_function
import ptodsl.tilelib as tilelib

from ._common import ub_row_major_constraints
from ._elementwise import traversal_metadata


_DTYPES = [
    ("i8", "f32", "f32", "f32", "f32"),
    ("i8", "f16", "f16", "f16", "f16"),
    ("i8", "i8", "i8", "i8", "i8"),
]


def _f32_select_masks(raw_mask, full_mask_b16):
    select_mask = pto.pbitcast(raw_mask, pto.mask_b16)
    select_mask0, select_mask1 = pto.pintlv_b16(
        select_mask,
        full_mask_b16,
    )
    return (
        pto.pbitcast(select_mask0, pto.mask_b32),
        pto.pbitcast(select_mask1, pto.mask_b32),
    )


def _f32_paired_cols(valid_cols, lanes):
    repeat_times = (valid_cols + lanes - 1) // lanes
    return (repeat_times // 2) * lanes * 2


@rewrite_jit_function
def _emit_tsel_1d(mask, src0, src1, dst):
    dtype = dst.dtype
    valid_rows, valid_cols = dst.valid_shape
    lanes = pto.elements_per_vreg(dtype)
    total_elements = valid_rows * valid_cols
    mask_ptr = pto.castptr(mask.as_ptr(), pto.ptr(pto.ui8, "ub"))
    src0_ptr = src0.as_ptr()
    src1_ptr = src1.as_ptr()
    dst_ptr = dst.as_ptr()

    if str(dtype) == "f32":
        full_mask_b16 = pto.pset_b16(pto.PAT.ALL)
        remained = total_elements
        for offset in range(0, total_elements, lanes * 2):
            raw_mask = pto.plds(
                mask_ptr,
                offset // 8,
                dist=pto.PredicateDist.US,
            )
            select_mask0, select_mask1 = _f32_select_masks(
                raw_mask,
                full_mask_b16,
            )
            pred0, remained = pto.make_mask(dtype, remained)
            pred1, remained = pto.make_mask(dtype, remained)
            lhs0 = pto.vlds(src0_ptr, offset)
            rhs0 = pto.vlds(src1_ptr, offset)
            lhs1 = pto.vlds(src0_ptr, offset + lanes)
            rhs1 = pto.vlds(src1_ptr, offset + lanes)
            selected0 = pto.vsel(lhs0, rhs0, select_mask0)
            selected1 = pto.vsel(lhs1, rhs1, select_mask1)
            pto.vsts(selected0, dst_ptr, offset, pred0)
            pto.vsts(selected1, dst_ptr, offset + lanes, pred1)
    elif str(dtype) == "f16":
        remained = total_elements
        for offset in range(0, total_elements, lanes):
            pred, remained = pto.make_mask(dtype, remained)
            select_mask = pto.plds(
                mask_ptr,
                offset // 8,
                dist=pto.PredicateDist.US,
            )
            select_mask = pto.pbitcast(select_mask, pto.mask_b16)
            lhs = pto.vlds(src0_ptr, offset)
            rhs = pto.vlds(src1_ptr, offset)
            result = pto.vsel(lhs, rhs, select_mask)
            pto.vsts(result, dst_ptr, offset, pred)
    else:
        remained = total_elements
        for offset in range(0, total_elements, lanes):
            pred, remained = pto.make_mask(dtype, remained)
            select_mask = pto.plds(
                mask_ptr,
                offset // 8,
                dist=pto.PredicateDist.NORM,
            )
            lhs = pto.vlds(src0_ptr, offset)
            rhs = pto.vlds(src1_ptr, offset)
            result = pto.vsel(lhs, rhs, select_mask)
            pto.vsts(result, dst_ptr, offset, pred)


@rewrite_jit_function
def _emit_tsel_2d(mask, src0, src1, dst):
    dtype = dst.dtype
    valid_rows, valid_cols = dst.valid_shape
    lanes = pto.elements_per_vreg(dtype)
    mask_ptr = pto.castptr(mask.as_ptr(), pto.ptr(pto.ui8, "ub"))
    mask_stride = mask.shape[1]

    if str(dtype) == "f32":
        full_mask_b16 = pto.pset_b16(pto.PAT.ALL)
        pair_width = lanes * 2
        paired_cols = _f32_paired_cols(valid_cols, lanes)
        for row in range(0, valid_rows, 1):
            remained = valid_cols
            for col in range(0, paired_cols, pair_width):
                mask_offset = row * mask_stride + col // 8
                raw_mask = pto.plds(
                    mask_ptr,
                    mask_offset,
                    dist=pto.PredicateDist.US,
                )
                select_mask0, select_mask1 = _f32_select_masks(
                    raw_mask,
                    full_mask_b16,
                )
                pred0, remained = pto.make_mask(dtype, remained)
                pred1, remained = pto.make_mask(dtype, remained)
                lhs0 = pto.vlds(src0[row, col:])
                rhs0 = pto.vlds(src1[row, col:])
                lhs1 = pto.vlds(src0[row, col + lanes:])
                rhs1 = pto.vlds(src1[row, col + lanes:])
                selected0 = pto.vsel(lhs0, rhs0, select_mask0)
                selected1 = pto.vsel(lhs1, rhs1, select_mask1)
                pto.vsts(selected0, dst[row, col:], pred0)
                pto.vsts(selected1, dst[row, col + lanes:], pred1)

            for col in range(paired_cols, valid_cols, lanes):
                mask_offset = row * mask_stride + col // 8
                raw_mask = pto.plds(
                    mask_ptr,
                    mask_offset,
                    dist=pto.PredicateDist.US,
                )
                select_mask = pto.pbitcast(raw_mask, pto.mask_b16)
                select_mask = pto.punpack(
                    select_mask,
                    pto.PredicatePart.LOWER,
                )
                select_mask = pto.pbitcast(select_mask, pto.mask_b32)
                pred, remained = pto.make_mask(dtype, remained)
                lhs = pto.vlds(src0[row, col:])
                rhs = pto.vlds(src1[row, col:])
                selected = pto.vsel(lhs, rhs, select_mask)
                pto.vsts(selected, dst[row, col:], pred)
    elif str(dtype) == "f16":
        for row in range(0, valid_rows, 1):
            remained = valid_cols
            for col in range(0, valid_cols, lanes):
                pred, remained = pto.make_mask(dtype, remained)
                mask_offset = row * mask_stride + col // 8
                select_mask = pto.plds(
                    mask_ptr,
                    mask_offset,
                    dist=pto.PredicateDist.US,
                )
                select_mask = pto.pbitcast(select_mask, pto.mask_b16)
                lhs = pto.vlds(src0[row, col:])
                rhs = pto.vlds(src1[row, col:])
                result = pto.vsel(lhs, rhs, select_mask)
                pto.vsts(result, dst[row, col:], pred)
    else:
        for row in range(0, valid_rows, 1):
            remained = valid_cols
            for col in range(0, valid_cols, lanes):
                pred, remained = pto.make_mask(dtype, remained)
                mask_offset = row * mask_stride + col // 8
                select_mask = pto.plds(
                    mask_ptr,
                    mask_offset,
                    dist=pto.PredicateDist.NORM,
                )
                lhs = pto.vlds(src0[row, col:])
                rhs = pto.vlds(src1[row, col:])
                result = pto.vsel(lhs, rhs, select_mask)
                pto.vsts(result, dst[row, col:], pred)


def _register_tsel(*, name, traversal):
    constraints = ub_row_major_constraints(
        "src0",
        "src1",
        "tmp",
        "dst",
        require_same_valid_shape=False,
    )
    loop_depth, priority, candidate_id = traversal_metadata(traversal)
    if traversal == "1d":
        constraints.append(
            tilelib.require_predicate_select_1d(
                "mask",
                "src0",
                "src1",
                "dst",
                temporary_operand="tmp",
                memory_spaces=("ub",),
            )
        )

    @tilelib.tile_template(
        op="pto.tsel",
        target="a5",
        name=name,
        dtypes=_DTYPES,
        iteration_axis="none",
        op_engine="vector",
        op_class="other",
        constraints=constraints,
        priority=priority,
        id=candidate_id,
        loop_depth=loop_depth,
        is_post_update=False,
        tags=("select", "predicate-load"),
    )
    def template(
        mask: pto.Tile,
        src0: pto.Tile,
        src1: pto.Tile,
        tmp: pto.Tile,
        dst: pto.Tile,
    ):
        _ = tmp
        if traversal == "1d":
            _emit_tsel_1d(mask, src0, src1, dst)
        else:
            _emit_tsel_2d(mask, src0, src1, dst)

    return template


template_tsel = _register_tsel(
    name="template_tsel",
    traversal="2d",
)

template_tsel_1d = _register_tsel(
    name="template_tsel_1d",
    traversal="1d",
)
