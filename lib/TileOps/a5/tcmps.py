# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""PTODSL TileLib templates for pto.tcmps."""

from ptodsl import pto
from ptodsl._ast_rewrite import rewrite_jit_function
import ptodsl.tilelib as tilelib

from ._elementwise import traversal_metadata


_DTYPES = [
    ("f32", "f32", "ui8"),
    ("i32", "i32", "ui8"),
    ("f16", "f16", "ui8"),
    ("i16", "i16", "ui8"),
    ("i8", "i8", "ui8"),
    ("ui8", "ui8", "ui8"),
]


def _ub_or_vec_row_major(
    operand_memory_spaces,
    operand_b_layouts,
    operand_s_layouts,
    **_,
):
    return (
        all(space in {"ub", "vec"} for space in operand_memory_spaces)
        and all(layout == "row_major" for layout in operand_b_layouts)
        and all(layout == "none_box" for layout in operand_s_layouts)
    )


@rewrite_jit_function
def _emit_tcmps_1d(src, scalar, dst):
    dtype = src.dtype
    valid_rows, valid_cols = src.valid_shape
    lanes = pto.elements_per_vreg(dtype)
    total_elements = valid_rows * valid_cols
    cmp_mode = pto.get_op_attr("cmp_mode", "eq")
    src_ptr = src.as_ptr()
    dst_ptr = dst.as_ptr()

    if str(dtype) in {"f32", "i32"}:
        remained = total_elements
        for offset in range(0, total_elements, lanes * 2):
            first_mask, remained = pto.make_mask(dtype, remained)
            first = pto.vlds(src_ptr, offset)
            first_cmp = pto.vcmps(first, scalar, first_mask, cmp_mode)
            first_cmp_b8 = pto.pbitcast(first_cmp, pto.mask_b8)

            second_mask, remained = pto.make_mask(dtype, remained)
            second = pto.vlds(src_ptr, offset + lanes)
            second_cmp = pto.vcmps(
                second,
                scalar,
                second_mask,
                cmp_mode,
            )
            second_cmp_b8 = pto.pbitcast(second_cmp, pto.mask_b8)

            packed_low, _ = pto.pdintlv_b8(
                first_cmp_b8,
                second_cmp_b8,
            )
            pto.psts(
                packed_low,
                dst_ptr,
                offset // 8,
                dist=pto.PredicateDist.PK,
            )
    elif str(dtype) in {"f16", "i16"}:
        remained = total_elements
        for offset in range(0, total_elements, lanes):
            mask, remained = pto.make_mask(dtype, remained)
            value = pto.vlds(src_ptr, offset)
            cmp = pto.vcmps(value, scalar, mask, cmp_mode)
            pto.psts(
                cmp,
                dst_ptr,
                offset // 8,
                dist=pto.PredicateDist.PK,
            )
    else:
        remained = total_elements
        for offset in range(0, total_elements, lanes):
            mask, remained = pto.make_mask(dtype, remained)
            value = pto.vlds(src_ptr, offset)
            cmp = pto.vcmps(value, scalar, mask, cmp_mode)
            pto.psts(
                cmp,
                dst_ptr,
                offset // 8,
                dist=pto.PredicateDist.NORM,
            )


@rewrite_jit_function
def _emit_tcmps_2d(src, scalar, dst):
    dtype = src.dtype
    valid_rows, valid_cols = src.valid_shape
    lanes = pto.elements_per_vreg(dtype)
    cmp_mode = pto.get_op_attr("cmp_mode", "eq")
    dst_ptr = dst.as_ptr()

    # The destination is a linear packed-predicate byte stream. Its physical
    # column count provides capacity; it is not the stride between logical
    # predicate rows. Each row starts after the packed bytes emitted for the
    # preceding row, preserving the existing TCMPS layout contract.
    if str(dtype) in {"f32", "i32"}:
        repeat_times = (valid_cols + lanes - 1) // lanes + 1
        iterations = repeat_times // 2
        packed_row_bytes = iterations * 16
        for row in range(0, valid_rows, 1):
            remained = valid_cols
            for col in range(0, iterations, 1):
                first_offset = col * lanes * 2
                second_offset = (col * 2 + 1) * lanes

                first_mask, remained = pto.make_mask(dtype, remained)
                first = pto.vlds(src[row, first_offset:])
                first_cmp = pto.vcmps(
                    first,
                    scalar,
                    first_mask,
                    cmp_mode,
                )
                first_cmp_b8 = pto.pbitcast(first_cmp, pto.mask_b8)

                second_mask, remained = pto.make_mask(dtype, remained)
                second = pto.vlds(src[row, second_offset:])
                second_cmp = pto.vcmps(
                    second,
                    scalar,
                    second_mask,
                    cmp_mode,
                )
                second_cmp_b8 = pto.pbitcast(second_cmp, pto.mask_b8)

                packed_low, _ = pto.pdintlv_b8(
                    first_cmp_b8,
                    second_cmp_b8,
                )
                store_offset = row * packed_row_bytes + col * 16
                pto.psts(
                    packed_low,
                    dst_ptr,
                    store_offset,
                    dist=pto.PredicateDist.PK,
                )
    elif str(dtype) in {"f16", "i16"}:
        bytes_per_iter = 16
        iters_per_row = (valid_cols + lanes - 1) // lanes
        for row in range(0, valid_rows, 1):
            remained = valid_cols
            for col in range(0, valid_cols, lanes):
                mask, remained = pto.make_mask(dtype, remained)
                value = pto.vlds(src[row, col:])
                cmp = pto.vcmps(value, scalar, mask, cmp_mode)
                store_offset = (
                    row * iters_per_row + col // lanes
                ) * bytes_per_iter
                pto.psts(
                    cmp,
                    dst_ptr,
                    store_offset,
                    dist=pto.PredicateDist.PK,
                )
    else:
        bytes_per_iter = 32
        iters_per_row = (valid_cols + lanes - 1) // lanes
        for row in range(0, valid_rows, 1):
            remained = valid_cols
            for col in range(0, valid_cols, lanes):
                mask, remained = pto.make_mask(dtype, remained)
                value = pto.vlds(src[row, col:])
                cmp = pto.vcmps(value, scalar, mask, cmp_mode)
                store_offset = (
                    row * iters_per_row + col // lanes
                ) * bytes_per_iter
                pto.psts(
                    cmp,
                    dst_ptr,
                    store_offset,
                    dist=pto.PredicateDist.NORM,
                )


def _register_tcmps(*, name, traversal):
    constraints = [_ub_or_vec_row_major]
    loop_depth, priority, candidate_id = traversal_metadata(traversal)
    if traversal == "1d":
        constraints.append(
            tilelib.require_predicate_compare_1d(
                "src",
                predicate_operand="dst",
                flattened_destination=True,
            )
        )

    @tilelib.tile_template(
        op="pto.tcmps",
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
        tags=("compare", "scalar", "predicate-store"),
    )
    def template(src: pto.Tile, scalar, dst: pto.Tile):
        if traversal == "1d":
            _emit_tcmps_1d(src, scalar, dst)
        else:
            _emit_tcmps_2d(src, scalar, dst)

    return template


template_tcmps = _register_tcmps(
    name="template_tcmps",
    traversal="2d",
)

template_tcmps_1d = _register_tcmps(
    name="template_tcmps_1d",
    traversal="1d",
)
