# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Shared PTODSL implementations for partition-combine TileOps."""

from ptodsl import pto
import ptodsl.tilelib as tilelib

from ._common import NUMERIC_DTYPES


def _ub_or_vec_row_major(operand_memory_spaces, operand_b_layouts, operand_s_layouts, **_):
    return (
        all(space in {"ub", "vec"} for space in operand_memory_spaces)
        and all(layout == "row_major" for layout in operand_b_layouts)
        and all(layout == "none_box" for layout in operand_s_layouts)
    )


def _valid_within(valid_shape, dst_valid_shape):
    return (
        len(valid_shape) == 2
        and len(dst_valid_shape) == 2
        and valid_shape[0] <= dst_valid_shape[0]
        and valid_shape[1] <= dst_valid_shape[1]
    )


def _valid_add_mul_partition(src0_valid_shape=(), src1_valid_shape=(), dst_valid_shape=(), **_):
    if not (
        _valid_within(src0_valid_shape, dst_valid_shape)
        and _valid_within(src1_valid_shape, dst_valid_shape)
    ):
        return False
    return src0_valid_shape == dst_valid_shape or src1_valid_shape == dst_valid_shape


def _valid_extreme_partition(src0_valid_shape=(), src1_valid_shape=(), dst_valid_shape=(), **_):
    return (
        _valid_within(src0_valid_shape, dst_valid_shape)
        and _valid_within(src1_valid_shape, dst_valid_shape)
    )


def register_part_binary(*, op, name, vector_op):
    @tilelib.tile_template(
        op=op,
        target="a5",
        name=name,
        dtypes=[(dtype, dtype, dtype) for dtype in NUMERIC_DTYPES],
        iteration_axis="none",
        op_engine="vector",
        op_class="other",
        constraints=[_ub_or_vec_row_major, _valid_add_mul_partition],
        id=0,
        loop_depth=2,
        is_post_update=False,
        tags=("partition", "binary"),
    )
    def template(src0: pto.Tile, src1: pto.Tile, dst: pto.Tile):
        dst_valid_rows, dst_valid_cols = dst.valid_shape
        src0_valid_rows, src0_valid_cols = src0.valid_shape
        src1_valid_rows, src1_valid_cols = src1.valid_shape

        src0_eq_dst = (src0_valid_rows == dst_valid_rows) & (src0_valid_cols == dst_valid_cols)
        src1_eq_dst = (src1_valid_rows == dst_valid_rows) & (src1_valid_cols == dst_valid_cols)

        with pto.if_(src0_eq_dst) as src0_full:
            with src0_full.then_:
                _emit_overlay_binary(
                    dst,
                    src0,
                    src1,
                    vector_op,
                    dst_valid_rows,
                    dst_valid_cols,
                    src1_valid_rows,
                    src1_valid_cols,
                )
            with src0_full.else_:
                with pto.if_(src1_eq_dst) as src1_full:
                    with src1_full.then_:
                        _emit_overlay_binary(
                            dst,
                            src1,
                            src0,
                            vector_op,
                            dst_valid_rows,
                            dst_valid_cols,
                            src0_valid_rows,
                            src0_valid_cols,
                        )

    return template


def register_part_extreme(*, op, name, vector_op, pad_value):
    @tilelib.tile_template(
        op=op,
        target="a5",
        name=name,
        dtypes=[(dtype, dtype, dtype) for dtype in NUMERIC_DTYPES],
        iteration_axis="none",
        op_engine="vector",
        op_class="other",
        constraints=[_ub_or_vec_row_major, _valid_extreme_partition],
        id=0,
        loop_depth=2,
        is_post_update=False,
        tags=("partition", "extreme"),
    )
    def template(src0: pto.Tile, src1: pto.Tile, dst: pto.Tile):
        dtype = dst.dtype
        dst_valid_rows, dst_valid_cols = dst.valid_shape
        src0_valid_rows, src0_valid_cols = src0.valid_shape
        src1_valid_rows, src1_valid_cols = src1.valid_shape
        lanes = pto.elements_per_vreg(dtype)

        pad_vec = pto.vbr(pad_value(dtype))
        with pto.for_(0, dst_valid_rows, step=1) as row:
            col_loop = pto.for_(0, dst_valid_cols, step=lanes).carry(remained=dst_valid_cols)
            with col_loop:
                col = col_loop.iv
                mask, remained = pto.make_mask(dtype, col_loop.remained)
                pto.vsts(pad_vec, dst[row, col:], mask)
                col_loop.update(remained=remained)

        pto.mem_bar(pto.BarrierType.VST_VLD)

        _emit_copy(dst, src0, src0_valid_rows, src0_valid_cols, start_row=0)

        pto.mem_bar(pto.BarrierType.VST_VLD)

        with pto.for_(0, src1_valid_rows, step=1) as row:
            col_loop = pto.for_(0, src1_valid_cols, step=lanes).carry(remained=src1_valid_cols)
            with col_loop:
                col = col_loop.iv
                mask, remained = pto.make_mask(dtype, col_loop.remained)
                lhs = pto.vlds(dst[row, col:])
                rhs = pto.vlds(src1[row, col:])
                pto.vsts(vector_op(lhs, rhs, mask), dst[row, col:], mask)
                col_loop.update(remained=remained)

    return template


def _emit_overlay_binary(dst, full_src, part_src, vector_op, dst_rows, dst_cols,
                         part_rows, part_cols):
    part_eq_dst = (part_rows == dst_rows) & (part_cols == dst_cols)

    with pto.if_(part_eq_dst) as full_part:
        with full_part.then_:
            _emit_binary(dst, full_src, part_src, vector_op, dst_rows, dst_cols)
        with full_part.else_:
            with pto.if_(part_cols < dst_cols) as col_partial:
                with col_partial.then_:
                    _emit_copy(dst, full_src, dst_rows, dst_cols, start_row=0)
                    _emit_binary(dst, full_src, part_src, vector_op, part_rows, part_cols)
                with col_partial.else_:
                    _emit_binary(dst, full_src, part_src, vector_op, part_rows, part_cols)
                    _emit_copy(dst, full_src, dst_rows, dst_cols, start_row=part_rows)


def _emit_binary(dst, src0, src1, vector_op, valid_rows, valid_cols):
    dtype = dst.dtype
    lanes = pto.elements_per_vreg(dtype)

    with pto.for_(0, valid_rows, step=1) as row:
        col_loop = pto.for_(0, valid_cols, step=lanes).carry(remained=valid_cols)
        with col_loop:
            col = col_loop.iv
            mask, remained = pto.make_mask(dtype, col_loop.remained)
            lhs = pto.vlds(src0[row, col:])
            rhs = pto.vlds(src1[row, col:])
            pto.vsts(vector_op(lhs, rhs, mask), dst[row, col:], mask)
            col_loop.update(remained=remained)


def _emit_copy(dst, src, valid_rows, valid_cols, start_row):
    dtype = dst.dtype
    lanes = pto.elements_per_vreg(dtype)

    with pto.for_(start_row, valid_rows, step=1) as row:
        col_loop = pto.for_(0, valid_cols, step=lanes).carry(remained=valid_cols)
        with col_loop:
            col = col_loop.iv
            mask, remained = pto.make_mask(dtype, col_loop.remained)
            pto.vsts(pto.vlds(src[row, col:]), dst[row, col:], mask)
            col_loop.update(remained=remained)


def pad_min(dtype):
    name = str(dtype)
    if name == "f32":
        return pto.f32(-3.4028234663852886e38)
    if name == "f16":
        return pto.f16(-65504.0)
    if name == "bf16":
        return pto.bf16(-3.3895313892515355e38)
    if name == "ui32":
        return pto.ui32(0)
    if name == "ui16":
        return pto.ui16(0)
    if name == "ui8":
        return pto.ui8(0)
    if name in {"i32", "si32"}:
        return pto.i32(-2147483648)
    if name in {"i16", "si16"}:
        return pto.i16(-32768)
    return pto.i8(-128)


def pad_max(dtype):
    name = str(dtype)
    if name == "f32":
        return pto.f32(3.4028234663852886e38)
    if name == "f16":
        return pto.f16(65504.0)
    if name == "bf16":
        return pto.bf16(3.3895313892515355e38)
    if name == "ui32":
        return pto.ui32(4294967295)
    if name == "ui16":
        return pto.ui16(65535)
    if name == "ui8":
        return pto.ui8(255)
    if name in {"i32", "si32"}:
        return pto.i32(2147483647)
    if name in {"i16", "si16"}:
        return pto.i16(32767)
    return pto.i8(127)


__all__ = ["pad_max", "pad_min", "register_part_binary", "register_part_extreme"]
