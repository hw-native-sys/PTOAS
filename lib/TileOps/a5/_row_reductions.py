# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Shared PTODSL implementations for row-wise reductions."""

from ptodsl import pto
import ptodsl.tilelib as tilelib

from ._common import NUMERIC_DTYPES, element_store_dist
from ._part import pad_max, pad_min


def _single_output_col(dst_shape=(), dst_valid_shape=(), **_):
    if len(dst_valid_shape) != 2:
        return False
    return dst_valid_shape[1] == 1 or (
        _is_unknown_dim(dst_valid_shape[1])
        and len(dst_shape) == 2
        and dst_shape[1] == 1
    )


def _ub_or_vec_row_major(operand_memory_spaces, operand_b_layouts, operand_s_layouts, **_):
    return (
        all(space in {"ub", "vec"} for space in operand_memory_spaces)
        and all(layout == "row_major" for layout in operand_b_layouts)
        and all(layout == "none_box" for layout in operand_s_layouts)
    )


def _row_reduction_layout(src_config, tmp_config, dst_config, dst_shape=(), operand_memory_spaces=(), **_):
    if not all(space in {"ub", "vec"} for space in operand_memory_spaces):
        return False
    if not (src_config and tmp_config and dst_config):
        return False
    if src_config.b_layout != "row_major" or src_config.s_layout != "none_box":
        return False

    dst_row_major = dst_config.b_layout == "row_major"
    dst_col_major_single_col = (
        dst_config.b_layout == "col_major"
        and len(dst_shape) == 2
        and dst_shape[1] == 1
    )
    return (
        (dst_row_major and dst_config.s_layout == "none_box")
        or (
            dst_col_major_single_col
            and dst_config.s_layout in {"none_box", "row_major"}
        )
    )


def _is_unknown_dim(value):
    return value is None or value in {-1, -(2**63)}


def _rowprod_reduction_steps(dtype):
    return 7 if str(dtype) in {"f16", "i16"} else 6


def _one(dtype):
    name = str(dtype)
    if name == "f32":
        return pto.f32(1.0)
    if name == "f16":
        return pto.f16(1.0)
    if name in {"i32", "si32"}:
        return pto.i32(1)
    return pto.i16(1)


def _zero(dtype):
    name = str(dtype)
    if name == "f32":
        return pto.f32(0.0)
    if name == "f16":
        return pto.f16(0.0)
    if name == "bf16":
        return pto.bf16(0.0)
    if name in {"ui32", "si32", "i32"}:
        return pto.i32(0)
    if name in {"ui16", "si16", "i16"}:
        return pto.i16(0)
    return pto.i8(0)


def register_row_extreme(*, op, name, reduce_op, combine_op):
    @tilelib.tile_template(
        op=op,
        target="a5",
        name=name,
        dtypes=[(dtype, dtype, dtype) for dtype in NUMERIC_DTYPES],
        iteration_axis="row",
        op_engine="vector",
        op_class="reduction",
        constraints=[
            _row_reduction_layout,
            _single_output_col,
        ],
        id=0,
        loop_depth=2,
        is_post_update=False,
        tags=("reduction", "row"),
    )
    def template(src: pto.Tile, tmp: pto.Tile, dst: pto.Tile):
        _ = tmp
        dtype = dst.dtype
        valid_rows, valid_cols = src.valid_shape
        src_cols = src.shape[1]
        dst_cols = dst.shape[1]
        src_ptr = src.as_ptr()
        dst_ptr = dst.as_ptr()
        lanes = pto.elements_per_vreg(dtype)
        one_mask, _ = pto.make_mask(dtype, 1)
        init = pad_min(dtype) if combine_op is pto.vmax else pad_max(dtype)

        for row in range(0, valid_rows, 1):
            remained = valid_cols
            acc = pto.vbr(init)
            for col in range(0, valid_cols, lanes):
                mask, remained = pto.make_mask(dtype, remained)
                src_addr = pto.addptr(src_ptr, row * src_cols + col)
                value = pto.vlds(src_addr, 0)
                reduced = reduce_op(value, mask)
                if str(dtype) in {"f16", "f32"}:
                    reduced = pto.vsel(reduced, acc, mask)
                acc = combine_op(acc, reduced, one_mask)
            dst_addr = pto.addptr(dst_ptr, row * dst_cols)
            pto.vsts(acc, dst_addr, 0, one_mask, dist=element_store_dist(dtype))

    return template


def register_rowsum():
    @tilelib.tile_template(
        op="pto.trowsum",
        target="a5",
        name="template_trowsum",
        dtypes=[("f16", "f16", "f16"), ("f32", "f32", "f32"), ("i16", "i16", "i16"), ("i32", "i32", "i32")],
        iteration_axis="row",
        op_engine="vector",
        op_class="reduction",
        constraints=[
            _row_reduction_layout,
            _single_output_col,
        ],
        id=0,
        loop_depth=2,
        is_post_update=False,
        tags=("reduction", "row", "sum"),
    )
    def template(src: pto.Tile, tmp: pto.Tile, dst: pto.Tile):
        _ = tmp
        dtype = dst.dtype
        valid_rows, valid_cols = src.valid_shape
        src_cols = src.shape[1]
        dst_cols = dst.shape[1]
        src_ptr = src.as_ptr()
        dst_ptr = dst.as_ptr()
        lanes = pto.elements_per_vreg(dtype)
        one_mask, _ = pto.make_mask(dtype, 1)

        if str(dtype) == "i16":
            acc_mask, _ = pto.make_mask(pto.i32, 1)
            zero = pto.vbr(pto.i32(0))
            for row in range(0, valid_rows, 1):
                remained = valid_cols
                acc = zero
                for col in range(0, valid_cols, lanes):
                    mask, remained = pto.make_mask(dtype, remained)
                    src_addr = pto.addptr(src_ptr, row * src_cols + col)
                    value = pto.vlds(src_addr, 0)
                    reduced = pto.vcadd(value, mask)
                    acc = pto.vadd(acc, reduced, acc_mask)
                converted = pto.vcvt(
                    acc,
                    dtype,
                    acc_mask,
                    sat=pto.VcvtSatMode.NOSAT,
                    part=pto.VcvtPartMode.EVEN,
                )
                dst_addr = pto.addptr(dst_ptr, row * dst_cols)
                pto.vsts(converted, dst_addr, 0, one_mask, dist=element_store_dist(dtype))
        else:
            for row in range(0, valid_rows, 1):
                remained = valid_cols
                acc = pto.vbr(_zero(dtype))
                for col in range(0, valid_cols, lanes):
                    mask, remained = pto.make_mask(dtype, remained)
                    src_addr = pto.addptr(src_ptr, row * src_cols + col)
                    value = pto.vlds(src_addr, 0)
                    reduced = pto.vcadd(value, mask)
                    acc = pto.vadd(acc, reduced, one_mask)
                dst_addr = pto.addptr(dst_ptr, row * dst_cols)
                pto.vsts(acc, dst_addr, 0, one_mask, dist=element_store_dist(dtype))

    return template


def register_rowprod():
    @tilelib.tile_template(
        op="pto.trowprod",
        target="a5",
        name="template_trowprod",
        dtypes=[("f16", "f16", "f16"), ("f32", "f32", "f32"), ("i16", "i16", "i16"), ("i32", "i32", "i32")],
        iteration_axis="row",
        op_engine="vector",
        op_class="reduction",
        constraints=[
            _row_reduction_layout,
            _single_output_col,
        ],
        id=0,
        loop_depth=2,
        is_post_update=False,
        tags=("reduction", "row", "prod"),
    )
    def template(src: pto.Tile, tmp: pto.Tile, dst: pto.Tile):
        _ = tmp
        dtype = dst.dtype
        valid_rows, valid_cols = src.valid_shape
        src_cols = src.shape[1]
        dst_cols = dst.shape[1]
        src_ptr = src.as_ptr()
        dst_ptr = dst.as_ptr()
        lanes = pto.elements_per_vreg(dtype)
        one_mask, _ = pto.make_mask(dtype, 1)
        full_mask, _ = pto.make_mask(dtype, lanes)
        one = pto.vbr(_one(dtype))

        for row in range(0, valid_rows, 1):
            remained = valid_cols
            acc = one
            for col in range(0, valid_cols, lanes):
                mask, remained = pto.make_mask(dtype, remained)
                src_addr = pto.addptr(src_ptr, row * src_cols + col)
                value = pto.vlds(src_addr, 0)
                prod = pto.vmul(acc, value, mask)
                acc = pto.vsel(prod, acc, mask)

            for _ in pto.static_range(_rowprod_reduction_steps(dtype)):
                lhs, rhs = pto.vintlv(acc, one)
                acc = pto.vmul(lhs, rhs, full_mask)
            dst_addr = pto.addptr(dst_ptr, row * dst_cols)
            pto.vsts(acc, dst_addr, 0, one_mask, dist=element_store_dist(dtype))

    return template


__all__ = ["register_row_extreme", "register_rowsum", "register_rowprod"]
