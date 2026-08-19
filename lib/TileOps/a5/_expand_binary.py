# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Shared PTODSL templates for row/column expand binary TileOps."""

from ptodsl import pto
import ptodsl.tilelib as tilelib


NUMERIC_SIGNATURES = [
    ("i8", "i8", "i8"),
    ("i16", "i16", "i16"),
    ("i32", "i32", "i32"),
    ("f16", "f16", "f16"),
    ("bf16", "bf16", "bf16"),
    ("f32", "f32", "f32"),
]

FLOAT_SIGNATURES = [
    ("f16", "f16", "f16"),
    ("f32", "f32", "f32"),
]


def _ub_or_vec_row_major(operand_memory_spaces, operand_b_layouts, operand_s_layouts, **_):
    return (
        all(space in {"ub", "vec"} for space in operand_memory_spaces)
        and all(layout == "row_major" for layout in operand_b_layouts)
        and all(layout == "none_box" for layout in operand_s_layouts)
    )


def _row_expand_layout(src0_config, src1_config, dst_config, src1_shape=(), operand_memory_spaces=(), **_):
    if not all(space in {"ub", "vec"} for space in operand_memory_spaces):
        return False
    if not (src0_config and src1_config and dst_config):
        return False
    if src0_config.b_layout != "row_major" or src0_config.s_layout != "none_box":
        return False
    if dst_config.b_layout != "row_major" or dst_config.s_layout != "none_box":
        return False
    src1_row_major = src1_config.b_layout == "row_major"
    src1_col_major_row_scalar = (
        src1_config.b_layout == "col_major"
        and len(src1_shape) == 2
        and src1_shape[1] == 1
    )
    return (
        (src1_row_major and src1_config.s_layout == "none_box")
        or (
            src1_col_major_row_scalar
            and src1_config.s_layout in {"none_box", "row_major"}
        )
    )


def _is_unknown_dim(value):
    return value is None or value in {-1, -(2**63)}


def _known_eq(lhs, rhs) -> bool:
    return _is_unknown_dim(lhs) or _is_unknown_dim(rhs) or lhs == rhs


def _known_ge(lhs, rhs) -> bool:
    return _is_unknown_dim(lhs) or _is_unknown_dim(rhs) or lhs >= rhs


def register_row_expand_binary(*, op, name, vector_op, dtypes):
    @tilelib.tile_template(
        op=op,
        target="a5",
        name=name,
        dtypes=dtypes,
        iteration_axis="row",
        op_engine="vector",
        op_class="broadcast",
        constraints=[
            _row_expand_layout,
            _valid_row_expand_binary,
        ],
        id=0,
        loop_depth=2,
        is_post_update=False,
        tags=("row_expand", "binary"),
    )
    def template(src0: pto.Tile, src1: pto.Tile, dst: pto.Tile):
        _emit_row_expand_body(src0, src1, dst, vector_op)

    return template


def register_column_expand_binary(*, op, name, vector_op, dtypes):
    @tilelib.tile_template(
        op=op,
        target="a5",
        name=name,
        dtypes=dtypes,
        iteration_axis="column",
        op_engine="vector",
        op_class="broadcast",
        constraints=[
            _ub_or_vec_row_major,
            _valid_column_expand_binary,
        ],
        id=0,
        loop_depth=2,
        is_post_update=False,
        tags=("column_expand", "binary"),
    )
    def template(src0: pto.Tile, src1: pto.Tile, dst: pto.Tile):
        _emit_column_expand_body(src0, src1, dst, vector_op)

    return template


def register_row_expand_expdif():
    @tilelib.tile_template(
        op="pto.trowexpandexpdif",
        target="a5",
        name="template_trowexpandexpdif_f32",
        dtypes=[("f32", "f32", "f32")],
        iteration_axis="row",
        op_engine="vector",
        op_class="broadcast",
        constraints=[
            _row_expand_layout,
            _valid_row_expand_binary,
        ],
        id=0,
        loop_depth=2,
        is_post_update=False,
        tags=("row_expand", "expdif"),
    )
    def template_f32(src0: pto.Tile, src1: pto.Tile, dst: pto.Tile):
        _emit_row_expand_body(
            src0,
            src1,
            dst,
            lambda lhs, rhs, mask: pto.vexpdif(lhs, rhs, mask, pto.VcvtPartMode.EVEN),
        )

    @tilelib.tile_template(
        op="pto.trowexpandexpdif",
        target="a5",
        name="template_trowexpandexpdif_f16",
        dtypes=[("f16", "f16", "f16")],
        iteration_axis="row",
        op_engine="vector",
        op_class="broadcast",
        constraints=[
            _row_expand_layout,
            _valid_row_expand_binary,
        ],
        id=0,
        loop_depth=2,
        is_post_update=False,
        tags=("row_expand", "expdif"),
    )
    def template_f16(src0: pto.Tile, src1: pto.Tile, dst: pto.Tile):
        _emit_row_expand_body(
            src0,
            src1,
            dst,
            lambda lhs, rhs, mask: pto.vexp(pto.vsub(lhs, rhs, mask), mask),
        )

    return template_f32, template_f16


def register_column_expand_expdif():
    @tilelib.tile_template(
        op="pto.tcolexpandexpdif",
        target="a5",
        name="template_tcolexpandexpdif_f32",
        dtypes=[("f32", "f32", "f32")],
        iteration_axis="column",
        op_engine="vector",
        op_class="broadcast",
        constraints=[
            _ub_or_vec_row_major,
            _valid_column_expand_binary,
        ],
        id=0,
        loop_depth=2,
        is_post_update=False,
        tags=("column_expand", "expdif"),
    )
    def template_f32(src0: pto.Tile, src1: pto.Tile, dst: pto.Tile):
        _emit_column_expand_body(
            src0,
            src1,
            dst,
            lambda lhs, rhs, mask: pto.vexpdif(lhs, rhs, mask, pto.VcvtPartMode.ODD),
        )

    @tilelib.tile_template(
        op="pto.tcolexpandexpdif",
        target="a5",
        name="template_tcolexpandexpdif_f16",
        dtypes=[("f16", "f16", "f16")],
        iteration_axis="column",
        op_engine="vector",
        op_class="broadcast",
        constraints=[
            _ub_or_vec_row_major,
            _valid_column_expand_binary,
        ],
        id=0,
        loop_depth=2,
        is_post_update=False,
        tags=("column_expand", "expdif"),
    )
    def template_f16(src0: pto.Tile, src1: pto.Tile, dst: pto.Tile):
        _emit_column_expand_body(
            src0,
            src1,
            dst,
            lambda lhs, rhs, mask: pto.vexp(pto.vsub(lhs, rhs, mask), mask),
        )

    return template_f32, template_f16


def _emit_row_expand_body(src0, src1, dst, vector_op):
    dtype = dst.dtype
    valid_rows, valid_cols = dst.valid_shape
    lanes = pto.elements_per_vreg(dtype)
    broadcast_dist = {
        "i8": "BRC_B8",
        "i16": "BRC_B16",
        "i32": "BRC_B32",
        "f16": "BRC_B16",
        "bf16": "BRC_B16",
        "f32": "BRC_B32",
    }[str(dtype)]

    sinkhorn_grouped_form = (
        str(dtype) == "f32"
        and tuple(src0.shape) == (8, 8)
        and src0._template_static_valid_shape in {(8, 4), (8, 8)}
        and tuple(src1.shape) == (8, 1)
        and src1._template_static_valid_shape == (8, 1)
        and tuple(dst.shape) == (8, 8)
        and dst._template_static_valid_shape
        == src0._template_static_valid_shape
    )
    if sinkhorn_grouped_form:
        full_mask, _ = pto.make_mask(dtype, 64)
        lane_ids = pto.vci(pto.i32(0), "ASC")
        row_ids = pto.vshrs(lane_ids, pto.i16(3), full_mask)
        with pto.for_(0, 1, step=1):
            lhs = pto.vlds(src0[0, 0:])
            rhs = pto.vgather2_bc(src1.as_ptr(), row_ids, full_mask)
            result = vector_op(lhs, rhs, full_mask)
            pto.vsts(result, dst[0, 0:], full_mask)
        return

    with pto.for_(0, valid_rows, step=1) as row:
        col_loop = pto.for_(0, valid_cols, step=lanes).carry(remained=valid_cols)
        with col_loop:
            col = col_loop.iv
            mask, remained = pto.make_mask(dtype, col_loop.remained)
            lhs = pto.vlds(src0[row, col:])
            rhs = pto.vlds(src1[row, :], dist=broadcast_dist)
            result = vector_op(lhs, rhs, mask)
            pto.vsts(result, dst[row, col:], mask)
            col_loop.update(remained=remained)


def _emit_column_expand_body(src0, src1, dst, vector_op):
    dtype = dst.dtype
    valid_rows, valid_cols = dst.valid_shape
    lanes = pto.elements_per_vreg(dtype)

    with pto.for_(0, valid_rows, step=1) as row:
        col_loop = pto.for_(0, valid_cols, step=lanes).carry(remained=valid_cols)
        with col_loop:
            col = col_loop.iv
            mask, remained = pto.make_mask(dtype, col_loop.remained)
            lhs = pto.vlds(src0[row, col:])
            rhs = pto.vlds(src1[0, col:])
            result = vector_op(lhs, rhs, mask)
            pto.vsts(result, dst[row, col:], mask)
            col_loop.update(remained=remained)


def _valid_row_expand_binary(src0_valid_shape=(), src1_valid_shape=(), dst_valid_shape=(), **_):
    return (
        len(src0_valid_shape) == 2
        and len(src1_valid_shape) == 2
        and len(dst_valid_shape) == 2
        and _known_eq(src0_valid_shape[0], dst_valid_shape[0])
        and _known_eq(src0_valid_shape[1], dst_valid_shape[1])
        and _known_eq(src1_valid_shape[0], dst_valid_shape[0])
        and _known_ge(src1_valid_shape[1], 1)
    )


def _valid_column_expand_binary(src0_valid_shape=(), src1_valid_shape=(), dst_valid_shape=(), **_):
    return (
        len(src0_valid_shape) == 2
        and len(src1_valid_shape) == 2
        and len(dst_valid_shape) == 2
        and _known_eq(src0_valid_shape[0], dst_valid_shape[0])
        and _known_eq(src0_valid_shape[1], dst_valid_shape[1])
        and _known_ge(src1_valid_shape[0], 1)
        and _known_eq(src1_valid_shape[1], dst_valid_shape[1])
    )


__all__ = [
    "FLOAT_SIGNATURES",
    "NUMERIC_SIGNATURES",
    "register_column_expand_binary",
    "register_column_expand_expdif",
    "register_row_expand_binary",
    "register_row_expand_expdif",
]
