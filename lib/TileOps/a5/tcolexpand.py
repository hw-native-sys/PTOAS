# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""PTODSL TileLib template for pto.tcolexpand."""

from ptodsl import pto
import ptodsl.tilelib as tilelib


def _valid_column_expand(src_valid_shape=(), dst_valid_shape=(), **_):
    return (
        len(src_valid_shape) == 2
        and len(dst_valid_shape) == 2
        and src_valid_shape[0] >= 1
        and src_valid_shape[1] == dst_valid_shape[1]
    )


def _ub_or_vec_row_major(operand_memory_spaces, operand_b_layouts, operand_s_layouts, **_):
    return (
        all(space in {"ub", "vec"} for space in operand_memory_spaces)
        and all(layout == "row_major" for layout in operand_b_layouts)
        and all(layout == "none_box" for layout in operand_s_layouts)
    )


@tilelib.tile_template(
    op="pto.tcolexpand",
    target="a5",
    name="template_tcolexpand",
    dtypes=[
        ("i8", "i8"),
        ("i16", "i16"),
        ("i32", "i32"),
        ("f16", "f16"),
        ("bf16", "bf16"),
        ("f32", "f32"),
    ],
    iteration_axis="column",
    op_engine="vector",
    op_class="broadcast",
    constraints=[
        _ub_or_vec_row_major,
        _valid_column_expand,
    ],
    id=0,
    loop_depth=2,
    is_post_update=False,
    tags=("broadcast", "column"),
)
def template_tcolexpand(src: pto.Tile, dst: pto.Tile):
    dtype = dst.dtype
    valid_rows, valid_cols = dst.valid_shape
    lanes = pto.elements_per_vreg(dtype)

    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, lanes):
            mask, remained = pto.make_mask(dtype, remained)
            value = pto.vlds(src[0, col:])
            pto.vsts(value, dst[row, col:], mask)


from ._vmi_common import (  # noqa: E402
    canonical_vmi_template,
    col_expand_vmi_constraint,
    emit_col_expand_vmi,
)


@canonical_vmi_template(
    target="a5",
    op="tcolexpand",
    name="vmi_tcolexpand",
    dtypes=(("f32", "f32"),),
    constraints=(col_expand_vmi_constraint,),
    requires_full_physical_row=False,
    tags=("supports_partial_valid_shape",),
)
def vmi_tcolexpand(src: pto.Tile, dst: pto.Tile):
    emit_col_expand_vmi(src, dst)
