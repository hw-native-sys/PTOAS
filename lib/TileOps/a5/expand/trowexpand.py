# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""PTODSL TileLib template for pto.trowexpand."""

from ptodsl import pto
import ptodsl.tilelib as tilelib


def _valid_row_expand(src_valid_shape=(), dst_valid_shape=(), **_):
    return (
        len(src_valid_shape) == 2
        and len(dst_valid_shape) == 2
        and src_valid_shape[0] == dst_valid_shape[0]
        and src_valid_shape[1] >= 1
    )


@tilelib.tile_template(
    op="pto.trowexpand",
    target="a5",
    name="template_trowexpand",
    dtypes=[
        ("i8", "i8"),
        ("i16", "i16"),
        ("i32", "i32"),
        ("f16", "f16"),
        ("bf16", "bf16"),
        ("f32", "f32"),
    ],
    iteration_axis="row",
    op_engine="vector",
    op_class="broadcast",
    constraints=[
        tilelib.check_memory_space("ub"),
        tilelib.check_layout("row_major"),
        tilelib.check_s_layout("none_box"),
        _valid_row_expand,
    ],
    id=0,
    loop_depth=2,
    is_post_update=False,
    tags=("broadcast", "row"),
)
def template_trowexpand(src: pto.Tile, dst: pto.Tile):
    dtype = dst.dtype
    valid_rows, valid_cols = dst.valid_shape
    lanes = pto.elements_per_vreg(dtype)

    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, lanes):
            mask, remained = pto.make_mask(dtype, remained)
            value = pto.vlds(src[row, :])
            broadcast = pto.vdup(value, mask)
            pto.vsts(broadcast, dst[row, col:], mask)
