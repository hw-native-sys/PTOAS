# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""PTODSL TileLib template for ``pto.tconcat`` (ports A5 CCE ``TConcat``).

Column-wise concat: ``dst[:, 0:c0] = src0``; ``dst[:, c0:] = src1``. The CCE
``vscatter`` for the offset half becomes a ``dst[row, c0 + col:]`` slice (cf.
``textract``), so both halves are plain ``vlds``/``vsts`` chunked copies.
"""

from ptodsl import pto
import ptodsl.tilelib as tilelib

from ._common import same_dtype_signatures


def _concat_layout(operand_memory_spaces, operand_b_layouts, operand_s_layouts, **_):
    return (
        all(space in {"ub", "vec"} for space in operand_memory_spaces)
        and all(layout == "row_major" for layout in operand_b_layouts)
        and all(layout == "none_box" for layout in operand_s_layouts)
    )


def _concat_shapes(src0_valid_shape, src1_valid_shape, dst_valid_shape, **_):
    return (
        len(src0_valid_shape) == 2
        and len(src1_valid_shape) == 2
        and len(dst_valid_shape) == 2
        and src0_valid_shape[0] == dst_valid_shape[0]
        and src1_valid_shape[0] == dst_valid_shape[0]
        and src0_valid_shape[1] + src1_valid_shape[1] == dst_valid_shape[1]
    )


@tilelib.tile_template(
    op="pto.tconcat",
    target="a5",
    name="template_tconcat",
    dtypes=same_dtype_signatures(3),
    iteration_axis="none",
    op_engine="vector",
    op_class="movement",
    constraints=[_concat_layout, _concat_shapes],
    id=0,
    loop_depth=2,
    is_post_update=False,
    tags=("movement", "concat"),
)
def template_tconcat(src0: pto.Tile, src1: pto.Tile, dst: pto.Tile):
    dtype = dst.dtype
    valid_rows = dst.valid_shape[0]
    cols0 = src0.valid_shape[1]
    cols1 = src1.valid_shape[1]
    lanes = pto.elements_per_vreg(dtype)

    for row in range(0, valid_rows, 1):
        remained0 = cols0
        for col in range(0, cols0, lanes):
            mask0, remained0 = pto.make_mask(dtype, remained0)
            value0 = pto.vlds(src0[row, col:])
            pto.vsts(value0, dst[row, col:], mask0)

        remained1 = cols1
        for col in range(0, cols1, lanes):
            mask1, remained1 = pto.make_mask(dtype, remained1)
            value1 = pto.vlds(src1[row, col:])
            pto.vsts(value1, dst[row, cols0 + col:], mask1)
