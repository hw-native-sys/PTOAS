# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Shared PTODSL implementation for straightforward A5 column reductions."""

from ptodsl import pto
import ptodsl.tilelib as tilelib


def _has_single_output_row(dst_valid_shape=(), **_):
    return len(dst_valid_shape) == 2 and dst_valid_shape[0] == 1


def _ub_or_vec_row_major(operand_memory_spaces, operand_b_layouts, operand_s_layouts, **_):
    return (
        all(space in {"ub", "vec"} for space in operand_memory_spaces)
        and all(layout == "row_major" for layout in operand_b_layouts)
        and all(layout == "none_box" for layout in operand_s_layouts)
    )


def register_column_reduction(*, op, name, vector_op, dtypes):
    @tilelib.tile_template(
        op=op,
        target="a5",
        name=name,
        dtypes=dtypes,
        iteration_axis="column",
        op_engine="vector",
        op_class="reduction",
        constraints=[
            _ub_or_vec_row_major,
            _has_single_output_row,
        ],
        id=0,
        loop_depth=2,
        is_post_update=False,
        tags=("reduction", "column"),
    )
    def template(src: pto.Tile, dst: pto.Tile):
        dtype = dst.dtype
        valid_rows, valid_cols = src.valid_shape
        lanes = pto.elements_per_vreg(dtype)
        remained = valid_cols

        for col in range(0, valid_cols, lanes):
            mask, remained = pto.make_mask(dtype, remained)
            accumulator = pto.vlds(src[0, col:])

            for row in range(1, valid_rows, 1):
                value = pto.vlds(src[row, col:])
                accumulator = vector_op(accumulator, value, mask)

            pto.vsts(accumulator, dst[0, col:], mask)

    return template


__all__ = ["register_column_reduction"]
