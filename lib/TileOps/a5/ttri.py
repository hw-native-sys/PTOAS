# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""PTODSL TileLib template for pto.ttri."""

from ptodsl import pto
from ptodsl import pto
import ptodsl.tilelib as tilelib
from ._common import NUMERIC_DTYPES
from ._elementwise import _ub_or_vec_row_major


TRI_DTYPES = [
    ("i32", dtype) for dtype in NUMERIC_DTYPES
]

def _scalar_tile(operand_kinds=(), **_):
    return operand_kinds == ("scalar", "tile")


def _is_lower(upper_or_lower=0, **_):
    if isinstance(upper_or_lower, str):
        return upper_or_lower in ("0", "lower")
    return int(upper_or_lower) == 0


def _is_upper(upper_or_lower=0, **_):
    if isinstance(upper_or_lower, str):
        return upper_or_lower in ("1", "upper")
    return int(upper_or_lower) == 1


def _as_pto_type(dtype):
    return getattr(pto, str(dtype))


@tilelib.tile_template(
    op="pto.ttri",
    target="a5",
    name="template_ttri_lower",
    dtypes=TRI_DTYPES,
    iteration_axis="none",
    op_engine="vector",
    op_class="other",
    constraints=[_is_lower, _ub_or_vec_row_major, _scalar_tile],
    loop_depth=2,
    is_post_update=False,
    id=0,
)
def template_ttri_lower(diagonal, dst: pto.Tile):
    """Lower triangular mask: mask[i,j]=1 if j<=i+diagonal, else 0."""
    dtype = dst.dtype
    valid_rows, valid_cols = dst.valid_shape
    lanes = pto.elements_per_vreg(dtype)
    v_zeros = pto.vbr(_as_pto_type(dtype)(0))
    v_ones = pto.vbr(_as_pto_type(dtype)(1))

    if diagonal >= 0:
        start_row = 0
    else:
        start_row = 0 - diagonal
    start_num = diagonal + 1

    # Pass 1: fill all with zeros
    for row in range(valid_rows):
        remained = valid_cols
        for col in range(0, valid_cols, lanes):
            mask, remained = pto.make_mask(dtype, remained)
            pto.vsts(v_zeros, dst[row, col:], mask)

    # Pass 2: overwrite first (row+diagonal+1) cols with ones
    for row in range(start_row, valid_rows):
        num_ones = row + start_num
        for col in range(0, valid_cols, lanes):
            mask, num_ones = pto.make_mask(dtype, num_ones)
            pto.vsts(v_ones, dst[row, col:], mask)


@tilelib.tile_template(
    op="pto.ttri",
    target="a5",
    name="template_ttri_upper",
    dtypes=TRI_DTYPES,
    iteration_axis="none",
    op_engine="vector",
    op_class="other",
    constraints=[_is_upper, _ub_or_vec_row_major, _scalar_tile],
    loop_depth=2,
    is_post_update=False,
    id=1,
)
def template_ttri_upper(diagonal, dst: pto.Tile):
    """Upper triangular mask: mask[i,j]=1 if j>=i+diagonal, else 0."""
    dtype = dst.dtype
    valid_rows, valid_cols = dst.valid_shape
    lanes = pto.elements_per_vreg(dtype)
    v_zeros = pto.vbr(_as_pto_type(dtype)(0))
    v_ones = pto.vbr(_as_pto_type(dtype)(1))

    if diagonal > 0:
        start_row = 0
    else:
        start_row = 1 - diagonal
    
    start_num = diagonal
    for row in range(valid_rows):
        remained = valid_cols
        for col in range(0, valid_cols, lanes):
            mask, remained = pto.make_mask(dtype, remained)
            pto.vsts(v_ones, dst[row, col:], mask)

    for row in range(start_row, valid_rows):
        num_zeros = row + start_num
        for col in range(0, valid_cols, lanes):
            mask, num_zeros = pto.make_mask(dtype, num_zeros)
            pto.vsts(v_zeros, dst[row, col:], mask)
