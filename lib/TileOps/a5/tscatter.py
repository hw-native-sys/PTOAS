# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""PTODSL TileLib templates for ``pto.tscatter``."""

from ptodsl import pto
import ptodsl.tilelib as tilelib

from ._common import NUMERIC_DTYPES
from ._row_arg import _scalar_literal


def _init_ub_buffer(dst: pto.Tile):
    dtype = dst.dtype
    tile_rows, tile_cols = dst.shape
    lanes = pto.elements_per_vreg(dtype)
    zeros = pto.vbr(_scalar_literal(dtype, 0))
    mask, _ = pto.make_mask(dtype, lanes)
    for row in range(0, tile_rows, 1):
        for col in range(0, tile_cols, lanes):
            pto.vsts(zeros, dst[row, col:], mask)
    pto.mem_bar(pto.BarrierType.VST_VST)


_SCATTER_MASK_DTYPES = [(dtype, dtype) for dtype in NUMERIC_DTYPES]

_SCATTER_INDEX_DTYPES = [
    (dtype, dtype, idx)
    for dtype in NUMERIC_DTYPES
    for idx in (
        ("i16", "ui16")
        if dtype in ("i8", "ui8", "i16", "ui16", "f16", "bf16")
        else ("i32", "ui32")
    )
]


def _no_mask_pattern(mask_pattern=True):
    return mask_pattern is True


def _axis_is_row(axis_value, **_):
    return axis_value == "row"


def _axis_is_col(axis_value, **_):
    return axis_value == "col"


@tilelib.tile_template(
    op="pto.tscatter",
    target="a5",
    name="template_tscatter_index",
    dtypes=_SCATTER_INDEX_DTYPES,
    iteration_axis="none",
    op_engine="vector",
    op_class="other",
    layouts=("row_major",),
    loop_depth=2,
    is_post_update=False,
    constraints=(_no_mask_pattern,),
    id=0,
)
def template_tscatter_index(src: pto.Tile, dst: pto.Tile, idx: pto.Tile):
    _init_ub_buffer(dst)
    valid_rows, valid_cols = dst.valid_shape
    dtype = dst.dtype
    idx_dtype = idx.dtype
    dst_ptr = dst.as_ptr()
    src_dist = "UNPK_B8" if pto.bytewidth(dtype) == 1 else None
    lanes = pto.elements_per_vreg(idx_dtype)
    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, lanes):
            idx_mask, remained = pto.make_mask(idx_dtype, remained)
            src_reg = pto.vlds(src[row, col:], dist=src_dist)
            idx_reg = pto.vlds(idx[row, col:])
            pto.vscatter(src_reg, dst_ptr, idx_reg, idx_mask)


_MASK_PATTERN_TO_INTERLEAVE = {
    "P1111": (),
    "P0101": (True,),
    "P1010": (False,),
    "P0001": (True, True),
    "P0010": (True, False),
    "P0100": (False, True),
    "P1000": (False, False),
}

_MASK_PATTERN_TO_STRIDE = {
    "P1111": (0, 1),
    "P0101": (0, 2),
    "P1010": (1, 2),
    "P0001": (0, 4),
    "P0010": (1, 4),
    "P0100": (2, 4),
    "P1000": (3, 4),
}


@tilelib.tile_template(
    op="pto.tscatter",
    target="a5",
    name="template_tscatter_mask_row",
    dtypes=_SCATTER_MASK_DTYPES,
    iteration_axis="none",
    op_engine="vector",
    op_class="other",
    layouts=("row_major",),
    loop_depth=2,
    is_post_update=False,
    tags=("scatter", "mask"),
    constraints=(_axis_is_row,),
    id=1,
)
def template_tscatter_mask_row(src: pto.Tile, dst: pto.Tile):
    mask_pattern = pto.get_op_attr("mask_pattern", "P1111")
    interleave_args = _MASK_PATTERN_TO_INTERLEAVE[mask_pattern]
    dtype = dst.dtype
    valid_rows, valid_cols = src.valid_shape
    lanes = pto.elements_per_vreg(dtype)
    times = 1 << len(interleave_args)
    dst_valid_col = valid_cols * times
    zeros = pto.vbr(_scalar_literal(dtype, 0))
    for row in range(0, valid_rows, 1):
        py_rem = dst_valid_col
        for col in range(0, valid_cols, lanes):
            src_reg = pto.vlds(src[row, col:])
            if not interleave_args:
                mask, _ = pto.make_mask(dtype, py_rem)
                pto.vsts(src_reg, dst[row, col:], mask)
                py_rem -= lanes
            elif len(interleave_args) == 1:
                if interleave_args[0]:
                    reg0, reg1 = pto.vintlv(src_reg, zeros)
                else:
                    reg0, reg1 = pto.vintlv(zeros, src_reg)
                mask, _ = pto.make_mask(dtype, py_rem)
                pto.vsts(reg0, dst[row, col * times:], mask)
                py_rem -= lanes
                if py_rem > 0:
                    mask, _ = pto.make_mask(dtype, py_rem)
                    pto.vsts(reg1, dst[row, col * times + lanes:], mask)
                    py_rem -= lanes
            else:
                if interleave_args[0]:
                    tmp0, tmp1 = pto.vintlv(src_reg, zeros)
                else:
                    tmp0, tmp1 = pto.vintlv(zeros, src_reg)
                if interleave_args[1]:
                    reg0, reg1 = pto.vintlv(tmp0, zeros)
                    reg2, reg3 = pto.vintlv(tmp1, zeros)
                else:
                    reg0, reg1 = pto.vintlv(zeros, tmp0)
                    reg2, reg3 = pto.vintlv(zeros, tmp1)
                mask, _ = pto.make_mask(dtype, py_rem)
                pto.vsts(reg0, dst[row, col * times:], mask)
                py_rem -= lanes
                if py_rem > 0:
                    mask, _ = pto.make_mask(dtype, py_rem)
                    pto.vsts(reg1, dst[row, col * times + lanes:], mask)
                    py_rem -= lanes
                if py_rem > 0:
                    mask, _ = pto.make_mask(dtype, py_rem)
                    pto.vsts(reg2, dst[row, col * times + lanes * 2:], mask)
                    py_rem -= lanes
                if py_rem > 0:
                    mask, _ = pto.make_mask(dtype, py_rem)
                    pto.vsts(reg3, dst[row, col * times + lanes * 3:], mask)
                    py_rem -= lanes


@tilelib.tile_template(
    op="pto.tscatter",
    target="a5",
    name="template_tscatter_mask_col",
    dtypes=_SCATTER_MASK_DTYPES,
    iteration_axis="none",
    op_engine="vector",
    op_class="other",
    layouts=("row_major",),
    loop_depth=2,
    is_post_update=False,
    tags=("scatter", "mask"),
    constraints=(_axis_is_col,),
    id=2,
)
def template_tscatter_mask_col(src: pto.Tile, dst: pto.Tile):
    mask_pattern = pto.get_op_attr("mask_pattern", "P1111")
    if mask_pattern != "P1111":
        _init_ub_buffer(dst)
    start, stride = _MASK_PATTERN_TO_STRIDE[mask_pattern]
    dtype = dst.dtype
    valid_rows, valid_cols = src.valid_shape
    lanes = pto.elements_per_vreg(dtype)
    for row in range(0, valid_rows, 1):
        remained = valid_cols
        row_dst = row * stride + start
        for col in range(0, valid_cols, lanes):
            src_reg = pto.vlds(src[row, col:])
            mask, remained = pto.make_mask(dtype, remained)
            pto.vsts(src_reg, dst[row_dst, col:], mask)
