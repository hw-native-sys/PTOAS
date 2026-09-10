# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""PTODSL TileLib templates for ``pto.ttrans`` (A5 2D tile transpose).

Ports the 2D-vectorized logic from ``include/pto/npu/a5/TTrans.hpp`` of the
pto-isa repo. Only the 2D row-major transpose path (``TTransTile``) is covered;
ConvTile format conversions are out of scope.

The ``tmp`` operand is required by the ``pto.ttrans`` op contract (matching
the C++ ``TTRANS(dst, src, tmp)`` intrinsic) but is unused in the 2D
row-major path; the C++ ``TTrans.hpp`` 2D path also does not use ``tmp``.
"""

from ptodsl import pto
import ptodsl.tilelib as tilelib


B32_DTYPES = ("f32", "i32", "ui32")
B16_DTYPES = ("f16", "bf16", "i16", "ui16")
B8_DTYPES = ("i8", "ui8")

_BYTEWIDTH_BY_NAME = {"f32": 4, "i32": 4, "ui32": 4,
                      "f16": 2, "bf16": 2, "i16": 2, "ui16": 2,
                      "i8": 1, "ui8": 1}


def _ub_row_major_2d(operand_memory_spaces, operand_b_layouts, operand_s_layouts, **_):
    return (
        all(space in {"ub", "vec"} for space in operand_memory_spaces)
        and all(layout == "row_major" for layout in operand_b_layouts)
        and all(layout == "none_box" for layout in operand_s_layouts)
    )


def _wide_shape(src_valid_shape, **_):
    rows, cols = src_valid_shape
    return rows < cols


def _tall_shape(src_valid_shape, **_):
    rows, cols = src_valid_shape
    return rows >= cols


def _major_32byte_aligned(src_valid_shape, src_dtype, dst_dtype, **_):
    dtype_name = getattr(src_dtype, "name", str(src_dtype))
    bytewidth = _BYTEWIDTH_BY_NAME.get(dtype_name)
    if bytewidth is None:
        return False
    rows, cols = src_valid_shape
    return (cols * bytewidth) % 32 == 0


@tilelib.tile_template(
    op="pto.ttrans",
    target="a5",
    name="template_ttrans_b32_rowwise",
    dtypes=[(d, d, d) for d in B32_DTYPES],
    iteration_axis="none",
    op_engine="vector",
    op_class="movement",
    constraints=[
        _ub_row_major_2d,
        _wide_shape,
        _major_32byte_aligned,
    ],
    id=0,
    loop_depth=2,
    is_post_update=False,
    tags=("trans", "ub", "b32", "rowwise"),
)
def template_ttrans_b32_rowwise(src: pto.Tile, tmp: pto.Tile, dst: pto.Tile):
    dtype = dst.dtype
    valid_rows, valid_cols = src.valid_shape
    lanes = pto.elements_per_vreg(dtype)
    dst_stride = dst.shape[1]
    valid_cols_minus_1 = valid_cols - 1
    dst_ptr = dst.as_ptr()
    # vci generates [0,1,...,lanes-1] once outside the loop (base index sequence).
    base_idx = pto.vci(pto.i32(0), "ASC")

    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, lanes):
            mask, remained = pto.make_mask(dtype, remained)
            data = pto.vlds(src[row, col:])
            # transpose dst col index = col + [0..lanes-1], clamp, *dst_stride, +row
            idx = pto.vadds(base_idx, col, mask)
            idx = pto.vmins(idx, valid_cols_minus_1, mask)
            idx = pto.vmuls(idx, dst_stride, mask)
            idx = pto.vadds(idx, row, mask)
            pto.vscatter(data, dst_ptr, idx, mask)


@tilelib.tile_template(
    op="pto.ttrans",
    target="a5",
    name="template_ttrans_b32_colwise",
    dtypes=[(d, d, d) for d in B32_DTYPES],
    iteration_axis="none",
    op_engine="vector",
    op_class="movement",
    constraints=[
        _ub_row_major_2d,
        _tall_shape,
        _major_32byte_aligned,
    ],
    id=1,
    loop_depth=2,
    is_post_update=False,
    tags=("trans", "ub", "b32", "colwise"),
)
def template_ttrans_b32_colwise(src: pto.Tile, tmp: pto.Tile, dst: pto.Tile):
    """dst[j,i] = src[i,j] for tall (Rows>=Cols) b32 tiles. Port of TTransB32ColWise."""
    dtype = dst.dtype
    valid_rows, valid_cols = src.valid_shape
    lanes = pto.elements_per_vreg(dtype)
    src_stride = src.shape[1]
    valid_rows_minus_1 = valid_rows - 1
    src_ptr = src.as_ptr()
    base_idx = pto.vci(pto.i32(0), "ASC")

    for col in range(0, valid_cols, 1):
        remained = valid_rows
        for row in range(0, valid_rows, lanes):
            mask, remained = pto.make_mask(dtype, remained)
            idx = pto.vadds(base_idx, row, mask)
            idx = pto.vmins(idx, valid_rows_minus_1, mask)
            idx = pto.vmuls(idx, src_stride, mask)
            idx = pto.vadds(idx, col, mask)
            data = pto.vgather2(src_ptr, idx, mask)
            pto.vsts(data, dst[col, row:], mask)


@tilelib.tile_template(
    op="pto.ttrans",
    target="a5",
    name="template_ttrans_b16_colwise",
    dtypes=[(d, d, d) for d in B16_DTYPES],
    iteration_axis="none",
    op_engine="vector",
    op_class="movement",
    constraints=[
        _ub_row_major_2d,
        _tall_shape,
        _major_32byte_aligned,
    ],
    id=3,
    loop_depth=2,
    is_post_update=False,
    tags=("trans", "ub", "b16", "colwise"),
)
def template_ttrans_b16_colwise(src: pto.Tile, tmp: pto.Tile, dst: pto.Tile):
    """dst[j,i] = src[i,j] for tall b16 tiles. Port of TTransB16ColWise (TTrans.hpp:111).

    Only ColWise is implemented by this template. VPTO vscatter supports the
    16-bit offsets needed by a future RowWise b16 template, but that template
    has not yet been added.
    """
    dtype = dst.dtype
    valid_rows, valid_cols = src.valid_shape
    lanes = pto.elements_per_vreg(dtype)
    src_stride = src.shape[1]
    valid_rows_minus_1 = valid_rows - 1
    src_ptr = src.as_ptr()
    base_idx = pto.vci(pto.i16(0), "ASC")

    for col in range(0, valid_cols, 1):
        remained = valid_rows
        for row in range(0, valid_rows, lanes):
            mask, remained = pto.make_mask(dtype, remained)
            idx = pto.vadds(base_idx, row, mask)
            idx = pto.vmins(idx, valid_rows_minus_1, mask)
            idx = pto.vmuls(idx, src_stride, mask)
            idx = pto.vadds(idx, col, mask)
            data = pto.vgather2(src_ptr, idx, mask)
            pto.vsts(data, dst[col, row:], mask)


@tilelib.tile_template(
    op="pto.ttrans",
    target="a5",
    name="template_ttrans_b8_colwise",
    dtypes=[(d, d, d) for d in B8_DTYPES],
    iteration_axis="none",
    op_engine="vector",
    op_class="movement",
    constraints=[
        _ub_row_major_2d,
        _tall_shape,
        _major_32byte_aligned,
    ],
    id=5,
    loop_depth=2,
    is_post_update=False,
    tags=("trans", "ub", "b8", "colwise"),
)
def template_ttrans_b8_colwise(src: pto.Tile, tmp: pto.Tile, dst: pto.Tile):
    """dst[j,i] = src[i,j] for tall b8 tiles (paired-packed as b16).

    Port of TTransB8ColWise (TTrans.hpp:200). Two b8 elements pack into one
    b16 lane (C++ sregLower = elementsPerRepeat >> 1). vgather2 produces a
    128xi16 result (paired-pack) from the i8 source; mask is b16.

    Only ColWise is implemented by this template. VPTO vscatter supports the
    16-bit offsets and even-byte source convention needed by a future RowWise
    b8 template, but that template has not yet been added. This template is not
    yet covered by the ST harness (b8 transpose hits a deeper emitc/intrinsic
    issue under investigation).
    """
    dtype = dst.dtype  # i8/ui8
    valid_rows, valid_cols = src.valid_shape
    # packed lanes: two b8 per b16 lane
    packed_lanes = pto.elements_per_vreg(dtype) >> 1   # 256/2 = 128
    src_stride = src.shape[1]
    valid_rows_minus_1 = valid_rows - 1
    src_ptr = src.as_ptr()
    base_idx = pto.vci(pto.i16(0), "ASC")
    # b8 gather uses paired-pack: i8 source -> i16 result, ui8 -> ui16 result
    _b8_index_elem = pto.ui16 if str(dtype) in ("ui8",) else pto.i16
    result_ty = pto.vreg_type(packed_lanes, _b8_index_elem)

    for col in range(0, valid_cols, 1):
        remained = valid_rows
        for row in range(0, valid_rows, packed_lanes):
            mask, remained = pto.make_mask(pto.i16, remained)
            idx = pto.vadds(base_idx, row, mask)
            idx = pto.vmins(idx, valid_rows_minus_1, mask)
            idx = pto.vmuls(idx, src_stride, mask)
            idx = pto.vadds(idx, col, mask)
            data = pto.vgather2(src_ptr, idx, mask, result_vreg_type=result_ty)
            pto.vsts(data, dst[col, row:], mask)
