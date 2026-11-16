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
pto-isa repo: all six variants (b32/b16/b8 x RowWise/ColWise) with template
ids matching the upstream slots (0..5). Only the 2D row-major transpose path
(``TTransTile``) is covered; ConvTile format conversions are out of scope.

Upstream guards the b8/b16 variants with ``static_assert``s capping gather
and scatter indices at ``0xFFFF``. b16 tiles can never exceed that inside
the VEC buffer (253952 bytes on A5 cap a src+dst pair at ``2**16`` elements)
and the b32 forms take 32-bit offsets, so only b8 needs extra care here: a
b8 tile pair fits up to 126976 elements, twice the 16-bit offset range. The
b8 emit paths therefore index one ``lanes``-wide chunk at a time through a
base pointer rebased to the chunk start with ``pto.addptr``, so every
16-bit offset stays inside a single-chunk span for tiles of any size.

The ``tmp`` operand is required by the ``pto.ttrans`` op contract (matching
the C++ ``TTRANS(dst, src, tmp)`` intrinsic) but is unused in the 2D
row-major path; the C++ ``TTrans.hpp`` 2D path also does not use ``tmp``.
"""

from ptodsl import pto
from ptodsl._ast_rewrite import rewrite_jit_function
import ptodsl.tilelib as tilelib


B32_DTYPES = ("f32", "i32", "ui32")
B16_DTYPES = ("f16", "bf16", "i16", "ui16")
B8_DTYPES = ("i8", "ui8")

_BYTEWIDTH_BY_NAME = {"f32": 4, "i32": 4, "ui32": 4,
                      "f16": 2, "bf16": 2, "i16": 2, "ui16": 2,
                      "i8": 1, "ui8": 1}

# vgather2/vscatter take 16-bit element offsets for b8/b16 values (the b32 forms
# use 32-bit offsets), so every gather/scatter index must stay below 2**16.
_UI16_OFFSET_LIMIT = 1 << 16


def _ensure_ui16_chunk_span(lanes, stride):
    """Reject strides whose single ``lanes``-wide chunk cannot fit 16-bit offsets.

    The b8 emit paths index one ``lanes``-wide chunk at a time, so the widest
    offset a chunk produces is ``lanes * stride - 1``; once that product
    exceeds the offset field, splitting the tile into smaller chunks cannot
    help — the vector width itself is the addressing unit. A VEC-sized b8
    tile pair keeps ``stride < 2**16 / lanes``, so this only fires for
    out-of-budget tiles and mirrors upstream's ``static_assert``.
    """
    if lanes * stride > _UI16_OFFSET_LIMIT:
        raise ValueError(
            f"ttrans b8: one {lanes}-lane chunk spans {lanes * stride} element "
            f"offsets, past the {_UI16_OFFSET_LIMIT - 1} the 16-bit "
            f"vgather2/vscatter offset field holds; no chunking can bring "
            f"this back in range"
        )


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


@rewrite_jit_function
def _emit_ttrans_rowwise(
    src, dst, *, base_idx, lanes, mask_dtype, load_dist=None, chunk_rebase=False,
):
    """Emit the RowWise transpose loops (port of the TTrans*RowWise bodies).

    Per source row, load one ``lanes``-wide contiguous chunk and scatter it
    into the transposed destination: chunk element k goes to dst[col + k, row].

    With ``chunk_rebase`` (b8), each chunk scatters through a destination
    pointer rebased to the chunk start with ``pto.addptr``, so the vector
    offsets only encode ``[0..lanes-1] * dst_stride + row`` — a span that
    never depends on the tile size and stays inside the 16-bit offset field.
    """
    valid_rows, valid_cols = src.valid_shape
    dst_stride = dst.shape[1]
    valid_cols_minus_1 = valid_cols - 1
    dst_ptr = dst.as_ptr()

    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, lanes):
            mask, remained = pto.make_mask(mask_dtype, remained)
            data = pto.vlds(src[row, col:], dist=load_dist)
            if chunk_rebase:
                chunk_ptr = pto.addptr(dst_ptr, col * dst_stride)
                idx = pto.vmins(base_idx, valid_cols_minus_1 - col, mask)
            else:
                chunk_ptr = dst_ptr
                # transpose dst col index = col + [0..lanes-1], clamp, *dst_stride, +row
                idx = pto.vadds(base_idx, col, mask)
                idx = pto.vmins(idx, valid_cols_minus_1, mask)
            idx = pto.vmuls(idx, dst_stride, mask)
            idx = pto.vadds(idx, row, mask)
            pto.vscatter(data, chunk_ptr, idx, mask)


@rewrite_jit_function
def _emit_ttrans_colwise(
    src, dst, *, base_idx, lanes, mask_dtype,
    gather_result_type=None, store_dist=None, chunk_rebase=False,
):
    """Emit the ColWise transpose loops (port of the TTrans*ColWise bodies).

    Per source column, gather one ``lanes``-wide chunk of rows and store it
    into the transposed destination: chunk element k goes to dst[col, row + k].

    With ``chunk_rebase`` (b8), each chunk gathers through a source pointer
    rebased to the chunk start with ``pto.addptr``, so the vector offsets
    only encode ``[0..lanes-1] * src_stride + col`` — a span that never
    depends on the tile size and stays inside the 16-bit offset field.
    """
    valid_rows, valid_cols = src.valid_shape
    src_stride = src.shape[1]
    valid_rows_minus_1 = valid_rows - 1
    src_ptr = src.as_ptr()

    for col in range(0, valid_cols, 1):
        remained = valid_rows
        for row in range(0, valid_rows, lanes):
            mask, remained = pto.make_mask(mask_dtype, remained)
            if chunk_rebase:
                chunk_ptr = pto.addptr(src_ptr, row * src_stride)
                idx = pto.vmins(base_idx, valid_rows_minus_1 - row, mask)
            else:
                chunk_ptr = src_ptr
                idx = pto.vadds(base_idx, row, mask)
                idx = pto.vmins(idx, valid_rows_minus_1, mask)
            idx = pto.vmuls(idx, src_stride, mask)
            idx = pto.vadds(idx, col, mask)
            data = pto.vgather2(chunk_ptr, idx, mask, result_vreg_type=gather_result_type)
            pto.vsts(data, dst[col, row:], mask, dist=store_dist)


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
    # vci generates [0,1,...,lanes-1] once outside the loop (base index sequence).
    base_idx = pto.vci(pto.i32(0), "ASC")
    _emit_ttrans_rowwise(
        src, dst, base_idx=base_idx,
        lanes=pto.elements_per_vreg(dtype), mask_dtype=dtype,
    )


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
    base_idx = pto.vci(pto.i32(0), "ASC")
    _emit_ttrans_colwise(
        src, dst, base_idx=base_idx,
        lanes=pto.elements_per_vreg(dtype), mask_dtype=dtype,
    )


@tilelib.tile_template(
    op="pto.ttrans",
    target="a5",
    name="template_ttrans_b16_rowwise",
    dtypes=[(d, d, d) for d in B16_DTYPES],
    iteration_axis="none",
    op_engine="vector",
    op_class="movement",
    constraints=[
        _ub_row_major_2d,
        _wide_shape,
        _major_32byte_aligned,
    ],
    id=2,
    loop_depth=2,
    is_post_update=False,
    tags=("trans", "ub", "b16", "rowwise"),
)
def template_ttrans_b16_rowwise(src: pto.Tile, tmp: pto.Tile, dst: pto.Tile):
    """dst[j,i] = src[i,j] for wide (Rows<Cols) b16 tiles.

    Port of TTransB16RowWise (TTrans.hpp:157). Each row chunk loads
    contiguous source elements with ``vlds`` and scatters them to the
    transposed destination with 16-bit offsets (vscatter b16 form).
    """
    dtype = dst.dtype
    base_idx = pto.vci(pto.i16(0), "ASC")
    _emit_ttrans_rowwise(
        src, dst, base_idx=base_idx,
        lanes=pto.elements_per_vreg(dtype), mask_dtype=dtype,
    )


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

    Wide (Rows<Cols) b16 tiles are handled by ``template_ttrans_b16_rowwise``.
    """
    dtype = dst.dtype
    base_idx = pto.vci(pto.i16(0), "ASC")
    _emit_ttrans_colwise(
        src, dst, base_idx=base_idx,
        lanes=pto.elements_per_vreg(dtype), mask_dtype=dtype,
    )


@tilelib.tile_template(
    op="pto.ttrans",
    target="a5",
    name="template_ttrans_b8_rowwise",
    dtypes=[(d, d, d) for d in B8_DTYPES],
    iteration_axis="none",
    op_engine="vector",
    op_class="movement",
    constraints=[
        _ub_row_major_2d,
        _wide_shape,
        _major_32byte_aligned,
    ],
    id=4,
    loop_depth=2,
    is_post_update=False,
    tags=("trans", "ub", "b8", "rowwise"),
)
def template_ttrans_b8_rowwise(src: pto.Tile, tmp: pto.Tile, dst: pto.Tile):
    """dst[j,i] = src[i,j] for wide (Rows<Cols) b8 tiles.

    Port of TTransB8RowWise (TTrans.hpp:248). Each row chunk loads
    ``sregLower = elementsPerRepeat >> 1`` source bytes with an unpacking
    ``vlds`` (``dist = "UNPK_B8"`` places each byte in the even-byte lane of a
    256-lane b8 vector), then scatters them with the vscatter b8 form, which
    stores the even-numbered bytes of its value vector. Every chunk scatters
    through a destination pointer rebased to the chunk start, so the 16-bit
    offsets only span one ``sregLower``-wide chunk regardless of tile size.
    """
    dtype = dst.dtype  # i8/ui8
    # packed lanes: two b8 per b16 lane
    packed_lanes = pto.elements_per_vreg(dtype) >> 1   # 256/2 = 128
    _ensure_ui16_chunk_span(packed_lanes, dst.shape[1])
    base_idx = pto.vci(pto.i16(0), "ASC")
    _emit_ttrans_rowwise(
        src, dst, base_idx=base_idx,
        lanes=packed_lanes, mask_dtype=pto.i16, load_dist="UNPK_B8",
        chunk_rebase=True,
    )


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
    """dst[j,i] = src[i,j] for tall b8 tiles.

    Port of TTransB8ColWise (TTrans.hpp:200). Each column chunk gathers
    ``sregLower = elementsPerRepeat >> 1`` bytes: ``vgather2`` zero-extends
    every gathered i8/ui8 byte into one i16/ui16 lane of a 128-lane result
    (see docs/isa/micro-isa/03-vector-load-store.md), and the store packs the
    low byte of each lane back into contiguous bytes with ``dist = "PK_B16"``
    (the C++ path stores with ``DIST_PK_B16``). Mask granularity is b16.
    Every chunk gathers through a source pointer rebased to the chunk start,
    so the 16-bit offsets only span one ``sregLower``-wide chunk regardless
    of tile size.
    """
    dtype = dst.dtype  # i8/ui8
    # packed lanes: two b8 per b16 lane
    packed_lanes = pto.elements_per_vreg(dtype) >> 1   # 256/2 = 128
    _ensure_ui16_chunk_span(packed_lanes, src.shape[1])
    base_idx = pto.vci(pto.i16(0), "ASC")
    # b8 gather: i8 source -> i16 result, ui8 -> ui16 result
    _b8_index_elem = pto.ui16 if str(dtype) in ("ui8",) else pto.i16
    _emit_ttrans_colwise(
        src, dst, base_idx=base_idx,
        lanes=packed_lanes, mask_dtype=pto.i16,
        gather_result_type=pto.vreg_type(packed_lanes, _b8_index_elem),
        store_dist=pto.VStoreDist.PK_B16,
        chunk_rebase=True,
    )
