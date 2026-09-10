# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Shared PTODSL implementations for straightforward A5 elementwise TileOps.

The module-level traversal cores opt into PTODSL's source rewrite so they can
author runtime control flow with ordinary Python ``for range(...)`` syntax.
"""

from ptodsl import pto
from ptodsl._ast_rewrite import rewrite_jit_function
import ptodsl.tilelib as tilelib


FALLBACK_TRAVERSAL_PRIORITY = 0
PREFERRED_TRAVERSAL_PRIORITY = 10


def _ub_or_vec_row_major(operand_memory_spaces, operand_b_layouts, operand_s_layouts, **_):
    return (
        all(space in {"ub", "vec"} for space in operand_memory_spaces)
        and all(layout == "row_major" for layout in operand_b_layouts)
        and all(layout == "none_box" for layout in operand_s_layouts)
    )


def _common_constraints(*operand_names):
    return [
        _ub_or_vec_row_major,
        tilelib.require_same_valid_shape(*operand_names),
    ]


def _traversal_loop_depth(traversal):
    if traversal == "1d":
        return 1
    if traversal == "2d":
        return 2
    raise ValueError(
        f"unsupported element-wise traversal {traversal!r}; expected '1d' or '2d'"
    )


def traversal_metadata(
    traversal,
    *,
    priority=None,
    candidate_id=None,
    fallback_candidate_id=0,
    candidate_count=1,
):
    """Resolve stable candidate metadata for a 2D/1D traversal pair."""

    loop_depth = _traversal_loop_depth(traversal)
    if priority is None:
        priority = (
            PREFERRED_TRAVERSAL_PRIORITY
            if loop_depth == 1
            else FALLBACK_TRAVERSAL_PRIORITY
        )
    if candidate_id is None:
        candidate_id = fallback_candidate_id
        if loop_depth == 1:
            candidate_id += candidate_count
    return loop_depth, priority, candidate_id


def _with_traversal_constraint(traversal, operand_names, constraints):
    loop_depth = _traversal_loop_depth(traversal)
    result = list(constraints)
    if loop_depth == 1:
        result.append(tilelib.require_elementwise_1d(*operand_names))
    return result


@rewrite_jit_function
def emit_elementwise_1d(anchor, emit_chunk):
    """Traverse ``anchor`` as one contiguous logical element range.

    The caller is responsible for proving that every pointer used by
    ``emit_chunk`` describes the same gap-free range. ``emit_chunk`` receives
    the flattened element offset and the mask for that vector iteration.
    """

    dtype = anchor.dtype
    valid_rows, valid_cols = anchor.valid_shape
    lanes = pto.elements_per_vreg(dtype)
    total_elements = valid_rows * valid_cols
    remained = total_elements
    for offset in range(0, total_elements, lanes):
        mask, remained = pto.make_mask(dtype, remained)
        emit_chunk(offset, mask)


@rewrite_jit_function
def emit_elementwise_2d(anchor, emit_chunk):
    """Traverse ``anchor`` row by row with one vector loop per row.

    ``emit_chunk`` receives the row, column, and mask for the current vector
    iteration. The per-row mask state is deliberately restarted at
    ``valid_cols``.
    """

    dtype = anchor.dtype
    valid_rows, valid_cols = anchor.valid_shape
    lanes = pto.elements_per_vreg(dtype)
    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, lanes):
            mask, remained = pto.make_mask(dtype, remained)
            emit_chunk(row, col, mask)


def emit_unary_1d(src, dst, vector_op):
    """Emit a flattened unary element-wise body."""

    src_ptr = src.as_ptr()
    dst_ptr = dst.as_ptr()

    def emit_chunk(offset, mask):
        value = pto.vlds(src_ptr, offset)
        result = vector_op(value, mask)
        pto.vsts(result, dst_ptr, offset, mask)

    emit_elementwise_1d(dst, emit_chunk)


def emit_unary_2d(src, dst, vector_op):
    """Emit a row-wise unary element-wise body."""

    def emit_chunk(row, col, mask):
        value = pto.vlds(src[row, col:])
        result = vector_op(value, mask)
        pto.vsts(result, dst[row, col:], mask)

    emit_elementwise_2d(dst, emit_chunk)


def emit_binary_1d(src0, src1, dst, vector_op):
    """Emit a flattened Tile-Tile element-wise body."""

    src0_ptr = src0.as_ptr()
    src1_ptr = src1.as_ptr()
    dst_ptr = dst.as_ptr()

    def emit_chunk(offset, mask):
        lhs = pto.vlds(src0_ptr, offset)
        rhs = pto.vlds(src1_ptr, offset)
        result = vector_op(lhs, rhs, mask)
        pto.vsts(result, dst_ptr, offset, mask)

    emit_elementwise_1d(dst, emit_chunk)


def emit_binary_2d(src0, src1, dst, vector_op):
    """Emit a row-wise Tile-Tile element-wise body."""

    def emit_chunk(row, col, mask):
        lhs = pto.vlds(src0[row, col:])
        rhs = pto.vlds(src1[row, col:])
        result = vector_op(lhs, rhs, mask)
        pto.vsts(result, dst[row, col:], mask)

    emit_elementwise_2d(dst, emit_chunk)


def _emit_scalar_compute(value, scalar, mask, vector_op, broadcast_scalar,
                         scalar_lhs):
    if broadcast_scalar or scalar_lhs:
        scalar_value = pto.vbr(scalar)
        lhs, rhs = (
            (scalar_value, value) if scalar_lhs else (value, scalar_value)
        )
        return vector_op(lhs, rhs, mask)
    return vector_op(value, scalar, mask)


def emit_scalar_binary_1d(src, scalar, dst, vector_op, broadcast_scalar=False,
                          scalar_lhs=False):
    """Emit a flattened Tile-Scalar or Scalar-Tile element-wise body."""

    src_ptr = src.as_ptr()
    dst_ptr = dst.as_ptr()

    def emit_chunk(offset, mask):
        value = pto.vlds(src_ptr, offset)
        result = _emit_scalar_compute(
            value,
            scalar,
            mask,
            vector_op,
            broadcast_scalar,
            scalar_lhs,
        )
        pto.vsts(result, dst_ptr, offset, mask)

    emit_elementwise_1d(dst, emit_chunk)


def emit_scalar_binary_2d(src, scalar, dst, vector_op, broadcast_scalar=False,
                          scalar_lhs=False):
    """Emit a row-wise Tile-Scalar or Scalar-Tile element-wise body."""

    def emit_chunk(row, col, mask):
        value = pto.vlds(src[row, col:])
        result = _emit_scalar_compute(
            value,
            scalar,
            mask,
            vector_op,
            broadcast_scalar,
            scalar_lhs,
        )
        pto.vsts(result, dst[row, col:], mask)

    emit_elementwise_2d(dst, emit_chunk)


def emit_scalar_fill_1d(scalar, dst):
    """Emit a flattened scalar-fill body."""

    dst_ptr = dst.as_ptr()

    def emit_chunk(offset, mask):
        value = pto.vdup(scalar, mask)
        pto.vsts(value, dst_ptr, offset, mask)

    emit_elementwise_1d(dst, emit_chunk)


def emit_scalar_fill_2d(scalar, dst):
    """Emit a row-wise scalar-fill body."""

    def emit_chunk(row, col, mask):
        value = pto.vdup(scalar, mask)
        pto.vsts(value, dst[row, col:], mask)

    emit_elementwise_2d(dst, emit_chunk)


def _f32_select_masks(raw_mask, full_mask_b16):
    """Expand one raw f32 mask register into the two lane select masks."""
    select_mask = pto.pbitcast(raw_mask, pto.mask_b16)
    select_mask0, select_mask1 = pto.pintlv_b16(
        select_mask,
        full_mask_b16,
    )
    return (
        pto.pbitcast(select_mask0, pto.mask_b32),
        pto.pbitcast(select_mask1, pto.mask_b32),
    )


def _f32_paired_cols(valid_cols, lanes):
    repeat_times = (valid_cols + lanes - 1) // lanes
    return (repeat_times // 2) * lanes * 2


@rewrite_jit_function
def _emit_select_2d_f32_pair_tail(
    mask_ptr,
    mask_stride,
    dtype,
    valid_rows,
    valid_cols,
    lanes,
    dst,
    full_mask_b16,
    paired_cols,
    load_pair,
    load_tail,
):
    """Emit the per-row paired + tail masked-select loops for f32 payloads.

    ``load_pair(row, col)`` returns the four vsel operands for the paired
    lanes at columns ``col`` and ``col + lanes`` as ``(lhs0, rhs0, lhs1,
    rhs1)``; ``load_tail(row, col)`` returns ``(lhs, rhs)`` for one tail lane.
    """
    pair_width = lanes * 2
    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, paired_cols, pair_width):
            mask_offset = row * mask_stride + col // 8
            raw_mask = pto.plds(
                mask_ptr,
                mask_offset,
                dist=pto.PredicateDist.US,
            )
            select_mask0, select_mask1 = _f32_select_masks(
                raw_mask,
                full_mask_b16,
            )
            pred0, remained = pto.make_mask(dtype, remained)
            pred1, remained = pto.make_mask(dtype, remained)
            lhs0, rhs0, lhs1, rhs1 = load_pair(row, col)
            selected0 = pto.vsel(lhs0, rhs0, select_mask0)
            selected1 = pto.vsel(lhs1, rhs1, select_mask1)
            pto.vsts(selected0, dst[row, col:], pred0)
            pto.vsts(selected1, dst[row, col + lanes:], pred1)

        for col in range(paired_cols, valid_cols, lanes):
            mask_offset = row * mask_stride + col // 8
            raw_mask = pto.plds(
                mask_ptr,
                mask_offset,
                dist=pto.PredicateDist.US,
            )
            select_mask = pto.pbitcast(raw_mask, pto.mask_b16)
            select_mask = pto.punpack(
                select_mask,
                pto.PredicatePart.LOWER,
            )
            select_mask = pto.pbitcast(select_mask, pto.mask_b32)
            pred, remained = pto.make_mask(dtype, remained)
            lhs, rhs = load_tail(row, col)
            selected = pto.vsel(lhs, rhs, select_mask)
            pto.vsts(selected, dst[row, col:], pred)


def register_unary(*, op, name, vector_op, dtypes, constraints=(),
                   traversal="2d", priority=None, candidate_id=None):
    """Register a unary tile traversal using a public PTODSL vector operation."""

    loop_depth, priority, candidate_id = traversal_metadata(
        traversal,
        priority=priority,
        candidate_id=candidate_id,
    )
    candidate_constraints = _with_traversal_constraint(
        traversal,
        ("src", "dst"),
        _common_constraints("src", "dst") + list(constraints),
    )

    @tilelib.tile_template(
        op=op,
        target="a5",
        name=name,
        dtypes=dtypes,
        iteration_axis="none",
        op_engine="vector",
        op_class="elementwise",
        constraints=candidate_constraints,
        priority=priority,
        id=candidate_id,
        loop_depth=loop_depth,
        is_post_update=False,
        tags=("elementwise", "unary"),
    )
    def template(src: pto.Tile, dst: pto.Tile):
        if traversal == "1d":
            emit_unary_1d(src, dst, vector_op)
        else:
            emit_unary_2d(src, dst, vector_op)

    return template


def register_binary(*, op, name, vector_op, dtypes, has_tmp=False,
                    traversal="2d", priority=None, candidate_id=None):
    """Register a binary tile traversal, retaining an optional TileOp tmp operand."""

    loop_depth, priority, candidate_id = traversal_metadata(
        traversal,
        priority=priority,
        candidate_id=candidate_id,
    )
    if has_tmp:
        candidate_constraints = _with_traversal_constraint(
            traversal,
            ("src0", "src1", "tmp", "dst"),
            _common_constraints("src0", "src1", "tmp", "dst"),
        )

        @_elementwise_template_decorator(
            op=op, name=name, dtypes=dtypes, constraints=candidate_constraints,
            priority=priority, candidate_id=candidate_id, loop_depth=loop_depth,
            tags=("elementwise", "binary"),
        )
        def template(src0: pto.Tile, src1: pto.Tile, tmp: pto.Tile, dst: pto.Tile):
            _ = tmp
            if traversal == "1d":
                emit_binary_1d(src0, src1, dst, vector_op)
            else:
                emit_binary_2d(src0, src1, dst, vector_op)

        return template

    candidate_constraints = _with_traversal_constraint(
        traversal,
        ("src0", "src1", "dst"),
        _common_constraints("src0", "src1", "dst"),
    )

    @_elementwise_template_decorator(
        op=op, name=name, dtypes=dtypes, constraints=candidate_constraints,
        priority=priority, candidate_id=candidate_id, loop_depth=loop_depth,
        tags=("elementwise", "binary"),
    )
    def template(src0: pto.Tile, src1: pto.Tile, dst: pto.Tile):
        if traversal == "1d":
            emit_binary_1d(src0, src1, dst, vector_op)
        else:
            emit_binary_2d(src0, src1, dst, vector_op)

    return template


def _elementwise_template_decorator(*, op, name, dtypes, constraints, priority,
                                    candidate_id, loop_depth, tags):
    """Return the shared a5 elementwise ``tilelib.tile_template`` decorator."""
    return tilelib.tile_template(
        op=op, target="a5", name=name, dtypes=dtypes,
        iteration_axis="none", op_engine="vector", op_class="elementwise",
        constraints=constraints, priority=priority, id=candidate_id,
        loop_depth=loop_depth, is_post_update=False, tags=tags,
    )


def _scalar_binary_constraints(traversal, has_tmp, tmp_matches_src_dst):
    """Build the shared tile/scalar binary constraint list."""
    constraints = _common_constraints("src", "dst")
    if not has_tmp:
        return _with_traversal_constraint(traversal, ("src", "dst"), constraints)
    if tmp_matches_src_dst:
        constraints = _common_constraints("src", "tmp", "dst")
    else:
        constraints = _common_constraints("src", "dst") + [_ub_or_vec_row_major]
    return _with_traversal_constraint(traversal, ("src", "tmp", "dst"), constraints)


def _emit_scalar_binary_dispatch(traversal, src, scalar, dst, vector_op,
                                 broadcast_scalar, scalar_lhs):
    """Route one tile/scalar template body to the traversal-specific emitter."""
    if traversal == "1d":
        emit_scalar_binary_1d(
            src,
            scalar,
            dst,
            vector_op,
            broadcast_scalar,
            scalar_lhs,
        )
    else:
        emit_scalar_binary_2d(
            src,
            scalar,
            dst,
            vector_op,
            broadcast_scalar,
            scalar_lhs,
        )


def _scalar_binary_template_variant(*, op, name, dtypes, constraints, priority,
                                    candidate_id, loop_depth, traversal,
                                    vector_op, broadcast_scalar, scalar_lhs,
                                    has_tmp, scalar_first):
    """Build one decorated tile/scalar template closure variant."""
    decorator = _elementwise_template_decorator(
        op=op, name=name, dtypes=dtypes, constraints=constraints,
        priority=priority, candidate_id=candidate_id, loop_depth=loop_depth,
        tags=("elementwise", "scalar"),
    )

    if scalar_first:
        if has_tmp:
            @decorator
            def variant(scalar, src: pto.Tile, tmp: pto.Tile, dst: pto.Tile):
                _ = tmp
                _emit_scalar_binary_dispatch(
                    traversal, src, scalar, dst, vector_op, broadcast_scalar,
                    scalar_lhs,
                )
        else:
            @decorator
            def variant(scalar, src: pto.Tile, dst: pto.Tile):
                _emit_scalar_binary_dispatch(
                    traversal, src, scalar, dst, vector_op, broadcast_scalar,
                    scalar_lhs,
                )
    elif has_tmp:
        @decorator
        def variant(src: pto.Tile, scalar, tmp: pto.Tile, dst: pto.Tile):
            _ = tmp
            _emit_scalar_binary_dispatch(
                traversal, src, scalar, dst, vector_op, broadcast_scalar,
                scalar_lhs,
            )
    else:
        @decorator
        def variant(src: pto.Tile, scalar, dst: pto.Tile):
            _emit_scalar_binary_dispatch(
                traversal, src, scalar, dst, vector_op, broadcast_scalar,
                scalar_lhs,
            )

    return variant


def _register_scalar_binary_variants(*, op, name, vector_op, dtypes,
                                     broadcast_scalar, has_tmp,
                                     tmp_matches_src_dst, reverse_name,
                                     traversal, priority, candidate_id,
                                     loop_depth, reverse_candidate_id):
    """Register the forward and optional reverse tile/scalar template variants."""

    constraints = _scalar_binary_constraints(traversal, has_tmp, tmp_matches_src_dst)
    template = _scalar_binary_template_variant(
        op=op, name=name, dtypes=dtypes, constraints=constraints,
        priority=priority, candidate_id=candidate_id, loop_depth=loop_depth,
        traversal=traversal, vector_op=vector_op,
        broadcast_scalar=broadcast_scalar, scalar_lhs=False, has_tmp=has_tmp,
        scalar_first=False,
    )
    if not reverse_name:
        return template

    reverse_template = _scalar_binary_template_variant(
        op=op, name=reverse_name, dtypes=dtypes, constraints=constraints,
        priority=priority, candidate_id=reverse_candidate_id,
        loop_depth=loop_depth, traversal=traversal, vector_op=vector_op,
        broadcast_scalar=True, scalar_lhs=True, has_tmp=has_tmp,
        scalar_first=True,
    )
    return template, reverse_template


def register_scalar_binary(*, op, name, vector_op, dtypes, broadcast_scalar=False,
                           has_tmp=False, tmp_matches_src_dst=True,
                           reverse_name=None, traversal="2d", priority=None,
                           candidate_id=None, reverse_candidate_id=None):
    """Register a tile/scalar traversal using either a vector-scalar or broadcast op."""

    loop_depth, priority, candidate_id = traversal_metadata(
        traversal,
        priority=priority,
        candidate_id=candidate_id,
        candidate_count=2 if reverse_name else 1,
    )
    if reverse_candidate_id is None:
        reverse_candidate_id = candidate_id + 1
    return _register_scalar_binary_variants(
        op=op,
        name=name,
        vector_op=vector_op,
        dtypes=dtypes,
        broadcast_scalar=broadcast_scalar,
        has_tmp=has_tmp,
        tmp_matches_src_dst=tmp_matches_src_dst,
        reverse_name=reverse_name,
        traversal=traversal,
        priority=priority,
        candidate_id=candidate_id,
        loop_depth=loop_depth,
        reverse_candidate_id=reverse_candidate_id,
    )


def register_scalar_fill(*, op, name, dtypes, traversal="2d", priority=None,
                         candidate_id=None):
    """Register a scalar-to-tile fill traversal."""

    loop_depth, priority, candidate_id = traversal_metadata(
        traversal,
        priority=priority,
        candidate_id=candidate_id,
    )
    constraints = [
        tilelib.check_memory_space("ub"),
        tilelib.check_layout("row_major"),
        tilelib.check_s_layout("none_box"),
    ]
    constraints = _with_traversal_constraint(
        traversal,
        ("dst",),
        constraints,
    )
    @tilelib.tile_template(
        op=op,
        target="a5",
        name=name,
        dtypes=dtypes,
        iteration_axis="none",
        op_engine="vector",
        op_class="elementwise",
        constraints=constraints,
        priority=priority,
        id=candidate_id,
        loop_depth=loop_depth,
        is_post_update=False,
        tags=("elementwise", "scalar", "fill"),
    )
    def template(scalar, dst: pto.Tile):
        if traversal == "1d":
            emit_scalar_fill_1d(scalar, dst)
        else:
            emit_scalar_fill_2d(scalar, dst)

    return template


__all__ = [
    "FALLBACK_TRAVERSAL_PRIORITY",
    "PREFERRED_TRAVERSAL_PRIORITY",
    "emit_binary_1d",
    "emit_binary_2d",
    "emit_elementwise_1d",
    "emit_elementwise_2d",
    "emit_scalar_binary_1d",
    "emit_scalar_binary_2d",
    "emit_scalar_fill_1d",
    "emit_scalar_fill_2d",
    "emit_unary_1d",
    "emit_unary_2d",
    "register_binary",
    "register_scalar_binary",
    "register_scalar_fill",
    "register_unary",
    "traversal_metadata",
]
