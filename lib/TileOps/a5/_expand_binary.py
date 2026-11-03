# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Shared PTODSL templates for row/column expand binary TileOps."""

from collections import namedtuple

from SoftOps.div_uint import div_uint_soft
from TileOps.a5.elementwise import div_dispatch

from ptodsl import pto
import ptodsl.tilelib as tilelib


NUMERIC_SIGNATURES = [
    ("i8", "i8", "i8"),
    ("i16", "i16", "i16"),
    ("i32", "i32", "i32"),
    ("ui8", "ui8", "ui8"),
    ("ui16", "ui16", "ui16"),
    ("ui32", "ui32", "ui32"),
    ("f16", "f16", "f16"),
    ("bf16", "bf16", "bf16"),
    ("f32", "f32", "f32"),
]

FLOAT_SIGNATURES = [
    ("f16", "f16", "f16"),
    ("bf16", "bf16", "bf16"),
    ("f32", "f32", "f32"),
]

UNSIGNED_SIGNATURES = [(dtype, dtype, dtype) for dtype in ("ui8", "ui16", "ui32")]


def _ub_or_vec_row_major(operand_memory_spaces, operand_b_layouts, operand_s_layouts, **_):
    return (
        all(space in {"ub", "vec"} for space in operand_memory_spaces)
        and all(layout == "row_major" for layout in operand_b_layouts)
        and all(layout == "none_box" for layout in operand_s_layouts)
    )


# One operand's layout and shapes, as the expand constraints see them.
_ExpandOperand = namedtuple("_ExpandOperand", "config shape valid_shape")


def _expand_operands(src0_config, src1_config, dst_config, context):
    """Per-operand layout/shape facts for the expand constraints, in operand order."""

    def _of(name, config):
        return _ExpandOperand(
            config,
            context.get(f"{name}_shape", ()),
            context.get(f"{name}_valid_shape", ()),
        )

    return _of("src0", src0_config), _of("src1", src1_config), _of("dst", dst_config)


def _row_expand_layout(src0_config, src1_config, dst_config, operand_memory_spaces=(), **context):
    """Row-expand layout constraint: either operand may be the expanded one."""
    if not all(space in {"ub", "vec"} for space in operand_memory_spaces):
        return False
    return _row_expand_layout_ok(
        _expand_operands(src0_config, src1_config, dst_config, context)
    )


def _row_expand_layout_ok(operands) -> bool:
    """The full-sized operand is a plain RowMajor tile and the other one is expandable."""
    src0, src1, dst = operands
    if not all(operand.config for operand in operands):
        return False
    if dst.config.b_layout != "row_major" or dst.config.s_layout != "none_box":
        return False
    index = _full_operand_index(src0.valid_shape, src1.valid_shape, dst.valid_shape)
    if index is None:
        return False
    full, expanded = (src0, src1) if index == 0 else (src1, src0)
    if full.config.b_layout != "row_major" or full.config.s_layout != "none_box":
        return False
    return _row_expand_operand_ok(expanded)


def _is_unknown_dim(value):
    return value is None or value in {-1, -(2**63)}


def _known_eq(lhs, rhs) -> bool:
    return _is_unknown_dim(lhs) or _is_unknown_dim(rhs) or lhs == rhs


def _known_ge(lhs, rhs) -> bool:
    return _is_unknown_dim(lhs) or _is_unknown_dim(rhs) or lhs >= rhs


def _shape_pair_eq(lhs, rhs) -> bool:
    """True when two rank-2 shapes agree on both dims (unknown dims match)."""
    try:
        return _known_eq(lhs[0], rhs[0]) and _known_eq(lhs[1], rhs[1])
    except (IndexError, KeyError, TypeError):
        return False


def _static_valid_pair(tile):
    """Rank-2 valid shape carried by the tile's ``TileSpec`` (a trace-time value)."""
    valid = getattr(tile, "valid_shape_static", None)
    if valid is None:
        raise RuntimeError(
            "row-expand template cannot read the tile's static valid shape from "
            f"{type(tile).__name__}; refusing to guess which operand is row-expanded"
        )
    return tuple(valid)


def _tile_is_full_sized(tile, dst) -> bool:
    """True when ``tile`` matches dst, i.e. the *other* operand is row-expanded."""
    return _shape_pair_eq(_static_valid_pair(tile), _static_valid_pair(dst))


def _full_operand_index(src0_valid_shape, src1_valid_shape, dst_valid_shape):
    """Index (0 or 1) of the full-sized operand, or None when neither matches dst.

    ``src0`` wins when both match, matching pto-isa's ``src0eqdst`` precedence.  Both
    shapes must be rank-2 so callers can index them without guarding again.
    """
    if not all(
        len(shape) == 2
        for shape in (src0_valid_shape, src1_valid_shape, dst_valid_shape)
    ):
        return None
    if _shape_pair_eq(src0_valid_shape, dst_valid_shape):
        return 0
    if _shape_pair_eq(src1_valid_shape, dst_valid_shape):
        return 1
    return None


def _expanded_valid_shape(src0_valid_shape, src1_valid_shape, dst_valid_shape):
    """Valid shape of the expanded operand, or None when neither operand matches dst."""
    index = _full_operand_index(src0_valid_shape, src1_valid_shape, dst_valid_shape)
    if index is None:
        return None
    return src1_valid_shape if index == 0 else src0_valid_shape


def _row_expand_operand_ok(operand) -> bool:
    """The expanded operand is either a RowMajor block row or a ColMajor single column."""
    config, shape, valid_shape = operand.config, operand.shape, operand.valid_shape
    if config.b_layout == "row_major":
        return config.s_layout == "none_box"
    one_column = (len(shape) == 2 and shape[1] == 1) or (
        len(valid_shape) == 2 and valid_shape[1] == 1
    )
    return (
        config.b_layout == "col_major"
        and one_column
        and config.s_layout in {"none_box", "row_major"}
    )


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


def _row_operand_repeats_block(tile) -> bool:
    """True when the row operand is a RowMajor 32-byte block per row (ISA Mode 2).

    ``TROWEXPAND*`` picks its broadcast form from the layout of the expanded operand
    (see ``pto-isa/docs/isa/TROWEXPANDADD.md`` and
    ``pto-isa/include/pto/cpu/TRowExpandOp.hpp``):

    * **Mode 1** - ColMajor with exactly one column: one scalar per row.
    * **Mode 2** - RowMajor with ``32/sizeof(dtype)`` columns: that 32-byte block
      *repeats* across the row,
      ``dst[i, j] = src0[i, j] + b_i[j % (32 / sizeof(dtype))]``.

    ``_row_expand_layout`` has already restricted the operand to one of these two
    modes, so anything that is not explicitly ColMajor single-column is Mode 2.
    The TileLib render runtime hands every tile operand its ``TileSpec`` config, so
    the body can pick the form without registering a second template.  Reading the
    mode is read from the operand layout and is never guessed.
    """
    layout = getattr(getattr(tile, "config", None), "b_layout", None)
    if layout not in {"row_major", "col_major"}:
        raise RuntimeError(
            "row-expand template cannot read the expanded operand's b_layout from "
            f"{type(tile).__name__}; refusing to guess the broadcast mode "
            "(Mode 1 = one scalar per row, Mode 2 = 32-byte block repeat)"
        )
    return layout == "row_major"


def _mul_i8(lhs, rhs, mask, dtype):
    """Multiply byte lanes modulo 256 using the supported 16-bit VMUL form."""
    lhs16 = pto.vbitcast(lhs, pto.ui16)
    rhs16 = pto.vbitcast(rhs, pto.ui16)
    mask16 = pto.make_mask(pto.ui16, pto.PAT.ALL)
    low = pto.vands(pto.vmul(lhs16, rhs16, mask16), pto.ui16(0xff), mask16)
    high_lhs = pto.vshrs(lhs16, 8, mask16)
    high_rhs = pto.vshrs(rhs16, 8, mask16)
    high = pto.vshls(pto.vmul(high_lhs, high_rhs, mask16), 8, mask16)
    return pto.vbitcast(pto.vor(low, high, mask16), dtype)


def _div_bf16(lhs, rhs, mask):
    mask32 = pto.make_mask(pto.f32, pto.PAT.ALL)
    lhs_even = pto.vcvt(lhs, pto.f32, mask, part=pto.VcvtPartMode.EVEN)
    lhs_odd = pto.vcvt(lhs, pto.f32, mask, part=pto.VcvtPartMode.ODD)
    rhs_even = pto.vcvt(rhs, pto.f32, mask, part=pto.VcvtPartMode.EVEN)
    rhs_odd = pto.vcvt(rhs, pto.f32, mask, part=pto.VcvtPartMode.ODD)
    even = pto.vcvt(pto.vdiv(lhs_even, rhs_even, mask32), pto.bf16, mask32,
                    rnd=pto.VcvtRoundMode.R, sat=pto.VcvtSatMode.NOSAT,
                    part=pto.VcvtPartMode.EVEN)
    odd = pto.vcvt(pto.vdiv(lhs_odd, rhs_odd, mask32), pto.bf16, mask32,
                   rnd=pto.VcvtRoundMode.R, sat=pto.VcvtSatMode.NOSAT,
                   part=pto.VcvtPartMode.ODD)
    return pto.vbitcast(
        pto.vor(pto.vbitcast(even, pto.ui16), pto.vbitcast(odd, pto.ui16), mask),
        pto.bf16,
    )


def _div_by_precision(lhs, rhs, mask, dtype):
    """Pointwise divide honouring the op's precisionType selector.

    Row/column expand carries the same ``DivAlgo`` attribute as ``pto.tdiv`` /
    ``pto.tdivs``, so it dispatches through the shared table instead of always
    issuing a native divide. The compensated sequence is pointwise, which is why
    these ops need no scratch operand any more.
    """
    precision = pto.get_op_attr("precisionType", "intrinsic")
    return div_dispatch.divide(lhs, rhs, dtype, mask, precision)


def _binary_result(lhs, rhs, mask, dtype, vector_op):
    if vector_op == pto.vmul and str(dtype) in {"i8", "ui8"}:
        return _mul_i8(lhs, rhs, mask, dtype)
    if vector_op == pto.vdiv:
        if str(dtype) == "bf16":
            return _div_bf16(lhs, rhs, mask)
        if str(dtype) in {"ui8", "ui16", "ui32"}:
            return div_uint_soft(lhs, rhs, mask, dtype)
        return _div_by_precision(lhs, rhs, mask, dtype)
    return vector_op(lhs, rhs, mask)


def _emit_row_expand_body(src0, src1, dst, vector_op):
    dtype = dst.dtype
    valid_rows, valid_cols = dst.valid_shape
    lanes = pto.elements_per_vreg(dtype)
    # pto-isa expands whichever operand does not match dst; src0 wins when both do.
    src0_is_full = _tile_is_full_sized(src0, dst)
    full = src0 if src0_is_full else src1
    expanded = src1 if src0_is_full else src0
    # Mode 1 is a per-row scalar splat, which is exactly `vdup {position = LOWEST}`.
    # Mode 2 carries a whole 32-byte block per row, so that operand is read with a
    # block-replicate load (`BRC_BLK`), mirroring pto-isa's `vlds(..., BLK)`; every
    # column of the block then takes part in the operation.
    # The block width always divides the vreg width (64/128/256 lanes vs 8/16/32
    # elements), so one BRC_BLK per column chunk is correct for every dtype.
    repeats_block = _row_operand_repeats_block(expanded)

    with pto.for_(0, valid_rows, step=1) as row:
        col_loop = pto.for_(0, valid_cols, step=lanes).carry(remained=valid_cols)
        with col_loop:
            col = col_loop.iv
            mask, remained = pto.make_mask(dtype, col_loop.remained)
            full_vec = pto.vlds(full[row, col:])
            if repeats_block:
                expanded_vec = pto.vlds(expanded[row, :], dist="BRC_BLK")
            else:
                expanded_vec = pto.vdup(pto.vlds(expanded[row, :]), mask)
            # Keep the authored operand order: dst = src0 OP src1.
            if src0_is_full:
                result = _binary_result(full_vec, expanded_vec, mask, dtype, vector_op)
            else:
                result = _binary_result(expanded_vec, full_vec, mask, dtype, vector_op)
            pto.vsts(result, dst[row, col:], mask)
            col_loop.update(remained=remained)


def _emit_column_expand_loop(src0, src1, dst, apply_op):
    """Shared column-expand loop.

    ``apply_op(lhs, rhs, mask, remained)`` receives the two operands in the authored
    order ``src0, src1`` and returns the result vector of the current column chunk.
    """
    dtype = dst.dtype
    valid_rows, valid_cols = dst.valid_shape
    lanes = pto.elements_per_vreg(dtype)
    # pto-isa expands whichever operand does not match dst; src0 wins when both do.
    src0_is_full = _tile_is_full_sized(src0, dst)
    full = src0 if src0_is_full else src1
    broadcast_row = src1 if src0_is_full else src0

    with pto.for_(0, valid_rows, step=1) as row:
        col_loop = pto.for_(0, valid_cols, step=lanes).carry(remained=valid_cols)
        with col_loop:
            col = col_loop.iv
            mask, remained = pto.make_mask(dtype, col_loop.remained)
            full_vec = pto.vlds(full[row, col:])
            broadcast_vec = pto.vlds(broadcast_row[0, col:])
            if src0_is_full:
                result = apply_op(full_vec, broadcast_vec, mask, remained)
            else:
                result = apply_op(broadcast_vec, full_vec, mask, remained)
            pto.vsts(result, dst[row, col:], mask)
            col_loop.update(remained=remained)


def _emit_column_expand_body(src0, src1, dst, vector_op):
    """Column expand built from a single ``dst = src0 OP src1`` vector op."""
    _emit_column_expand_loop(
        src0, src1, dst, lambda lhs, rhs, mask, _remained: _binary_result(lhs, rhs, mask, dst.dtype, vector_op)
    )


def _emit_column_expand_integer_div(src0, src1, dst, divide_i16, divide_i32):
    """Column expand for the integer division templates (i16 via f32, i32 in software)."""

    def _apply(lhs, rhs, mask, remained):
        if str(dst.dtype) == "i16":
            f32_mask, _ = pto.make_mask(pto.f32, (remained + 1) // 2)
            return divide_i16(lhs, rhs, mask, f32_mask)
        return divide_i32(lhs, rhs, mask)

    _emit_column_expand_loop(src0, src1, dst, _apply)


def _valid_row_expand_binary(src0_valid_shape=(), src1_valid_shape=(), dst_valid_shape=(), **_):
    """Either operand may be the row-expanded one, as long as the other matches dst."""
    expanded = _expanded_valid_shape(src0_valid_shape, src1_valid_shape, dst_valid_shape)
    if expanded is None:
        return False
    return _known_eq(expanded[0], dst_valid_shape[0]) and _known_ge(expanded[1], 1)


def _valid_column_expand_binary(src0_valid_shape=(), src1_valid_shape=(), dst_valid_shape=(), **_):
    """Either operand may carry the broadcast row, as long as the other matches dst."""
    expanded = _expanded_valid_shape(src0_valid_shape, src1_valid_shape, dst_valid_shape)
    if expanded is None:
        return False
    return _known_eq(expanded[1], dst_valid_shape[1]) and _known_ge(expanded[0], 1)


__all__ = [
    "FLOAT_SIGNATURES",
    "NUMERIC_SIGNATURES",
    "UNSIGNED_SIGNATURES",
    "register_column_expand_binary",
    "register_column_expand_expdif",
    "register_row_expand_binary",
    "register_row_expand_expdif",
]
