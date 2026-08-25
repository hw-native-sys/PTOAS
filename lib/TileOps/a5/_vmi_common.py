# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Shared helpers for A5 VMI TileLib candidates.

Per-op VMI candidates live next to the ordinary A5 TileLib template for the
same TileOp (for example ``tadd.py`` owns both the normal and VMI ``tadd``
candidates).  This module only contains common emitters, legality helpers, and
algorithm fragments reused by those per-op candidates.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from ptodsl import pto, scalar
from ptodsl._surface_values import unwrap_surface_value
from ptodsl._surface_types import Tile
from ptodsl._runtime_scalar_ops import emit_runtime_binary_op
from ptodsl._tile_template_tracing import (
    CanonicalBlockMap,
    CanonicalBlockCoordinate,
    _MaskValue,
    _TileProxy,
    _Value,
    _VectorValue,
    ScalarType,
    f16,
    bf16,
    f32,
    i8,
    i16,
    i32,
    ui8,
    for_,
    index_add,
    index_mul,
    tile_template as _trace_tile_template,
)
from ptodsl._vmi_namespace import vmi as _vmi_builder
from ptodsl.tilelib import registry as _tilelib_registry
from ptodsl.tilelib.registry import TileTemplateRegistry
from ptoas.mlir.dialects import pto as _pto_dialect
from ptodsl._types import VMI_LANE_COUNTS


def _snap_lanes(lanes: int) -> int:
    """Snap a logical lane count up to the nearest legal VMI lane count.

    VMI vreg/mask sizes must be one of (1, 2, 4, 8, 64, 128, 256).  When a
    tile column count (e.g. 32 or 512) is not in this set, round up to the
    next legal value so the mask/vreg is always valid.  The physical
    chunking into native vregs is left to the lowering passes.
    """
    for legal in VMI_LANE_COUNTS:
        if legal >= lanes:
            return legal
    return VMI_LANE_COUNTS[-1]


ElementwiseCompute = Callable[[Sequence[_VectorValue], _MaskValue], _VectorValue]
# A5 VMI elementwise/math helpers reuse this tuple as the default
# ``allowed_dtypes`` set. It mirrors lib/TileOps/a5/_common.py::FLOAT_DTYPES
# (f16, bf16, f32) so VMI candidates no longer lock f32-only and bf16 tiles
# stop falling back to the ordinary PTODSL path. See ADR-0003 PR1.
FLOAT_DTYPES = (f32, f16, bf16)
ui32 = ScalarType("ui32", lanes=64, mask_bits=32, bytewidth=4)
ui16 = ScalarType("ui16", lanes=128, mask_bits=16, bytewidth=2)
# Full dtype set aligned with lib/TileOps/a5/_common.py::NUMERIC_DTYPES. Used as
# the integer-inclusive ``allowed_dtypes`` default once PR2 widens elementwise
# candidates to int. Order matches _common.NUMERIC_DTYPES for greppability.
NUMERIC_DTYPES = (
    i8, i16, i32, ui8, ui16, ui32, f16, bf16, f32,
)
# String-name sets used by per-dtype constraint gates that compare against the
# ``*_dtype`` string metadata carried by canonical_vmi_template.
_FLOAT_DTYPE_NAMES = frozenset(d.name for d in FLOAT_DTYPES)
_NUMERIC_DTYPE_NAMES = frozenset(d.name for d in NUMERIC_DTYPES)

# The helpers below only adapt traced TileLib values and dtype metadata.  Use
# the raw VMI builder alias instead of pto.vmi so tile-template tracing helpers
# cannot shadow the builder namespace while a VMI template is being traced.


def _pto_dtype(dtype: ScalarType):
    descriptors = {
        "f32": pto.f32,
        "f16": pto.f16,
        "bf16": pto.bf16,
        "i8": pto.i8,
        "i16": pto.i16,
        "i32": pto.i32,
        "ui8": pto.ui8,
        "ui16": pto.ui16,
        "ui32": pto.ui32,
    }
    try:
        return descriptors[dtype.name]
    except KeyError as exc:
        raise ValueError(f"unsupported VMI TileLib dtype {dtype}") from exc


def _wrap_vreg(value, dtype: ScalarType) -> _VectorValue:
    return _VectorValue(unwrap_surface_value(value), dtype)


def _wrap_mask(value, dtype: ScalarType) -> _MaskValue:
    return _MaskValue(unwrap_surface_value(value), dtype)


def _has_null_pad(value) -> bool:
    return str(value).lower() in {"null", "0", "0x0", "0x00"}


def row_reduce_vmi_constraint(
    src_shape=(),
    src_valid_shape=(),
    workspace_shape=(),
    workspace_valid_shape=(),
    dst_shape=(),
    dst_valid_shape=(),
    src_dtype=None,
    workspace_dtype=None,
    dst_dtype=None,
    src_config=None,
    workspace_config=None,
    dst_config=None,
    **_,
):
    if (
        src_dtype not in _FLOAT_DTYPE_NAMES
        or workspace_dtype not in _FLOAT_DTYPE_NAMES
        or dst_dtype not in _FLOAT_DTYPE_NAMES
        or len(src_shape) != 2
        or len(src_valid_shape) != 2
        or len(workspace_shape) != 2
        or len(workspace_valid_shape) != 2
        or len(dst_shape) != 2
        or len(dst_valid_shape) != 2
    ):
        return False
    rows, cols = src_shape
    valid_rows, valid_cols = src_valid_shape
    workspace_rows, workspace_cols = workspace_shape
    sinkhorn_grouped_form = (
        src_shape == (8, 8)
        and src_valid_shape in {(8, 4), (8, 8)}
    )
    return (
        isinstance(rows, int)
        and isinstance(cols, int)
        and isinstance(workspace_rows, int)
        and isinstance(workspace_cols, int)
        and rows > 0
        and cols > 0
        and valid_rows == rows
        and isinstance(valid_cols, int)
        and 0 < valid_cols <= cols
        and (src_valid_shape == src_shape or sinkhorn_grouped_form)
        and (
            cols * _DTYPE_BYTEWIDTH[src_dtype] >= 128
            or sinkhorn_grouped_form
        )
        # The grouped row-reduce emit path loads the whole tile as one vector
        # (total_lanes = rows * physical_cols) through _vload_linear, which
        # snaps to a single VMI vreg. VMI vregs max out at 256 lanes, so any
        # tile wider than that would silently truncate the back rows. Reject
        # these shapes until the emit path gains proper 256-lane chunking; the
        # ordinary ptodsl template already chunks correctly. See P1-2.
        and rows * cols <= 256
        and workspace_rows == rows
        and workspace_cols >= 1
        and workspace_valid_shape == workspace_shape
        and dst_shape == (rows, 1)
        and dst_valid_shape == (rows, 1)
        and src_config is not None
        and src_config.b_layout == "row_major"
        and src_config.s_layout == "none_box"
        # Padding metadata is irrelevant when every physical lane is valid.
        and workspace_config is not None
        and _has_null_pad(workspace_config.pad_value)
        and dst_config is not None
        and dst_config.b_layout == "col_major"
        and dst_config.s_layout == "none_box"
        and _has_null_pad(dst_config.pad_value)
    )


def row_reduce_streaming_vmi_constraint(**context):
    """Use row streaming when the full static tile exceeds one A5 VREG."""

    if not row_reduce_vmi_constraint(**context):
        return False
    src_shape = context.get("src_shape", ())
    src_valid_shape = context.get("src_valid_shape", ())
    if len(src_shape) != 2 or src_valid_shape != src_shape:
        return False
    rows, cols = src_shape
    src_dtype = context.get("src_dtype")
    src_bytewidth = _DTYPE_BYTEWIDTH.get(src_dtype)
    if src_bytewidth is None:
        return False
    return (
        isinstance(rows, int)
        and isinstance(cols, int)
        and rows * cols * src_bytewidth > 256
        # The streaming emit path loads each row with lanes=physical_cols via
        # _vload_linear, which snaps down to a single 256-lane vreg. Columns
        # beyond 256 would be silently dropped. Reject until the emit path
        # gains 256-lane chunking; the ordinary ptodsl row-reduce template
        # already chunks correctly. See P1-2.
        and cols <= 256
    )


def sinkhorn_row_reduce_streaming_vmi_constraint(**context):
    """Recognize the statically bounded 8x8 Sinkhorn row-reduce forms."""

    if (
        context.get("src_shape") != (8, 8)
        or context.get("src_valid_shape") not in {(8, 4), (8, 8)}
    ):
        return False
    full_context = dict(context)
    full_context["src_valid_shape"] = (8, 8)
    return row_reduce_vmi_constraint(**full_context)


def _vreg_lanes(value: _VectorValue) -> int:
    return _pto_dialect.VMIVRegType(value.value.type).element_count


def _validate_same_dtype(operation: str, *values: _VectorValue) -> ScalarType:
    if not values:
        raise ValueError(f"{operation} expects at least one vector")
    dtype = values[0].dtype
    if any(value.dtype != dtype for value in values):
        raise TypeError(f"{operation} operands must use the same dtype")
    return dtype


def _validate_mask(operation: str, mask: _MaskValue, dtype: ScalarType) -> None:
    if not isinstance(mask, _MaskValue):
        raise TypeError(f"{operation} expects a VMI mask")
    if mask.dtype.mask_bits != dtype.mask_bits:
        raise TypeError(
            f"{operation} mask granularity b{mask.dtype.mask_bits} is incompatible "
            f"with {dtype} lanes using b{dtype.mask_bits}"
        )


def _validate_block_access(
    tile: _TileProxy,
    coordinate: CanonicalBlockCoordinate,
    *,
    operation: str,
) -> None:
    if not isinstance(tile, _TileProxy):
        raise TypeError(f"{operation} expects a traced Tile argument")
    if not isinstance(coordinate, CanonicalBlockCoordinate):
        raise TypeError(f"{operation} expects a CanonicalBlockCoordinate")
    if tile._spec.shape != coordinate.block_map.shape:
        raise ValueError(
            f"{operation} tile shape {tile._spec.shape} does not match "
            f"CanonicalBlockMap shape {coordinate.block_map.shape}"
        )


def _create_mask_lanes(
    active_lanes: int,
    vector_lanes: int,
    dtype: ScalarType,
    *,
    trace,
) -> _MaskValue:
    if not isinstance(dtype, ScalarType):
        raise TypeError("_create_mask_lanes expects a tile-template ScalarType")
    vector_lanes = _snap_lanes(vector_lanes)
    active_lanes = min(active_lanes, vector_lanes)
    if not 0 < active_lanes <= vector_lanes:
        raise ValueError("active_lanes must be in the range [1, vector_lanes]")
    active = trace.index_const(active_lanes)
    return _wrap_mask(_vmi_builder.create_mask(active.value, size=vector_lanes), dtype)


def _create_mask(
    block_map: CanonicalBlockMap,
    dtype: ScalarType,
    *,
    trace,
) -> _MaskValue:
    if not isinstance(block_map, CanonicalBlockMap):
        raise TypeError("_create_mask expects a CanonicalBlockMap")
    return _create_mask_lanes(
        block_map.logical_lanes,
        block_map.logical_lanes,
        dtype,
        trace=trace,
    )


def _prepare_tile_access(*tiles: _TileProxy) -> None:
    if not tiles:
        raise ValueError("_prepare_tile_access requires at least one Tile")
    for tile in tiles:
        if not isinstance(tile, _TileProxy):
            raise TypeError("_prepare_tile_access expects traced Tile arguments")
        tile._trace.ensure_tile_ptr(tile)


def _vload(tile: _TileProxy, coordinate: CanonicalBlockCoordinate) -> _VectorValue:
    _validate_block_access(tile, coordinate, operation="_vload")
    ptr_value = tile._trace.ensure_tile_ptr(tile)
    offset = tile._trace._coerce_index(coordinate.linear_offset)
    return _wrap_vreg(
        _vmi_builder.vload(
            ptr_value.value,
            offset.value,
            size=_snap_lanes(coordinate.block_map.logical_lanes),
        ),
        tile.element_type,
    )


def _vload_linear(tile: _TileProxy, offset, *, lanes: int) -> _VectorValue:
    if not isinstance(tile, _TileProxy):
        raise TypeError("_vload_linear expects a traced Tile argument")
    if not isinstance(lanes, int) or lanes <= 0:
        raise ValueError("_vload_linear lanes must be a positive integer")
    ptr_value = tile._trace.ensure_tile_ptr(tile)
    offset_value = tile._trace._coerce_index(offset)
    return _wrap_vreg(
        _vmi_builder.vload(ptr_value.value, offset_value.value,
                           size=_snap_lanes(lanes)),
        tile.element_type,
    )


def _vstore(
    vec: _VectorValue,
    tile: _TileProxy,
    coordinate: CanonicalBlockCoordinate,
    mask: _MaskValue,
) -> None:
    _validate_block_access(tile, coordinate, operation="_vstore")
    if vec.dtype != tile.element_type:
        raise TypeError("_vstore value and destination must use the same dtype")
    _validate_mask("_vstore", mask, vec.dtype)
    ptr_value = tile._trace.ensure_tile_ptr(tile)
    offset = tile._trace._coerce_index(coordinate.linear_offset)
    _vmi_builder.vstore(vec.value, ptr_value.value, offset.value, mask.value)


def _vstore_linear(
    vec: _VectorValue,
    tile: _TileProxy,
    offset,
    mask: _MaskValue,
) -> None:
    if not isinstance(tile, _TileProxy):
        raise TypeError("_vstore_linear expects a traced Tile destination")
    if vec.dtype != tile.element_type:
        raise TypeError("_vstore_linear value and destination must use the same dtype")
    _validate_mask("_vstore_linear", mask, vec.dtype)
    ptr_value = tile._trace.ensure_tile_ptr(tile)
    offset_value = tile._trace._coerce_index(offset)
    _vmi_builder.vstore(vec.value, ptr_value.value, offset_value.value, mask.value)


def _vbinary(name: str, lhs: _VectorValue, rhs: _VectorValue, mask: _MaskValue) -> _VectorValue:
    dtype = _validate_same_dtype(f"pto.vmi.{name}", lhs, rhs)
    _validate_mask(f"pto.vmi.{name}", mask, dtype)
    builder = getattr(_vmi_builder, name)
    return _wrap_vreg(builder(lhs.value, rhs.value, mask.value), dtype)


def _vadd(lhs: _VectorValue, rhs: _VectorValue, mask: _MaskValue) -> _VectorValue:
    return _vbinary("vadd", lhs, rhs, mask)


def _vsub(lhs: _VectorValue, rhs: _VectorValue, mask: _MaskValue) -> _VectorValue:
    return _vbinary("vsub", lhs, rhs, mask)


def _vmul(lhs: _VectorValue, rhs: _VectorValue, mask: _MaskValue) -> _VectorValue:
    return _vbinary("vmul", lhs, rhs, mask)


def _vdiv(lhs: _VectorValue, rhs: _VectorValue, mask: _MaskValue) -> _VectorValue:
    return _vbinary("vdiv", lhs, rhs, mask)


def _vmax(lhs: _VectorValue, rhs: _VectorValue, mask: _MaskValue) -> _VectorValue:
    return _vbinary("vmax", lhs, rhs, mask)


def _vmin(lhs: _VectorValue, rhs: _VectorValue, mask: _MaskValue) -> _VectorValue:
    return _vbinary("vmin", lhs, rhs, mask)


def _vand(lhs: _VectorValue, rhs: _VectorValue, mask: _MaskValue) -> _VectorValue:
    return _vbinary("vand", lhs, rhs, mask)


def _vor(lhs: _VectorValue, rhs: _VectorValue, mask: _MaskValue) -> _VectorValue:
    return _vbinary("vor", lhs, rhs, mask)


def _vshl(lhs: _VectorValue, rhs: _VectorValue, mask: _MaskValue) -> _VectorValue:
    return _vbinary("vshl", lhs, rhs, mask)


def _vshr(lhs: _VectorValue, rhs: _VectorValue, mask: _MaskValue) -> _VectorValue:
    return _vbinary("vshr", lhs, rhs, mask)


def _vunary(name: str, source: _VectorValue, mask: _MaskValue) -> _VectorValue:
    _validate_mask(f"pto.vmi.{name}", mask, source.dtype)
    builder = getattr(_vmi_builder, name)
    return _wrap_vreg(builder(source.value, mask.value), source.dtype)


def _vexp(source: _VectorValue, mask: _MaskValue) -> _VectorValue:
    return _vunary("vexp", source, mask)


def _vabs(source: _VectorValue, mask: _MaskValue) -> _VectorValue:
    return _vunary("vabs", source, mask)


def _vneg(source: _VectorValue, mask: _MaskValue) -> _VectorValue:
    return _vunary("vneg", source, mask)


def _vsqrt(source: _VectorValue, mask: _MaskValue) -> _VectorValue:
    return _vunary("vsqrt", source, mask)


def _vvec_scalar(
    name: str,
    source: _VectorValue,
    scalar: _Value,
    mask: _MaskValue,
) -> _VectorValue:
    _validate_mask(f"pto.vmi.{name}", mask, source.dtype)
    builder = getattr(_vmi_builder, name)
    return _wrap_vreg(builder(source.value, scalar.value, mask.value), source.dtype)


def _vadds(source: _VectorValue, scalar: _Value, mask: _MaskValue) -> _VectorValue:
    return _vvec_scalar("vadds", source, scalar, mask)


def _negate_scalar(scalar: _Value, dtype: ScalarType) -> _Value:
    # Pick the neutral zero literal matching the dtype kind: integer dtypes
    # need an int literal (0) — feeding `0.0` to an int materializer raises
    # `cannot materialize 0.0 as an integer constant`. See ADR-0003 PR2.3.
    zero_value = 0 if dtype.name[0] in {"i", "u"} else 0.0
    zero = _scalar_constant(zero_value, dtype)
    return _Value(emit_runtime_binary_op("sub", zero.value, scalar.value))


def _vmuls(source: _VectorValue, scalar: _Value, mask: _MaskValue) -> _VectorValue:
    return _vvec_scalar("vmuls", source, scalar, mask)


def _vmaxs(source: _VectorValue, scalar: _Value, mask: _MaskValue) -> _VectorValue:
    return _vvec_scalar("vmaxs", source, scalar, mask)


def _vmins(source: _VectorValue, scalar: _Value, mask: _MaskValue) -> _VectorValue:
    return _vvec_scalar("vmins", source, scalar, mask)


def _vcmp(
    lhs: _VectorValue,
    rhs: _VectorValue,
    seed: _MaskValue,
    cmp: str,
) -> _MaskValue:
    dtype = _validate_same_dtype("pto.vmi.vcmp", lhs, rhs)
    _validate_mask("pto.vmi.vcmp", seed, dtype)
    return _wrap_mask(_vmi_builder.vcmp(lhs.value, rhs.value, seed.value, cmp), dtype)


def _vcmps(
    source: _VectorValue,
    scalar: _Value,
    seed: _MaskValue,
    cmp: str,
) -> _MaskValue:
    _validate_mask("pto.vmi.vcmps", seed, source.dtype)
    return _wrap_mask(
        _vmi_builder.vcmps(source.value, scalar.value, seed.value, cmp),
        source.dtype,
    )


def _vsel(
    true_value: _VectorValue,
    false_value: _VectorValue,
    mask: _MaskValue,
) -> _VectorValue:
    dtype = _validate_same_dtype("pto.vmi.vsel", true_value, false_value)
    _validate_mask("pto.vmi.vsel", mask, dtype)
    return _wrap_vreg(
        _vmi_builder.vsel(mask.value, true_value.value, false_value.value),
        dtype,
    )


def _vmula(
    acc: _VectorValue,
    lhs: _VectorValue,
    rhs: _VectorValue,
    mask: _MaskValue,
) -> _VectorValue:
    dtype = _validate_same_dtype("pto.vmi.vmula", acc, lhs, rhs)
    _validate_mask("pto.vmi.vmula", mask, dtype)
    return _wrap_vreg(
        _vmi_builder.vmula(acc.value, lhs.value, rhs.value, mask.value),
        dtype,
    )


def _pand(lhs: _MaskValue, rhs: _MaskValue) -> _MaskValue:
    if lhs.dtype.mask_bits != rhs.dtype.mask_bits:
        raise TypeError("pto.vmi.vand mask operands must use the same granularity")
    return _wrap_mask(_vmi_builder.vand(lhs.value, rhs.value), lhs.dtype)


def _por(lhs: _MaskValue, rhs: _MaskValue) -> _MaskValue:
    if lhs.dtype.mask_bits != rhs.dtype.mask_bits:
        raise TypeError("pto.vmi.vor mask operands must use the same granularity")
    return _wrap_mask(_vmi_builder.vor(lhs.value, rhs.value), lhs.dtype)


def _pnot(mask: _MaskValue) -> _MaskValue:
    return _wrap_mask(_vmi_builder.vnot(mask.value), mask.dtype)


def _scalar_constant(value: float | int, dtype: ScalarType) -> _Value:
    return _Value(unwrap_surface_value(pto.const(value, dtype=_pto_dtype(dtype))))


def _vbrc(source: _VectorValue, *, lanes: int) -> _VectorValue:
    if not isinstance(lanes, int) or lanes <= 0:
        raise ValueError("_vbrc lanes must be a positive integer")
    return _wrap_vreg(_vmi_builder.vbrc(source.value, size=lanes), source.dtype)


def _vbrc_scalar(
    scalar: _Value,
    *,
    like: _VectorValue | None = None,
    dtype: ScalarType | None = None,
) -> _VectorValue:
    if like is None and dtype is None:
        raise TypeError("_vbrc_scalar requires like= or dtype=")
    ref_dtype = like.dtype if like is not None else dtype
    size = _vreg_lanes(like) if like is not None else dtype.lanes
    return _wrap_vreg(_vmi_builder.vbrc(scalar.value, size=size), ref_dtype)


def _vconstant(
    value: float | int,
    dtype: ScalarType,
    *,
    like: _VectorValue | None = None,
    lanes: int | None = None,
) -> _VectorValue:
    if like is None and lanes is None:
        raise TypeError("_vconstant requires like= or lanes=")
    if like is not None and lanes is not None:
        raise TypeError("_vconstant lanes cannot be combined with like=")
    scalar = _scalar_constant(value, dtype)
    if like is not None:
        return _vbrc_scalar(scalar, like=like)
    return _wrap_vreg(_vmi_builder.vbrc(scalar.value, size=lanes), dtype)


def _vreduce_max(source: _VectorValue, mask: _MaskValue) -> _VectorValue:
    _validate_mask("pto.vmi.vcmax", mask, source.dtype)
    return _wrap_vreg(_vmi_builder.vcmax(source.value, mask.value), source.dtype)


def _vreduce_add(source: _VectorValue, mask: _MaskValue) -> _VectorValue:
    _validate_mask("pto.vmi.vcadd", mask, source.dtype)
    return _wrap_vreg(
        _vmi_builder.vcadd(source.value, mask.value, reassoc=True),
        source.dtype,
    )


def _vcvt(
    source: _VectorValue,
    dst_dtype: ScalarType,
    *,
    rounding: str | None = None,
    saturate: str | None = None,
) -> _VectorValue:
    if not isinstance(dst_dtype, ScalarType):
        raise TypeError("_vcvt expects a tile-template destination ScalarType")
    return _wrap_vreg(
        _vmi_builder.vcvt(
            source.value,
            to_dtype=_pto_dtype(dst_dtype),
            rounding=rounding,
            saturate=saturate,
        ),
        dst_dtype,
    )


def _vinterpret_cast(source: _VectorValue, dst_dtype: ScalarType) -> _VectorValue:
    if not isinstance(dst_dtype, ScalarType):
        raise TypeError("_vinterpret_cast expects a destination ScalarType")
    source_type = _pto_dialect.VMIVRegType(source.value.type)
    source_bits = source_type.element_count * source.dtype.bytewidth * 8
    dst_bits = dst_dtype.bytewidth * 8
    if source_bits % dst_bits != 0:
        raise ValueError("_vinterpret_cast requires matching total bit width")
    return _wrap_vreg(
        _vmi_builder.vinterpret_cast(source.value, to_dtype=_pto_dtype(dst_dtype)),
        dst_dtype,
    )


def _qualify_op_name(op: str) -> str:
    return op if op.startswith("pto.") else f"pto.{op}"


def _normalize_op_name(op: str) -> str:
    return op[4:] if op.startswith("pto.") else op


class _VMITileTemplateRegistry(TileTemplateRegistry):
    def lookup(self, op: str, target: str) -> list:
        candidates = super().lookup(op, target)
        if candidates:
            return candidates
        qualified = _qualify_op_name(op)
        if qualified != op:
            candidates = super().lookup(qualified, target)
            if candidates:
                return candidates
        normalized = _normalize_op_name(op)
        if normalized != op:
            return super().lookup(normalized, target)
        return []


VMI_TILELIB_REGISTRY = _VMITileTemplateRegistry()


_DTYPE_BYTEWIDTH = {
    "f32": 4,
    "i32": 4,
    "ui32": 4,
    "f16": 2,
    "bf16": 2,
    "i16": 2,
    "ui16": 2,
    "i8": 1,
    "ui8": 1,
}

# Logical lanes per A5 physical VREG (256 bytes) for each dtype. Used by
# constraints that previously hardcoded ``f32.lanes`` (64). Mirrors the
# ``ScalarType.lanes`` values from ptodsl._tile_template_tracing.
_DTYPE_LANES = {
    "f32": 64,
    "i32": 64,
    "ui32": 64,
    "f16": 128,
    "bf16": 128,
    "i16": 128,
    "ui16": 128,
    "i8": 256,
    "ui8": 256,
}


def _lanes_for_dtype(dtype_name: str | None) -> int | None:
    """Return the per-dtype A5 VREG lane count, or ``None`` if unsupported."""
    if dtype_name is None:
        return None
    return _DTYPE_LANES.get(dtype_name)


def _physical_row_vmi_constraint(min_row_bytes: int, **metadata) -> bool:
    """Check the minimum byte width needed by a VMI logical row."""

    if not isinstance(min_row_bytes, int) or min_row_bytes <= 0:
        raise ValueError("VMI row byte constraint must be a positive integer")

    row_bytes = []
    for name, shape in metadata.items():
        if not name.endswith("_shape") or name.endswith("_valid_shape"):
            continue
        if not isinstance(shape, (tuple, list)) or len(shape) != 2:
            continue
        cols = shape[1]
        dtype = metadata.get(f"{name[:-6]}_dtype")
        bytewidth = _DTYPE_BYTEWIDTH.get(dtype)
        if isinstance(cols, int) and cols > 0 and bytewidth is not None:
            row_bytes.append(cols * bytewidth)
    return not row_bytes or max(row_bytes) >= min_row_bytes


def full_physical_row_vmi_constraint(**metadata) -> bool:
    """Require at least one complete 256-byte physical row."""

    return _physical_row_vmi_constraint(256, **metadata)


def min_128b_row_vmi_constraint(**metadata) -> bool:
    """Allow statically full rows with at least 128 bytes of data.

    Sub-VL rows need a separately validated A5 access and mask contract; static
    allocation bounds alone are not sufficient to make them VMI candidates.
    """

    return _physical_row_vmi_constraint(128, **metadata)


def sinkhorn_compact_elementwise_vmi_constraint(**metadata) -> bool:
    """Accept only the static compact f32 forms used by DSv4 Sinkhorn."""

    operands = []
    for name, shape in metadata.items():
        if not name.endswith("_shape") or name.endswith("_valid_shape"):
            continue
        operand = name[:-6]
        valid_shape = metadata.get(f"{operand}_valid_shape")
        dtype = metadata.get(f"{operand}_dtype")
        config = metadata.get(f"{operand}_config")
        if not isinstance(shape, (tuple, list)) or not isinstance(
            valid_shape, (tuple, list)
        ):
            return False
        operands.append((tuple(shape), tuple(valid_shape), dtype, config))

    if not operands:
        return False
    shape, valid_shape, _, _ = operands[0]
    accepted_form = (shape, valid_shape) in {
        ((8, 8), (8, 8)),
        ((8, 8), (8, 4)),
        ((1, 8), (1, 8)),
    }
    return accepted_form and all(
        operand_shape == shape
        and operand_valid_shape == valid_shape
        and dtype in _FLOAT_DTYPE_NAMES
        and config is not None
        and config.b_layout == "row_major"
        and config.s_layout == "none_box"
        for operand_shape, operand_valid_shape, dtype, config in operands
    )


def _is_safe_static_row_prefix(
    shape: tuple[int, ...],
    valid_shape: tuple[int, ...],
    *,
    native_lanes: int,
) -> bool:
    """Whether a static valid row prefix can be read as whole physical chunks."""

    if len(shape) != 2 or len(valid_shape) != 2:
        return False
    rows, physical_cols = shape
    valid_rows, logical_cols = valid_shape
    if not all(
        isinstance(dim, int)
        for dim in (rows, physical_cols, valid_rows, logical_cols)
    ):
        return False
    if rows <= 0 or physical_cols <= 0 or valid_rows != rows:
        return False
    if logical_cols <= 0 or logical_cols > physical_cols:
        return False
    physical_read_cols = (
        (logical_cols + native_lanes - 1) // native_lanes
    ) * native_lanes
    return valid_shape == shape or physical_read_cols <= physical_cols


def row_expand_binary_vmi_constraint(
    src_shape=(),
    src_valid_shape=(),
    src_dtype=None,
    src_config=None,
    row_values_shape=(),
    row_values_valid_shape=(),
    row_values_config=None,
    dst_shape=(),
    dst_valid_shape=(),
    dst_dtype=None,
    dst_config=None,
    **_,
):
    """Accept the static DSv4 row-tensor/[rows, 1] broadcast form."""

    if not all(
        len(shape) == 2
        for shape in (
            src_shape,
            src_valid_shape,
            row_values_shape,
            row_values_valid_shape,
            dst_shape,
            dst_valid_shape,
        )
    ):
        return False
    src_lanes = _lanes_for_dtype(src_dtype)
    dst_lanes = _lanes_for_dtype(dst_dtype)
    src_bytewidth = _DTYPE_BYTEWIDTH.get(src_dtype)
    if src_lanes is None or dst_lanes is None or src_bytewidth is None:
        return False
    rows, logical_cols = src_valid_shape
    safe_row_access = (
        _is_safe_static_row_prefix(
            src_shape, src_valid_shape, native_lanes=src_lanes
        )
        and _is_safe_static_row_prefix(
            dst_shape, dst_valid_shape, native_lanes=dst_lanes
        )
    )
    return (
        rows > 0
        and logical_cols > 0
        and logical_cols * src_bytewidth >= 128
        # The row-expand binary emit path loads each row with lanes=io_lanes
        # (rounded to logical_cols) via _vload_linear, which snaps down to a
        # single 256-lane vreg. Columns beyond 256 would be silently dropped.
        # Reject until the emit path gains 256-lane chunking; the ordinary
        # ptodsl row-expand template already chunks correctly. See P1-2.
        and logical_cols <= 256
        and src_valid_shape == src_shape
        and dst_valid_shape == dst_shape
        and safe_row_access
        and dst_shape == src_shape
        and dst_valid_shape == src_valid_shape
        and row_values_shape == (rows, 1)
        and row_values_valid_shape == row_values_shape
        and src_config is not None
        and src_config.b_layout == "row_major"
        and src_config.s_layout == "none_box"
        and row_values_config is not None
        and row_values_config.b_layout == "col_major"
        and row_values_config.s_layout == "none_box"
        and dst_config is not None
        and dst_config.b_layout == "row_major"
        and dst_config.s_layout == "none_box"
    )


def sinkhorn_row_expand_vmi_constraint(
    src_shape=(),
    src_valid_shape=(),
    src_config=None,
    row_values_shape=(),
    row_values_valid_shape=(),
    row_values_config=None,
    dst_shape=(),
    dst_valid_shape=(),
    dst_config=None,
    **_,
):
    """Accept Sinkhorn row expands whose compact state uses gather loading."""

    return (
        src_shape == (8, 8)
        and src_valid_shape in {(8, 4), (8, 8)}
        and dst_shape == src_shape
        and dst_valid_shape == src_valid_shape
        and row_values_shape == (8, 1)
        and row_values_valid_shape == row_values_shape
        and src_config is not None
        and src_config.b_layout == "row_major"
        and src_config.s_layout == "none_box"
        and row_values_config is not None
        and row_values_config.b_layout == "col_major"
        and row_values_config.s_layout == "none_box"
        and dst_config is not None
        and dst_config.b_layout == "row_major"
        and dst_config.s_layout == "none_box"
    )


def col_expand_vmi_constraint(
    src_shape=(),
    src_valid_shape=(),
    src_dtype=None,
    src_config=None,
    dst_shape=(),
    dst_valid_shape=(),
    dst_dtype=None,
    dst_config=None,
    **_,
):
    """Accept a static full source row broadcast over destination rows."""

    if not all(
        len(shape) == 2
        for shape in (src_shape, src_valid_shape, dst_shape, dst_valid_shape)
    ):
        return False
    src_bytewidth = _DTYPE_BYTEWIDTH.get(src_dtype)
    if src_bytewidth is None:
        return False
    rows, cols = dst_shape
    sinkhorn_grouped_form = (
        src_shape == (1, 8)
        and src_valid_shape in {(1, 4), (1, 8)}
        and dst_shape == (8, 8)
        and dst_valid_shape == (8, src_valid_shape[1])
    )
    return (
        rows > 0
        and cols > 0
        and src_shape == (1, cols)
        and (
            (
                src_valid_shape == (1, cols)
                and dst_valid_shape == dst_shape
                and cols * src_bytewidth >= 128
                # The col-expand emit path loads the single source row with
                # lanes=cols via _vload_linear, which snaps to a single
                # 256-lane vreg. Columns beyond 256 would be silently dropped
                # from the broadcast. Reject until the emit path gains
                # 256-lane chunking; the ordinary ptodsl col-expand template
                # already chunks correctly. See P1-2.
                and cols <= 256
            )
            or sinkhorn_grouped_form
        )
        and src_config is not None
        and src_config.b_layout == "row_major"
        and src_config.s_layout == "none_box"
        and dst_config is not None
        and dst_config.b_layout == "row_major"
        and dst_config.s_layout == "none_box"
    )


def col_reduce_vmi_constraint(
    src_shape=(),
    src_valid_shape=(),
    src_dtype=None,
    src_config=None,
    dst_shape=(),
    dst_valid_shape=(),
    dst_dtype=None,
    dst_config=None,
    **_,
):
    """Accept a static full [rows, cols] -> [1, cols] col-reduce.

    Mirrors the shape checks in `_validate_col_reduce_tiles` plus a
    wide-column guard: `emit_col_reduce_vmi` issues one wide load/mask/store
    with `lanes = block_map.cols = cols` per surviving row, and `_snap_lanes`
    silently caps that at 256. Columns beyond 256 would be dropped (or trip a
    lane-count verifier mismatch when the accumulator is materialized at the
    full width), so reject until the emit path gains 256-lane chunking; the
    ordinary ptodsl col-reduce template already chunks correctly. See P1-2.
    """

    if not all(
        len(shape) == 2
        for shape in (src_shape, src_valid_shape, dst_shape, dst_valid_shape)
    ):
        return False
    if src_dtype not in _NUMERIC_DTYPE_NAMES or dst_dtype not in _NUMERIC_DTYPE_NAMES:
        return False
    src_bytewidth = _DTYPE_BYTEWIDTH.get(src_dtype)
    if src_bytewidth is None:
        return False
    rows, cols = src_shape
    return (
        rows > 0
        and cols > 0
        and cols * src_bytewidth >= 128
        and cols <= 256
        and src_valid_shape == src_shape
        and dst_shape == (1, cols)
        and dst_valid_shape == dst_shape
        and src_config is not None
        and src_config.b_layout == "row_major"
        and src_config.s_layout == "none_box"
        and dst_config is not None
        and dst_config.b_layout == "row_major"
        and dst_config.s_layout == "none_box"
    )


def col_expand_binary_vmi_constraint(
    src_shape=(),
    src_valid_shape=(),
    src_dtype=None,
    src_config=None,
    col_values_shape=(),
    col_values_valid_shape=(),
    col_values_config=None,
    dst_shape=(),
    dst_valid_shape=(),
    dst_dtype=None,
    dst_config=None,
    **_,
):
    """Accept the static [rows, cols] + [1, cols] -> [rows, cols] col-expand.

    Mirrors the shape checks in `_validate_col_expand_binary_tiles` plus a
    wide-column guard: `emit_col_expand_binary_vmi` issues one wide load of the
    [1, cols] broadcast row (`_vload_linear(..., lanes=cols)`) plus a wide
    load/mask/store per dst row, all of which snap to a single 256-lane vreg.
    Columns beyond 256 would be silently dropped (only the front 256 of each
    row gets the broadcast applied). Reject until the emit path gains
    256-lane chunking; the ordinary ptodsl col-expand-binary template already
    chunks correctly. See P1-2.
    """

    if not all(
        len(shape) == 2
        for shape in (
            src_shape,
            src_valid_shape,
            col_values_shape,
            col_values_valid_shape,
            dst_shape,
            dst_valid_shape,
        )
    ):
        return False
    src_bytewidth = _DTYPE_BYTEWIDTH.get(src_dtype)
    if src_bytewidth is None:
        return False
    rows, cols = src_shape
    return (
        rows > 0
        and cols > 0
        and cols * src_bytewidth >= 128
        and cols <= 256
        and src_valid_shape == src_shape
        and dst_shape == src_shape
        and dst_valid_shape == dst_shape
        and col_values_shape == (1, cols)
        and col_values_valid_shape == col_values_shape
        and src_config is not None
        and src_config.b_layout == "row_major"
        and src_config.s_layout == "none_box"
        and col_values_config is not None
        and col_values_config.b_layout == "row_major"
        and col_values_config.s_layout == "none_box"
        and dst_config is not None
        and dst_config.b_layout == "row_major"
        and dst_config.s_layout == "none_box"
    )


def convert_vmi_constraint(
    src_shape=(),
    src_valid_shape=(),
    src_dtype=None,
    src_config=None,
    dst_shape=(),
    dst_valid_shape=(),
    dst_dtype=None,
    dst_config=None,
    round_mode=None,
    sat_mode=None,
    **_,
):
    """Accept static full row-major conversions with matching logical shape."""

    supported_round_modes = {
        ("bf16", "f32"): {"ROUND"},
        ("f16", "f32"): {"RINT", "ROUND"},
        ("i32", "f32"): {"RINT", "ROUND"},
        ("f32", "bf16"): {"RINT", "ROUND"},
        ("f32", "f16"): {"RINT", "ROUND"},
        # The current fp-to-int candidate is validated only for truncation.
        # RINT/ROUND require separate semantic validation and remain on the
        # ordinary fallback path.
        ("f32", "i32"): {"TRUNC"},
        ("i32", "f16"): {"ROUND"},
    }
    allowed_round_modes = supported_round_modes.get((src_dtype, dst_dtype))
    if not all(
        len(shape) == 2
        for shape in (src_shape, src_valid_shape, dst_shape, dst_valid_shape)
    ):
        return False
    rows, cols = src_shape
    src_bytewidth = _DTYPE_BYTEWIDTH.get(src_dtype)
    dst_bytewidth = _DTYPE_BYTEWIDTH.get(dst_dtype)
    return (
        rows > 0
        and cols > 0
        and src_bytewidth is not None
        and dst_bytewidth is not None
        and max(cols * src_bytewidth, cols * dst_bytewidth) >= 128
        and allowed_round_modes is not None
        and round_mode in allowed_round_modes
        and sat_mode in ("DEFAULT", "ON", "OFF")
        and src_valid_shape == src_shape
        and dst_shape == src_shape
        and dst_valid_shape == dst_shape
        and src_config is not None
        and src_config.b_layout == "row_major"
        and src_config.s_layout == "none_box"
        and dst_config is not None
        and dst_config.b_layout == "row_major"
        and dst_config.s_layout == "none_box"
    )


# Reduce kind -> (merge op, identity element). The identity mirrors pto-isa
# `TColReduceOps.hpp` `InstrOp::InitVal` / a5 `Padding<T>::Min/Max`:
#   max -> vmax, init -inf (Padding<T>::Min)
#   min -> vmin, init +inf (Padding<T>::Max)
#   add -> vadd, init 0
#   prod-> vmul, init 1
# `emit_col_reduce_vmi` exercises max/min/add today; prod maps to a vmi
# merge op the VMI tilelib does not yet expose as an elementwise-vector form
# (only the -s scalar variant), so it raises if used.
_REDUCE_MERGE_OP = {
    "max": _vmax,
    "min": _vmin,
    "add": _vadd,
}


_ONE_VECTOR_CANDIDATES = {"texpands", "tmov", "tcolexpand"}
_THREE_VECTOR_CANDIDATES = {
    "add",
    "div",
    "mul",
    "sub",
    "tadd",
    "tdiv",
    "tmax",
    "tmul",
    "tsub",
    "tcolexpandadd",
    "tcolexpanddiv",
    "tcolexpandmul",
    "tcolexpandsub",
    "tcolmax",
    "tcolmin",
    "tcolsum",
    "trowexpanddiv",
    "trowexpandmul",
    "trowexpandsub",
    "tcvt",
}


def _default_resource_vector_values(op: str) -> int:
    """Return the conservative peak wide values for a canonical candidate."""

    unqualified = op.removeprefix("pto.")
    if unqualified in _ONE_VECTOR_CANDIDATES:
        return 1
    if unqualified in _THREE_VECTOR_CANDIDATES:
        return 3
    # Unary and vector-scalar candidates materialize an input and result.
    return 2


_vmi_candidate_id_counters: dict[str, int] = {}


def _next_vmi_candidate_id(qualified_op: str) -> int:
    """Return the next per-op auto-incremented candidate id.

    The first VMI candidate for an op gets id 1000 (matching the historical
    default); subsequent ones get 1001, 1002, etc. so they never collide.
    Templates that pass an explicit ``candidate_id=`` are left alone.
    """
    value = _vmi_candidate_id_counters.get(qualified_op, 1000)
    _vmi_candidate_id_counters[qualified_op] = value + 1
    return value


def canonical_vmi_template(
    *,
    target: str = "a5",
    op: str,
    name: str | None = None,
    dtypes: tuple | list = (),
    context_constraints: dict[str, tuple[object, ...]] | None = None,
    constraints: tuple[object, ...] | list[object] = (),
    tags: tuple[str, ...] | list[str] = (),
    priority: int = 100,
    candidate_id: int | None = None,
    single_logical_row_loop: bool = True,
    requires_full_physical_row: bool = True,
    min_row_bytes: int = 256,
    resource_scope: str = "row",
    resource_vector_values: int | None = None,
    resource_chunk_streaming: bool = False,
):
    """Register one canonical VMI implementation in this provider module."""

    def decorator(fn):
        qualified_op = _qualify_op_name(op)
        effective_constraints = tuple(constraints)
        if requires_full_physical_row:
            if min_row_bytes == 256:
                row_constraint = full_physical_row_vmi_constraint
            elif min_row_bytes == 128:
                row_constraint = min_128b_row_vmi_constraint
            else:
                raise ValueError(
                    "canonical VMI templates support only 128B or 256B row constraints"
                )
            effective_constraints = (
                row_constraint,
                *effective_constraints,
            )
        effective_id = candidate_id
        if effective_id is None:
            effective_id = _next_vmi_candidate_id(qualified_op)
        descriptor = _trace_tile_template(
            target=target,
            op=qualified_op,
            name=name,
            ir_level="vmi",
            dtypes=dtypes,
            context_constraints=context_constraints,
            constraints=effective_constraints,
            tags=tuple(tags),
            priority=priority,
            candidate_id=effective_id,
            single_logical_row_loop=single_logical_row_loop,
            resource_scope=resource_scope,
            resource_vector_values=(
                resource_vector_values
                if resource_vector_values is not None
                else _default_resource_vector_values(op)
            ),
            resource_chunk_streaming=resource_chunk_streaming,
        )(fn)
        _tilelib_registry.register(descriptor)
        VMI_TILELIB_REGISTRY.register(descriptor)
        return descriptor

    return decorator


def emit_elementwise_vmi(
    dst: _TileProxy,
    sources: Sequence[_TileProxy],
    compute: ElementwiseCompute,
    *,
    logical_lanes: int | None = None,
    allowed_dtypes: Sequence[ScalarType] = FLOAT_DTYPES,
) -> None:
    """Emit one flat principal loop for a standalone elementwise candidate.

    Static full-shape row-major tiles use the equivalent contiguous
    native-chunk domain when they are either one multi-VL row or several
    short rows that exactly pack a native vector. This matches the PTO-ISA 1D
    implementation while preserving a common principal loop for compatible
    elementwise chains. Partial or grouped shapes retain row-aware domains.
    """

    if not sources:
        raise ValueError("emit_elementwise_vmi requires at least one source tile")
    if logical_lanes is None:
        logical_lanes = dst._spec.shape[1]
    _validate_elementwise_tiles(
        dst,
        sources,
        logical_lanes=logical_lanes,
        allowed_dtypes=allowed_dtypes,
    )

    rows, cols = dst._spec.shape
    native_lanes = dst.element_type.lanes
    valid_shape = dst._spec.effective_valid_shape
    if dst.element_type == f32 and (
        ((rows, cols), valid_shape)
        in {
            ((8, 8), (8, 8)),
            ((8, 8), (8, 4)),
            ((1, 8), (1, 8)),
        }
    ):
        _emit_elementwise_sinkhorn_grouped_vmi(dst, sources, compute)
        return
    if logical_lanes == cols and _can_use_contiguous_native_chunks(
        dst, sources, chunk_lanes=native_lanes
    ):
        _emit_elementwise_contiguous_blocks_vmi(
            dst, sources, compute, block_lanes=native_lanes
        )
        return

    block_map = CanonicalBlockMap.from_tile(dst, logical_lanes=logical_lanes)

    _prepare_tile_access(*sources, dst)
    mask = _create_mask(block_map, dst.element_type, trace=dst._trace)
    with for_(0, block_map.logical_block_count, step=1) as logical_block:
        coordinate = block_map.coordinate(logical_block)
        values = tuple(_vload(source, coordinate) for source in sources)
        result = compute(values, mask)
        _vstore(result, dst, coordinate, mask)


def _emit_elementwise_sinkhorn_grouped_vmi(
    dst: _TileProxy,
    sources: Sequence[_TileProxy],
    compute: ElementwiseCompute,
) -> None:
    """Process one statically proven compact Sinkhorn tile safely."""

    rows, cols = dst._spec.shape
    _, active_cols = dst._spec.effective_valid_shape
    total_lanes = rows * cols
    _prepare_tile_access(*sources, dst)

    if rows == 1:
        mask = _create_mask_lanes(
            active_cols, cols, dst.element_type, trace=dst._trace
        )
        zero = dst._trace.index_const(0)
        with for_(0, 1, step=1):
            values = tuple(
                _vload_linear(source, zero, lanes=cols) for source in sources
            )
            result = compute(values, mask)
            _vstore_linear(result, dst, zero, mask)
        return

    # The compact 8x8 storage is one contiguous 64-lane physical tile. The
    # grouped mask describes valid columns in each row; it is not a strided
    # memory layout. Loading each 8-lane row separately would advance the A5
    # vector load by only 32 bytes and is unsafe after row zero.
    zero = dst._trace.index_const(0)
    active = dst._trace.index_const(active_cols)
    mask = _wrap_mask(
        _vmi_builder.create_mask(active.value, size=total_lanes, group=rows),
        dst.element_type,
    )
    with for_(0, 1, step=1):
        values = tuple(
            _vload_linear(source, zero, lanes=total_lanes) for source in sources
        )
        result = compute(values, mask)
        _vstore_linear(result, dst, zero, mask)


def _emit_elementwise_contiguous_blocks_vmi(
    dst: _TileProxy,
    sources: Sequence[_TileProxy],
    compute: ElementwiseCompute,
    *,
    block_lanes: int,
) -> None:
    """Process a full contiguous tile as one native-chunk principal loop."""

    rows, cols = dst._spec.shape
    total_lanes = rows * cols
    if not _can_use_contiguous_native_chunks(
        dst, sources, chunk_lanes=block_lanes
    ):
        raise ValueError("contiguous VMI blocks require an exactly tiled full shape")

    _prepare_tile_access(*sources, dst)
    mask = _create_mask_lanes(
        block_lanes, block_lanes, dst.element_type, trace=dst._trace
    )
    with for_(0, total_lanes, step=block_lanes) as offset:
        values = tuple(
            _vload_linear(source, offset, lanes=block_lanes)
            for source in sources
        )
        result = compute(values, mask)
        _vstore_linear(result, dst, offset, mask)


def _can_use_contiguous_native_chunks(
    dst: _TileProxy,
    sources: Sequence[_TileProxy] = (),
    *,
    chunk_lanes: int,
) -> bool:
    """Return whether a tile chain is a full contiguous linear stream."""

    if not isinstance(chunk_lanes, int) or chunk_lanes <= 0:
        return False
    tiles = (dst, *sources)
    if any(not isinstance(tile, _TileProxy) for tile in tiles):
        return False
    rows, cols = dst._spec.shape
    if not isinstance(rows, int) or not isinstance(cols, int):
        return False
    total_lanes = rows * cols
    one_row_multi_chunk = (
        rows == 1 and cols > chunk_lanes and cols % chunk_lanes == 0
    )
    packed_short_rows = (
        rows > 1
        and cols < chunk_lanes
        and chunk_lanes % cols == 0
        and total_lanes % chunk_lanes == 0
    )
    multi_row_multi_chunk = (
        rows > 1 and cols > chunk_lanes and cols % chunk_lanes == 0
    )
    if (
        not one_row_multi_chunk
        and not packed_short_rows
        and not multi_row_multi_chunk
    ):
        return False
    return all(
        tile._spec.shape == dst._spec.shape
        and tile._spec.effective_valid_shape == tile._spec.shape
        and tile._spec.b_layout == "row_major"
        and getattr(tile._spec, "s_layout", "none_box") == "none_box"
        for tile in tiles
    )


def emit_scalar_fill_vmi(
    scalar: _Value,
    dst: _TileProxy,
    *,
    allowed_dtypes: Sequence[ScalarType] = FLOAT_DTYPES,
) -> None:
    """Broadcast a runtime scalar into each wide logical row of ``dst``."""

    if not isinstance(dst, _TileProxy):
        raise TypeError("scalar-fill VMI candidate destination must be a traced Tile")
    if dst.element_type not in allowed_dtypes:
        raise ValueError(
            "VMI scalar-fill candidate dtype is not supported; "
            f"got {dst.element_type}, expected one of {tuple(allowed_dtypes)}"
        )
    if dst._spec.b_layout != "row_major":
        raise ValueError("VMI scalar-fill candidates require row-major tiles")

    rows, cols = dst._spec.shape
    native_lanes = dst.element_type.lanes
    if _can_use_contiguous_native_chunks(dst, chunk_lanes=native_lanes):
        total_lanes = rows * cols
        _prepare_tile_access(dst)
        mask = _create_mask_lanes(
            native_lanes, native_lanes, dst.element_type, trace=dst._trace
        )
        fill = _wrap_vreg(
            _vmi_builder.vbrc(scalar.value, size=native_lanes),
            dst.element_type,
        )
        with for_(0, total_lanes, step=native_lanes) as offset:
            _vstore_linear(fill, dst, offset, mask)
        return

    block_map = CanonicalBlockMap.from_tile(dst)
    _prepare_tile_access(dst)
    mask = _create_mask(block_map, dst.element_type, trace=dst._trace)
    fill = _wrap_vreg(
        _vmi_builder.vbrc(scalar.value, size=block_map.logical_lanes),
        dst.element_type,
    )
    with for_(0, block_map.logical_block_count, step=1) as logical_block:
        _vstore(fill, dst, block_map.coordinate(logical_block), mask)


def _validate_elementwise_tiles(
    dst: _TileProxy,
    sources: Sequence[_TileProxy],
    *,
    logical_lanes: int,
    allowed_dtypes: Sequence[ScalarType],
) -> None:
    if not isinstance(dst, _TileProxy):
        raise TypeError("elementwise VMI candidate destination must be a traced Tile")
    if dst.element_type not in allowed_dtypes:
        raise ValueError(
            "VMI elementwise candidate dtype is not supported; "
            f"got {dst.element_type}, expected one of {tuple(allowed_dtypes)}"
        )
    if dst._spec.b_layout != "row_major":
        raise ValueError("VMI elementwise candidates require row-major tiles")
    for source in sources:
        if not isinstance(source, _TileProxy):
            raise TypeError("elementwise VMI candidate sources must be traced Tiles")
        if source._spec.shape != dst._spec.shape:
            raise ValueError(
                "elementwise VMI candidate source and destination shapes must match; "
                f"got {source._spec.shape} and {dst._spec.shape}"
            )
        if source.element_type != dst.element_type:
            raise ValueError(
                "elementwise VMI candidate source and destination dtypes must match; "
                f"got {source.element_type} and {dst.element_type}"
            )
        if source._spec.b_layout != dst._spec.b_layout:
            raise ValueError("elementwise VMI candidate layouts must match")
        if source._spec.effective_valid_shape != dst._spec.effective_valid_shape:
            raise ValueError(
                "elementwise VMI candidate valid shapes must match"
            )


# Elementwise compute closures used by the per-op VMI candidates below. The
# integer cases of `_add`/`_mul`/`_sub` follow A5 `vadd`/`vmul`/`vsub` default
# semantics: **wrap-around** (two's-complement wrap on overflow), NOT
# saturating. This matches the A5 vector ISA and the ordinary PTODSL
# `pto.vadd`/`pto.vmul` path. Saturating integer add/mul is not modeled here —
# it would require a `sat_mode` context attr plumbed into vadd/vmul lowering
# (currently only `vcvt` has saturation; see VMILowerUnifiedToLegacy.cpp:431+),
# and is listed out-of-scope by ADR-0003. Per-op candidates only declare the
# integer dtypes their TileOp ODS actually accepts (see each op's dtypes=).

def _add(values: Sequence[_VectorValue], mask: _MaskValue) -> _VectorValue:
    if len(values) != 2:
        raise ValueError("tadd VMI candidate expects two source vectors")
    return _vadd(values[0], values[1], mask)


def _exp(values: Sequence[_VectorValue], mask: _MaskValue) -> _VectorValue:
    if len(values) != 1:
        raise ValueError("texp VMI candidate expects one source vector")
    return _vexp(values[0], mask)


def _abs(values: Sequence[_VectorValue], mask: _MaskValue) -> _VectorValue:
    if len(values) != 1:
        raise ValueError("tabs VMI candidate expects one source vector")
    return _vabs(values[0], mask)


def _neg(values: Sequence[_VectorValue], mask: _MaskValue) -> _VectorValue:
    if len(values) != 1:
        raise ValueError("tneg VMI candidate expects one source vector")
    return _vneg(values[0], mask)


def _sub(values: Sequence[_VectorValue], mask: _MaskValue) -> _VectorValue:
    if len(values) != 2:
        raise ValueError("tsub VMI candidate expects two source vectors")
    return _vsub(values[0], values[1], mask)


def _mul(values: Sequence[_VectorValue], mask: _MaskValue) -> _VectorValue:
    if len(values) != 2:
        raise ValueError("tmul VMI candidate expects two source vectors")
    return _vmul(values[0], values[1], mask)


def _max(values: Sequence[_VectorValue], mask: _MaskValue) -> _VectorValue:
    if len(values) != 2:
        raise ValueError("tmax VMI candidate expects two source vectors")
    return _vmax(values[0], values[1], mask)


def _move(values: Sequence[_VectorValue], mask: _MaskValue) -> _VectorValue:
    if len(values) != 1:
        raise ValueError("tmov VMI candidate expects one source vector")
    return values[0]


def _divide_by_scalar(
    value: _VectorValue, scalar: _Value, mask: _MaskValue
) -> _VectorValue:
    scalar_vector = _vbrc_scalar(scalar, like=value)
    return _vdiv(value, scalar_vector, mask)


def _divide_scalar_by_vector(
    scalar: _Value, value: _VectorValue, mask: _MaskValue
) -> _VectorValue:
    scalar_vector = _vbrc_scalar(scalar, like=value)
    return _vdiv(scalar_vector, value, mask)


def _mask_as(mask: _MaskValue, dtype: ScalarType) -> _MaskValue:
    return _MaskValue(mask.value, dtype)


def _vbrc_constant(
    value: float | int, dtype: ScalarType, like: _VectorValue
) -> _VectorValue:
    if dtype.name.startswith("ui"):
        return _vconstant(value, dtype, like=like)
    return _vbrc_scalar(_scalar_constant(value, dtype), like=like)


def _div_three_candidate_search_f32(
    lhs: _VectorValue, rhs: _VectorValue, mask: _MaskValue
) -> _VectorValue:
    lhs_u32 = _vinterpret_cast(lhs, ui32)
    inf_bound = _vbrc_constant(0x7F800000, ui32, like=lhs_u32)
    sign_bit = _vbrc_constant(0x80000000, ui32, like=lhs_u32)
    zero = _vbrc_constant(0.0, f32, like=lhs)
    one = _vbrc_constant(1.0, f32, like=lhs)
    neg_one = _vbrc_constant(-1.0, f32, like=lhs)

    z = _vdiv(lhs, rhs, mask)
    z_u32 = _vinterpret_cast(z, ui32)
    z_or_sign = _vor(z_u32, sign_bit, _mask_as(mask, ui32))
    is_inf_nan = _vcmp(z_or_sign, inf_bound, _mask_as(mask, ui32), "ge")
    is_zero = _vcmp(z, zero, mask, "eq")
    special_mask = _por(is_inf_nan, is_zero)

    y = _vmul(rhs, neg_one, mask)
    residual = _vmula(lhs, z, y, mask)
    z_pre = _vadd(z, neg_one, mask)
    z_next = _vadd(z, one, mask)
    residual_pre = _vmula(lhs, z_pre, y, mask)
    residual_next = _vmula(lhs, z_next, y, mask)

    residual_abs = _vabs(residual, mask)
    residual_pre_abs = _vabs(residual_pre, mask)
    residual_next_abs = _vabs(residual_next, mask)
    better_pre = _vcmp(residual_pre_abs, residual_abs, mask, "lt")
    z_best = _vsel(z_pre, z, better_pre)
    residual_best_abs = _vsel(residual_pre_abs, residual_abs, better_pre)
    better_next = _vcmp(residual_next_abs, residual_best_abs, mask, "lt")
    z_best = _vsel(z_next, z_best, better_next)
    return _vsel(z, z_best, special_mask)


def _div_ieee754_f32_vmi(
    src0: _VectorValue, src1: _VectorValue, mask: _MaskValue
) -> _VectorValue:
    int_mask = _mask_as(mask, ui32)
    src0_u32 = _vinterpret_cast(src0, ui32)
    f32_inf = _vbrc_constant(0x7F800000, ui32, like=src0_u32)
    sign_extractor = _vbrc_constant(0x80000000, ui32, like=src0_u32)
    exponent_extractor = _vbrc_constant(0x807FFFFF, ui32, like=src0_u32)
    exponent_normalizer = _vbrc_constant(0x3F800000, ui32, like=src0_u32)
    subnormal_threshold = _vbrc_constant(0x007FFFFF, ui32, like=src0_u32)
    nan_value = _vbrc_constant(0x7FC00000, ui32, like=src0_u32)
    min_denormal = _vbrc_constant(0x1, ui32, like=src0_u32)
    zero_u32 = _vbrc_constant(0, ui32, like=src0_u32)
    normalize_scale_enlarge = _vbrc_constant(8388608.0, f32, like=src0)
    normalize_scale_reduce = _vbrc_constant(1.1920928955078125e-07, f32, like=src0)

    src0_abs = _vabs(src0, mask)
    src1_abs = _vabs(src1, mask)
    src0_abs_u32 = _vinterpret_cast(src0_abs, ui32)
    src1_abs_u32 = _vinterpret_cast(src1_abs, ui32)

    mask_inf_src0 = _vcmp(src0_abs_u32, f32_inf, int_mask, "eq")
    mask_inf_src1 = _vcmp(src1_abs_u32, f32_inf, int_mask, "eq")
    mask_invalid = _por(mask_inf_src0, mask_inf_src1)
    mask_zero_src0 = _vcmp(src0_abs_u32, zero_u32, int_mask, "eq")
    mask_invalid = _por(mask_invalid, mask_zero_src0)
    mask_zero_src1 = _vcmp(src1_abs_u32, zero_u32, int_mask, "eq")
    mask_invalid = _por(mask_invalid, mask_zero_src1)
    mask_valid = _pnot(mask_invalid)

    mask_src0_subnormal = _vcmp(src0_abs_u32, subnormal_threshold, int_mask, "eq")
    mask_src0_normal = _pnot(mask_src0_subnormal)
    src0_subnormal = _vmul(
        src0, normalize_scale_enlarge, _mask_as(mask_src0_subnormal, f32)
    )
    mask_src1_subnormal = _vcmp(src1_abs_u32, subnormal_threshold, int_mask, "lt")
    mask_src1_normal = _pnot(mask_src1_subnormal)
    src1_subnormal = _vmul(
        src1, normalize_scale_enlarge, _mask_as(mask_src1_subnormal, f32)
    )

    src0_all = _vsel(src0, src0_subnormal, _mask_as(mask_src0_normal, f32))
    src1_all = _vsel(src1, src1_subnormal, _mask_as(mask_src1_normal, f32))
    src0_all_u32 = _vinterpret_cast(src0_all, ui32)
    src1_all_u32 = _vinterpret_cast(src1_all, ui32)

    src0_norm_u32 = _vand(src0_all_u32, exponent_extractor, mask_valid)
    src1_norm_u32 = _vand(src1_all_u32, exponent_extractor, mask_valid)
    src0_norm_u32 = _vadd(src0_norm_u32, exponent_normalizer, mask_valid)
    src1_norm_u32 = _vadd(src1_norm_u32, exponent_normalizer, mask_valid)
    src0_norm = _vsel(
        _vinterpret_cast(src0_norm_u32, f32), src0_all, _mask_as(mask_valid, f32)
    )
    src1_norm = _vsel(
        _vinterpret_cast(src1_norm_u32, f32), src1_all, _mask_as(mask_valid, f32)
    )

    divided = _div_three_candidate_search_f32(
        src0_norm, src1_norm, _mask_as(mask_valid, f32)
    )
    mask0 = _pand(mask_src0_subnormal, mask_src1_normal)
    divided = _vsel(
        _vmul(divided, normalize_scale_reduce, _mask_as(mask0, f32)),
        divided,
        _mask_as(mask0, f32),
    )
    mask0 = _pand(mask_src0_normal, mask_src1_subnormal)
    divided = _vsel(
        _vmul(divided, normalize_scale_enlarge, _mask_as(mask0, f32)),
        divided,
        _mask_as(mask0, f32),
    )

    divided_u32 = _vinterpret_cast(divided, ui32)
    divided_sign = _vand(divided_u32, sign_extractor, int_mask)
    src0_exponent = _vand(src0_all_u32, f32_inf, int_mask)
    src1_exponent = _vand(src1_all_u32, f32_inf, int_mask)
    shift23 = _vbrc_constant(23, ui32, like=src0_exponent)
    src0_exp_shifted = _vshr(src0_exponent, shift23, int_mask)
    src1_exp_shifted = _vshr(src1_exponent, shift23, int_mask)

    scale = _vinterpret_cast(
        _vsub(src0_exp_shifted, src1_exp_shifted, int_mask), i32
    )
    scale_mask = _mask_as(mask, i32)
    scale = _vadds(scale, _scalar_constant(127, i32), scale_mask)

    neg23 = _vbrc_constant(-23, i32, like=scale)
    mask_underflow1 = _vcmp(scale, neg23, scale_mask, "eq")
    mask_underflow1 = _pand(mask_underflow1, mask_valid)
    z1_u32 = _vadd(divided_sign, min_denormal, mask_underflow1)
    z2_u32 = _vadd(divided_sign, zero_u32, mask_underflow1)

    src0_norm_abs = _vabs(src0_norm, _mask_as(mask_valid, f32))
    src1_norm_abs = _vabs(src1_norm, _mask_as(mask_valid, f32))
    mask_norm = _vcmp(src0_norm_abs, src1_norm_abs, _mask_as(mask_valid, f32), "le")
    divided_u32_temp = _vsel(
        _vsel(z2_u32, z1_u32, mask_norm), divided_u32, mask_underflow1
    )

    mask_valid_temp = _pand(_pnot(mask_underflow1), mask_valid)
    mask_underflow2 = _vcmp(scale, neg23, scale_mask, "lt")
    mask_underflow2 = _pand(mask_underflow2, mask_valid_temp)
    divided_u32_temp = _vsel(
        _vadd(divided_sign, zero_u32, mask_underflow2),
        divided_u32_temp,
        mask_underflow2,
    )

    mask_valid_temp = _pand(_pnot(mask_underflow2), mask_valid_temp)
    max_exp = _vbrc_constant(255, i32, like=scale)
    mask_overflow1 = _vcmp(scale, max_exp, scale_mask, "eq")
    mask_overflow1 = _pand(mask_overflow1, mask_valid_temp)
    scale = _vsel(
        _vadds(scale, _scalar_constant(-1, i32), mask_overflow1),
        scale,
        mask_overflow1,
    )

    divided_f32_temp = _vinterpret_cast(divided_u32_temp, f32)
    divided_f32_temp = _vsel(
        _vmul(
            divided_f32_temp,
            _vbrc_constant(2.0, f32, like=src0),
            _mask_as(mask_overflow1, f32),
        ),
        divided_f32_temp,
        _mask_as(mask_overflow1, f32),
    )

    mask_overflow2 = _vcmp(scale, max_exp, scale_mask, "gt")
    mask_overflow2 = _pand(mask_overflow2, mask_valid_temp)
    divided_u32_temp = _vsel(
        _vadd(divided_sign, f32_inf, mask_overflow2),
        _vinterpret_cast(divided_f32_temp, ui32),
        mask_overflow2,
    )

    mask_valid_final = _pand(_pnot(mask_overflow2), mask_valid_temp)
    zero_exp = _vbrc_constant(0, i32, like=scale)
    mask_pos_exp = _vcmp(scale, zero_exp, _mask_as(mask_valid_final, i32), "gt")
    scale_u32 = _vinterpret_cast(scale, ui32)
    exp_shifted = _vshl(scale_u32, shift23, _mask_as(mask_pos_exp, ui32))
    exp_factor_f32 = _vinterpret_cast(exp_shifted, f32)
    divided_f32_temp = _vinterpret_cast(divided_u32_temp, f32)
    divided_f32_temp = _vsel(
        _vmul(divided_f32_temp, exp_factor_f32, _mask_as(mask_pos_exp, f32)),
        divided_f32_temp,
        _mask_as(mask_pos_exp, f32),
    )

    mask_pos_exp_not = _pnot(mask_pos_exp)
    scale_abs = _vabs(scale, mask_pos_exp_not)
    shr_factor_u32 = _vshr(
        _vbrc_constant(4194304, ui32, like=scale_u32),
        _vinterpret_cast(scale_abs, ui32),
        _mask_as(mask_pos_exp_not, ui32),
    )
    divided_f32_temp = _vsel(
        _vmul(
            divided_f32_temp,
            _vinterpret_cast(shr_factor_u32, f32),
            _mask_as(mask_pos_exp_not, f32),
        ),
        divided_f32_temp,
        _mask_as(mask_pos_exp_not, f32),
    )

    mask_nan = _por(
        _vcmp(src0_abs, src0_abs, mask, "ne"),
        _vcmp(src1_abs, src1_abs, mask, "ne"),
    )
    return _vsel(
        _vinterpret_cast(nan_value, f32), divided_f32_temp, mask_nan
    )


def _div_ieee754_f16_vmi(
    src0: _VectorValue, src1: _VectorValue, mask: _MaskValue
) -> _VectorValue:
    int_mask = _mask_as(mask, ui16)
    src0_u16 = _vinterpret_cast(src0, ui16)
    f16_inf = _vbrc_constant(0x7C00, ui16, like=src0_u16)
    exponent_extractor = _vbrc_constant(0x83FF, ui16, like=src0_u16)
    exponent_normalizer = _vbrc_constant(0x3C00, ui16, like=src0_u16)
    sign_extractor = _vbrc_constant(0x8000, ui16, like=src0_u16)
    subnormal_threshold = _vbrc_constant(0x03FF, ui16, like=src0_u16)
    nan_value = _vbrc_constant(0x7E00, ui16, like=src0_u16)
    min_denormal = _vbrc_constant(0x1, ui16, like=src0_u16)
    zero_u16 = _vbrc_constant(0, ui16, like=src0_u16)
    normalize_scale_enlarge = _vbrc_constant(1024.0, f16, like=src0)
    normalize_scale_reduce = _vbrc_constant(0.0009765625, f16, like=src0)

    src0_abs = _vabs(src0, mask)
    src1_abs = _vabs(src1, mask)
    src0_abs_u16 = _vinterpret_cast(src0_abs, ui16)
    src1_abs_u16 = _vinterpret_cast(src1_abs, ui16)

    mask_inf_src0 = _vcmp(src0_abs_u16, f16_inf, int_mask, "eq")
    mask_inf_src1 = _vcmp(src1_abs_u16, f16_inf, int_mask, "eq")
    mask_invalid = _por(mask_inf_src0, mask_inf_src1)
    mask_zero_src0 = _vcmp(src0_abs_u16, zero_u16, int_mask, "eq")
    mask_invalid = _por(mask_invalid, mask_zero_src0)
    mask_zero_src1 = _vcmp(src1_abs_u16, zero_u16, int_mask, "eq")
    mask_invalid = _por(mask_invalid, mask_zero_src1)
    mask_valid = _pnot(mask_invalid)

    mask_src0_subnormal = _vcmp(src0_abs_u16, subnormal_threshold, int_mask, "lt")
    mask_src0_normal = _pnot(mask_src0_subnormal)
    src0_subnormal = _vmul(
        src0, normalize_scale_enlarge, _mask_as(mask_src0_subnormal, f16)
    )
    mask_src1_subnormal = _vcmp(src1_abs_u16, subnormal_threshold, int_mask, "lt")
    mask_src1_normal = _pnot(mask_src1_subnormal)
    src1_subnormal = _vmul(
        src1, normalize_scale_enlarge, _mask_as(mask_src1_subnormal, f16)
    )

    src0_all = _vsel(src0, src0_subnormal, _mask_as(mask_src0_normal, f16))
    src1_all = _vsel(src1, src1_subnormal, _mask_as(mask_src1_normal, f16))
    src0_all_u16 = _vinterpret_cast(src0_all, ui16)
    src1_all_u16 = _vinterpret_cast(src1_all, ui16)

    src0_norm_u16 = _vand(src0_all_u16, exponent_extractor, mask_valid)
    src1_norm_u16 = _vand(src1_all_u16, exponent_extractor, mask_valid)
    src0_norm_u16 = _vadd(src0_norm_u16, exponent_normalizer, mask_valid)
    src1_norm_u16 = _vadd(src1_norm_u16, exponent_normalizer, mask_valid)
    src0_norm = _vsel(
        _vinterpret_cast(src0_norm_u16, f16), src0_all, _mask_as(mask_valid, f16)
    )
    src1_norm = _vsel(
        _vinterpret_cast(src1_norm_u16, f16), src1_all, _mask_as(mask_valid, f16)
    )

    src0_norm_abs = _vabs(src0_norm, _mask_as(mask_valid, f16))
    src1_norm_abs = _vabs(src1_norm, _mask_as(mask_valid, f16))
    mask_norm = _vcmp(src0_norm_abs, src1_norm_abs, _mask_as(mask_valid, f16), "le")
    divided = _vdiv(src0_norm, src1_norm, _mask_as(mask_valid, f16))

    mask0 = _pand(mask_src0_subnormal, mask_src1_normal)
    divided = _vsel(
        _vmul(divided, normalize_scale_reduce, _mask_as(mask0, f16)),
        divided,
        _mask_as(mask0, f16),
    )
    mask0 = _pand(mask_src0_normal, mask_src1_subnormal)
    divided = _vsel(
        _vmul(divided, normalize_scale_enlarge, _mask_as(mask0, f16)),
        divided,
        _mask_as(mask0, f16),
    )

    divided_u16 = _vinterpret_cast(divided, ui16)
    divided_sign = _vand(divided_u16, sign_extractor, int_mask)
    src0_exponent = _vand(src0_all_u16, f16_inf, int_mask)
    src1_exponent = _vand(src1_all_u16, f16_inf, int_mask)
    shift10 = _vbrc_constant(10, ui16, like=src0_exponent)
    src0_exp_shifted = _vshr(src0_exponent, shift10, int_mask)
    src1_exp_shifted = _vshr(src1_exponent, shift10, int_mask)

    scale = _vinterpret_cast(
        _vsub(src0_exp_shifted, src1_exp_shifted, int_mask), i16
    )
    scale_mask = _mask_as(mask, i16)
    scale = _vadds(scale, _scalar_constant(15, i16), scale_mask)

    neg9 = _vbrc_constant(-9, i16, like=scale)
    mask_underflow1 = _vcmp(scale, neg9, scale_mask, "eq")
    mask_underflow1 = _pand(mask_underflow1, mask_valid)
    z1_u16 = _vadd(divided_sign, min_denormal, mask_underflow1)
    z2_u16 = _vadd(divided_sign, zero_u16, mask_underflow1)
    divided_u16_temp = _vsel(
        _vsel(z2_u16, z1_u16, mask_norm), divided_u16, mask_underflow1
    )

    mask_valid_temp = _pand(_pnot(mask_underflow1), mask_valid)
    mask_underflow2 = _vcmp(scale, neg9, scale_mask, "lt")
    mask_underflow2 = _pand(mask_underflow2, mask_valid_temp)
    divided_u16_temp = _vsel(
        _vadd(divided_sign, zero_u16, mask_underflow2),
        divided_u16_temp,
        mask_underflow2,
    )

    mask_valid_temp = _pand(_pnot(mask_underflow2), mask_valid_temp)
    max_exp = _vbrc_constant(31, i16, like=scale)
    mask_overflow1 = _vcmp(scale, max_exp, scale_mask, "eq")
    mask_overflow1 = _pand(mask_overflow1, mask_valid_temp)
    scale = _vsel(
        _vadds(scale, _scalar_constant(-1, i16), mask_overflow1),
        scale,
        mask_overflow1,
    )

    divided_f16_temp = _vinterpret_cast(divided_u16_temp, f16)
    divided_f16_temp = _vsel(
        _vmul(
            divided_f16_temp,
            _vbrc_constant(2.0, f16, like=src0),
            _mask_as(mask_overflow1, f16),
        ),
        divided_f16_temp,
        _mask_as(mask_overflow1, f16),
    )

    mask_overflow2 = _vcmp(scale, max_exp, scale_mask, "gt")
    mask_overflow2 = _pand(mask_overflow2, mask_valid_temp)
    divided_u16_temp = _vsel(
        _vadd(divided_sign, f16_inf, mask_overflow2),
        _vinterpret_cast(divided_f16_temp, ui16),
        mask_overflow2,
    )

    mask_valid_final = _pand(_pnot(mask_overflow2), mask_valid_temp)
    zero_exp = _vbrc_constant(0, i16, like=scale)
    mask_pos_exp = _vcmp(scale, zero_exp, _mask_as(mask_valid_final, i16), "gt")
    scale_u16 = _vinterpret_cast(scale, ui16)
    exp_factor_f16 = _vinterpret_cast(
        _vshl(scale_u16, shift10, _mask_as(mask_pos_exp, ui16)), f16
    )
    divided_f16_temp = _vinterpret_cast(divided_u16_temp, f16)
    divided_f16_temp = _vsel(
        _vmul(divided_f16_temp, exp_factor_f16, _mask_as(mask_pos_exp, f16)),
        divided_f16_temp,
        _mask_as(mask_pos_exp, f16),
    )

    mask_pos_exp_not = _pnot(mask_pos_exp)
    scale_abs = _vabs(scale, mask_pos_exp_not)
    shr_factor_u16 = _vshr(
        _vbrc_constant(512, ui16, like=scale_u16),
        _vinterpret_cast(scale_abs, ui16),
        _mask_as(mask_pos_exp_not, ui16),
    )
    divided_f16_temp = _vsel(
        _vmul(
            divided_f16_temp,
            _vinterpret_cast(shr_factor_u16, f16),
            _mask_as(mask_pos_exp_not, f16),
        ),
        divided_f16_temp,
        _mask_as(mask_pos_exp_not, f16),
    )

    mask_nan = _por(
        _vcmp(src0_abs, src0_abs, mask, "ne"),
        _vcmp(src1_abs, src1_abs, mask, "ne"),
    )
    return _vsel(
        _vinterpret_cast(nan_value, f16), divided_f16_temp, mask_nan
    )


def _div_high_precision(
    lhs: _VectorValue, rhs: _VectorValue, mask: _MaskValue
) -> _VectorValue:
    if lhs.dtype != rhs.dtype:
        raise ValueError("high-precision VMI division requires matching dtypes")
    if lhs.dtype == f32:
        return _div_ieee754_f32_vmi(lhs, rhs, mask)
    if lhs.dtype == f16:
        return _div_ieee754_f16_vmi(lhs, rhs, mask)
    raise ValueError("high-precision VMI division requires f16 or f32")


def _divide_by_scalar_high_precision(
    value: _VectorValue, scalar: _Value, mask: _MaskValue
) -> _VectorValue:
    return _div_high_precision(value, _vbrc_scalar(scalar, like=value), mask)


def _divide_scalar_by_vector_high_precision(
    scalar: _Value, value: _VectorValue, mask: _MaskValue
) -> _VectorValue:
    return _div_high_precision(_vbrc_scalar(scalar, like=value), value, mask)


def _sqrt_high_precision_f16(source: _VectorValue, mask: _MaskValue) -> _VectorValue:
    subnormal_mask = _vcmps(
        source,
        _scalar_constant(6.097555160522461e-05, f16),
        mask,
        "lt",
    )
    scaled_source = _vmuls(source, _scalar_constant(4096.0, f16), subnormal_mask)
    source_adjusted = _vsel(scaled_source, source, subnormal_mask)
    root = _vsqrt(source_adjusted, mask)
    scaled_root = _vmuls(root, _scalar_constant(0.015625, f16), subnormal_mask)
    return _vsel(scaled_root, root, subnormal_mask)


def _sqrt_high_precision_f32(source: _VectorValue, mask: _MaskValue) -> _VectorValue:
    subnormal_mask = _vcmps(source, _scalar_constant(1.0, f32), mask, "lt")
    scaled_source = _vmuls(
        source, _scalar_constant(16777216.0, f32), subnormal_mask
    )
    source_adjusted = _vsel(scaled_source, source, subnormal_mask)

    one = _vbrc_scalar(_scalar_constant(1.0, f32), like=source)
    root = _vsqrt(source_adjusted, mask)
    reciprocal = _vdiv(one, root, mask)
    neg_reciprocal = _vmuls(reciprocal, _scalar_constant(-1.0, f32), mask)
    err = _vmul(reciprocal, source_adjusted, mask)
    one_adjusted = _vmula(one, err, neg_reciprocal, mask)
    half_reciprocal = _vmuls(reciprocal, _scalar_constant(0.5, f32), mask)
    refined = _vmula(reciprocal, one_adjusted, half_reciprocal, mask)

    result = _vmul(refined, source_adjusted, mask)
    neg_result = _vmuls(result, _scalar_constant(-1.0, f32), mask)
    err = _vmula(source_adjusted, result, neg_result, mask)
    half_refined = _vmuls(refined, _scalar_constant(0.5, f32), mask)
    correction = _vmul(err, half_refined, mask)
    corrected = _vadd(correction, result, mask)

    scaled_corrected = _vmuls(
        corrected, _scalar_constant(0.000244140625, f32), mask
    )
    result = _vsel(scaled_corrected, corrected, subnormal_mask)

    source_bits = _vinterpret_cast(source_adjusted, ui32)
    is_inf = _vcmp(
        source_bits,
        _vbrc_constant(0x7F800000, ui32, like=source_bits),
        _mask_as(mask, ui32),
        "eq",
    )
    sign_bit = _vbrc_constant(0x80000000, ui32, like=source_bits)
    source_with_sign = _vor(source_bits, sign_bit, _mask_as(mask, ui32))
    is_zero = _vcmp(
        source_with_sign,
        _vbrc_constant(0x80000000, ui32, like=source_bits),
        _mask_as(mask, ui32),
        "eq",
    )
    return _vsel(source_adjusted, result, _por(is_zero, is_inf))


def _sqrt_high_precision(
    values: Sequence[_VectorValue], mask: _MaskValue
) -> _VectorValue:
    if len(values) != 1:
        raise ValueError("tsqrt high-precision VMI candidate expects one source vector")
    source = values[0]
    if source.dtype == f16:
        return _sqrt_high_precision_f16(source, mask)
    if source.dtype == f32:
        return _sqrt_high_precision_f32(source, mask)
    raise ValueError("tsqrt high-precision VMI candidate requires f16 or f32")


def _context_attr(tile: _TileProxy, name: str, default=None):
    return getattr(tile._trace, "context_attrs", {}).get(name, default)


def _operand_kinds_are(expected: tuple[str, ...]):
    def predicate(operand_kinds=(), **_):
        return tuple(operand_kinds) == expected

    return predicate


def emit_sqrt_vmi(src: _TileProxy, dst: _TileProxy) -> None:
    emit_elementwise_vmi(
        dst,
        (src,),
        lambda values, mask: _vsqrt(values[0], mask),
        allowed_dtypes=FLOAT_DTYPES,
    )


def emit_sqrt_high_precision_vmi(src: _TileProxy, dst: _TileProxy) -> None:
    emit_elementwise_vmi(
        dst,
        (src,),
        _sqrt_high_precision,
        allowed_dtypes=FLOAT_DTYPES,
    )


def emit_recip_vmi(src: _TileProxy, dst: _TileProxy, *, high_precision: bool) -> None:
    def reciprocal(values, mask):
        one = _vbrc_scalar(
            _scalar_constant(1.0, values[0].dtype), like=values[0]
        )
        if high_precision:
            return _div_high_precision(one, values[0], mask)
        return _vdiv(one, values[0], mask)

    emit_elementwise_vmi(dst, (src,), reciprocal, allowed_dtypes=FLOAT_DTYPES)


def emit_rsqrt_vmi(
    src: _TileProxy,
    dst: _TileProxy,
    *,
    high_precision: bool,
) -> None:
    def reciprocal_sqrt(values, mask):
        root = (
            _sqrt_high_precision(values, mask)
            if high_precision
            else _vsqrt(values[0], mask)
        )
        one = _vbrc_scalar(
            _scalar_constant(1.0, values[0].dtype), like=values[0]
        )
        if high_precision:
            return _div_high_precision(one, root, mask)
        return _vdiv(one, root, mask)

    emit_elementwise_vmi(dst, (src,), reciprocal_sqrt, allowed_dtypes=FLOAT_DTYPES)


# Row-reduce is ODS-validated only for {f32, i32} on A5 (trowmax/trowsum
# reject i8/i16/f16/bf16/ui*); other dtypes fall back to the ordinary PTODSL
# path. This set is checked against the TileOp ODS, not just the VMI candidate.
_ROW_REDUCE_DTYPES = (f32, i32)


def _validate_row_reduce_tiles(
    src: _TileProxy, workspace: _TileProxy, dst: _TileProxy
) -> tuple[int, int, int]:
    if (
        src.element_type != workspace.element_type
        or src.element_type != dst.element_type
    ):
        raise ValueError("row-reduce VMI candidate requires matching src/workspace/dst dtype")
    if src.element_type not in _ROW_REDUCE_DTYPES:
        raise ValueError(
            f"row-reduce VMI candidate dtype {src.element_type} not supported; "
            f"expected one of {tuple(d.name for d in _ROW_REDUCE_DTYPES)} "
            "(A5 row-reduce ODS accepts only f32/i32)"
        )
    if src._spec.b_layout != "row_major":
        raise ValueError("row-reduce source must be row-major")
    rows, physical_cols = src._spec.shape
    valid_rows, valid_cols = src._spec.effective_valid_shape
    if valid_rows != rows or valid_cols <= 0 or valid_cols > physical_cols:
        raise ValueError("row-reduce valid shape must fit the physical source tile")
    src_lanes = src.element_type.lanes
    safe_read_cols = ((valid_cols + src_lanes - 1) // src_lanes) * src_lanes
    sinkhorn_grouped_form = (
        src._spec.shape == (8, 8)
        and src._spec.effective_valid_shape == (8, 4)
    )
    if (
        src._spec.effective_valid_shape != src._spec.shape
        and safe_read_cols > physical_cols
        and not sinkhorn_grouped_form
    ):
        raise ValueError(
            "row-reduce source must contain every physical lane read by its mask"
        )
    workspace_rows, workspace_cols = workspace._spec.shape
    if workspace_rows != rows or workspace_cols < 1:
        raise ValueError("row-reduce workspace must have matching rows")
    if workspace._spec.effective_valid_shape != workspace._spec.shape:
        raise ValueError("row-reduce VMI candidates require a full workspace tile")
    if dst._spec.shape != (rows, 1) or dst._spec.b_layout != "col_major":
        raise ValueError("row-reduce destination must be a col-major [rows, 1] tile")
    if dst._spec.effective_valid_shape != (rows, 1):
        raise ValueError("row-reduce destination valid shape must be [rows, 1]")
    return rows, physical_cols, valid_cols


def emit_row_reduce_vmi(
    src: _TileProxy,
    workspace: _TileProxy,
    dst: _TileProxy,
    *,
    kind: str,
) -> None:
    rows, physical_cols, valid_cols = _validate_row_reduce_tiles(src, workspace, dst)
    sinkhorn_grouped_form = (
        src._spec.shape == (8, 8)
        and src._spec.effective_valid_shape == (8, 4)
    )
    if valid_cols != physical_cols and not sinkhorn_grouped_form:
        raise ValueError(
            "grouped row-reduce requires a full static source tile or the "
            "registered Sinkhorn 8x8/8x4 form"
        )
    _prepare_tile_access(src, workspace, dst)
    total_lanes = rows * physical_cols
    active = src._trace.index_const(valid_cols)
    src_dtype = src.element_type
    full_mask = _wrap_mask(
        _vmi_builder.create_mask(
            active.value, size=total_lanes, group=rows
        ),
        src_dtype,
    )
    source = _vload_linear(src, 0, lanes=total_lanes)
    if kind == "max":
        reduced_value = _vmi_builder.vcmax(
            source.value, full_mask.value, group=rows
        )
    else:
        reduced_value = _vmi_builder.vcadd(
            source.value, full_mask.value, group=rows, reassoc=True
        )
    reduced = _wrap_vreg(reduced_value, src_dtype)

    offset = dst._trace._coerce_index(0)
    if physical_cols < src_dtype.lanes:
        # The compact reduction result intentionally has group_slots layout:
        # one value per source row. This is the memory form group_store models.
        dst_ptr = dst._trace.ensure_tile_ptr(dst)
        row_stride = dst._trace._coerce_index(1)
        _vmi_builder.vstore(
            reduced.value,
            dst_ptr.value,
            offset.value,
            stride=row_stride.value,
            group=rows,
        )
        return

    # The grouped reduction produces one logical value per row. Scatter these
    # compact values from an aligned UB base instead of reinterpreting them as
    # a grouped strided memory store.
    dst_ptr = dst._trace.ensure_tile_ptr(dst)
    compact_mask = _create_mask_lanes(rows, rows, src_dtype, trace=dst._trace)
    zero_i32 = dst._trace.scalar_const(0, i32)
    offsets = _wrap_vreg(
        _vmi_builder.vci(zero_i32.value, size=rows, order="ASC"),
        i32,
    )
    _vmi_builder.vscatter(
        reduced.value, dst_ptr.value, offsets.value, compact_mask.value
    )


def emit_row_reduce_streaming_vmi(
    src: _TileProxy,
    workspace: _TileProxy,
    dst: _TileProxy,
    *,
    kind: str,
) -> None:
    """Reduce one logical row per iteration for compatible VMI fusion."""

    rows, physical_cols, valid_cols = _validate_row_reduce_tiles(src, workspace, dst)
    sinkhorn_row_form = (
        (rows, physical_cols) == (8, 8) and valid_cols in {4, 8}
    )
    if valid_cols != physical_cols and not sinkhorn_row_form:
        raise ValueError("row-streaming reduction requires a full static source tile")
    _prepare_tile_access(src, dst)
    active = src._trace.index_const(valid_cols)
    src_dtype = src.element_type
    row_mask = _wrap_mask(
        _vmi_builder.create_mask(active.value, size=physical_cols), src_dtype
    )
    row_stride = dst._trace._coerce_index(1)
    with for_(0, rows, step=1) as row:
        src_offset = index_mul(row, physical_cols)
        source = _vload_linear(src, src_offset, lanes=physical_cols)
        if kind == "max":
            reduced_value = _vmi_builder.vcmax(source.value, row_mask.value)
        else:
            reduced_value = _vmi_builder.vcadd(
                source.value, row_mask.value, reassoc=True
            )
        reduced = _wrap_vreg(reduced_value, src_dtype)
        dst_ptr = dst._trace.ensure_tile_ptr(dst)
        dst_offset = dst._trace._coerce_index(row)
        _vmi_builder.vstore(
            reduced.value,
            dst_ptr.value,
            dst_offset.value,
            stride=row_stride.value,
            group=1,
        )


def emit_row_expand_binary_vmi(
    row_tensor: _TileProxy,
    compact_row_state: _TileProxy,
    output: _TileProxy,
    operation: str,
) -> None:
    """Apply one compact per-row value to each wide logical row."""

    operations = {
        "sub": _vsub,
        "mul": _vmul,
        "div": _vdiv,
    }
    if operation not in operations:
        raise ValueError(
            f"row-expand VMI candidate does not support {operation!r}; "
            f"expected one of {sorted(operations)}"
        )
    if (
        row_tensor.element_type != f32
        or compact_row_state.element_type != f32
        or output.element_type != f32
    ):
        raise ValueError("row-expand VMI candidates currently support only f32")
    if (
        row_tensor._spec.b_layout != "row_major"
        or output._spec.b_layout != "row_major"
    ):
        raise ValueError("row-expand source and destination must be row-major")
    logical_shape = row_tensor._spec.effective_valid_shape
    if output._spec.effective_valid_shape != logical_shape:
        raise ValueError(
            "row-expand source and destination logical shapes must match"
        )
    sinkhorn_grouped_form = (
        row_tensor._spec.shape == (8, 8)
        and logical_shape in {(8, 4), (8, 8)}
        and output._spec.shape == (8, 8)
        and output._spec.effective_valid_shape == logical_shape
    )
    if not sinkhorn_grouped_form and (
        not _is_safe_static_row_prefix(
            row_tensor._spec.shape,
            logical_shape,
            native_lanes=f32.lanes,
        )
        or not _is_safe_static_row_prefix(
            output._spec.shape,
            output._spec.effective_valid_shape,
            native_lanes=f32.lanes,
        )
    ):
        raise ValueError("row-expand logical row exceeds its physical storage")
    rows, cols = logical_shape
    if (
        compact_row_state._spec.shape != (rows, 1)
        or compact_row_state._spec.effective_valid_shape != (rows, 1)
        or compact_row_state._spec.b_layout != "col_major"
    ):
        raise ValueError(
            "row-expand compact state must be a col-major [rows, 1] tile"
        )
    src_physical_cols = row_tensor._spec.shape[1]
    dst_physical_cols = output._spec.shape[1]
    dtype = row_tensor.element_type
    io_lanes = ((cols + dtype.lanes - 1) // dtype.lanes) * dtype.lanes

    _prepare_tile_access(row_tensor, compact_row_state, output)
    if sinkhorn_grouped_form:
        total_lanes = rows * src_physical_cols
        zero = row_tensor._trace.index_const(0)
        one = row_tensor._trace.index_const(1)
        active = row_tensor._trace.index_const(cols)
        mask = _wrap_mask(
            _vmi_builder.create_mask(
                active.value, size=total_lanes, group=rows
            ),
            f32,
        )
        state_ptr = compact_row_state._trace.ensure_tile_ptr(compact_row_state)
        with for_(0, 1, step=1):
            # Load one compact scalar per row into group slots, then broadcast
            # each slot to that row's physical lanes. This keeps the 8x8 tile
            # in one aligned 64-lane domain instead of issuing 32-byte row
            # loads that a following grouped TileOp cannot consume safely.
            slots = _wrap_vreg(
                _vmi_builder.vload(
                    state_ptr.value,
                    zero.value,
                    size=rows,
                    stride=one.value,
                    group=rows,
                ),
                f32,
            )
            broadcast = _wrap_vreg(
                _vmi_builder.vbrc(
                    slots.value, size=total_lanes, group=rows
                ),
                f32,
            )
            value = _vload_linear(row_tensor, zero, lanes=total_lanes)
            result = operations[operation](value, broadcast, mask)
            _vstore_linear(result, output, zero, mask)
        return

    full_mask = _create_mask_lanes(cols, io_lanes, dtype, trace=row_tensor._trace)
    state_ptr = compact_row_state._trace.ensure_tile_ptr(compact_row_state)
    with for_(0, rows, step=1) as row:
        # Match PTO-ISA TRowExpandBinOps: load compact state[row] with the
        # scalar-broadcast distribution, then consume it in the same row loop.
        broadcast = _wrap_vreg(
            _vmi_builder.vload(
                state_ptr.value,
                row.value,
                size=_snap_lanes(io_lanes),
                dist_mode="brc",
            ),
            dtype,
        )
        src_offset = index_mul(row, src_physical_cols)
        dst_offset = index_mul(row, dst_physical_cols)
        value = _vload_linear(row_tensor, src_offset, lanes=io_lanes)
        result = operations[operation](value, broadcast, full_mask)
        _vstore_linear(result, output, dst_offset, full_mask)


def emit_row_expand_sub_vmi(
    src: _TileProxy, row_values: _TileProxy, dst: _TileProxy
) -> None:
    emit_row_expand_binary_vmi(src, row_values, dst, "sub")


def emit_col_expand_vmi(src: _TileProxy, dst: _TileProxy) -> None:
    """Broadcast the single logical source row to every destination row."""

    if src.element_type != f32 or dst.element_type != f32:
        raise ValueError("tcolexpand VMI candidate currently supports only f32")
    if src._spec.b_layout != "row_major" or dst._spec.b_layout != "row_major":
        raise ValueError("tcolexpand source and destination must be row-major")
    rows, cols = dst._spec.shape
    if src._spec.shape != (1, cols):
        raise ValueError("tcolexpand source must be a row-major [1, cols] tile")
    _, valid_cols = dst._spec.effective_valid_shape
    if src._spec.effective_valid_shape != (1, valid_cols):
        raise ValueError(
            "tcolexpand source and destination valid columns must match"
        )
    if (
        src._spec.shape == (1, 8)
        and dst._spec.shape == (8, 8)
        and dst._spec.effective_valid_shape in {(8, 4), (8, 8)}
    ):
        _prepare_tile_access(src, dst)
        mask = _create_mask_lanes(valid_cols, cols, f32, trace=dst._trace)
        broadcast = _vload_linear(src, 0, lanes=cols)
        with for_(0, rows, step=1) as row:
            dst_offset = index_mul(row, cols)
            _vstore_linear(broadcast, dst, dst_offset, mask)
        return

    block_map = CanonicalBlockMap.from_tile(dst, logical_lanes=cols)

    _prepare_tile_access(src, dst)
    full_mask = _create_mask(block_map, dst.element_type, trace=dst._trace)
    broadcast = _vload_linear(src, 0, lanes=cols)
    with for_(0, rows, step=1) as row:
        dst_offset = index_mul(row, cols)
        _vstore_linear(broadcast, dst, dst_offset, full_mask)


def _reduce_identity(kind: str, dtype: ScalarType):
    """Return the reduce-neutral identity element for ``kind`` in ``dtype``.

    Mirrors the C++ ``createReduceNeutralInit`` (VMILowerUnifiedToLegacy.cpp
    :148-188) for the Python reduction emitters. Integer reductions get
    integer neutrals (INT_MIN/INT_MAX/0/1) instead of float ``-inf``/``inf``
    — the float literals would break the int literal materializer. Unsigned
    max uses 0, min uses ``2**bits - 1``; signed max uses ``-2**(bits-1)``,
    min uses ``2**(bits-1) - 1``. Bit width comes from ``mask_bits`` (the
    element bit width on A5), shared between signed and unsigned of the same
    width (the A5 reduction instruction is sign-agnostic at the identity
    stage; the merge op itself dispatches by dtype).
    """
    name = dtype.name
    if name.startswith("i") or name.startswith("ui"):
        unsigned = name.startswith("u")
        bits = dtype.mask_bits
        if kind == "max":
            return 0 if unsigned else -(2 ** (bits - 1))
        if kind == "min":
            return (2 ** bits - 1) if unsigned else (2 ** (bits - 1) - 1)
        if kind == "prod":
            return 1
        # add and any other kind default to 0.
        return 0
    return {
        "max": float("-inf"),
        "min": float("inf"),
        "add": 0.0,
        "prod": 1.0,
    }[kind]


def _validate_col_reduce_tiles(
    src: _TileProxy, dst: _TileProxy
) -> CanonicalBlockMap:
    """Validate tiles for a ColReduce (tcolmax / tcolsum) VMI candidate.

    Mirror of `_validate_row_reduce_tiles` but the surviving axis is the column
    dimension: src is [rows, cols] row-major, dst is [1, cols] row-major, and the
    reduction runs across all rows as a single logical row-width vector.
    """
    if src.element_type != dst.element_type:
        raise ValueError("col-reduce VMI candidate requires matching src/dst dtype")
    if src.element_type not in NUMERIC_DTYPES:
        raise ValueError(
            f"col-reduce VMI candidate dtype {src.element_type} not supported; "
            f"expected one of {tuple(d.name for d in NUMERIC_DTYPES)}"
        )
    if src._spec.b_layout != "row_major" or dst._spec.b_layout != "row_major":
        raise ValueError("col-reduce source and destination must be row-major")
    rows, cols = src._spec.shape
    if dst._spec.shape != (1, cols):
        raise ValueError("col-reduce destination must be a row-major [1, cols] tile")
    return CanonicalBlockMap.from_tile(src, logical_lanes=cols)


def emit_col_reduce_vmi(
    src: _TileProxy,
    dst: _TileProxy,
    *,
    kind: str,
    split: int = 1,
) -> None:
    """Emit a ColReduce (tcolmax / tcolsum / tcolmin / ...) VMI candidate.

    Mirrors pto-isa `TColReduceInstr_NoPostUpdate` over one logical row:
      acc = vbr(InitVal)                        # row-wide, reduce-neutral init
      for row in 0..rows: acc = op(acc, load(row))   # runtime scf.for
      store(acc, dst)

    The accumulator stays row-wide for the whole reduction (the column axis is
    the surviving axis). This intentionally avoids `_vreduce_max`/`vmi_vcmax`,
    which collapse to a 1-lane scalar — wrong for a column-preserving ColMax.

    The init is the reduce's identity element (max->-inf, min->+inf, add->0,
    prod->1), broadcast to the logical row via `vbr` — exactly pto-isa's
    `vbr(dstVReg, InstrOp::InitVal)` (see a5/common.hpp `Padding<T>::Min/Max`).
    The reduce runs from row 0 (not 1): iteration 0 does op(InitVal, load(0))
    which absorbs row 0 through the op (e.g. max(-inf, x) = x, 0 + x = x), so a
    c0..rows header matches the element-wise VMI candidates' c0..N header and
    the downstream loop-fusion pass can merge this reduce with its same-index
    neighbors into one scf.for.

    The cross-row reduction is a runtime ``scf.for`` carrying the row-wide
    accumulator as loop state (one ``vmi.vmax``/``vmi.vadd`` per iteration),
    matching the pto-isa repeat loop. It must NOT be a Python ``range`` here:
    a trace-time ``range`` would statically unroll one merge per row (e.g. 127
    for ``rows=128``), producing a flat vmax chain with no surrounding loop.

    ``split`` (1 or 2) controls the reduction width:

    * ``split=1`` (default): one accumulator, ``scf.for c0..rows step 1``,
      one merge per row. This is the fusion-friendly form: its loop header
      (lb/ub/step) is structurally identical to the element-wise VMI candidates'
      ``c0..N`` header, so ``PTOVmiLoopFusion`` merges them into one
      ``scf.for``.
    * ``split=2``: two accumulators (``acc_a``/``acc_b``) each seeded with the
      reduce identity, ``scf.for c0..rows step 2`` where each iteration loads
      two rows — row ``i`` merges into ``acc_a``, row ``i+1`` into ``acc_b`` —
      and the two partial results are merged once after the loop
      (``merge(acc_a, acc_b)``). This raises ILP (two independent vloads /
      vmerges per iteration, exposing pipeline parallelism) at the cost of
      **breaking fusion**: the ``step 2`` header no longer matches the
      element-wise candidates' ``step 1`` header (see
      ``PTOVmiLoopFusion::sameHeader``), so this reduce runs as a standalone
      loop and is no longer folded into the softmax single-loop body. Use it
      when the reduction itself is the rvec bottleneck and fusion is not
      profitable.

    ``split=2`` requires ``rows % 2 == 0``; otherwise it falls back to
    ``split=1`` (the reduction is still correct, just single-way).
    """
    # Reduce identity element per kind, dtype-aware (ADR-0003 PR3). The C++
    # `createReduceNeutralInit` (VMILowerUnifiedToLegacy.cpp:148-188) already
    # maps these per-bit-width; this Python path mirrors it so an int reduction
    # seeds its accumulator with a correct integer neutral (INT_MIN/INT_MAX/0)
    # rather than `float("-inf")` (which would break the int literal
    # materializer). max -> min representable, min -> max representable,
    # add/prod -> 0/1.
    dst_dtype = src.element_type
    reduce_identity = _reduce_identity(kind, dst_dtype)
    block_map = _validate_col_reduce_tiles(src, dst)
    merge_op = _REDUCE_MERGE_OP[kind]

    _prepare_tile_access(src, dst)
    full_mask = _create_mask(block_map, dst_dtype, trace=src._trace)
    # Seed the row-wide accumulator with the reduce-neutral identity (vbr InitVal,
    # matching pto-isa `TColReduceInstr_NoPostUpdate`), so the loop runs c0..rows
    # and absorbs row 0 via op(InitVal, load(0)) instead of preloading row 0.
    # The broadcast takes the element type/lanes from `dst_dtype` directly — no
    # dummy load needed (a vload would carry a Read memory effect and survive
    # DCE, duplicating the row-0 read the loop itself does).
    accumulator = _vconstant(reduce_identity, dst_dtype, lanes=block_map.cols)

    # Validate split: power-of-two widths 1/2/4/8 are supported, and split>1
    # requires ``rows % split == 0`` so every iteration loads ``split`` real rows
    # (no OOB tail). Any unsupported value / non-divisible row count silently
    # falls back to split=1, which is always correct.
    _SUPPORTED_SPLITS = (1, 2, 4, 8)
    if split not in _SUPPORTED_SPLITS or block_map.rows % split != 0:
        split = 1

    if split >= 2:
        # ``split`` independent row-wide accumulators, each carrying every
        # ``split``-th row. step=split: iteration i loads rows i..i+split-1, one
        # per accumulator (row i+k -> acc_k). The split merges per iteration are
        # mutually independent (acc_k does not depend on acc_j's load for j!=k),
        # exposing load/merge pipeline parallelism the single-way chain cannot.
        # The final cross-accumulator merge is a ``split``-way reduction tree
        # (split-1 extra merge ops) outside the loop. ``step`` no longer equals
        # 1, so this header is not structurally equivalent to the element-wise
        # candidates' ``c0..N step 1`` header and PTOVmiLoopFusion will NOT fold
        # this reduce into the softmax single-loop body (see sameHeader).
        acc_init = accumulator  # acc_0 already seeded above; seed the rest.
        acc_names = [f"acc_{k}" for k in range(split)]
        acc_state = {acc_names[0]: acc_init}
        for k in range(1, split):
            acc_state[acc_names[k]] = _vconstant(
                reduce_identity, dst_dtype, lanes=block_map.cols
            )
        with for_(0, block_map.rows, step=split, state=acc_state) as loop:
            row_base = index_mul(loop.iv, block_map.blocks_per_row)
            next_state = {}
            for k in range(split):
                row_k = index_mul(index_add(loop.iv, k), block_map.blocks_per_row)
                loaded_k = _vload(src, block_map.coordinate(row_k))
                next_state[acc_names[k]] = merge_op(
                    getattr(loop.state, acc_names[k]), loaded_k, full_mask
                )
            loop.yield_state(**next_state)
        # Cross-accumulator merge tree: fold the split partials into one
        # row-wide result. Sequential fold is correct (the merge op is
        # associative & commutative for max/min/add); a balanced tree would
        # expose a bit more ILP but the loop-internal parallelism already
        # dominates the rvec gain.
        accumulator = loop.results[0]
        for k in range(1, split):
            accumulator = merge_op(accumulator, loop.results[k], full_mask)
    else:
        # The whole reduction is a runtime scf.for from row 0 carrying the
        # row-wide accumulator; each iteration does one element-wise merge over
        # the full logical row. Row r maps to logical block r*blocks_per_row.
        with for_(0, block_map.rows, step=1, state={"acc": accumulator}) as loop:
            row_block_base = index_mul(loop.iv, block_map.blocks_per_row)
            loaded = _vload(src, block_map.coordinate(row_block_base))
            merged = merge_op(loop.state.acc, loaded, full_mask)
            loop.yield_state(acc=merged)
        accumulator = loop.results[0]
    # dst [1, cols] is one logical row; store via linear offset to avoid the
    # src/dst shape mismatch in CanonicalBlockCoordinate validation (src is
    # [rows, cols], dst is [1, cols]).
    _vstore_linear(accumulator, dst, 0, full_mask)


def _validate_col_expand_binary_tiles(
    src: _TileProxy, col_values: _TileProxy, dst: _TileProxy
) -> CanonicalBlockMap:
    """Validate tiles for a ColExpandBinary (tcolexpandsub/...) VMI candidate.

    src is [rows, cols] row-major, col_values is [1, cols] row-major (one
    logical row of surviving reduce result), dst is [rows, cols] row-major.
    """
    if (
        src.element_type != f32
        or col_values.element_type != f32
        or dst.element_type != f32
    ):
        raise ValueError("col-expand-binary VMI candidates currently support only f32")
    if src._spec.shape != dst._spec.shape:
        raise ValueError("col-expand-binary source and destination shapes must match")
    if src._spec.b_layout != "row_major" or dst._spec.b_layout != "row_major":
        raise ValueError("col-expand-binary source and destination must be row-major")
    rows, cols = src._spec.shape
    if (
        col_values._spec.shape != (1, cols)
        or col_values._spec.b_layout != "row_major"
    ):
        raise ValueError(
            "col-expand-binary col_values must be a row-major [1, cols] tile"
        )
    return CanonicalBlockMap.from_tile(src, logical_lanes=cols)


def emit_col_expand_binary_vmi(
    src: _TileProxy,
    col_values: _TileProxy,
    dst: _TileProxy,
    *,
    binop: str,
) -> None:
    """Emit a ColExpandBinary (tcolexpandsub/add/mul/div) VMI candidate.

    Mirrors pto-isa `TColExpandBinOp`: the single logical row of col_values is
    broadcast to every row, then a binary op is applied per row block.
    """
    binop_dispatch = {
        "sub": _vsub,
        "add": _vadd,
        "mul": _vmul,
        "div": _vdiv,
    }
    if binop not in binop_dispatch:
        raise ValueError(
            f"col-expand-binary VMI candidate does not support op {binop!r}; "
            f"expected one of {sorted(binop_dispatch)}"
        )
    op_fn = binop_dispatch[binop]
    block_map = _validate_col_expand_binary_tiles(src, col_values, dst)

    _prepare_tile_access(src, col_values, dst)
    _prepare_tile_access(src, col_values, dst)
    full_mask = _create_mask(block_map, src.element_type, trace=src._trace)
    # pto-isa TColExpandBinOp broadcasts by reloading the same col_values row
    # block per row (vlds with fixed offset), NOT a 1-lane vbrc. col_values is
    # [1, cols] (one logical row), so the broadcast load is loop-invariant:
    # hoist it out of the row loop so a later mem2reg (Stage C) can forward the
    # ColMax result directly to the consumer without a per-row reload.
    broadcast = _vload_linear(col_values, 0, lanes=block_map.cols)
    with for_(0, block_map.rows, step=1) as row:
        coordinate = block_map.coordinate(index_mul(row, block_map.blocks_per_row))
        value = _vload(src, coordinate)
        result = op_fn(value, broadcast, full_mask)
        _vstore(result, dst, coordinate, full_mask)


def emit_convert_vmi(src: _TileProxy, dst: _TileProxy) -> None:
    supported_forms = {
        (bf16, f32),
        (f16, f32),
        (i32, f32),
        (f32, bf16),
        (f32, f16),
        (f32, i32),
        (i32, f16),
    }
    if (src.element_type, dst.element_type) not in supported_forms:
        raise ValueError(
            "tcvt VMI candidate does not support "
            f"{src.element_type} -> {dst.element_type}"
        )
    if src._spec.shape != dst._spec.shape:
        raise ValueError("tcvt source and destination shapes must match")
    if src._spec.b_layout != "row_major" or dst._spec.b_layout != "row_major":
        raise ValueError("tcvt VMI candidate requires row-major tiles")
    rows, cols = src._spec.shape
    round_mode = _context_attr(src, "round_mode", "RINT")
    rounding = {
        "RINT": "R",
        "NONE": "R",
        "ROUND": "A",
        "TRUNC": "Z",
    }.get(round_mode)
    if rounding is None:
        raise ValueError(f"tcvt VMI candidate does not support {round_mode} rounding")
    sat_mode = _context_attr(src, "sat_mode", "DEFAULT")
    if sat_mode == "DEFAULT":
        # All narrowing forms currently admitted by this candidate use the
        # A5 TCVT overload default, which is saturation ON. Explicit OFF must
        # remain distinguishable and lower to NOSAT.
        saturate = "SAT"
    else:
        saturate = "SAT" if sat_mode == "ON" else "NOSAT"

    def convert(source: _VectorValue) -> _VectorValue:
        kwargs = {}
        if src.element_type == f32 and dst.element_type in (f16, bf16):
            kwargs["rounding"] = rounding
            kwargs["saturate"] = saturate
        elif src.element_type == f32 and dst.element_type == i32:
            # VMIToVPTO defaults an omitted fp-to-int rounding mode to R.
            # Preserve TileOp TRUNC explicitly as the physical Z mode.
            kwargs["rounding"] = rounding
            kwargs["saturate"] = saturate
        if src.element_type == i32 and dst.element_type == f16:
            # Integer widening has no rounding semantics.  Apply the TileOp
            # rounding mode only to the subsequent f32 -> f16 narrowing.
            widened = _vcvt(source, f32)
            converted = _vcvt(
                widened,
                f16,
                rounding=rounding,
                saturate=saturate,
            )
        else:
            converted = _vcvt(source, dst.element_type, **kwargs)
        return converted

    chunk_lanes = min(src.element_type.lanes, dst.element_type.lanes)
    if _can_use_contiguous_native_chunks(dst, (src,), chunk_lanes=chunk_lanes):
        total_lanes = rows * cols
        _prepare_tile_access(src, dst)
        dst_mask = _create_mask_lanes(
            chunk_lanes, chunk_lanes, dst.element_type, trace=src._trace
        )
        with for_(0, total_lanes, step=chunk_lanes) as offset:
            source = _vload_linear(src, offset, lanes=chunk_lanes)
            converted = convert(source)
            _vstore_linear(converted, dst, offset, dst_mask)
        return

    block_map = CanonicalBlockMap.from_tile(src, logical_lanes=cols)
    _prepare_tile_access(src, dst)
    dst_mask = _create_mask_lanes(cols, cols, dst.element_type, trace=src._trace)
    with for_(0, block_map.logical_block_count, step=1) as logical_block:
        coordinate = block_map.coordinate(logical_block)
        source = _vload(src, coordinate)
        converted = convert(source)
        _vstore(converted, dst, coordinate, dst_mask)


__all__ = [
    "FLOAT_DTYPES",
    "Tile",
    "VMI_TILELIB_REGISTRY",
    "_abs",
    "_add",
    "_context_attr",
    "_divide_by_scalar",
    "_divide_by_scalar_high_precision",
    "_divide_scalar_by_vector",
    "_divide_scalar_by_vector_high_precision",
    "_div_high_precision",
    "_exp",
    "_max",
    "_move",
    "_mul",
    "_neg",
    "_negate_scalar",
    "_operand_kinds_are",
    "_sub",
    "_vadds",
    "_vdiv",
    "_vmaxs",
    "_vmins",
    "_vmuls",
    "canonical_vmi_template",
    "convert_vmi_constraint",
    "emit_elementwise_vmi",
    "emit_scalar_fill_vmi",
    "col_expand_vmi_constraint",
    "col_expand_binary_vmi_constraint",
    "col_reduce_vmi_constraint",
    "emit_col_expand_binary_vmi",
    "emit_col_expand_vmi",
    "emit_col_reduce_vmi",
    "emit_convert_vmi",
    "emit_recip_vmi",
    "emit_row_expand_sub_vmi",
    "emit_row_expand_binary_vmi",
    "row_expand_binary_vmi_constraint",
    "sinkhorn_compact_elementwise_vmi_constraint",
    "sinkhorn_row_expand_vmi_constraint",
    "emit_row_reduce_vmi",
    "emit_row_reduce_streaming_vmi",
    "emit_rsqrt_vmi",
    "emit_sqrt_high_precision_vmi",
    "emit_sqrt_vmi",
    "f32",
    "row_reduce_vmi_constraint",
    "row_reduce_streaming_vmi_constraint",
    "sinkhorn_row_reduce_streaming_vmi_constraint",
]
