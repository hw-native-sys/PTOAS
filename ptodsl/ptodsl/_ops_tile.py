# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Tile ops: views, allocation, and the T-op family."""

from functools import wraps
import warnings
from ._diagnostics import (
    PTODSLDeprecationWarning,
    explicit_mode_required_with_context_error,
    make_tensor_view_invalid_layout_error,
    make_tensor_view_missing_metadata_error,
    tile_row_alignment_error,
)
from ._host_tensors import resolve_tensor_data_entry
from ._scalar_coercion import coerce_scalar_to_type, materialize_scalar_literal
from ._scalar_adaptation import (
    classify_runtime_scalar_type,
    coerce_runtime_i1_value,
    coerce_runtime_index_value,
    coerce_runtime_integer_value,
)
from ._runtime_scalar_ops import emit_runtime_binary_op
from ._surface_values import (
    AllocatedBufferValue,
    MaskResultValue,
    PartitionTensorViewValue,
    TensorViewValue,
    TileSliceValue,
    TileValue,
    _coerce_index_value,
    _static_index_dims,
    _unwrap_sequence,
    compose_partition_spec,
    emit_as_ptr,
    infer_tile_element_type,
    is_runtime_scalar_ir_type,
    parse_tile_type_metadata,
    resolve_address_access,
    unwrap_surface_value,
    wrap_surface_value,
)
from ._types import (
    _is_struct_type,
    _isinstance_pto_type,
    _materialize_integer_literal,
    _normalize_address_space,
    _resolve,
    _strip_integer_signedness,
    mask_type,
    part_tensor_view_type,
    part_tensor_view_type_from_dims,
    ptr,
    tensor_view_type,
    tensor_view_type_from_dims,
    vreg_type,
)
from ptoas.mlir.dialects import arith, pto as _pto
from ptoas.mlir.ir import (
    Attribute,
    BF16Type,
    F16Type,
    F32Type,
    Float8E4M3FNType,
    Float8E5M2Type,
    FloatAttr,
    IndexType,
    IntegerAttr,
    IntegerType,
    MemRefType,
    Operation,
    Type,
    TypeAttr,
    UnitAttr,
    VectorType,
)

from ._ops_common import (
    _UNSET,
    _cmp_mode_attr,
    _coerce_i32,
    _coerce_i64,
    _coerce_index,
    _constant_like,
    _element_bytewidth,
    _normalize_acc_to_vec_mode,
    _normalize_cmp_mode,
    _normalize_enum_attr,
    _require_explicit_mode,
    _require_target_arch,
    _resolve_precision_argument,
    _tile_mask_pattern_attr,
)

from ._ops_core import (
    const,
)


def _coerce_tensor_view_layout_attr(layout):
    if layout is None:
        return None
    if isinstance(layout, str):
        canonical = layout.upper()
        if canonical not in {"ND", "DN", "NZ"}:
            raise make_tensor_view_invalid_layout_error(layout)
        return _pto.LayoutAttr.get(getattr(_pto.Layout, canonical))
    if isinstance(layout, Attribute):
        return layout
    try:
        return _pto.LayoutAttr.get(layout)
    except Exception as exc:  # pragma: no cover - defensive pybind fallback
        raise make_tensor_view_invalid_layout_error(layout) from exc


def make_tensor_view(ptr, *, shape=None, strides=None, layout=None):
    """
    ``pto.make_tensor_view`` – wrap a pointer as a tensor view.

    Type is inferred: rank from ``len(shape)``, element type from ``ptr``.
    """
    if shape is None or strides is None:
        raise make_tensor_view_missing_metadata_error(ptr)
    ptr = resolve_tensor_data_entry(ptr)
    rank = len(shape)
    raw_ptr = unwrap_surface_value(ptr)
    elem = _pto.PtrType(raw_ptr.type).element_type
    normalized_shape = [
        _coerce_index(dim, context="make_tensor_view(shape=...)")
        for dim in shape
    ]
    normalized_strides = [
        _coerce_index(dim, context="make_tensor_view(strides=...)")
        for dim in strides
    ]
    static_dims = _static_index_dims(normalized_shape)
    tv_type = (
        tensor_view_type_from_dims(static_dims, elem)
        if static_dims is not None
        else tensor_view_type(rank, elem)
    )
    layout_attr = _coerce_tensor_view_layout_attr(layout)
    value = _pto.MakeTensorViewOp(
        tv_type,
        raw_ptr,
        _unwrap_sequence(normalized_shape),
        _unwrap_sequence(normalized_strides),
        layout=layout_attr,
    ).result
    return TensorViewValue(value, shape=tuple(shape), strides=tuple(strides))


def _normalize_static_tile_shape(shape):
    static_shape = []
    for dim in shape:
        if isinstance(dim, bool) or not isinstance(dim, int):
            raise TypeError(
                "alloc_tile(shape=...) currently requires a static physical tile shape. "
                "Use constexpr/static integers for shape and place runtime metadata in valid_shape."
            )
        static_shape.append(dim)
    return tuple(static_shape)


def _authored_tile_physical_shape(shape):
    if len(shape) == 1:
        return (1, shape[0])
    return tuple(shape)


def _split_valid_shape(shape, valid_shape):
    logical_rank = len(shape)
    if valid_shape is None:
        return _authored_tile_physical_shape(shape), None, None, tuple(shape)

    if len(valid_shape) != logical_rank:
        raise TypeError(
            f"alloc_tile(valid_shape=...) rank mismatch: expected {logical_rank} dims, got {len(valid_shape)}"
        )

    surface_valid_shape = []
    if logical_rank == 1:
        dim = valid_shape[0]
        surface_valid_shape.append(dim)
        if isinstance(dim, bool):
            raise TypeError("alloc_tile(valid_shape=...) does not accept bool dimensions")
        if isinstance(dim, int):
            return (1, dim), None, None, tuple(surface_valid_shape)
        return (-1, -1), 1, dim, tuple(surface_valid_shape)

    type_valid_shape = []
    valid_row = None
    valid_col = None
    for index, dim in enumerate(valid_shape):
        surface_valid_shape.append(dim)
        if isinstance(dim, bool):
            raise TypeError("alloc_tile(valid_shape=...) does not accept bool dimensions")
        if isinstance(dim, int):
            type_valid_shape.append(dim)
            continue
        type_valid_shape.append(-1)
        if index == 0:
            valid_row = dim
            continue
        if index == 1:
            valid_col = dim
            continue
        raise TypeError(
            "alloc_tile(valid_shape=...) currently only supports dynamic runtime metadata "
            "for the first two dimensions"
        )
    return tuple(type_valid_shape), valid_row, valid_col, tuple(surface_valid_shape)


def _uses_row_major_none_box_layout(blayout, slayout) -> bool:
    return str(blayout).lower() == "rowmajor" and str(slayout).lower() == "nonebox"


def _validate_authored_tile_row_alignment(shape, dtype, *, blayout, slayout):
    if not _uses_row_major_none_box_layout(blayout, slayout):
        return
    if not shape:
        return
    elem_bytewidth = _element_bytewidth(_resolve(dtype))
    row_bytes = shape[-1] * elem_bytewidth
    required_alignment = 32
    if row_bytes % required_alignment == 0:
        return
    raise tile_row_alignment_error(
        shape=shape,
        dtype=str(_resolve(dtype)),
        row_bytes=row_bytes,
        required_alignment=required_alignment,
    )


def partition_view(tv, *, offsets, sizes):
    """
    ``pto.partition_view`` – slice a tensor view.

    Type is inferred from the source tensor-view type.
    """
    spec = compose_partition_spec(tv, offsets=offsets, sizes=sizes)
    if spec is not None:
        source = spec.root_tensor_view
        offsets = spec.offsets
        sizes = spec.sizes
    else:
        source = tv

    raw_source = unwrap_surface_value(source)
    src_type = _pto.TensorViewType(raw_source.type)
    rank = src_type.rank
    elem = src_type.element_type
    normalized_offsets = [
        _coerce_index(offset, context="partition_view(offsets=...)")
        for offset in offsets
    ]
    normalized_sizes = [
        _coerce_index(size, context="partition_view(sizes=...)")
        for size in sizes
    ]
    static_dims = _static_index_dims(normalized_sizes)
    ptv_type = (
        part_tensor_view_type_from_dims(static_dims, elem)
        if static_dims is not None
        else part_tensor_view_type(rank, elem)
    )
    value = _pto.PartitionViewOp(
        ptv_type,
        raw_source,
        _unwrap_sequence(normalized_offsets),
        _unwrap_sequence(normalized_sizes),
    ).result
    return wrap_surface_value(
        value,
        root_tensor_view=source if spec is None else spec.root_tensor_view,
        offsets=tuple(offsets),
        sizes=tuple(sizes),
    )


def _tile_logical_rank(tile, *, context: str) -> int:
    shape = getattr(tile, "shape", None)
    if shape is not None:
        return len(shape)
    parsed = parse_tile_type_metadata(unwrap_surface_value(tile).type)
    if parsed is not None:
        return len(parsed["shape_dims"])
    raise TypeError(f"{context} requires tile shape metadata to infer sizes")


def _source_view_rank(tv, *, context: str) -> int:
    shape = getattr(tv, "shape", None)
    if shape is not None:
        return len(shape)
    raw_type = unwrap_surface_value(tv).type
    try:
        return _pto.TensorViewType(raw_type).rank
    except Exception:
        try:
            return _pto.PartitionTensorViewType(raw_type).rank
        except Exception as exc:
            raise TypeError(f"{context} expects a tensor view or partition tensor view, got {raw_type}") from exc


def _is_partition_tensor_view(value) -> bool:
    try:
        _pto.PartitionTensorViewType(unwrap_surface_value(value).type)
        return True
    except Exception:
        return False


def _normalize_transfer_offsets(tv, *, offsets, context: str):
    if offsets is None:
        return [0] * _source_view_rank(tv, context=context)
    if isinstance(offsets, tuple):
        return list(offsets)
    if isinstance(offsets, list):
        return offsets
    return [offsets]


def _normalize_transfer_sizes(sizes):
    if isinstance(sizes, tuple):
        return list(sizes)
    if isinstance(sizes, list):
        return sizes
    return [sizes]


def _infer_tile_transfer_sizes(tile, *, context: str):
    valid_shape = getattr(tile, "valid_shape", None)
    if valid_shape is None:
        raise TypeError(f"{context} requires tile valid_shape metadata to infer sizes")
    sizes = []
    for index in range(_tile_logical_rank(tile, context=context)):
        try:
            dim = valid_shape[index]
        except Exception as exc:
            raise TypeError(
                f"{context} could not read tile.valid_shape[{index}] to infer sizes; "
                "pass sizes= explicitly"
            ) from exc
        if dim is None:
            raise ValueError(
                f"{context} cannot infer partition sizes because tile.valid_shape[{index}] is None; "
                "pass sizes= explicitly"
            )
        sizes.append(dim)
    return sizes


def _tile_transfer_partition(tv, tile, *, offsets=None, sizes=None, context: str):
    normalized_offsets = _normalize_transfer_offsets(
        tv,
        offsets=offsets,
        context=context,
    )
    normalized_sizes = (
        _infer_tile_transfer_sizes(tile, context=context)
        if sizes is None
        else _normalize_transfer_sizes(sizes)
    )
    if len(normalized_offsets) != len(normalized_sizes):
        if sizes is None:
            raise ValueError(
                f"{context} cannot infer partition sizes for rank-{len(normalized_offsets)} view "
                f"from rank-{len(normalized_sizes)} tile; pass sizes= explicitly"
            )
        raise ValueError(
            f"{context} expects offset rank and sizes rank to match, got "
            f"{len(normalized_offsets)} and {len(normalized_sizes)}"
        )
    return partition_view(tv, offsets=normalized_offsets, sizes=normalized_sizes)


def alloc_buffer(shape, dtype, **kwargs):
    """
    Allocate explicit-body scratch storage and return an address-like value.

    The allocation emits an LLVM stack allocation in the surrounding explicit
    kernel or helper body. UB scratch uses explicit ``pto.castptr`` /
    ``pto.addptr`` pointer authoring and an appropriate host launch wrapper.
    """
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(
            f"pto.alloc_buffer(...) does not accept keyword argument(s): {unexpected}. "
            "It only allocates explicit-body local buffers; author UB scratch explicitly with "
            "pto.castptr/pto.addptr and pass the dynamic UB byte count at launch."
        )
    _require_explicit_mode("pto.alloc_buffer(...)")
    element_type = _resolve(dtype)
    element_count = _static_alloc_buffer_element_count(shape)
    elem_bytes = _element_bytewidth(element_type)
    byte_size = element_count * elem_bytes

    return _alloc_local_buffer(
        shape,
        dtype,
        element_type,
        element_count,
        byte_size,
    )


def _static_alloc_buffer_element_count(shape):
    if isinstance(shape, int):
        dims = (shape,)
    elif isinstance(shape, (list, tuple)):
        dims = tuple(shape)
    else:
        raise TypeError("pto.alloc_buffer(shape, ...) expects an int or a tuple/list of static dimensions")
    if not dims:
        raise ValueError("pto.alloc_buffer(shape, ...) expects at least one dimension")
    count = 1
    for dim in dims:
        raw_dim = unwrap_surface_value(dim)
        if isinstance(raw_dim, bool):
            raise TypeError("pto.alloc_buffer(shape, ...) does not accept bool dimensions")
        if not isinstance(raw_dim, int):
            raise TypeError(
                "pto.alloc_buffer(shape, ...) requires static integer dimensions; "
                f"got {getattr(raw_dim, 'type', type(raw_dim).__name__)}"
            )
        if raw_dim <= 0:
            raise ValueError(f"pto.alloc_buffer(shape, ...) dimensions must be positive, got {raw_dim}")
        count *= raw_dim
    return count


def _alloc_local_buffer(shape, dtype, element_type, element_count, byte_size):
    i32 = IntegerType.get_signless(32)
    count = _materialize_integer_literal(i32, element_count)
    llvm_ptr_type = Type.parse("!llvm.ptr")
    attributes = {
        "elem_type": TypeAttr.get(element_type),
    }
    if _is_persistent_alloc_buffer_candidate():
        attributes["pto.persistent"] = UnitAttr.get()
    alloca = Operation.create(
        "llvm.alloca",
        results=[llvm_ptr_type],
        operands=[count],
        attributes=attributes,
    ).results[0]
    return AllocatedBufferValue(
        alloca,
        shape=_normalize_alloc_buffer_shape_metadata(shape),
        dtype=dtype,
        element_type=element_type,
        element_count=element_count,
        byte_size=byte_size,
    )


def _is_persistent_alloc_buffer_candidate() -> bool:
    try:
        from ._tracing.active import current_session
        session = current_session()
    except Exception:
        session = None
    if session is None:
        return False
    if getattr(session.module_spec, "entry", False) is not True:
        return False
    if session.current_function is not session.entry_function:
        return False
    return session.current_subkernel is None


def _normalize_alloc_buffer_shape_metadata(shape):
    if isinstance(shape, int):
        return (shape,)
    return tuple(unwrap_surface_value(dim) for dim in shape)


def _authored_alloc_tile_type(
    shape,
    dtype,
    memory_space,
    valid_shape,
    blayout,
    slayout,
    fractal_size,
    pad,
):
    if shape is None or dtype is None:
        raise TypeError("alloc_tile() requires either tile_type or both shape= and dtype=")
    logical_shape = _normalize_static_tile_shape(shape)
    physical_shape = _authored_tile_physical_shape(logical_shape)
    _validate_authored_tile_row_alignment(
        physical_shape,
        dtype,
        blayout=blayout,
        slayout=slayout,
    )
    type_valid_shape, valid_row, valid_col, surface_valid_shape = _split_valid_shape(
        logical_shape,
        valid_shape,
    )
    from ._types import tile_buf_type

    tile_type = tile_buf_type(
        physical_shape,
        dtype,
        type_valid_shape,
        blayout=blayout,
        address_space=memory_space,
        slayout=slayout,
        fractal_size=fractal_size,
        pad=pad,
    )
    return tile_type, logical_shape, physical_shape, surface_valid_shape, valid_row, valid_col


def _explicit_tile_valid_shape(tile_type, shape, valid_row, valid_col):
    if valid_row is None and valid_col is None:
        return None
    parsed_tile_type = parse_tile_type_metadata(_resolve(tile_type))
    rank = len(shape) if shape is not None else len(parsed_tile_type["shape_dims"])
    surface_valid_shape = [None] * rank
    if rank >= 1:
        surface_valid_shape[0] = valid_row
    if rank >= 2:
        surface_valid_shape[1] = valid_col
    return tuple(surface_valid_shape)


def _normalize_alloc_tile_request(
    tile_type,
    shape,
    dtype,
    memory_space,
    valid_shape,
    blayout,
    slayout,
    fractal_size,
    pad,
    valid_row,
    valid_col,
):
    if tile_type is not None and shape is not None:
        raise TypeError("alloc_tile() accepts either tile_type or shape=/dtype=, not both")
    if tile_type is None:
        if valid_row is not None or valid_col is not None:
            raise TypeError(
                "alloc_tile(shape=..., dtype=...) uses the authored surface form; "
                "use valid_shape=... instead of valid_row=/valid_col="
            )
        return _authored_alloc_tile_type(
            shape,
            dtype,
            memory_space,
            valid_shape,
            blayout,
            slayout,
            fractal_size,
            pad,
        )
    return (
        tile_type,
        shape,
        None,
        _explicit_tile_valid_shape(tile_type, shape, valid_row, valid_col),
        valid_row,
        valid_col,
    )


def _emit_alloc_tile(tile_type, addr, valid_row, valid_col):
    return _pto.AllocTileOp(
        _resolve(tile_type),
        addr=_coerce_i64(addr, context="alloc_tile(addr)") if addr is not None else None,
        valid_row=_coerce_index(valid_row, context="alloc_tile(valid_row)") if valid_row is not None else None,
        valid_col=_coerce_index(valid_col, context="alloc_tile(valid_col)") if valid_col is not None else None,
    ).result


def _wrap_allocated_tile(value, shape, physical_shape, dtype, memory_space, valid_shape):
    return wrap_surface_value(
        value,
        tile_metadata={
            "shape": shape,
            "physical_shape": physical_shape,
            "dtype": dtype,
            "memory_space": memory_space,
            "valid_shape": valid_shape,
        },
    )


def _materialize_alloc_tile(
    tile_type,
    shape,
    physical_shape,
    dtype,
    memory_space,
    surface_valid_shape,
    addr,
    valid_row,
    valid_col,
):
    value = _emit_alloc_tile(tile_type, addr, valid_row, valid_col)
    return _wrap_allocated_tile(
        value,
        shape,
        physical_shape,
        dtype,
        memory_space,
        surface_valid_shape,
    )


def alloc_tile(
    tile_type=None,
    *,
    shape=None,
    dtype=None,
    memory_space="ub",
    valid_shape=None,
    blayout: str = "RowMajor",
    slayout: str = "NoneBox",
    fractal_size: int = 512,
    pad: str = "Null",
    addr=None,
    valid_row=None,
    valid_col=None,
):
    """
    ``pto.alloc_tile``.

    Accepts either the authored surface form:
    ``alloc_tile(shape=[...], dtype=..., memory_space=..., valid_shape=..., addr=...)``
    or the low-level explicit-type form:
    ``alloc_tile(tile_type, addr=..., valid_row=..., valid_col=...)``.
    """
    (
        tile_type,
        shape,
        physical_shape,
        surface_valid_shape,
        valid_row,
        valid_col,
    ) = _normalize_alloc_tile_request(
        tile_type,
        shape,
        dtype,
        memory_space,
        valid_shape,
        blayout,
        slayout,
        fractal_size,
        pad,
        valid_row,
        valid_col,
    )
    return _materialize_alloc_tile(
        tile_type, shape, physical_shape, dtype, memory_space,
        surface_valid_shape, addr, valid_row, valid_col,
    )


def set_tile_valid_shape(tile, valid_shape):
    """Update the runtime valid-shape metadata of an authored dynamic tile."""
    parsed_tile_type = parse_tile_type_metadata(unwrap_surface_value(tile).type)
    if parsed_tile_type is None:
        raise TypeError("tile.valid_shape assignment expects a tile_buf-backed value")
    if len(parsed_tile_type["shape_dims"]) != 2:
        raise TypeError("tile.valid_shape assignment currently only supports rank-2 tiles")
    logical_rank = len(tile.shape) if getattr(tile, "shape", None) is not None else 2
    if logical_rank == 1:
        if len(valid_shape) != 1:
            raise TypeError("rank-1 tile.valid_shape assignment expects exactly one dimension")
        if parsed_tile_type["valid_dims"] != (None, None):
            raise TypeError(
                "rank-1 tile.valid_shape assignment requires a tile allocated with "
                "valid_shape=[...] so the physical valid row/col metadata remain dynamic"
            )
        valid_row = _coerce_index_value(1)
        valid_col = _coerce_index(valid_shape[0], context="tile.valid_shape assignment")
    else:
        if len(valid_shape) != 2:
            raise TypeError("tile.valid_shape assignment currently expects exactly two dimensions")
        if parsed_tile_type["valid_dims"] != (None, None):
            raise TypeError(
                "tile.valid_shape assignment requires a tile allocated with fully dynamic "
                "valid_shape=[..., ...]"
            )
        valid_row = _coerce_index(valid_shape[0], context="tile.valid_shape assignment")
        valid_col = _coerce_index(valid_shape[1], context="tile.valid_shape assignment")
    _pto.SetValidShapeOp(
        unwrap_surface_value(tile),
        valid_row,
        valid_col,
    )


def tload(part, tile):
    """``pto.tload ins(part) outs(tile)``."""
    _pto.TLoadOp(None, unwrap_surface_value(part), unwrap_surface_value(tile))


def tstore(tile, part, *, fp=None):
    """``pto.tstore ins(tile) outs(part)``."""
    kwargs = {}
    if fp is not None:
        kwargs["fp"] = unwrap_surface_value(fp)
    _pto.TStoreOp(
        None, unwrap_surface_value(tile), unwrap_surface_value(part), **kwargs
    )


def tmov(src, dst, *, fp=None, mode=None):
    """``pto.tmov ins(src) outs(dst)`` – move data between tile domains."""
    kwargs = {}
    if fp is not None:
        kwargs["fp"] = unwrap_surface_value(fp)
    if mode is not None:
        kwargs["accToVecMode"] = _normalize_acc_to_vec_mode(mode, context="tmov(..., mode=...)")
    _pto.TMovOp(None, unwrap_surface_value(src), unwrap_surface_value(dst), **kwargs)


def ttrans(src, tmp, dst):
    """``pto.ttrans ins(src, tmp) outs(dst)`` – tile transpose (DPS)."""
    _pto.TTransOp(
        unwrap_surface_value(src),
        unwrap_surface_value(dst),
        tmp=unwrap_surface_value(tmp),
    )


def textract(src, dst, index_row, index_col, *, fp=None, mode=None):
    """``pto.textract ins(src, index_row, index_col) outs(dst)``."""
    kwargs = {}
    if fp is not None:
        kwargs["fp"] = unwrap_surface_value(fp)
    if mode is not None:
        kwargs["accToVecMode"] = _normalize_acc_to_vec_mode(
            mode, context="textract(..., mode=...)"
        )
    _pto.TExtractOp(
        unwrap_surface_value(src),
        _coerce_index(index_row, context="textract(index_row)"),
        _coerce_index(index_col, context="textract(index_col)"),
        unwrap_surface_value(dst),
        **kwargs,
    )


def tinsert(src, dst, index_row, index_col, *, fp=None, mode=None):
    """``pto.tinsert ins(src, index_row, index_col) outs(dst)``."""
    kwargs = {}
    if fp is not None:
        kwargs["fp"] = unwrap_surface_value(fp)
    if mode is not None:
        kwargs["accToVecMode"] = _normalize_acc_to_vec_mode(
            mode, context="tinsert(..., mode=...)"
        )
    _pto.TInsertOp(
        unwrap_surface_value(src),
        _coerce_index(index_row, context="tinsert(index_row)"),
        _coerce_index(index_col, context="tinsert(index_col)"),
        unwrap_surface_value(dst),
        **kwargs,
    )


def tconcat(src0, src1, dst):
    """``pto.tconcat ins(src0, src1) outs(dst)``."""
    _pto.tconcat(
        unwrap_surface_value(src0),
        unwrap_surface_value(src1),
        unwrap_surface_value(dst),
    )


def tmatmul(lhs, rhs, dst):
    """``pto.tmatmul ins(lhs, rhs) outs(dst)``."""
    _pto.TMatmulOp(
        None,
        unwrap_surface_value(lhs),
        unwrap_surface_value(rhs),
        unwrap_surface_value(dst),
    )


def tmatmul_acc(acc_in, lhs, rhs, dst):
    """``pto.tmatmul.acc ins(acc_in, lhs, rhs) outs(dst)``."""
    _pto.TMatmulAccOp(
        None,
        unwrap_surface_value(acc_in),
        unwrap_surface_value(lhs),
        unwrap_surface_value(rhs),
        unwrap_surface_value(dst),
    )


def tmatmul_mx(lhs, lhs_scale, rhs, rhs_scale, dst, *, acc_phase=None):
    """``pto.tmatmul.mx ins(lhs, lhs_scale, rhs, rhs_scale) outs(dst)``."""
    _pto.TMatmulMxOp(
        None,
        unwrap_surface_value(lhs),
        unwrap_surface_value(lhs_scale),
        unwrap_surface_value(rhs),
        unwrap_surface_value(rhs_scale),
        unwrap_surface_value(dst),
        accPhase=acc_phase,
    )


def tmatmul_mx_acc(acc_in, lhs, lhs_scale, rhs, rhs_scale, dst, *, acc_phase=None):
    """``pto.tmatmul.mx.acc ins(acc_in, lhs, lhs_scale, rhs, rhs_scale) outs(dst)``."""
    _pto.TMatmulMxAccOp(
        None,
        unwrap_surface_value(acc_in),
        unwrap_surface_value(lhs),
        unwrap_surface_value(lhs_scale),
        unwrap_surface_value(rhs),
        unwrap_surface_value(rhs_scale),
        unwrap_surface_value(dst),
        accPhase=acc_phase,
    )


def tmatmul_mx_bias(lhs, lhs_scale, rhs, rhs_scale, bias, dst):
    """``pto.tmatmul.mx.bias ins(lhs, lhs_scale, rhs, rhs_scale, bias) outs(dst)``."""
    _pto.TMatmulMxBiasOp(
        None,
        unwrap_surface_value(lhs),
        unwrap_surface_value(lhs_scale),
        unwrap_surface_value(rhs),
        unwrap_surface_value(rhs_scale),
        unwrap_surface_value(bias),
        unwrap_surface_value(dst),
    )


def tgemv(lhs, rhs, dst, *, acc_phase=None):
    """``pto.tgemv ins(lhs, rhs) outs(dst)``."""
    _pto.TGemvOp(
        None,
        unwrap_surface_value(lhs),
        unwrap_surface_value(rhs),
        unwrap_surface_value(dst),
        accPhase=acc_phase,
    )


def tgemv_acc(acc_in, lhs, rhs, dst, *, acc_phase=None):
    """``pto.tgemv.acc ins(acc_in, lhs, rhs) outs(dst)``."""
    _pto.TGemvAccOp(
        None,
        unwrap_surface_value(acc_in),
        unwrap_surface_value(lhs),
        unwrap_surface_value(rhs),
        unwrap_surface_value(dst),
        accPhase=acc_phase,
    )


def tgemv_bias(lhs, rhs, bias, dst):
    """``pto.tgemv.bias ins(lhs, rhs, bias) outs(dst)``."""
    _pto.TGemvBiasOp(
        None,
        unwrap_surface_value(lhs),
        unwrap_surface_value(rhs),
        unwrap_surface_value(bias),
        unwrap_surface_value(dst),
    )


def tgemv_mx(lhs, lhs_scale, rhs, rhs_scale, dst, *, acc_phase=None):
    """``pto.tgemv.mx ins(lhs, lhs_scale, rhs, rhs_scale) outs(dst)``."""
    _pto.TGemvMxOp(
        None,
        unwrap_surface_value(lhs),
        unwrap_surface_value(lhs_scale),
        unwrap_surface_value(rhs),
        unwrap_surface_value(rhs_scale),
        unwrap_surface_value(dst),
        accPhase=acc_phase,
    )


def tgemv_mx_acc(acc_in, lhs, lhs_scale, rhs, rhs_scale, dst, *, acc_phase=None):
    """``pto.tgemv.mx.acc ins(acc_in, lhs, lhs_scale, rhs, rhs_scale) outs(dst)``."""
    _pto.TGemvMxAccOp(
        None,
        unwrap_surface_value(acc_in),
        unwrap_surface_value(lhs),
        unwrap_surface_value(lhs_scale),
        unwrap_surface_value(rhs),
        unwrap_surface_value(rhs_scale),
        unwrap_surface_value(dst),
        accPhase=acc_phase,
    )


def tgemv_mx_bias(lhs, lhs_scale, rhs, rhs_scale, bias, dst):
    """``pto.tgemv.mx.bias ins(lhs, lhs_scale, rhs, rhs_scale, bias) outs(dst)``."""
    _pto.TGemvMxBiasOp(
        None,
        unwrap_surface_value(lhs),
        unwrap_surface_value(lhs_scale),
        unwrap_surface_value(rhs),
        unwrap_surface_value(rhs_scale),
        unwrap_surface_value(bias),
        unwrap_surface_value(dst),
    )


def _coerce_tile_scalar_operand(tile, scalar, *, context: str):
    return _constant_like(scalar, infer_tile_element_type(wrap_surface_value(tile)))


def tadd(src0, src1, dst):
    """``pto.tadd ins(src0, src1) outs(dst)``."""
    _pto.tadd(
        unwrap_surface_value(src0),
        unwrap_surface_value(src1),
        unwrap_surface_value(dst),
    )


def taddrelu(src0, src1, dst):
    """``pto.taddrelu ins(src0, src1) outs(dst)``."""
    _require_target_arch("pto.tile.addrelu", {"a2", "a3"})
    _pto.taddrelu(
        unwrap_surface_value(src0),
        unwrap_surface_value(src1),
        unwrap_surface_value(dst),
    )


def tsub(src0, src1, dst):
    """``pto.tsub ins(src0, src1) outs(dst)``."""
    _pto.tsub(
        unwrap_surface_value(src0),
        unwrap_surface_value(src1),
        unwrap_surface_value(dst),
    )


def tmul(src0, src1, dst):
    """``pto.tmul ins(src0, src1) outs(dst)``."""
    _pto.tmul(
        unwrap_surface_value(src0),
        unwrap_surface_value(src1),
        unwrap_surface_value(dst),
    )


def tdiv(src0, src1, dst, *, precision=None, div_precision=_UNSET):
    """``pto.tdiv ins(src0, src1) outs(dst)``."""
    precision = _resolve_precision_argument(
        precision, div_precision, legacy_name="div_precision", context="pto.tdiv"
    )
    _pto.tdiv(
        unwrap_surface_value(src0),
        unwrap_surface_value(src1),
        unwrap_surface_value(dst),
        precision_type=_normalize_enum_attr(
            precision,
            enum_cls=_pto.DivPrecision,
            attr_cls=_pto.DivPrecisionAttr,
            context="tdiv precision",
        ),
    )


def trem(src0, src1, dst, *, tmp=None, precision=None):
    """``pto.trem ins(src0, src1[, tmp]) outs(dst)``."""
    tmp = _resolve_a5_mandatory_tmp(dst, tmp, context="trem")
    kwargs = {
        "precision_type": _normalize_enum_attr(
            precision,
            enum_cls=_pto.RemPrecision,
            attr_cls=_pto.RemPrecisionAttr,
            context="trem precision",
        ),
    }
    if tmp is not None:
        kwargs["tmp"] = unwrap_surface_value(tmp)
    _pto.trem(
        unwrap_surface_value(src0),
        unwrap_surface_value(src1),
        unwrap_surface_value(dst),
        **kwargs,
    )


def trems(src, scalar, dst, *, tmp=None):
    """``pto.trems ins(src, scalar[, tmp]) outs(dst)``."""
    tmp = _resolve_a5_mandatory_tmp(dst, tmp, context="trems")
    kwargs = {}
    if tmp is not None:
        kwargs["tmp"] = unwrap_surface_value(tmp)
    _pto.trems(
        unwrap_surface_value(src),
        _coerce_tile_scalar_operand(src, scalar, context="trems"),
        unwrap_surface_value(dst),
        **kwargs,
    )


def tfmod(src0, src1, dst, *, precision=None):
    """``pto.tfmod ins(src0, src1) outs(dst)``."""
    _pto.tfmod(
        unwrap_surface_value(src0),
        unwrap_surface_value(src1),
        unwrap_surface_value(dst),
        precision_type=_normalize_enum_attr(
            precision,
            enum_cls=_pto.FmodPrecision,
            attr_cls=_pto.FmodPrecisionAttr,
            context="tfmod precision",
        ),
    )


def tfmods(src, scalar, dst):
    """``pto.tfmods ins(src, scalar) outs(dst)``."""
    _pto.tfmods(
        unwrap_surface_value(src),
        _coerce_tile_scalar_operand(src, scalar, context="tfmods"),
        unwrap_surface_value(dst),
    )


def tmax(src0, src1, dst):
    """``pto.tmax ins(src0, src1) outs(dst)``."""
    _pto.tmax(
        unwrap_surface_value(src0),
        unwrap_surface_value(src1),
        unwrap_surface_value(dst),
    )


def tmin(src0, src1, dst):
    """``pto.tmin ins(src0, src1) outs(dst)``."""
    _pto.tmin(
        unwrap_surface_value(src0),
        unwrap_surface_value(src1),
        unwrap_surface_value(dst),
    )


def tadds(src, scalar, dst):
    """``pto.tadds ins(src, scalar) outs(dst)``."""
    _pto.tadds(
        unwrap_surface_value(src),
        _coerce_tile_scalar_operand(src, scalar, context="tadds"),
        unwrap_surface_value(dst),
    )


def tsubs(src, scalar, dst):
    """``pto.tsubs ins(src, scalar) outs(dst)``."""
    _pto.tsubs(
        unwrap_surface_value(src),
        _coerce_tile_scalar_operand(src, scalar, context="tsubs"),
        unwrap_surface_value(dst),
    )


def tmuls(src, scalar, dst):
    """``pto.tmuls ins(src, scalar) outs(dst)``."""
    _pto.tmuls(
        unwrap_surface_value(src),
        _coerce_tile_scalar_operand(src, scalar, context="tmuls"),
        unwrap_surface_value(dst),
    )


def tdivs(src, scalar, dst, *, precision=None, div_precision=_UNSET):
    """``pto.tdivs ins(src, scalar) outs(dst)``.

    Accepts both ``(tile, scalar, dst)`` and ``(scalar, tile, dst)`` operand
    orders; the scalar-lhs form mirrors the A5 TileLib ``scalar_tile``
    templates.
    """
    precision = _resolve_precision_argument(
        precision, div_precision, legacy_name="div_precision", context="pto.tdivs"
    )
    precision_attr = _normalize_enum_attr(
        precision,
        enum_cls=_pto.DivPrecision,
        attr_cls=_pto.DivPrecisionAttr,
        context="tdivs precision",
    )
    if is_runtime_scalar_ir_type(getattr(src, "type", None)):
        _pto.tdivs(
            unwrap_surface_value(src),
            unwrap_surface_value(scalar),
            unwrap_surface_value(dst),
            precision_type=precision_attr,
        )
    else:
        _pto.tdivs(
            unwrap_surface_value(src),
            _coerce_tile_scalar_operand(src, scalar, context="tdivs"),
            unwrap_surface_value(dst),
            precision_type=precision_attr,
        )


def tmaxs(src, scalar, dst):
    """``pto.tmaxs ins(src, scalar) outs(dst)``."""
    _pto.tmaxs(
        unwrap_surface_value(src),
        _coerce_tile_scalar_operand(src, scalar, context="tmaxs"),
        unwrap_surface_value(dst),
    )


def tmins(src, scalar, dst):
    """``pto.tmins ins(src, scalar) outs(dst)``."""
    _pto.tmins(
        unwrap_surface_value(src),
        _coerce_tile_scalar_operand(src, scalar, context="tmins"),
        unwrap_surface_value(dst),
    )


def texp(src, dst, *, precision=None, exp_precision=_UNSET):
    """``pto.texp ins(src) outs(dst)``."""
    precision = _resolve_precision_argument(
        precision, exp_precision, legacy_name="exp_precision", context="pto.texp"
    )
    _pto.texp(
        unwrap_surface_value(src),
        unwrap_surface_value(dst),
        precision_type=_normalize_enum_attr(
            precision,
            enum_cls=_pto.ExpPrecision,
            attr_cls=_pto.ExpPrecisionAttr,
            context="texp precision",
        ),
    )


def tlog(src, dst, *, precision=None, log_precision=_UNSET):
    """``pto.tlog ins(src) outs(dst)``."""
    precision = _resolve_precision_argument(
        precision, log_precision, legacy_name="log_precision", context="pto.tlog"
    )
    _pto.tlog(
        unwrap_surface_value(src),
        unwrap_surface_value(dst),
        precision_type=_normalize_enum_attr(
            precision,
            enum_cls=_pto.LogPrecision,
            attr_cls=_pto.LogPrecisionAttr,
            context="tlog precision",
        ),
    )


def tsqrt(src, dst, *, precision=None, sqrt_precision=_UNSET):
    """``pto.tsqrt ins(src) outs(dst)``."""
    precision = _resolve_precision_argument(
        precision, sqrt_precision, legacy_name="sqrt_precision", context="pto.tsqrt"
    )
    _pto.tsqrt(
        unwrap_surface_value(src),
        unwrap_surface_value(dst),
        precision_type=_normalize_enum_attr(
            precision,
            enum_cls=_pto.SqrtPrecision,
            attr_cls=_pto.SqrtPrecisionAttr,
            context="tsqrt precision",
        ),
    )


def trsqrt(src, dst, *, tmp=None, precision=None, rsqrt_precision=_UNSET):
    """``pto.trsqrt ins(src, tmp?) outs(dst)``."""
    precision = _resolve_precision_argument(
        precision, rsqrt_precision, legacy_name="rsqrt_precision", context="pto.trsqrt"
    )
    _pto.trsqrt(
        unwrap_surface_value(src),
        unwrap_surface_value(dst),
        tmp=None if tmp is None else unwrap_surface_value(tmp),
        precision_type=_normalize_enum_attr(
            precision,
            enum_cls=_pto.RsqrtPrecision,
            attr_cls=_pto.RsqrtPrecisionAttr,
            context="trsqrt precision",
        ),
    )


def trecip(src, dst, *, precision=None, recip_precision=_UNSET):
    """``pto.trecip ins(src) outs(dst)``."""
    precision = _resolve_precision_argument(
        precision, recip_precision, legacy_name="recip_precision", context="pto.trecip"
    )
    _pto.trecip(
        unwrap_surface_value(src),
        unwrap_surface_value(dst),
        precision_type=_normalize_enum_attr(
            precision,
            enum_cls=_pto.RecipPrecision,
            attr_cls=_pto.RecipPrecisionAttr,
            context="trecip precision",
        ),
    )


def tabs(src, dst):
    """``pto.tabs ins(src) outs(dst)``."""
    _pto.tabs(
        unwrap_surface_value(src),
        unwrap_surface_value(dst),
    )


def tneg(src, dst):
    """``pto.tneg ins(src) outs(dst)``."""
    _pto.tneg(
        unwrap_surface_value(src),
        unwrap_surface_value(dst),
    )


def tdequant(src, scale, offset, dst):
    """``pto.tdequant ins(src, scale, offset) outs(dst)``."""
    _pto.tdequant(
        unwrap_surface_value(src),
        unwrap_surface_value(scale),
        unwrap_surface_value(offset),
        unwrap_surface_value(dst),
    )


_PRINT_FORMAT_ALIASES = {
    None: None,
    "width8_precision4": ("Width8_Precision4", "width8_precision4"),
    "Width8_Precision4": ("Width8_Precision4", "width8_precision4"),
    "width8_precision2": ("Width8_Precision2", "width8_precision2"),
    "Width8_Precision2": ("Width8_Precision2", "width8_precision2"),
    "width10_precision6": ("Width10_Precision6", "width10_precision6"),
    "Width10_Precision6": ("Width10_Precision6", "width10_precision6"),
}


def _normalize_print_format(value, *, context: str):
    if value is None:
        return None
    if hasattr(value, "type"):
        return value
    names = _PRINT_FORMAT_ALIASES.get(value)
    if names is None:
        allowed = ", ".join(sorted(k for k in _PRINT_FORMAT_ALIASES if isinstance(k, str)))
        raise ValueError(f"{context} expects one of {allowed}, got {value!r}")
    enum_name, asm_name = names
    if hasattr(_pto, "PrintFormat") and hasattr(_pto, "PrintFormatAttr"):
        enum_value = getattr(_pto.PrintFormat, enum_name)
        return _pto.PrintFormatAttr.get(enum_value)
    return Attribute.parse(f"#pto<print_format {asm_name}>")


def _require_emitc_tprint_backend():
    from ._tracing.active import current_runtime

    runtime = current_runtime()
    module_spec = getattr(runtime, "module_spec", None)
    backend = getattr(module_spec, "backend", None)
    if backend == "vpto":
        raise ValueError("pto.tprint is supported only by the EmitC backend; got backend='vpto'")


def print(fmt, scalar):
    """``pto.print ins(fmt, scalar)``."""
    raw_scalar = unwrap_surface_value(scalar)
    if IndexType.isinstance(raw_scalar.type):
        raw_scalar = arith.IndexCastOp(IntegerType.get_signless(32), raw_scalar).result
    _pto.PrintOp(str(fmt), raw_scalar)


def tprint(src, tmp=None, *, print_format=None):
    """``pto.tprint ins(src[, tmp])``."""
    _require_emitc_tprint_backend()
    kwargs = {}
    fmt_attr = _normalize_print_format(print_format, context="tprint(..., print_format=...)")
    if fmt_attr is not None:
        kwargs["printFormat"] = fmt_attr
    if tmp is None:
        _pto.TPrintOp(unwrap_surface_value(src), **kwargs)
    else:
        _pto.TPrintOp(unwrap_surface_value(src), tmp=unwrap_surface_value(tmp), **kwargs)


def trelu(src, dst):
    """``pto.trelu ins(src) outs(dst)``."""
    _pto.trelu(
        unwrap_surface_value(src),
        unwrap_surface_value(dst),
    )


def tprelu(src0, src1, dst, *, tmp=None):
    """``pto.tprelu ins(src0, src1[, tmp]) outs(dst)``."""
    tmp = _resolve_a5_mandatory_tmp(dst, tmp, context="tprelu")
    kwargs = {}
    if tmp is not None:
        kwargs["tmp"] = unwrap_surface_value(tmp)
    _pto.TPReluOp(
        unwrap_surface_value(src0),
        unwrap_surface_value(src1),
        unwrap_surface_value(dst),
        **kwargs,
    )


def _require_trandom_i32_scalar(value, *, name: str):
    """Return one signless i32 runtime scalar accepted by ``pto.trandom``."""
    raw_value = unwrap_surface_value(value)
    value_type = getattr(raw_value, "type", None)
    if (
        not isinstance(value_type, Type)
        or not IntegerType.isinstance(value_type)
        or IntegerType(value_type).width != 32
        or IntegerType(value_type).is_signed
        or IntegerType(value_type).is_unsigned
    ):
        raise TypeError(
            f"pto.trandom {name} must be a signless i32 runtime scalar, got {value_type}"
        )
    return raw_value


def trandom(key0, key1, counter0, counter1, counter2, counter3, dst, *, rounds=10):
    """``pto.trandom`` – generate A5 Philox random words into ``dst``.

    The six scalar operands are 32-bit key/counter words.  ``rounds`` is the
    compile-time Philox round count supported by the A5 operation (7 or 10).
    """
    if not isinstance(rounds, int) or isinstance(rounds, bool):
        raise TypeError("pto.trandom rounds must be a Python integer")
    if rounds not in (7, 10):
        raise ValueError(f"pto.trandom rounds must be 7 or 10, got {rounds}")
    _pto.TRandomOp(
        _require_trandom_i32_scalar(key0, name="key0"),
        _require_trandom_i32_scalar(key1, name="key1"),
        _require_trandom_i32_scalar(counter0, name="counter0"),
        _require_trandom_i32_scalar(counter1, name="counter1"),
        _require_trandom_i32_scalar(counter2, name="counter2"),
        _require_trandom_i32_scalar(counter3, name="counter3"),
        unwrap_surface_value(dst),
        rounds=rounds,
    )


def tlrelu(src, slope, dst):
    """``pto.tlrelu ins(src, slope) outs(dst)``."""
    _pto.tlrelu(
        unwrap_surface_value(src),
        _coerce_tile_scalar_operand(src, slope, context="tlrelu"),
        unwrap_surface_value(dst),
    )


def trowsum(src, tmp, dst):
    """``pto.trowsum ins(src, tmp) outs(dst)``."""
    _pto.trowsum(
        unwrap_surface_value(src),
        unwrap_surface_value(dst),
        tmp=unwrap_surface_value(tmp),
    )


def trowmax(src, tmp, dst):
    """``pto.trowmax ins(src, tmp) outs(dst)``."""
    _pto.trowmax(
        unwrap_surface_value(src),
        unwrap_surface_value(dst),
        tmp=unwrap_surface_value(tmp),
    )


def trowmin(src, tmp, dst):
    """``pto.trowmin ins(src, tmp) outs(dst)``."""
    _pto.trowmin(
        unwrap_surface_value(src),
        unwrap_surface_value(dst),
        tmp=unwrap_surface_value(tmp),
    )


def trowprod(src, tmp, dst):
    """``pto.trowprod ins(src, tmp) outs(dst)``."""
    _pto.trowprod(
        unwrap_surface_value(src),
        unwrap_surface_value(dst),
        tmp=unwrap_surface_value(tmp),
    )


def trowargmax(src, tmp, dst):
    """``pto.trowargmax ins(src, tmp) outs(dst)``."""
    _pto.trowargmax(
        unwrap_surface_value(src),
        unwrap_surface_value(dst),
        tmp=unwrap_surface_value(tmp),
    )


def trowargmin(src, tmp, dst):
    """``pto.trowargmin ins(src, tmp) outs(dst)``."""
    _pto.trowargmin(
        unwrap_surface_value(src),
        unwrap_surface_value(dst),
        tmp=unwrap_surface_value(tmp),
    )


def tcolsum(src, dst, *, tmp=None, is_binary=None):
    """``pto.tcolsum ins(src, tmp?) outs(dst)``."""
    _pto.tcolsum(
        unwrap_surface_value(src),
        unwrap_surface_value(dst),
        tmp=None if tmp is None else unwrap_surface_value(tmp),
        is_binary=is_binary,
    )


def tcolmax(src, dst):
    """``pto.tcolmax ins(src) outs(dst)``."""
    _pto.tcolmax(
        unwrap_surface_value(src),
        unwrap_surface_value(dst),
    )


def tcolmin(src, dst):
    """``pto.tcolmin ins(src) outs(dst)``."""
    _pto.tcolmin(
        unwrap_surface_value(src),
        unwrap_surface_value(dst),
    )


def tcolprod(src, dst):
    """``pto.tcolprod ins(src) outs(dst)``."""
    _pto.tcolprod(
        unwrap_surface_value(src),
        unwrap_surface_value(dst),
    )


def tcolargmax(src, tmp, dst):
    """``pto.tcolargmax ins(src, tmp) outs(dst)``."""
    _pto.tcolargmax(
        unwrap_surface_value(src),
        unwrap_surface_value(dst),
        tmp=unwrap_surface_value(tmp),
    )


def tcolargmin(src, tmp, dst):
    """``pto.tcolargmin ins(src, tmp) outs(dst)``."""
    _pto.tcolargmin(
        unwrap_surface_value(src),
        unwrap_surface_value(dst),
        tmp=unwrap_surface_value(tmp),
    )


def tcmp(src0, src1, dst, *, cmp_mode=None):
    """``pto.tcmp ins(src0, src1) outs(dst)``."""
    _pto.tcmp(
        unwrap_surface_value(src0),
        unwrap_surface_value(src1),
        unwrap_surface_value(dst),
        cmp_mode=None if cmp_mode is None else _cmp_mode_attr(cmp_mode),
    )


def tcmps(src, scalar, dst, *, cmp_mode=None):
    """``pto.tcmps ins(src, scalar) outs(dst)``."""
    _pto.tcmps(
        unwrap_surface_value(src),
        _coerce_tile_scalar_operand(src, scalar, context="tcmps"),
        unwrap_surface_value(dst),
        cmp_mode=None if cmp_mode is None else _cmp_mode_attr(cmp_mode),
    )


def texpands(scalar, dst):
    """``pto.texpands ins(scalar) outs(dst)``."""
    _pto.texpands(
        _coerce_tile_scalar_operand(dst, scalar, context="texpands"),
        unwrap_surface_value(dst),
    )


def _tile_numel(shape, *, context: str):
    numel = 1
    for dim in shape:
        if isinstance(dim, bool) or not isinstance(dim, int):
            raise TypeError(f"{context} currently requires a static shape")
        numel *= dim
    return numel


def treshape(src, *, shape, dtype=None, blayout=None):
    """``pto.treshape ins(src) -> result``."""
    src_value = unwrap_surface_value(src)
    src_shape = getattr(src, "shape", None)
    src_dtype = getattr(src, "dtype", None)
    src_memory_space = getattr(src, "memory_space", None)
    src_metadata = parse_tile_type_metadata(src_value.type)
    if src_shape is None and src_metadata is not None:
        src_shape = tuple(src_metadata["shape_dims"])
    if src_dtype is None and src_metadata is not None:
        src_dtype = src_metadata["element_type"]
    if src_memory_space is None and src_metadata is not None:
        src_memory_space = src_metadata["memory_space"]
    if src_shape is None or src_dtype is None or src_memory_space is None:
        raise TypeError("treshape(...) expects a tile_buf-backed Tile value")

    result_shape = _normalize_static_tile_shape(shape)
    result_dtype = dtype if dtype is not None else src_dtype
    result_blayout = blayout if blayout is not None else "RowMajor"

    src_numel = _tile_numel(src_shape, context="treshape(src, shape=...) source")
    dst_numel = _tile_numel(result_shape, context="treshape(src, shape=...) result")
    src_bytes = src_numel * _element_bytewidth(_resolve(src_dtype))
    dst_bytes = dst_numel * _element_bytewidth(_resolve(result_dtype))
    if src_bytes != dst_bytes:
        raise ValueError(
            "treshape(src, shape=..., dtype=...) requires source and result to have the same total byte size"
        )

    result_memory_space = src_memory_space
    result_physical_shape = _authored_tile_physical_shape(result_shape)
    _validate_authored_tile_row_alignment(
        result_physical_shape, result_dtype, blayout=result_blayout,
        slayout="NoneBox")

    from ._types import tile_buf_type

    result_type = tile_buf_type(
        result_physical_shape,
        result_dtype,
        blayout=result_blayout,
        address_space=result_memory_space,
    )
    value = _pto.treshape(result_type, src_value)
    return wrap_surface_value(
        value,
        tile_metadata={
            "shape": result_shape,
            "physical_shape": result_physical_shape,
            "dtype": result_dtype,
            "memory_space": result_memory_space,
        },
    )


def trowexpand(src, dst):
    """``pto.trowexpand ins(src) outs(dst)``."""
    _pto.trowexpand(
        unwrap_surface_value(src),
        unwrap_surface_value(dst),
    )


def tcolexpand(src, dst):
    """``pto.tcolexpand ins(src) outs(dst)``."""
    _pto.tcolexpand(
        unwrap_surface_value(src),
        unwrap_surface_value(dst),
    )


def trowexpandadd(src0, src1, dst):
    """``pto.trowexpandadd ins(src0, src1) outs(dst)``."""
    _pto.trowexpandadd(
        unwrap_surface_value(src0),
        unwrap_surface_value(src1),
        unwrap_surface_value(dst),
    )


def trowexpandsub(src0, src1, dst, *, tmp=None):
    """``pto.trowexpandsub ins(src0, src1, tmp?) outs(dst)``."""
    _pto.trowexpandsub(
        unwrap_surface_value(src0),
        unwrap_surface_value(src1),
        unwrap_surface_value(dst),
        tmp=None if tmp is None else unwrap_surface_value(tmp),
    )


def trowexpandmul(src0, src1, dst, *, tmp=None):
    """``pto.trowexpandmul ins(src0, src1, tmp?) outs(dst)``."""
    _pto.trowexpandmul(
        unwrap_surface_value(src0),
        unwrap_surface_value(src1),
        unwrap_surface_value(dst),
        tmp=None if tmp is None else unwrap_surface_value(tmp),
    )


def trowexpanddiv(src0, src1, dst, *, tmp=None, precision=None, div_precision=_UNSET):
    """``pto.trowexpanddiv ins(src0, src1, tmp?) outs(dst)``."""
    precision = _resolve_precision_argument(
        precision, div_precision, legacy_name="div_precision", context="pto.trowexpanddiv"
    )
    _pto.trowexpanddiv(
        unwrap_surface_value(src0),
        unwrap_surface_value(src1),
        unwrap_surface_value(dst),
        tmp=None if tmp is None else unwrap_surface_value(tmp),
        precision_type=_normalize_enum_attr(
            precision,
            enum_cls=_pto.DivPrecision,
            attr_cls=_pto.DivPrecisionAttr,
            context="trowexpanddiv precision",
        ),
    )


def trowexpandmax(src0, src1, dst, *, tmp=None):
    """``pto.trowexpandmax ins(src0, src1, tmp?) outs(dst)``."""
    _pto.trowexpandmax(
        unwrap_surface_value(src0),
        unwrap_surface_value(src1),
        unwrap_surface_value(dst),
        tmp=None if tmp is None else unwrap_surface_value(tmp),
    )


def trowexpandmin(src0, src1, dst, *, tmp=None):
    """``pto.trowexpandmin ins(src0, src1, tmp?) outs(dst)``."""
    _pto.trowexpandmin(
        unwrap_surface_value(src0),
        unwrap_surface_value(src1),
        unwrap_surface_value(dst),
        tmp=None if tmp is None else unwrap_surface_value(tmp),
    )


def trowexpandexpdif(src0, src1, dst, *, tmp=None):
    """``pto.trowexpandexpdif ins(src0, src1, tmp?) outs(dst)``."""
    _pto.trowexpandexpdif(
        unwrap_surface_value(src0),
        unwrap_surface_value(src1),
        unwrap_surface_value(dst),
        tmp=None if tmp is None else unwrap_surface_value(tmp),
    )


def tcolexpandadd(src0, src1, dst):
    """``pto.tcolexpandadd ins(src0, src1) outs(dst)``."""
    _pto.tcolexpandadd(
        unwrap_surface_value(src0),
        unwrap_surface_value(src1),
        unwrap_surface_value(dst),
    )


def tcolexpandsub(src0, src1, dst):
    """``pto.tcolexpandsub ins(src0, src1) outs(dst)``."""
    _pto.tcolexpandsub(
        unwrap_surface_value(src0),
        unwrap_surface_value(src1),
        unwrap_surface_value(dst),
    )


def tcolexpandmul(src0, src1, dst):
    """``pto.tcolexpandmul ins(src0, src1) outs(dst)``."""
    _pto.tcolexpandmul(
        unwrap_surface_value(src0),
        unwrap_surface_value(src1),
        unwrap_surface_value(dst),
    )


def tcolexpanddiv(src0, src1, dst, *, precision=None, div_precision=_UNSET):
    """``pto.tcolexpanddiv ins(src0, src1) outs(dst)``."""
    precision = _resolve_precision_argument(
        precision, div_precision, legacy_name="div_precision", context="pto.tcolexpanddiv"
    )
    _pto.tcolexpanddiv(
        unwrap_surface_value(src0),
        unwrap_surface_value(src1),
        unwrap_surface_value(dst),
        precision_type=_normalize_enum_attr(
            precision,
            enum_cls=_pto.DivPrecision,
            attr_cls=_pto.DivPrecisionAttr,
            context="tcolexpanddiv precision",
        ),
    )


def tcolexpandmax(src0, src1, dst):
    """``pto.tcolexpandmax ins(src0, src1) outs(dst)``."""
    _pto.tcolexpandmax(
        unwrap_surface_value(src0),
        unwrap_surface_value(src1),
        unwrap_surface_value(dst),
    )


def tcolexpandmin(src0, src1, dst):
    """``pto.tcolexpandmin ins(src0, src1) outs(dst)``."""
    _pto.tcolexpandmin(
        unwrap_surface_value(src0),
        unwrap_surface_value(src1),
        unwrap_surface_value(dst),
    )


def tcolexpandexpdif(src0, src1, dst):
    """``pto.tcolexpandexpdif ins(src0, src1) outs(dst)``."""
    _pto.tcolexpandexpdif(
        unwrap_surface_value(src0),
        unwrap_surface_value(src1),
        unwrap_surface_value(dst),
    )


def _resolve_selection_tmp(dst, tmp, *, context: str):
    if tmp is not None:
        return tmp

    session = None
    try:
        from ._tracing.active import current_session
        session = current_session()
    except Exception:
        session = None

    if session is not None and getattr(session.module_spec, "target_arch", None) == "a5":
        return dst

    return alloc_tile(tile_type=unwrap_surface_value(dst).type)


def _resolve_a5_mandatory_tmp(dst, tmp, *, context: str):
    """Supply an A5-only metadata placeholder for mandatory tmp operands.

    TileLib candidate selection runs before the compiler pass that materializes
    implicit scratch tiles.  A5 templates still bind a four-operand callable,
    although their template bodies do not consume the temporary.  Passing the
    destination as a placeholder preserves that selection contract; A2/A3
    continue to emit the implicit three-operand form for compiler-side
    scratch-size materialization.
    """
    if tmp is not None:
        return tmp

    session = None
    try:
        from ._tracing.active import current_session
        session = current_session()
    except Exception:
        session = None

    if session is not None and getattr(session.module_spec, "target_arch", None) == "a5":
        return dst
    return None


def tsort32(src, idx, dst, *, tmp=None):
    """``pto.tsort32 ins(src, idx, tmp?) outs(dst)``."""
    _pto.tsort32(
        unwrap_surface_value(src),
        unwrap_surface_value(idx),
        unwrap_surface_value(dst),
        tmp=None if tmp is None else unwrap_surface_value(tmp),
    )


def _unwrap_optional_integer(value):
    if value is None:
        return None
    if isinstance(value, int):
        value = const(value, dtype=IntegerType.get_signless(32))
    return unwrap_surface_value(value)


def tmrgsort(src, dst, block_len=None, *, tmp=None, excuted=None, exhausted=None):
    """``pto.tmrgsort`` tile merge-sort wrapper.

    Format 1 uses ``tmrgsort(src, dst, block_len)``.  Format 2 can pass
    ``src`` and ``dst`` as sequences and provide ``tmp`` plus ``excuted``.
    """
    srcs = src if isinstance(src, (list, tuple)) else [src]
    dsts = dst if isinstance(dst, (list, tuple)) else [dst]
    _pto.tmrgsort(
        [unwrap_surface_value(value) for value in srcs],
        [unwrap_surface_value(value) for value in dsts],
        block_len=_unwrap_optional_integer(block_len),
        tmp=None if tmp is None else unwrap_surface_value(tmp),
        excuted=None if excuted is None else unwrap_surface_value(excuted),
        exhausted=exhausted,
    )


def tgather(
    src,
    dst,
    *,
    cdst=None,
    indices=None,
    tmp=None,
    k_value=None,
    mask_pattern=None,
    axis=None,
    cmp_mode=None,
    offset=None,
):
    """``pto.tgather`` tile gather/select wrapper."""
    _validate_tgather_mode(indices, axis, mask_pattern)
    _pto.tgather(
        unwrap_surface_value(src),
        unwrap_surface_value(dst),
        cdst=None if cdst is None else unwrap_surface_value(cdst),
        indices=None if indices is None else unwrap_surface_value(indices),
        tmp=None if tmp is None else unwrap_surface_value(tmp),
        k_value=None if k_value is None else unwrap_surface_value(k_value),
        mask_pattern=None if mask_pattern is None else _tile_mask_pattern_attr(mask_pattern),
        axis=None if axis is None else axis,
        cmp_mode=None if cmp_mode is None else _normalize_cmp_mode(cmp_mode),
        offset=offset,
    )


def _validate_tgather_mode(indices, axis, mask_pattern):
    if indices is not None and (axis is not None or mask_pattern is not None):
        raise ValueError("indices and axis/mask_pattern cannot be provided together")
    if mask_pattern is not None and axis is None:
        raise ValueError("axis must be provided when mask_pattern is specified")
    if axis is not None and axis not in ("row", "col"):
        raise ValueError(f"axis must be 'row' or 'col', got {axis!r}")

def ttri(diagonal, dst, *, upper_or_lower="lower"):
    """``pto.ttri ins(diagonal) outs(dst)``."""
    if isinstance(upper_or_lower, str):
        if upper_or_lower == "lower":
            upper_or_lower = 0
        elif upper_or_lower == "upper":
            upper_or_lower = 1
        else:
            raise ValueError(
                f"upper_or_lower must be 'lower' or 'upper', got {upper_or_lower!r}"
            )
    elif upper_or_lower not in (0, 1):
        raise ValueError(f"upper_or_lower must be 0 (lower) or 1 (upper), got {upper_or_lower}")
    if not isinstance(diagonal, int):
        raise TypeError(
            f"ttri(diagonal) expects an int, got {type(diagonal).__name__}"
        )
    dst_dtype = str(infer_tile_element_type(dst))
    _TRI_ALLOWED_DTYPES = {
        "i8", "i16", "i32", "ui8", "ui16", "ui32", "f16", "bf16", "f32",
    }
    if dst_dtype not in _TRI_ALLOWED_DTYPES:
        raise ValueError(
            f"ttri dst dtype must be one of {sorted(_TRI_ALLOWED_DTYPES)}, "
            f"got {dst_dtype!r}"
        )
    _pto.ttri(
        _coerce_i32(diagonal, context="ttri(diagonal)"),
        unwrap_surface_value(dst),
        upper_or_lower=upper_or_lower,
    )

def tthistogram(src, idx, dst, *, byte=None):
    """``pto.thistogram ins(src, idx) outs(dst)``."""
    if byte is not None:
        if not isinstance(byte, int) or byte not in (0, 1, 2, 3):
            raise ValueError(f"byte must be an int in [0, 3], got {byte!r}")
    src_dtype = str(infer_tile_element_type(src))
    idx_dtype = str(infer_tile_element_type(idx))
    dst_dtype = str(infer_tile_element_type(dst))
    if src_dtype not in ("ui16", "ui32"):
        raise ValueError(
            f"thistogram src dtype must be ui16 or ui32, got {src_dtype!r}"
        )
    if idx_dtype != "ui8":
        raise ValueError(
            f"thistogram idx dtype must be ui8, got {idx_dtype!r}"
        )
    if dst_dtype != "ui32":
        raise ValueError(
            f"thistogram dst dtype must be ui32, got {dst_dtype!r}"
        )
    effective_byte = 1 if byte is None else byte
    if src_dtype == "ui16" and effective_byte not in (0, 1):
        raise ValueError(
            f"thistogram with ui16 src only supports byte 0 (LSB) or 1 (MSB), "
            f"got byte={effective_byte}"
        )
    _pto.thistogram(
        unwrap_surface_value(src),
        unwrap_surface_value(idx),
        unwrap_surface_value(dst),
        byte=byte,
    )

def tgatherb(src, offsets, dst):
    """``pto.tgatherb`` - gather 32-byte blocks using byte addresses (DPS)."""
    _pto.tgatherb(
        unwrap_surface_value(src),
        unwrap_surface_value(offsets),
        unwrap_surface_value(dst),
    )


def tci(start, dst, *, tmp=None, descending=False):
    """``pto.tci`` – generate contiguous integer sequence into dst tile (DPS)."""
    _pto.tci(
        _unwrap_optional_integer(start),
        unwrap_surface_value(dst),
        tmp=None if tmp is None else unwrap_surface_value(tmp),
        descending=descending,
    )


def tscatter(src, dst, *, indexes=None, axis=None, mask_pattern=None):
    """``pto.tscatter`` tile scatter wrapper."""
    if indexes is not None and (axis is not None or mask_pattern is not None):
        raise ValueError("indexes and axis/mask_pattern cannot be provided together")
    if indexes is None and (axis is None or mask_pattern is None):
        raise ValueError(f"non indexes mode must provide both axis and mask_pattern")
    if axis is not None and axis not in ("row", "col"):
        raise ValueError(f"axis must be 'row' or 'col', got {axis!r}")
    _pto.tscatter(
        unwrap_surface_value(src),
        unwrap_surface_value(dst),
        indexes=None if indexes is None else unwrap_surface_value(indexes),
        axis=None if axis is None else axis,
        mask_pattern=None if mask_pattern is None else _tile_mask_pattern_attr(mask_pattern),
    )


def tsel(mask, src0, src1, dst, *, tmp=None):
    """``pto.tsel ins(mask, src0, src1, tmp) outs(dst)`` with synthesized scratch when omitted."""
    resolved_tmp = tmp if tmp is not None else _resolve_selection_tmp(dst, tmp, context="tsel")
    _pto.tsel(
        unwrap_surface_value(mask),
        unwrap_surface_value(src0),
        unwrap_surface_value(src1),
        unwrap_surface_value(dst),
        tmp=unwrap_surface_value(resolved_tmp),
    )


def tsels(mask, src, scalar, dst, *, tmp=None):
    """``pto.tsels ins(mask, src, tmp, scalar) outs(dst)`` with synthesized scratch when omitted."""
    resolved_tmp = tmp if tmp is not None else _resolve_selection_tmp(dst, tmp, context="tsels")
    _pto.tsels(
        unwrap_surface_value(mask),
        unwrap_surface_value(src),
        _coerce_tile_scalar_operand(src, scalar, context="tsels"),
        unwrap_surface_value(dst),
        tmp=unwrap_surface_value(resolved_tmp),
    )


def tcvt(src, dst, *, tmp=None, rmode=None, sat_mode=None):
    """``pto.tcvt ins(src) outs(dst)``.

    The ``tmp`` parameter is retained for backward compatibility but is not
    supported by the current PTO backend; passing a non-None value raises.
    """
    if tmp is not None:
        raise TypeError("pto.tile.cvt(..., tmp=...) is not supported by the current PTO Python bindings")
    _pto.tcvt(
        unwrap_surface_value(src),
        unwrap_surface_value(dst),
        rmode=_normalize_enum_attr(
            rmode,
            enum_cls=_pto.RoundMode,
            attr_cls=_pto.RoundModeAttr,
            context="tile.cvt(..., rmode=...)",
        ),
        sat_mode=_normalize_enum_attr(
            sat_mode,
            enum_cls=_pto.SaturationMode,
            attr_cls=_pto.SaturationModeAttr,
            context="tile.cvt(..., sat_mode=...)",
        ),
    )


def tnot(src, dst):
    """``pto.tnot ins(src) outs(dst)``."""
    _pto.tnot(
        unwrap_surface_value(src),
        unwrap_surface_value(dst),
    )


def tand(src0, src1, dst):
    """``pto.tand ins(src0, src1) outs(dst)``."""
    _pto.tand(
        unwrap_surface_value(src0),
        unwrap_surface_value(src1),
        unwrap_surface_value(dst),
    )


def tands(src, scalar, dst):
    """``pto.tands ins(src, scalar) outs(dst)``."""
    _pto.tands(
        unwrap_surface_value(src),
        _coerce_tile_scalar_operand(src, scalar, context="tands"),
        unwrap_surface_value(dst),
    )


def tor(src0, src1, dst):
    """``pto.tor ins(src0, src1) outs(dst)``."""
    _pto.tor(
        unwrap_surface_value(src0),
        unwrap_surface_value(src1),
        unwrap_surface_value(dst),
    )


def tors(src, scalar, dst):
    """``pto.tors ins(src, scalar) outs(dst)``."""
    _pto.tors(
        unwrap_surface_value(src),
        _coerce_tile_scalar_operand(src, scalar, context="tors"),
        unwrap_surface_value(dst),
    )


def txor(src0, src1, tmp, dst):
    """``pto.txor ins(src0, src1, tmp) outs(dst)``."""
    _pto.txor(
        unwrap_surface_value(src0),
        unwrap_surface_value(src1),
        unwrap_surface_value(dst),
        tmp=unwrap_surface_value(tmp),
    )


def txors(src, scalar, tmp, dst):
    """``pto.txors ins(src, scalar, tmp) outs(dst)``."""
    _pto.txors(
        unwrap_surface_value(src),
        _coerce_tile_scalar_operand(src, scalar, context="txors"),
        unwrap_surface_value(dst),
        tmp=unwrap_surface_value(tmp),
    )


def tshl(src0, src1, dst):
    """``pto.tshl ins(src0, src1) outs(dst)``."""
    _pto.tshl(
        unwrap_surface_value(src0),
        unwrap_surface_value(src1),
        unwrap_surface_value(dst),
    )


def tshls(src, scalar, dst):
    """``pto.tshls ins(src, scalar) outs(dst)``."""
    _pto.tshls(
        unwrap_surface_value(src),
        _coerce_tile_scalar_operand(src, scalar, context="tshls"),
        unwrap_surface_value(dst),
    )


def tshr(src0, src1, dst):
    """``pto.tshr ins(src0, src1) outs(dst)``."""
    _pto.tshr(
        unwrap_surface_value(src0),
        unwrap_surface_value(src1),
        unwrap_surface_value(dst),
    )


def tshrs(src, scalar, dst):
    """``pto.tshrs ins(src, scalar) outs(dst)``."""
    _pto.tshrs(
        unwrap_surface_value(src),
        _coerce_tile_scalar_operand(src, scalar, context="tshrs"),
        unwrap_surface_value(dst),
    )


def tpartadd(src0, src1, dst):
    """``pto.tpartadd ins(src0, src1) outs(dst)``."""
    _pto.tpartadd(
        unwrap_surface_value(src0),
        unwrap_surface_value(src1),
        unwrap_surface_value(dst),
    )


def tpartmul(src0, src1, dst):
    """``pto.tpartmul ins(src0, src1) outs(dst)``."""
    _pto.tpartmul(
        unwrap_surface_value(src0),
        unwrap_surface_value(src1),
        unwrap_surface_value(dst),
    )


def tpartmax(src0, src1, dst):
    """``pto.tpartmax ins(src0, src1) outs(dst)``."""
    _pto.tpartmax(
        unwrap_surface_value(src0),
        unwrap_surface_value(src1),
        unwrap_surface_value(dst),
    )


def tpartmin(src0, src1, dst):
    """``pto.tpartmin ins(src0, src1) outs(dst)``."""
    _pto.tpartmin(
        unwrap_surface_value(src0),
        unwrap_surface_value(src1),
        unwrap_surface_value(dst),
    )


def tfillpad(src, dst):
    """``pto.tfillpad ins(src) outs(dst)`` with compiler-inferred lowering."""
    _pto.tfillpad(
        unwrap_surface_value(src),
        unwrap_surface_value(dst),
    )
