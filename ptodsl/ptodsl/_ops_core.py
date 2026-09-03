# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Core PTODSL ops: constants, structs, and pointer arithmetic."""

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




def const(value: int, *, dtype=None):
    """
    Emit an ``arith.constant``.

    ``dtype`` is a ``_DType`` descriptor or a concrete ``ptoas.mlir.ir.Type``.
    Defaults to ``index`` when omitted.
    """
    from ._types import index as _idx_dtype
    mlir_type = _resolve(dtype) if dtype is not None else _resolve(_idx_dtype)
    if any(cls.isinstance(mlir_type) for cls in (F16Type, BF16Type, F32Type)):
        return wrap_surface_value(arith.ConstantOp(mlir_type, FloatAttr.get(mlir_type, value)).result)
    if IntegerType.isinstance(mlir_type):
        return wrap_surface_value(_materialize_integer_literal(mlir_type, value))
    return wrap_surface_value(arith.ConstantOp(mlir_type, value).result)


# ── Stack-local structs ──────────────────────────────────────────

def _normalize_struct_path(path, *, op_name: str) -> tuple[int, ...]:
    if isinstance(path, bool):
        raise TypeError(f"{op_name}: path must be a static int, tuple[int, ...], or list[int]")
    if isinstance(path, int):
        indices = (path,)
    elif isinstance(path, (tuple, list)):
        indices = tuple(path)
    else:
        raise TypeError(f"{op_name}: path must be a static int, tuple[int, ...], or list[int]")
    if not indices:
        raise ValueError(f"{op_name}: path must not be empty")
    for depth, index in enumerate(indices):
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError(
                f"{op_name}: path[{depth}] must be a static Python int, got {index!r}"
            )
        if index < 0:
            raise ValueError(f"{op_name}: path[{depth}] must be non-negative, got {index}")
    return indices


def _require_struct_value(struct, *, op_name: str):
    raw_struct = unwrap_surface_value(struct)
    struct_type = getattr(raw_struct, "type", None)
    if not _is_struct_type(struct_type):
        raise TypeError(
            f"{op_name}: struct must be a value of !pto.struct<...>, got "
            f"{getattr(raw_struct, 'type', type(raw_struct).__name__)}"
        )
    return raw_struct, _pto.StructType(struct_type)


def _resolve_struct_path(struct_type, path: tuple[int, ...], *, op_name: str):
    current_type = struct_type
    for depth, index in enumerate(path):
        fields = tuple(_pto.StructType(current_type).field_types)
        if index >= len(fields):
            raise ValueError(
                f"{op_name}: path[{depth}] = {index} is out of range for "
                f"a struct with {len(fields)} field(s)"
            )
        current_type = fields[index]
        if depth + 1 < len(path) and not _is_struct_type(current_type):
            raise TypeError(
                f"{op_name}: path[{depth}] selects scalar field {current_type}; "
                "cannot descend further"
            )
    if _is_struct_type(current_type):
        raise TypeError(
            f"{op_name}: path must end at a scalar field, but ends at {current_type}; "
            "extend the path to reach a scalar field"
        )
    return current_type


def _materialize_struct_value(value, field_type, *, op_name: str):
    raw_value = unwrap_surface_value(value)
    if isinstance(raw_value, bool):
        raise TypeError(f"{op_name}: value does not accept bool literals")
    if isinstance(raw_value, int):
        return materialize_scalar_literal(raw_value, field_type, context=op_name)
    if isinstance(raw_value, float):
        if not any(cls.isinstance(field_type) for cls in (F16Type, BF16Type, F32Type)):
            raise TypeError(
                f"{op_name}: cannot materialize a floating-point literal for integer field type {field_type}"
            )
        return materialize_scalar_literal(raw_value, field_type, context=op_name)
    if not hasattr(raw_value, "type"):
        raise TypeError(
            f"{op_name}: value must be a Python int/float literal or an SSA value, got {value!r}"
        )
    if raw_value.type != field_type:
        raise TypeError(
            f"{op_name}: SSA value type {raw_value.type} must exactly match struct field type {field_type}"
        )
    return raw_value


def declare_struct(struct_type):
    """Declare an uninitialized stack-local ``!pto.struct`` value."""
    resolved_type = _resolve(struct_type)
    if not _is_struct_type(resolved_type):
        raise TypeError(
            "pto.declare_struct(...): expected a pto.struct_type(...) descriptor or !pto.struct type, "
            f"got {resolved_type}"
        )
    return wrap_surface_value(_pto.DeclareStructOp(resolved_type).s)


def struct_get(struct, path):
    """Read one scalar field from a stack-local struct at a static field path."""
    raw_struct, struct_type = _require_struct_value(struct, op_name="pto.struct_get")
    indices = _normalize_struct_path(path, op_name="pto.struct_get")
    field_type = _resolve_struct_path(struct_type, indices, op_name="pto.struct_get")
    return wrap_surface_value(_pto.StructGetOp(field_type, raw_struct, indices).value)


def struct_set(struct, path, value):
    """Write one scalar field of a stack-local struct at a static field path."""
    raw_struct, struct_type = _require_struct_value(struct, op_name="pto.struct_set")
    indices = _normalize_struct_path(path, op_name="pto.struct_set")
    field_type = _resolve_struct_path(struct_type, indices, op_name="pto.struct_set")
    _pto.StructSetOp(raw_struct, indices, _materialize_struct_value(
        value,
        field_type,
        op_name="pto.struct_set",
    ))


def get_op_attr(name: str, default=None):
    """Return a TileLib render-time op attribute supplied by the C++ bridge."""
    from ._tracing.active import current_runtime

    runtime = current_runtime()
    attrs = getattr(runtime, "context_attrs", None)
    if not attrs:
        return default
    return attrs.get(name, default)


# ── Pointer ops ───────────────────────────────────────────────────────────────
