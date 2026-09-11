# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Core PTODSL ops: constants, structs, and pointer arithmetic."""

from __future__ import annotations

# Shared import surface consolidated in _ops_imports to avoid repeating the
# identical block across the sibling _ops modules. See _ops_imports.__all__.
from ._ops_imports import *  # noqa: F401,F403




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
