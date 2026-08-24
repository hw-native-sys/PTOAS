# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Builtin MLIR vector helpers for PTODSL scalar-contiguous access."""

from collections.abc import Mapping

from ._scalar_coercion import coerce_scalar_to_type
from ._surface_values import _SurfaceValue, VecValue, unwrap_surface_value
from ._types import _resolve, _signless_integer_type, _validate_vec_size, vec_type

from ptoas.mlir.dialects import arith
from ptoas.mlir.dialects import llvm
from ptoas.mlir.ir import IntegerType, VectorType


def Vec(dtype, size: int, *, init=None):
    """Create a builtin vector type descriptor, broadcast vector value, or per-element vector value.

    ``pto.Vec(dtype, size)`` returns a lazy vector type descriptor. ``pto.Vec(dtype, size, init=scalar)``
    broadcasts one scalar to every element. ``pto.Vec(dtype, size, init=sequence)`` packs one distinct
    scalar per element, preserving input order. ``pto.Vec(dtype, size, init=vector)`` passes a builtin
    vector through. A sequence init is any iterable of scalars that is not a mapping or str/bytes.
    """
    size = _validate_vec_size(size, context="pto.Vec(...)")
    descriptor = vec_type(dtype, size)
    if init is None:
        return descriptor
    return _init_vec_value(descriptor, init)


def _init_vec_value(descriptor, init):
    """Dispatch one authored ``init=`` input to broadcast, passthrough, or per-element packing."""
    if isinstance(init, (str, bytes, bytearray, Mapping)):
        raise TypeError(
            f"pto.Vec(..., init=...) expects a scalar, a builtin vector, or a sequence of scalars, got {type(init).__name__}"
        )
    # A tuple / list / generator / other iterable of scalars packs one distinct element per entry.
    if not isinstance(init, _SurfaceValue) and hasattr(init, "__iter__"):
        return _per_element_vec_value(descriptor, init)
    return _broadcast_vec_value(descriptor, init)


def _per_element_vec_value(descriptor, values):
    """Build a builtin vector value from one distinct authored scalar per element.

    ``values`` must be a sequence whose length equals ``descriptor.size``. Each entry is
    coerced to the vector element type with the same scalar coercion rules used by the
    broadcast path, then inserted at its positional index (``undef + insertelement``).
    """
    if isinstance(values, (str, bytes, bytearray)):
        raise TypeError("pto.Vec(..., init=sequence) expects a sequence of scalar values, got str/bytes")
    if isinstance(values, Mapping):
        raise TypeError(
            f"pto.Vec(..., init=sequence) expects a sequence of scalar values, got a {type(values).__name__} mapping"
        )
    if isinstance(values, _SurfaceValue):
        raise TypeError(
            f"pto.Vec(..., init=sequence) expects a sequence of scalar values, got {type(values).__name__}"
        )
    element_values = tuple(values)
    if len(element_values) != descriptor.size:
        raise ValueError(
            f"pto.Vec(..., init=sequence) expects exactly {descriptor.size} element(s), got {len(element_values)}"
        )
    vector_type = _resolve_vector_type_with_signless_elements(descriptor)
    element_type = VectorType(vector_type).element_type
    i32 = IntegerType.get_signless(32)
    current = llvm.UndefOp(vector_type).res
    for index, element in enumerate(element_values):
        raw_element = unwrap_surface_value(element)
        if hasattr(raw_element, "type") and VectorType.isinstance(raw_element.type):
            raise TypeError(
                f"pto.Vec(..., init=sequence) element {index} is a vector value ({raw_element.type}); each element must be a scalar"
            )
        scalar_value = coerce_scalar_to_type(
            element,
            element_type,
            context=f"pto.Vec(..., init=sequence) element {index}",
        )
        element_index = arith.ConstantOp(i32, index).result
        current = llvm.InsertElementOp(current, scalar_value, element_index).res
    return VecValue(current)


def _resolve_vector_type_with_signless_elements(descriptor):
    """Resolve a descriptor to an LLVM-compatible builtin vector type.

    LLVM dialect vectors require signless integer element types, so signed /
    unsigned integer dtypes (siN / uiN) resolve to the same-width signless iN
    vector; scalar coercion keeps the element bit patterns. Signless inputs are
    returned unchanged (no-op).
    """
    vector_type = _resolve(descriptor)
    element_type = VectorType(vector_type).element_type
    if IntegerType.isinstance(element_type):
        signless_element_type = _signless_integer_type(element_type)
        if signless_element_type != element_type:
            vector_type = VectorType.get([descriptor.size], signless_element_type)
    return vector_type


def _vector_types_bit_compatible(lhs, rhs):
    """Return whether two vector types store the same element bits."""
    if lhs == rhs:
        return True
    if VectorType.isinstance(lhs) and VectorType.isinstance(rhs):
        lhs_element = VectorType(lhs).element_type
        rhs_element = VectorType(rhs).element_type
        if IntegerType.isinstance(lhs_element) and IntegerType.isinstance(rhs_element):
            if IntegerType(lhs_element).width == IntegerType(rhs_element).width:
                return True
    return False


def _broadcast_vec_value(descriptor, init):
    vector_type = _resolve_vector_type_with_signless_elements(descriptor)
    element_type = VectorType(vector_type).element_type
    raw_init = unwrap_surface_value(init)

    if hasattr(raw_init, "type") and VectorType.isinstance(raw_init.type):
        vec_value = VecValue(raw_init)
        if not _vector_types_bit_compatible(vec_value.type, vector_type):
            raise TypeError(f"pto.Vec(..., init=vector) expected {vector_type}, got {vec_value.type}")
        return vec_value

    scalar_value = coerce_scalar_to_type(init, element_type, context="pto.Vec(..., init=...)")
    current = llvm.UndefOp(vector_type).res
    i32 = IntegerType.get_signless(32)
    for index in range(descriptor.size):
        element_index = arith.ConstantOp(i32, index).result
        current = llvm.InsertElementOp(current, scalar_value, element_index).res
    return VecValue(current)


__all__ = ["Vec"]
