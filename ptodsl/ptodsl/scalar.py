# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
Scalar arithmetic helpers – exposed as top-level ``scalar.*`` from the
``ptodsl`` package (for example ``from ptodsl import scalar``).

Arithmetic helpers operate on raw ``ptoas.mlir.ir.Value`` objects and emit the
corresponding arith dialect operations at the active insertion point.
Scalar memory helpers (`load` / `store`) also accept PTODSL surface-level
address views such as `tile[row, col]` and `tile.as_ptr() + offset`.
"""

from ._scalar_coercion import coerce_scalar_to_type
from ._scalar_adaptation import classify_runtime_scalar_type
from ._runtime_scalar_ops import (
    emit_runtime_abs,
    emit_runtime_binary_op,
    emit_runtime_max,
    emit_runtime_min,
)
from ._surface_values import (
    AddressOffsetValue,
    AllocatedBufferValue,
    VecValue,
    resolve_address_access,
    unwrap_surface_value,
    wrap_surface_value,
)
from ._types import _resolve

from ptoas.mlir.dialects import arith
from ptoas.mlir.dialects import llvm
from ptoas.mlir.dialects import math
from ptoas.mlir.ir import IndexType, IntegerType, MemRefType, Operation, VectorType
from ptoas.mlir.dialects import pto as _pto


def muli(lhs, rhs):
    """arith.muli"""
    return wrap_surface_value(emit_runtime_binary_op("mul", unwrap_surface_value(lhs), unwrap_surface_value(rhs)))


def addi(lhs, rhs):
    """arith.addi"""
    return wrap_surface_value(emit_runtime_binary_op("add", unwrap_surface_value(lhs), unwrap_surface_value(rhs)))


def subi(lhs, rhs):
    """arith.subi"""
    return wrap_surface_value(emit_runtime_binary_op("sub", unwrap_surface_value(lhs), unwrap_surface_value(rhs)))


def index_cast(type_or_val, val=None):
    """
    arith.index_cast.

    Two calling conventions::

        index_cast(result_type, value)   # explicit result type
        index_cast(value)                # result type = index (1-arg shorthand)
    """
    if val is None:
        # 1-arg form: cast to index
        return wrap_surface_value(arith.IndexCastOp(IndexType.get(), unwrap_surface_value(type_or_val)).result)
    return wrap_surface_value(arith.IndexCastOp(_resolve(type_or_val), unwrap_surface_value(val)).result)


def select(cond, true_val, false_val):
    """arith.select"""
    return wrap_surface_value(arith.SelectOp(
        unwrap_surface_value(cond),
        unwrap_surface_value(true_val),
        unwrap_surface_value(false_val),
    ).result)


def cast(value, dtype):
    """Cast a runtime scalar or builtin vector to ``dtype``.

    For builtin vectors, a scalar ``dtype`` denotes the destination element
    type and the source vector shape is preserved.  An explicit ``pto.Vec``
    descriptor is also accepted when its shape matches the source.
    """
    raw_value = unwrap_surface_value(value)
    target_type = _resolve(dtype)
    source_is_vector = hasattr(raw_value, "type") and VectorType.isinstance(raw_value.type)
    target_is_vector = VectorType.isinstance(target_type)

    if source_is_vector:
        source_vector_type = VectorType(raw_value.type)
        if target_is_vector:
            target_vector_type = VectorType(target_type)
            if tuple(source_vector_type.shape) != tuple(target_vector_type.shape):
                raise TypeError(
                    "scalar.cast(vector, dtype) requires the destination vector shape "
                    f"to match the source: got {target_type}, expected shape "
                    f"{tuple(source_vector_type.shape)}"
                )
        else:
            target_type = VectorType.get(list(source_vector_type.shape), target_type)
    elif target_is_vector:
        raise TypeError(
            "scalar.cast(scalar, dtype) cannot produce a vector; "
            "use pto.Vec(..., init=value) to broadcast"
        )

    return wrap_surface_value(
        coerce_scalar_to_type(value, target_type, context="scalar.cast(...)")
    )


def max(lhs, rhs):
    """Runtime scalar maximum across float / integer / index values."""
    return wrap_surface_value(emit_runtime_max(
        unwrap_surface_value(lhs),
        unwrap_surface_value(rhs),
    ))


def min(lhs, rhs):
    """Runtime scalar minimum across float / integer / index values."""
    return wrap_surface_value(emit_runtime_min(
        unwrap_surface_value(lhs),
        unwrap_surface_value(rhs),
    ))


def exp(value):
    """Runtime scalar exponential for floating-point values."""
    raw_value = unwrap_surface_value(value)
    kind = classify_runtime_scalar_type(raw_value.type)
    if kind != "float":
        raise TypeError(f"scalar.exp(...) expects a floating-point runtime scalar, got {raw_value.type}")
    return wrap_surface_value(math.ExpOp(raw_value).result)


def log(value):
    """Runtime scalar natural logarithm for floating-point values."""
    raw_value = unwrap_surface_value(value)
    kind = classify_runtime_scalar_type(raw_value.type)
    if kind != "float":
        raise TypeError(f"scalar.log(...) expects a floating-point runtime scalar, got {raw_value.type}")
    return wrap_surface_value(math.LogOp(raw_value).result)


def sqrt(value):
    """Runtime scalar square root for floating-point values."""
    raw_value = unwrap_surface_value(value)
    kind = classify_runtime_scalar_type(raw_value.type)
    if kind != "float":
        raise TypeError(f"scalar.sqrt(...) expects a floating-point runtime scalar, got {raw_value.type}")
    return wrap_surface_value(math.SqrtOp(raw_value).result)


def abs(value):
    """Runtime scalar absolute value across float / integer / index values."""
    return wrap_surface_value(emit_runtime_abs(unwrap_surface_value(value)))


def load(ptr_or_ref, offset=None, *, contiguous=None):
    """Load one scalar element or a contiguous builtin vector from a PTODSL address view."""
    width = _normalize_contiguous(contiguous, context="scalar.load(...)")
    allocated_buffer = _allocated_buffer_target(ptr_or_ref)
    buffer_value, index_value = resolve_address_access(ptr_or_ref, offset)
    result_type = _infer_buffer_element_type(buffer_value.type, allocated_buffer=allocated_buffer)
    if width > 1:
        return VecValue(_emit_contiguous_load(buffer_value, index_value, result_type, width))
    if allocated_buffer is not None:
        ptr_value = _emit_llvm_byte_pointer(buffer_value, index_value, result_type)
        return wrap_surface_value(llvm.LoadOp(result_type, ptr_value).res)
    return wrap_surface_value(Operation.create(
        "pto.load",
        results=[result_type],
        operands=[buffer_value, index_value],
    ).results[0])


def store(value, ptr_or_ref, offset=None, *, contiguous=None):
    """Store one scalar element or a builtin vector to a PTODSL address view."""
    allocated_buffer = _allocated_buffer_target(ptr_or_ref)
    buffer_value, index_value = resolve_address_access(ptr_or_ref, offset)
    elem_type = _infer_buffer_element_type(buffer_value.type, allocated_buffer=allocated_buffer)
    raw_value = unwrap_surface_value(value)
    if hasattr(raw_value, "type") and VectorType.isinstance(raw_value.type):
        vec_value = value if isinstance(value, VecValue) else VecValue(raw_value)
        width = _normalize_contiguous(contiguous, context="scalar.store(...)", default=vec_value.size)
        if width != vec_value.size:
            raise ValueError(
                f"scalar.store(..., contiguous={width}) does not match vector size {vec_value.size}"
            )
        if not _vector_store_element_compatible(vec_value.element_type, elem_type):
            raise TypeError(
                "scalar.store(vector, ...) element type must be bit-compatible with the "
                "destination pointer element type: "
                f"got {vec_value.element_type}, expected {elem_type}"
            )
        _emit_contiguous_store(raw_value, buffer_value, index_value)
        return

    width = _normalize_contiguous(contiguous, context="scalar.store(...)")
    if width > 1:
        raise TypeError("scalar.store(scalar, ..., contiguous=N) is not supported; pass a vector value")
    coerced_value = coerce_scalar_to_type(value, elem_type, context="scalar.store(...)")
    if allocated_buffer is not None:
        ptr_value = _emit_llvm_byte_pointer(buffer_value, index_value, elem_type)
        llvm.StoreOp(coerced_value, ptr_value)
        return
    Operation.create(
        "pto.store",
        operands=[buffer_value, index_value, coerced_value],
    )


def _normalize_contiguous(contiguous, *, context: str, default: int = 1) -> int:
    if contiguous is None:
        return default
    if isinstance(contiguous, bool) or not isinstance(contiguous, int):
        raise TypeError(f"{context} expects contiguous to be a positive Python integer")
    if contiguous <= 0:
        raise ValueError(f"{context} expects contiguous to be positive")
    return contiguous


def _allocated_buffer_target(target):
    if isinstance(target, AllocatedBufferValue):
        return target
    if isinstance(target, AddressOffsetValue) and isinstance(target.base, AllocatedBufferValue):
        return target.base
    return None


def _is_local_allocated_buffer(allocated_buffer) -> bool:
    return allocated_buffer is not None


def _infer_buffer_element_type(buffer_type, *, allocated_buffer=None):
    if allocated_buffer is not None:
        return allocated_buffer.element_type
    try:
        return _pto.PtrType(buffer_type).element_type
    except Exception:
        return MemRefType(buffer_type).element_type


def _emit_contiguous_load(buffer_value, index_value, elem_type, width: int):
    vector_type = VectorType.get([width], elem_type)
    ptr_value = _emit_llvm_byte_pointer(buffer_value, index_value, elem_type)
    return llvm.LoadOp(vector_type, ptr_value).res


def _vector_store_element_compatible(vector_element_type, elem_type) -> bool:
    """Return whether a vector element type and a destination element type share one bit layout.

    Exact type equality always passes. Same-width integers also pass: signed / unsigned /
    signless forms of one width store the same bits, mirroring the scalar coercion rules
    (for example a signless ``i32`` builtin vector packed from ``si32``/``ui32`` scalars).

    This relaxation is store-path only: ``scalar.load(..., contiguous=N)`` keeps the declared
    pointer element type (e.g. a ``ui32`` pointer yields ``vector<2xui32>``), while
    ``pto.Vec(...)`` builds signless vectors. Stores accept both because the bit layout is
    identical; element-level APIs (extract/insert/arithmetic) may still see the type
    mismatch and should be revisited if such APIs are added.
    """
    if vector_element_type == elem_type:
        return True
    if IntegerType.isinstance(vector_element_type) and IntegerType.isinstance(elem_type):
        if IntegerType(vector_element_type).width == IntegerType(elem_type).width:
            return True
    return False


def _emit_contiguous_store(vector_value, buffer_value, index_value):
    elem_type = VectorType(vector_value.type).element_type
    ptr_value = _emit_llvm_byte_pointer(buffer_value, index_value, elem_type)
    llvm.StoreOp(vector_value, ptr_value)


def _emit_llvm_byte_pointer(buffer_value, index_value, elem_type):
    byte_offset = _emit_byte_offset(index_value, elem_type)
    llvm_ptr_type = _as_llvm_ptr_type(buffer_value.type)
    if llvm_ptr_type is not None:
        return _emit_llvm_gep(
            llvm_ptr_type,
            buffer_value,
            [byte_offset],
            [-2147483648],
            IntegerType.get_signless(8),
        )

    pto_ptr_type = _as_pto_ptr_type(buffer_value.type)
    i64 = IntegerType.get_signless(64)
    addr_as_i64 = _pto.CastPtrOp(i64, buffer_value).result
    llvm_ptr_type = llvm.PointerType.get(_pto_ptr_llvm_address_space(pto_ptr_type))
    llvm_base = llvm.IntToPtrOp(llvm_ptr_type, addr_as_i64).res
    return _emit_llvm_gep(
        llvm_ptr_type,
        llvm_base,
        [byte_offset],
        [-2147483648],
        IntegerType.get_signless(8),
    )


def _emit_llvm_gep(result_type, base, dynamic_indices, raw_constant_indices, elem_type):
    try:
        return llvm.GEPOp(
            result_type,
            base,
            dynamic_indices,
            raw_constant_indices,
            elem_type,
            None,
        ).res
    except TypeError as exc:
        if "positional" not in str(exc) and "argument" not in str(exc):
            raise
        return llvm.GEPOp(
            result_type,
            base,
            dynamic_indices,
            raw_constant_indices,
            elem_type,
        ).res


def _as_llvm_ptr_type(type_obj):
    try:
        return llvm.PointerType(type_obj)
    except Exception:
        return None


def _emit_byte_offset(index_value, elem_type):
    bytewidth = _element_bytewidth(elem_type)
    bytewidth_const = arith.ConstantOp(IndexType.get(), bytewidth).result
    byte_index = arith.MulIOp(index_value, bytewidth_const).result
    return arith.IndexCastOp(IntegerType.get_signless(64), byte_index).result


def _as_pto_ptr_type(type_obj):
    try:
        return _pto.PtrType(type_obj)
    except Exception as exc:
        raise TypeError(
            "contiguous scalar.load/store currently expects a PTO pointer-backed address"
        ) from exc


def _pto_ptr_llvm_address_space(ptr_type) -> int:
    memory_space = getattr(ptr_type, "memory_space", None)
    value = getattr(memory_space, "value", None)
    if value is not None:
        return int(value)
    text = str(ptr_type)
    if ", ub>" in text or ", vec>" in text:
        return 6
    if ", gm>" in text or text.endswith(">"):
        return 1
    raise TypeError(f"unable to infer LLVM address space for pointer type {ptr_type}")


def _element_bytewidth(elem_type):
    if str(elem_type) == "f32":
        return 4
    if str(elem_type) in {"f16", "bf16"}:
        return 2
    if IntegerType.isinstance(elem_type):
        width = IntegerType(elem_type).width
        if width % 8 != 0:
            raise TypeError(f"unsupported sub-byte integer element type {elem_type}")
        return width // 8
    if str(elem_type).startswith("f8") or str(elem_type).startswith("!pto."):
        return 1
    raise TypeError(f"unsupported element type {elem_type}")


__all__ = [
    "muli", "addi", "subi",
    "index_cast",
    "cast",
    "select",
    "max", "min", "exp", "log", "sqrt", "abs",
    "load", "store",
]
