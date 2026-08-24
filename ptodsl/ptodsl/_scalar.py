# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
Implementation of the public ``pto.*`` scalar value and memory helpers.

This module is an internal implementation module. User code should import
``pto`` and use its unified public helpers.

Arithmetic helpers operate on raw ``ptoas.mlir.ir.Value`` objects and emit
frontend ``pto.*`` scalar operations at the active insertion point. PTOAS
legalizes those operations to arith/math/LLVM after execution-scope validation.
Scalar memory helpers (`load` / `store`) also accept PTODSL surface-level
address views such as `tile[row, col]` and `tile.as_ptr() + offset`.
"""

from ._scalar_coercion import coerce_scalar_to_type
from ._scalar_adaptation import (
    _integer_signedness,
    _pto_cast,
    _restore_authored_integer_type,
    _signedness_attr,
    _signless_integer_type,
    _to_common_integer_value,
    classify_runtime_scalar_type,
)
from ._runtime_scalar_ops import (
    emit_runtime_binary_op,
    emit_runtime_compare,
    emit_runtime_unary_op,
    normalize_runtime_binary_operands,
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

from ptoas.mlir.dialects import llvm
from ptoas.mlir.ir import (
    Attribute,
    IndexType,
    IntegerAttr,
    IntegerType,
    MemRefType,
    Operation,
    VectorType,
)
from ptoas.mlir.dialects import pto as _pto


def mul(lhs, rhs, *, overflow=None, fastmath=None):
    """Multiply runtime values and select integer or floating PTO IR."""
    return wrap_surface_value(emit_runtime_binary_op(
        "mul", unwrap_surface_value(lhs), unwrap_surface_value(rhs),
        attributes=_arithmetic_attrs(overflow=overflow, fastmath=fastmath),
    ))


def add(lhs, rhs, *, overflow=None, fastmath=None):
    """Add runtime values and select integer or floating PTO IR."""
    return wrap_surface_value(emit_runtime_binary_op(
        "add", unwrap_surface_value(lhs), unwrap_surface_value(rhs),
        attributes=_arithmetic_attrs(overflow=overflow, fastmath=fastmath),
    ))


def sub(lhs, rhs, *, overflow=None, fastmath=None):
    """Subtract runtime values and select integer or floating PTO IR."""
    return wrap_surface_value(emit_runtime_binary_op(
        "sub", unwrap_surface_value(lhs), unwrap_surface_value(rhs),
        attributes=_arithmetic_attrs(overflow=overflow, fastmath=fastmath),
    ))


def neg(value, *, overflow=None, fastmath=None):
    """Frontend ``pto.neg``."""
    return wrap_surface_value(emit_runtime_unary_op(
        "neg", unwrap_surface_value(value),
        attributes=_arithmetic_attrs(overflow=overflow, fastmath=fastmath),
    ))


def ceildiv(lhs, rhs):
    """Frontend integer ``pto.ceildiv``."""
    return wrap_surface_value(emit_runtime_binary_op("ceildiv", unwrap_surface_value(lhs), unwrap_surface_value(rhs)))


def div(lhs, rhs, *, fastmath=None):
    """Divide runtime values and select integer or floating PTO IR."""
    return wrap_surface_value(emit_runtime_binary_op(
        "truediv", unwrap_surface_value(lhs), unwrap_surface_value(rhs),
        attributes=_arithmetic_attrs(fastmath=fastmath),
    ))


def floordiv(lhs, rhs):
    """Floor-divide integer or index runtime values."""
    return wrap_surface_value(emit_runtime_binary_op(
        "floordiv", unwrap_surface_value(lhs), unwrap_surface_value(rhs)
    ))


def rem(lhs, rhs, *, fastmath=None):
    """Compute integer or floating-point remainder."""
    return wrap_surface_value(emit_runtime_binary_op(
        "mod", unwrap_surface_value(lhs), unwrap_surface_value(rhs),
        attributes=_arithmetic_attrs(fastmath=fastmath),
    ))


def shl(lhs, rhs, *, overflow=None):
    """Frontend integer ``pto.shl``."""
    return wrap_surface_value(emit_runtime_binary_op(
        "shl", unwrap_surface_value(lhs), unwrap_surface_value(rhs),
        attributes=_arithmetic_attrs(overflow=overflow),
    ))


def shr(lhs, rhs):
    """Frontend signedness-aware integer ``pto.shr``."""
    return wrap_surface_value(emit_runtime_binary_op("shr", unwrap_surface_value(lhs), unwrap_surface_value(rhs)))


def select(cond, true_val, false_val):
    """Frontend ``pto.select`` lowered to ``arith.select`` by PTOAS."""
    raw_true = unwrap_surface_value(true_val)
    raw_false = unwrap_surface_value(false_val)
    result = Operation.create(
        "pto.select",
        results=[_signless_integer_type(raw_true.type)],
        operands=[
            unwrap_surface_value(cond),
            _to_common_integer_value(raw_true),
            _to_common_integer_value(raw_false),
        ],
    ).results[0]
    return wrap_surface_value(
        _restore_authored_integer_type(result, raw_true.type)
    )


def cast(value, dtype, *, rounding=None, saturation=None, overflow=None, fastmath=None):
    """Cast a runtime scalar or builtin vector to ``dtype``.

    For builtin vectors, a scalar ``dtype`` denotes the destination element
    type and the source vector shape is preserved.  An explicit ``pto.Vec``
    descriptor is also accepted when its shape matches the source.

    Index/integer conversions use the authored integer signedness and emit
    ``pto.index_cast``. Numeric conversion attributes are not valid for this
    conversion category.
    """
    raw_value = unwrap_surface_value(value)
    target_type = _resolve_shape_preserving_target(raw_value, dtype, "pto.cast")
    attrs = _arithmetic_attrs(
        rounding=rounding, saturation=saturation, overflow=overflow, fastmath=fastmath
    )

    if not hasattr(raw_value, "type"):
        if attrs:
            raise TypeError("pto.cast(...) semantic attributes require a traced runtime value")
        return wrap_surface_value(
            coerce_scalar_to_type(value, target_type, context="pto.cast(...)")
        )
    if raw_value.type == target_type:
        if attrs:
            raise ValueError(
                "pto.cast(...) semantic attributes require different source and destination types"
            )
        return wrap_surface_value(raw_value)
    return wrap_surface_value(_pto_cast(
        raw_value,
        target_type,
        attributes=attrs,
    ))


def bitcast(value, dtype):
    """Reinterpret a runtime scalar or builtin vector without changing its bits."""
    raw_value = unwrap_surface_value(value)
    if not hasattr(raw_value, "type"):
        raise TypeError("pto.bitcast(...) requires a traced runtime value")
    target_type = _resolve_shape_preserving_target(raw_value, dtype, "pto.bitcast")
    if raw_value.type == target_type:
        return wrap_surface_value(raw_value)
    common_target_type = _signless_integer_type(target_type)
    result = Operation.create(
        "pto.bitcast", results=[common_target_type],
        operands=[_to_common_integer_value(raw_value)]
    ).results[0]
    return wrap_surface_value(
        _restore_authored_integer_type(result, target_type)
    )


def cmp(lhs, rhs, predicate, *, fastmath=None):
    """Compare matching runtime values with an explicit PTO predicate."""
    if not isinstance(predicate, str):
        raise TypeError("pto.cmp(...) predicate must be a string")
    return wrap_surface_value(emit_runtime_compare(
        predicate, unwrap_surface_value(lhs), unwrap_surface_value(rhs),
        attributes=_arithmetic_attrs(fastmath=fastmath),
    ))


_OVERFLOW_FLAGS = {"nsw", "nuw"}
_FAST_MATH_FLAGS = {"reassoc", "nnan", "ninf", "nsz", "arcp", "contract", "afn", "fast"}
_ROUNDING_MODES = {
    "to_nearest_even": 0,
    "downward": 1,
    "upward": 2,
    "toward_zero": 3,
    "to_nearest_away": 4,
    "to_odd": 5,
    "hybrid": 6,
}
_ROUNDING_ALIASES = {
    "r": "to_nearest_even",
    "a": "to_nearest_away",
    "f": "downward",
    "c": "upward",
    "z": "toward_zero",
    "o": "to_odd",
    "h": "hybrid",
}
_SATURATION_MODES = {"sat", "nosat", "on", "off"}


def _normalize_flag_list(value, *, name, allowed):
    if value is None:
        return None
    values = [value] if isinstance(value, str) else list(value)
    if not values or any(flag not in allowed for flag in values):
        raise ValueError(f"{name} must contain only {sorted(allowed)}, got {value!r}")
    return values


def _arithmetic_attrs(*, overflow=None, fastmath=None, rounding=None, saturation=None):
    attrs = {}
    overflow_flags = _normalize_flag_list(
        overflow, name="overflow", allowed=_OVERFLOW_FLAGS
    )
    if overflow_flags:
        attrs["overflowFlags"] = Attribute.parse(
            f"#pto.overflow<{', '.join(overflow_flags)}>"
        )
    fastmath_flags = _normalize_flag_list(
        fastmath, name="fastmath", allowed=_FAST_MATH_FLAGS
    )
    if fastmath_flags:
        attrs["fastmath"] = Attribute.parse(
            f"#pto.fastmath<{','.join(fastmath_flags)}>"
        )
    if rounding is not None:
        mode = _ROUNDING_ALIASES.get(rounding, rounding)
        if mode not in _ROUNDING_MODES:
            raise ValueError(
                "rounding must be one of "
                f"{sorted((*_ROUNDING_MODES, *_ROUNDING_ALIASES))}, got {rounding!r}"
            )
        attrs["roundingmode"] = IntegerAttr.get(
            IntegerType.get_signless(32), _ROUNDING_MODES[mode]
        )
    if saturation is not None:
        if saturation not in _SATURATION_MODES:
            raise ValueError(
                "saturation must be one of "
                f"{sorted(_SATURATION_MODES)}, got {saturation!r}"
            )
        mode = {"on": "sat", "off": "nosat"}.get(saturation, saturation)
        attrs["saturation"] = Attribute.parse(f"#pto.saturation<{mode}>")
    return attrs


def _resolve_shape_preserving_target(raw_value, dtype, operation):
    target_type = _resolve(dtype)
    source_is_vector = hasattr(raw_value, "type") and VectorType.isinstance(raw_value.type)
    target_is_vector = VectorType.isinstance(target_type)
    if source_is_vector:
        source_vector_type = VectorType(raw_value.type)
        if target_is_vector:
            target_vector_type = VectorType(target_type)
            if (tuple(source_vector_type.shape) != tuple(target_vector_type.shape) or
                    tuple(source_vector_type.scalable_dims) !=
                    tuple(target_vector_type.scalable_dims)):
                raise TypeError(
                    f"{operation}(vector, dtype) requires the destination vector shape "
                    f"to match the source: got {target_type}, expected shape "
                    f"{tuple(source_vector_type.shape)}"
                )
        else:
            target_type = VectorType.get(
                list(source_vector_type.shape),
                target_type,
                scalable=list(source_vector_type.scalable_dims),
            )
    elif target_is_vector:
        raise TypeError(
            f"{operation}(scalar, dtype) cannot produce a vector; "
            "use pto.Vec(..., init=value) to broadcast"
        )
    return target_type


def max(lhs, rhs, *, fastmath=None):
    """Runtime scalar maximum.

    Floating-point operands use LLVM ``maxnum`` semantics; integer operands
    select explicit signed/unsigned operations; index uses signed ordering.
    """
    raw_lhs, raw_rhs, kind = normalize_runtime_binary_operands(
        unwrap_surface_value(lhs), unwrap_surface_value(rhs),
        require_matching_signedness=True)
    if fastmath is not None and kind != "float":
        raise TypeError("pto.max(..., fastmath=...) expects floating-point operands")
    return wrap_surface_value(
        _emit_extremum("max", raw_lhs, raw_rhs, kind, fastmath=fastmath)
    )


def min(lhs, rhs, *, fastmath=None):
    """Runtime scalar minimum.

    Floating-point operands use LLVM ``minnum`` semantics; integer operands
    select explicit signed/unsigned operations; index uses signed ordering.
    """
    raw_lhs, raw_rhs, kind = normalize_runtime_binary_operands(
        unwrap_surface_value(lhs), unwrap_surface_value(rhs),
        require_matching_signedness=True)
    if fastmath is not None and kind != "float":
        raise TypeError("pto.min(..., fastmath=...) expects floating-point operands")
    return wrap_surface_value(
        _emit_extremum("min", raw_lhs, raw_rhs, kind, fastmath=fastmath)
    )


def _emit_extremum(mnemonic, lhs, rhs, kind, *, fastmath=None):
    authored_type = lhs.type
    attrs = _arithmetic_attrs(fastmath=fastmath)
    if kind != "float":
        attrs["signedness"] = _signedness_attr(_integer_signedness(authored_type))
    result = Operation.create(
        f"pto.{mnemonic}{'f' if kind == 'float' else 'i'}",
        results=[_signless_integer_type(authored_type)],
        operands=[_to_common_integer_value(lhs), _to_common_integer_value(rhs)],
        attributes=attrs,
    ).results[0]
    return _restore_authored_integer_type(result, authored_type)


def _float_extremum(mnemonic, lhs, rhs, *, fastmath=None):
    raw_lhs, raw_rhs, kind = normalize_runtime_binary_operands(
        unwrap_surface_value(lhs), unwrap_surface_value(rhs),
        require_matching_signedness=True)
    if kind != "float":
        raise TypeError(f"pto.{mnemonic}(...) expects floating-point operands")
    return wrap_surface_value(Operation.create(
        f"pto.{mnemonic}", results=[raw_lhs.type], operands=[raw_lhs, raw_rhs],
        attributes=_arithmetic_attrs(fastmath=fastmath),
    ).results[0])


def maximum(lhs, rhs, *, fastmath=None):
    """NaN-propagating floating-point maximum."""
    return _float_extremum("maximum", lhs, rhs, fastmath=fastmath)


def minimum(lhs, rhs, *, fastmath=None):
    """NaN-propagating floating-point minimum."""
    return _float_extremum("minimum", lhs, rhs, fastmath=fastmath)


def addui_extended(lhs, rhs):
    """Return unsigned wrapped sum and overflow bit."""
    raw_lhs, raw_rhs, kind = normalize_runtime_binary_operands(
        unwrap_surface_value(lhs), unwrap_surface_value(rhs)
    )
    if kind not in {"integer", "index"}:
        raise TypeError("pto.addui_extended(...) expects integer or index operands")
    bool_type = IntegerType.get_signless(1)
    if VectorType.isinstance(raw_lhs.type):
        vector_type = VectorType(raw_lhs.type)
        bool_type = VectorType.get(
            list(vector_type.shape), bool_type,
            scalable=list(vector_type.scalable_dims),
        )
    authored_type = raw_lhs.type
    op = Operation.create(
        "pto.addui_extended",
        results=[_signless_integer_type(authored_type), bool_type],
        operands=[_to_common_integer_value(raw_lhs), _to_common_integer_value(raw_rhs)],
    )
    return (
        wrap_surface_value(_restore_authored_integer_type(op.results[0], authored_type)),
        wrap_surface_value(op.results[1]),
    )


def mul_extended(lhs, rhs, *, signedness=None):
    raw_lhs, raw_rhs, kind = normalize_runtime_binary_operands(
        unwrap_surface_value(lhs), unwrap_surface_value(rhs)
    )
    if kind not in {"integer", "index"}:
        raise TypeError("pto.mul_extended(...) expects integer or index operands")
    signedness = signedness or _integer_signedness(raw_lhs.type)
    authored_type = raw_lhs.type
    op = Operation.create(
        "pto.mul_extended",
        results=[_signless_integer_type(authored_type)] * 2,
        operands=[_to_common_integer_value(raw_lhs), _to_common_integer_value(raw_rhs)],
        attributes={"signedness": _signedness_attr(signedness)},
    )
    return tuple(
        wrap_surface_value(_restore_authored_integer_type(result, authored_type))
        for result in op.results
    )


def exp(value):
    """Runtime scalar exponential for floating-point values."""
    raw_value = unwrap_surface_value(value)
    kind = classify_runtime_scalar_type(raw_value.type)
    if not VectorType.isinstance(raw_value.type) and kind != "float":
        raise TypeError(f"pto.exp(...) expects a floating-point runtime scalar, got {raw_value.type}")
    return wrap_surface_value(_pto.ExpOp(raw_value).result)


def log(value):
    """Runtime scalar natural logarithm for floating-point values."""
    raw_value = unwrap_surface_value(value)
    kind = classify_runtime_scalar_type(raw_value.type)
    if not VectorType.isinstance(raw_value.type) and kind != "float":
        raise TypeError(f"pto.log(...) expects a floating-point runtime scalar, got {raw_value.type}")
    return wrap_surface_value(_pto.LogOp(raw_value).result)


def sqrt(value):
    """Runtime scalar square root for floating-point values."""
    raw_value = unwrap_surface_value(value)
    kind = classify_runtime_scalar_type(raw_value.type)
    if not VectorType.isinstance(raw_value.type) and kind != "float":
        raise TypeError(f"pto.sqrt(...) expects a floating-point runtime scalar, got {raw_value.type}")
    return wrap_surface_value(_pto.SqrtOp(raw_value).result)


def abs(value):
    """Runtime scalar absolute value across float / integer / index values."""
    raw_value = unwrap_surface_value(value)
    kind = classify_runtime_scalar_type(raw_value.type)
    attrs = {}
    if kind != "float":
        attrs["signedness"] = _signedness_attr(_integer_signedness(raw_value.type))
    result = Operation.create(
        "pto.absf" if kind == "float" else "pto.absi",
        results=[_signless_integer_type(raw_value.type)],
        operands=[_to_common_integer_value(raw_value)],
        attributes=attrs,
    ).results[0]
    return wrap_surface_value(
        _restore_authored_integer_type(result, raw_value.type)
    )


def load(ptr_or_ref, offset=None, *, contiguous=None):
    """Load a scalar or contiguous builtin vector from a PTODSL address view."""
    width = _normalize_contiguous(contiguous, context="pto.load(...)")
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
        width = _normalize_contiguous(contiguous, context="pto.store(...)", default=vec_value.size)
        if width != vec_value.size:
            raise ValueError(
                f"pto.store(..., contiguous={width}) does not match vector size {vec_value.size}"
            )
        if not _vector_store_element_compatible(vec_value.element_type, elem_type):
            raise TypeError(
                "pto.store(vector, ...) element type must be bit-compatible with the "
                "destination pointer element type: "
                f"got {vec_value.element_type}, expected {elem_type}"
            )
        _emit_contiguous_store(raw_value, buffer_value, index_value)
        return

    width = _normalize_contiguous(contiguous, context="pto.store(...)")
    if width > 1:
        raise TypeError("pto.store(scalar, ..., contiguous=N) is not supported; pass a vector value")
    coerced_value = coerce_scalar_to_type(value, elem_type, context="pto.store(...)")
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
    pto_pointer = _as_pto_ptr_type(buffer_value.type)
    if pto_pointer is not None:
        vector_pointer, zero = _emit_pto_contiguous_pointer(
            buffer_value, index_value, elem_type, vector_type, pto_pointer
        )
        return Operation.create(
            "pto.load",
            results=[vector_type],
            operands=[vector_pointer, zero],
        ).results[0]
    ptr_value = _emit_llvm_byte_pointer(buffer_value, index_value, elem_type)
    return llvm.LoadOp(vector_type, ptr_value).res


def _vector_store_element_compatible(vector_element_type, elem_type) -> bool:
    """Return whether vector and destination elements have the same bit layout."""
    if vector_element_type == elem_type:
        return True
    if IntegerType.isinstance(vector_element_type) and IntegerType.isinstance(elem_type):
        return IntegerType(vector_element_type).width == IntegerType(elem_type).width
    return False


def _emit_contiguous_store(vector_value, buffer_value, index_value):
    elem_type = VectorType(vector_value.type).element_type
    pto_pointer = _as_pto_ptr_type(buffer_value.type)
    if pto_pointer is not None:
        vector_pointer, zero = _emit_pto_contiguous_pointer(
            buffer_value,
            index_value,
            elem_type,
            vector_value.type,
            pto_pointer,
        )
        Operation.create(
            "pto.store",
            operands=[vector_pointer, zero, vector_value],
        )
        return
    ptr_value = _emit_llvm_byte_pointer(buffer_value, index_value, elem_type)
    llvm.StoreOp(vector_value, ptr_value)


def _emit_pto_contiguous_pointer(
    buffer_value, index_value, elem_type, vector_type, pto_pointer
):
    """Retype one PTO pointer at an authored scalar-element offset.

    A vector-typed PTO pointer advances by one whole vector, while the
    ``pto.load/store(..., contiguous=N)`` offset is authored in scalar
    elements.  Apply the scalar byte offset before retyping the pointer, then
    access element zero through the vector pointer.
    """
    i64 = IntegerType.get_signless(64)
    base_address = _pto.CastPtrOp(i64, buffer_value).result
    byte_offset = _emit_byte_offset(index_value, elem_type)
    vector_address = Operation.create(
        "pto.addi", results=[i64], operands=[base_address, byte_offset]
    ).results[0]
    vector_pointer_type = _pto.PtrType.get(
        vector_type, memory_space=pto_pointer.memory_space
    )
    vector_pointer = _pto.CastPtrOp(vector_pointer_type, vector_address).result
    index_type = IndexType.get()
    zero = Operation.create(
        "pto.constant",
        results=[index_type],
        attributes={"value": IntegerAttr.get(index_type, 0)},
    ).results[0]
    return vector_pointer, zero


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
    index_type = IndexType.get()
    bytewidth_const = Operation.create(
        "pto.constant",
        results=[index_type],
        attributes={"value": IntegerAttr.get(index_type, bytewidth)},
    ).results[0]
    byte_index = Operation.create(
        "pto.muli", results=[index_type], operands=[index_value, bytewidth_const]
    ).results[0]
    i64 = IntegerType.get_signless(64)
    return _pto_cast(byte_index, i64, index=True)


def _as_pto_ptr_type(type_obj):
    try:
        return _pto.PtrType(type_obj)
    except Exception:
        return None


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
    "mul", "add", "sub", "neg", "div", "floordiv", "ceildiv", "rem", "shl", "shr",
    "cast",
    "bitcast",
    "cmp",
    "select",
    "max", "min", "maximum", "minimum",
    "addui_extended", "mul_extended",
    "exp", "log", "sqrt", "abs",
    "load", "store",
]
