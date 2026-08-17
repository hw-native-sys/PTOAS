# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Shared scalar adaptation helpers for PTODSL frontend lowering."""

from __future__ import annotations

from ._types import (
    _integer_signedness,
    _materialize_integer_literal,
    _restore_integer_signedness,
    _signless_integer_type,
    _strip_integer_signedness,
)

from ptoas.mlir.dialects import arith
from ptoas.mlir.ir import BF16Type, F16Type, F32Type, FloatAttr, IndexType, IntegerType, VectorType


def classify_runtime_scalar_type(type_obj):
    if IndexType.isinstance(type_obj):
        return "index"
    if IntegerType.isinstance(type_obj):
        return "integer"
    if any(cls.isinstance(type_obj) for cls in (BF16Type, F16Type, F32Type)):
        return "float"
    if VectorType.isinstance(type_obj):
        element_type = VectorType(type_obj).element_type
        if any(cls.isinstance(element_type) for cls in (BF16Type, F16Type, F32Type)):
            return "float"
    raise TypeError(f"runtime scalar operators only support index/int/float values, got {type_obj}")


def is_mlir_value(value) -> bool:
    return not isinstance(value, (bool, int, float)) and hasattr(value, "type")


def materialize_scalar_literal(value, target_type, *, context: str):
    """Materialize one Python literal as an MLIR scalar constant of *target_type*."""
    if isinstance(value, bool):
        raise TypeError(f"{context} does not accept bool literals")

    target_kind = classify_runtime_scalar_type(target_type)
    if isinstance(value, float) and target_kind != "float":
        raise TypeError(
            f"{context} cannot materialize a floating-point literal against non-floating "
            f"target type {target_type}"
        )

    if target_kind == "float":
        return arith.ConstantOp(target_type, FloatAttr.get(target_type, float(value))).result
    if target_kind == "index":
        return arith.ConstantOp(target_type, int(value)).result

    return _materialize_integer_literal(target_type, value)


def coerce_scalar_value_to_type(value, target_type, *, context: str):
    """Normalize one already-unwrapped authored scalar value/literal to *target_type*."""
    if not hasattr(value, "type"):
        return materialize_scalar_literal(value, target_type, context=context)

    if value.type == target_type:
        return value

    source_kind = classify_runtime_scalar_type(value.type)
    target_kind = classify_runtime_scalar_type(target_type)

    if source_kind == "index" and target_kind == "integer":
        return coerce_integer_like(value, target_type)
    if source_kind == "integer" and target_kind == "index":
        return arith.IndexCastOp(target_type, _strip_integer_signedness(value)).result
    if source_kind == "integer" and target_kind == "integer":
        return coerce_integer_like(value, target_type)
    if source_kind == "float" and target_kind == "float":
        return _coerce_float_like(value, target_type)

    raise TypeError(
        f"{context} cannot coerce the authored value to the expected scalar type: "
        f"got {value.type}, expected {target_type}"
    )


def coerce_runtime_index_value(value, *, context: str):
    """Normalize one already-unwrapped authored value/literal to an MLIR index value."""
    if isinstance(value, bool):
        raise TypeError(f"{context} does not accept bool values")
    if isinstance(value, int):
        return arith.ConstantOp(IndexType.get(), value).result
    if not hasattr(value, "type"):
        raise TypeError(
            f"{context} expects a Python int, an index value, or an integer runtime scalar; "
            f"got {value!r}"
        )

    value_type = value.type
    if IndexType.isinstance(value_type):
        return value
    if IntegerType.isinstance(value_type):
        return arith.IndexCastOp(IndexType.get(), _strip_integer_signedness(value)).result

    raise TypeError(f"{context} expects an index or integer runtime scalar, got {value_type}")


def coerce_runtime_integer_value(value, target_type, *, context: str):
    """Normalize one authored integer-like value/literal to *target_type*."""
    if isinstance(value, bool):
        raise TypeError(f"{context} does not accept bool values")
    if isinstance(value, int):
        return _materialize_integer_literal(target_type, value)
    if not hasattr(value, "type"):
        raise TypeError(f"{context} expects an integer-like scalar, got {value!r}")

    kind = classify_runtime_scalar_type(value.type)
    if kind == "float":
        raise TypeError(f"{context} expects an integer-like scalar, got {value.type}")
    return coerce_integer_like(value, target_type)


def coerce_runtime_integer_to_i1(value, *, context: str):
    """Convert one runtime integer-like value to ``i1`` using nonzero truthiness."""
    if not hasattr(value, "type"):
        raise TypeError(f"{context} expects an integer-like runtime scalar, got {value!r}")

    if IndexType.isinstance(value.type):
        zero = arith.ConstantOp(IndexType.get(), 0).result
        return arith.CmpIOp(arith.CmpIPredicate.ne, value, zero).result

    if not IntegerType.isinstance(value.type):
        raise TypeError(f"{context} expects an integer-like runtime scalar, got {value.type}")

    signless_type = _signless_integer_type(value.type)
    signless_value = _strip_integer_signedness(value)
    if IntegerType(signless_type).width == 1:
        return signless_value
    zero = arith.ConstantOp(signless_type, 0).result
    return arith.CmpIOp(arith.CmpIPredicate.ne, signless_value, zero).result


def coerce_runtime_i1_value(value, *, context: str):
    """Normalize one authored bool/integer-like value/literal to signless i1."""
    i1_type = IntegerType.get_signless(1)
    if isinstance(value, bool):
        return _materialize_integer_literal(i1_type, int(value))
    if isinstance(value, int):
        if value not in (0, 1):
            raise ValueError(f"{context} expects a bool or 0/1 integer, got {value}")
        return _materialize_integer_literal(i1_type, value)
    if not hasattr(value, "type"):
        raise TypeError(f"{context} expects a bool or integer-like scalar, got {value!r}")

    kind = classify_runtime_scalar_type(value.type)
    if kind == "float":
        raise TypeError(f"{context} expects a bool or integer-like scalar, got {value.type}")
    return coerce_runtime_integer_to_i1(value, context=context)


def normalize_runtime_binary_operands(lhs, rhs):
    lhs_is_value = is_mlir_value(lhs)
    rhs_is_value = is_mlir_value(rhs)

    if not lhs_is_value and not rhs_is_value:
        raise TypeError("runtime scalar operators require at least one traced runtime operand")

    if lhs_is_value and rhs_is_value:
        return reconcile_typed_runtime_binary_operands(lhs, rhs)

    anchor_type = lhs.type if lhs_is_value else rhs.type
    lhs = lhs if lhs_is_value else _materialize_runtime_literal(lhs, anchor_type)
    rhs = rhs if rhs_is_value else _materialize_runtime_literal(rhs, anchor_type)
    return reconcile_typed_runtime_binary_operands(lhs, rhs)


def reconcile_typed_runtime_binary_operands(lhs, rhs):
    lhs_type = lhs.type
    rhs_type = rhs.type

    if lhs_type == rhs_type:
        return lhs, rhs, classify_runtime_scalar_type(lhs_type)

    if IndexType.isinstance(lhs_type) and IntegerType.isinstance(rhs_type):
        rhs = arith.IndexCastOp(IndexType.get(), _strip_integer_signedness(rhs)).result
        return lhs, rhs, "index"

    if IntegerType.isinstance(lhs_type) and IndexType.isinstance(rhs_type):
        lhs = arith.IndexCastOp(IndexType.get(), _strip_integer_signedness(lhs)).result
        return lhs, rhs, "index"

    if IntegerType.isinstance(lhs_type) and IntegerType.isinstance(rhs_type):
        lhs_width = IntegerType(lhs_type).width
        rhs_width = IntegerType(rhs_type).width
        target_type = lhs_type if lhs_width >= rhs_width else rhs_type
        lhs = coerce_integer_like(lhs, target_type)
        rhs = coerce_integer_like(rhs, target_type)
        return lhs, rhs, "integer"

    raise TypeError(
        "runtime scalar operators require matching scalar types or an index/integer pair; "
        f"got {lhs_type} and {rhs_type}"
    )


def coerce_integer_like(value, target_type):
    if IndexType.isinstance(value.type):
        signless_target = _signless_integer_type(target_type)
        adapted = arith.IndexCastOp(signless_target, value).result
        return _restore_integer_signedness(adapted, target_type)

    source_type = value.type
    source_width = IntegerType(source_type).width
    target_width = IntegerType(target_type).width
    signless_source = _strip_integer_signedness(value)
    signless_target = _signless_integer_type(target_type)

    if source_width < target_width:
        source_signedness = _integer_signedness(source_type)
        # i1 carries boolean truth values; widening must preserve 0/1 storage.
        if source_width == 1 or source_signedness == "unsigned":
            widened = arith.ExtUIOp(signless_target, signless_source).result
        else:
            widened = arith.ExtSIOp(signless_target, signless_source).result
        return _restore_integer_signedness(widened, target_type)
    if source_width > target_width:
        truncated = arith.TruncIOp(signless_target, signless_source).result
        return _restore_integer_signedness(truncated, target_type)
    return _restore_integer_signedness(signless_source, target_type)


def _materialize_runtime_literal(value, anchor_type):
    if isinstance(value, bool):
        raise TypeError("runtime scalar operators do not accept bool literals")

    kind = classify_runtime_scalar_type(anchor_type)
    if isinstance(value, float) and kind != "float":
        raise TypeError(
            "runtime scalar operators cannot materialize a floating-point literal "
            f"against non-floating operand type {anchor_type}"
        )

    if kind == "float":
        return arith.ConstantOp(anchor_type, FloatAttr.get(anchor_type, float(value))).result
    if kind == "index":
        return arith.ConstantOp(anchor_type, int(value)).result

    return _materialize_integer_literal(anchor_type, value)


def _coerce_float_like(value, target_type):
    if value.type == target_type:
        return value

    source_shape = _float_shape(value.type)
    target_shape = _float_shape(target_type)
    if source_shape != target_shape:
        raise TypeError(
            "cannot coerce floating-point values with different shapes: "
            f"{value.type} and {target_type}"
        )

    source_width = _float_bytewidth(value.type)
    target_width = _float_bytewidth(target_type)
    if source_width < target_width:
        return arith.ExtFOp(target_type, value).result
    if source_width > target_width:
        return arith.TruncFOp(target_type, value).result
    raise TypeError(
        "cannot coerce between different floating-point types of the same width: "
        f"{value.type} and {target_type}"
    )


def _float_shape(type_obj):
    if VectorType.isinstance(type_obj):
        return tuple(VectorType(type_obj).shape)
    return ()


def _float_bytewidth(type_obj):
    if VectorType.isinstance(type_obj):
        return _float_bytewidth(VectorType(type_obj).element_type)
    if BF16Type.isinstance(type_obj) or F16Type.isinstance(type_obj):
        return 2
    if F32Type.isinstance(type_obj):
        return 4
    raise TypeError(f"unsupported floating-point type {type_obj}")


__all__ = [
    "classify_runtime_scalar_type",
    "coerce_integer_like",
    "coerce_runtime_i1_value",
    "coerce_runtime_integer_to_i1",
    "coerce_runtime_index_value",
    "coerce_runtime_integer_value",
    "coerce_scalar_value_to_type",
    "is_mlir_value",
    "materialize_scalar_literal",
    "normalize_runtime_binary_operands",
    "reconcile_typed_runtime_binary_operands",
]
