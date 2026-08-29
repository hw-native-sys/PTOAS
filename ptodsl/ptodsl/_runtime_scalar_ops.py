# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Tracing-time authored scalar operator lowering for runtime values."""

from __future__ import annotations

from ._scalar_adaptation import (
    _integer_signedness,
    _restore_authored_integer_type,
    _signedness_attr,
    _signless_integer_type,
    _to_common_integer_value,
    classify_runtime_scalar_type,
    normalize_runtime_binary_operands,
)
from ptoas.mlir.ir import Attribute, IntegerType, Operation, VectorType


_SHORT_CMP_PREDICATES = {"eq", "ne", "lt", "le", "gt", "ge"}
_INTEGER_CMP_PREDICATES = _SHORT_CMP_PREDICATES
_FLOAT_CMP_PREDICATES = _SHORT_CMP_PREDICATES | {
    "false", "oeq", "ogt", "oge", "olt", "ole", "one", "ord",
    "ueq", "ugt", "uge", "ult", "ule", "une", "uno", "true",
}


def emit_runtime_binary_op(op_name: str, lhs, rhs, *, attributes=None):
    """Lower one authored runtime scalar binary operator."""
    lhs, rhs, kind = normalize_runtime_binary_operands(
        lhs, rhs, require_matching_signedness=op_name in
        {"truediv", "floordiv", "ceildiv", "mod", "shr"})
    attributes = dict(attributes or {})
    if kind in {"index", "integer"}:
        if "fastmath" in attributes:
            raise TypeError(
                f"runtime scalar operator '{op_name}' does not accept fastmath for integer/index values"
            )
        mnemonic = {
            "add": "addi",
            "sub": "subi",
            "mul": "muli",
        }.get(op_name)
        if op_name == "shl" and kind == "integer":
            mnemonic = "shl"
        if op_name in {"truediv", "floordiv", "ceildiv", "mod", "shr"}:
            signedness = "signed" if kind == "index" else _integer_signedness(lhs.type)
            mnemonic = _SIGNED_INTEGER_BINARY_OPS[op_name]
            attributes["signedness"] = _signedness_attr(signedness)
        if mnemonic is None:
            raise TypeError(f"runtime scalar operator '{op_name}' is not supported for integer/index values")
        return _emit_binary_pto_op(mnemonic, lhs, rhs, attributes=attributes)
    if kind == "float":
        if "overflowFlags" in attributes:
            raise TypeError(
                f"runtime scalar operator '{op_name}' does not accept overflow for floating-point values"
            )
        mnemonic = {
            "add": "addf",
            "sub": "subf",
            "mul": "mulf",
            "truediv": "divf",
            "mod": "remf",
        }.get(op_name)
        if mnemonic is None:
            raise TypeError(f"runtime scalar operator '{op_name}' is not supported for floating-point values")
        return _emit_binary_pto_op(mnemonic, lhs, rhs, attributes=attributes)
    raise TypeError(f"unsupported runtime scalar operand category '{kind}'")


def emit_runtime_unary_op(op_name: str, value, *, attributes=None):
    """Lower one authored runtime scalar unary operator."""
    kind = classify_runtime_scalar_type(value.type)
    if op_name != "neg" or kind not in {"index", "integer", "float"}:
        raise TypeError(
            f"runtime scalar unary operator '{op_name}' is not supported for {value.type}"
        )
    mnemonic = "negf" if kind == "float" else "negi"
    attributes = dict(attributes or {})
    if kind == "float" and "overflowFlags" in attributes:
        raise TypeError("floating-point negation does not accept overflow")
    if kind != "float" and "fastmath" in attributes:
        raise TypeError("integer/index negation does not accept fastmath")
    result = Operation.create(
        f"pto.{mnemonic}",
        results=[_signless_integer_type(value.type)],
        operands=[_to_common_integer_value(value)],
        attributes=attributes,
    ).results[0]
    return _restore_authored_integer_type(result, value.type)


def _emit_binary_pto_op(mnemonic: str, lhs, rhs, *, attributes=None):
    authored_type = lhs.type
    common_type = _signless_integer_type(authored_type)
    result = Operation.create(
        f"pto.{mnemonic}", results=[common_type],
        operands=[_to_common_integer_value(lhs), _to_common_integer_value(rhs)],
        attributes=attributes or {},
    ).results[0]
    return _restore_authored_integer_type(result, authored_type)


_SIGNED_INTEGER_BINARY_OPS = {
    "truediv": "divi",
    "floordiv": "floordiv",
    "ceildiv": "ceildiv",
    "mod": "remi",
    "shr": "shr",
}


def emit_runtime_compare(op_name: str, lhs, rhs, *, attributes=None):
    """Lower one authored runtime scalar comparison operator."""
    lhs, rhs, kind = normalize_runtime_binary_operands(
        lhs, rhs, require_matching_signedness=True)
    if kind == "float":
        supported_predicates = _FLOAT_CMP_PREDICATES
    else:
        signedness = "signed" if kind == "index" else _integer_signedness(lhs.type)
        supported_predicates = _INTEGER_CMP_PREDICATES
    if op_name not in supported_predicates:
        raise TypeError(f"runtime scalar comparison '{op_name}' is not supported")
    predicate = Attribute.parse(f"#pto.scalar_cmp_predicate<{op_name}>")
    result_type = IntegerType.get_signless(1)
    if VectorType.isinstance(lhs.type):
        vector_type = VectorType(lhs.type)
        result_type = VectorType.get(
            list(vector_type.shape),
            result_type,
            scalable=list(vector_type.scalable_dims),
        )
    op_attributes = dict(attributes or {})
    op_attributes["predicate"] = predicate
    if kind != "float":
        op_attributes["signedness"] = _signedness_attr(signedness)
    return Operation.create(
        "pto.cmpf" if kind == "float" else "pto.cmpi",
        results=[result_type],
        operands=[_to_common_integer_value(lhs), _to_common_integer_value(rhs)],
        attributes=op_attributes,
    ).results[0]


def emit_runtime_bitwise_op(op_name: str, lhs, rhs):
    """Lower one authored runtime scalar bitwise operator."""
    lhs, rhs, kind = normalize_runtime_binary_operands(lhs, rhs)
    if op_name not in {"and", "or", "xor"}:
        raise TypeError(f"unsupported runtime scalar bitwise operator '{op_name}'")

    if kind == "index":
        return _emit_binary_pto_op(op_name, lhs, rhs)

    if kind != "integer":
        raise TypeError(
            f"runtime scalar bitwise operator '{op_name}' expects integer-like operands, got {lhs.type} and {rhs.type}"
        )

    return _emit_binary_pto_op(op_name, lhs, rhs)


__all__ = [
    "classify_runtime_scalar_type",
    "emit_runtime_binary_op",
    "emit_runtime_unary_op",
    "emit_runtime_compare",
    "emit_runtime_bitwise_op",
    "normalize_runtime_binary_operands",
]
