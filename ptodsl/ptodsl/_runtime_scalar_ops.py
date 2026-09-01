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
    classify_runtime_scalar_type,
    normalize_runtime_binary_operands,
)
from ._types import (
    _integer_signedness,
    _restore_integer_signedness,
    _strip_integer_signedness,
)

from ptoas.mlir.dialects import arith, math
from ptoas.mlir.ir import IndexType, IntegerType


_FLOAT_BINARY_OPS = {
    "add": arith.AddFOp,
    "sub": arith.SubFOp,
    "mul": arith.MulFOp,
    "truediv": arith.DivFOp,
}


def emit_runtime_binary_op(op_name: str, lhs, rhs):
    """Lower one authored runtime scalar binary operator."""
    lhs, rhs, kind, authored_type = normalize_runtime_binary_operands(lhs, rhs)
    if kind in {"index", "integer"}:
        op_cls = _integer_binary_op(op_name, authored_type)
        if op_cls is None:
            raise TypeError(f"runtime scalar operator '{op_name}' is not supported for integer/index values")
        if kind == "integer":
            lhs = _strip_integer_signedness(lhs)
            rhs = _strip_integer_signedness(rhs)
        result = op_cls(lhs, rhs).result
        if kind == "index":
            return result
        return _restore_runtime_integer_result(result, authored_type)
    if kind == "float":
        op_cls = _FLOAT_BINARY_OPS.get(op_name)
        if op_cls is None:
            raise TypeError(f"runtime scalar operator '{op_name}' is not supported for floating-point values")
        return op_cls(lhs, rhs).result
    raise TypeError(f"unsupported runtime scalar operand category '{kind}'")


def emit_runtime_max(lhs, rhs):
    """Lower one authored runtime scalar max operation."""
    lhs, rhs, kind, authored_type = normalize_runtime_binary_operands(lhs, rhs)
    if kind == "float":
        return arith.MaximumFOp(lhs, rhs).result
    if kind == "integer":
        signedness = _integer_signedness(authored_type)
        signless_lhs = _strip_integer_signedness(lhs)
        signless_rhs = _strip_integer_signedness(rhs)
        if signedness == "unsigned":
            result = arith.MaxUIOp(signless_lhs, signless_rhs).result
        else:
            result = arith.MaxSIOp(signless_lhs, signless_rhs).result
        return _restore_integer_signedness(result, authored_type)
    if kind == "index":
        cond = arith.CmpIOp(arith.CmpIPredicate.sge, lhs, rhs).result
        return arith.SelectOp(cond, lhs, rhs).result
    raise TypeError(f"unsupported runtime scalar operand category '{kind}'")


def emit_runtime_min(lhs, rhs):
    """Lower one authored runtime scalar min operation."""
    lhs, rhs, kind, authored_type = normalize_runtime_binary_operands(lhs, rhs)
    if kind == "float":
        return arith.MinimumFOp(lhs, rhs).result
    if kind == "integer":
        signedness = _integer_signedness(authored_type)
        signless_lhs = _strip_integer_signedness(lhs)
        signless_rhs = _strip_integer_signedness(rhs)
        if signedness == "unsigned":
            result = arith.MinUIOp(signless_lhs, signless_rhs).result
        else:
            result = arith.MinSIOp(signless_lhs, signless_rhs).result
        return _restore_integer_signedness(result, authored_type)
    if kind == "index":
        cond = arith.CmpIOp(arith.CmpIPredicate.sle, lhs, rhs).result
        return arith.SelectOp(cond, lhs, rhs).result
    raise TypeError(f"unsupported runtime scalar operand category '{kind}'")


def _restore_runtime_integer_result(result, authored_type):
    if IndexType.isinstance(authored_type):
        return result
    if not IntegerType.isinstance(authored_type):
        return result
    return _restore_integer_signedness(result, authored_type)


def emit_runtime_compare(op_name: str, lhs, rhs):
    """Lower one authored runtime scalar comparison operator."""
    lhs, rhs, kind, authored_type = normalize_runtime_binary_operands(lhs, rhs)

    if kind == "float":
        predicate = {
            "lt": arith.CmpFPredicate.OLT,
            "le": arith.CmpFPredicate.OLE,
            "gt": arith.CmpFPredicate.OGT,
            "ge": arith.CmpFPredicate.OGE,
            "eq": arith.CmpFPredicate.OEQ,
            "ne": arith.CmpFPredicate.ONE,
        }.get(op_name)
        if predicate is None:
            raise TypeError(f"runtime scalar comparison '{op_name}' is not supported for floating-point values")
        return arith.CmpFOp(predicate, lhs, rhs).result

    if kind == "index":
        predicate = {
            "lt": arith.CmpIPredicate.slt,
            "le": arith.CmpIPredicate.sle,
            "gt": arith.CmpIPredicate.sgt,
            "ge": arith.CmpIPredicate.sge,
            "eq": arith.CmpIPredicate.eq,
            "ne": arith.CmpIPredicate.ne,
        }.get(op_name)
        if predicate is None:
            raise TypeError(f"runtime scalar comparison '{op_name}' is not supported for index values")
        return arith.CmpIOp(predicate, lhs, rhs).result

    if kind == "integer":
        signedness = _integer_signedness(authored_type)
        signed_predicates = {
            "lt": arith.CmpIPredicate.slt,
            "le": arith.CmpIPredicate.sle,
            "gt": arith.CmpIPredicate.sgt,
            "ge": arith.CmpIPredicate.sge,
            "eq": arith.CmpIPredicate.eq,
            "ne": arith.CmpIPredicate.ne,
        }
        unsigned_predicates = {
            "lt": arith.CmpIPredicate.ult,
            "le": arith.CmpIPredicate.ule,
            "gt": arith.CmpIPredicate.ugt,
            "ge": arith.CmpIPredicate.uge,
            "eq": arith.CmpIPredicate.eq,
            "ne": arith.CmpIPredicate.ne,
        }
        predicate = (unsigned_predicates if signedness == "unsigned" else signed_predicates).get(op_name)
        if predicate is None:
            raise TypeError(f"runtime scalar comparison '{op_name}' is not supported for integer values")
        return arith.CmpIOp(predicate, _strip_integer_signedness(lhs), _strip_integer_signedness(rhs)).result

    raise TypeError(f"unsupported runtime scalar operand category '{kind}'")


def emit_runtime_bitwise_op(op_name: str, lhs, rhs):
    """Lower one authored runtime scalar bitwise operator."""
    lhs, rhs, kind, authored_type = normalize_runtime_binary_operands(lhs, rhs)
    op_cls = {
        "and": arith.AndIOp,
        "or": arith.OrIOp,
        "xor": arith.XOrIOp,
    }.get(op_name)
    if op_cls is None:
        raise TypeError(f"unsupported runtime scalar bitwise operator '{op_name}'")

    if kind == "index":
        return op_cls(lhs, rhs).result

    if kind != "integer":
        raise TypeError(
            f"runtime scalar bitwise operator '{op_name}' expects integer-like operands, got {lhs.type} and {rhs.type}"
        )

    result = op_cls(_strip_integer_signedness(lhs), _strip_integer_signedness(rhs)).result
    return _restore_integer_signedness(result, authored_type)


def emit_runtime_abs(value):
    """Lower one authored runtime scalar absolute-value operation."""
    kind = classify_runtime_scalar_type(value.type)
    if kind == "float":
        return math.AbsFOp(value).result
    if kind == "index":
        return value
    if kind == "integer":
        signedness = _integer_signedness(value.type)
        if signedness == "unsigned":
            return value
        result = math.AbsIOp(_strip_integer_signedness(value)).result
        return _restore_integer_signedness(result, value.type)
    raise TypeError(f"unsupported runtime scalar operand category '{kind}'")


def _integer_binary_op(op_name: str, authored_type):
    if IndexType.isinstance(authored_type):
        return {
            "add": arith.AddIOp,
            "sub": arith.SubIOp,
            "mul": arith.MulIOp,
            "floordiv": arith.FloorDivSIOp,
            "mod": arith.RemSIOp,
        }.get(op_name)

    signedness = _integer_signedness(authored_type)
    if op_name in {"add", "sub", "mul"}:
        return {
            "add": arith.AddIOp,
            "sub": arith.SubIOp,
            "mul": arith.MulIOp,
        }[op_name]
    if op_name == "floordiv":
        if signedness == "unsigned":
            return arith.DivUIOp
        return arith.FloorDivSIOp
    if op_name == "mod":
        if signedness == "unsigned":
            return arith.RemUIOp
        return arith.RemSIOp
    return None


__all__ = [
    "classify_runtime_scalar_type",
    "emit_runtime_abs",
    "emit_runtime_binary_op",
    "emit_runtime_compare",
    "emit_runtime_bitwise_op",
    "emit_runtime_max",
    "emit_runtime_min",
    "normalize_runtime_binary_operands",
]
