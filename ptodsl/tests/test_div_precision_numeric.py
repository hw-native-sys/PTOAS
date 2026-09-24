#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Evaluate the materialized scalar SoftLib integer IR against NumPy."""

import operator

import numpy as np

from ptoas.mlir.ir import IntegerAttr
from ptodsl._context import make_context
from ptodsl.softlib._compiler_runtime import materialize


_BINARY = {
    "pto.addi": operator.add, "pto.subi": operator.sub,
    "pto.muli": operator.mul, "pto.and": operator.and_,
    "pto.or": operator.or_, "pto.xor": operator.xor,
    "pto.shl": operator.lshift, "pto.shr": operator.rshift,
    "pto.divi": operator.floordiv,
}
_COMPARE = {
    "eq": operator.eq, "ne": operator.ne, "lt": operator.lt,
    "le": operator.le, "gt": operator.gt, "ge": operator.ge,
}


def _constant_value(op, _args):
    return np.uint32(IntegerAttr(op.attributes["value"]).value & 0xffffffff)


def _same_value(_op, args):
    return args[0]


def _bitcast_value(op, args):
    dtype = np.float32 if str(op.results[0].type) == "f32" else np.uint32
    return np.asarray(args[0]).view(dtype)


def _cmpi_value(op, args):
    predicate = str(op.attributes["predicate"]).split("<")[1].rstrip(">")
    if str(op.attributes["signedness"]) == "#pto.signedness<signed>":
        args = [np.asarray(arg).view(np.int32) for arg in args]
    return _COMPARE[predicate](*args)


def _binary_value(op, args):
    name = op.name
    if name in ("pto.shl", "pto.shr") and np.any(args[1] >= 32):
        raise AssertionError(f"invalid shift in {op}")
    if name == "pto.divi" and np.any(args[1] == 0):
        raise AssertionError("integer division by zero")
    return _BINARY[name](*args)


_EVALUATORS = {
    "pto.constant": _constant_value,
    "builtin.unrealized_conversion_cast": _same_value,
    "pto.bitcast": _bitcast_value,
    "pto.select": lambda _op, args: np.where(*args),
    "pto.cmpi": _cmpi_value,
    "pto.divf": lambda _op, args: np.divide(*args, dtype=np.float32),
}


def _evaluate_op(op, args):
    if op.name in _BINARY:
        return _binary_value(op, args)
    evaluator = _EVALUATORS.get(op.name)
    if evaluator is None:
        raise AssertionError(f"unsupported operation: {op.name}")
    return evaluator(op, args)


def _evaluate(block, lhs, rhs):
    values = dict(zip(block.arguments, (lhs, rhs)))
    for op in block.operations:
        args = [values[value] for value in op.operands]
        if op.name == "func.return":
            return args[0]
        values[op.results[0]] = _evaluate_op(op, args)
    raise AssertionError("missing return")


def test_materialized_scalar_division_matches_numpy():
    rng = np.random.default_rng(20260803)
    lhs = rng.integers(0, 2**32, 100000, dtype=np.uint32)
    rhs = rng.integers(0, 2**32, 100000, dtype=np.uint32)
    boundaries = np.array([
        0, 1, 2, 3, 0x003fffff, 0x007fffff, 0x00800000, 0x00800001,
        0x3f000000, 0x3f800000, 0x3f800001, 0x40000000, 0x40e00000,
        0x7f7fffff, 0x7f800000, 0x7fc00000,
    ], dtype=np.uint32)
    boundaries = np.concatenate((boundaries, boundaries | np.uint32(0x80000000)))
    lhs = np.concatenate((lhs, np.repeat(boundaries, boundaries.size))).view(np.float32)
    rhs = np.concatenate((rhs, np.tile(boundaries, boundaries.size))).view(np.float32)
    module, _ = materialize(
        "a5", "pto.divf", '{"dtype":"f32","precision":"high_precision"}', make_context()
    )
    block = list(module.body.operations)[0].regions[0].blocks[0]
    with np.errstate(all="ignore"):
        expected = np.divide(lhs, rhs, dtype=np.float32)
        actual = _evaluate(block, lhs, rhs)
    # NaN payload/sign are native behavior; all other results must match bitwise.
    nan = np.isnan(expected)
    np.testing.assert_array_equal(np.isnan(actual), nan)
    np.testing.assert_array_equal(actual[~nan].view(np.uint32), expected[~nan].view(np.uint32))


if __name__ == "__main__":
    test_materialized_scalar_division_matches_numpy()
