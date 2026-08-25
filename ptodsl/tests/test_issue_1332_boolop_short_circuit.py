#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import inspect

from ptodsl import pto, scalar
from ptodsl._ast_rewrite import rewrite_jit_function


# Issue #1332 original kernels: Python ``and``/``or`` between runtime
# predicates must lower to device-side short-circuit ``scf.if`` regions
# instead of calling ``__bool__`` on a PTODSL runtime value at trace time.


@pto.jit(name="issue_1332_and_kernel", kernel_kind="vector", target="a5")
def issue_1332_and_kernel(value: pto.i32, divisor: pto.i32, out: pto.ptr(pto.i8, "gm")):
    pred = (divisor != 0) and ((value // divisor) > 0)
    scalar.store(pred, out, 0)


@pto.jit(name="issue_1332_or_kernel", kernel_kind="vector", target="a5")
def issue_1332_or_kernel(value: pto.i32, divisor: pto.i32, out: pto.ptr(pto.i8, "gm")):
    pred = (divisor == 0) or ((value // divisor) > 0)
    scalar.store(pred, out, 0)


# Three-operand chains fold right, matching Python semantics.


@pto.jit(name="issue_1332_and_chain", kernel_kind="vector", target="a5")
def issue_1332_and_chain(a: pto.i32, b: pto.i32, c: pto.i32, out: pto.ptr(pto.i8, "gm")):
    pred = (a > 0) and (b > 0) and (c > 0)
    scalar.store(pred, out, 0)


@pto.jit(name="issue_1332_or_chain", kernel_kind="vector", target="a5")
def issue_1332_or_chain(a: pto.i32, b: pto.i32, c: pto.i32, out: pto.ptr(pto.i8, "gm")):
    pred = (a > 0) or (b > 0) or (c > 0)
    scalar.store(pred, out, 0)


# ``and``/``or`` inside an ``if`` condition and a native ``while`` test.


@pto.jit(name="issue_1332_if_condition", kernel_kind="vector", target="a5")
def issue_1332_if_condition(a: pto.i32, b: pto.i32, out: pto.ptr(pto.i8, "gm")):
    if (a > 0) and (b > 0):
        scalar.store(pto.const(1, dtype=pto.i8), out, 0)
    else:
        scalar.store(pto.const(0, dtype=pto.i8), out, 0)


@pto.jit(name="issue_1332_while_test", kernel_kind="vector", target="a5")
def issue_1332_while_test(x: pto.i32, y: pto.i32, out: pto.ptr(pto.i8, "gm")):
    i = pto.const(0, dtype=pto.i32)
    while (i < y) and (x > 0):
        scalar.store(pto.const(7, dtype=pto.i8), out, i)
        i = i + 1
    scalar.store(pto.const(9, dtype=pto.i8), out, 0)


# ``and``/``or`` in call-argument position and in ``return``.


@pto.jit(name="issue_1332_call_argument", kernel_kind="vector", target="a5")
def issue_1332_call_argument(a: pto.i32, b: pto.i32, out: pto.ptr(pto.i8, "gm")):
    pred = (a > 0) and (b > 0)
    result = scalar.select(
        pred,
        pto.const(1, dtype=pto.i32),
        pto.const(0, dtype=pto.i32),
    )
    scalar.store(result, out, 0)


@pto.jit(name="issue_1332_return_position", kernel_kind="vector", target="a5")
def issue_1332_return_position(a: pto.i32, b: pto.i32) -> pto.i1:
    return (a > 0) or (b > 0)


# Static operands short-circuit at trace time: ``1 // 0`` would raise
# ZeroDivisionError if the RHS were ever traced, so compiling these kernels
# proves the RHS lambda is not evaluated.


@pto.jit(name="issue_1332_static_short_circuit", kernel_kind="vector", target="a5")
def issue_1332_static_short_circuit(out: pto.ptr(pto.i8, "gm")):
    pred_and = False and (1 // 0)
    pred_or = True or (1 // 0)
    if pred_and is False and pred_or is True:
        scalar.store(pto.const(1, dtype=pto.i8), out, 0)
    else:
        scalar.store(pto.const(0, dtype=pto.i8), out, 0)


# Runtime integer left operand: the control condition uses non-zero
# truthiness while the branch merge reuses the existing integer-to-i1
# reconciliation rule.


@pto.jit(name="issue_1332_integer_lhs", kernel_kind="vector", target="a5")
def issue_1332_integer_lhs(x: pto.i32, y: pto.i32, out: pto.ptr(pto.i8, "gm")):
    pred = x and (y > 0)
    scalar.store(pred, out, 0)


@pto.jit(name="issue_1332_integer_operands", kernel_kind="vector", target="a5")
def issue_1332_integer_operands(x: pto.i32, y: pto.i32, out: pto.ptr(pto.i32, "gm")):
    pred = x and y
    scalar.store(pred, out, 0)


# ``runtime_value and True`` materializes the Python bool to an i1 const in
# the branch merge.


@pto.jit(name="issue_1332_bool_literal_rhs", kernel_kind="vector", target="a5")
def issue_1332_bool_literal_rhs(f: pto.i1, out: pto.ptr(pto.i8, "gm")):
    pred = f and True
    scalar.store(pred, out, 0)


# and/or on plain Python operands (lists, strings) keep native truthiness
# inside rewritten kernels: TileOps templates use the loops-or-None pattern where
# loops is a regular Python list that must not be traced as a runtime condition
# (and must never reach the surface-value validator).

@pto.jit(name="issue_1332_plain_python_operands", kernel_kind="vector", target="a5")
def issue_1332_plain_python_operands(out: pto.ptr(pto.i8, "gm")):
    pairs = [(1, 2), (3, 4)] or None
    if pairs is None or len(pairs) != 2:
        return
    empty = [] or None
    if empty is not None:
        return
    kept = "x" or None
    if kept != "x":
        return
    value = pairs and pairs[0]
    if value != (1, 2):
        return
    scalar.store(pto.const(11, dtype=pto.i8), out, 0)


# Diagnostics: floating-point controls, incompatible branch merges, and the
# native behavior preserved when AST rewrite is disabled.


@pto.jit(name="issue_1332_float_control", kernel_kind="vector", target="a5")
def issue_1332_float_control(x: pto.f32, out: pto.ptr(pto.i8, "gm")):
    pred = x and (x > 0)
    scalar.store(pred, out, 0)


@pto.jit(name="issue_1332_incompatible_merge", kernel_kind="vector", target="a5")
def issue_1332_incompatible_merge(x: pto.i32, f: pto.f32, out: pto.ptr(pto.i8, "gm")):
    pred = (x > 0) and f
    scalar.store(pred, out, 0)


@pto.jit(name="issue_1332_no_ast_rewrite", kernel_kind="vector", target="a5", ast_rewrite=False)
def issue_1332_no_ast_rewrite(x: pto.i32, y: pto.i32, out: pto.ptr(pto.i8, "gm")):
    pred = (x > 0) and (y > 0)
    scalar.store(pred, out, 0)


def _expect_error(fn, needle):
    try:
        fn()
    except Exception as exc:
        if needle in str(exc):
            return
        raise AssertionError(
            f"expected error containing {needle!r}, got: {type(exc).__name__}: {exc}"
        ) from exc
    raise AssertionError(f"expected {needle!r} error, but no error was raised")


def main():
    and_mlir = issue_1332_and_kernel.compile().mlir_text()
    or_mlir = issue_1332_or_kernel.compile().mlir_text()
    # The guarded RHS (division) must live inside an scf.if region.
    assert "scf.if" in and_mlir and "arith.floordivsi" in and_mlir
    assert "scf.if" in or_mlir and "arith.floordivsi" in or_mlir
    # Short-circuit control uses the LHS predicate, not the division result.
    assert "arith.cmpi ne, %" in and_mlir or "arith.cmpi ne" in and_mlir

    chain_and = issue_1332_and_chain.compile().mlir_text()
    chain_or = issue_1332_or_chain.compile().mlir_text()
    # a and b and c lowers to two nested scf.if regions.
    assert chain_and.count("scf.if") == 2, chain_and
    assert chain_or.count("scf.if") == 2, chain_or

    issue_1332_if_condition.compile().mlir_text()
    issue_1332_while_test.compile().mlir_text()
    issue_1332_call_argument.compile().mlir_text()
    ret_mlir = issue_1332_return_position.compile().mlir_text()
    assert "scf.if" in ret_mlir

    static_mlir = issue_1332_static_short_circuit.compile().mlir_text()
    # A traced RHS would raise ZeroDivisionError; static operands must also
    # leave no division in the IR at all.
    assert "floordiv" not in static_mlir, static_mlir

    int_lhs = issue_1332_integer_lhs.compile().mlir_text()
    # Integer LHS becomes a non-zero i1 control condition.
    assert int_lhs.count("arith.cmpi ne") == 2, int_lhs
    int_operands = issue_1332_integer_operands.compile().mlir_text()
    # Both integer operands merge back to i32 (Python operand semantics).
    assert "-> (i32)" in int_operands, int_operands

    plain = issue_1332_plain_python_operands.compile().mlir_text()
    bool_lit = issue_1332_bool_literal_rhs.compile().mlir_text()
    assert "arith.constant true" in bool_lit, bool_lit

    _expect_error(
        lambda: issue_1332_float_control.compile().mlir_text(),
        "short-circuit",
    )
    _expect_error(
        lambda: issue_1332_incompatible_merge.compile().mlir_text(),
        "type mismatch",
    )
    _expect_error(
        lambda: issue_1332_no_ast_rewrite.compile().mlir_text(),
        "native Python if/while condition cannot consume a PTODSL runtime value",
    )

    # Source-less functions keep the pre-rewrite tracing behavior: the
    # rewriter returns them untouched instead of synthesizing helper calls.
    exec("def _source_less(x):\n    return x\n")
    source_less_fn = locals()["_source_less"]
    try:
        inspect.getsource(source_less_fn)
    except (OSError, TypeError):
        pass
    else:
        raise AssertionError("test function unexpectedly has retrievable source")
    assert rewrite_jit_function(source_less_fn) is source_less_fn

    print("issue 1332 short-circuit checks passed")


if __name__ == "__main__":
    main()