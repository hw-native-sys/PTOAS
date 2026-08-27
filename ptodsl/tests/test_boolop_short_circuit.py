#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import inspect
from types import FunctionType

from ptodsl import pto, scalar
from ptoas.mlir.dialects import arith
from ptodsl._ast_rewrite import PTODSLAstRewriteError, rewrite_jit_function


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




@pto.jit(name="issue_1332_integer_or_lhs", kernel_kind="vector", target="a5")
def issue_1332_integer_or_lhs(x: pto.i32, y: pto.i32, out: pto.ptr(pto.i8, "gm")):
    pred = x or (y > 0)
    scalar.store(pred, out, 0)


@pto.jit(name="issue_1332_flag_integer_rhs", kernel_kind="vector", target="a5")
def issue_1332_flag_integer_rhs(flag: pto.i1, y: pto.i32, out: pto.ptr(pto.i8, "gm")):
    pred = flag and y
    scalar.store(pred, out, 0)


# Static floats and ints keep native Python truthiness and operand results at
# trace time (runtime float controls are still diagnosed).


@pto.jit(name="issue_1332_static_float", kernel_kind="vector", target="a5")
def issue_1332_static_float(f: pto.i1, out: pto.ptr(pto.i8, "gm")):
    pred_a = 0.0 or True
    if pred_a is not True:
        return
    pred_b = 1.0 and f
    scalar.store(pred_b, out, 0)


@pto.jit(name="issue_1332_static_int_operands", kernel_kind="vector", target="a5")
def issue_1332_static_int_operands(out: pto.ptr(pto.i8, "gm")):
    a = 2 and True
    if a is not True:
        return
    b = 0 and True
    if b != 0:
        return
    c = 5 or (1 > 2)
    if c != 5:
        return
    d = 0 or (1 > 2)
    if d is not False:
        return
    scalar.store(pto.const(3, dtype=pto.i8), out, 0)


# and/or inside an @pto.func helper body.


@pto.func(returns=pto.i1)
def issue_1332_func_and(a: pto.i1, b: pto.i1) -> pto.i1:
    return a and b


@pto.jit(name="issue_1332_func_call", kernel_kind="vector", target="a5")
def issue_1332_func_call(x: pto.i32, y: pto.i32, out: pto.ptr(pto.i8, "gm")):
    pred = issue_1332_func_and(x > 0, y > 0)
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


@pto.jit(name="issue_1332_index_lhs", kernel_kind="vector", target="a5")
def issue_1332_index_lhs(idx: pto.index, y: pto.i32, out: pto.ptr(pto.i8, "gm")):
    pred = idx and (y > 0)
    scalar.store(pred, out, 0)


@pto.jit(name="issue_1332_int_literal_anchored", kernel_kind="vector", target="a5")
def issue_1332_int_literal_anchored(x: pto.i32, out: pto.ptr(pto.i32, "gm")):
    pred = x or 2
    scalar.store(pred, out, 0)


@pto.jit(name="issue_1332_int_literal_vs_i1", kernel_kind="vector", target="a5")
def issue_1332_int_literal_vs_i1(flag: pto.i1, out: pto.ptr(pto.i8, "gm")):
    pred = flag and 2
    scalar.store(pred, out, 0)


@pto.jit(name="issue_1332_walrus_rhs", kernel_kind="vector", target="a5")
def issue_1332_walrus_rhs(flag: pto.i1, out: pto.ptr(pto.i8, "gm")):
    pred = flag and (bound := flag)
    scalar.store(pred, out, 0)


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


@pto.jit(name="issue_1332_float_rhs_literal", kernel_kind="vector", target="a5")
def issue_1332_float_rhs_literal(flag: pto.i1, out: pto.ptr(pto.i8, "gm")):
    pred = flag and 0.5
    scalar.store(pred, out, 0)


@pto.jit(name="issue_1332_no_ast_rewrite", kernel_kind="vector", target="a5", ast_rewrite=False)
def issue_1332_no_ast_rewrite(x: pto.i32, y: pto.i32, out: pto.ptr(pto.i8, "gm")):
    pred = (x > 0) and (y > 0)
    scalar.store(pred, out, 0)


def _expect_error(fn, needle, error_type=None):
    try:
        fn()
    except Exception as exc:
        if error_type is not None and not isinstance(exc, error_type):
            raise AssertionError(
                f"expected {error_type.__name__}, got: {type(exc).__name__}: {exc}"
            ) from exc
        if needle in str(exc):
            return
        raise AssertionError(
            f"expected error containing {needle!r}, got: {type(exc).__name__}: {exc}"
        ) from exc
    raise AssertionError(f"expected {needle!r} error, but no error was raised")


def main():
    # The guarded RHS (division) must live inside the scf.if branch and the
    # condition must come from the LHS comparison.  Verify by walking the
    # emitted module rather than matching text fragments.
    def _walk(op):
        yield op
        for region in op.operation.regions:
            for block in region.blocks:
                for child in block.operations:
                    yield from _walk(child)

    def _ops(module, name):
        return [op for op in _walk(module) if op.operation.name == name]

    def _has_result_type(module, name, expected_type):
        return any(
            str(result.type) == expected_type
            for op in _ops(module, name)
            for result in op.operation.results
        )

    def _has_constant(module, value, expected_type):
        for op in _ops(module, "arith.constant"):
            if not any(str(result.type) == expected_type for result in op.operation.results):
                continue
            attributes = op.operation.attributes
            if "value" in attributes and str(attributes["value"]).split(" :", 1)[0] == str(value):
                return True
        return False

    def _guarded_division(module, region_index):
        scf_ifs = [op for op in _walk(module) if op.operation.name == "scf.if"]
        assert scf_ifs, "expected scf.if in the short-circuit kernel"
        scf_if = scf_ifs[0]
        cond_owner = scf_if.operation.operands[0].owner
        assert cond_owner is not None and cond_owner.operation.name == "arith.cmpi", (
            "scf.if condition must be the LHS comparison",
        )
        names = []
        for block in scf_if.operation.regions[region_index].blocks:
            for child in block.operations:
                names.extend(op.operation.name for op in _walk(child))
        assert "arith.floordivsi" in names, f"guarded division missing from region {region_index}"

    _guarded_division(issue_1332_and_kernel.mlir_module(), 0)  # and: then region
    _guarded_division(issue_1332_or_kernel.mlir_module(), 1)   # or: else region

    chain_and_module = issue_1332_and_chain.mlir_module()
    chain_or_module = issue_1332_or_chain.mlir_module()
    # a and b and c lowers to two nested scf.if regions.
    assert len(_ops(chain_and_module, "scf.if")) == 2
    assert len(_ops(chain_or_module, "scf.if")) == 2

    issue_1332_if_condition.compile().mlir_text()
    issue_1332_while_test.compile().mlir_text()
    issue_1332_call_argument.compile().mlir_text()
    assert _ops(issue_1332_return_position.mlir_module(), "scf.if")

    # A traced RHS would raise ZeroDivisionError; static operands must also
    # leave no division in the IR at all.
    assert not _ops(issue_1332_static_short_circuit.mlir_module(), "arith.floordivsi")

    int_lhs_module = issue_1332_integer_lhs.mlir_module()
    # Integer LHS becomes a non-zero i1 control condition; the integer operand
    # is preserved and the i1 RHS widens to the integer type.
    int_lhs_ifs = _ops(int_lhs_module, "scf.if")
    assert len(int_lhs_ifs) == 1
    int_lhs_condition = int_lhs_ifs[0].operation.operands[0].owner
    assert int_lhs_condition is not None
    assert int_lhs_condition.operation.name == "arith.cmpi"
    predicate = int_lhs_condition.operation.attributes["predicate"]
    assert predicate.value == int(arith.CmpIPredicate.ne)
    assert _ops(int_lhs_module, "arith.extui")
    assert _has_result_type(int_lhs_module, "scf.if", "i32")
    int_operands_module = issue_1332_integer_operands.mlir_module()
    # Both integer operands merge back to i32 (Python operand semantics).
    assert _has_result_type(int_operands_module, "scf.if", "i32")

    or_lhs_module = issue_1332_integer_or_lhs.mlir_module()
    assert _ops(or_lhs_module, "arith.extui")
    assert _has_result_type(or_lhs_module, "scf.if", "i32")
    flag_and_module = issue_1332_flag_integer_rhs.mlir_module()
    assert _ops(flag_and_module, "arith.extui")
    assert _has_result_type(flag_and_module, "scf.if", "i32")
    index_lhs_module = issue_1332_index_lhs.mlir_module()
    # index + i1: the index type is kept and the i1 side widens via index_cast.
    assert _ops(index_lhs_module, "arith.index_cast")
    assert _has_result_type(index_lhs_module, "scf.if", "index")
    # The i1 -> index widening must keep the 0/1 truth value: a direct
    # arith.index_cast from i1 sign-extends (1 becomes -1 on the simulator),
    # so the widening must zero-extend through i32 first.
    index_casts = _ops(index_lhs_module, "arith.index_cast")
    assert any(
        op.operation.operands[0].owner is not None
        and op.operation.operands[0].owner.operation.name == "arith.extui"
        for op in index_casts
    ), "i1->index widening must zero-extend via extui first"
    anchored_module = issue_1332_int_literal_anchored.mlir_module()
    # A Python int literal anchors to the integer branch type.
    assert _has_constant(anchored_module, 2, "i32")
    assert _has_result_type(anchored_module, "scf.if", "i32")
    _expect_error(
        lambda: issue_1332_int_literal_vs_i1.compile().mlir_text(),
        "cannot infer an integer width",
    )
    _expect_error(
        lambda: issue_1332_walrus_rhs.compile().mlir_text(),
        "assignment expression",
        PTODSLAstRewriteError,
    )
    issue_1332_static_float.compile().mlir_text()
    issue_1332_static_int_operands.compile().mlir_text()
    issue_1332_func_call.compile().mlir_text()

    issue_1332_plain_python_operands.compile().mlir_text()
    bool_lit_module = issue_1332_bool_literal_rhs.mlir_module()
    assert _has_result_type(bool_lit_module, "arith.constant", "i1")

    _expect_error(
        lambda: issue_1332_float_control.compile().mlir_text(),
        "short-circuit",
    )
    _expect_error(
        lambda: issue_1332_incompatible_merge.compile().mlir_text(),
        "type mismatch",
    )
    _expect_error(
        lambda: issue_1332_float_rhs_literal.compile().mlir_text(),
        "float literals",
    )
    _expect_error(
        lambda: issue_1332_no_ast_rewrite.compile().mlir_text(),
        "native Python if/while condition cannot consume a PTODSL runtime value",
    )

    # Source-less functions keep the pre-rewrite tracing behavior: the
    # rewriter returns them untouched instead of synthesizing helper calls.
    source_less_fn = FunctionType(
        (lambda value: value).__code__.replace(co_filename="<source-less>"),
        globals(),
        "_source_less",
    )
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
