#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from dataclasses import replace
from pathlib import Path
import inspect
import os
import re
import subprocess
import sys
from tempfile import TemporaryDirectory
from importlib.util import module_from_spec, spec_from_file_location
from typing import Optional
from unittest import mock

# Prefer the build-tree MLIR Python bindings during CMake/CTest runs.  An
# editable scikit-build install otherwise redirects ptoas.mlir imports to the
# previously installed generated bindings, which can miss newly added ops.
sys.meta_path[:] = [finder for finder in sys.meta_path if "editable" not in repr(finder).lower()]

from ptodsl import pto, scalar
from ptodsl import _types as pto_types
import ptodsl._vmi_namespace as vmi_namespace
from ptodsl._ast_rewrite import PTODSLAstRewriteError
from ptodsl._context import make_context
from ptodsl._cache_signature import cache_signature_atom
from ptodsl._kernel_signature import DeviceParameterSpec, HelperMarkerParameterSpec, RuntimeScalarParameterSpec
from ptodsl._tracing.runtime import SignatureTracingRuntime
from ptodsl._runtime import native_build as native_build_runtime
from ptodsl._runtime.cache import NativeBuildArtifacts, artifact_paths
from ptodsl._runtime.codegen import generate_launch_cpp
from ptodsl._runtime.launch import _marshal_launch_args
from ptodsl._tracing import current_session
from ptoas.mlir.ir import InsertionPoint, Location, Module


@pto.jit(target='a5')
def ast_nested_partial_assign_entry_isolated_probe(cond3: pto.i1, cond1: pto.i1):
    # Sibling branch isolation: the outer then-branch rebinds lo_bits before the
    # inner conditional is traced, so the inner else (which never assigns lo_bits)
    # must still fall back to the value captured before the outer if.
    lo_bits = pto.const(0, dtype=pto.ui32)
    hi_bits = pto.const(0x40000000, dtype=pto.ui32)

    q1_bits = pto.const(0x10000000, dtype=pto.ui32)
    q2_bits = pto.const(0x20000000, dtype=pto.ui32)

    if cond3:
        lo_bits = q2_bits
    else:
        if cond1:
            lo_bits = q1_bits
            hi_bits = q2_bits
        # else: lo_bits/hi_bits must remain at their pre-if values.

    _ = hi_bits
    _ = lo_bits


@pto.jit(target='a5')
def ast_nested_partial_assign_loop_carry_probe(
    rows: pto.i32,
    cond3: pto.i1,
    cond2: pto.i1,
    cond1: pto.i1,
):
    # Issue 1252 structure: three nested conditionals inside a runtime loop,
    # where the innermost else only assigns hi_bits. On that path lo_bits must
    # remain the loop-carried value from the previous iteration.
    lo_bits = pto.const(0, dtype=pto.ui32)
    hi_bits = pto.const(0x40000000, dtype=pto.ui32)

    q1_bits = pto.const(0x10000000, dtype=pto.ui32)
    q2_bits = pto.const(0x20000000, dtype=pto.ui32)
    q3_bits = pto.const(0x30000000, dtype=pto.ui32)

    for _ in range(rows):
        if cond3:
            lo_bits = q3_bits
        else:
            if cond2:
                lo_bits = q2_bits
                hi_bits = q3_bits
            else:
                if cond1:
                    lo_bits = q1_bits
                    hi_bits = q2_bits
                else:
                    hi_bits = q1_bits  # lo_bits must remain unchanged

    _ = hi_bits
    _ = lo_bits


@pto.jit(target='a5')
def ast_nested_partial_assign_slot_entry_isolated_probe(cond3: pto.i1, cond1: pto.i1):
    # Static-subscript analogue of the entry isolation probe: the slot value is
    # rewritten into a shared Python temporary, so the same sibling pollution
    # would leak unless the temporary is restored at each branch entry.
    zero = pto.const(0, dtype=pto.ui32)
    q1_bits = pto.const(0x10000000, dtype=pto.ui32)
    q2_bits = pto.const(0x20000000, dtype=pto.ui32)

    values = [zero]
    if cond3:
        values[0] = q2_bits
    else:
        if cond1:
            values[0] = q1_bits
        # else: values[0] must remain at its pre-if value.

    out = values[0]
    _ = out


@pto.jit(target='a5')
def ast_unbound_partial_liveout_loop_probe(rows: pto.i32, cond: pto.i1):
    # value is never bound before the loop, so it must stay on the
    # last-iteration-only diagnostics path instead of becoming an implicit carry.
    for _ in range(rows):
        if cond:
            value = 0
    _ = value


@pto.jit(target='a5')
def ast_unbound_partial_liveout_controlled_loop_probe(rows: pto.i32, cond: pto.i1, stop: pto.i1):
    # break/continue/for-else loops must apply the same definite-binding gate.
    one = pto.const(1, dtype=pto.i32)
    for _ in range(rows):
        if cond:
            value = one
        if stop:
            break
    _ = value


@pto.jit(target='a5')
def ast_param_bound_partial_liveout_loop_probe(rows: pto.i32, cond: pto.i1, value: pto.i32):
    one = pto.const(1, dtype=pto.i32)
    for _ in range(rows):
        if cond:
            value = one
    _ = value


@pto.jit(target='a5')
def ast_outer_bound_partial_liveout_loop_probe(rows: pto.i32, cond: pto.i1, outer: pto.i1):
    one = pto.const(1, dtype=pto.i32)
    zero = pto.const(0, dtype=pto.i32)
    value = zero
    if outer:
        for _ in range(rows):
            if cond:
                value = one
    _ = value


@pto.jit(target='a5')
def ast_both_branch_bound_partial_liveout_loop_probe(rows: pto.i32, cond: pto.i1, choose: pto.i1):
    one = pto.const(1, dtype=pto.i32)
    zero = pto.const(0, dtype=pto.i32)
    if choose:
        value = zero
    else:
        value = one
    for _ in range(rows):
        if cond:
            value = one
    _ = value


@pto.jit(target='a5')
def ast_slot_partial_assign_loop_carry_probe(rows: pto.i32, cond: pto.i1):
    zero = pto.const(0, dtype=pto.ui32)
    one = pto.const(1, dtype=pto.ui32)
    values = [zero]
    for _ in range(rows):
        if cond:
            values[0] = one
    _ = values[0]


@pto.jit(target='a5', backend='vpto', mode='explicit')
def ast_with_entry_bound_partial_liveout_probe(rows: pto.i32, cond: pto.i1, value: pto.i32):
    # A function parameter must remain definitely bound inside a vecscope body.
    one = pto.const(1, dtype=pto.i32)
    with pto.vecscope():
        for _ in range(rows):
            if cond:
                value = one
        _ = value
    _ = value


@pto.jit(target='a5', backend='vpto', mode='explicit')
def ast_with_body_bound_partial_liveout_probe(rows: pto.i32, cond: pto.i1):
    # A name bound inside the with body is definite after the with statement.
    one = pto.const(1, dtype=pto.i32)
    zero = pto.const(0, dtype=pto.i32)
    with pto.vecscope():
        value = zero
    for _ in range(rows):
        if cond:
            value = one
    _ = value


@pto.jit(target='a5', backend='vpto', mode='explicit')
def ast_with_as_bound_partial_liveout_probe(rows: pto.i32, cols: pto.i32, cond: pto.i1):
    # The with-as target is definitely bound inside the body; an inner runtime
    # loop that partially rebinds it must see it as a valid partial carry.
    with pto.for_(0, rows, step=1) as value:
        for replacement in range(cols):
            if cond:
                value = replacement
        pto.wait_flag(pto.Pipe.V, pto.Pipe.MTE2, event_id=value)


@pto.jit(target='a5', backend='vpto', mode='explicit')
def ast_for_iv_bound_partial_liveout_probe(rows: pto.i32, cols: pto.i32, cond: pto.i1):
    # The runtime for induction variable is bound at the top of its own body;
    # an inner loop that partially rebinds it must see it as a valid carry.
    for value in range(rows):
        for replacement in range(cols):
            if cond:
                value = replacement
        pto.wait_flag(pto.Pipe.V, pto.Pipe.MTE2, event_id=value)


@pto.jit(target='a5', backend='vpto', mode='explicit')
def ast_for_iv_augassign_probe(rows: pto.i32):
    # Augmenting the induction variable must not infer it as a loop carry (it
    # is re-bound at the top of every iteration) and must not read it before
    # the loop is entered.
    for value in range(rows):
        value += 1
        pto.wait_flag(pto.Pipe.V, pto.Pipe.MTE2, event_id=value)


@pto.jit(target='a5', backend='vpto', mode='explicit')
def ast_static_range_empty_orelse_partial_probe(rows: pto.i32, cond: pto.i1):
    # An empty static range jumps straight to the else clause with the target
    # unbound; the inner partial live-out must keep the explicit diagnostics.
    for value in pto.static_range(0):
        pass
    else:
        for replacement in range(rows):
            if cond:
                value = replacement
        pto.wait_flag(pto.Pipe.V, pto.Pipe.MTE2, event_id=value)


@pto.jit(target='a5', backend='vpto', mode='explicit')
def ast_static_range_subscript_target_probe(rows: pto.i32):
    # Non-Name static_range targets keep their trace-time semantics and must
    # not crash the rewrite with an AttributeError.
    zero = pto.const(0, dtype=pto.i32)
    values = [zero]
    for values[0] in pto.static_range(1):
        pto.wait_flag(pto.Pipe.V, pto.Pipe.MTE2, event_id=values[0])
    _ = values[0]


def ast_static_range_nonempty_definite_gate_fn(rows, cond):
    # Source-only helper for the rewrite-level gate assertion: the static_range
    # target must be credited as definitely bound once it has run, so the later
    # partial loop can carry the value without being rejected.
    for value in pto.static_range(1):
        pass
    for replacement in range(rows):
        if cond:
            value = replacement
    _ = value


def ast_static_range_body_init_definite_gate_fn(rows, cond, seed):
    for _ in pto.static_range(1):
        value = seed
    for replacement in range(rows):
        if cond:
            value = replacement
    _ = value


def ast_static_range_delete_target_gate_fn(rows, cond, seed):
    value = seed
    for value in pto.static_range(1):
        del value
    for replacement in range(rows):
        if cond:
            value = replacement
    _ = value


def ast_static_range_empty_orelse_init_gate_fn(rows, cond, seed):
    for _ in pto.static_range(0):
        pass
    else:
        value = seed
    for replacement in range(rows):
        if cond:
            value = replacement
    _ = value


def ast_static_range_orelse_delete_gate_fn(rows, cond, seed):
    value = seed
    for _ in pto.static_range(1):
        pass
    else:
        del value
    for replacement in range(rows):
        if cond:
            value = replacement
    _ = value


@pto.jit(target='a5', backend='vpto', mode='explicit')
def ast_del_then_partial_liveout_probe(rows: pto.i32, cond: pto.i1):
    zero = pto.const(0, dtype=pto.i32)
    one = pto.const(1, dtype=pto.i32)
    value = zero
    del value
    for _ in range(rows):
        if cond:
            value = one
    _ = value


def _all_operations(operation):
    yield operation
    for region in operation.regions:
        for block in region.blocks:
            for nested in block.operations:
                yield from _all_operations(nested)


def _ui32_const_results(operation, value):
    # The frontend materializes ui32 scalars as i32 constants wrapped in
    # builtin.unrealized_conversion_cast ops, so match constants by value only.
    results = []
    for op in _all_operations(operation):
        if op.operation.name == 'arith.constant':
            attr = op.attributes['value']
            if attr.value == value:
                results.append(op.results[0])
    return results


def _cast_result_of_const(const_result):
    # Resolve the ui32 value of a constant through its feeding conversion cast.
    for use in const_result.uses:
        user = getattr(use, 'owner', None)
        if user is not None and user.operation.name == 'builtin.unrealized_conversion_cast':
            return user.results[0]
    return const_result


def _same_ssa(a, b):
    try:
        return a == b
    except Exception:
        return str(a) == str(b)


def _scf_if_count(block):
    if_ops = [op for op in block.operations if op.operation.name == 'scf.if']
    if not if_ops:
        return 0
    return 1 + _scf_if_count(if_ops[0].regions[1].blocks[0])


def _innermost_else_yield_operands(block):
    if_ops = [op for op in block.operations if op.operation.name == 'scf.if']
    if not if_ops:
        for op in block.operations:
            if op.operation.name == 'scf.yield':
                return list(op.operands)
        return []
    return _innermost_else_yield_operands(if_ops[0].regions[1].blocks[0])


def _find_op(region, name):
    for block in region.blocks:
        for op in block.operations:
            if op.operation.name == name:
                return op
    return None


def _find_op_anywhere(operation, name):
    for op in _all_operations(operation):
        if op.operation.name == name:
            return op
    return None


def _assert_ast_rewrite_nested_partial_assign_ssa_identity():
    # Focused probe 1: sibling-tracing isolation without a loop.
    isolated_text = ast_nested_partial_assign_entry_isolated_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(
        isolated_text,
        'AST-rewritten nested partial assignment entry isolation',
    )
    with make_context() as ctx:
        isolated_module = Module.parse(isolated_text, ctx)
        isolated_func = _find_op_anywhere(isolated_module.operation, 'func.func')
        expect(isolated_func is not None, 'entry isolation probe should contain a func.func op')
        func_block = isolated_func.regions[0].blocks[0]
        zero_consts = _ui32_const_results(isolated_module.operation, 0)
        expect(len(zero_consts) >= 1, 'entry isolation probe should materialize the pre-if 0 constant')
        zero_value = _cast_result_of_const(zero_consts[0])
        q2_consts = _ui32_const_results(isolated_module.operation, 0x20000000)
        expect(len(q2_consts) >= 1, 'entry isolation probe should materialize the q2 constant')
        expect(
            _scf_if_count(func_block) == 2,
            'entry isolation probe should contain two nested scf.if ops',
        )
        yield_operands = _innermost_else_yield_operands(func_block)
        expect(len(yield_operands) == 2, 'innermost else should yield two values')
        expect(
            _same_ssa(yield_operands[1], zero_value),
            'innermost else must yield the pre-if lo_bits value, not the sibling branch rebinding',
        )
        q2_cast = [_cast_result_of_const(q) for q in q2_consts]
        expect(
            not any(_same_ssa(yield_operands[1], q) for q in q2_cast),
            'innermost else must not yield the sibling branch q2 value',
        )

    # Focused probe 2: the issue-1252 runtime loop structure.
    loop_text = ast_nested_partial_assign_loop_carry_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(
        loop_text,
        'AST-rewritten issue-1252 nested partial assignment loop',
    )
    with make_context() as ctx:
        loop_module = Module.parse(loop_text, ctx)
        loop_func = _find_op_anywhere(loop_module.operation, 'func.func')
        expect(loop_func is not None, 'issue-1252 probe should contain a func.func op')
        for_op = _find_op(loop_func.regions[0], 'scf.for')
        expect(for_op is not None, 'issue-1252 probe should lower to a runtime scf.for')
        loop_body = for_op.regions[0].blocks[0]
        arguments = list(loop_body.arguments)
        expect(
            len(arguments) == 3,
            'scf.for should carry the induction variable plus two ui32 iter_args',
        )
        expect(
            str(arguments[1].type) == 'ui32' and str(arguments[2].type) == 'ui32',
            'both loop iter_args should be ui32',
        )
        expect(
            _scf_if_count(loop_body) == 3,
            'issue-1252 probe should contain three nested scf.if ops',
        )
        yield_operands = _innermost_else_yield_operands(loop_body)
        expect(len(yield_operands) == 2, 'innermost else should yield two values')
        expect(
            _same_ssa(yield_operands[1], arguments[2]),
            'innermost else must yield the lo_bits loop-carried block argument',
        )
        q2_consts = _ui32_const_results(loop_module.operation, 0x20000000)
        q2_cast = [_cast_result_of_const(q) for q in q2_consts]
        expect(
            not any(_same_ssa(yield_operands[1], q) for q in q2_cast),
            'innermost else must not yield the sibling branch q2 constant',
        )
        init_operands = list(for_op.operands)[3:]
        expect(len(init_operands) == 2, 'scf.for should have exactly two iter_arg init operands')
        hi_consts = _ui32_const_results(loop_module.operation, 0x40000000)
        lo_consts = _ui32_const_results(loop_module.operation, 0)
        expect(
            hi_consts and _same_ssa(init_operands[0], _cast_result_of_const(hi_consts[0])),
            'first iter_arg should be hi_bits initialized to 0x40000000',
        )
        expect(
            lo_consts and _same_ssa(init_operands[1], _cast_result_of_const(lo_consts[0])),
            'second iter_arg should be lo_bits initialized to 0',
        )

    # Focused probe 3: static-subscript partial assignment keeps the entry slot.
    slot_text = ast_nested_partial_assign_slot_entry_isolated_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(
        slot_text,
        'AST-rewritten nested static-subscript partial assignment isolation',
    )
    with make_context() as ctx:
        slot_module = Module.parse(slot_text, ctx)
        slot_func = _find_op_anywhere(slot_module.operation, 'func.func')
        expect(slot_func is not None, 'slot isolation probe should contain a func.func op')
        slot_block = slot_func.regions[0].blocks[0]
        zero_consts = _ui32_const_results(slot_module.operation, 0)
        expect(len(zero_consts) >= 1, 'slot isolation probe should materialize the entry 0 constant')
        zero_value = _cast_result_of_const(zero_consts[0])
        q2_consts = _ui32_const_results(slot_module.operation, 0x20000000)
        q2_cast = [_cast_result_of_const(q) for q in q2_consts]
        expect(
            _scf_if_count(slot_block) == 2,
            'slot isolation probe should contain two nested scf.if ops',
        )
        slot_yield = _innermost_else_yield_operands(slot_block)
        expect(len(slot_yield) == 1, 'innermost else of the slot probe should yield one value')
        expect(
            _same_ssa(slot_yield[0], zero_value),
            'innermost else of the slot probe must yield the entry slot value',
        )
        expect(
            not any(_same_ssa(slot_yield[0], q) for q in q2_cast),
            'innermost else of the slot probe must not yield the sibling branch q2 value',
        )

    # Focused probe 4: with-items bind their optional_vars in source order.
    # The second context expression below reads the value produced by the first
    # item, so it must not become a live-in slot.  A subscript target also kills
    # only the slot it overwrites, rather than the whole list binding.
    import ast as _slot_ast
    from ptodsl._ast_rewrite import _SubscriptSlot, _read_before_assignment_slots

    def with_slot_liveness(source, live_after=()):
        with_stmt = _slot_ast.parse(source).body[0]
        return _read_before_assignment_slots(
            [with_stmt], {}, live_after=set(live_after)
        )

    values0 = _SubscriptSlot('values', 0)
    values1 = _SubscriptSlot('values', 1)
    expect(
        not with_slot_liveness(
            'with cm() as values, cm(values[0]):\n    pass'
        ),
        'a later with-item may use the binding produced by an earlier item',
    )
    expect(
        with_slot_liveness('with cm(values[0]) as values:\n    pass') == {values0},
        'a with context expression must still keep an outer slot live-in',
    )
    expect(
        with_slot_liveness(
            'with cm() as values[0]:\n    pass',
            live_after={values0, values1},
        ) == {values1},
        'a subscript with-as target must kill only the overwritten slot',
    )

    # Augmenting the induction variable must not invent a carry for it.
    iv_augassign_text = ast_for_iv_augassign_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(iv_augassign_text, 'AST-rewritten induction-variable augment-assignment')
    expect(
        'iter_args(' not in iv_augassign_text,
        'augmenting the induction variable should not infer a spurious loop carry',
    )

    # Non-Name static_range targets keep trace-time semantics (no scf.for).
    subscript_target_text = ast_static_range_subscript_target_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(subscript_target_text, 'AST-rewritten static-range subscript target')

    # A known non-empty static_range definitely binds its target afterwards, so
    # the rewrite gate must credit it for a later implicit loop carry instead of
    # rejecting it as last-iteration-only (verified at the rewrite level).
    import ast as _ast
    import inspect as _inspect
    import textwrap as _textwrap
    from ptodsl._ast_rewrite import _ControlFlowRewriter as _CFR
    def rewrite_gate(fn):
        gate_src = _textwrap.dedent(_inspect.getsource(fn))
        gate_tree = _ast.parse(gate_src)
        gate_fn = next(
            n
            for n in _ast.walk(gate_tree)
            if isinstance(n, _ast.FunctionDef) and n.name == fn.__name__
        )
        gate_rewriter = _CFR(static_env={})
        gate_body = gate_rewriter.rewrite_block(
            gate_fn.body,
            live_after={'value'},
            bound_on_entry={arg.arg for arg in gate_fn.args.args},
        )
        gate_module = _ast.Module(body=gate_body, type_ignores=[])
        _ast.fix_missing_locations(gate_module)
        return _ast.unparse(gate_module)

    gate_text = rewrite_gate(ast_static_range_nonempty_definite_gate_fn)
    expect(
        'carry(value=value)' in gate_text,
        'known non-empty static_range target should be credited as bound for an implicit carry',
    )
    body_init_text = rewrite_gate(ast_static_range_body_init_definite_gate_fn)
    expect(
        'carry(value=value)' in body_init_text,
        'known non-empty static_range body initialization should make an implicit carry safe',
    )
    empty_orelse_text = rewrite_gate(ast_static_range_empty_orelse_init_gate_fn)
    expect(
        'carry(value=value)' in empty_orelse_text,
        'known empty static_range else initialization should make an implicit carry safe',
    )
    expect_raises(
        PTODSLAstRewriteError,
        lambda: rewrite_gate(ast_static_range_delete_target_gate_fn),
        'last-iteration-only',
    )
    expect_raises(
        PTODSLAstRewriteError,
        lambda: rewrite_gate(ast_static_range_orelse_delete_gate_fn),
        'last-iteration-only',
    )
    del _ast, _inspect, _textwrap, _CFR

    # Unbound partial live-outs must stay on the explicit diagnostics path.
    expect_raises(
        PTODSLAstRewriteError,
        lambda: ast_unbound_partial_liveout_loop_probe.compile(),
        'last-iteration-only',
    )
    expect_raises(
        PTODSLAstRewriteError,
        lambda: ast_unbound_partial_liveout_controlled_loop_probe.compile(),
        'last-iteration-only',
    )
    expect_raises(
        PTODSLAstRewriteError,
        lambda: ast_static_range_empty_orelse_partial_probe.compile(),
        'last-iteration-only',
    )
    expect_raises(
        PTODSLAstRewriteError,
        lambda: ast_del_then_partial_liveout_probe.compile(),
        'last-iteration-only',
    )

    # Definitely-bound partial live-outs (parameters, outer blocks, both branches
    # of a prior conditional, and static subscript slots) are loop carries.
    for probe_text, label in (
        (ast_param_bound_partial_liveout_loop_probe.compile().mlir_text(), 'parameter-bound partial live-out'),
        (ast_outer_bound_partial_liveout_loop_probe.compile().mlir_text(), 'outer-block-bound partial live-out'),
        (ast_both_branch_bound_partial_liveout_loop_probe.compile().mlir_text(), 'both-branches-bound partial live-out'),
        (ast_slot_partial_assign_loop_carry_probe.compile().mlir_text(), 'loop static-subscript partial assignment'),
        (ast_with_entry_bound_partial_liveout_probe.compile().mlir_text(), 'with-entry-bound partial live-out'),
        (ast_with_body_bound_partial_liveout_probe.compile().mlir_text(), 'with-body-bound partial live-out'),
        (ast_with_as_bound_partial_liveout_probe.compile().mlir_text(), 'with-as-bound partial live-out'),
        (ast_for_iv_bound_partial_liveout_probe.compile().mlir_text(), 'for-induction-variable-bound partial live-out'),
    ):
        expect_parse_roundtrip_and_verify(probe_text, 'AST-rewritten ' + label)
        expect('iter_args(' in probe_text, label + ' should lower through scf.for iter_args')


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def expect_raises(exc_type, func, message_substring: Optional[str] = None) -> Exception:
    try:
        func()
    except exc_type as exc:
        if message_substring is not None and message_substring not in str(exc):
            raise AssertionError(
                f"expected {exc_type.__name__} containing {message_substring!r}, got {exc!r}"
            ) from exc
        return exc
    except Exception as exc:
        raise AssertionError(
            f"expected {exc_type.__name__}, got {exc.__class__.__name__}: {exc}"
        ) from exc
    raise AssertionError(f"expected {exc_type.__name__} to be raised")


def expect_parse_roundtrip_and_verify(text: str, label: str) -> None:
    with make_context() as ctx:
        parsed = Module.parse(text, ctx)
        parsed.operation.verify()
        roundtrip_text = str(parsed)
    expect(
        roundtrip_text == text,
        f"{label} should survive Module.parse(...) round-trip without textual drift",
    )


def mlir_op_sequence(text: str) -> list[str]:
    ops = []
    for line in text.splitlines():
        stripped = line.strip()
        match = re.search(r"(?:%[\w#]+(?:\s*:\s*[^=]+)?\s*=\s*)?([a-z][\w]*\.[\w_]+)", stripped)
        if match is not None:
            ops.append(match.group(1))
    return ops


def run_python_snippet(source: str) -> str:
    env = dict(os.environ)
    if "PYTHONPYCACHEPREFIX" not in env:
        env["PYTHONPYCACHEPREFIX"] = "/tmp/ptoas-pycache"
    result = subprocess.run(
        [sys.executable, "-c", source],
        check=True,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def ptodsl_func_stable_symbol_from_subprocess() -> str:
    source = r'''
import re
from ptodsl import pto

@pto.func(returns=pto.i32)
def stable_dtype_return_helper(x: pto.i32):
    return x

@pto.jit(target="a5")
def stable_dtype_symbol_probe(x: pto.i32):
    _ = stable_dtype_return_helper(x)

text = stable_dtype_symbol_probe.compile().mlir_text()
match = re.search(r"func\.func @(stable_dtype_return_helper__ptodsl_[0-9a-f]+)\(", text)
if match is None:
    raise RuntimeError(text)
print(match.group(1))
'''
    return run_python_snippet(source)


expect_raises(
    TypeError,
    lambda: pto.for_(0, 1, step=1, iter_args=(0,)),
    "iter_args",
)


class _FakeTileWithoutValidShape:
    shape = (1, 16)
    valid_shape = None


class _FakeTileWithPartialValidShape:
    shape = (1, 16)
    valid_shape = [1, None]


expect_raises(
    TypeError,
    lambda: pto.tile.load(object(), _FakeTileWithoutValidShape(), offsets=[0, 0]),
    "requires tile valid_shape metadata",
)
expect_raises(
    ValueError,
    lambda: pto.tile.load(object(), _FakeTileWithPartialValidShape(), offsets=[0, 0]),
    "tile.valid_shape[1] is None",
)
expect_raises(
    TypeError,
    lambda: pto.mte_l1_l0a_mx(None, None, 16, 64, transpose=True),
    "transpose",
)
expect_raises(
    TypeError,
    lambda: pto.mte_l1_l0b_mx(None, None, 64, 16, transpose=True),
    "transpose",
)


@pto.jit(target="a5")
def host_vec_copy(
    A_ptr: pto.ptr(pto.f32, "gm"),
    O_ptr: pto.ptr(pto.f32, "gm"),
    rows: pto.i32,
    cols: pto.i32,
    *,
    BLOCK: pto.const_expr = 128,
):
    a_view = pto.make_tensor_view(A_ptr, shape=[rows, cols], strides=[cols, 1])
    o_view = pto.make_tensor_view(O_ptr, shape=[rows, cols], strides=[cols, 1])
    a_tile = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)
    o_tile = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)
    part = pto.partition_view(a_view, offsets=[0, 0], sizes=[rows, cols])
    out = pto.partition_view(o_view, offsets=[0, 0], sizes=[rows, cols])
    pto.tile.load(part, a_tile)
    pto.tile.store(o_tile, out)


@pto.jit(target="a5", backend="vpto", mode="explicit")
def scalar_pipeline_access_modes_probe(
    src: pto.ptr(pto.i32, "gm"),
    dst: pto.ptr(pto.i32, "gm"),
):
    normal = pto.load_scalar(src, 1)
    pto.store_scalar(dst, 2, normal)
    bypass = pto.load_scalar(src, 3, bypass_l1=True)
    pto.store_scalar(dst, 4, bypass, bypass_l1=True)


@pto.jit(target="a5", backend="vpto", mode="explicit")
def scalar_pipeline_bypass_ub_invalid_probe():
    src = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.i32, "ub"))
    pto.load_scalar(src, 0, bypass_l1=True)


@pto.jit(target="a5", backend="vpto", mode="explicit")
def scalar_pipeline_bypass_float_invalid_probe(
    src: pto.ptr(pto.f32, "gm"),
):
    pto.load_scalar(src, 0, bypass_l1=True)


@pto.jit(target="a5", mode="explicit")
def host_vec_copy_explicit(
    A_ptr: pto.ptr(pto.f32, "gm"),
    O_ptr: pto.ptr(pto.f32, "gm"),
    rows: pto.i32,
    cols: pto.i32,
    *,
    BLOCK: pto.const_expr = 128,
):
    a_view = pto.make_tensor_view(A_ptr, shape=[rows, cols], strides=[cols, 1])
    o_view = pto.make_tensor_view(O_ptr, shape=[rows, cols], strides=[cols, 1])
    a_tile = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)
    o_tile = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)
    part = pto.partition_view(a_view, offsets=[0, 0], sizes=[rows, cols])
    out = pto.partition_view(o_view, offsets=[0, 0], sizes=[rows, cols])
    pto.tile.load(part, a_tile)
    pto.tile.store(o_tile, out)


@pto.jit(target="a5", mode="explicit")
def host_vec_copy_explicit_addr(
    A_ptr: pto.ptr(pto.f32, "gm"),
    O_ptr: pto.ptr(pto.f32, "gm"),
    rows: pto.i32,
    cols: pto.i32,
    *,
    BLOCK: pto.const_expr = 128,
):
    a_view = pto.make_tensor_view(A_ptr, shape=[rows, cols], strides=[cols, 1])
    o_view = pto.make_tensor_view(O_ptr, shape=[rows, cols], strides=[cols, 1])
    a_tile = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32, addr=0)
    o_tile = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32, addr=4096)
    part = pto.partition_view(a_view, offsets=[0, 0], sizes=[rows, cols])
    out = pto.partition_view(o_view, offsets=[0, 0], sizes=[rows, cols])
    pto.tile.load(part, a_tile)
    pto.tile.store(o_tile, out)


@pto.jit(target="a5", backend="emitc")
def host_vec_copy_emitc(
    A_ptr: pto.ptr(pto.f32, "gm"),
    O_ptr: pto.ptr(pto.f32, "gm"),
    rows: pto.i32,
    cols: pto.i32,
):
    a_view = pto.make_tensor_view(A_ptr, shape=[rows, cols], strides=[cols, 1])
    o_view = pto.make_tensor_view(O_ptr, shape=[rows, cols], strides=[cols, 1])
    a_tile = pto.alloc_tile(shape=[1, 16], dtype=pto.f32)
    o_tile = pto.alloc_tile(shape=[1, 16], dtype=pto.f32)
    part = pto.partition_view(a_view, offsets=[0, 0], sizes=[rows, cols])
    out = pto.partition_view(o_view, offsets=[0, 0], sizes=[rows, cols])
    pto.tile.load(part, a_tile)
    pto.tile.store(o_tile, out)


@pto.jit(target="a5", entry=False, backend="vpto")
def non_entry_metadata_probe(
    A_ptr: pto.ptr(pto.f32, "gm"),
    rows: pto.i32,
):
    a_view = pto.make_tensor_view(A_ptr, shape=[rows, 1], strides=[1, 1])
    _ = a_view


@pto.jit(target="a5", entry=False, backend="emitc", kernel_kind="vector")
def emitc_vector_kernel_module_metadata_probe(
    src_gm: pto.ptr(pto.f32, "gm"),
    dst_gm: pto.ptr(pto.f32, "gm"),
    row: pto.i32,
):
    _ = src_gm
    _ = dst_gm
    _ = row


@pto.jit(target="a5", backend="emitc")
def emitc_entry_calls_emitc_vector_kernel_module_metadata_probe(
    src_gm: pto.ptr(pto.f32, "gm"),
    dst_gm: pto.ptr(pto.f32, "gm"),
    rows: pto.i32,
):
    with pto.for_(0, rows, step=1) as row:
        emitc_vector_kernel_module_metadata_probe(src_gm, dst_gm, row)


@pto.jit(target="a5")
def explicit_layout_tensor_view_probe(
    K_ptr: pto.ptr(pto.f16, "gm"),
    rows: pto.i32,
    cols: pto.i32,
):
    k_view = pto.make_tensor_view(K_ptr, shape=[rows, cols], strides=[1, rows], layout="DN")
    _ = k_view


@pto.jit(target="a5", entry=False, backend="vpto")
def helper_device_abi_surface_probe(
    tile: pto.Tile,
    view: pto.TensorView,
    part: pto.PartitionTensorView,
    ub_ptr: pto.ptr(pto.f32, "ub"),
    rows: pto.i32,
):
    _ = tile
    _ = view
    _ = part
    _ = ub_ptr
    _ = rows


@pto.jit(target="a5", entry=False, backend="vpto")
def kernel_module_return_probe(
    ptr: pto.ptr(pto.f32, "gm"),
    rows: pto.i32,
):
    return rows


@pto.jit(target="a5", entry=False, backend="vpto")
def process_tile_module(
    a_tile: pto.Tile,
    b_tile: pto.Tile,
    o_tile: pto.Tile,
    rows: pto.i32,
    cols: pto.i32,
):
    with pto.tileop():
        vec = pto.elements_per_vreg(pto.f32)
        initial_remained = cols
        with pto.for_(0, rows, step=1) as r:
            col_loop = pto.for_(0, cols, step=vec).carry(remained=initial_remained)
            with col_loop:
                c = col_loop.iv
                remained = col_loop.remained
                mask, remained = pto.make_mask(pto.f32, remained)
                a_vec = pto.vlds(a_tile[r, c:])
                b_vec = pto.vlds(b_tile[r, c:])
                o_vec = pto.vadd(a_vec, b_vec, mask)
                pto.vsts(o_vec, o_tile[r, c:], mask)
                col_loop.update(remained=remained)


@pto.jit(target="a5", entry=False, backend="vpto", mode="explicit", insert_sync=False)
def explicit_vpto_kernel_module(
    a_tile: pto.Tile,
    o_tile: pto.Tile,
    cols: pto.i32,
):
    with pto.tileop():
        remained = cols
        vec = pto.elements_per_vreg(pto.f32)
        loop = pto.for_(0, cols, step=vec).carry(remained=remained)
        with loop:
            c = loop.iv
            mask, remained = pto.make_mask(pto.f32, loop.remained)
            a_vec = pto.vlds(a_tile[0, c:])
            pto.vsts(a_vec, o_tile[0, c:], mask)
            loop.update(remained=remained)


@pto.jit(target="a5", entry=False, backend="vpto", mode="explicit", insert_sync=False)
def process_row_ptr_kernel_module(
    src_gm: pto.ptr(pto.f32, "gm"),
    dst_gm: pto.ptr(pto.f32, "gm"),
    row: pto.i32,
):
    c0_i64 = pto.const(0, dtype=pto.i64)
    row_offset = row * 16
    src_row = pto.addptr(src_gm, row_offset)
    dst_row = pto.addptr(dst_gm, row_offset)
    ub_ptr = pto.castptr(c0_i64, pto.ptr(pto.f32, "ub"))

    pto.get_buf(pto.Pipe.MTE2, 0)
    pto.mte_gm_ub(src_row, ub_ptr, 0, 64, nburst=(1, 64, 64))
    pto.rls_buf(pto.Pipe.MTE2, 0)

    pto.get_buf(pto.Pipe.MTE3, 0)
    pto.mte_ub_gm(ub_ptr, dst_row, 64, nburst=(1, 64, 64))
    pto.rls_buf(pto.Pipe.MTE3, 0)
    pto.pipe_barrier(pto.Pipe.ALL)


@pto.jit(target="a5", entry=False, backend="vpto", mode="explicit", insert_sync=False)
def dynamic_buf_kernel_module(
    src_gm: pto.ptr(pto.f32, "gm"),
    dst_gm: pto.ptr(pto.f32, "gm"),
    row: pto.i32,
):
    c0_i64 = pto.const(0, dtype=pto.i64)
    row_offset = row * 16
    src_row = pto.addptr(src_gm, row_offset)
    dst_row = pto.addptr(dst_gm, row_offset)
    ub_ptr = pto.castptr(c0_i64, pto.ptr(pto.f32, "ub"))

    # Compute dynamic buf-id from the row index: row & 1
    buf_id = row & 1
    pto.get_buf(pto.Pipe.MTE2, buf_id, 0)
    pto.mte_gm_ub(src_row, ub_ptr, 0, 64, nburst=(1, 64, 64))
    pto.rls_buf(pto.Pipe.MTE2, buf_id, 0)

    buf_id2 = row & 1
    pto.get_buf(pto.Pipe.MTE3, buf_id2, 0)
    pto.mte_ub_gm(ub_ptr, dst_row, 64, nburst=(1, 64, 64))
    pto.rls_buf(pto.Pipe.MTE3, buf_id2, 0)
    pto.pipe_barrier(pto.Pipe.ALL)


@pto.jit(target="a5", entry=False, backend="vpto", mode="explicit", insert_sync=False)
def ast_rewrite_kernel_module_probe(
    src_ptr: pto.ptr(pto.f32, "ub"),
    dst_ptr: pto.ptr(pto.f32, "ub"),
    rows: pto.i32,
    cols: pto.i32,
):
    lanes = pto.elements_per_vreg(pto.f32)
    for row in range(0, rows, 1):
        row_base = row * cols
        remained = cols
        for col in range(0, cols, lanes):
            mask, remained = pto.make_mask(pto.f32, remained)
            vec = pto.vlds(src_ptr, row_base + col, dist="NORM")
            pto.vsts(vec, dst_ptr, row_base + col, mask, dist="NORM_B32")


@pto.jit(target="a5")
def entry_calls_kernel_module_probe(
    A_ptr: pto.ptr(pto.f32, "gm"),
    B_ptr: pto.ptr(pto.f32, "gm"),
    O_ptr: pto.ptr(pto.f32, "gm"),
    rows: pto.i32,
    cols: pto.i32,
    *,
    BLOCK: pto.const_expr = 128,
):
    a_view = pto.make_tensor_view(A_ptr, shape=[rows, cols], strides=[cols, 1])
    b_view = pto.make_tensor_view(B_ptr, shape=[rows, cols], strides=[cols, 1])
    o_view = pto.make_tensor_view(O_ptr, shape=[rows, cols], strides=[cols, 1])

    a_tile = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)
    b_tile = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)
    o_tile = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)

    with pto.for_(0, rows, step=1) as row:
        a_part = pto.partition_view(a_view, offsets=[row, 0], sizes=[1, cols])
        b_part = pto.partition_view(b_view, offsets=[row, 0], sizes=[1, cols])
        o_part = pto.partition_view(o_view, offsets=[row, 0], sizes=[1, cols])

        pto.tile.load(a_part, a_tile)
        pto.tile.load(b_part, b_tile)
        process_tile_module(a_tile, b_tile, o_tile, 1, cols)
        pto.tile.store(o_tile, o_part)


@pto.jit(target="a5", backend="emitc")
def emitc_entry_calls_vpto_kernel_module_probe(
    A_ptr: pto.ptr(pto.f32, "gm"),
    O_ptr: pto.ptr(pto.f32, "gm"),
    rows: pto.i32,
):
    a_view = pto.make_tensor_view(A_ptr, shape=[rows, 16], strides=[16, 1])
    o_view = pto.make_tensor_view(O_ptr, shape=[rows, 16], strides=[16, 1])
    a_tile = pto.alloc_tile(shape=[1, 16], dtype=pto.f32)
    o_tile = pto.alloc_tile(shape=[1, 16], dtype=pto.f32)

    with pto.for_(0, rows, step=1) as row:
        a_part = pto.partition_view(a_view, offsets=[row, 0], sizes=[1, 16])
        o_part = pto.partition_view(o_view, offsets=[row, 0], sizes=[1, 16])
        pto.tile.load(a_part, a_tile)
        pto.tile.adds(a_tile, 1.0, o_tile)
        pto.tile.store(o_tile, o_part)
        process_row_ptr_kernel_module(A_ptr, O_ptr, row)


@pto.jit(target="a5", backend="emitc")
def emitc_entry_calls_dynamic_buf_kernel_module_probe(
    A_ptr: pto.ptr(pto.f32, "gm"),
    O_ptr: pto.ptr(pto.f32, "gm"),
    rows: pto.i32,
):
    a_view = pto.make_tensor_view(A_ptr, shape=[rows, 16], strides=[16, 1])
    o_view = pto.make_tensor_view(O_ptr, shape=[rows, 16], strides=[16, 1])
    a_tile = pto.alloc_tile(shape=[1, 16], dtype=pto.f32)
    o_tile = pto.alloc_tile(shape=[1, 16], dtype=pto.f32)

    with pto.for_(0, rows, step=1) as row:
        a_part = pto.partition_view(a_view, offsets=[row, 0], sizes=[1, 16])
        o_part = pto.partition_view(o_view, offsets=[row, 0], sizes=[1, 16])
        pto.tile.load(a_part, a_tile)
        pto.tile.adds(a_tile, 1.0, o_tile)
        pto.tile.store(o_tile, o_part)
        dynamic_buf_kernel_module(A_ptr, O_ptr, row)


def emitc_vpto_kernel_module_callsite_simd_helper(
    src_tile: pto.Tile,
    dst_tile: pto.Tile,
    cols: pto.i32,
):
    explicit_vpto_kernel_module(src_tile, dst_tile, cols)


@pto.jit(target="a5", backend="emitc")
def emitc_entry_calls_vpto_kernel_module_via_decorated_simd_probe(
    A_ptr: pto.ptr(pto.f32, "gm"),
    O_ptr: pto.ptr(pto.f32, "gm"),
    rows: pto.i32,
):
    a_view = pto.make_tensor_view(A_ptr, shape=[rows, 16], strides=[16, 1])
    o_view = pto.make_tensor_view(O_ptr, shape=[rows, 16], strides=[16, 1])
    a_tile = pto.alloc_tile(shape=[1, 16], dtype=pto.f32)
    o_tile = pto.alloc_tile(shape=[1, 16], dtype=pto.f32)

    with pto.for_(0, rows, step=1) as row:
        a_part = pto.partition_view(a_view, offsets=[row, 0], sizes=[1, 16])
        o_part = pto.partition_view(o_view, offsets=[row, 0], sizes=[1, 16])
        pto.tile.load(a_part, a_tile)
        emitc_vpto_kernel_module_callsite_simd_helper(a_tile, o_tile, 16)
        pto.tile.store(o_tile, o_part)


@pto.jit(target="a5", backend="emitc")
def entry_calls_ast_rewrite_kernel_module_probe(
    A_ptr: pto.ptr(pto.f32, "gm"),
    O_ptr: pto.ptr(pto.f32, "gm"),
    rows: pto.i32,
    cols: pto.i32,
):
    a_view = pto.make_tensor_view(A_ptr, shape=[rows, cols], strides=[cols, 1])
    o_view = pto.make_tensor_view(O_ptr, shape=[rows, cols], strides=[cols, 1])
    a_tile = pto.alloc_tile(shape=[1, 128], dtype=pto.f32)
    o_tile = pto.alloc_tile(shape=[1, 128], dtype=pto.f32)
    part = pto.partition_view(a_view, offsets=[0, 0], sizes=[rows, cols])
    out = pto.partition_view(o_view, offsets=[0, 0], sizes=[rows, cols])
    pto.tile.load(part, a_tile)
    ast_rewrite_kernel_module_probe(a_tile.as_ptr(), o_tile.as_ptr(), rows, cols)
    pto.tile.store(o_tile, out)


@pto.jit(target="a5")
def entry_calls_kernel_module_multiple_abi_probe():
    src16 = pto.alloc_tile(shape=[1, 16], dtype=pto.f32)
    tmp16 = pto.alloc_tile(shape=[1, 16], dtype=pto.f32)
    dst16 = pto.alloc_tile(shape=[1, 16], dtype=pto.f32)
    process_tile_module(src16, tmp16, dst16, 1, 16)

    src32 = pto.alloc_tile(shape=[1, 32], dtype=pto.f32)
    tmp32 = pto.alloc_tile(shape=[1, 32], dtype=pto.f32)
    dst32 = pto.alloc_tile(shape=[1, 32], dtype=pto.f32)
    process_tile_module(src32, tmp32, dst32, 1, 32)


@pto.jit(target="a5")
def pointer_runtime_shape_specialization_probe(
    x_ptr: pto.ptr(pto.f32, "gm"),
    rows: pto.i32,
    cols: pto.i32,
    row_stride: pto.i32,
    *,
    BLOCK: pto.const_expr = 128,
):
    x_view = pto.make_tensor_view(x_ptr, shape=[rows, cols], strides=[row_stride, 1])
    x_part = pto.partition_view(x_view, offsets=[0, 0], sizes=[rows, cols])
    x_tile = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32, valid_shape=[1, cols])
    pto.tile.load(x_part, x_tile)


@pto.jit(target="a5")
def tile_transfer_surface_probe(
    A_ptr: pto.ptr(pto.f32, "gm"),
    O_ptr: pto.ptr(pto.f32, "gm"),
    rows: pto.i32,
    cols: pto.i32,
    *,
    BLOCK: pto.const_expr = 128,
):
    a_view = pto.make_tensor_view(A_ptr, shape=[rows, cols], strides=[cols, 1])
    o_view = pto.make_tensor_view(O_ptr, shape=[rows, cols], strides=[cols, 1])
    a_tile = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32, valid_shape=[rows, cols])
    o_tile = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32, valid_shape=[rows, cols])
    pto.tile.load(a_view, a_tile)
    pto.tile.store(o_tile, o_view)


@pto.jit(target="a5", insert_sync=False)
def host_vec_copy_no_insert_sync(
    A_ptr: pto.ptr(pto.f32, "gm"),
    O_ptr: pto.ptr(pto.f32, "gm"),
    rows: pto.i32,
    cols: pto.i32,
):
    a_view = pto.make_tensor_view(A_ptr, shape=[rows, cols], strides=[cols, 1])
    o_view = pto.make_tensor_view(O_ptr, shape=[rows, cols], strides=[cols, 1])
    a_tile = pto.alloc_tile(shape=[1, 16], dtype=pto.f32)
    o_tile = pto.alloc_tile(shape=[1, 16], dtype=pto.f32)
    part = pto.partition_view(a_view, offsets=[0, 0], sizes=[rows, cols])
    out = pto.partition_view(o_view, offsets=[0, 0], sizes=[rows, cols])
    pto.tile.load(part, a_tile)
    pto.tile.store(o_tile, out)


@pto.jit(target="a5", mode="explicit", insert_sync=True)
def host_vec_copy_explicit_insert_sync(
    A_ptr: pto.ptr(pto.f32, "gm"),
    O_ptr: pto.ptr(pto.f32, "gm"),
    rows: pto.i32,
    cols: pto.i32,
):
    a_view = pto.make_tensor_view(A_ptr, shape=[rows, cols], strides=[cols, 1])
    o_view = pto.make_tensor_view(O_ptr, shape=[rows, cols], strides=[cols, 1])
    a_tile = pto.alloc_tile(shape=[1, 16], dtype=pto.f32)
    o_tile = pto.alloc_tile(shape=[1, 16], dtype=pto.f32)
    part = pto.partition_view(a_view, offsets=[0, 0], sizes=[rows, cols])
    out = pto.partition_view(o_view, offsets=[0, 0], sizes=[rows, cols])
    pto.tile.load(part, a_tile)
    pto.tile.store(o_tile, out)


@pto.jit(target="a5")
def runtime_metadata_kernel(
    A_ptr: pto.ptr(pto.f32, "gm"),
    O_ptr: pto.ptr(pto.f32, "gm"),
    rows: pto.i32,
    cols: pto.i32,
    row_stride: pto.i32,
    col_stride: pto.i32,
    *,
    BLOCK: pto.const_expr = 128,
):
    a_view = pto.make_tensor_view(A_ptr, shape=[rows, cols], strides=[row_stride, col_stride])
    o_view = pto.make_tensor_view(O_ptr, shape=[rows, cols], strides=[row_stride, col_stride])
    a_tile = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32, valid_shape=[rows, cols])
    o_tile = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32, valid_shape=[rows, cols])
    part = pto.partition_view(a_view, offsets=[0, 0], sizes=[rows, cols])
    out = pto.partition_view(o_view, offsets=[0, 0], sizes=[rows, cols])
    pto.tile.load(part, a_tile)
    pto.tile.store(o_tile, out)


@pto.jit(target="a5", mode="explicit")
def authored_addr_tile_surface_probe(
    rows: pto.i32,
    cols: pto.i32,
):
    tile = pto.alloc_tile(shape=[1, 128], dtype=pto.f32, addr=0, valid_shape=[rows, cols])
    _ = tile


@pto.jit(target="a5", mode="explicit")
def dynamic_addr_tile_surface_probe(
    rows: pto.i32,
    cols: pto.i32,
):
    tile = pto.alloc_tile(
        shape=[1, 128],
        dtype=pto.f32,
        addr=scalar.index_cast(pto.i64, scalar.index_cast(rows)),
        valid_shape=[rows, cols],
    )
    _ = tile


@pto.jit(target="a5")
def tile_sort_gather_surface_probe():
    src = pto.alloc_tile(shape=[1, 32], dtype=pto.f32)
    idx = pto.alloc_tile(shape=[1, 32], dtype=pto.ui32)
    sort = pto.alloc_tile(shape=[1, 64], dtype=pto.f32)
    tmp = pto.alloc_tile(shape=[1, 64], dtype=pto.f32)
    gather_scores = pto.alloc_tile(shape=[1, 32], dtype=pto.f32)
    gather_indices = pto.alloc_tile(shape=[1, 32], dtype=pto.f32)

    pto.tile.sort32(src, idx, sort)
    pto.tile.mrgsort(sort, tmp, pto.const(64, dtype=pto.i32))
    pto.tile.gather(tmp, gather_scores, mask_pattern="P0101", axis="row")
    pto.tgather(tmp, gather_indices, mask_pattern="P1010", axis="row")


@pto.jit(target="a5")
def tile_ci_surface_probe():
    dst_i32 = pto.alloc_tile(shape=[1, 64], dtype=pto.i32, valid_shape=[1, 64])
    pto.tile.ci(0, dst_i32)

    dst_ui32 = pto.alloc_tile(shape=[1, 64], dtype=pto.ui32, valid_shape=[1, 64])
    pto.tile.ci(pto.ui32(0), dst_ui32, descending=True)


@pto.jit(target="a5")
def tile_surface_compute_probe():
    lhs = pto.alloc_tile(shape=[2, 16], dtype=pto.f32)
    rhs = pto.alloc_tile(shape=[2, 16], dtype=pto.f32)
    out = pto.alloc_tile(shape=[2, 16], dtype=pto.f32)
    cmp_out = pto.alloc_tile(shape=[2, 32], dtype=pto.i8, valid_shape=[2, 16])
    reshape_src = pto.alloc_tile(shape=[8, 64], dtype=pto.f32, valid_shape=[8, 64])

    pto.tile.expands(1.0, lhs)
    pto.tile.expands(2.0, rhs)
    pto.tile.add(lhs, rhs, out)
    pto.tile.adds(out, 3.0, out)
    pto.tile.cmps(out, 0.0, cmp_out, cmp_mode=pto.CmpMode.GT)
    reshape_1d = pto.tile.reshape(reshape_src, shape=[512])
    reshape_col = pto.tile.reshape(reshape_src, shape=[8, 64], blayout="ColMajor")
    expect(reshape_1d.shape == (512,), "pto.tile.reshape(..., shape=[...]) should expose the authored logical result shape")
    expect(reshape_1d.physical_shape == (1, 512), "rank-1 pto.tile.reshape(...) should materialize the authored 1D physical shape")
    expect(str(reshape_1d.dtype) == str(reshape_src.dtype), "pto.tile.reshape(..., dtype omitted) should preserve the source element type")
    expect(reshape_1d.memory_space == reshape_src.memory_space, "pto.tile.reshape(...) should preserve the source memory space")
    expect(
        reshape_1d.static_valid_shape is None,
        "pto.tile.reshape(...) should not preserve or infer reshaped valid_shape metadata on the current public surface",
    )
    expect(reshape_col.shape == (8, 64), "same-rank pto.tile.reshape(...) should preserve the authored logical result rank")
    expect(
        reshape_col.static_valid_shape is None,
        "same-rank pto.tile.reshape(...) should also leave valid_shape metadata unset on the current public surface",
    )
    expect(
        "valid_shape" in reshape_col.surface_metadata and reshape_col.surface_metadata["valid_shape"] is None,
        "tile.reshape result should still expose valid_shape surface metadata even when it is unset",
    )
    _ = reshape_1d
    _ = reshape_col


@pto.jit(target="a5")
def tile_surface_window_matmul_probe():
    src_mat = pto.alloc_tile(
        shape=[64, 64],
        dtype=pto.f16,
        memory_space=pto.MemorySpace.MAT,
        valid_shape=[64, 64],
    )
    dst_mat = pto.alloc_tile(
        shape=[64, 64],
        dtype=pto.f32,
        memory_space=pto.MemorySpace.MAT,
        blayout="ColMajor",
        slayout="RowMajor",
        valid_shape=[64, 64],
    )
    lhs_l0a = pto.alloc_tile(
        shape=[16, 16],
        dtype=pto.f16,
        memory_space=pto.MemorySpace.LEFT,
        blayout="ColMajor",
        slayout="RowMajor",
        valid_shape=[16, 16],
    )
    rhs_l0b = pto.alloc_tile(
        shape=[16, 16],
        dtype=pto.f16,
        memory_space=pto.MemorySpace.RIGHT,
        blayout="RowMajor",
        slayout="ColMajor",
        valid_shape=[16, 16],
    )
    acc_prev = pto.alloc_tile(
        shape=[16, 16],
        dtype=pto.f32,
        memory_space=pto.MemorySpace.ACC,
        blayout="ColMajor",
        slayout="RowMajor",
        valid_shape=[16, 16],
    )
    acc_insert = pto.alloc_tile(
        shape=[16, 16],
        dtype=pto.f32,
        memory_space=pto.MemorySpace.ACC,
        blayout="ColMajor",
        slayout="RowMajor",
        valid_shape=[16, 16],
    )
    acc_out = pto.alloc_tile(
        shape=[16, 16],
        dtype=pto.f32,
        memory_space=pto.MemorySpace.ACC,
        blayout="ColMajor",
        slayout="RowMajor",
        valid_shape=[16, 16],
    )

    pto.tile.extract(src_mat, lhs_l0a, 8, 16)
    pto.tile.insert(acc_insert, dst_mat, 0, 32)
    pto.tile.matmul(lhs_l0a, rhs_l0b, acc_out)
    pto.tile.matmul_acc(acc_prev, lhs_l0a, rhs_l0b, acc_out)


SUBKERNEL_OBSERVATIONS = []
INLINE_SUBKERNEL_SCOPE_OBSERVATIONS = []


@pto.tileop
def nested_simd_probe():
    session = current_session()
    frame = session.current_subkernel
    SUBKERNEL_OBSERVATIONS.append((frame.role, frame.symbol_name, session.subkernel_stack_depth))
    tileop_noop_simt_probe[1, 1, 1]()


@pto.tileop
def top_level_cube_probe():
    session = current_session()
    frame = session.current_subkernel
    SUBKERNEL_OBSERVATIONS.append((frame.role, frame.symbol_name, session.subkernel_stack_depth))
    tileop_noop_simt_probe[1, 1, 1]()


@pto.tileop
def top_level_simd_probe():
    session = current_session()
    frame = session.current_subkernel
    SUBKERNEL_OBSERVATIONS.append((frame.role, frame.symbol_name, session.subkernel_stack_depth))
    tileop_noop_simt_probe[1, 1, 1]()


@pto.jit(target="a5")
def shared_subkernel_lowering_probe(*, TRACE_TOKEN: pto.const_expr = 0):
    top_level_cube_probe()
    top_level_simd_probe()
    nested_simd_probe()


@pto.jit(target="a5", kernel_kind="vector")
def explicit_vector_inline_tileop_probe(*, TRACE_TOKEN: pto.const_expr = 0):
    with pto.tileop():
        pto.vbr(1.0)


@pto.jit(target="a5", mode="explicit")
def inline_subkernel_scope_probe(*, TRACE_TOKEN: pto.const_expr = 0):
    session = current_session()
    meta_tile = pto.alloc_tile(shape=[1, 8], dtype=pto.i32, valid_shape=[1, 1])

    with pto.simt():
        frame = session.current_subkernel
        INLINE_SUBKERNEL_SCOPE_OBSERVATIONS.append((frame.role, frame.symbol_name, session.subkernel_stack_depth))
        scalar.store(0, meta_tile.as_ptr() + 0)
    with pto.tileop():
        frame = session.current_subkernel
        INLINE_SUBKERNEL_SCOPE_OBSERVATIONS.append((frame.role, frame.symbol_name, session.subkernel_stack_depth))
        pto.vbr(1.0)


@pto.jit(target="a5", mode="explicit")
def inline_simt_launch_dims_probe(
    gm: pto.ptr(pto.i32, "gm"),
    *,
    TRACE_TOKEN: pto.const_expr = 0,
):
    with pto.simt(32, 2, 1):
        tid = pto.get_tid_x()
        pto.stg(tid, gm, scalar.index_cast(tid))


@pto.jit(target="a5", mode="explicit")
def inline_simt_dynamic_launch_dim_probe(*, TRACE_TOKEN: pto.const_expr = 0):
    dynamic_dim = pto.const(32, dtype=pto.i32)
    with pto.simt(dynamic_dim, 1, 1):
        pto.get_tid_x()


@pto.simt
def simt_tid_probe():
    pto.get_tid_x()
    pto.get_tid_y()
    pto.get_tid_z()


@pto.simt(max_threads=1, ast_rewrite=False)
def tileop_noop_simt_probe():
    pass


@pto.simt
def simt_query_probe():
    pto.get_tid()
    pto.get_block_dim()
    pto.get_grid_dim()
    pto.get_block_idx_x()
    pto.get_block_idx_y()
    pto.get_block_idx_z()
    pto.get_veccoreid()
    pto.get_clock32()
    pto.get_clock64()
    pto.get_laneid()
    pto.get_lanemask_eq()
    pto.get_lanemask_le()
    pto.get_lanemask_lt()
    pto.get_lanemask_ge()
    pto.get_lanemask_gt()


@pto.simt
def simt_grouped_query_probe():
    tid_x, tid_y, tid_z = pto.get_tid()
    block_x, block_y, block_z = pto.get_block_dim()
    grid_x, grid_y, grid_z = pto.get_grid_dim()
    pto.keep(tid_x, slot=0)
    pto.keep(tid_y, slot=1)
    pto.keep(tid_z, slot=2)
    pto.keep(block_x, slot=3)
    pto.keep(block_y, slot=4)
    pto.keep(block_z, slot=5)
    pto.keep(grid_x, slot=6)
    pto.keep(grid_y, slot=7)
    pto.keep(grid_z, slot=8)


@pto.simt(max_threads=256)
def simt_resource_attr_probe():
    pto.get_tid_x()


@pto.simt
def simt_collective_math_probe():
    lane = pto.get_laneid()
    pred = pto.const(1, dtype=pto.i1)

    pto.vote_all(pred)
    pto.vote_any(pred)
    pto.vote_uni(pred)
    pto.vote_ballot(pred)

    pto.shuffle_idx(lane, lane, width=32)
    pto.shuffle_up(lane, 1, width=32)
    pto.shuffle_down(lane, 1, width=32)
    pto.shuffle_bfly(lane, 1, width=32)

    pto.redux_add(lane, signedness="signed")
    pto.redux_max(lane, signedness="signed")
    pto.redux_min(lane, signedness="signed")

    pto.prmt(lane, lane, lane)
    pto.mulhi(lane, lane, signedness="signed")
    pto.mul_i32toi64(lane, lane, signedness="unsigned")

    as_f32 = pto.convert(lane, pto.f32, rounding="r", saturation="nosat", signedness="signed")
    as_f16 = pto.const(1.0, dtype=pto.f16)
    pto.convert(as_f32, pto.i32, rounding="z", saturation="sat", signedness="signed")
    pto.absf(as_f32)
    pto.sqrt(as_f32)
    pto.exp(as_f32)
    pto.log(as_f32)
    pto.sin(as_f32)
    pto.cos(as_f32)
    pto.pow(as_f32, as_f32)
    pto.ceil(as_f32)
    pto.floor(as_f32)
    pto.rint(as_f32)
    pto.round(as_f32)
    pto.fmin(as_f32, as_f32)
    pto.fmax(as_f32, as_f32)
    pto.fmin(as_f16, as_f16)
    pto.fmax(as_f16, as_f16)
    pto.fma(as_f32, as_f32, as_f32)


@pto.simt
def simt_memory_atomic_probe(
    gm: pto.ptr(pto.i32, "gm"),
):
    idx = scalar.index_cast(pto.get_tid_x())
    value = pto.ldg(gm, idx, l1cache="cache", l2cache="nmfv")
    pto.stg(value, gm, idx, l1cache="uncache", l2cache="wtsred")
    pto.stg(1, gm, idx)

    old = pto.atomic_add(gm, value, l2cache="nmfv", signedness="signed")
    pto.atomic_exch(gm, value, signedness="signed")
    pto.atomic_sub(gm, value, signedness="signed")
    pto.atomic_min(gm, value, signedness="signed")
    pto.atomic_max(gm, value, signedness="signed")
    pto.atomic_and(gm, value, signedness="unsigned")
    pto.atomic_or(gm, value, signedness="unsigned")
    pto.atomic_xor(gm, value, signedness="unsigned")
    pto.atomic_cas(gm, old, value, signedness="signed")

    pto.syncthreads()
    pto.threadfence()
    pto.threadfence_block()


@pto.simt
def simt_fp8_ext_ldg_stg_probe(
    gm_f8e4x4: pto.ptr(pto.f8e4m3x4, "gm"),
    gm_f8e4x8: pto.ptr(pto.f8e4m3x8, "gm"),
    gm_f8e5x4: pto.ptr(pto.f8e5m2x4, "gm"),
    gm_f8e5x8: pto.ptr(pto.f8e5m2x8, "gm"),
):
    idx = scalar.index_cast(pto.get_tid_x())
    v_f8e4x4 = pto.ldg(gm_f8e4x4, idx)
    v_f8e4x8 = pto.ldg(gm_f8e4x8, idx)
    v_f8e5x4 = pto.ldg(gm_f8e5x4, idx)
    v_f8e5x8 = pto.ldg(gm_f8e5x8, idx)
    pto.stg(v_f8e4x4, gm_f8e4x4, idx)
    pto.stg(v_f8e4x8, gm_f8e4x8, idx)
    pto.stg(v_f8e5x4, gm_f8e5x4, idx)
    pto.stg(v_f8e5x8, gm_f8e5x8, idx)


@pto.simt
def simt_specialized_i32_ptr_probe(ptr: pto.ptr(pto.i32, "gm")):
    value = scalar.load(ptr)
    _ = value


@pto.simt
def simt_specialized_f32_ptr_probe(ptr: pto.ptr(pto.f32, "gm")):
    value = scalar.load(ptr)
    _ = value


@pto.simt
def simt_specialized_flag_probe(*, FLAG):
    if FLAG:
        pto.get_tid_x()
    else:
        pto.get_tid_y()


@pto.simt
def simt_keep_stage():
    pto.keep(pto.get_tid_x(), slot=0)


@pto.simt
def simt_resume_stage(gm: pto.ptr(pto.i32, "gm")):
    resumed = pto.resume(pto.i32, slot=0)
    idx = scalar.index_cast(pto.get_tid_x())
    scalar.store(resumed, gm, idx)


@pto.simt
def simt_invalid_redux_signedness_probe():
    pto.redux_max(pto.get_laneid())


@pto.simt
def simt_invalid_convert_signedness_probe():
    pto.convert(pto.get_laneid(), pto.f32, rounding="r", saturation="nosat")


@pto.simt
def simt_invalid_atomic_signedness_probe(gm: pto.ptr(pto.f32, "gm")):
    value = pto.ldg(gm, 0)
    pto.atomic_add(gm, value, signedness="signed")


@pto.tileop
def ast_subkernel_runtime_for_helper(inp: pto.Tile, out: pto.Tile, rows: pto.i32):
    mask, _ = pto.make_mask(pto.f32, pto.const(64, dtype=pto.i32))
    for row in range(0, rows, 1):
        vec = pto.vlds(inp[row, 0:])
        pto.vsts(vec, out[row, 0:], mask)


@pto.jit(target="a5")
def simt_helper_lowering_probe(*, TRACE_TOKEN: pto.const_expr = 0):
    simt_tid_probe()
    simt_tid_probe()


@pto.simt
def alloc_buffer_local_helper():
    _ = pto.alloc_buffer((32,), pto.f32)


@pto.jit(target="a5", mode="explicit")
def alloc_buffer_local_probe():
    alloc_buffer_local_helper()


@pto.jit(target="a5", mode="explicit")
def alloc_buffer_outside_simt_probe():
    _ = pto.alloc_buffer((32,), pto.f32)


@pto.tileop
def alloc_buffer_tileop_helper():
    _ = pto.alloc_buffer((32,), pto.f32)
    tileop_noop_simt_probe[1, 1, 1]()


@pto.jit(target="a5", mode="explicit", kernel_kind="vector")
def alloc_buffer_tileop_helper_probe():
    alloc_buffer_tileop_helper()


@pto.jit(target="a5", mode="explicit")
def alloc_buffer_inline_simt_probe():
    with pto.simt(32, 1, 1):
        _ = pto.alloc_buffer((32,), pto.f32)


@pto.simt
def rmsnorm_alloc_buffer_frag_helper(
    w_ub: pto.ptr(pto.f32, pto.MemorySpace.UB),
    x_ub: pto.ptr(pto.f32, pto.MemorySpace.UB),
):
    _ = pto.get_tid_x()
    _ = w_ub
    _ = x_ub
    _ = pto.alloc_buffer((32,), pto.f32)
    _ = pto.alloc_buffer((1,), pto.f32)


@pto.jit(target="a5", mode="explicit")
def rmsnorm_alloc_buffer_layout_probe(
    X: pto.ptr(pto.f32, "gm"),
    W: pto.ptr(pto.f32, "gm"),
    Y: pto.ptr(pto.f32, "gm"),
    RSTD: pto.ptr(pto.f32, "gm"),
):
    ub_base = pto.castptr(pto.const(0, dtype=pto.ui64), pto.ptr(pto.f32, "ub"))
    w_ub = pto.addptr(ub_base, 0)
    reduce_scratch = pto.addptr(ub_base, 4096)
    x_ub = pto.addptr(ub_base, 4224)
    y_ub = pto.addptr(ub_base, 12416)
    rstd_ub = pto.addptr(ub_base, 20608)

    pto.mte_gm_ub(W, w_ub, 0, 4096 * 4, nburst=(1, 0, 0))
    pto.mte_gm_ub(X, x_ub, 0, 4096 * 4, nburst=(1, 0, 0))
    rmsnorm_alloc_buffer_frag_helper(w_ub, x_ub)
    pto.mte_ub_gm(y_ub, Y, 4096 * 4, nburst=(1, 0, 0))
    pto.mte_ub_gm(rstd_ub, RSTD, 4, nburst=(1, 0, 0))
    _ = reduce_scratch


@pto.simt
def simt_alloc_buffer_arg_probe(buf: pto.ptr(pto.f32, "ub")):
    _ = buf


@pto.jit(target="a5", mode="explicit")
def simt_helper_reject_alloc_buffer_probe():
    scratch = pto.alloc_buffer((32,), pto.f32)
    pto.simt_launch(simt_alloc_buffer_arg_probe, scratch, dims=(32, 1, 1))


@pto.jit(target="a5", mode="explicit")
def simt_helper_reject_alloc_buffer_offset_probe():
    scratch = pto.alloc_buffer((32,), pto.f32)
    pto.simt_launch(simt_alloc_buffer_arg_probe, scratch + 4, dims=(32, 1, 1))


@pto.jit(target="a5")
def simt_explicit_launch_probe(*, TRACE_TOKEN: pto.const_expr = 0):
    pto.simt_launch(simt_query_probe, dims=(32, 2, 1))


@pto.jit(target="a5")
def simt_launch_index_sugar_probe(*, TRACE_TOKEN: pto.const_expr = 0):
    simt_query_probe[32, 2, 1]()


@pto.jit(target="a5")
def simt_grouped_query_launch_probe(*, TRACE_TOKEN: pto.const_expr = 0):
    simt_grouped_query_probe[32, 1, 1]()


@pto.jit(target="a5")
def simt_resource_attr_launch_probe(*, TRACE_TOKEN: pto.const_expr = 0):
    pto.simt_launch(simt_resource_attr_probe, dims=(128, 1, 1))


@pto.jit(target="a5")
def simt_full_surface_probe(
    gm: pto.ptr(pto.i32, "gm"),
    gm_f8e4x4: pto.ptr(pto.f8e4m3x4, "gm"),
    gm_f8e4x8: pto.ptr(pto.f8e4m3x8, "gm"),
    gm_f8e5x4: pto.ptr(pto.f8e5m2x4, "gm"),
    gm_f8e5x8: pto.ptr(pto.f8e5m2x8, "gm"),
    *,
    TRACE_TOKEN: pto.const_expr = 0,
):
    pto.simt_launch(simt_collective_math_probe, dims=(32, 1, 1))
    pto.simt_launch(simt_memory_atomic_probe, gm, dims=(32, 1, 1))
    pto.simt_launch(
        simt_fp8_ext_ldg_stg_probe,
        gm_f8e4x4,
        gm_f8e4x8,
        gm_f8e5x4,
        gm_f8e5x8,
        dims=(32, 1, 1),
    )
    pto.simt_launch(simt_keep_stage, dims=(32, 1, 1))
    pto.simt_launch(simt_resume_stage, gm, dims=(32, 1, 1))


@pto.jit(target="a5")
def simt_specialized_arg_type_probe(
    gm_i32: pto.ptr(pto.i32, "gm"),
    gm_f32: pto.ptr(pto.f32, "gm"),
    *,
    TRACE_TOKEN: pto.const_expr = 0,
):
    pto.simt_launch(simt_specialized_i32_ptr_probe, gm_i32, dims=(32, 1, 1))
    pto.simt_launch(simt_specialized_f32_ptr_probe, gm_f32, dims=(32, 1, 1))


@pto.jit(target="a5")
def simt_specialized_static_kwarg_probe(*, TRACE_TOKEN: pto.const_expr = 0):
    pto.simt_launch(simt_specialized_flag_probe, dims=(32, 1, 1), FLAG=False)
    pto.simt_launch(simt_specialized_flag_probe, dims=(32, 1, 1), FLAG=True)


@pto.jit(target="a5")
def simt_invalid_redux_signedness_launch(*, TRACE_TOKEN: pto.const_expr = 0):
    pto.simt_launch(simt_invalid_redux_signedness_probe, dims=(32, 1, 1))


@pto.jit(target="a5")
def simt_invalid_convert_signedness_launch(*, TRACE_TOKEN: pto.const_expr = 0):
    pto.simt_launch(simt_invalid_convert_signedness_probe, dims=(32, 1, 1))


@pto.jit(target="a5")
def simt_invalid_atomic_signedness_launch(
    gm: pto.ptr(pto.f32, "gm"),
    *,
    TRACE_TOKEN: pto.const_expr = 0,
):
    pto.simt_launch(simt_invalid_atomic_signedness_probe, gm, dims=(32, 1, 1))


@pto.jit(target="a5")
def ast_subkernel_runtime_for_probe(rows: pto.i32):
    inp = pto.alloc_tile(shape=[1, 64], dtype=pto.f32)
    out = pto.alloc_tile(shape=[1, 64], dtype=pto.f32)
    ast_subkernel_runtime_for_helper(inp, out, rows)


@pto.jit(target="a5")
def carry_loop_lowering_probe(*, BLOCK: pto.const_expr = 128):
    m_prev = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)
    l_prev = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)
    o_prev = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)
    m_next = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)
    l_next = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)
    o_next = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)

    m_prev.fill(0.0)
    l_prev.fill(0.0)
    o_prev.fill(0.0)

    m = m_prev
    l = l_prev
    o = o_prev
    for _ in range(0, 4, 1):
        m.fill(1.0)
        l.fill(2.0)
        o.fill(3.0)
        m = m_next
        l = l_next
        o = o_next

    o.fill(4.0)


@pto.jit(target="a5")
def branch_handle_then_only_probe():
    cond = pto.const(1, dtype=pto.i1)
    with pto.if_(cond) as br:
        with br.then_:
            pto.pipe_barrier(pto.Pipe.ALL)


@pto.jit(target="a5")
def branch_handle_side_effect_probe():
    lhs = pto.const(4, dtype=pto.i32)
    rhs = pto.const(2, dtype=pto.i32)
    with pto.if_(lhs > rhs) as br:
        with br.then_:
            pto.pipe_barrier(pto.Pipe.ALL)
        with br.else_:
            pto.mem_bar(pto.BarrierType.VST_VLD)


@pto.jit(target="a5")
def branch_handle_merge_probe():
    lhs = pto.const(4, dtype=pto.i32)
    rhs = pto.const(2, dtype=pto.i32)
    with pto.if_(lhs > rhs) as br:
        with br.then_:
            br.assign(total=lhs + rhs, diff=lhs - rhs)
        with br.else_:
            br.assign(total=rhs + lhs, diff=rhs - lhs)
    total = br.total
    diff = br.diff
    merged = total + diff
    _ = merged


@pto.jit(target="a5")
def ast_if_side_effect_probe():
    cond = pto.const(1, dtype=pto.i1)
    if cond:
        pto.pipe_barrier(pto.Pipe.ALL)


@pto.jit(target="a5")
def ast_if_merge_probe():
    lhs = pto.const(4, dtype=pto.i32)
    rhs = pto.const(2, dtype=pto.i32)
    if lhs > rhs:
        total = lhs + rhs
        diff = lhs - rhs
    else:
        total = rhs + lhs
        diff = rhs - lhs
    merged = total + diff
    _ = merged


@pto.jit(target="a5")
def ast_if_old_value_merge_probe():
    lhs = pto.const(4, dtype=pto.i32)
    rhs = pto.const(2, dtype=pto.i32)
    total = rhs
    if lhs > rhs:
        total = lhs + rhs
    merged = total + rhs
    _ = merged


@pto.jit(target="a5")
def ast_if_static_subscript_slot_merge_probe():
    done = [None]
    done[0] = pto.const(0, dtype=pto.i32)
    condition = pto.const(1, dtype=pto.i1)

    if condition:
        done[0] = pto.const(1, dtype=pto.i32)

    if done[0]:
        pto.pipe_barrier(pto.Pipe.ALL)


_AST_BRANCH_SLOT_INDEX = 1


@pto.jit(target="a5")
def ast_if_static_subscript_expression_merge_probe(*, SLOT: pto.const_expr = 1):
    values = [
        pto.const(0, dtype=pto.i1),
        pto.const(0, dtype=pto.i1),
    ]
    condition = pto.const(1, dtype=pto.i1)

    if condition:
        values[_AST_BRANCH_SLOT_INDEX + (SLOT - SLOT)] = pto.const(1, dtype=pto.i1)

    if values[SLOT]:
        pto.pipe_barrier(pto.Pipe.ALL)


@pto.jit(target="a5")
def ast_if_static_subscript_negative_index_merge_probe():
    values = [pto.const(0, dtype=pto.i1)]
    condition = pto.const(1, dtype=pto.i1)

    if condition:
        values[-1] = pto.const(1, dtype=pto.i1)

    if values[-1]:
        pto.pipe_barrier(pto.Pipe.ALL)


@pto.jit(target="a5")
def ast_if_branch_local_temp_liveness_probe():
    c0 = pto.const(0, dtype=pto.i1)
    c1 = pto.const(1, dtype=pto.i1)
    one = pto.const(1, dtype=pto.i32)
    zero = pto.const(0, dtype=pto.i32)

    if c0:
        tmp = one
        _ = tmp

    if c1:
        tmp = one
        out = tmp + one
    else:
        out = zero

    _ = out


@pto.jit(target="a5")
def ast_nested_with_if_merge_probe():
    lhs = pto.const(4, dtype=pto.i32)
    rhs = pto.const(2, dtype=pto.i32)
    meta_tile = pto.alloc_tile(shape=[1, 8], dtype=pto.i32, valid_shape=[1, 1])
    with pto.simt():
        if lhs > rhs:
            value = lhs + rhs
        else:
            value = rhs + lhs
        merged = value + rhs
        scalar.store(merged, meta_tile.as_ptr() + 0)


@pto.jit(target="a5")
def ast_runtime_for_probe(rows: pto.i32):
    for row in range(0, rows, 1):
        _ = row
        pto.pipe_barrier(pto.Pipe.ALL)


@pto.jit(target="a5", mode="explicit")
def ast_nested_runtime_for_induction_scope_probe(rows: pto.i32):
    for outer in range(rows):
        _ = outer
        with pto.vecscope():
            for lane in range(4):
                _ = lane
                pto.mem_bar(pto.BarrierType.VST_VLD)


@pto.jit(target="a5", mode="explicit")
def ast_nested_runtime_for_local_temp_scope_probe(rows: pto.i32):
    for outer in range(rows):
        _ = outer
        with pto.vecscope():
            for row in range(4):
                for col in range(2):
                    ob = row + col
                    _ = ob
        pto.pipe_barrier(pto.Pipe.ALL)


@pto.jit(target="a5")
def ast_runtime_for_carry_probe(rows: pto.i32):
    one = pto.const(1, dtype=pto.i32)
    acc = pto.const(0, dtype=pto.i32)
    for _ in range(rows):
        acc = acc + one
    _ = acc


@pto.jit(target="a5")
def ast_runtime_for_augassign_carry_probe(rows: pto.i32):
    one = pto.const(1, dtype=pto.i32)
    acc = pto.const(0, dtype=pto.i32)
    for _ in range(rows):
        acc += one
    _ = acc


@pto.jit(target="a5")
def ast_runtime_for_branch_local_temp_probe(rows: pto.i32):
    cond = pto.const(1, dtype=pto.i1)
    one = pto.const(1, dtype=pto.i32)
    zero = pto.const(0, dtype=pto.i32)

    for _ in range(rows):
        if cond:
            tmp = one
            out = tmp + one
        else:
            tmp = zero
            out = tmp + zero
        _ = out


@pto.jit(target="a5")
def ast_runtime_ifexp_assign_probe(rows: pto.i32):
    limit = pto.const(64, dtype=pto.index)
    zero = pto.const(0, dtype=pto.index)

    for row in range(rows):
        cnt = row if row < limit else limit
        out = cnt + zero
        _ = out


@pto.jit(target="a5")
def ast_runtime_ifexp_python_literal_assign_probe(rows: pto.i32):
    limit = pto.const(64, dtype=pto.index)
    zero = pto.const(0, dtype=pto.index)

    for row in range(rows):
        cnt = row if row < limit else 64
        out = cnt + zero
        _ = out


@pto.jit(target="a5")
def ast_runtime_for_sibling_iv_reuse_probe(rows: pto.i32, cols: pto.i32):
    stride = pto.const(64, dtype=pto.index)
    one = pto.const(1, dtype=pto.index)
    zero = pto.const(0, dtype=pto.index)

    for t in range(rows):
        for c in range(cols):
            col = c * stride
            sink = col + one
            _ = sink

        for stage in pto.static_range(2):
            acc = zero
            for c in range(cols):
                col = c * stride
                acc = acc + col
            _ = acc
            _ = stage
        _ = t


@pto.jit(target="a5")
def ast_runtime_for_static_range_name_reuse_probe(cols: pto.i32):
    zero = pto.const(0, dtype=pto.index)

    for phase_a_chunk in range(cols):
        d_heads = [phase_a_chunk + h for h in pto.static_range(4)]
        acc = zero
        for h in pto.static_range(4):
            acc = acc + d_heads[h]
        for mi in pto.static_range(2):
            inner = zero
            for h in pto.static_range(4):
                inner = inner + d_heads[h]
            acc = acc + inner
            _ = mi
        _ = acc


@pto.jit(target="a5")
def ast_runtime_for_static_slot_carry_probe(cols: pto.i32):
    zero = pto.const(0, dtype=pto.index)
    accs = [zero for _ in pto.static_range(4)]

    for c in range(cols):
        for h in pto.static_range(4):
            accs[h] = accs[h] + c

    total = zero
    for h in pto.static_range(4):
        total = total + accs[h]
    _ = total


@pto.jit(target="a5", mode="explicit")
def ast_runtime_for_static_slot_nested_vecscope_probe(cols: pto.i32):
    zero = pto.const(0, dtype=pto.index)

    for _ in range(cols):
        with pto.vecscope():
            values = [zero, zero, zero, zero]
            for c in range(64):
                values[0] = c
                values[1] = c
                values[2] = c
                values[3] = c
                total = values[0] + values[1] + values[2] + values[3]
                _ = total


_STATIC_SLOT_GLOBAL_INDEX = 2


@pto.jit(target="a5")
def ast_runtime_for_static_slot_global_index_probe(cols: pto.i32):
    zero = pto.const(0, dtype=pto.index)
    accs = [zero for _ in pto.static_range(4)]

    for c in range(cols):
        accs[_STATIC_SLOT_GLOBAL_INDEX] = accs[_STATIC_SLOT_GLOBAL_INDEX] + c

    total = zero
    for h in pto.static_range(4):
        total = total + accs[h]
    _ = total


@pto.jit(target="a5")
def ast_runtime_for_static_slot_binop_index_probe(cols: pto.i32):
    zero = pto.const(0, dtype=pto.index)
    accs = [zero for _ in pto.static_range(5)]

    for c in range(cols):
        for h in pto.static_range(4):
            accs[h + 1] = accs[h + 1] + c

    total = zero
    for h in pto.static_range(5):
        total = total + accs[h]
    _ = total


@pto.jit(target="a5")
def ast_runtime_for_static_slot_constexpr_index_probe(cols: pto.i32, *, SLOT: pto.const_expr = 2):
    zero = pto.const(0, dtype=pto.index)
    accs = [zero for _ in pto.static_range(4)]

    for c in range(cols):
        accs[SLOT] = accs[SLOT] + c

    total = zero
    for h in pto.static_range(4):
        total = total + accs[h]
    _ = total


@pto.jit(target="a5")
def ast_runtime_for_dynamic_slot_store_error_probe(cols: pto.i32):
    zero = pto.const(0, dtype=pto.index)
    accs = [zero for _ in pto.static_range(4)]

    for idx in range(cols):
        accs[idx] = zero


class _StaticSlotHolder:
    pass


@pto.jit(target="a5")
def ast_runtime_for_complex_slot_store_error_probe(cols: pto.i32):
    zero = pto.const(0, dtype=pto.index)
    holder = _StaticSlotHolder()
    holder.accs = [zero for _ in pto.static_range(4)]

    for _ in range(cols):
        for h in pto.static_range(4):
            holder.accs[h] = zero


@pto.jit(target="a5", ast_rewrite=False)
def ast_rewrite_disabled_nested_helper_python_control_probe():
    def helper(enabled):
        if enabled:
            for _ in range(2):
                pto.pipe_barrier(pto.Pipe.ALL)

    helper(True)


@pto.jit(target="a5")
def ast_nested_helper_ast_rewrite_probe(rows: pto.i32):
    cond = pto.const(1, dtype=pto.i1)

    def helper(limit, enabled):
        one = pto.const(1, dtype=pto.i32)
        acc = pto.const(0, dtype=pto.i32)
        for _ in range(limit):
            acc += one
        if enabled:
            acc = acc + one
        return acc

    value = helper(rows, cond)
    _ = value


@pto.func(returns=pto.i32)
def func_runtime_for_return_helper(limit: pto.i32, initial: pto.i32):
    one = pto.const(1, dtype=pto.i32)
    total = initial
    for _ in range(limit):
        total = total + one
    return total


@pto.func(returns=pto.i32)
def func_runtime_if_return_helper(lhs: pto.i32, rhs: pto.i32):
    if lhs > rhs:
        total = lhs + rhs
    else:
        total = rhs + lhs
    return total


@pto.func(returns=pto.i32)
def func_runtime_if_early_return_helper(lhs: pto.i32, rhs: pto.i32):
    if lhs > rhs:
        return lhs
    return rhs


@pto.func(returns=pto.i32)
def func_runtime_for_early_return_helper(limit: pto.i32, initial: pto.i32):
    for _ in range(limit):
        return initial
    return initial


@pto.func(returns=None)
def func_runtime_if_yield_helper(lhs: pto.i32, rhs: pto.i32):
    if lhs > rhs:
        yield lhs


@pto.func(returns=(pto.i32, pto.i32))
def func_multi_return_helper(value: pto.i32):
    one = pto.const(1, dtype=pto.i32)
    return value, value + one


@pto.func(returns=None)
def func_void_helper():
    pto.pipe_barrier(pto.Pipe.ALL)


@pto.func(returns=None)
def func_i32_argument_helper(value: pto.i32):
    _ = value


@pto.func(returns=pto.i32)
def func_partition_metadata_helper(part: pto.PartitionTensorView, cols: pto.i32):
    return part.sizes[0] + cols


@pto.func(returns=pto.i32)
def func_constexpr_static_helper(value: pto.i32, *, BLOCK: pto.const_expr = 2):
    total = value
    one = pto.const(1, dtype=pto.i32)
    for _ in pto.static_range(BLOCK):
        total = total + one
    return total


def _make_future_annotations_ptodsl_func_helpers():
    namespace = {"pto": pto}
    exec(
        """
from __future__ import annotations

@pto.func
def func_future_i32_return_helper(x: pto.i32) -> pto.i32:
    return x + pto.const(1, dtype=pto.i32)

@pto.func
def func_future_i64_literal_helper(x: pto.i64) -> pto.i64:
    return x + pto.const(1, dtype=pto.i64)

@pto.func
def func_future_void_helper(x: pto.i32) -> None:
    _ = x
    pto.pipe_barrier(pto.Pipe.ALL)
""",
        namespace,
    )
    return (
        namespace["func_future_i32_return_helper"],
        namespace["func_future_i64_literal_helper"],
        namespace["func_future_void_helper"],
    )


(
    func_future_i32_return_helper,
    func_future_i64_literal_helper,
    func_future_void_helper,
) = _make_future_annotations_ptodsl_func_helpers()


@pto.jit(target="a5")
def ptodsl_func_call_probe(rows: pto.i32):
    init = pto.const(0, dtype=pto.i32)
    total = func_runtime_for_return_helper(rows, init)
    merged = func_runtime_if_return_helper(total, init)
    first, second = func_multi_return_helper(merged)
    _ = first + second
    func_void_helper()
    func_void_helper()


@pto.jit(target="a5")
def ptodsl_func_partition_metadata_probe(
    A_ptr: pto.ptr(pto.f32, "gm"),
    cols: pto.i32,
):
    a_view = pto.make_tensor_view(A_ptr, shape=[1, 16], strides=[16, 1])
    part = pto.partition_view(a_view, offsets=[0, 0], sizes=[1, 16])
    _ = func_partition_metadata_helper(part, cols)


@pto.jit(target="a5")
def ptodsl_func_future_annotations_probe(rows: pto.i32):
    i32_value = func_future_i32_return_helper(rows)
    i64_value = func_future_i64_literal_helper(1)
    func_future_void_helper(i32_value)
    _ = i64_value


@pto.jit(target="a5")
def ptodsl_func_constexpr_probe(rows: pto.i32):
    first = func_constexpr_static_helper(rows, BLOCK=2)
    second = func_constexpr_static_helper(rows, BLOCK=4)
    _ = first + second


@pto.jit(target="a5")
def ptodsl_func_constexpr_runtime_value_probe(rows: pto.i32):
    _ = func_constexpr_static_helper(rows, BLOCK=rows)


@pto.jit(target="a5")
def ptodsl_func_traced_argument_coercion_probe(value: pto.i64):
    func_i32_argument_helper(value)


@pto.jit(target="a5")
def ptodsl_func_traced_argument_type_error_probe(value: pto.f32):
    func_i32_argument_helper(value)


@pto.jit(target="a5")
def ptodsl_func_runtime_closure_capture_probe(value: pto.i32):
    @pto.func(returns=None)
    def helper():
        _ = value

    helper()


@pto.jit(target="a5")
def ptodsl_func_if_early_return_probe(rows: pto.i32):
    init = pto.const(0, dtype=pto.i32)
    _ = func_runtime_if_early_return_helper(rows, init)


@pto.jit(target="a5")
def ptodsl_func_for_early_return_probe(rows: pto.i32):
    init = pto.const(0, dtype=pto.i32)
    _ = func_runtime_for_early_return_helper(rows, init)


@pto.jit(target="a5")
def ptodsl_func_if_yield_probe(rows: pto.i32):
    init = pto.const(0, dtype=pto.i32)
    func_runtime_if_yield_helper(rows, init)


@pto.jit(target="a5")
def ast_nested_helper_freevar_if_merge_probe():
    lhs = pto.const(4, dtype=pto.i32)
    rhs = pto.const(2, dtype=pto.i32)

    if lhs > rhs:
        x = lhs + rhs
    else:
        x = rhs + lhs

    def helper():
        return x + rhs

    y = helper()
    _ = y


@pto.jit(target="a5")
def ast_nested_helper_name_store_liveness_probe():
    lhs = pto.const(4, dtype=pto.i32)
    rhs = pto.const(2, dtype=pto.i32)

    if lhs > rhs:
        helper = lhs
    else:
        helper = rhs

    def helper():
        return lhs + rhs

    y = helper()
    _ = y


@pto.jit(target="a5", name="ast_control_flow_equiv_explicit")
def ast_control_flow_equiv_explicit_probe(rows: pto.i32):
    one = pto.const(1, dtype=pto.i32)
    acc = pto.const(0, dtype=pto.i32)

    def helper(limit, initial):
        total = initial
        loop = pto.for_(0, limit, step=1).carry(total=total)
        with loop:
            i = loop.iv
            total = loop.total
            cond = i > pto.const(0)
            with pto.if_(cond) as br:
                with br.then_:
                    br.assign(total=total + one)
                with br.else_:
                    br.assign(total=total)
            total = br.total
            loop.update(total=total)
        return loop.final("total")

    acc = helper(rows, acc)
    _ = acc


@pto.jit(target="a5", name="ast_control_flow_equiv_native")
def ast_control_flow_equiv_native_probe(rows: pto.i32):
    one = pto.const(1, dtype=pto.i32)
    acc = pto.const(0, dtype=pto.i32)

    def helper(limit, initial):
        total = initial
        for i in range(limit):
            if i > pto.const(0):
                total = total + one
        return total

    acc = helper(rows, acc)
    _ = acc


def make_ast_closure_kernel(limit: int):
    @pto.jit(target="a5")
    def ast_closure_kernel():
        for _ in pto.static_range(limit):
            pto.pipe_barrier(pto.Pipe.ALL)

    return ast_closure_kernel


ast_closure_kernel_probe = make_ast_closure_kernel(3)


def make_ast_rebound_closure_kernel():
    limit = 2

    @pto.jit(target="a5")
    def ast_rebound_closure_kernel():
        for _ in pto.static_range(limit):
            pto.pipe_barrier(pto.Pipe.ALL)

    limit = 4
    return ast_rebound_closure_kernel


ast_rebound_closure_kernel_probe = make_ast_rebound_closure_kernel()


def make_ast_mutable_closure_cache_kernel():
    limit = 2

    @pto.jit(target="a5")
    def ast_mutable_closure_cache_kernel():
        for _ in pto.static_range(limit):
            pto.pipe_barrier(pto.Pipe.ALL)

    def set_limit(value: int):
        nonlocal limit
        limit = value

    return ast_mutable_closure_cache_kernel, set_limit


ast_mutable_closure_cache_kernel_probe, set_ast_mutable_closure_cache_limit = (
    make_ast_mutable_closure_cache_kernel()
)


def make_ast_signature_closure_default_kernel(limit: int):
    @pto.jit(target="a5")
    def ast_signature_closure_default_kernel(*, BLOCK: pto.const_expr = limit):
        for _ in pto.static_range(BLOCK):
            pto.pipe_barrier(pto.Pipe.ALL)

    return ast_signature_closure_default_kernel


ast_signature_closure_default_kernel_probe = make_ast_signature_closure_default_kernel(2)


def make_ast_rebound_subkernel_probe():
    limit = 2

    @pto.tileop
    def helper():
        for _ in pto.static_range(limit):
            tileop_noop_simt_probe[1, 1, 1]()

    limit = 4

    @pto.jit(target="a5")
    def ast_rebound_subkernel_probe(*, TRACE_TOKEN: pto.const_expr = 0):
        helper()

    return ast_rebound_subkernel_probe


ast_rebound_subkernel_probe = make_ast_rebound_subkernel_probe()


def make_sourceless_ast_rewrite_kernel():
    namespace = {"pto": pto}
    exec(
        """
@pto.jit(target="a5")
def sourceless_ast_rewrite_kernel():
    if True:
        pto.pipe_barrier(pto.Pipe.ALL)
""",
        namespace,
    )
    return namespace["sourceless_ast_rewrite_kernel"]


sourceless_ast_rewrite_kernel_probe = make_sourceless_ast_rewrite_kernel()


def make_sourceless_subkernel_entry():
    namespace = {"pto": pto, "tileop_noop_simt_probe": tileop_noop_simt_probe}
    exec(
        """
@pto.tileop
def sourceless_subkernel_helper():
    if True:
        tileop_noop_simt_probe[1, 1, 1]()
""",
        namespace,
    )
    helper = namespace["sourceless_subkernel_helper"]

    @pto.jit(target="a5")
    def sourceless_subkernel_entry_probe(*, TRACE_TOKEN: pto.const_expr = 0):
        helper()

    return sourceless_subkernel_entry_probe


sourceless_subkernel_entry_probe = make_sourceless_subkernel_entry()


def make_sourceless_ptodsl_func_probe():
    namespace = {"pto": pto}
    exec(
        """
@pto.func(returns=None)
def sourceless_ptodsl_func_helper():
    if True:
        pto.pipe_barrier(pto.Pipe.ALL)
""",
        namespace,
    )
    helper = namespace["sourceless_ptodsl_func_helper"]

    @pto.jit(target="a5")
    def sourceless_ptodsl_func_probe(*, TRACE_TOKEN: pto.const_expr = 0):
        helper()

    return sourceless_ptodsl_func_probe


sourceless_ptodsl_func_probe = make_sourceless_ptodsl_func_probe()


def make_entry_closure_kernel_module_probe():
    @pto.jit(target="a5", entry=False)
    def closure_helper():
        pto.pipe_barrier(pto.Pipe.ALL)

    @pto.jit(target="a5")
    def closure_entry(*, TRACE_TOKEN: pto.const_expr = 0):
        closure_helper()

    return closure_entry


entry_closure_kernel_module_probe = make_entry_closure_kernel_module_probe()


@pto.jit(target="a5")
def ast_static_control_flow_probe(*, ENABLE: pto.const_expr = True):
    if pto.const_expr(ENABLE):
        for _ in pto.static_range(2):
            pto.pipe_barrier(pto.Pipe.ALL)


@pto.jit(target="a5")
def ast_python_bool_guard_probe(
    *,
    BLOCK: pto.const_expr = 128,
    ENABLE: pto.const_expr = True,
):
    if BLOCK == 128:
        pto.pipe_barrier(pto.Pipe.ALL)
    if ENABLE:
        pto.pipe_barrier(pto.Pipe.ALL)


@pto.jit(target="a5")
def ast_static_range_loop_target_live_after_probe():
    for stage in pto.static_range(3):
        pto.pipe_barrier(pto.Pipe.ALL)
    _ = stage


@pto.jit(target="a5")
def ast_static_range_break_continue_probe():
    for stage in pto.static_range(4):
        if pto.const_expr(stage == 2):
            break
        pto.pipe_barrier(pto.Pipe.ALL)

    for stage in pto.static_range(4):
        if pto.const_expr(stage % 2 == 0):
            continue
        pto.pipe_barrier(pto.Pipe.ALL)


@pto.jit(target="a5")
def runtime_scalar_operator_probe(
    O_ptr: pto.ptr(pto.f32, "gm"),
    rows: pto.i32,
    cols: pto.i32,
    row_stride: pto.i32,
    *,
    BLOCK: pto.const_expr = 8,
):
    block_idx = pto.get_block_idx()
    o_view = pto.make_tensor_view(O_ptr, shape=[rows, cols], strides=[row_stride, 1])
    o_part = pto.partition_view(o_view, offsets=[0, 0], sizes=[rows, cols])
    o_ptr = o_part.as_ptr()

    batch_idx = block_idx // rows
    head_idx = block_idx % rows
    chunks = (cols + BLOCK - 1) // BLOCK
    tail = cols % BLOCK

    x = pto.const(2.0, dtype=pto.f32)
    y = (x + 1.0) * 2.0
    z = 4.0 - y
    w = 1.0 / z
    m = scalar.max(w, x)
    n = scalar.min(m, x)
    e = scalar.exp(m)
    lg = scalar.log(e)
    rt = scalar.sqrt(e)
    mag = scalar.abs(z)
    gt_zero = m > x
    eq_self = x == x
    in_range = (m >= x) & (m <= e)
    scalar.store(e, o_ptr + 0)

    _ = batch_idx
    _ = head_idx
    _ = chunks
    _ = tail
    _ = w
    _ = m
    _ = n
    _ = e
    _ = lg
    _ = rt
    _ = mag
    _ = gt_zero
    _ = eq_self
    _ = in_range


@pto.jit(target="a5")
def host_runtime_scalar_entry_probe(
    A_ptr: pto.ptr(pto.f32, "gm"),
    O_ptr: pto.ptr(pto.f32, "gm"),
    rows: pto.i32,
    cols: pto.i32,
    limit: pto.i32,
    alpha: pto.f32,
):
    a_view = pto.make_tensor_view(A_ptr, shape=[rows, cols], strides=[cols, 1])
    o_view = pto.make_tensor_view(O_ptr, shape=[rows, cols], strides=[cols, 1])
    a_part = pto.partition_view(a_view, offsets=[0, 0], sizes=[rows, cols])
    o_part = pto.partition_view(o_view, offsets=[0, 0], sizes=[rows, cols])
    a_tile = pto.alloc_tile(shape=[1, 8], dtype=pto.f32, valid_shape=[1, cols])
    o_tile = pto.alloc_tile(shape=[1, 8], dtype=pto.f32, valid_shape=[1, cols])
    pto.tile.load(a_part, a_tile)
    row_limit = limit // pto.const(2, dtype=pto.i32)
    scaled = alpha + 1.0
    _ = row_limit
    _ = scaled
    pto.tile.store(o_tile, o_part)


@pto.tileop
def tile_slice_vector_probe(inp_tile: pto.Tile, out_tile: pto.Tile, row: pto.index):
    mask, _ = pto.plt_b32(pto.const(64, dtype=pto.i32))
    vec = pto.vlds(inp_tile[row, 0:])
    pto.vsts(vec, out_tile[row, 0:], mask)


@pto.jit(target="a5")
def tile_slice_surface_probe(*, BLOCK: pto.const_expr = 128):
    inp_tile = pto.alloc_tile(shape=[2, BLOCK], dtype=pto.f32)
    out_tile = pto.alloc_tile(shape=[2, BLOCK], dtype=pto.f32)
    for row in range(0, 1, 1):
        tile_slice_vector_probe(inp_tile, out_tile, row)


@pto.jit(target="a5")
def tile_slice_1d_surface_probe(*, BLOCK: pto.const_expr = 128):
    inp_tile = pto.alloc_tile(shape=[BLOCK], dtype=pto.f32)
    out_tile = pto.alloc_tile(shape=[BLOCK], dtype=pto.f32)
    start = pto.const(0, dtype=pto.i32)
    mask, _ = pto.plt_b32(pto.const(64, dtype=pto.i32))
    align = pto.vldas(inp_tile[start:])
    vec, align = pto.vldus(inp_tile[start:], align)
    pto.vsts(vec, out_tile[start:], mask)
    _ = align


@pto.jit(target="a5")
def tile_valid_shape_update_probe(
    rows: pto.i32,
    cols: pto.i32,
    *,
    BLOCK: pto.const_expr = 128,
):
    tile = pto.alloc_tile(
        shape=[1, BLOCK],
        dtype=pto.f32,
        valid_shape=[pto.const(1), cols],
    )
    tile.valid_shape = [rows, cols]


@pto.jit(target="a5")
def tile_valid_shape_update_1d_probe(
    length: pto.i32,
    *,
    BLOCK: pto.const_expr = 128,
):
    tile = pto.alloc_tile(
        shape=[BLOCK],
        dtype=pto.f32,
        valid_shape=[pto.const(BLOCK)],
    )
    tile.valid_shape = [length]


@pto.jit(target="a5", mode="explicit")
def make_mask_index_roundtrip_probe(
    cols: pto.i32,
):
    remained = cols
    for _ in range(0, cols, 64):
        mask, remained = pto.make_mask(pto.f32, remained)
        _ = mask


@pto.jit(target="a5", mode="explicit")
def carry_static_pyint_init_probe():
    col_loop = pto.for_(0, 64, step=64).carry(remained=64)
    with col_loop:
        remained = col_loop.remained
        mask, remained_after_pack = pto.make_mask(pto.f32, remained)
        _ = mask
        col_loop.update(remained=remained_after_pack)


@pto.jit(target="a5")
def fixed_integer_index_coercion_probe():
    count = pto.const(4)
    mask, remained = pto.make_mask(pto.f32, count)
    _ = mask
    _ = remained


@pto.jit(target="a5")
def integer_loop_bound_probe(*, BLOCK: pto.const_expr = 8):
    row_start = pto.const(0, dtype=pto.i32)
    row_stop = pto.const(BLOCK, dtype=pto.i32)
    valid_dim = pto.const(BLOCK // 2, dtype=pto.i32)
    for row in range(row_start, row_stop, 1):
        for col in range(0, valid_dim, 1):
            _ = row
            _ = col


@pto.jit(target="a5")
def scalar_pointer_offset_probe():
    meta_tile = pto.alloc_tile(shape=[1, 8], dtype=pto.i32, valid_shape=[1, 3])
    meta_ptr = meta_tile.as_ptr()
    scalar.store(0, meta_ptr, 0)
    scalar.store(1, meta_ptr, 1)
    scalar.store(2, meta_ptr + 2)
    row_start = scalar.load(meta_ptr, 0)
    row_stop = scalar.load(meta_ptr, 1)
    valid_cols = scalar.load(meta_ptr + 2)
    _ = row_start
    _ = row_stop
    _ = valid_cols


@pto.jit(target="a5")
def scalar_contiguous_vector_probe():
    data_tile = pto.alloc_tile(shape=[1, 16], dtype=pto.f32, valid_shape=[1, 16])
    data_ptr = data_tile.as_ptr()
    x4 = scalar.load(data_ptr, 0, contiguous=4)
    scale4 = pto.Vec(pto.f32, size=4, init=1.0)
    y4 = x4 * scale4
    scalar.store(y4, data_ptr, 4)


@pto.simt
def scalar_contiguous_vector_arith_simt_body(data_ptr):
    tid = pto.get_tid_x()
    base = tid * 4
    x4 = scalar.load(data_ptr, base, contiguous=4)
    y4 = scalar.load(data_ptr, 32 + base, contiguous=4)
    sum4 = x4 + y4
    diff4 = x4 - y4
    prod4 = x4 * y4
    scalar.store(sum4, data_ptr, 64 + base)
    scalar.store(diff4, data_ptr, 96 + base)
    scalar.store(prod4, data_ptr, 128 + base)


@pto.jit(target="a5", mode="explicit")
def scalar_contiguous_vector_arith_probe():
    scalar_contiguous_vector_arith_simt_body[8, 1, 1](
        pto.castptr(pto.const(0, dtype=pto.ui64), pto.ptr(pto.f32, "ub"))
    )
    pto.pipe_barrier(pto.Pipe.ALL)


@pto.simt
def scalar_contiguous_local_alloc_buffer_helper():
    data = pto.alloc_buffer((16,), pto.f32)
    x4 = scalar.load(data, 0, contiguous=4)
    scale4 = pto.Vec(pto.f32, 4, init=1.0)
    y4 = x4 * scale4
    scalar.store(y4, data, 4)


@pto.jit(target="a5", mode="explicit")
def scalar_contiguous_local_alloc_buffer_probe():
    scalar_contiguous_local_alloc_buffer_helper()


@pto.jit(target="a5")
def scalar_contiguous_width_mismatch_probe():
    data_tile = pto.alloc_tile(shape=[1, 16], dtype=pto.f32, valid_shape=[1, 16])
    data_ptr = data_tile.as_ptr()
    x4 = scalar.load(data_ptr, 0, contiguous=4)
    scalar.store(x4, data_ptr, 4, contiguous=2)


@pto.jit(target="a5")
def scalar_contiguous_scalar_store_probe():
    data_tile = pto.alloc_tile(shape=[1, 16], dtype=pto.f32, valid_shape=[1, 16])
    data_ptr = data_tile.as_ptr()
    scalar.store(1.0, data_ptr, 0, contiguous=4)


@pto.simt
def per_element_vec_f32_simt_body(data_ptr):
    base = pto.get_tid_x() * 2
    v0 = scalar.load(data_ptr, base)
    v1 = scalar.load(data_ptr, base + 1)
    pair = pto.Vec(pto.f32, 2, init=(v0, v1))
    scalar.store(pair, data_ptr, 16 + base)


@pto.jit(target="a5", mode="explicit")
def per_element_vec_f32_probe():
    per_element_vec_f32_simt_body[1, 1, 1](
        pto.castptr(pto.const(0, dtype=pto.ui64), pto.ptr(pto.f32, "ub"))
    )
    pto.pipe_barrier(pto.Pipe.ALL)


@pto.simt
def per_element_vec_i32_simt_body(data_ptr):
    base = pto.get_tid_x() * 2
    v0 = scalar.load(data_ptr, base)
    v1 = scalar.load(data_ptr, base + 1)
    pair = pto.Vec(pto.i32, 2, init=(v0, v1))
    scalar.store(pair, data_ptr, 16 + base)


@pto.jit(target="a5", mode="explicit")
def per_element_vec_i32_probe():
    per_element_vec_i32_simt_body[1, 1, 1](
        pto.castptr(pto.const(0, dtype=pto.ui64), pto.ptr(pto.i32, "ub"))
    )
    pto.pipe_barrier(pto.Pipe.ALL)


@pto.simt
def per_element_vec_ui32_simt_body(data_ptr):
    base = pto.get_tid_x() * 2
    v0 = scalar.load(data_ptr, base)
    v1 = scalar.load(data_ptr, base + 1)
    pair = pto.Vec(pto.ui32, 2, init=(v0, v1))
    scalar.store(pair, data_ptr, 16 + base)


@pto.jit(target="a5", mode="explicit")
def per_element_vec_ui32_probe():
    per_element_vec_ui32_simt_body[1, 1, 1](
        pto.castptr(pto.const(0, dtype=pto.ui64), pto.ptr(pto.ui32, "ub"))
    )
    pto.pipe_barrier(pto.Pipe.ALL)


@pto.jit(target="a5")
def vec_init_size_mismatch_probe():
    data_tile = pto.alloc_tile(shape=[1, 16], dtype=pto.f32, valid_shape=[1, 16])
    data_ptr = data_tile.as_ptr()
    pair = pto.Vec(pto.f32, 2, init=(1.0, 2.0, 3.0))
    scalar.store(pair, data_ptr, 0)


@pto.jit(target="a5")
def vec_init_str_probe():
    data_tile = pto.alloc_tile(shape=[1, 16], dtype=pto.f32, valid_shape=[1, 16])
    data_ptr = data_tile.as_ptr()
    pair = pto.Vec(pto.f32, 2, init="ab")
    scalar.store(pair, data_ptr, 0)


@pto.jit(target="a5")
def vec_init_literals_probe():
    signed_tile = pto.alloc_tile(shape=[1, 16], dtype=pto.si32, valid_shape=[1, 16])
    unsigned_tile = pto.alloc_tile(shape=[1, 16], dtype=pto.ui32, valid_shape=[1, 16])
    signed_ptr = signed_tile.as_ptr()
    unsigned_ptr = unsigned_tile.as_ptr()
    signed_pair = pto.Vec(pto.si32, 2, init=(-1, 2))
    unsigned_pair = pto.Vec(pto.ui32, 2, init=(-1, 2))
    scalar.store(signed_pair, signed_ptr, 0)
    scalar.store(unsigned_pair, unsigned_ptr, 0)


@pto.jit(target="a5")
def vec_init_mapping_probe():
    data_tile = pto.alloc_tile(shape=[1, 16], dtype=pto.f32, valid_shape=[1, 16])
    data_ptr = data_tile.as_ptr()
    pair = pto.Vec(pto.f32, 2, init={0: 1.0, 1: 2.0})
    scalar.store(pair, data_ptr, 0)


@pto.jit(target="a5")
def vec_broadcast_unsigned_probe():
    unsigned_tile = pto.alloc_tile(shape=[1, 16], dtype=pto.ui32, valid_shape=[1, 16])
    unsigned_ptr = unsigned_tile.as_ptr()
    broadcast = pto.Vec(pto.ui32, 2, init=7)
    scalar.store(broadcast, unsigned_ptr, 0)


@pto.jit(target="a5")
def vec_init_16b_probe():
    f16_tile = pto.alloc_tile(shape=[1, 16], dtype=pto.f16, valid_shape=[1, 16])
    i16_tile = pto.alloc_tile(shape=[1, 16], dtype=pto.i16, valid_shape=[1, 16])
    ui16_tile = pto.alloc_tile(shape=[1, 16], dtype=pto.ui16, valid_shape=[1, 16])
    f16_ptr = f16_tile.as_ptr()
    i16_ptr = i16_tile.as_ptr()
    ui16_ptr = ui16_tile.as_ptr()
    pair_f16 = pto.Vec(pto.f16, 2, init=(1.5, -2.25))
    quad_f16 = pto.Vec(pto.f16, 4, init=(1.5, -2.25, 3.0, 4.5))
    quad_i16 = pto.Vec(pto.i16, 4, init=(-1, 2, 3, -4))
    quad_ui16 = pto.Vec(pto.ui16, 4, init=(1, 2, 3, 4))
    scalar.store(pair_f16, f16_ptr, 0)
    scalar.store(quad_f16, f16_ptr, 2)
    scalar.store(quad_i16, i16_ptr, 0)
    scalar.store(quad_ui16, ui16_ptr, 0)


@pto.jit(target="a5")
def addptr_surface_probe():
    meta_tile = pto.alloc_tile(shape=[1, 8], dtype=pto.i32, valid_shape=[1, 4])
    meta_ptr = meta_tile.as_ptr()
    ptr_pyint = pto.addptr(meta_ptr, 2)
    ptr_i32 = pto.addptr(meta_ptr, pto.i32(3))
    scalar.store(11, ptr_pyint)
    scalar.store(13, ptr_i32)
    val_pyint = scalar.load(ptr_pyint)
    val_i32 = scalar.load(ptr_i32)
    _ = val_pyint
    _ = val_i32


@pto.simt
def simt_pointer_offset_helper(meta_ptr: pto.ptr(pto.i32, pto.MemorySpace.UB)):
    scalar.store(7, meta_ptr + 0)
    scalar.store(9, meta_ptr + 1)


@pto.simt
def simt_reserved_buffer_peer():
    pto.reserve_buffer("simt_c2v_fifo", size=8192, location="vec")


@pto.simt
def simt_reserved_buffer_ambiguous_peer(*, FLAG):
    if FLAG:
        pto.get_tid_x()
    pto.reserve_buffer("simt_c2v_fifo", size=8192, location="vec")


@pto.jit(target="a5")
def simt_pointer_offset_probe():
    meta_tile = pto.alloc_tile(shape=[1, 8], dtype=pto.i32, valid_shape=[1, 2])
    simt_pointer_offset_helper(meta_tile.as_ptr())
    first = scalar.load(meta_tile.as_ptr() + 0)
    second = scalar.load(meta_tile.as_ptr() + 1)
    _ = first
    _ = second


@pto.jit(target="a5")
def simt_reserved_buffer_peer_probe():
    simt_reserved_buffer_peer()
    imported = pto.import_reserved_buffer("simt_c2v_fifo", peer_func=simt_reserved_buffer_peer)
    _ = imported


@pto.jit(target="a5")
def simt_reserved_buffer_ambiguous_peer_probe():
    simt_reserved_buffer_ambiguous_peer(FLAG=True)
    simt_reserved_buffer_ambiguous_peer(FLAG=False)
    pto.import_reserved_buffer("simt_c2v_fifo", peer_func=simt_reserved_buffer_ambiguous_peer)


@pto.jit(target="a5")
def scalar_store_element_coercion_probe():
    meta_tile = pto.alloc_tile(shape=[1, 8], dtype=pto.i32, valid_shape=[1, 4])
    meta_ptr = meta_tile.as_ptr()
    row_start = pto.const(0)
    row_stop = pto.const(4)
    scalar.store(row_start, meta_ptr + 0)
    scalar.store(row_stop, meta_ptr + 1)
    scalar.store(pto.const(2, dtype=pto.i64), meta_ptr + 2)
    scalar.store(3, meta_ptr + 3)


@pto.jit(target="a5")
def shared_index_coercion_probe():
    limit = pto.const(4, dtype=pto.i32)
    step = pto.const(1, dtype=pto.i32)
    meta_tile = pto.alloc_tile(shape=[1, 8], dtype=pto.i32, valid_shape=[1, 4])
    meta_ptr = meta_tile.as_ptr()
    with pto.for_(0, limit, step=step) as i:
        ptr = pto.addptr(meta_ptr, limit)
        scalar.store(i, ptr)
        pto.wait_flag(pto.Pipe.V, pto.Pipe.MTE2, event_id=limit)


@pto.tileop
def public_vector_surface_probe(inp_tile: pto.Tile, out_tile: pto.Tile, stats_tile: pto.Tile):
    col_mask = pto.make_mask(pto.f32, pto.const(16, dtype=pto.i32))
    row = pto.const(0)
    s_row = pto.vlds(inp_tile[row, 0:])
    pto.vsubs(s_row, pto.const(1.0, dtype=pto.f32), col_mask)
    row_max = pto.vcgmax(s_row, col_mask)
    row_max_broadcast = pto.vdup(row_max, col_mask)
    s_shifted = pto.vsub(s_row, row_max_broadcast, col_mask)
    p_row = pto.vexp(s_shifted, col_mask)
    row_sum = pto.vcgadd(p_row, col_mask)
    # Register-level probe: only assert that pto.vsqz emits the op. The
    # compacted result is intentionally discarded here — a real compress_store
    # would consume it via pto.init_align + pto.vstur(POST_UPDATE) + pto.vstar
    # (see docs/user_guide/08-compute-operations.md). This probe deliberately
    # does NOT chain a masked pto.vsts, because the original col_mask selects
    # source lanes, not compacted positions, and would scatter the survivors.
    _ = pto.vsqz(p_row, col_mask)
    pto.vsts(p_row, out_tile[row, 0:], col_mask)
    pto.vsts(row_max, stats_tile.as_ptr(), row, col_mask, dist="1PT_B32")
    pto.vsts(row_sum, stats_tile.as_ptr(), row + 1, col_mask, dist="1PT_B32")


@pto.jit(target="a5", entry=False, mode="explicit", kernel_kind="cube")
def public_cube_surface_probe(
    lhs_tile: pto.Tile,
    rhs_tile: pto.Tile,
    lhs_tile_mx: pto.Tile,
    rhs_tile_mx: pto.Tile,
    lhs_l0a: pto.Tile,
    rhs_l0b: pto.Tile,
    lhs_l0a_f32: pto.Tile,
    rhs_l0b_f32: pto.Tile,
    lhs_l0a_mx: pto.Tile,
    rhs_l0b_mx: pto.Tile,
    lhs_scale_mx: pto.Tile,
    rhs_scale_mx: pto.Tile,
    acc_tile: pto.Tile,
    bias_tile: pto.Tile,
    l1_out_tile: pto.Tile,
    out_tile: pto.Tile,
):
    m = pto.const(16)
    k = pto.const(16)
    n = pto.const(16)
    start_row = pto.const(2)
    start_col = pto.const(4)
    pto.mte_l1_l0a(
        lhs_tile.as_ptr(),
        lhs_l0a.as_ptr(),
        m=m,
        k=k,
        start_row=start_row,
        start_col=start_col,
    )
    pto.mte_l1_l0b(
        rhs_tile.as_ptr(),
        rhs_l0b.as_ptr(),
        k=k,
        n=n,
        start_row=start_col,
        start_col=start_row,
        transpose=True,
    )
    pto.mte_l1_l0a_mx(
        lhs_tile_mx.as_ptr(),
        lhs_l0a_mx.as_ptr(),
        m,
        k,
        start_row=start_row,
        start_col=start_col,
    )
    pto.mte_l1_l0b_mx(
        rhs_tile_mx.as_ptr(),
        rhs_l0b_mx.as_ptr(),
        k,
        n,
        start_row=start_col,
        start_col=start_row,
    )
    pto.mad(
        lhs_l0a.as_ptr(),
        rhs_l0b.as_ptr(),
        acc_tile.as_ptr(),
        m,
        n,
        k,
        unit_flag=pto.MadUnitFlagMode.CHECK_ONLY,
        disable_gemv=True,
        sat=pto.SatMode.OFF,
        n_dir=True,
    )
    pto.mad(lhs_l0a_f32.as_ptr(), rhs_l0b_f32.as_ptr(), acc_tile.as_ptr(), m, n, k, tf32_mode=pto.Tf32Mode.ROUND_EVEN)
    pto.mad_acc(lhs_l0a.as_ptr(), rhs_l0b.as_ptr(), acc_tile.as_ptr(), m, n, k, unit_flag=pto.MadUnitFlagMode.CHECK_AND_SET)
    pto.mad_bias(lhs_l0a.as_ptr(), rhs_l0b.as_ptr(), acc_tile.as_ptr(), bias_tile.as_ptr(), m, n, k, sat=pto.SatMode.ON)
    pto.mad_mx(lhs_l0a_mx.as_ptr(), rhs_l0b_mx.as_ptr(), acc_tile.as_ptr(), m, n, k, unit_flag=pto.MadUnitFlagMode.CHECK_ONLY)
    pto.mad_mx_acc(lhs_l0a_mx.as_ptr(), rhs_l0b_mx.as_ptr(), acc_tile.as_ptr(), m, n, k, disable_gemv=True)
    pto.mad_mx_bias(lhs_l0a_mx.as_ptr(), rhs_l0b_mx.as_ptr(), acc_tile.as_ptr(), bias_tile.as_ptr(), m, n, k, n_dir=True)
    pto.mte_l0c_l1(
        acc_tile.as_ptr(),
        l1_out_tile.as_ptr(),
        m,
        n,
        n,
        n,
        unit_flag=pto.AccStoreUnitFlagCtrl.CHECK_ONLY,
        layout="nz2nd",
        loop3=(1, n, n),
        sat=pto.SatMode.ON,
    )
    pto.mte_l0c_ub(acc_tile.as_ptr(), out_tile.as_ptr(), m, n, n, n, 0)
    pto.mte_l0c_ub(acc_tile.as_ptr(), out_tile.as_ptr(), m, n, n, n, split=pto.SplitMode.M, layout="nz2nd")


@pto.jit(target="a5", mode="explicit")
def acc_store_pre_quant_surface_probe(
    dst: pto.ptr(pto.bf16, "gm"),
):
    size = pto.const(16)
    zero = pto.const(0)
    src = pto.alloc_tile(
        shape=[16, 16],
        dtype=pto.f32,
        memory_space=pto.MemorySpace.ACC,
        valid_shape=[16, 16],
        blayout="ColMajor",
        slayout="RowMajor",
    )
    pto.mte_l0c_gm(
        src.as_ptr(),
        dst,
        size,
        size,
        size,
        size,
        zero,
        zero,
        pre_quant=(pto.bf16(1.0), "f32_bf16"),
        layout="nz2nd",
    )


@pto.jit(target="a5", mode="explicit")
def acc_store_bad_pre_quant_mode_probe(
    dst: pto.ptr(pto.bf16, "gm"),
):
    size = pto.const(16)
    zero = pto.const(0)
    src = pto.alloc_tile(
        shape=[16, 16],
        dtype=pto.f32,
        memory_space=pto.MemorySpace.ACC,
        valid_shape=[16, 16],
        blayout="ColMajor",
        slayout="RowMajor",
    )
    pto.mte_l0c_gm(
        src.as_ptr(),
        dst,
        size,
        size,
        size,
        size,
        zero,
        zero,
        pre_quant=(pto.bf16(1.0), "not_a_mode"),
        layout="nz2nd",
    )


@pto.jit(target="a5", mode="explicit")
def acc_store_bad_pre_quant_payload_probe(
    dst: pto.ptr(pto.bf16, "gm"),
):
    size = pto.const(16)
    zero = pto.const(0)
    src = pto.alloc_tile(
        shape=[16, 16],
        dtype=pto.f32,
        memory_space=pto.MemorySpace.ACC,
        valid_shape=[16, 16],
        blayout="ColMajor",
        slayout="RowMajor",
    )
    payload = pto.alloc_tile(
        shape=[16],
        dtype=pto.bf16,
        memory_space=pto.MemorySpace.SCALING,
        valid_shape=[16],
    )
    pto.mte_l0c_gm(
        src.as_ptr(),
        dst,
        size,
        size,
        size,
        size,
        zero,
        zero,
        pre_quant=(payload.as_ptr(), "f32_bf16"),
        layout="nz2nd",
    )


@pto.jit(target="a5", entry=False, mode="explicit", kernel_kind="cube")
def public_cube_tile_mx_probe(
    mat_lhs: pto.Tile,
    mat_lhs_scale: pto.Tile,
    mat_rhs: pto.Tile,
    mat_rhs_scale: pto.Tile,
    mat_acc: pto.Tile,
    mat_bias: pto.Tile,
    gemv_lhs: pto.Tile,
    gemv_lhs_scale: pto.Tile,
    gemv_rhs: pto.Tile,
    gemv_rhs_scale: pto.Tile,
    gemv_acc: pto.Tile,
    gemv_bias: pto.Tile,
):
    pto.tile.matmul_mx(mat_lhs, mat_lhs_scale, mat_rhs, mat_rhs_scale, mat_acc)
    pto.tile.matmul_mx_acc(mat_acc, mat_lhs, mat_lhs_scale, mat_rhs, mat_rhs_scale, mat_acc)
    pto.tile.matmul_mx_bias(mat_lhs, mat_lhs_scale, mat_rhs, mat_rhs_scale, mat_bias, mat_acc)
    pto.tile.gemv_mx(gemv_lhs, gemv_lhs_scale, gemv_rhs, gemv_rhs_scale, gemv_acc)
    pto.tile.gemv_mx_acc(gemv_acc, gemv_lhs, gemv_lhs_scale, gemv_rhs, gemv_rhs_scale, gemv_acc)
    pto.tile.gemv_mx_bias(gemv_lhs, gemv_lhs_scale, gemv_rhs, gemv_rhs_scale, gemv_bias, gemv_acc)


@pto.jit(target="a5", mode="explicit")
def public_surface_exports_probe(
    A_ptr: pto.ptr(pto.f32, "gm"),
    O_ptr: pto.ptr(pto.f32, "gm"),
    rows: pto.i32,
    cols: pto.i32,
):
    pto.mem_bar(pto.BarrierType.VST_VLD)
    pto.pipe_barrier(pto.Pipe.ALL)

    a_view = pto.make_tensor_view(A_ptr, shape=[rows, cols], strides=[cols, 1])
    o_view = pto.make_tensor_view(O_ptr, shape=[rows, cols], strides=[cols, 1])
    a_part = pto.partition_view(a_view, offsets=[0, 0], sizes=[1, 16])
    o_part = pto.partition_view(o_view, offsets=[0, 0], sizes=[1, 16])
    dma_tile = pto.alloc_tile(shape=[1, 128], dtype=pto.f32, valid_shape=[1, 16])
    pto.mte_load(a_part.as_ptr(), dma_tile.as_ptr(), 0, 64, nburst=(1, 0, 128))
    pto.pipe_barrier(pto.Pipe.ALL)
    pto.mte_store(dma_tile.as_ptr(), o_part.as_ptr(), 64, nburst=(1, 128, 0))

    vec_in = pto.alloc_tile(shape=[1, 128], dtype=pto.f32, valid_shape=[1, 16])
    vec_out = pto.alloc_tile(shape=[1, 128], dtype=pto.f32, valid_shape=[1, 16])
    stats_tile = pto.alloc_tile(shape=[1, 8], dtype=pto.f32, valid_shape=[1, 2])
    public_vector_surface_probe(vec_in, vec_out, stats_tile)

    lhs_tile = pto.alloc_tile(
        shape=[16, 16],
        dtype=pto.f16,
        memory_space=pto.MemorySpace.MAT,
        valid_shape=[16, 16],
    )
    rhs_tile = pto.alloc_tile(
        shape=[16, 16],
        dtype=pto.f16,
        memory_space=pto.MemorySpace.MAT,
        valid_shape=[16, 16],
    )
    lhs_tile_mx = pto.alloc_tile(
        shape=[16, 32],
        dtype=pto.f8e4m3,
        memory_space=pto.MemorySpace.MAT,
        valid_shape=[16, 16],
    )
    rhs_tile_mx = pto.alloc_tile(
        shape=[16, 32],
        dtype=pto.f8e4m3,
        memory_space=pto.MemorySpace.MAT,
        valid_shape=[16, 16],
    )
    lhs_l0a = pto.alloc_tile(
        shape=[16, 16],
        dtype=pto.f16,
        memory_space=pto.MemorySpace.LEFT,
        valid_shape=[16, 16],
    )
    rhs_l0b = pto.alloc_tile(
        shape=[16, 16],
        dtype=pto.f16,
        memory_space=pto.MemorySpace.RIGHT,
        valid_shape=[16, 16],
    )
    lhs_l0a_f32 = pto.alloc_tile(
        shape=[16, 16],
        dtype=pto.f32,
        memory_space=pto.MemorySpace.LEFT,
        valid_shape=[16, 16],
    )
    rhs_l0b_f32 = pto.alloc_tile(
        shape=[16, 16],
        dtype=pto.f32,
        memory_space=pto.MemorySpace.RIGHT,
        valid_shape=[16, 16],
    )
    pto.mte_l1_l0a(
        lhs_tile.as_ptr(),
        lhs_l0a.as_ptr(),
        m_start=4,
        k_start=0,
        m_step=4,
        k_step=16,
        src_stride=16,
        dst_stride=4,
        transpose=True,
    )
    pto.mte_l1_l0b(
        rhs_tile.as_ptr(),
        rhs_l0b.as_ptr(),
        m_start=4,
        k_start=0,
        m_step=4,
        k_step=16,
        src_stride=16,
        dst_stride=4,
        transpose=True,
    )
    lhs_l0a_mx = pto.alloc_tile(
        shape=[16, 32],
        dtype=pto.f8e4m3,
        memory_space=pto.MemorySpace.LEFT,
        valid_shape=[16, 16],
        blayout="ColMajor",
        slayout="RowMajor",
    )
    rhs_l0b_mx = pto.alloc_tile(
        shape=[32, 16],
        dtype=pto.f8e4m3,
        memory_space=pto.MemorySpace.RIGHT,
        valid_shape=[16, 16],
        blayout="RowMajor",
        slayout="ColMajor",
    )
    lhs_scale_mx = pto.alloc_tile(
        shape=[16, 2],
        dtype=pto.f16,
        memory_space=pto.MemorySpace.SCALING,
        valid_shape=[16, 2],
        blayout="RowMajor",
        slayout="RowMajor",
        fractal_size=32,
    )
    rhs_scale_mx = pto.alloc_tile(
        shape=[2, 16],
        dtype=pto.f16,
        memory_space=pto.MemorySpace.SCALING,
        valid_shape=[2, 16],
        blayout="ColMajor",
        slayout="ColMajor",
        fractal_size=32,
    )
    acc_tile = pto.alloc_tile(
        shape=[16, 16],
        dtype=pto.f32,
        memory_space=pto.MemorySpace.ACC,
        valid_shape=[16, 16],
        blayout="ColMajor",
        slayout="RowMajor",
    )
    mat_lhs_tile_mx = pto.alloc_tile(
        shape=[16, 64],
        dtype=pto.f8e4m3,
        memory_space=pto.MemorySpace.LEFT,
        valid_shape=[16, 64],
        blayout="ColMajor",
        slayout="RowMajor",
    )
    mat_rhs_tile_mx = pto.alloc_tile(
        shape=[64, 16],
        dtype=pto.f8e4m3,
        memory_space=pto.MemorySpace.RIGHT,
        valid_shape=[64, 16],
        blayout="RowMajor",
        slayout="ColMajor",
    )
    mat_lhs_scale_tile_mx = pto.alloc_tile(
        shape=[16, 2],
        dtype=pto.f16,
        memory_space=pto.MemorySpace.SCALING,
        valid_shape=[16, 2],
        blayout="RowMajor",
        slayout="RowMajor",
        fractal_size=32,
    )
    mat_rhs_scale_tile_mx = pto.alloc_tile(
        shape=[2, 16],
        dtype=pto.f16,
        memory_space=pto.MemorySpace.SCALING,
        valid_shape=[2, 16],
        blayout="ColMajor",
        slayout="ColMajor",
        fractal_size=32,
    )
    mat_acc_tile_mx = pto.alloc_tile(
        shape=[16, 16],
        dtype=pto.f32,
        memory_space=pto.MemorySpace.ACC,
        valid_shape=[16, 16],
        blayout="ColMajor",
        slayout="RowMajor",
    )
    mat_bias_tile_mx = pto.alloc_tile(
        shape=[1, 16],
        dtype=pto.f32,
        memory_space=pto.MemorySpace.BIAS,
        valid_shape=[1, 16],
    )
    gemv_lhs_tile_mx = pto.alloc_tile(
        shape=[1, 64],
        dtype=pto.f8e4m3,
        memory_space=pto.MemorySpace.LEFT,
        valid_shape=[1, 64],
        blayout="ColMajor",
        slayout="RowMajor",
    )
    gemv_rhs_tile_mx = pto.alloc_tile(
        shape=[64, 16],
        dtype=pto.f8e4m3,
        memory_space=pto.MemorySpace.RIGHT,
        valid_shape=[64, 16],
        blayout="RowMajor",
        slayout="ColMajor",
    )
    gemv_lhs_scale_tile_mx = pto.alloc_tile(
        shape=[1, 2],
        dtype=pto.f16,
        memory_space=pto.MemorySpace.SCALING,
        valid_shape=[1, 2],
        blayout="RowMajor",
        slayout="RowMajor",
        fractal_size=32,
    )
    gemv_rhs_scale_tile_mx = pto.alloc_tile(
        shape=[2, 16],
        dtype=pto.f16,
        memory_space=pto.MemorySpace.SCALING,
        valid_shape=[2, 16],
        blayout="ColMajor",
        slayout="ColMajor",
        fractal_size=32,
    )
    gemv_acc_tile_mx = pto.alloc_tile(
        shape=[1, 16],
        dtype=pto.f32,
        memory_space=pto.MemorySpace.ACC,
        valid_shape=[1, 16],
        blayout="ColMajor",
        slayout="RowMajor",
    )
    gemv_bias_tile_mx = pto.alloc_tile(
        shape=[1, 16],
        dtype=pto.f32,
        memory_space=pto.MemorySpace.BIAS,
        valid_shape=[1, 16],
    )
    bias_tile = pto.alloc_tile(
        shape=[16, 16],
        dtype=pto.f32,
        memory_space=pto.MemorySpace.BIAS,
        valid_shape=[16, 16],
    )
    l1_out = pto.alloc_tile(
        shape=[16, 16],
        dtype=pto.f32,
        memory_space=pto.MemorySpace.MAT,
        valid_shape=[16, 16],
    )
    cube_out = pto.alloc_tile(shape=[16, 16], dtype=pto.f32, valid_shape=[16, 16])
    public_cube_surface_probe(
        lhs_tile,
        rhs_tile,
        lhs_tile_mx,
        rhs_tile_mx,
        lhs_l0a,
        rhs_l0b,
        lhs_l0a_f32,
        rhs_l0b_f32,
        lhs_l0a_mx,
        rhs_l0b_mx,
        lhs_scale_mx,
        rhs_scale_mx,
        acc_tile,
        bias_tile,
        l1_out,
        cube_out,
    )
    pto.mte_l0c_gm(
        acc_tile.as_ptr(),
        O_ptr,
        pto.const(16),
        pto.const(16),
        pto.const(16),
        pto.const(16),
        0,
        0,
        layout=("nz2dn", pto.const(16)),
        loop3=(1, pto.const(16), pto.const(16)),
        sat=pto.SatMode.OFF,
        atomic=("f32", "add"),
    )
    public_cube_tile_mx_probe(
        mat_lhs_tile_mx,
        mat_lhs_scale_tile_mx,
        mat_rhs_tile_mx,
        mat_rhs_scale_tile_mx,
        mat_acc_tile_mx,
        mat_bias_tile_mx,
        gemv_lhs_tile_mx,
        gemv_lhs_scale_tile_mx,
        gemv_rhs_tile_mx,
        gemv_rhs_scale_tile_mx,
        gemv_acc_tile_mx,
        gemv_bias_tile_mx,
    )


@pto.jit(target="a5", backend="vpto", mode="explicit")
def vmi_wrapper_dispatch_probe():
    src_tile = pto.alloc_tile(shape=[1, 64], dtype=pto.f32)
    other_tile = pto.alloc_tile(shape=[1, 64], dtype=pto.f32)
    dst_tile = pto.alloc_tile(shape=[1, 64], dtype=pto.f32)
    int_src_tile = pto.alloc_tile(shape=[1, 64], dtype=pto.i32)
    int_other_tile = pto.alloc_tile(shape=[1, 64], dtype=pto.i32)
    hist_acc_tile = pto.alloc_tile(shape=[1, 256], dtype=pto.ui16)
    hist_src_tile = pto.alloc_tile(shape=[1, 256], dtype=pto.ui8)
    half_src_tile = pto.alloc_tile(shape=[1, 128], dtype=pto.f16)
    half_max_tile = pto.alloc_tile(shape=[1, 128], dtype=pto.f16)

    src_ptr = src_tile.as_ptr()
    other_ptr = other_tile.as_ptr()
    dst_ptr = dst_tile.as_ptr()
    int_src_ptr = int_src_tile.as_ptr()
    int_other_ptr = int_other_tile.as_ptr()
    hist_acc_ptr = hist_acc_tile.as_ptr()
    hist_src_ptr = hist_src_tile.as_ptr()
    half_src_ptr = half_src_tile.as_ptr()
    half_max_ptr = half_max_tile.as_ptr()

    offset = pto.const(0, dtype=pto.index)
    active_lanes = pto.const(64, dtype=pto.index)
    active_per_group = pto.const(8, dtype=pto.index)

    mask = pto.vmi.create_mask(
        active_lanes,
        size=64,
    )
    group_mask = pto.vmi.create_mask(
        active_per_group,
        size=64,
        group=8,
    )
    lhs = pto.vmi.vload(src_ptr, offset, size=64)
    rhs = pto.vmi.vload(other_ptr, offset, size=64)
    compact = pto.vmi.vload(src_ptr, offset, size=8)
    idx = pto.vmi.vci(pto.i32(0), size=64, order="ASC")
    bias = pto.vmi.vbrc(pto.f32(0.0), size=64)
    expanded = pto.vmi.vbrc(compact, size=64, group=8)
    hist_acc = pto.vmi.vload(hist_acc_ptr, offset, size=256)
    hist_src = pto.vmi.vload(hist_src_ptr, offset, size=256)
    int_lhs = pto.vmi.vload(int_src_ptr, offset, size=64)
    int_rhs = pto.vmi.vload(int_other_ptr, offset, size=64)
    hist_mask = pto.vmi.create_mask(pto.const(256, dtype=pto.index), size=256)
    half_mask = pto.vmi.create_mask(pto.const(128, dtype=pto.index), size=128)
    half_lhs = pto.vmi.vload(half_src_ptr, offset, size=128)
    half_max = pto.vmi.vload(half_max_ptr, offset, size=128)
    added = pto.vmi.vadd(lhs, rhs, mask)
    carry_sum, carry = pto.vmi.vaddc(int_lhs, int_rhs, mask)
    carry_next, carry_out = pto.vmi.vaddcs(carry_sum, int_rhs, carry, mask)
    subtracted = pto.vmi.vsub(lhs, rhs, mask)
    multiplied = pto.vmi.vmul(lhs, rhs, mask)
    divided = pto.vmi.vdiv(lhs, rhs, mask)
    maximum = pto.vmi.vmax(lhs, rhs, mask)
    minimum = pto.vmi.vmin(lhs, rhs, mask)
    absolute = pto.vmi.vabs(lhs, mask)
    negated = pto.vmi.vneg(lhs, mask)
    relu = pto.vmi.vrelu(added, mask)
    exponent = pto.vmi.vexp(lhs, mask)
    logarithm = pto.vmi.vln(lhs, mask)
    square_root = pto.vmi.vsqrt(lhs, mask)
    int_and = pto.vmi.vand(int_lhs, int_rhs, mask)
    int_or = pto.vmi.vor(int_lhs, int_rhs, mask)
    int_xor = pto.vmi.vxor(int_lhs, int_rhs, mask)
    int_not = pto.vmi.vnot(int_lhs, mask)
    int_shl = pto.vmi.vshl(int_lhs, int_rhs, mask)
    int_shr = pto.vmi.vshr(int_lhs, int_rhs, mask)
    scalar_added = pto.vmi.vadd(relu, 1.0, mask)
    scalar_multiplied = pto.vmi.vmul(relu, 2.0, mask)
    scalar_maximum = pto.vmi.vmax(relu, 1.0, mask)
    scalar_minimum = pto.vmi.vmin(relu, 1.0, mask)
    scalar_shl = pto.vmi.vshl(int_lhs, pto.i32(1), mask)
    scalar_shr = pto.vmi.vshr(int_lhs, pto.i32(1), mask)
    scaled = pto.vmi.vadd(scalar_multiplied, bias, mask)
    pred = pto.vmi.vcmp(scaled, lhs, mask, "ogt")
    scalar_pred = pto.vmi.vcmps(scaled, 0.0, mask, "ogt")
    selected = pto.vmi.vsel(pred, scaled, expanded)
    shuffled = pto.vmi.vselr(selected, idx)
    total = pto.vmi.vcadd(shuffled, mask, reassoc=False)
    explicit_total = pto.vmi.vcadd(shuffled, mask, group=1, reassoc=False)
    peak = pto.vmi.vcmax(shuffled, mask)
    explicit_peak = pto.vmi.vcmax(shuffled, mask, group=1)
    floor = pto.vmi.vcmin(shuffled, mask)
    explicit_floor = pto.vmi.vcmin(shuffled, mask, group=1)
    group_peak = pto.vmi.vcmax(shuffled, group_mask, group=8)
    gather = pto.vmi.vgather(src_ptr, idx, mask)
    gatherb = pto.vmi.vgatherb(src_ptr, idx, mask)
    hist = pto.vmi.vdhist(hist_acc, hist_src, hist_mask)
    cumul = pto.vmi.vchist(hist_acc, hist_src, hist_mask)
    exp_difference = pto.vmi.vexpdif(lhs, rhs, mask)
    half_exp_difference = pto.vmi.vexpdif(half_lhs, half_max, half_mask)
    axpy = pto.vmi.vaxpy(lhs, rhs, 2.0, mask)
    leaky_relu = pto.vmi.vlrelu(lhs, 0.125, mask)
    parametric_relu = pto.vmi.vprelu(lhs, rhs, mask)
    low, high = pto.vmi.vmull(int_lhs, int_rhs, mask)
    multiply_accumulate = pto.vmi.vmula(lhs, lhs, rhs, mask)
    widened = pto.vmi.vadd(low, high, mask)
    casted = pto.vmi.vcvt(shuffled, pto.f16)
    casted_r = pto.vmi.vcvt(shuffled, pto.f16, rounding="R", saturate="SAT")
    casted_a = pto.vmi.vcvt(shuffled, pto.f16, rounding="A", saturate="SAT")
    casted_h = pto.vmi.vcvt(shuffled, pto.f16, rounding="H", saturate="SAT")
    casted_z = pto.vmi.vcvt(shuffled, pto.f16, rounding="Z", saturate="SAT")
    recast = pto.vmi.vinterpret_cast(
        lhs,
        pto.i32,
    )
    recast_narrow = pto.vmi.vinterpret_cast(lhs, pto.f16)
    lo, hi = pto.vmi.vintlv(selected, shuffled, mask)
    even, odd = pto.vmi.vdintlv(lo, hi, mask)
    pto.vmi.vscatter(selected, dst_ptr, idx, mask)
    pto.vmi.vstore(lo, dst_ptr, offset, mask)
    pto.vmi.vsstb(hi, dst_ptr, offset, pto.i16(8), mask)

    _ = group_mask
    _ = carry_next
    _ = carry_out
    _ = total
    _ = explicit_total
    _ = peak
    _ = explicit_peak
    _ = floor
    _ = explicit_floor
    _ = group_peak
    _ = gather
    _ = gatherb
    _ = hist
    _ = cumul
    _ = widened
    _ = casted
    _ = (casted_r, casted_a, casted_h, casted_z)
    _ = recast
    _ = recast_narrow
    _ = hi
    _ = (subtracted, multiplied, divided, maximum, minimum, absolute, negated)
    _ = (exponent, logarithm, square_root, int_and, int_or, int_xor, int_not)
    _ = (int_shl, int_shr, scalar_added, scalar_maximum, scalar_minimum)
    _ = (scalar_shl, scalar_shr, scalar_pred, exp_difference, axpy)
    _ = (leaky_relu, parametric_relu, multiply_accumulate, even, odd)


@pto.jit(target="a5", backend="vpto", mode="explicit")
def vmi_vhist_signless_source_probe():
    # acc/result stay ui16 (DSL-enforced); source is signless i8 to exercise
    # the VMI verifier's "signless == unsigned" relaxation for vdhist/vchist.
    hist_acc_tile = pto.alloc_tile(shape=[1, 256], dtype=pto.ui16)
    hist_src_tile = pto.alloc_tile(shape=[1, 256], dtype=pto.i8)
    hist_acc_ptr = hist_acc_tile.as_ptr()
    hist_src_ptr = hist_src_tile.as_ptr()
    offset = pto.const(0, dtype=pto.index)
    hist_acc = pto.vmi.vload(hist_acc_ptr, offset, size=256)
    hist_src = pto.vmi.vload(hist_src_ptr, offset, size=256)
    hist_mask = pto.vmi.create_mask(pto.const(256, dtype=pto.index), size=256)
    _ = pto.vmi.vdhist(hist_acc, hist_src, hist_mask)
    _ = pto.vmi.vchist(hist_acc, hist_src, hist_mask)


@pto.jit(target="a5", backend="vpto", mode="explicit")
def vmi_vhist_signless_acc_probe():
    # acc is signless i16 - DSL should accept it (treated as ui16) and emit a
    # ui16 result for both vdhist and vchist.
    hist_acc_tile = pto.alloc_tile(shape=[1, 256], dtype=pto.i16)
    hist_src_tile = pto.alloc_tile(shape=[1, 256], dtype=pto.ui8)
    hist_acc_ptr = hist_acc_tile.as_ptr()
    hist_src_ptr = hist_src_tile.as_ptr()
    offset = pto.const(0, dtype=pto.index)
    hist_acc = pto.vmi.vload(hist_acc_ptr, offset, size=256)
    hist_src = pto.vmi.vload(hist_src_ptr, offset, size=256)
    hist_mask = pto.vmi.create_mask(pto.const(256, dtype=pto.index), size=256)
    _ = pto.vmi.vdhist(hist_acc, hist_src, hist_mask)
    _ = pto.vmi.vchist(hist_acc, hist_src, hist_mask)


@pto.jit(target="a5", backend="vpto", mode="explicit")
def vmi_vhist_bad_acc_probe():
    # acc is signed si16 - must be rejected by the PTODSL surface with a
    # TypeError before the MLIR verifier runs.
    hist_acc_tile = pto.alloc_tile(shape=[1, 256], dtype=pto.si16)
    hist_src_tile = pto.alloc_tile(shape=[1, 256], dtype=pto.ui8)
    hist_acc_ptr = hist_acc_tile.as_ptr()
    hist_src_ptr = hist_src_tile.as_ptr()
    offset = pto.const(0, dtype=pto.index)
    hist_acc = pto.vmi.vload(hist_acc_ptr, offset, size=256)
    hist_src = pto.vmi.vload(hist_src_ptr, offset, size=256)
    hist_mask = pto.vmi.create_mask(pto.const(256, dtype=pto.index), size=256)
    _ = pto.vmi.vdhist(hist_acc, hist_src, hist_mask)


@pto.jit(target="a5", backend="vpto", mode="explicit")
def vmi_missing_binding_probe():
    src_tile = pto.alloc_tile(shape=[1, 64], dtype=pto.f32)
    src_ptr = src_tile.as_ptr()
    offset = pto.const(0, dtype=pto.index)
    active_lanes = pto.const(64, dtype=pto.index)

    src = pto.vmi.vload(src_ptr, offset, size=64)
    mask = pto.vmi.create_mask(
        active_lanes,
        size=64,
    )
    _ = pto.vmi.vadd(src, src, mask)


@pto.jit(target="a5", backend="vpto", mode="explicit")
def vmi_masked_vcvt_probe():
    src_tile = pto.alloc_tile(shape=[1, 64], dtype=pto.f32)
    src_ptr = src_tile.as_ptr()
    offset = pto.const(0, dtype=pto.index)
    active_lanes = pto.const(64, dtype=pto.index)

    src = pto.vmi.vload(src_ptr, offset, size=64)
    mask = pto.vmi.create_mask(
        active_lanes,
        size=64,
    )
    _ = pto.vmi.vcvt(src, pto.f16, mask)


@pto.jit(target="a5", backend="vpto", mode="explicit")
def vmi_invalid_rounding_vcvt_probe():
    src_tile = pto.alloc_tile(shape=[1, 64], dtype=pto.f32)
    src_ptr = src_tile.as_ptr()
    offset = pto.const(0, dtype=pto.index)

    src = pto.vmi.vload(src_ptr, offset, size=64)
    _ = pto.vmi.vcvt(
        src,
        pto.f8e4m3,
        rounding=pto.VcvtRoundMode.F,
        saturate=pto.VcvtSatMode.SAT,
    )


@pto.jit(target="a5", backend="vpto", mode="explicit")
def vmi_round_r_vcvt_probe():
    src_tile = pto.alloc_tile(shape=[1, 64], dtype=pto.f32)
    src_ptr = src_tile.as_ptr()
    offset = pto.const(0, dtype=pto.index)

    src = pto.vmi.vload(src_ptr, offset, size=64)
    _ = pto.vmi.vcvt(
        src,
        pto.f8e4m3,
        rounding=pto.VcvtRoundMode.R,
        saturate=pto.VcvtSatMode.SAT,
    )


@pto.jit(target="a5", backend="vpto", mode="explicit")
def vmi_bf16x2_to_f4x2_vcvt_probe():
    src_tile = pto.alloc_tile(shape=[1, 256], dtype=pto.bf16)
    src_ptr = src_tile.as_ptr()
    offset = pto.const(0, dtype=pto.index)

    wide = pto.vmi.vload(src_ptr, offset, size=256)
    pair = pto.vmi.vinterpret_cast(wide, to_dtype=pto.vmi.bf16x2)
    default_e1 = pto.vmi.vcvt(pair, pto.f4e1m2x2)
    default_e2 = pto.vmi.vcvt(pair, pto.f4e2m1x2)
    round_r = pto.vmi.vcvt(pair, pto.f4e1m2x2, rounding=pto.VcvtRoundMode.R)
    round_a = pto.vmi.vcvt(pair, pto.f4e1m2x2, rounding=pto.VcvtRoundMode.A)
    round_f = pto.vmi.vcvt(pair, pto.f4e1m2x2, rounding=pto.VcvtRoundMode.F)
    round_z = pto.vmi.vcvt(pair, pto.f4e2m1x2, rounding=pto.VcvtRoundMode.Z)
    round_c = pto.vmi.vcvt(pair, pto.f4e2m1x2, rounding=pto.VcvtRoundMode.C)

    _ = (default_e1, default_e2, round_r, round_a, round_f, round_z, round_c)


@pto.jit(target="a5", backend="vpto", mode="explicit")
def vmi_bf16x2_to_f4x2_vstore_probe():
    src_tile = pto.alloc_tile(shape=[1, 512], dtype=pto.bf16)
    dst_e1_tile = pto.alloc_tile(shape=[1, 128], dtype=pto.f4e1m2x2)
    dst_e2_tile = pto.alloc_tile(shape=[1, 128], dtype=pto.f4e2m1x2)
    src_offset = pto.const(64, dtype=pto.index)
    dst_offset = pto.const(16, dtype=pto.index)

    wide = pto.vmi.vload(src_tile.as_ptr(), src_offset, size=128)
    pair = pto.vmi.vinterpret_cast(wide, to_dtype=pto.vmi.bf16x2)
    out_e1 = pto.vmi.vcvt(pair, pto.f4e1m2x2)
    out_e2 = pto.vmi.vcvt(pair, pto.f4e2m1x2)

    pto.vmi.vstore(out_e1, dst_e1_tile.as_ptr(), dst_offset)
    pto.vmi.vstore(out_e2, dst_e2_tile.as_ptr(), dst_offset)


@pto.jit(target="a5", backend="vpto", mode="explicit")
def vmi_bf16x2_to_f4x2_invalid_rounding_probe():
    src_tile = pto.alloc_tile(shape=[1, 256], dtype=pto.bf16)
    src_ptr = src_tile.as_ptr()
    offset = pto.const(0, dtype=pto.index)

    wide = pto.vmi.vload(src_ptr, offset, size=256)
    pair = pto.vmi.vinterpret_cast(wide, to_dtype=pto.vmi.bf16x2)
    _ = pto.vmi.vcvt(pair, pto.f4e1m2x2, rounding=pto.VcvtRoundMode.H)


@pto.jit(target="a5", backend="vpto", mode="explicit")
def vmi_bf16x2_to_f4x2_sat_probe():
    src_tile = pto.alloc_tile(shape=[1, 256], dtype=pto.bf16)
    src_ptr = src_tile.as_ptr()
    offset = pto.const(0, dtype=pto.index)

    wide = pto.vmi.vload(src_ptr, offset, size=256)
    pair = pto.vmi.vinterpret_cast(wide, to_dtype=pto.vmi.bf16x2)
    _ = pto.vmi.vcvt(pair, pto.f4e1m2x2, saturate=pto.VcvtSatMode.SAT)


@pto.jit(target="a5", backend="vpto", mode="explicit")
def vmi_bf16x2_to_f4x2_nosat_probe():
    src_tile = pto.alloc_tile(shape=[1, 256], dtype=pto.bf16)
    src_ptr = src_tile.as_ptr()
    offset = pto.const(0, dtype=pto.index)

    wide = pto.vmi.vload(src_ptr, offset, size=256)
    pair = pto.vmi.vinterpret_cast(wide, to_dtype=pto.vmi.bf16x2)
    _ = pto.vmi.vcvt(pair, pto.f4e1m2x2, saturate=pto.VcvtSatMode.NOSAT)


@pto.jit(target="a5", backend="vpto", mode="explicit")
def vmi_bf16x2_unsupported_pair_probe():
    src_tile = pto.alloc_tile(shape=[1, 256], dtype=pto.bf16)
    src_ptr = src_tile.as_ptr()
    offset = pto.const(0, dtype=pto.index)

    wide = pto.vmi.vload(src_ptr, offset, size=256)
    pair = pto.vmi.vinterpret_cast(wide, to_dtype=pto.vmi.bf16x2)
    _ = pto.vmi.vcvt(pair, pto.f16)


@pto.jit(target="a5", backend="vpto", mode="explicit")
def vmi_f4x2_to_bf16x2_vcvt_probe():
    src_tile = pto.alloc_tile(shape=[1, 128], dtype=pto.f4e1m2x2)
    src_ptr = src_tile.as_ptr()
    offset = pto.const(0, dtype=pto.index)

    packed = pto.vmi.vload(src_ptr, offset, size=128)
    _ = pto.vmi.vcvt(packed, pto.vmi.bf16x2)


@pto.jit(target="a5", backend="vpto", mode="explicit")
def vmi_f4x2_to_bf16x2_rounding_probe():
    src_tile = pto.alloc_tile(shape=[1, 128], dtype=pto.f4e1m2x2)
    src_ptr = src_tile.as_ptr()
    offset = pto.const(0, dtype=pto.index)

    packed = pto.vmi.vload(src_ptr, offset, size=128)
    _ = pto.vmi.vcvt(packed, pto.vmi.bf16x2, rounding=pto.VcvtRoundMode.R)


@pto.jit(target="a5", backend="vpto", mode="explicit")
def vmi_f4x2_to_bf16x2_sat_probe():
    src_tile = pto.alloc_tile(shape=[1, 128], dtype=pto.f4e1m2x2)
    src_ptr = src_tile.as_ptr()
    offset = pto.const(0, dtype=pto.index)

    packed = pto.vmi.vload(src_ptr, offset, size=128)
    _ = pto.vmi.vcvt(packed, pto.vmi.bf16x2, saturate=pto.VcvtSatMode.SAT)


@pto.jit(target="a5", backend="vpto", mode="explicit")
def vmi_brc_vload_probe():
    src_tile = pto.alloc_tile(shape=[1, 64], dtype=pto.f32)
    src_ptr = src_tile.as_ptr()
    offset = pto.const(0, dtype=pto.index)

    _ = pto.vmi.vload(
        src_ptr,
        offset,
        size=64,
        dist_mode="brc",
    )


@pto.jit(target="a5", backend="vpto", mode="explicit")
def vmi_group_brc_vload_probe():
    src_tile = pto.alloc_tile(shape=[1, 64], dtype=pto.f32)
    src_ptr = src_tile.as_ptr()
    offset = pto.const(0, dtype=pto.index)
    row_stride = pto.const(1, dtype=pto.index)

    _ = pto.vmi.vload(
        src_ptr,
        offset,
        size=64,
        group=8,
        stride=row_stride,
        dist_mode="brc",
    )


@pto.jit(target="a5", backend="vpto", mode="explicit")
def vmi_block_stride_memory_probe():
    src_tile = pto.alloc_tile(shape=[1, 64], dtype=pto.f32)
    dst_tile = pto.alloc_tile(shape=[1, 64], dtype=pto.f32)
    offset = pto.const(0, dtype=pto.index)

    value = pto.vmi.vload(
        src_tile.as_ptr(), offset, size=64, block_stride=pto.i16(2)
    )
    pto.vmi.vstore(
        value, dst_tile.as_ptr(), offset, block_stride=pto.i16(2)
    )


@pto.jit(target="a5")
def compile_time_query_probe():
    f32_bw = pto.bytewidth(pto.f32)
    f16_bw = pto.bytewidth(pto.f16)
    i8_bw = pto.bytewidth(pto.i8)
    f32_vec = pto.elements_per_vreg(pto.f32)
    f16_vec = pto.elements_per_vreg(pto.f16)
    i8_vec = pto.elements_per_vreg(pto.i8)

    expect(f32_bw == 4, "pto.bytewidth(pto.f32) should evaluate to 4")
    expect(f16_bw == 2, "pto.bytewidth(pto.f16) should evaluate to 2")
    expect(i8_bw == 1, "pto.bytewidth(pto.i8) should evaluate to 1")
    expect(f32_vec == 64, "pto.elements_per_vreg(pto.f32) should evaluate to 64")
    expect(f16_vec == 128, "pto.elements_per_vreg(pto.f16) should evaluate to 128")
    expect(i8_vec == 256, "pto.elements_per_vreg(pto.i8) should evaluate to 256")


@pto.jit(target="a5")
def eager_scalar_constructor_probe():
    i32_val = pto.i32(1024)
    ui16_val = pto.ui16(7)
    si32_val = pto.si32(-7)
    ui8_val = pto.ui8(255)
    si8_neg1 = pto.si8(-1)
    ui8_bits = pto.ui8("0xFF")
    si16_bits = pto.si16("0xFFFF")
    f16_val = pto.f16(-1.5)
    f32_val = pto.f32("inf")
    f16_bits = pto.f16("0xFC00")
    i32_bits = pto.i32("0x80000000")

    _ = i32_val
    _ = ui16_val
    _ = si32_val
    _ = ui8_val
    _ = si8_neg1
    _ = ui8_bits
    _ = si16_bits
    _ = f16_val
    _ = f32_val
    _ = f16_bits
    _ = i32_bits


@pto.jit(target="a5")
def signed_integer_scalar_probe():
    signed_tile = pto.alloc_tile(shape=[1, 8], dtype=pto.si32, valid_shape=[1, 3])
    unsigned_tile = pto.alloc_tile(shape=[1, 8], dtype=pto.ui32, valid_shape=[1, 3])

    signed_ptr = signed_tile.as_ptr()
    unsigned_ptr = unsigned_tile.as_ptr()

    scalar.store(pto.si32(-7), signed_ptr + 0)
    scalar.store(pto.si32(5), signed_ptr + 1)
    scalar.store(pto.ui32("0xFFFFFFFF"), unsigned_ptr + 0)
    scalar.store(pto.ui32(9), unsigned_ptr + 1)

    s0 = scalar.load(signed_ptr + 0)
    s1 = scalar.load(signed_ptr + 1)
    u0 = scalar.load(unsigned_ptr + 0)
    u1 = scalar.load(unsigned_ptr + 1)

    s_add = s0 + 1
    u_add = u1 + 2
    s_max = scalar.max(s0, s1)
    s_min = scalar.min(s0, s1)
    u_max = scalar.max(u0, u1)
    u_min = scalar.min(u0, u1)
    s_abs = scalar.abs(s0)
    u_abs = scalar.abs(u0)
    s_cmp = s1 > s0
    u_cmp = u0 > u1

    scalar.store(s_add, signed_ptr + 2)
    scalar.store(u_add, unsigned_ptr + 2)

    _ = s_max
    _ = s_min
    _ = u_max
    _ = u_min
    _ = s_abs
    _ = u_abs
    _ = s_cmp
    _ = u_cmp


@pto.jit(target="a5")
def low_precision_storage_probe():
    lp_tile = pto.alloc_tile(shape=[128, 64], dtype=pto.f8e4m3)
    lp_tile_hif8 = pto.alloc_tile(shape=[64, 64], dtype=pto.hif8)
    lp_tile_ty = pto_types.tile_buf_type([16, 16], pto.f4e2m1x2, [16, 16])
    _ = lp_tile
    _ = lp_tile_hif8
    _ = lp_tile_ty


@pto.jit(target="a5")
def pointer_vlds_inference_probe(*, BLOCK: pto.const_expr = 128):
    tile = pto.alloc_tile(shape=[2, BLOCK], dtype=pto.f32)
    vec = pto.vlds(tile.as_ptr(), pto.const(0))
    vec_brc = pto.vlds(tile.as_ptr(), pto.const(0), dist="BRC_B32")
    ivec = pto.vbitcast(vec, pto.i32)
    f16_vec = pto.vbitcast(vec, pto.f16)
    _ = vec
    _ = vec_brc
    _ = ivec
    _ = f16_vec


@pto.jit(target="a5")
def public_mask_bitcast_probe():
    mask_b8, _ = pto.make_mask(pto.i8, pto.const(256, dtype=pto.i32))
    mask_b16 = pto.pbitcast(mask_b8, pto.mask_b16)
    mask_b32 = pto.pbitcast(mask_b16, pto.mask_b32)
    _ = mask_b8
    _ = mask_b16
    _ = mask_b32


@pto.jit(target="a5")
def public_mask_surface_probe():
    remained = pto.const(16, dtype=pto.i32)

    mask8_full = pto.pset_b8(pto.MaskPattern.ALL)
    mask16_full = pto.pset_b16(pto.MaskPattern.ALL)
    mask32_full = pto.pset_b32(pto.MaskPattern.ALL)
    mask8_prefix = pto.pge_b8(pto.MaskPattern.VL32)
    mask16_prefix = pto.pge_b16(pto.MaskPattern.VL16)
    mask32_prefix = pto.pge_b32(pto.MaskPattern.VL8)

    mask8_tail, remained8 = pto.plt_b8(remained)
    mask16_tail, remained16 = pto.plt_b16(remained)
    mask32_tail, remained32 = pto.plt_b32(remained)

    merged = pto.pand(mask32_full, mask32_prefix, mask32_full)
    union = pto.por(mask32_full, mask32_prefix, mask32_full)
    flipped = pto.pxor(mask32_full, mask32_prefix, mask32_full)
    inverted = pto.pnot(mask32_prefix, mask32_full)
    selected = pto.psel(mask32_full, mask32_prefix, mask32_tail)

    packed = pto.ppack(mask32_full, pto.PredicatePart.LOWER)
    unpacked = pto.punpack(packed, pto.PredicatePart.LOWER)
    packed_hi = pto.ppack(mask32_full, pto.PredicatePart.HIGHER)
    unpacked_hi = pto.punpack(packed_hi, pto.PredicatePart.HIGHER)
    packed_hi_b16 = pto.ppack(mask32_full, pto.PredicatePart.HIGHER, to_type=pto.mask_b16)
    unpacked_hi_b32 = pto.punpack(packed_hi_b16, pto.PredicatePart.HIGHER, to_type=pto.mask_b32)
    lo8, hi8 = pto.pintlv_b8(mask8_full, mask8_prefix)
    dlo8, dhi8 = pto.pdintlv_b8(lo8, hi8)
    lo16, hi16 = pto.pintlv_b16(mask16_full, mask16_prefix)
    dlo16, dhi16 = pto.pdintlv_b16(lo16, hi16)
    lo32, hi32 = pto.pintlv_b32(mask32_full, mask32_prefix)
    dlo32, dhi32 = pto.pdintlv_b32(lo32, hi32)

    vec_tile = pto.alloc_tile(shape=[1, 64], dtype=pto.f32, valid_shape=[1, 64])
    scores = pto.vlds(vec_tile.as_ptr(), pto.const(0))
    cmp_eq = pto.vcmp(scores, scores, mask32_full, pto.CmpMode.EQ)
    cmp_gt = pto.vcmps(scores, pto.f32(0.0), mask32_full, pto.CmpMode.GT)

    mask8_buf = pto.alloc_tile(shape=[1, 64], dtype=pto.ui8, valid_shape=[1, 64])
    mask16_buf = pto.alloc_tile(shape=[1, 64], dtype=pto.ui16, valid_shape=[1, 64])
    mask32_buf = pto.alloc_tile(shape=[1, 64], dtype=pto.ui32, valid_shape=[1, 64])
    pto.psts(mask8_full, mask8_buf.as_ptr(), pto.const(0), dist=pto.PredicateDist.NORM)
    pto.psts(mask16_full, mask16_buf.as_ptr(), pto.const(0), dist=pto.PredicateDist.PK)
    loaded8 = pto.plds(mask8_buf.as_ptr(), pto.const(0), dist=pto.PredicateDist.NORM)
    loaded16 = pto.plds(mask16_buf.as_ptr(), pto.const(0), dist=pto.PredicateDist.US)
    loaded32 = pto.plds(mask32_buf.as_ptr(), pto.const(0), dist=pto.PredicateDist.DS)

    align0 = pto.init_align()
    align1, base1 = pto.pstu(align0, mask16_full, mask16_buf.as_ptr())
    align2, base2 = pto.pstu(align1, mask32_full, mask32_buf.as_ptr())

    _ = mask8_tail
    _ = mask16_tail
    _ = mask32_tail
    _ = remained8
    _ = remained16
    _ = remained32
    _ = merged
    _ = union
    _ = flipped
    _ = inverted
    _ = selected
    _ = packed
    _ = unpacked
    _ = packed_hi
    _ = unpacked_hi
    _ = packed_hi_b16
    _ = unpacked_hi_b32
    _ = dlo8
    _ = dhi8
    _ = dlo16
    _ = dhi16
    _ = dlo32
    _ = dhi32
    _ = cmp_eq
    _ = cmp_gt
    _ = loaded8
    _ = loaded16
    _ = loaded32
    _ = align2
    _ = base1
    _ = base2


@pto.jit(target="a5")
def public_sync_surface_probe():
    dynamic_event = pto.const(3)
    pto.get_buf(pto.Pipe.V, 0)
    pto.rls_buf(pto.Pipe.MTE2, 1, 2)
    pto.set_flag(pto.Pipe.MTE2, pto.Pipe.V, event_id=0)
    pto.wait_flag(pto.Pipe.MTE2, pto.Pipe.V, event_id=0)
    pto.set_flag(pto.Pipe.V, pto.Pipe.MTE3, event_id=dynamic_event)
    pto.wait_flag(pto.Pipe.V, pto.Pipe.MTE3, event_id=dynamic_event)
    pto.set_cross_block(pto.Pipe.FIX, 0)
    pto.set_intra_block(pto.Pipe.MTE3, dynamic_event)
    pto.set_intra_block(pto.Pipe.FIX, 4)
    pto.wait_cross_block(pto.Pipe.FIX, 0)
    pto.wait_intra_block(pto.Pipe.V, dynamic_event)
    pto.wait_intra_block(pto.Pipe.FIX, 20)
    pto.wait_intra_block(pto.Pipe.MTE3, dynamic_event)
    pto.wait_intra_block(pto.Pipe.MTE3, 31)


@pto.jit(target="a5")
def public_trap_surface_probe():
    pto.trap()


@pto.jit(target="a5")
def public_dynamic_buf_sync_surface_probe():
    const_buf_id = pto.const(3)
    pto.get_buf(pto.Pipe.V, const_buf_id)
    pto.rls_buf(pto.Pipe.MTE2, const_buf_id, 2)

    with pto.for_(0, 4, step=1) as iter:
        buf_id = iter & 1
        pto.get_buf(pto.Pipe.MTE2, buf_id, 0)
        pto.rls_buf(pto.Pipe.MTE2, buf_id, 0)


@pto.jit(target="a5", ast_rewrite=False)
def explicit_runtime_index_bitwise_event_probe():
    with pto.for_(0, 4, step=1) as i:
        pto.wait_flag(pto.Pipe.V, pto.Pipe.MTE2, event_id=i & 1)
        pto.set_flag(pto.Pipe.MTE2, pto.Pipe.V, event_id=(i | 0) ^ 1)
        pto.wait_flag(pto.Pipe.V, pto.Pipe.MTE2, event_id=1 & i)
        pto.set_flag(pto.Pipe.MTE2, pto.Pipe.V, event_id=0 | i)
        pto.wait_flag(pto.Pipe.V, pto.Pipe.MTE2, event_id=1 ^ i)


@pto.jit(target="a5", ast_rewrite=False)
def explicit_runtime_index_integer_bitwise_event_probe():
    one = pto.const(1, dtype=pto.i32)
    zero = pto.const(0, dtype=pto.i32)
    with pto.for_(0, 4, step=1) as i:
        pto.wait_flag(pto.Pipe.V, pto.Pipe.MTE2, event_id=i & one)
        pto.set_flag(pto.Pipe.MTE2, pto.Pipe.V, event_id=zero | i)
        pto.wait_flag(pto.Pipe.V, pto.Pipe.MTE2, event_id=one ^ i)


@pto.jit(target="a5")
def ast_runtime_index_bitwise_event_probe():
    for i in range(0, 4):
        pto.wait_flag(pto.Pipe.V, pto.Pipe.MTE2, event_id=i & 1)


@pto.jit(target="a5", mode="explicit")
def ui8_pad_surface_probe():
    """Explicit ui8 pointer + ui8 pad value must compile (issue #1245)."""
    zero = pto.const(0, dtype=pto.i64)
    gm_src = pto.castptr(zero, pto.ptr(pto.ui8, "gm"))
    ub_dst = pto.castptr(zero, pto.ptr(pto.ui8, "ub"))
    pad_value = pto.const(0, dtype=pto.ui8)
    pto.mte_gm_ub(gm_src, ub_dst, 0, 3, nburst=(1, 3, 32), pad=(pad_value, 0, 29))


@pto.jit(target="a5", mode="explicit")
def public_data_movement_surface_probe():
    zero_u64 = pto.const(0, dtype=pto.ui64)
    gm_src = pto.castptr(zero_u64, pto.ptr(pto.f16, "gm"))
    gm_dst = pto.castptr(zero_u64, pto.ptr(pto.f16, "gm"))
    ub_src = pto.castptr(zero_u64, pto.ptr(pto.f16, "ub"))
    ub_dst = pto.castptr(zero_u64, pto.ptr(pto.f16, "ub"))
    l1_dst = pto.castptr(zero_u64, pto.ptr(pto.f16, "mat"))
    bias_dst = pto.castptr(zero_u64, pto.ptr(pto.f32, pto.MemorySpace.BIAS))
    scaling_dst = pto.castptr(zero_u64, pto.ptr(pto.f32, pto.MemorySpace.SCALING))
    ub_src_f32 = pto.castptr(zero_u64, pto.ptr(pto.f32, "ub"))
    ub_dst_f32 = pto.castptr(zero_u64, pto.ptr(pto.f32, "ub"))

    pto.mte_gm_ub(gm_src, ub_dst, 0, 256, nburst=(8, 256, 256), loops=[(4, 2048, 2048)])
    pto.mte_gm_ub(gm_src, ub_dst, 0, 200, nburst=(64, 200, 256), pad=(0.0, 0, 0))
    pto.mte_ub_gm(ub_src, gm_dst, 256, nburst=(64, 256, 1024))
    pto.set_store_atomic_cfg(4)
    pto.mte_ub_gm(ub_src, gm_dst, 128, nburst=(1, 128, 128), l2_cache="nared")
    pto.set_store_atomic_cfg(0)
    pto.set_atomic_s32()
    pto.set_atomic_add()
    pto.set_atomic_max()
    pto.set_atomic_min()
    pto.set_atomic_f32()
    pto.set_atomic_f16()
    pto.set_atomic_bf16()
    pto.set_atomic_s16()
    pto.set_atomic_s8()
    pto.set_atomic_none()
    pto.mte_ub_gm(ub_src, gm_dst, 64, nburst=(1, 64, 64), l2_cache="wtsred")
    pto.mte_ub_ub(ub_src, ub_dst, 8, nburst=(16, 0, 4))
    pto.mte_ub_l1(ub_src, l1_dst, 8, nburst=(16, 0, 4))
    pto.mte_gm_l1(gm_src, l1_dst, 256, nburst=(8, 256, 256), loops=[(2, 2048, 2048)])
    pto.raw_fill_l1(
        l1_dst,
        byte_offset=32,
        raw_value=0,
        repeat_times=4,
        block_num_32b=2,
        dst_gap_32b=6,
        fill_word_bits=16,
    )
    pto.mte_l1_ub(l1_dst, ub_dst, 256, nburst=(8, 256, 256), loops=[(2, 2048, 2048)])
    pto.mte_gm_l1_frac(
        gm_src,
        l1_dst,
        pto.FractalMode.ND2NZ,
        shape=(16, 16),
        src_layout=(16, 256),
        dst_group=(1, 0, 0, 0),
        ctrl=(0, False),
    )
    pto.mte_l1_bt(l1_dst, bias_dst, 8, nburst=(1, 0, 0))
    pto.mte_l1_fb(l1_dst, scaling_dst, 8, nburst=(1, 0, 0))

    load_align0 = pto.vldas(ub_src)
    vec0, load_align1 = pto.vldus(ub_src, load_align0)
    low0, high0 = pto.vldsx2(ub_src, pto.const(0), pto.DeinterleaveDist.DINTLV_B16)
    store_align0 = pto.init_align()
    store_align1 = pto.vstur(store_align0, vec0, ub_dst, pto.PostUpdate.OFF)
    pto.vstar(store_align1, ub_dst)

    store_align2 = pto.init_align()
    store_align3 = pto.vstus(store_align2, pto.const(32), vec0, ub_dst)
    pto.vstas(store_align3, ub_dst, pto.const(64))

    vec_f32 = pto.vlds(ub_src_f32, pto.const(0))
    offsets_i16 = pto.vbitcast(vec0, pto.i16)
    offsets_i32 = pto.vbitcast(vec_f32, pto.i32)
    mask16_full = pto.pset_b16(pto.MaskPattern.ALL)
    mask32_full = pto.pset_b32(pto.MaskPattern.ALL)
    gather0 = pto.vgather2(ub_src_f32, offsets_i32, mask32_full)
    gather1 = pto.vgather2_bc(ub_src_f32, offsets_i32, mask32_full)
    gatherb = pto.vgatherb(ub_src_f32, offsets_i32, mask32_full)
    pto.vscatter(gather0, ub_dst_f32, offsets_i32, mask32_full)
    blocked = pto.vsldb(ub_src, pto.i16(32), pto.i16(0), mask32_full)
    pto.vsstb(vec0, ub_dst, pto.i16(32), pto.i16(0), mask32_full)
    pto.vstsx2(low0, high0, ub_dst, pto.const(0), pto.InterleaveDist.INTLV_B16, mask16_full)

    _ = vec0
    _ = vec_f32
    _ = load_align1
    _ = low0
    _ = high0
    _ = gather0
    _ = gather1
    _ = gatherb
    _ = blocked


@pto.jit(target="a5", mode="explicit")
def explicit_mx_scale_staging_surface_probe():
    zero_u64 = pto.const(0, dtype=pto.ui64)
    stage1_u64 = pto.const(32768, dtype=pto.ui64)
    a_scale_l1 = pto.castptr(zero_u64, pto.ptr(pto.f8e8m0, "mat"))
    b_scale_l1 = pto.castptr(zero_u64, pto.ptr(pto.f8e8m0, "mat"))
    a_stage1_l0 = pto.castptr(stage1_u64, pto.ptr(pto.f8e4m3, "left"))
    b_stage1_l0 = pto.castptr(stage1_u64, pto.ptr(pto.f8e4m3, "right"))

    pto.mte_l1_l0a_mx(
        a_scale_l1, a_stage1_l0,
        x_start=3, y_start=5, x_step=16, y_step=2, src_stride=8, dst_stride=2,
    )
    pto.mte_l1_l0b_mx(
        b_scale_l1, b_stage1_l0,
        x_start=3, y_start=5, x_step=16, y_step=2, src_stride=8, dst_stride=2,
    )


@pto.jit(target="a5", mode="explicit")
def explicit_fp4_s4_staging_surface_probe():
    zero_u64 = pto.const(0, dtype=pto.ui64)
    a_l1 = pto.castptr(zero_u64, pto.ptr(pto.f4e1m2x2, "mat"))
    a_l0 = pto.castptr(zero_u64, pto.ptr(pto.f4e1m2x2, "left"))
    b_l1 = pto.castptr(zero_u64, pto.ptr(pto.f4e2m1x2, "mat"))
    b_l0 = pto.castptr(zero_u64, pto.ptr(pto.f4e2m1x2, "right"))

    pto.mte_l1_l0a(
        a_l1, a_l0,
        m_start=0, k_start=4, m_step=16, k_step=4,
        src_stride=16, dst_stride=16,
    )
    pto.mte_l1_l0b(
        b_l1, b_l0,
        m_start=0, k_start=4, m_step=16, k_step=4,
        src_stride=16, dst_stride=16, transpose=True,
    )


@pto.jit(target="a5", mode="explicit")
def fixed_width_integer_specialization_probe():
    zero_u64 = pto.const(0, dtype=pto.ui64)
    gm_src = pto.castptr(zero_u64, pto.ptr(pto.f16, "gm"))
    ub_src = pto.castptr(zero_u64, pto.ptr(pto.f16, "ub"))
    ub_dst = pto.castptr(zero_u64, pto.ptr(pto.f16, "ub"))
    l1_dst = pto.castptr(zero_u64, pto.ptr(pto.f16, "mat"))

    mask32_full = pto.pset_b32(pto.MaskPattern.ALL)
    index_stride = pto.const(32)
    index_zero = pto.const(0)
    blocked = pto.vsldb(ub_src, index_stride, index_zero, mask32_full)
    pto.vsstb(blocked, ub_dst, index_stride, index_zero, mask32_full)

    pto.mte_gm_l1_frac(
        gm_src,
        l1_dst,
        pto.FractalMode.ND2NZ,
        shape=(16, 4),
        src_layout=(16, 256),
        dst_group=(1, 0, 0, 0),
        ctrl=(0, True),
    )
    pto.mte_gm_l1_frac(
        gm_src,
        l1_dst,
        pto.FractalMode.ND2NZ,
        shape=(16, 4),
        src_layout=(16, 256),
        dst_group=(1, 0, 0, 0),
        ctrl=(0, 0),
    )
    pto.mte_gm_l1_frac(
        gm_src,
        l1_dst,
        pto.FractalMode.ND2NZ,
        shape=(16, 4),
        src_layout=(16, 256),
        dst_group=(1, 0, 0, 0),
        ctrl=(0, 1),
    )


@pto.jit(target="a5", mode="explicit")
def public_vector_conversion_surface_probe():
    zero_u64 = pto.const(0, dtype=pto.ui64)
    ub_f32 = pto.castptr(zero_u64, pto.ptr(pto.f32, "ub"))
    ub_i32 = pto.castptr(zero_u64, pto.ptr(pto.i32, "ub"))
    ub_f16 = pto.castptr(zero_u64, pto.ptr(pto.f16, "ub"))

    mask32_full = pto.pset_b32(pto.MaskPattern.ALL)
    vec_f32, ub_f32_next = pto.vlds(ub_f32, pto.const(0), post_update=pto.PostUpdate.ON)
    vec_i32 = pto.vlds(ub_i32, pto.const(0))
    converted = pto.vcvt(
        vec_f32,
        pto.f16,
        mask32_full,
        rnd=pto.VcvtRoundMode.R,
        sat=pto.VcvtSatMode.SAT,
        part=pto.VcvtPartMode.EVEN,
    )
    ub_f16_next = pto.vsts(
        converted,
        ub_f16,
        pto.const(0),
        mask32_full,
        dist=pto.VStoreDist.PK_B32,
        post_update=pto.PostUpdate.ON,
    )
    packed = pto.vpack(vec_i32, pto.VPackPart.LOWER)

    _ = ub_f32_next
    _ = ub_f16_next
    _ = packed


@pto.jit(target="a5", mode="explicit")
def low_precision_vector_memory_surface_probe():
    zero_u64 = pto.const(0, dtype=pto.ui64)
    f8_src = pto.castptr(zero_u64, pto.ptr(pto.f8e4m3, "ub"))
    f8_dst = pto.castptr(zero_u64, pto.ptr(pto.f8e4m3, "ub"))
    hif8_src = pto.castptr(zero_u64, pto.ptr(pto.hif8, "ub"))
    hif8_dst = pto.castptr(zero_u64, pto.ptr(pto.hif8, "ub"))
    mask_b8 = pto.pset_b8(pto.MaskPattern.ALL)
    f8 = pto.vlds(f8_src, pto.const(0))
    hif8 = pto.vlds(hif8_src, pto.const(0))
    pto.vsts(f8, f8_dst, pto.const(0), mask_b8)
    pto.vsts(hif8, hif8_dst, pto.const(0), mask_b8)


@pto.jit(target="a5", mode="explicit")
def low_precision_vcvt_surface_probe():
    zero_u64 = pto.const(0, dtype=pto.ui64)
    ub_f32 = pto.castptr(zero_u64, pto.ptr(pto.f32, "ub"))
    ub_bf16 = pto.castptr(zero_u64, pto.ptr(pto.bf16, "ub"))
    mask_b8 = pto.pset_b8(pto.MaskPattern.ALL)
    mask_b16 = pto.pset_b16(pto.MaskPattern.ALL)
    mask_b32 = pto.pset_b32(pto.MaskPattern.ALL)
    vec_f32 = pto.vlds(ub_f32, pto.const(0))
    vec_bf16 = pto.vlds(ub_bf16, pto.const(0))

    f8e4 = pto.vcvt(
        vec_f32,
        pto.f8e4m3,
        mask_b32,
        rnd=pto.VcvtRoundMode.R,
        sat=pto.VcvtSatMode.NOSAT,
        part=pto.VcvtPartMode.P0,
    )
    _ = pto.vcvt(f8e4, pto.f32, mask_b8, part=pto.VcvtPartMode.P0)
    hif8 = pto.vcvt(
        vec_f32,
        pto.hif8,
        mask_b32,
        rnd=pto.VcvtRoundMode.H,
        sat=pto.VcvtSatMode.NOSAT,
        part=pto.VcvtPartMode.P0,
    )
    _ = pto.vcvt(hif8, pto.f32, mask_b8, part=pto.VcvtPartMode.P0)
    f4e1 = pto.vcvt(
        vec_bf16,
        pto.f4e1m2x2,
        mask_b16,
        rnd=pto.VcvtRoundMode.R,
        part=pto.VcvtPartMode.P0,
    )
    _ = pto.vcvt(f4e1, pto.bf16, mask_b8, part=pto.VcvtPartMode.P0)


@pto.jit(target="a5", mode="explicit")
def low_precision_vadd_invalid_probe():
    zero_u64 = pto.const(0, dtype=pto.ui64)
    f8_src = pto.castptr(zero_u64, pto.ptr(pto.f8e4m3, "ub"))
    mask_b8 = pto.pset_b8(pto.MaskPattern.ALL)
    f8 = pto.vlds(f8_src, pto.const(0))
    _ = pto.vadd(f8, f8, mask_b8)


@pto.jit(target="a5", mode="explicit")
def low_precision_vexp_invalid_probe():
    zero_u64 = pto.const(0, dtype=pto.ui64)
    f8_src = pto.castptr(zero_u64, pto.ptr(pto.f8e4m3, "ub"))
    mask_b8 = pto.pset_b8(pto.MaskPattern.ALL)
    f8 = pto.vlds(f8_src, pto.const(0))
    _ = pto.vexp(f8, mask_b8)


@pto.jit(target="a5", mode="explicit")
def low_precision_vmuls_invalid_probe():
    zero_u64 = pto.const(0, dtype=pto.ui64)
    hif8_src = pto.castptr(zero_u64, pto.ptr(pto.hif8, "ub"))
    mask_b8 = pto.pset_b8(pto.MaskPattern.ALL)
    hif8 = pto.vlds(hif8_src, pto.const(0))
    _ = pto.vmuls(hif8, 1.0, mask_b8)


@pto.jit(target="a5", mode="explicit")
def low_precision_vsel_invalid_probe():
    zero_u64 = pto.const(0, dtype=pto.ui64)
    f8_src = pto.castptr(zero_u64, pto.ptr(pto.f8e4m3, "ub"))
    mask_b8 = pto.pset_b8(pto.MaskPattern.ALL)
    f8 = pto.vlds(f8_src, pto.const(0))
    _ = pto.vsel(f8, f8, mask_b8)


@pto.jit(target="a5", mode="explicit")
def low_precision_vmula_invalid_probe():
    zero_u64 = pto.const(0, dtype=pto.ui64)
    hif8_src = pto.castptr(zero_u64, pto.ptr(pto.hif8, "ub"))
    mask_b8 = pto.pset_b8(pto.MaskPattern.ALL)
    hif8 = pto.vlds(hif8_src, pto.const(0))
    _ = pto.vmula(hif8, hif8, hif8, mask_b8)


@pto.jit(target="a5", mode="explicit")
def low_precision_vmadd_invalid_probe():
    zero_u64 = pto.const(0, dtype=pto.ui64)
    hif8_src = pto.castptr(zero_u64, pto.ptr(pto.hif8, "ub"))
    mask_b8 = pto.pset_b8(pto.MaskPattern.ALL)
    hif8 = pto.vlds(hif8_src, pto.const(0))
    _ = pto.vmadd(hif8, hif8, hif8, mask_b8)


@pto.jit(target="a5", mode="explicit")
def vdup_surface_probe():
    zero_u64 = pto.const(0, dtype=pto.ui64)
    ub_f32 = pto.castptr(zero_u64, pto.ptr(pto.f32, "ub"))
    mask32_full = pto.pset_b32(pto.MaskPattern.ALL)
    vec_f32 = pto.vlds(ub_f32, pto.const(0))
    scalar_dup = pto.vdup(0.0, mask32_full)
    lowest_dup = pto.vdup(vec_f32, mask32_full)
    highest_dup = pto.vdup(vec_f32, mask32_full, pto.PositionMode.HIGHEST)
    _ = scalar_dup
    _ = lowest_dup
    _ = highest_dup


@pto.jit(target="a5", mode="explicit")
def vecscope_surface_probe():
    zero_u64 = pto.const(0, dtype=pto.ui64)
    ub_f32 = pto.castptr(zero_u64, pto.ptr(pto.f32, "ub"))
    zero = pto.const(0)
    with pto.vecscope():
        mask32_full = pto.pset_b32(pto.MaskPattern.ALL)
        vec_f32 = pto.vlds(ub_f32, zero)
        pto.vsts(vec_f32, ub_f32, zero, mask32_full)


@pto.jit(target="a5", mode="explicit")
def vdup_surface_invalid_scalar_position_probe():
    mask32_full = pto.pset_b32(pto.MaskPattern.ALL)
    _ = pto.vdup(pto.f32(0), mask32_full, pto.PositionMode.HIGHEST)


@pto.jit(target="a5", mode="explicit")
def vmulscvt_surface_probe():
    zero_u64 = pto.const(0, dtype=pto.ui64)
    ub_f32 = pto.castptr(zero_u64, pto.ptr(pto.f32, "ub"))
    mask32_full = pto.pset_b32(pto.MaskPattern.ALL)
    vec_f32 = pto.vlds(ub_f32, pto.const(0))
    packed = pto.vmulscvt(
        vec_f32,
        1.0,
        mask32_full,
        rnd=pto.VcvtRoundMode.A,
        part=pto.PartMode.EVEN,
    )
    _ = packed


@pto.jit(target="a5", mode="explicit")
def vmula_surface_probe():
    zero_u64 = pto.const(0, dtype=pto.ui64)
    ub_f32 = pto.castptr(zero_u64, pto.ptr(pto.f32, "ub"))
    mask32_full = pto.pset_b32(pto.MaskPattern.ALL)
    acc = pto.vlds(ub_f32, pto.const(0))
    lhs = pto.vlds(ub_f32, pto.const(0))
    rhs = pto.vlds(ub_f32, pto.const(0))
    _ = pto.vmula(acc, lhs, rhs, mask32_full)


@pto.jit(target="a5", mode="explicit")
def vmadd_surface_probe():
    zero_u64 = pto.const(0, dtype=pto.ui64)
    ub_f32 = pto.castptr(zero_u64, pto.ptr(pto.f32, "ub"))
    mask32_full = pto.pset_b32(pto.MaskPattern.ALL)
    acc = pto.vlds(ub_f32, pto.const(0))
    lhs = pto.vlds(ub_f32, pto.const(0))
    rhs = pto.vlds(ub_f32, pto.const(0))
    _ = pto.vmadd(acc, lhs, rhs, mask32_full)


@pto.jit(target="a5", mode="explicit")
def vcvt_surface_invalid_dtype_pair_probe():
    zero_u64 = pto.const(0, dtype=pto.ui64)
    ub_f32 = pto.castptr(zero_u64, pto.ptr(pto.f32, "ub"))
    mask32_full = pto.pset_b32(pto.MaskPattern.ALL)
    vec_f32 = pto.vlds(ub_f32, pto.const(0))
    _ = pto.vcvt(vec_f32, pto.ui16, mask32_full)


@pto.jit(target="a5", mode="explicit")
def vcvt_low_precision_invalid_dtype_pair_probe():
    zero_u64 = pto.const(0, dtype=pto.ui64)
    ub_f8 = pto.castptr(zero_u64, pto.ptr(pto.f8e4m3, "ub"))
    mask_b8 = pto.pset_b8(pto.MaskPattern.ALL)
    vec_f8 = pto.vlds(ub_f8, pto.const(0))
    _ = pto.vcvt(vec_f8, pto.bf16, mask_b8, part=pto.VcvtPartMode.P0)


@pto.jit(target="a5", mode="explicit")
def vcvt_low_precision_invalid_dtype_pair_probe_2():
    zero_u64 = pto.const(0, dtype=pto.ui64)
    ub_f32 = pto.castptr(zero_u64, pto.ptr(pto.f32, "ub"))
    mask32_full = pto.pset_b32(pto.MaskPattern.ALL)
    vec_f32 = pto.vlds(ub_f32, pto.const(0))
    _ = pto.vcvt(
        vec_f32,
        pto.f4e1m2x2,
        mask32_full,
        rnd=pto.VcvtRoundMode.R,
        part=pto.VcvtPartMode.P0,
    )


@pto.jit(target="a5", mode="explicit")
def vcvt_low_precision_invalid_part_probe():
    zero_u64 = pto.const(0, dtype=pto.ui64)
    ub_f32 = pto.castptr(zero_u64, pto.ptr(pto.f32, "ub"))
    mask32_full = pto.pset_b32(pto.MaskPattern.ALL)
    vec_f32 = pto.vlds(ub_f32, pto.const(0))
    _ = pto.vcvt(
        vec_f32,
        pto.f8e4m3,
        mask32_full,
        rnd=pto.VcvtRoundMode.R,
        sat=pto.VcvtSatMode.NOSAT,
        part=pto.VcvtPartMode.EVEN,
    )


@pto.jit(target="a5", mode="explicit")
def vcvt_low_precision_missing_attr_probe():
    zero_u64 = pto.const(0, dtype=pto.ui64)
    ub_f32 = pto.castptr(zero_u64, pto.ptr(pto.f32, "ub"))
    mask32_full = pto.pset_b32(pto.MaskPattern.ALL)
    vec_f32 = pto.vlds(ub_f32, pto.const(0))
    _ = pto.vcvt(vec_f32, pto.f8e4m3, mask32_full, rnd=pto.VcvtRoundMode.R)


@pto.jit(target="a5", mode="explicit")
def vmulscvt_surface_invalid_dtype_pair_probe():
    zero_u64 = pto.const(0, dtype=pto.ui64)
    ub_i32 = pto.castptr(zero_u64, pto.ptr(pto.i32, "ub"))
    mask32_full = pto.pset_b32(pto.MaskPattern.ALL)
    vec_i32 = pto.vlds(ub_i32, pto.const(0))
    _ = pto.vmulscvt(
        vec_i32,
        1.0,
        mask32_full,
        rnd=pto.VcvtRoundMode.A,
        part=pto.PartMode.EVEN,
    )


@pto.jit(target="a5", mode="explicit")
def vpack_surface_invalid_shape_probe():
    zero_u64 = pto.const(0, dtype=pto.ui64)
    ub_i64 = pto.castptr(zero_u64, pto.ptr(pto.i64, "ub"))
    vec_i64 = pto.vlds(ub_i64, pto.const(0))
    _ = pto.vpack(vec_i64, pto.VPackPart.LOWER)


@pto.jit(target="a5", mode="explicit")
def vpack_surface_invalid_part_probe():
    zero_u64 = pto.const(0, dtype=pto.ui64)
    ub_i32 = pto.castptr(zero_u64, pto.ptr(pto.i32, "ub"))
    vec_i32 = pto.vlds(ub_i32, pto.const(0))
    _ = pto.vpack(vec_i32, "MIDDLE")


@pto.jit(target="a5", mode="explicit")
def vmulscvt_surface_invalid_attr_probe():
    zero_u64 = pto.const(0, dtype=pto.ui64)
    ub_f32 = pto.castptr(zero_u64, pto.ptr(pto.f32, "ub"))
    mask32_full = pto.pset_b32(pto.MaskPattern.ALL)
    vec_f32 = pto.vlds(ub_f32, pto.const(0))
    _ = pto.vmulscvt(
        vec_f32,
        1.0,
        mask32_full,
        rnd=pto.VcvtRoundMode.R,
        part=pto.PartMode.EVEN,
    )


@pto.jit(target="a5", mode="explicit")
def vsstb_post_update_surface_probe():
    zero_u64 = pto.const(0, dtype=pto.ui64)
    ub_f32 = pto.castptr(zero_u64, pto.ptr(pto.f32, "ub"))
    mask32_full = pto.pset_b32(pto.MaskPattern.ALL)
    vec_f32 = pto.vlds(ub_f32, pto.const(0))
    ub_f32_next = pto.vsstb(
        vec_f32,
        ub_f32,
        pto.i16(32),
        pto.i16(0),
        mask32_full,
        post_update=pto.PostUpdate.ON,
    )
    _ = ub_f32_next

@pto.jit(target="a5")
def auto_mode_explicit_surface_violation_probe():
    zero_u64 = pto.const(0, dtype=pto.ui64)
    gm_src = pto.castptr(zero_u64, pto.ptr(pto.f16, "gm"))
    ub_dst = pto.castptr(zero_u64, pto.ptr(pto.f16, "ub"))
    pto.mte_gm_ub(gm_src, ub_dst, 0, 256, nburst=(8, 256, 256))


@pto.jit(target="a5")
def auto_mode_vecscope_violation_probe():
    with pto.vecscope():
        pass


@pto.jit(target="a5")
def dynamic_buf_invalid_type_probe():
    bad = pto.const(1.0, dtype=pto.f32)
    pto.get_buf(pto.Pipe.V, bad)


@pto.jit(target="a5")
def dynamic_rls_buf_invalid_type_probe():
    bad = pto.const(0.0, dtype=pto.f32)
    pto.rls_buf(pto.Pipe.MTE2, bad)


class _FakeTensor:
    def __init__(self, shape):
        self.shape = tuple(shape)

    def new_empty(self, shape):
        return _FakeTensor(shape)


def main() -> None:
    expected_public_exports = [
        "f8e4m3",
        "f8e4m3x2",
        "f8e4m3x4",
        "f8e4m3x8",
        "f8e5m2",
        "f8e5m2x2",
        "f8e5m2x4",
        "f8e5m2x8",
        "hif8",
        "hif8x2",
        "f4e1m2x2",
        "f4e2m1x2",
        "si8",
        "si16",
        "si32",
        "si64",
        "ui8",
        "ui16",
        "ui32",
        "ui64",
        "i32",
        "f16",
        "f32",
        "bytewidth",
        "elements_per_vreg",
        "make_mask",
        "mask_b8",
        "mask_b16",
        "mask_b32",
        "MaskPattern",
        "CmpMode",
        "PredicatePart",
        "PredicateDist",
        "VStoreDist",
        "DeinterleaveDist",
        "InterleaveDist",
        "PostUpdate",
        "PartMode",
        "PositionMode",
        "VPackPart",
        "VcvtRoundMode",
        "VcvtSatMode",
        "VcvtPartMode",
        "RoundMode",
        "AlignType",
        "init_align",
        "plt_b8",
        "plt_b16",
        "pset_b8",
        "pset_b16",
        "pge_b8",
        "pge_b16",
        "pge_b32",
        "pand",
        "por",
        "pxor",
        "pnot",
        "psel",
        "pbitcast",
        "vcvt",
        "vpack",
        "vmulscvt",
        "ppack",
        "punpack",
        "pintlv_b8",
        "pintlv_b16",
        "pintlv_b32",
        "pdintlv_b8",
        "pdintlv_b16",
        "pdintlv_b32",
        "vcmp",
        "vcmps",
        "plds",
        "psts",
        "pstu",
        "vbitcast",
        "vexp",
        "vcgmax",
        "vcgadd",
        "vsubs",
        "mte_load",
        "mte_store",
        "mem_bar",
        "BarrierType",
        "Pipe",
        "pipe_barrier",
        "get_buf",
        "rls_buf",
        "set_cross_block",
        "wait_cross_block",
        "set_intra_block",
        "wait_intra_block",
        "mte_gm_ub",
        "mte_ub_gm",
        "mte_ub_ub",
        "mte_ub_l1",
        "mte_gm_l1",
        "raw_fill_l1",
        "mte_l1_ub",
        "mte_gm_l1_frac",
        "mte_l1_bt",
        "mte_l1_fb",
        "vldsx2",
        "vldas",
        "vldus",
        "vstsx2",
        "vgather2",
        "vgather2_bc",
        "vgatherb",
        "vscatter",
        "vsldb",
        "vsstb",
        "vstar",
        "vstas",
        "vstur",
        "vstus",
        "mte_l1_l0a",
        "mte_l1_l0b",
        "mte_l1_l0a_mx",
        "mte_l1_l0b_mx",
        "mte_l0c_l1",
        "mte_l0c_gm",
        "mte_l0c_ub",
        "mad",
        "mad_acc",
        "mad_bias",
        "mad_mx",
        "mad_mx_acc",
        "mad_mx_bias",
        "empty_like",
    ]
    for name in expected_public_exports:
        expect(hasattr(pto, name), f"pto.{name} should be exported from the public namespace")

    fake_tensor = _FakeTensor((2, 3, 4))
    fake_empty = pto.empty_like(fake_tensor)
    expect(isinstance(fake_empty, _FakeTensor), "pto.empty_like(...) should preserve host tensor factory type")
    expect(fake_empty.shape == fake_tensor.shape, "pto.empty_like(...) should preserve the logical tensor shape")
    expect(not hasattr(pto, "scalar"), "pto.scalar should not remain in the public pto namespace")
    expect(
        not hasattr(pto, "load_cbuf_to_ca"),
        "pto.load_cbuf_to_ca should not be exported; use pto.mte_l1_l0a(...) explicit control",
    )
    expect(
        not hasattr(pto, "load_cbuf_to_cb"),
        "pto.load_cbuf_to_cb should not be exported; use pto.mte_l1_l0b(...) explicit control",
    )
    expect(hasattr(pto, "tile"), "pto.tile should be exported from the public namespace")
    expect(hasattr(pto, "vmi"), "pto.vmi should be exported from the public namespace")
    expect(hasattr(pto.tile, "load"), "pto.tile.load should be exported from the public tile namespace")
    expect(hasattr(pto.tile, "add"), "pto.tile.add should be exported from the public tile namespace")
    expect(hasattr(pto.tile, "cmps"), "pto.tile.cmps should be exported from the public tile namespace")
    expect(hasattr(pto.vmi, "vreg"), "pto.vmi.vreg should be exported from the public VMI namespace")
    expect(hasattr(pto.vmi, "mask"), "pto.vmi.mask should be exported from the public VMI namespace")
    expect(hasattr(pto.vmi, "vadd"), "pto.vmi.vadd should be exported from the public VMI namespace")
    expect(hasattr(pto.vmi, "create_mask"), "pto.vmi.create_mask should be exported from the public VMI namespace")
    expect(not hasattr(pto, "load_tile"), "pto.load_tile should not remain on the public pto namespace")
    expect(not hasattr(pto, "store_tile"), "pto.store_tile should not remain on the public pto namespace")
    expect(hasattr(pto.tile, "matmul"), "pto.tile.matmul should be exported from the public tile namespace")
    expect(hasattr(pto.tile, "matmul_acc"), "pto.tile.matmul_acc should be exported from the public tile namespace")
    expect(hasattr(pto.tile, "matmul_mx"), "pto.tile.matmul_mx should be exported from the public tile namespace")
    expect(hasattr(pto.tile, "matmul_mx_acc"), "pto.tile.matmul_mx_acc should be exported from the public tile namespace")
    expect(hasattr(pto.tile, "matmul_mx_bias"), "pto.tile.matmul_mx_bias should be exported from the public tile namespace")
    expect(hasattr(pto.tile, "gemv_mx"), "pto.tile.gemv_mx should be exported from the public tile namespace")
    expect(hasattr(pto.tile, "gemv_mx_acc"), "pto.tile.gemv_mx_acc should be exported from the public tile namespace")
    expect(hasattr(pto.tile, "gemv_mx_bias"), "pto.tile.gemv_mx_bias should be exported from the public tile namespace")
    expect(not hasattr(pto, "tload"), "legacy pto.tload should not remain on the public pto namespace")
    expect(not hasattr(pto, "tstore"), "legacy pto.tstore should not remain on the public pto namespace")
    expect(not hasattr(pto, "tadd"), "legacy pto.tadd should not remain on the public pto namespace")
    expect(not hasattr(pto, "get_buf_dyn"), "pto.get_buf_dyn should not remain on the public pto namespace")
    expect(not hasattr(pto, "rls_buf_dyn"), "pto.rls_buf_dyn should not remain on the public pto namespace")
    expect(not hasattr(pto, "tile_buf_type"), "pto.tile_buf_type should not remain on the public pto namespace")
    expect(hasattr(pto, "vecscope"), "pto.vecscope should be exported from the public pto namespace")
    expect(not hasattr(pto, "as_ptr"), "pto.as_ptr should not remain on the public pto namespace")
    expect(not hasattr(pto, "vbrc_load"), "pto.vbrc_load should not remain on the public pto namespace")
    expect(not hasattr(pto, "vsts_1pt"), "pto.vsts_1pt should not remain on the public pto namespace")
    expect(not hasattr(pto, "constexpr"), "pto.const_expr should not remain on the public pto namespace")
    expect(not hasattr(pto, "copy_ubuf_to_ubuf"), "pto.copy_ubuf_to_ubuf should not remain on the public pto namespace")
    expect(not hasattr(pto, "vmi_vreg_type"), "pto.vmi_vreg_type should not remain on the public pto namespace")
    expect(not hasattr(pto, "vmi_mask_type"), "pto.vmi_mask_type should not remain on the public pto namespace")
    expect(not hasattr(scalar, "sts"), "scalar.sts should not remain in the public scalar namespace")
    expect(not hasattr(scalar, "cmpi"), "scalar.cmpi should not remain in the public scalar namespace")
    expect(not hasattr(scalar, "cmpi_sgt"), "scalar.cmpi_sgt should not remain in the public scalar namespace")
    removed_tile_buf_type = expect_raises(AttributeError, lambda: getattr(pto, "tile_buf_type"))
    expect(
        "pto.tile_buf_type is not a supported PTODSL public interface" in str(removed_tile_buf_type),
        "removed pto.tile_buf_type should diagnose the authored alloc_tile replacement",
    )
    expect(
        callable(pto.vecscope),
        "pto.vecscope should expose the public vecscope context manager",
    )
    removed_as_ptr = expect_raises(AttributeError, lambda: getattr(pto, "as_ptr"))
    expect(
        "pto.as_ptr is not a supported PTODSL public interface" in str(removed_as_ptr),
        "removed pto.as_ptr should diagnose the authored object-method replacements",
    )
    removed_vbrc_load = expect_raises(AttributeError, lambda: getattr(pto, "vbrc_load"))
    expect(
        "pto.vbrc_load is not a supported PTODSL public interface" in str(removed_vbrc_load),
        "removed pto.vbrc_load should diagnose the public vlds(dist=...) replacement",
    )
    removed_vsts_1pt = expect_raises(AttributeError, lambda: getattr(pto, "vsts_1pt"))
    expect(
        "pto.vsts_1pt is not a supported PTODSL public interface" in str(removed_vsts_1pt),
        "removed pto.vsts_1pt should diagnose the public vsts(dist=...) replacement",
    )
    removed_constexpr = expect_raises(AttributeError, lambda: getattr(pto, "constexpr"))
    expect(
        "pto.constexpr is not a supported PTODSL public interface" in str(removed_constexpr)
        and "Use pto.const_expr" in str(removed_constexpr),
        "removed pto.constexpr should diagnose pto.const_expr as the replacement",
    )
    removed_copy_ubuf_to_ubuf = expect_raises(AttributeError, lambda: getattr(pto, "copy_ubuf_to_ubuf"))
    expect(
        "pto.copy_ubuf_to_ubuf is not a supported PTODSL public interface" in str(removed_copy_ubuf_to_ubuf)
        and "Use pto.mte_ub_ub" in str(removed_copy_ubuf_to_ubuf),
        "removed pto.copy_ubuf_to_ubuf should diagnose pto.mte_ub_ub as the replacement",
    )
    for name in ("max", "min", "exp", "log", "sqrt", "abs"):
        expect(hasattr(scalar, name), f"scalar.{name} should be exported from the public scalar namespace")

    with make_context() as ctx, Location.unknown(ctx):
        tile_buf_ty = pto_types.tile_buf_type(
            [16, 32],
            pto.f32,
            [16, 8],
            address_space="mat",
            blayout="ColMajor",
            slayout="RowMajor",
        )
        expect(hasattr(tile_buf_ty, "memory_space"), "TileBufType should expose a memory_space accessor")
        expect(hasattr(tile_buf_ty, "shape"), "TileBufType should expose a shape accessor")
        expect(hasattr(tile_buf_ty, "valid_shape"), "TileBufType should expose a valid_shape accessor")
        expect(hasattr(tile_buf_ty, "element_type"), "TileBufType should expose an element_type accessor")
        expect(tile_buf_ty.memory_space.value == pto.MemorySpace.MAT.value, "TileBufType.memory_space should preserve the authored address space")
        expect(list(tile_buf_ty.shape) == [16, 32], "TileBufType.shape should preserve the authored physical shape")
        expect(list(tile_buf_ty.valid_shape) == [16, 8], "TileBufType.valid_shape should preserve the authored valid shape")
        expect(str(tile_buf_ty.element_type) == "f32", "TileBufType.element_type should preserve the authored element type")

    host_vec_copy.verify()
    runtime_metadata_kernel.verify()
    authored_addr_tile_surface_probe.verify()
    dynamic_addr_tile_surface_probe.verify()
    tile_surface_compute_probe.verify()
    shared_subkernel_lowering_probe.verify()
    simt_helper_lowering_probe.verify()
    carry_loop_lowering_probe.verify()
    branch_handle_then_only_probe.verify()
    branch_handle_side_effect_probe.verify()
    branch_handle_merge_probe.verify()
    runtime_scalar_operator_probe.verify()
    tile_slice_surface_probe.verify()
    tile_slice_1d_surface_probe.verify()
    tile_valid_shape_update_probe.verify()
    tile_valid_shape_update_1d_probe.verify()
    make_mask_index_roundtrip_probe.verify()
    fixed_integer_index_coercion_probe.verify()
    integer_loop_bound_probe.verify()
    scalar_pointer_offset_probe.verify()
    scalar_contiguous_vector_probe.verify()
    scalar_contiguous_vector_arith_probe.verify()
    per_element_vec_f32_probe.verify()
    per_element_vec_i32_probe.verify()
    per_element_vec_ui32_probe.verify()
    addptr_surface_probe.verify()
    simt_pointer_offset_probe.verify()
    scalar_store_element_coercion_probe.verify()
    shared_index_coercion_probe.verify()
    public_surface_exports_probe.verify()
    compile_time_query_probe.verify()
    eager_scalar_constructor_probe.verify()
    signed_integer_scalar_probe.verify()
    low_precision_storage_probe.verify()
    pointer_vlds_inference_probe.verify()
    public_mask_bitcast_probe.verify()
    public_mask_surface_probe.verify()
    public_sync_surface_probe.verify()
    public_trap_surface_probe.verify()
    explicit_runtime_index_bitwise_event_probe.verify()
    explicit_runtime_index_integer_bitwise_event_probe.verify()
    ast_runtime_index_bitwise_event_probe.verify()
    public_data_movement_surface_probe.verify()
    ui8_pad_surface_probe.verify()
    fixed_width_integer_specialization_probe.verify()
    public_vector_conversion_surface_probe.verify()
    vdup_surface_probe.verify()
    vmulscvt_surface_probe.verify()
    vmula_surface_probe.verify()
    vmadd_surface_probe.verify()
    vsstb_post_update_surface_probe.verify()

    with make_context() as ctx, Location.unknown(ctx):
        expect(
            pto.MaskPattern.ALL == "PAT_ALL",
            "pto.MaskPattern.ALL should expose the documented PAT_ALL token",
        )
        expect(
            pto.MaskPattern.VL16 == "PAT_VL16",
            "pto.MaskPattern.VL16 should expose the documented PAT_VL16 token",
        )
        expect(
            pto.CmpMode.GT == "gt",
            "pto.CmpMode.GT should lower through the documented compare-mode surface",
        )
        expect(
            pto.PredicatePart.LOWER == "LOWER",
            "pto.PredicatePart.LOWER should expose the documented pack/unpack token",
        )
        expect(
            pto.PredicateDist.PK == "PK",
            "pto.PredicateDist.PK should expose the documented predicate-store distribution token",
        )
        expect(
            str(pto.si8.resolve()) == "si8",
            "pto.si8 should resolve to a signed 8-bit integer type",
        )
        expect(
            str(pto.ui8.resolve()) == "ui8",
            "pto.ui8 should resolve to an unsigned 8-bit integer type",
        )
        bit_pattern_module = Module.create()
        with InsertionPoint(bit_pattern_module.body):
            _ = pto.si32("0xFFFFFFFF")
            _ = pto.ui32("0xFFFFFFFF")
            _ = pto.i32("0x80000000")
        bit_pattern_text = str(bit_pattern_module)
        expect(
            "unrealized_conversion_cast" in bit_pattern_text,
            "signed/unsigned integer bit-pattern constructors should bridge through unrealized_conversion_cast",
        )
        expect(
            "arith.constant -1 : i32" in bit_pattern_text,
            "pto.si32/ui32 bit-pattern initialization should materialize the expected signless constant payload",
        )
        expect(
            "arith.constant -2147483648 : i32" in bit_pattern_text,
            "pto.i32 bit-pattern initialization should materialize the documented signless bit pattern",
        )
        expect(
            str(pto.f8e4m3.resolve()) == "f8E4M3FN",
            "pto.f8e4m3 should resolve to the public E4M3 float8 type",
        )
        expect(
            str(pto.f8e4m3x4.resolve()) == "vector<4xf8E4M3FN>",
            "pto.f8e4m3x4 should resolve to vector<4xf8E4M3FN>",
        )
        expect(
            str(pto.f8e4m3x8.resolve()) == "vector<8xf8E4M3FN>",
            "pto.f8e4m3x8 should resolve to vector<8xf8E4M3FN>",
        )
        expect(
            str(pto.f8e5m2x4.resolve()) == "vector<4xf8E5M2>",
            "pto.f8e5m2x4 should resolve to vector<4xf8E5M2>",
        )
        expect(
            str(pto.f8e5m2x8.resolve()) == "vector<8xf8E5M2>",
            "pto.f8e5m2x8 should resolve to vector<8xf8E5M2>",
        )
        expect(
            str(pto.f8e8m0.resolve()) == "!pto.f8E8M0",
            "pto.f8e8m0 should resolve to the public E8M0 scale type",
        )
        expect(
            str(pto.bf16x2.resolve()) == "vector<2xbf16>",
            "pto.bf16x2 should remain the builtin two-lane bf16 vector type",
        )
        expect(
            str(pto.vmi.bf16x2.resolve()) == "!pto.bf16x2",
            "pto.vmi.bf16x2 should resolve to the VMI packed bf16 pair carrier",
        )
        expect(
            hasattr(pto_types._pto, "BF16x2Type"),
            "PTO Python dialect bindings should export BF16x2Type",
        )
        expect(
            "hif8" in str(pto.hif8.resolve()),
            "pto.hif8 should resolve to the public HiF8 type",
        )
        expect(
            "f4E1M2x2" in str(pto.f4e1m2x2.resolve()),
            "pto.f4e1m2x2 should resolve to the packed 4-bit float type",
        )
        expect(
            str(pto.mask_b8.resolve()) == "!pto.mask<b8>",
            "pto.mask_b8 should resolve to the public 8-bit mask type",
        )
        expect(
            str(pto.mask_b16.resolve()) == "!pto.mask<b16>",
            "pto.mask_b16 should resolve to the public 16-bit mask type",
        )
        expect(
            str(pto.mask_b32.resolve()) == "!pto.mask<b32>",
            "pto.mask_b32 should resolve to the public 32-bit mask type",
        )
        expect(
            str(pto.vmi.vreg(128, pto.f32).resolve()) == "!pto.vmi.vreg<128xf32>",
            "pto.vmi.vreg(...) should resolve to the public logical VMI vector type",
        )
        expect(
            str(pto.vmi.mask(128).resolve()) == "!pto.vmi.mask<128xpred>",
            "pto.vmi.mask(...) should resolve to the public logical VMI mask type",
        )
        expect(
            str(pto.vmi.vreg(128, pto.f8e4m3).resolve()) == "!pto.vmi.vreg<128xf8E4M3FN>",
            "pto.vmi.vreg(...) should preserve low-precision authored element types",
        )

        lp_tile_ty = pto_types.tile_buf_type([16, 16], pto.hif8, [16, 16])
        lp_tv_ty = pto_types.tensor_view_type(2, pto.f8e4m3)
        lp_part_ty = pto_types.part_tensor_view_type(2, pto.f4e2m1x2)
        expect(
            "hif8" in str(lp_tile_ty.element_type),
            "low-precision tile buffers should preserve their authored element type",
        )
        expect(
            str(lp_tv_ty.element_type) == "f8E4M3FN",
            "internal tensor-view type helper should preserve low-precision element types",
        )
        expect(
            "f4E2M1x2" in str(lp_part_ty.element_type),
            "internal partition tensor-view type helper should preserve low-precision element types",
        )

        expect(
            "!pto.ptr<!pto.hif8, ub>" == str(pto.ptr(pto.hif8).resolve()),
            "low-precision pointer types should be valid for device storage",
        )
        expect(
            "!pto.ptr<!pto.f8E8M0, l1>" == str(pto.ptr(pto.f8e8m0, "mat").resolve()),
            "f8e8m0 pointers should be valid for L1 scale storage",
        )
        expect(
            str(pto.vreg_type(256, pto.f8e4m3).resolve()) == "!pto.vreg<256xf8E4M3FN>",
            "low-precision vreg types should be valid for vector micro-ops",
        )
        expect_raises(
            TypeError,
            lambda: pto.hif8(1.0),
            "unsupported eager constructor target type",
        )
        expect_raises(
            TypeError,
            lambda: pto.f8e4m3(1.0),
            "unsupported eager constructor target type",
        )
        expect_raises(
            TypeError,
            lambda: pto.f8e8m0(1.0),
            "unsupported eager constructor target type",
        )

    expect_raises(
        AttributeError,
        lambda: pto.tensor_spec,
        "pto.tensor_spec is not a supported PTODSL public interface",
    )
    expect_raises(
        AttributeError,
        lambda: pto.TensorSpec,
        "pto.TensorSpec is not a supported PTODSL public interface",
    )
    expect(
        not hasattr(pto, "tensor_view_type"),
        "pto.tensor_view_type should remain an internal helper and not be exported on the public namespace",
    )
    expect(
        not hasattr(pto, "part_tensor_view_type"),
        "pto.part_tensor_view_type should remain an internal helper and not be exported on the public namespace",
    )

    default_compiled = host_vec_copy.compile()
    explicit_default = host_vec_copy.compile(BLOCK=128)
    block64 = host_vec_copy.compile(BLOCK=64)

    scalar_pipeline_text = scalar_pipeline_access_modes_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(
        scalar_pipeline_text,
        "scalar pipeline access mode specialization",
    )
    expect(
        scalar_pipeline_text.count("pto.load_scalar") == 1
        and scalar_pipeline_text.count("pto.store_scalar") == 1,
        "default scalar pipeline accesses should remain load_scalar/store_scalar",
    )
    expect(
        scalar_pipeline_text.count("pto.ld_dev") == 1
        and scalar_pipeline_text.count("pto.st_dev") == 1,
        "bypass_l1 scalar pipeline accesses should lower to ld_dev/st_dev",
    )
    expect_raises(
        TypeError,
        lambda: scalar_pipeline_bypass_ub_invalid_probe.compile().mlir_text(),
        "requires a GM pointer",
    )
    expect_raises(
        TypeError,
        lambda: scalar_pipeline_bypass_float_invalid_probe.compile().mlir_text(),
        "supports only i8, i16, i32 or i64",
    )

    expect(default_compiled is explicit_default, "default constexpr compile should hit specialization cache")
    expect(default_compiled is not block64, "different constexpr values should materialize different specializations")
    expect(len(host_vec_copy.cached_specializations()) == 2, "expected exactly two cached specializations")
    expect(default_compiled.constexpr_bindings == {"BLOCK": 128}, "default constexpr binding mismatch")
    expect(block64.constexpr_bindings == {"BLOCK": 64}, "BLOCK=64 constexpr binding mismatch")
    expect_raises(
        TypeError,
        lambda: host_vec_copy.compile(BLOCK=[128]),
        "@pto.jit constexpr parameter 'BLOCK' must be hashable",
    )
    expect(
        default_compiled.specialization_key.abi_signature == block64.specialization_key.abi_signature,
        "ABI signature should stay stable across constexpr-only specializations",
    )
    expect(
        default_compiled.specialization_key.constexpr_signature
        != block64.specialization_key.constexpr_signature,
        "constexpr specialization key should differ when BLOCK changes",
    )
    pointer_default = pointer_runtime_shape_specialization_probe.compile()
    pointer_explicit_default = pointer_runtime_shape_specialization_probe.compile(BLOCK=128)
    pointer_block64 = pointer_runtime_shape_specialization_probe.compile(BLOCK=64)
    expect(
        pointer_default is pointer_explicit_default,
        "pointer-first kernels should reuse the same specialization when only runtime launch values vary",
    )
    expect(
        pointer_default is not pointer_block64,
        "pointer-first kernels should still specialize on constexpr values",
    )
    expect(
        pointer_default.specialization_key.abi_signature == pointer_block64.specialization_key.abi_signature,
        "pointer-first runtime shape scalars should stay in the ABI signature, not the constexpr signature",
    )
    expect(
        pointer_default.specialization_key.constexpr_signature == (("BLOCK", 128),),
        "pointer-first specialization key should only capture constexpr bindings",
    )
    expect(
        pointer_block64.specialization_key.constexpr_signature == (("BLOCK", 64),),
        "pointer-first specialization key should change only with constexpr bindings",
    )
    source_native_build_compiled = None
    source_explicit_native_build_compiled = None
    source_no_insert_sync_native_build_compiled = None
    with TemporaryDirectory() as tmpdir:
        source_path = Path(tmpdir) / "source_kernel.pto"
        source_text_v1 = (
            "module {\n"
            "  func.func @selected_source_entry(%arg0: !pto.ptr<f32, gm>, %arg1: i32) {\n"
            "    return\n"
            "  }\n"
            "  func.func @other_source_entry(%arg0: !pto.ptr<f32, gm>) {\n"
            "    return\n"
            "  }\n"
            "}\n"
        )
        source_text_v2 = (
            "module {\n"
            "  func.func @selected_source_entry(%arg0: !pto.ptr<f32, gm>, %arg1: i32) {\n"
            "    %c0 = arith.constant 0 : i32\n"
            "    return\n"
            "  }\n"
            "  func.func @other_source_entry(%arg0: !pto.ptr<f32, gm>) {\n"
            "    return\n"
            "  }\n"
            "}\n"
        )
        source_path.write_text(source_text_v1, encoding="utf-8")

        @pto.jit(name="selected_source_entry", target="a5", source=str(source_path))
        def source_backed_probe(ptr: pto.ptr(pto.f32, "gm"), rows: pto.i32):
            raise RuntimeError("source-backed JIT should not trace the Python body")

        source_compiled_v1 = source_backed_probe.compile()
        source_compiled_v1_again = source_backed_probe.compile()
        expect(source_compiled_v1 is source_compiled_v1_again, "unchanged source-backed JIT should hit specialization cache")
        expect(
            source_compiled_v1.mlir_text() == source_text_v1,
            "source-backed JIT mlir_text() should preserve the authored source text",
        )
        expect(
            source_compiled_v1.ir_function_name == "selected_source_entry",
            "source-backed JIT should use name= for entry selection and launch wrapper naming",
        )
        expect(
            source_compiled_v1.build_metadata()["source_path"] == str(source_path.resolve()),
            "source-backed JIT metadata should expose the resolved source path",
        )

        source_path.write_text(source_text_v2, encoding="utf-8")
        source_compiled_v2 = source_backed_probe.compile()
        expect(
            source_compiled_v2 is not source_compiled_v1,
            "editing the source file should materialize a new specialization",
        )
        expect(
            source_compiled_v2.specialization_key != source_compiled_v1.specialization_key,
            "source-backed specialization key should include source content",
        )
        expect(
            source_compiled_v2.mlir_text() == source_text_v2,
            "source-backed JIT should reload changed source text",
        )
        source_auto_path = Path(tmpdir) / "source_auto.pto"
        source_auto_text = (
            "module {\n"
            "  func.func @source_auto_native(%arg0: !pto.ptr<f32, gm>) {\n"
            "    return\n"
            "  }\n"
            "}\n"
        )
        source_auto_path.write_text(source_auto_text, encoding="utf-8")

        @pto.jit(name="source_auto_native", target="a5", backend="vpto", source=str(source_auto_path))
        def source_auto_native(ptr: pto.ptr(pto.f32, "gm")):
            raise RuntimeError("source-backed JIT should not trace the Python body")

        source_native_build_compiled = source_auto_native.compile()

        source_explicit_path = Path(tmpdir) / "source_explicit.pto"
        source_explicit_text = (
            "module {\n"
            "  func.func @source_explicit_native(%arg0: !pto.ptr<f32, gm>) {\n"
            "    return\n"
            "  }\n"
            "}\n"
        )
        source_explicit_path.write_text(source_explicit_text, encoding="utf-8")

        @pto.jit(
            name="source_explicit_native",
            target="a5",
            backend="vpto",
            mode="explicit",
            source=str(source_explicit_path),
        )
        def source_explicit_native(ptr: pto.ptr(pto.f32, "gm")):
            raise RuntimeError("source-backed JIT should not trace the Python body")

        source_explicit_native_build_compiled = source_explicit_native.compile()

        source_no_insert_sync_path = Path(tmpdir) / "source_no_insert_sync.pto"
        source_no_insert_sync_text = (
            "module {\n"
            "  func.func @source_no_insert_sync_native(%arg0: !pto.ptr<f32, gm>) {\n"
            "    return\n"
            "  }\n"
            "}\n"
        )
        source_no_insert_sync_path.write_text(source_no_insert_sync_text, encoding="utf-8")

        @pto.jit(
            name="source_no_insert_sync_native",
            target="a5",
            backend="vpto",
            insert_sync=False,
            source=str(source_no_insert_sync_path),
        )
        def source_no_insert_sync_native(ptr: pto.ptr(pto.f32, "gm")):
            raise RuntimeError("source-backed JIT should not trace the Python body")

        source_no_insert_sync_native_build_compiled = source_no_insert_sync_native.compile()

        relative_case_dir = Path(tmpdir) / "relative_case"
        relative_case_dir.mkdir()
        relative_source_path = relative_case_dir / "relative_kernel.pto"
        relative_source_text = (
            "module {\n"
            "  func.func @relative_source_entry(%arg0: !pto.ptr<f32, gm>) {\n"
            "    return\n"
            "  }\n"
            "}\n"
        )
        relative_source_path.write_text(relative_source_text, encoding="utf-8")
        relative_module_path = relative_case_dir / "relative_case.py"
        relative_module_path.write_text(
            "from ptodsl import pto\n"
            "@pto.jit(name='relative_source_entry', target='a5', source='relative_kernel.pto')\n"
            "def relative_kernel(ptr: pto.ptr(pto.f32, 'gm')):\n"
            "    raise RuntimeError('source-backed JIT should not trace the Python body')\n",
            encoding="utf-8",
        )
        spec = spec_from_file_location("ptodsl_relative_source_case", relative_module_path)
        expect(spec is not None and spec.loader is not None, "relative source test module should be importable")
        relative_module = module_from_spec(spec)
        spec.loader.exec_module(relative_module)
        relative_compiled = relative_module.relative_kernel.compile()
        expect(
            relative_compiled.mlir_text() == relative_source_text,
            "relative source paths should resolve against the declaring Python file",
        )
        expect(
            relative_compiled.build_metadata()["source_path"] == str(relative_source_path.resolve()),
            "relative source-backed metadata should expose the resolved source path",
        )

        inline_source_text = (
            "module {\n"
            "  func.func @inline_source_entry(%arg0: !pto.ptr<f32, gm>, %arg1: i32) {\n"
            "    return\n"
            "  }\n"
            "}\n"
        )

        @pto.jit(name="inline_source_entry", target="a5", source=inline_source_text)
        def inline_source_backed_probe(ptr: pto.ptr(pto.f32, "gm"), rows: pto.i32):
            raise RuntimeError("source-backed JIT should not trace the Python body")

        inline_compiled = inline_source_backed_probe.compile()
        expect(
            inline_compiled.mlir_text() == inline_source_text,
            "inline source-backed JIT mlir_text() should preserve the authored source text",
        )
        inline_metadata = inline_compiled.build_metadata()
        expect(
            inline_metadata["source_kind"] == "inline",
            "inline source-backed JIT metadata should mark the source kind as inline",
        )
        expect(
            "source_path" not in inline_metadata,
            "inline source-backed JIT metadata should not synthesize a filesystem path",
        )
    pointer_artifacts_default = artifact_paths(
        pointer_default._py_name,
        pointer_default.ir_function_name,
        pointer_default.specialization_key,
    )
    pointer_artifacts_explicit_default = artifact_paths(
        pointer_explicit_default._py_name,
        pointer_explicit_default.ir_function_name,
        pointer_explicit_default.specialization_key,
    )
    pointer_artifacts_block64 = artifact_paths(
        pointer_block64._py_name,
        pointer_block64.ir_function_name,
        pointer_block64.specialization_key,
    )
    expect(
        pointer_artifacts_default.cache_dir == pointer_artifacts_explicit_default.cache_dir,
        "native artifact paths should stay stable for the same pointer-first specialization",
    )
    expect(
        pointer_artifacts_default.cache_dir != pointer_artifacts_block64.cache_dir,
        "native artifact paths should only change when constexpr specializations change",
    )
    marshaled_small = _marshal_launch_args(pointer_default._kernel_signature, (0x1000, 16, 32, 32))
    marshaled_padded = _marshal_launch_args(pointer_default._kernel_signature, (0x1000, 7, 29, 64))
    expect(len(marshaled_small) == 4, "pointer-first launch ABI should marshal one pointer plus rows/cols/stride scalars")
    expect(len(marshaled_padded) == 4, "pointer-first launch ABI width should stay constant across dynamic shape launches")
    expect(
        marshaled_small[1].value == 16 and marshaled_small[2].value == 32 and marshaled_small[3].value == 32,
        "contiguous runtime-shape launch should preserve rows/cols/stride values",
    )
    expect(
        marshaled_padded[1].value == 7 and marshaled_padded[2].value == 29 and marshaled_padded[3].value == 64,
        "same compiled kernel should accept padded runtime shape/stride values without changing specialization",
    )
    launch_cpp = generate_launch_cpp(
        ir_function_name=pointer_default.ir_function_name,
        kernel_signature=pointer_default._kernel_signature,
    )
    expect(
        "extern \"C\" void ptodsl_launch_" in launch_cpp and "int32_t rows" in launch_cpp,
        "launch wrapper codegen should resolve pointer-first runtime scalar annotations without requiring an ambient MLIR Context",
    )

    default_text = default_compiled.mlir_text()
    block64_text = block64.mlir_text()
    explicit_text = host_vec_copy_explicit.compile().mlir_text()
    expect_parse_roundtrip_and_verify(default_text, "default host_vec_copy specialization")
    expect_parse_roundtrip_and_verify(block64_text, "BLOCK=64 host_vec_copy specialization")
    expect_parse_roundtrip_and_verify(explicit_text, "explicit host_vec_copy specialization")
    expect("!pto.tile_buf<vec, 1x128xf32>" in default_text, "default specialization MLIR missing BLOCK=128 tile")
    expect("!pto.tile_buf<vec, 1x64xf32>" in block64_text, "BLOCK=64 specialization MLIR missing specialized tile")
    expect("pto.entry" in default_text, "default @pto.jit entry child should carry the explicit entry marker")
    expect("pto.entry" in explicit_text, "explicit @pto.jit entry child should carry the explicit entry marker")
    expect(default_text.count("module") == 2, "default @pto.jit should wrap an unspecified-kind kernel in a backend child module")
    expect(block64_text.count("module") == 2, "specialized @pto.jit should keep the backend child module shape")
    expect(
        'module attributes {pto.backend = "vpto", pto.target_arch = "a5"}' in default_text,
        "unpartitioned VPTO module should carry backend and target metadata",
    )
    expect('pto.mode = ' not in default_text, "generated PTODSL container IR should no longer expose public pto.mode")
    expect(
        'pto.backend = "vpto"' in default_text
        and 'pto.target_arch = "a5"' in default_text
        and 'pto.kernel_kind' not in default_text,
        "primary VPTO child module should carry PTOAS-facing backend metadata directly on the child module",
    )
    expect(
        'pto.backend = "vpto"' in explicit_text
        and 'pto.target_arch = "a5"' in explicit_text
        and 'pto.kernel_kind' not in explicit_text,
        "explicit specialization child module should keep the same VPTO child metadata shape",
    )
    expect(
        "ptodsl.compile_options" not in default_text,
        "backend-partitioned PTODSL child modules should no longer expose ptodsl.compile_options",
    )
    emitc_entry_text = host_vec_copy_emitc.compile().mlir_text()
    expect_parse_roundtrip_and_verify(emitc_entry_text, "emitc host_vec_copy specialization")
    expect(
        'module attributes {pto.backend = "emitc", pto.target_arch = "a5"}' in emitc_entry_text,
        "EmitC entry child module should encode the backend through pto.backend without VPTO kernel kind",
    )
    expect(
        '#pto.kernel_kind<' not in emitc_entry_text,
        "EmitC-only child modules should not carry VPTO kernel-kind metadata",
    )
    emitc_helper_text = (
        emitc_entry_calls_emitc_vector_kernel_module_metadata_probe.compile().mlir_text()
    )
    expect_parse_roundtrip_and_verify(
        emitc_helper_text,
        "emitc entry=False kernel-module specialization",
    )
    expect(
        'module attributes {pto.backend = "emitc", pto.kernel_kind = #pto.kernel_kind<vector>, pto.target_arch = "a5"}'
        in emitc_helper_text,
        "EmitC entry=False helper child should preserve kernel-kind metadata for PTOAS child compilation",
    )
    expect(
        host_vec_copy.compile()._module_spec.backend == "vpto",
        'default @pto.jit backend should stay "vpto"',
    )
    expect(
        host_vec_copy.compile()._module_spec.entry is True,
        "default @pto.jit should stay launch-entry oriented",
    )
    expect(
        host_vec_copy_emitc.compile()._module_spec.backend == "emitc",
        '@pto.jit(backend="emitc") should preserve the authored backend',
    )
    expect(
        host_vec_copy_emitc.compile()._module_spec.entry is True,
        "explicit backend selection should not change entry=True by default",
    )
    expect(
        non_entry_metadata_probe._compiler._module_spec.backend == "vpto",
        "non-entry helper should preserve its authored backend metadata",
    )
    expect(
        non_entry_metadata_probe._compiler._module_spec.entry is False,
        "@pto.jit(entry=False) should preserve kernel-module-vs-entry metadata",
    )
    helper_params = helper_device_abi_surface_probe._compiler._kernel_signature.positional_parameters
    expect(
        len(helper_params) == 5,
        "kernel-module ABI surface probe should keep all authored positional parameters",
    )
    expect(
        isinstance(helper_params[0], HelperMarkerParameterSpec) and helper_params[0].annotation is pto.Tile,
        "entry=False kernel module should accept pto.Tile parameters",
    )
    expect(
        isinstance(helper_params[1], HelperMarkerParameterSpec) and helper_params[1].annotation is pto.TensorView,
        "entry=False kernel module should accept pto.TensorView parameters",
    )
    expect(
        isinstance(helper_params[2], HelperMarkerParameterSpec) and helper_params[2].annotation is pto.PartitionTensorView,
        "entry=False kernel module should accept pto.PartitionTensorView parameters",
    )
    expect(
        isinstance(helper_params[3], DeviceParameterSpec),
        "entry=False kernel module should accept typed pointers from non-GM memory spaces",
    )
    expect(
        isinstance(helper_params[4], RuntimeScalarParameterSpec),
        "entry=False kernel module should accept PTO scalar parameters",
    )
    expect_raises(
        RuntimeError,
        helper_device_abi_surface_probe.compile,
        "is not directly compilable from Python",
    )
    helper_repr = repr(helper_device_abi_surface_probe)
    expect(
        "helper_device_abi_surface_probe" in helper_repr and "CompiledKernelHandle" not in helper_repr,
        "repr(@pto.jit(entry=False)) should expose lightweight metadata without triggering compilation",
    )
    helper_cache_signature = helper_device_abi_surface_probe.__ptodsl_cache_signature__()
    expect(
        helper_cache_signature[0] == "KernelHandle"
        and helper_cache_signature[1] == "helper_device_abi_surface_probe"
        and helper_cache_signature[3] == "helper_device_abi_surface_probe"
        and helper_cache_signature[4] is False,
        "@pto.jit(entry=False) handles should expose an explicit, stable cache-signature protocol",
    )
    expect(
        helper_cache_signature[7] is None and helper_cache_signature[8] is False,
        "default @pto.jit handles should leave the effective kernel kind unspecified",
    )
    expect_raises(
        RuntimeError,
        kernel_module_return_probe.compile,
        "is not directly compilable from Python",
    )
    kernel_module_runtime = SignatureTracingRuntime(
        kernel_module_return_probe._compiler._module_spec,
        kernel_module_return_probe._compiler._kernel_signature,
        kernel_module_return_probe._compiler._callback,
        constexpr_bindings={},
    )
    expect_raises(
        RuntimeError,
        lambda: kernel_module_runtime.trace_entry(None, 1),
        "@pto.jit(entry=False) kernel modules must return None",
    )
    expect(
        host_vec_copy.compile()._module_spec.insert_sync is None,
        "default @pto.jit insert_sync should stay unset and follow mode defaults",
    )
    expect(
        host_vec_copy_explicit.compile()._module_spec.insert_sync is None,
        "explicit @pto.jit insert_sync should stay unset and follow mode defaults",
    )
    expect(
        host_vec_copy_no_insert_sync.compile()._module_spec.insert_sync is False,
        "@pto.jit(insert_sync=False) should preserve the explicit override",
    )
    expect(
        host_vec_copy_explicit_insert_sync.compile()._module_spec.insert_sync is True,
        "@pto.jit(insert_sync=True) should preserve the explicit override",
    )
    closure_kernel_module_text = entry_closure_kernel_module_probe.compile().mlir_text()
    expect(
        "call @closure_helper__ptodsl_" in closure_kernel_module_text,
        "entry kernels that close over @pto.jit(entry=False) helpers should compile without implicitly building the helper",
    )
    kernel_module_compiled = entry_calls_kernel_module_probe.compile()
    kernel_module_call_text = kernel_module_compiled.mlir_text()
    expect_parse_roundtrip_and_verify(kernel_module_call_text, "entry calling kernel-module specialization")
    expect(
        "func.call @process_tile_module__ptodsl_" in kernel_module_call_text,
        "entry kernel should lower @pto.jit(entry=False) calls through the ABI-specialized symbol",
    )
    expect(
        "func.func public @process_tile_module__ptodsl_" in kernel_module_call_text,
        "kernel-module callee definition should be materialized as a public ABI-specialized symbol",
    )
    expect(
        'pto.visibility = "external"' in kernel_module_call_text,
        "kernel-module ABI-specialized primary definitions should carry explicit external artifact visibility",
    )
    expect(
        'pto.ptodsl.logical_name = "process_tile_module"' in kernel_module_call_text,
        "PTODSL-specialized kernel-module symbols should carry the authored logical symbol name as IR metadata",
    )
    expect(
        kernel_module_call_text.count("func.func private @process_tile_module__ptodsl_") >= 1,
        "kernel-module callsite lowering should materialize one private declaration for the ABI-specialized callee",
    )
    expect(
        kernel_module_call_text.count('pto.backend = "vpto"') >= 2
        and 'pto.kernel_kind' not in kernel_module_call_text,
        "entry-plus-helper specialization should materialize separate child modules for caller and callee",
    )
    ast_rewrite_kernel_module_text = entry_calls_ast_rewrite_kernel_module_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(
        ast_rewrite_kernel_module_text,
        "entry calling AST-rewritten kernel-module specialization",
    )
    expect(
        "func.func public @ast_rewrite_kernel_module_probe__ptodsl_" in ast_rewrite_kernel_module_text,
        "entry=False kernel-module lowering should materialize the AST-rewritten callee definition",
    )
    expect(
        'pto.visibility = "external"' in ast_rewrite_kernel_module_text,
        "AST-rewritten kernel-module primary definitions should carry explicit external artifact visibility",
    )
    expect(
        ast_rewrite_kernel_module_text.count("scf.for") >= 2,
        "entry=False kernel modules should rewrite Python range(...) loops before helper lowering",
    )
    kernel_module_graph = kernel_module_compiled.kernel_module_graph
    expect(
        kernel_module_graph is not None,
        "compiled @pto.jit artifacts should expose traced kernel-module import/dependency metadata",
    )
    expect(
        kernel_module_graph.dependencies == (("entry_calls_kernel_module_probe", ("process_tile_module",)),),
        "kernel-module callsite lowering should record one caller->callee dependency edge",
    )
    expect(
        len(kernel_module_graph.imports) == 1
        and kernel_module_graph.imports[0].caller_symbol_name == "entry_calls_kernel_module_probe"
        and kernel_module_graph.imports[0].target_symbol_name.startswith("process_tile_module__ptodsl_")
        and kernel_module_graph.imports[0].import_symbol_name.startswith("process_tile_module__ptodsl_"),
        "kernel-module import metadata should preserve caller/import/callee ownership including the ABI-specialized target symbol",
    )
    mixed_backend_text = emitc_entry_calls_vpto_kernel_module_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(mixed_backend_text, "emitc entry calling vpto kernel-module specialization")
    expect(
        'module attributes {pto.backend = "emitc", pto.target_arch = "a5"}' in mixed_backend_text,
        "mixed-backend caller child should encode the authored EmitC backend through pto.backend",
    )
    expect(
        'pto.backend = "vpto"' in mixed_backend_text
        and 'pto.target_arch = "a5"' in mixed_backend_text
        and 'pto.kernel_kind' not in mixed_backend_text,
        "mixed-backend callee child should preserve the callee's VPTO backend through child pto.backend metadata",
    )
    expect(
        "pto.tload" in mixed_backend_text and "pto.tstore" in mixed_backend_text,
        "mixed-backend EmitC entry should keep its top-level tile load/store path alongside the kernel-module call",
    )
    expect(
        "pto.section.vector {" not in mixed_backend_text,
        "PTODSL should leave the mixed-backend kernel-module body uncovered so PTOAS can infer its physical section",
    )
    expect(
        "pto.tload" in mixed_backend_text
        and "pto.tstore" in mixed_backend_text
        and "func.call @process_row_ptr_kernel_module__ptodsl_" in mixed_backend_text,
        "mixed-backend PTODSL IR should keep the naked entry tile path plus kernel-module call so PTOAS can infer the missing section later",
    )
    expect(
        "func.func public @process_row_ptr_kernel_module__ptodsl_(" not in mixed_backend_text,
        "ABI-specialized kernel-module public symbols should carry a stable specialization suffix",
    )
    expect(
        "func.func public @process_row_ptr_kernel_module__ptodsl_" in mixed_backend_text
        and "func.call @process_row_ptr_kernel_module__ptodsl_"
        in mixed_backend_text
        and ": (!pto.ptr<f32, gm>, !pto.ptr<f32, gm>, index) -> ()" in mixed_backend_text
        and "func.func private @process_row_ptr_kernel_module__ptodsl_"
        in mixed_backend_text,
        "mixed-backend kernel-module calls should currently lower through the C-ABI-compatible ptr/scalar subset",
    )
    expect(
        'pto.visibility = "external"' in mixed_backend_text,
        "mixed-backend kernel-module primary definitions should mark their artifact ABI visibility explicitly",
    )
    expect(
        mixed_backend_text.count("func.func private @process_row_ptr_kernel_module__ptodsl_") >= 1
        and "!pto.tile_buf" not in mixed_backend_text.split(
            "func.func private @process_row_ptr_kernel_module__ptodsl_",
            1,
        )[1].split("\n", 1)[0],
        "mixed-backend kernel-module calls should currently lower through the C-ABI-compatible ptr/scalar subset",
    )
    expect(
        "pto.mte_gm_ub" in mixed_backend_text and "pto.mte_ub_gm" in mixed_backend_text,
        "mixed-backend ptr/scalar kernel modules should be able to keep explicit VPTO data-movement ops in the callee child",
    )
    dynamic_buf_emitc_text = emitc_entry_calls_dynamic_buf_kernel_module_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(
        dynamic_buf_emitc_text,
        "emitc entry calling dynamic buf kernel-module specialization",
    )
    expect(
        'module attributes {pto.backend = "emitc", pto.target_arch = "a5"}' in dynamic_buf_emitc_text,
        "dynamic-buf emitc entry should encode the EmitC backend",
    )
    expect(
        "pto.get_buf_dyn" in dynamic_buf_emitc_text,
        "dynamic-buf kernel module IR should contain pto.get_buf_dyn ops",
    )
    expect(
        "pto.rls_buf_dyn" in dynamic_buf_emitc_text,
        "dynamic-buf kernel module IR should contain pto.rls_buf_dyn ops",
    )
    expect(
        dynamic_buf_emitc_text.count("pto.get_buf_dyn") == 2,
        "mixed-backend dynamic buf callee should keep both MTE2 and MTE3 get_buf_dyn ops",
    )
    expect(
        dynamic_buf_emitc_text.count("pto.rls_buf_dyn") == 2,
        "mixed-backend dynamic buf callee should keep both MTE2 and MTE3 rls_buf_dyn ops",
    )
    expect(
        "func.call @dynamic_buf_kernel_module__ptodsl_" in dynamic_buf_emitc_text,
        "emitc entry should call into the dynamic buf kernel module",
    )
    decorated_mixed_backend_text = emitc_entry_calls_vpto_kernel_module_via_decorated_simd_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(
        decorated_mixed_backend_text,
        "emitc entry calling vpto kernel-module through @pto.tileop specialization",
    )
    expect(
        re.search(
            r"call @explicit_vpto_kernel_module__ptodsl_[0-9a-f]+"
            r"\(%[a-zA-Z0-9_]+, %[a-zA-Z0-9_]+, %[a-zA-Z0-9_]+\)",
            decorated_mixed_backend_text,
        ) is not None,
        "plain Tile adapters should expand in the caller and invoke the kernel module directly",
    )
    expect(
        "emitc_vpto_kernel_module_callsite_simd_helper__ptodsl_" not in decorated_mixed_backend_text,
        "plain Tile adapters should not create an extra helper function boundary",
    )
    multi_abi_compiled = entry_calls_kernel_module_multiple_abi_probe.compile()
    multi_abi_text = multi_abi_compiled.mlir_text()
    expect_parse_roundtrip_and_verify(
        multi_abi_text,
        "entry calling one kernel-module symbol through multiple concrete ABIs",
    )
    expect(
        multi_abi_text.count("func.func public @process_tile_module__ptodsl_") == 2,
        "one kernel-module symbol called through two concrete Tile ABIs should materialize two specialized public callee definitions",
    )
    expect(
        multi_abi_text.count("func.func private @process_tile_module__ptodsl_") == 2,
        "one kernel-module symbol called through two concrete Tile ABIs should materialize two specialized private imports",
    )
    multi_abi_graph = multi_abi_compiled.kernel_module_graph
    expect(
        multi_abi_graph is not None and len(multi_abi_graph.imports) == 2,
        "multiple concrete kernel-module ABIs should be reflected in the traced import metadata",
    )
    expect(
        len({record.target_symbol_name for record in multi_abi_graph.imports}) == 2,
        "multiple concrete kernel-module ABIs should produce distinct target symbols in import metadata",
    )
    expect(
        multi_abi_graph.dependencies == (("entry_calls_kernel_module_multiple_abi_probe", ("process_tile_module",)),),
        "higher-level dependency metadata should still preserve the authored caller->callee edge",
    )
    host_vec_copy_compiled = host_vec_copy.compile()
    native_build_variants = (
        (
            "a3-pure-container",
            host_vec_copy_compiled,
            replace(host_vec_copy_compiled._module_spec, target_arch="a3"),
        ),
        ("explicit-level3-container", host_vec_copy_explicit_addr.compile(), None),
        ("same-backend-multi-child-container", kernel_module_compiled, None),
        ("mixed-backend-container", emitc_entry_calls_vpto_kernel_module_probe.compile(), None),
        ("source-auto", source_native_build_compiled, None),
        ("source-explicit", source_explicit_native_build_compiled, None),
        ("source-no-insert-sync", source_no_insert_sync_native_build_compiled, None),
    )
    native_build_observations = []
    launch_target_arches = []

    with TemporaryDirectory() as tmpdir:
        build_root = Path(tmpdir)

        def fake_artifacts(py_name, ir_function_name, specialization_key):
            cache_dir = build_root / f"{py_name}_{ir_function_name}"
            return NativeBuildArtifacts(
                cache_dir=cache_dir,
                mlir_path=cache_dir / "kernel.mlir",
                kernel_object=cache_dir / "kernel.o",
                launch_cpp=cache_dir / "launch.cpp",
                shared_library=cache_dir / f"lib{ir_function_name}.so",
                manifest_path=cache_dir / "manifest.json",
            )

        def fake_run_ptoas(mlir_path, kernel_object, *, target_arch, insert_sync=None, backend=None, pto_level=None):
            native_build_observations.append(
                {
                    "mlir_path": mlir_path,
                    "kernel_object": kernel_object,
                    "target_arch": target_arch,
                    "insert_sync": insert_sync,
                    "backend": backend,
                    "pto_level": pto_level,
                    "mlir_text": mlir_path.read_text(encoding="utf-8"),
                }
            )
            kernel_object.write_text("fake fatobj\n", encoding="utf-8")

        def fake_compile_launch_cpp(
            launch_cpp,
            launch_object,
            *,
            kernel_kind,
            target_arch,
            export_macro,
        ):
            expect(launch_cpp.is_file(), "native build should materialize launch.cpp before compiling it")
            expect(
                kernel_kind in {None, "vector", "cube"},
                "native build should forward the optional authored kernel kind",
            )
            launch_target_arches.append(target_arch)
            expect(export_macro.endswith("_EXPORTS"), "native build should preserve launch export macro naming")
            launch_object.write_text("fake launch object\n", encoding="utf-8")

        def fake_link_shared_library(launch_object, kernel_object, shared_library, *, kernel_kind):
            expect(launch_object.is_file(), "native build should compile launch.cpp before linking")
            expect(kernel_object.is_file(), "native build should run ptoas before shared-library link")
            expect(
                kernel_kind in {None, "vector", "cube"},
                "native build should preserve the optional kernel kind",
            )
            shared_library.write_text("fake shared library\n", encoding="utf-8")

        with mock.patch.object(native_build_runtime, "artifact_paths", side_effect=fake_artifacts), mock.patch.object(
            native_build_runtime, "is_native_build_current", return_value=False
        ), mock.patch.object(native_build_runtime, "_run_ptoas", side_effect=fake_run_ptoas), mock.patch.object(
            native_build_runtime, "_compile_launch_cpp", side_effect=fake_compile_launch_cpp
        ), mock.patch.object(
            native_build_runtime, "_link_shared_library", side_effect=fake_link_shared_library
        ), mock.patch.object(native_build_runtime, "runtime_library_flags", return_value=("-laclrt",)):
            for label, compiled, module_spec_override in native_build_variants:
                module_spec = module_spec_override or compiled._module_spec
                lib_path, launch_symbol = native_build_runtime.build_native_library(
                    py_name=compiled._py_name,
                    module_spec=module_spec,
                    kernel_signature=compiled._kernel_signature,
                    mlir_text=compiled.mlir_text(),
                    specialization_key=compiled.specialization_key,
                )
                expect(lib_path.is_file(), f"{label} native build should materialize the shared library artifact")
                expect(
                    launch_symbol.startswith("ptodsl_launch_"),
                    f"{label} native build should preserve PTODSL launch wrapper naming",
                )

    expect(
        len(native_build_observations) == len(native_build_variants),
        "native build should drive ptoas once per compiled container variant under test",
    )
    for (label, compiled, module_spec_override), observation, launch_target_arch in zip(
        native_build_variants, native_build_observations, launch_target_arches
    ):
        module_spec = module_spec_override or compiled._module_spec
        expect(
            observation["target_arch"] == module_spec.target_arch,
            f"{label} native build should still pass the target arch to ptoas",
        )
        expect(
            launch_target_arch == module_spec.target_arch,
            f"{label} native build should pass the target arch to launch compilation",
        )
        expected_insert_sync = (
            module_spec.insert_sync
            if module_spec.insert_sync is not None
            else module_spec.mode != "explicit"
        )
        expect(
            observation["insert_sync"] == expected_insert_sync,
            f"{label} native build should forward the effective insert_sync policy to ptoas",
        )
        expected_backend = module_spec.backend if module_spec.jit_source is not None else None
        expected_pto_level = "level3" if module_spec.mode == "explicit" else None
        expect(
            observation["backend"] == expected_backend,
            f"{label} native build should only forward ptoas backend overrides for source-backed kernels",
        )
        expect(
            observation["pto_level"] == expected_pto_level,
            f"{label} native build should derive the PTOAS level from the authored mode",
        )
        expect(
            observation["mlir_text"] == compiled.mlir_text(),
            f"{label} native build should hand the backend-partitioned container MLIR to ptoas unchanged",
        )
        if module_spec.jit_source is None:
            expected_module_count = 2
            expect(
                observation["mlir_text"].count("module") >= expected_module_count,
                f"{label} native build should route the authored module shape through ptoas",
            )
    with TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        mlir_path = tmpdir_path / "kernel.mlir"
        kernel_object = tmpdir_path / "kernel.o"
        mlir_path.write_text(default_text, encoding="utf-8")
        ptoas_cmds = []

        def fake_run_ptoas_cmd(cmd, *, cwd=None):
            ptoas_cmds.append(cmd)

        with mock.patch.object(native_build_runtime, "resolve_ptoas_binary", return_value=Path("/tmp/fake-ptoas")), mock.patch.object(
            native_build_runtime, "_run", side_effect=fake_run_ptoas_cmd
        ):
            native_build_runtime._run_ptoas(
                mlir_path,
                kernel_object,
                target_arch="a5",
            )

        expect(len(ptoas_cmds) == 1, "native build should issue exactly one ptoas command per kernel container")
        ptoas_cmd = ptoas_cmds[0]
        expect(
            ptoas_cmd[:2] == ["/tmp/fake-ptoas", "--pto-arch=a5"],
            "native build should still pass the ptoas binary plus target-arch flag",
        )
        expect(
            "--pto-backend=vpto" not in ptoas_cmd,
            "native build should no longer force a global VPTO backend when compiling backend-partitioned containers",
        )
        expect(
            "--pto-level=level3" not in ptoas_cmd,
            "native build should not pass a global pto-level flag by default",
        )
        expect(
            "--enable-insert-sync" not in ptoas_cmd,
            "native build should keep the default insert-sync policy unset when _run_ptoas is called directly",
        )
        expect(
            "--enable-tile-op-expand" in ptoas_cmd and str(mlir_path) in ptoas_cmd and str(kernel_object) in ptoas_cmd,
            "native build should still pass the shared PTOAS compile inputs and output path",
        )
        ptoas_cmds.clear()
        with mock.patch.object(native_build_runtime, "resolve_ptoas_binary", return_value=Path("/tmp/fake-ptoas")), mock.patch.object(
            native_build_runtime, "_run", side_effect=fake_run_ptoas_cmd
        ):
            native_build_runtime._run_ptoas(
                mlir_path,
                kernel_object,
                target_arch="a5",
                insert_sync=True,
            )
        expect(len(ptoas_cmds) == 1, "native build should issue exactly one ptoas command when insert_sync is forced on")
        expect(
            "--enable-insert-sync" in ptoas_cmds[0],
            "native build should pass --enable-insert-sync when the compiled module explicitly requests it",
        )
        ptoas_cmds.clear()
        with mock.patch.object(native_build_runtime, "resolve_ptoas_binary", return_value=Path("/tmp/fake-ptoas")), mock.patch.object(
            native_build_runtime, "_run", side_effect=fake_run_ptoas_cmd
        ):
            native_build_runtime._run_ptoas(
                mlir_path,
                kernel_object,
                target_arch="a5",
                backend="vpto",
                pto_level=native_build_runtime._effective_pto_level(mode="explicit"),
                insert_sync=True,
            )
        expect(len(ptoas_cmds) == 1, "native build should issue exactly one ptoas command with explicit-mode PTOAS policy")
        explicit_ptoas_cmd = ptoas_cmds[0]
        expect(
            "--pto-backend=vpto" in explicit_ptoas_cmd,
            "source-backed native build should pass the decorator backend to ptoas",
        )
        expect(
            "--pto-level=level3" in explicit_ptoas_cmd,
            'native build should pass --pto-level=level3 for mode="explicit"',
        )
        expect(
            "--enable-insert-sync" in explicit_ptoas_cmd,
            "source-backed native build should still pass explicit/effective insert-sync to ptoas",
        )
        ptoas_cmds.clear()
        vmi_mlir_text = vmi_wrapper_dispatch_probe.compile().mlir_text()
        mlir_path.write_text(vmi_mlir_text, encoding="utf-8")
        with mock.patch.object(native_build_runtime, "resolve_ptoas_binary", return_value=Path("/tmp/fake-ptoas")), mock.patch.object(
            native_build_runtime, "_run", side_effect=fake_run_ptoas_cmd
        ):
            native_build_runtime._run_ptoas(
                mlir_path,
                kernel_object,
                target_arch="a5",
            )
        expect(
            len(ptoas_cmds) == 1,
            "native build should issue exactly one ptoas command for PTODSL-generated VMI MLIR",
        )
    expect("valid=?" not in default_text, "default alloc_tile() should keep full static valid-shape when valid_shape= is omitted")
    auto_mode_violation = expect_raises(
        RuntimeError,
        auto_mode_explicit_surface_violation_probe.compile,
        '@pto.jit(mode="explicit")',
    )
    expect(
        "auto-mode contract violation" in str(auto_mode_violation),
        "explicit-only surface use in auto mode should be diagnosed as an auto-mode contract violation",
    )
    expect(
        "auto_mode_explicit_surface_violation_probe" in str(auto_mode_violation),
        "auto-mode DMA violation should identify the authored kernel name",
    )
    expect(
        __file__ in str(auto_mode_violation),
        "auto-mode DMA violation should preserve the authored source file",
    )
    auto_mode_vecscope_violation = expect_raises(
        RuntimeError,
        auto_mode_vecscope_violation_probe.compile,
        '@pto.jit(mode="explicit")',
    )
    expect(
        "auto-mode contract violation" in str(auto_mode_vecscope_violation),
        "auto-mode vecscope use should be diagnosed as an auto-mode contract violation",
    )
    expect(
        "pto.vecscope()" in str(auto_mode_vecscope_violation),
        "auto-mode vecscope violation should identify the explicit-only surface",
    )
    expect(
        "auto_mode_vecscope_violation_probe" in str(auto_mode_vecscope_violation),
        "auto-mode vecscope violation should identify the authored kernel name",
    )
    merged_cross_mode_text = str(pto.merge_jit_modules(host_vec_copy.compile(), host_vec_copy_explicit.compile()))
    expect_parse_roundtrip_and_verify(merged_cross_mode_text, "merged cross-mode PTODSL container")
    expect(
        'func.func @host_vec_copy(' in merged_cross_mode_text
        and 'func.func @host_vec_copy_explicit(' in merged_cross_mode_text,
        "merge_jit_modules() should no longer reject child modules that differ only in compile policy",
    )
    merged_same_mode_text = str(pto.merge_jit_modules(host_vec_copy.compile(), host_vec_copy.compile(BLOCK=64)))
    expect_parse_roundtrip_and_verify(merged_same_mode_text, "merged same-mode PTODSL container")
    expect(
        merged_same_mode_text.count('func.func @host_vec_copy(') == 2,
        "merge_jit_modules() should preserve same-named specializations in independent backend child modules",
    )

    runtime_metadata_text = runtime_metadata_kernel.compile().mlir_text()
    expect_parse_roundtrip_and_verify(runtime_metadata_text, "runtime metadata specialization")
    expect(
        re.search(
            r"pto\.make_tensor_view %arg0, shape = \[%[a-zA-Z0-9_]+, %[a-zA-Z0-9_]+\], strides = \[%[a-zA-Z0-9_]+, %[a-zA-Z0-9_]+\]",
            runtime_metadata_text,
        ) is not None,
        "make_tensor_view should preserve explicitly authored runtime shape/stride metadata",
    )

    explicit_layout_text = explicit_layout_tensor_view_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(explicit_layout_text, "explicit tensor_view layout specialization")
    expect(
        re.search(
            r'pto\.make_tensor_view %arg0, shape = \[%[a-zA-Z0-9_]+, %[a-zA-Z0-9_]+\], strides = \[%[a-zA-Z0-9_]+, %[a-zA-Z0-9_]+\] \{layout = #pto\.layout<dn>\}',
            explicit_layout_text,
        ) is not None,
        "make_tensor_view(layout='DN') should preserve the explicit layout attribute in MLIR",
    )

    tile_surface_text = tile_surface_compute_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(tile_surface_text, "tile surface compute specialization")
    tile_sort_gather_text = tile_sort_gather_surface_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(tile_sort_gather_text, "tile sort/gather surface specialization")
    expect("pto.texpands" in tile_surface_text, "pto.tile.expands should lower to pto.texpands")
    expect("pto.treshape" in tile_surface_text, "pto.tile.reshape should lower to pto.treshape")
    expect("pto.tadd " in tile_surface_text, "pto.tile.add should lower to pto.tadd")
    expect("pto.tadds" in tile_surface_text, "pto.tile.adds should lower to pto.tadds")
    expect("pto.tcmps" in tile_surface_text, "pto.tile.cmps should lower to pto.tcmps")
    expect("pto.tsort32" in tile_sort_gather_text, "pto.tile.sort32 should lower to pto.tsort32")
    expect("pto.tmrgsort" in tile_sort_gather_text, "pto.tile.mrgsort should lower to pto.tmrgsort")
    expect(tile_sort_gather_text.count("pto.tgather") == 2, "tile gather wrappers should lower to pto.tgather")
    expect("#pto.mask_pattern<P0101>" in tile_sort_gather_text, "pto.tile.gather should preserve P0101")
    expect("#pto.mask_pattern<P1010>" in tile_sort_gather_text, "pto.tgather should preserve P1010")
    tile_ci_text = tile_ci_surface_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(tile_ci_text, "tile ci surface specialization")
    expect(tile_ci_text.count("pto.tci") == 2, "pto.tile.ci should lower to pto.tci")
    expect("descending = true" in tile_ci_text, "pto.tile.ci(descending=True) should preserve the descending attribute in MLIR")
    expect(
        re.search(
            r"pto\.alloc_tile valid_row = %[a-zA-Z0-9_]+ valid_col = %[a-zA-Z0-9_]+ : !pto\.tile_buf<vec, 1x128xf32, valid=\?x\?>",
            runtime_metadata_text,
        ) is not None,
        "alloc_tile(valid_shape=[rows, cols]) should lower runtime metadata through valid_row/valid_col operands",
    )
    expect(
        re.search(
            r"sizes = \[%[a-zA-Z0-9_]+, %[a-zA-Z0-9_]+\]",
            runtime_metadata_text,
        ) is not None,
        "partition_view sizes derived from tensor metadata should remain runtime MLIR values",
    )

    tile_window_matmul_text = tile_surface_window_matmul_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(tile_window_matmul_text, "tile extract/insert/matmul specialization")
    expect("pto.textract" in tile_window_matmul_text, "pto.tile.extract should lower to pto.textract")
    expect("pto.tinsert" in tile_window_matmul_text, "pto.tile.insert should lower to pto.tinsert")
    expect("pto.tmatmul ins(" in tile_window_matmul_text, "pto.tile.matmul should lower to pto.tmatmul")
    expect("pto.tmatmul.acc ins(" in tile_window_matmul_text, "pto.tile.matmul_acc should lower to pto.tmatmul.acc")
    expect("!pto.tile_buf<left, 16x16xf16" in tile_window_matmul_text, "pto.tile.matmul lhs should preserve LEFT scratch tile typing")
    expect("!pto.tile_buf<right, 16x16xf16" in tile_window_matmul_text, "pto.tile.matmul rhs should preserve RIGHT scratch tile typing")
    expect("!pto.tile_buf<acc, 16x16xf32" in tile_window_matmul_text, "pto.tile.matmul/matmul_acc should preserve ACC destination typing")
    expect(
        "pto.tinsert ins(" in tile_window_matmul_text and ", %c0, %c32 :" in tile_window_matmul_text,
        "pto.tile.insert should preserve the authored insertion offsets in MLIR",
    )
    expect(
        "pto.tmatmul.acc ins(" in tile_window_matmul_text and tile_window_matmul_text.count("%") >= 3,
        "pto.tile.matmul_acc should materialize acc_in/lhs/rhs operands in MLIR",
    )

    tile_transfer_text = tile_transfer_surface_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(tile_transfer_text, "tile transfer surface specialization")
    expect("pto.tload" in tile_transfer_text, "pto.tile.load(tensor, tile) should lower to pto.tload")
    expect("pto.tstore" in tile_transfer_text, "pto.tile.store(tile, tensor) should lower to pto.tstore")
    expect(
        re.search(
            r"sizes = \[%[a-zA-Z0-9_]+, %[a-zA-Z0-9_]+\]",
            tile_transfer_text,
        ) is not None,
        "pto.tile.load/store overloads should infer partition sizes from tile.valid_shape",
    )

    authored_addr_tile_text = authored_addr_tile_surface_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(authored_addr_tile_text, "authored alloc_tile addr specialization")
    expect(
        re.search(
            r"pto\.alloc_tile addr = %c0_i64 valid_row = %[a-zA-Z0-9_]+ valid_col = %[a-zA-Z0-9_]+ : !pto\.tile_buf<vec, 1x128xf32, valid=\?x\?>",
            authored_addr_tile_text,
        ) is not None,
        "alloc_tile(shape=..., dtype=..., addr=int, valid_shape=...) should coerce Python ints to i64 operands",
    )

    dynamic_addr_tile_text = dynamic_addr_tile_surface_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(dynamic_addr_tile_text, "dynamic alloc_tile addr specialization")
    expect(
        "arith.index_cast %arg0 : i32 to index" in dynamic_addr_tile_text,
        "alloc_tile(addr=runtime integer metadata) should first bridge the public i32 scalar to index",
    )
    expect(
        re.search(
            r"arith\.index_cast %[a-zA-Z0-9_]+ : index to i64",
            dynamic_addr_tile_text,
        ) is not None,
        "alloc_tile(addr=runtime index) should still cast the bridged index metadata to i64 before lowering",
    )
    expect(
        re.search(
            r"pto\.alloc_tile addr = %[a-zA-Z0-9_]+ valid_row = %[a-zA-Z0-9_]+ valid_col = %[a-zA-Z0-9_]+ : !pto\.tile_buf<vec, 1x128xf32, valid=\?x\?>",
            dynamic_addr_tile_text,
        ) is not None,
        "alloc_tile(shape=..., dtype=..., addr=runtime value, valid_shape=...) should accept dynamic i64-like operands",
    )

    tile_valid_shape_text = tile_valid_shape_update_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(tile_valid_shape_text, "tile valid-shape update specialization")
    expect(
        re.search(
            r"pto\.set_validshape %[a-zA-Z0-9_]+, %[a-zA-Z0-9_]+, %[a-zA-Z0-9_]+ : !pto\.tile_buf<vec, 1x128xf32, valid=\?x\?>",
            tile_valid_shape_text,
        ) is not None,
        "tile.valid_shape = [rows, cols] should lower to pto.set_validshape on a dynamic-valid tile",
    )

    tile_valid_shape_1d_text = tile_valid_shape_update_1d_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(tile_valid_shape_1d_text, "1D tile valid-shape update specialization")
    expect(
        re.search(
            r"pto\.set_validshape %[a-zA-Z0-9_]+, %[a-zA-Z0-9_]+, %[a-zA-Z0-9_]+ : !pto\.tile_buf<vec, 1x128xf32, valid=\?x\?>",
            tile_valid_shape_1d_text,
        ) is not None,
        "tile.valid_shape = [length] should lower to pto.set_validshape on a rank-1 dynamic-valid tile",
    )

    make_mask_index_roundtrip_text = make_mask_index_roundtrip_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(make_mask_index_roundtrip_text, "make_mask index round-trip specialization")
    expect(
        "pto.plt_b32 %arg2 : i32 -> !pto.mask<b32>, i32" in make_mask_index_roundtrip_text,
        "make_mask(...) should accept public i32 runtime counts directly at the hardware tail-mask operand type",
    )
    expect(
        re.search(
            r"iter_args\(%[a-zA-Z0-9_]+ = %arg0\) -> \(i32\)",
            make_mask_index_roundtrip_text,
        ) is not None,
        "make_mask(...) should allow loop-carried public i32 remainders without manual index casts",
    )
    expect(
        re.search(
            r"scf\.yield %[a-zA-Z0-9_]+ : i32",
            make_mask_index_roundtrip_text,
        ) is not None,
        "make_mask(...) should keep the carried remainder in public i32 form after tail-mask generation",
    )

    carry_static_pyint_init_text = carry_static_pyint_init_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(carry_static_pyint_init_text, "carry static pyint init specialization")
    expect(
        re.search(
            r"iter_args\(%[a-zA-Z0-9_]+ = %c64_i32\) -> \(i32\)",
            carry_static_pyint_init_text,
        ) is not None,
        "pto.for_(...).carry(remained=64) should materialize Python int carry init values as public i32 constants",
    )
    expect(
        "pto.plt_b32" in carry_static_pyint_init_text,
        "a carried Python int should remain compatible with make_mask(...) without manual pto.const(...) wrapping",
    )

    fixed_integer_index_coercion_text = fixed_integer_index_coercion_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(fixed_integer_index_coercion_text, "fixed integer index coercion specialization")
    expect(
        re.search(
            r"arith\.index_cast %[a-zA-Z0-9_]+ : index to i32",
            fixed_integer_index_coercion_text,
        ) is not None,
        "fixed-width integer parameters should coerce runtime index values through shared scalar adaptation",
    )
    expect(
        "pto.plt_b32" in fixed_integer_index_coercion_text,
        "make_mask(...) should still lower runtime index counts to the predicate-load scalar path",
    )

    SUBKERNEL_OBSERVATIONS.clear()
    shared_subkernel_text = shared_subkernel_lowering_probe.compile(TRACE_TOKEN=1).mlir_text()
    expect_parse_roundtrip_and_verify(shared_subkernel_text, "shared subkernel lowering specialization")
    expect(
        SUBKERNEL_OBSERVATIONS == [
            ("tileop", "top_level_cube_probe", 1),
            ("tileop", "top_level_simd_probe", 1),
            ("tileop", "nested_simd_probe", 1),
        ],
        f"unexpected shared subkernel lowering observations: {SUBKERNEL_OBSERVATIONS!r}",
    )
    expect(
        re.search(r"call @top_level_cube_probe__ptodsl_[0-9a-f]+\(\)", shared_subkernel_text) is not None
        and re.search(r"call @top_level_simd_probe__ptodsl_[0-9a-f]+\(\)", shared_subkernel_text) is not None
        and re.search(r"call @nested_simd_probe__ptodsl_[0-9a-f]+\(\)", shared_subkernel_text) is not None,
        "@pto.tileop decorated helpers should lower to helper calls in the caller body",
    )
    expect(
        shared_subkernel_text.count("pto.tileop.helper") == 3
        and "pto.section.vector" not in shared_subkernel_text
        and "pto.section.cube" not in shared_subkernel_text,
        "outlined TileOp helpers should defer section kind materialization to PTOAS",
    )
    explicit_vector_inline_tileop_text = explicit_vector_inline_tileop_probe.compile(
        TRACE_TOKEN=1
    ).mlir_text()
    expect_parse_roundtrip_and_verify(
        explicit_vector_inline_tileop_text,
        "explicit vector jit calling inline TileOp specialization",
    )
    expect(
        "pto.kernel_kind = #pto.kernel_kind<vector>" in explicit_vector_inline_tileop_text
        and "pto.tileop.helper" in explicit_vector_inline_tileop_text
        and "pto.section.vector {" not in explicit_vector_inline_tileop_text,
        "inline pto.tileop() scopes should defer physical section materialization to PTOAS",
    )

    INLINE_SUBKERNEL_SCOPE_OBSERVATIONS.clear()
    inline_subkernel_scope_text = inline_subkernel_scope_probe.compile(TRACE_TOKEN=1).mlir_text()
    expect_parse_roundtrip_and_verify(inline_subkernel_scope_text, "inline subkernel scope specialization")
    expect(
        INLINE_SUBKERNEL_SCOPE_OBSERVATIONS == [
            ("simt", "inline_simt", 1),
            ("tileop", "inline_tileop", 1),
        ],
        f"unexpected inline subkernel scope observations: {INLINE_SUBKERNEL_SCOPE_OBSERVATIONS!r}",
    )
    expect(
        inline_subkernel_scope_text.count("pto.section.simt") == 1
        and re.search(r"pto\.section\.simt\s*<<<", inline_subkernel_scope_text) is not None,
        "inline pto.simt() should lower to one inline pto.section.simt region",
    )
    expect(
        re.search(r"call @inline_simt_[0-9]+__ptodsl_[0-9a-f]+\([^\\n]*\)", inline_subkernel_scope_text) is None
        and re.search(r"call @inline_tileop_[0-9]+__ptodsl_[0-9a-f]+\([^\\n]*\)", inline_subkernel_scope_text) is not None,
        "inline pto.simt() should stay in-place while inline pto.tileop() still lowers to one helper call",
    )
    expect(
        "pto.tileop.helper" in inline_subkernel_scope_text
        and "pto.section.vector {" not in inline_subkernel_scope_text
        and "pto.section.cube {" not in inline_subkernel_scope_text
        and "pto.store" in inline_subkernel_scope_text,
        "outlined inline TileOp helpers should mark TileOp helpers and preserve SIMT scalar ops",
    )

    inline_simt_launch_text = inline_simt_launch_dims_probe.compile(TRACE_TOKEN=1).mlir_text()
    expect_parse_roundtrip_and_verify(inline_simt_launch_text, "inline simt launch-dims specialization")
    expect(
        "pto.section.simt<<<32, 2, 1>>>" in inline_simt_launch_text,
        "with pto.simt(dim_x, dim_y, dim_z) should emit static inline pto.section.simt launch dims",
    )
    expect(
        "pto.simt_launch" not in inline_simt_launch_text
        and "pto.store_vfsimt_info" not in inline_simt_launch_text,
        "with pto.simt(dim_x, dim_y, dim_z) should not emit caller-side SIMT launch metadata",
    )
    expect(
        re.search(r"func\.func @inline_simt_[0-9]+__ptodsl_[0-9a-f]+", inline_simt_launch_text)
        is None
        and "pto.simt_entry" not in inline_simt_launch_text,
        "inline SIMT launch-dims body should remain in the enclosing kernel during DSL tracing",
    )
    expect_raises(
        TypeError,
        lambda: pto.simt(32, 1),
        "expects exactly three",
    )
    expect_raises(
        TypeError,
        lambda: inline_simt_dynamic_launch_dim_probe.compile(TRACE_TOKEN=1),
        "static Python int",
    )

    simt_text = simt_helper_lowering_probe.compile(TRACE_TOKEN=1).mlir_text()
    expect_parse_roundtrip_and_verify(simt_text, "simt helper lowering specialization")
    expect(
        simt_text.count("pto.store_vfsimt_info") == 2,
        "each @pto.simt callsite should materialize a caller-side store_vfsimt_info",
    )
    expect(
        re.search(r"call @simt_tid_probe__simt_\d+\(\)", simt_text) is not None,
        "each @pto.simt callsite should lower to a func.call of the helper symbol",
    )
    expect(
        len(re.findall(r"call @simt_tid_probe__simt_\d+\(\)", simt_text)) == 2,
        "both @pto.simt callsites should call the same helper specialization",
    )
    expect(
        len(
            re.findall(
                r"func\.func @simt_tid_probe__simt_\d+\(\) attributes \{[^}]*pto\.simt_entry[^}]*\}",
                simt_text,
            )
        )
        == 1,
        "@pto.simt helper should materialize exactly one reusable pto.simt_entry function",
    )
    expect(
        "pto.tileop.helper" not in simt_text,
        "@pto.simt helpers should not be modeled as TileOp helpers",
    )
    expect("pto.get_tid_x" in simt_text, "SIMT helper body should contain pto.get_tid_x")
    expect("pto.get_tid_y" in simt_text, "SIMT helper body should contain pto.get_tid_y")
    expect("pto.get_tid_z" in simt_text, "SIMT helper body should contain pto.get_tid_z")

    alloc_buffer_local_text = alloc_buffer_local_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(alloc_buffer_local_text, "alloc_buffer local specialization")
    expect(
        "llvm.alloca" in alloc_buffer_local_text and "x f32" in alloc_buffer_local_text,
        "alloc_buffer should lower to an LLVM stack allocation in the SIMT helper",
    )
    expect(
        "pto.persistent" not in alloc_buffer_local_text,
        "alloc_buffer inside a SIMT helper should remain section-local",
    )
    expect(
        re.search(
            r"func\.func @alloc_buffer_local_helper__simt_\d+\(\) attributes \{pto\.simt_entry\}",
            alloc_buffer_local_text,
        )
        is not None,
        "alloc_buffer probe should keep allocation inside the SIMT helper body",
    )
    alloc_buffer_top_level_text = alloc_buffer_outside_simt_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(alloc_buffer_top_level_text, "alloc_buffer top-level specialization")
    expect(
        "llvm.alloca" in alloc_buffer_top_level_text and "pto.persistent" in alloc_buffer_top_level_text,
        "alloc_buffer outside a SIMT helper should be marked as a persistent fragment candidate",
    )
    alloc_buffer_tileop_helper_text = alloc_buffer_tileop_helper_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(alloc_buffer_tileop_helper_text, "alloc_buffer tileop helper specialization")
    expect(
        "llvm.alloca" in alloc_buffer_tileop_helper_text
        and "pto.persistent" not in alloc_buffer_tileop_helper_text,
        "alloc_buffer inside a TileOp helper should remain local",
    )
    alloc_buffer_inline_simt_text = alloc_buffer_inline_simt_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(alloc_buffer_inline_simt_text, "alloc_buffer inline simt specialization")
    expect(
        "llvm.alloca" in alloc_buffer_inline_simt_text and "pto.persistent" not in alloc_buffer_inline_simt_text,
        "alloc_buffer inside an inline SIMT section should remain local",
    )
    rmsnorm_alloc_buffer_text = rmsnorm_alloc_buffer_layout_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(rmsnorm_alloc_buffer_text, "RMSNorm hand-authored UB layout specialization")
    for expected_offset in (4096, 4224, 12416, 20608):
        expect(
            f"arith.constant {expected_offset} : index" in rmsnorm_alloc_buffer_text,
            f"RMSNorm hand-authored UB layout should materialize f32 offset {expected_offset}",
        )
    expect(
        rmsnorm_alloc_buffer_text.count("llvm.alloca") == 2,
        "RMSNorm alloc_buffer fragment helper should allocate x_frag and sum_sq locally",
    )
    expect(
        re.search(r"call @rmsnorm_alloc_buffer_frag_helper__simt_\d+\(", rmsnorm_alloc_buffer_text)
        is not None,
        "RMSNorm alloc_buffer layout should pass explicitly authored UB pointers through the existing SIMT helper call path",
    )
    expect_raises(
        TypeError,
        lambda: simt_helper_reject_alloc_buffer_probe.compile(),
        "do not accept pto.alloc_buffer",
    )
    expect_raises(
        TypeError,
        lambda: simt_helper_reject_alloc_buffer_offset_probe.compile(),
        "do not accept pto.alloc_buffer",
    )

    simt_launch_text = simt_explicit_launch_probe.compile(TRACE_TOKEN=1).mlir_text()
    expect_parse_roundtrip_and_verify(simt_launch_text, "explicit simt launch specialization")
    expect(
        re.search(r"pto\.simt_launch @simt_query_probe__simt_\d+<<<", simt_launch_text) is not None,
        "pto.simt_launch(...) should emit VPTO simt_launch sugar",
    )
    expect(
        re.search(r"func\.func @simt_query_probe__simt_\d+\(\) attributes \{pto\.simt_entry\}", simt_launch_text) is not None,
        "explicit pto.simt_launch should materialize a reusable pto.simt_entry helper",
    )
    simt_launch_sugar_text = simt_launch_index_sugar_probe.compile(TRACE_TOKEN=1).mlir_text()
    expect_parse_roundtrip_and_verify(simt_launch_sugar_text, "indexed simt launch specialization")
    expect(
        re.search(r"pto\.simt_launch @simt_query_probe__simt_\d+<<<", simt_launch_sugar_text) is not None,
        "@pto.simt helper[x, y, z](...) should emit VPTO simt_launch sugar",
    )
    simt_grouped_query_text = simt_grouped_query_launch_probe.compile(TRACE_TOKEN=1).mlir_text()
    expect_parse_roundtrip_and_verify(simt_grouped_query_text, "grouped simt query specialization")
    expect(
        re.search(r"pto\.simt_launch @simt_grouped_query_probe__simt_\d+<<<", simt_grouped_query_text) is not None,
        "grouped SIMT query probe should be launchable through helper[x, y, z](...)",
    )
    for op_name in (
        "pto.get_tid_x",
        "pto.get_tid_y",
        "pto.get_tid_z",
        "pto.get_block_dim_x",
        "pto.get_block_dim_y",
        "pto.get_block_dim_z",
        "pto.get_grid_dim_x",
        "pto.get_grid_dim_y",
        "pto.get_grid_dim_z",
    ):
        expect(
            simt_grouped_query_text.count(op_name) == 1,
            f"grouped SIMT query helpers should lower exactly once to {op_name}",
        )
    expect(
        simt_grouped_query_text.count("pto.keep") == 9,
        "grouped SIMT query helpers should return values that can be consumed by later micro-ops",
    )
    expect_raises(
        TypeError,
        lambda: simt_query_probe[32, 1](),
        "helper[dim_x, dim_y, dim_z]",
    )
    expect_raises(
        TypeError,
        lambda: ast_subkernel_runtime_for_helper[32, 1, 1](),
        "only @pto.simt",
    )
    simt_resource_attr_text = simt_resource_attr_launch_probe.compile(TRACE_TOKEN=1).mlir_text()
    expect_parse_roundtrip_and_verify(simt_resource_attr_text, "simt resource attr launch specialization")
    expect(
        re.search(
            r"func\.func @simt_resource_attr_probe__simt_\d+\(\) attributes \{pto\.simt_entry, pto\.simt_max_threads = 256 : i32\}",
            simt_resource_attr_text,
        ) is not None,
        "@pto.simt(max_threads=...) should attach resource attrs to the helper function",
    )
    expect_raises(
        ValueError,
        lambda: pto.simt(max_threads=0)(lambda: None),
        "max_threads",
    )

    def _enter_inline_simt_with_resource_attr():
        with pto.simt(max_threads=256):
            pass

    expect_raises(
        TypeError,
        _enter_inline_simt_with_resource_attr,
        "function decorator",
    )
    for op_name in (
        "pto.get_tid_x",
        "pto.get_tid_y",
        "pto.get_tid_z",
        "pto.get_block_dim_x",
        "pto.get_block_dim_y",
        "pto.get_block_dim_z",
        "pto.get_grid_dim_x",
        "pto.get_grid_dim_y",
        "pto.get_grid_dim_z",
        "pto.get_block_idx_x",
        "pto.get_block_idx_y",
        "pto.get_block_idx_z",
        "pto.get_veccoreid",
        "pto.get_clock32",
        "pto.get_clock64",
        "pto.get_laneid",
        "pto.get_lanemask_eq",
        "pto.get_lanemask_le",
        "pto.get_lanemask_lt",
        "pto.get_lanemask_ge",
        "pto.get_lanemask_gt",
    ):
        expect(op_name in simt_launch_text, f"SIMT query body should contain {op_name}")

    simt_arg_type_text = simt_specialized_arg_type_probe.compile(TRACE_TOKEN=1).mlir_text()
    expect_parse_roundtrip_and_verify(simt_arg_type_text, "simt arg-type specialization")
    expect(
        re.search(r"func\.func @simt_specialized_i32_ptr_probe__simt_\d+\(", simt_arg_type_text) is not None
        and re.search(r"func\.func @simt_specialized_f32_ptr_probe__simt_\d+\(", simt_arg_type_text) is not None,
        "typed @pto.simt pointer helpers should materialize one helper per explicit ABI",
    )
    expect(
        "!pto.ptr<i32, gm>" in simt_arg_type_text and "!pto.ptr<f32, gm>" in simt_arg_type_text,
        "SIMT argument-type specializations should preserve distinct helper pointer types",
    )

    simt_static_kwarg_text = simt_specialized_static_kwarg_probe.compile(TRACE_TOKEN=1).mlir_text()
    expect_parse_roundtrip_and_verify(simt_static_kwarg_text, "simt static kwarg specialization")
    expect(
        len(re.findall(r"func\.func @simt_specialized_flag_probe__simt_\d+\(", simt_static_kwarg_text)) == 2,
        "same @pto.simt body launched with different static kwargs should materialize two helpers",
    )
    expect("pto.get_tid_x" in simt_static_kwarg_text, "FLAG=True SIMT specialization should emit get_tid_x")
    expect("pto.get_tid_y" in simt_static_kwarg_text, "FLAG=False SIMT specialization should emit get_tid_y")

    simt_full_text = simt_full_surface_probe.compile(TRACE_TOKEN=1).mlir_text()
    expect_parse_roundtrip_and_verify(simt_full_text, "full simt surface specialization")
    for op_name in (
        "pto.vote_all",
        "pto.vote_any",
        "pto.vote_uni",
        "pto.vote_ballot",
        "pto.shuffle_idx",
        "pto.shuffle_up",
        "pto.shuffle_down",
        "pto.shuffle_bfly",
        "pto.redux_add",
        "pto.redux_max",
        "pto.redux_min",
        "pto.ldg",
        "pto.stg",
        "pto.atomic_exch",
        "pto.atomic_add",
        "pto.atomic_sub",
        "pto.atomic_min",
        "pto.atomic_max",
        "pto.atomic_and",
        "pto.atomic_or",
        "pto.atomic_xor",
        "pto.atomic_cas",
        "pto.prmt",
        "pto.mulhi",
        "pto.mul_i32toi64",
        "pto.absf",
        "pto.sqrt",
        "pto.exp",
        "pto.log",
        "pto.sin",
        "pto.cos",
        "pto.pow",
        "pto.ceil",
        "pto.floor",
        "pto.rint",
        "pto.round",
        "pto.fmin",
        "pto.fmax",
        "pto.fma",
        "pto.convert",
        "pto.syncthreads",
        "pto.threadfence",
        "pto.threadfence_block",
        "pto.keep",
        "pto.resume",
    ):
        expect(op_name in simt_full_text, f"full SIMT surface should contain {op_name}")
    expect(
        re.search(r"pto\.fmin .* : f16, f16 -> f16", simt_full_text) is not None,
        "full SIMT surface should accept scalar f16 pto.fmin",
    )
    expect(
        re.search(r"pto\.fmax .* : f16, f16 -> f16", simt_full_text) is not None,
        "full SIMT surface should accept scalar f16 pto.fmax",
    )
    for fp8_vec in (
        "vector<4xf8E4M3FN>",
        "vector<8xf8E4M3FN>",
        "vector<4xf8E5M2>",
        "vector<8xf8E5M2>",
    ):
        expect(
            f"!pto.ptr<{fp8_vec}, gm>" in simt_full_text,
            f"SIMT FP8 extension should preserve !pto.ptr<{fp8_vec}, gm>",
        )
        expect(
            f"!pto.ptr<{fp8_vec}, gm>, {fp8_vec}" in simt_full_text,
            f"SIMT FP8 extension stg should accept exact {fp8_vec} payload",
        )

    expect_raises(
        TypeError,
        lambda: simt_invalid_redux_signedness_launch.compile(TRACE_TOKEN=1).mlir_text(),
        "requires signedness",
    )
    expect_raises(
        TypeError,
        lambda: simt_invalid_convert_signedness_launch.compile(TRACE_TOKEN=1).mlir_text(),
        "requires signedness",
    )
    expect_raises(
        TypeError,
        lambda: simt_invalid_atomic_signedness_launch.compile(TRACE_TOKEN=1).mlir_text(),
        "does not accept signedness",
    )

    ast_subkernel_runtime_for_text = ast_subkernel_runtime_for_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(
        ast_subkernel_runtime_for_text,
        "AST-rewritten subkernel runtime for specialization",
    )
    expect(
        ast_subkernel_runtime_for_text.count("scf.for") == 1,
        "@pto.tileop helper should rewrite Python range(...) loops into runtime scf.for",
    )
    expect(
        "pto.vlds" in ast_subkernel_runtime_for_text and "pto.vsts" in ast_subkernel_runtime_for_text,
        "rewritten @pto.tileop helper body should preserve Vector compute inside the caller trace",
    )

    carry_text = carry_loop_lowering_probe.compile(BLOCK=32).mlir_text()
    expect_parse_roundtrip_and_verify(carry_text, "carry loop specialization")
    expect("scf.for" in carry_text, "carry loop should lower to scf.for")
    expect("iter_args(" in carry_text, "carry loop should lower named state through scf.for iter_args")
    expect("scf.yield" in carry_text, "carry loop should lower loop.update(...) to scf.yield")
    expect(
        carry_text.count("!pto.tile_buf<vec, 1x32xf32>") >= 3,
        "carry loop MLIR should materialize the specialized carried tile types",
    )
    expect(
        re.search(r"outs\(%[^\s]+#2 : !pto\.tile_buf<vec, 1x32xf32>\)", carry_text) is not None,
        "loop.final(\"o\") should materialize the third scf.for result as the final carried state",
    )

    branch_then_only_text = branch_handle_then_only_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(branch_then_only_text, "branch handle then-only specialization")
    expect(branch_then_only_text.count("scf.if") == 1, "then-only branch handle should lower to one scf.if")
    expect("pto.barrier <PIPE_ALL>" in branch_then_only_text, "br.then_ body should lower into the scf.if then branch")
    expect("}, {" not in branch_then_only_text, "then-only branch handle should not materialize an else region")

    branch_side_effect_text = branch_handle_side_effect_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(branch_side_effect_text, "branch handle side-effect specialization")
    expect(branch_side_effect_text.count("scf.if") == 1, "side-effect branch handle should lower to one scf.if")
    expect("pto.barrier <PIPE_ALL>" in branch_side_effect_text, "br.then_ side effects should lower into the then branch")
    expect("pto.mem_bar" in branch_side_effect_text, "br.else_ side effects should lower into the else branch")
    expect("} else {" in branch_side_effect_text, "side-effect branch handle should materialize an explicit else region")

    branch_merge_text = branch_handle_merge_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(branch_merge_text, "branch handle merge specialization")
    expect(
        re.search(r"scf\.if %\d+ -> \(i32, i32\)", branch_merge_text) is not None,
        "br.assign(...) should infer scf.if result types from the assigned branch values",
    )
    expect(
        branch_merge_text.count("scf.yield") >= 2,
        "automatic branch merge should materialize one internal scf.yield per branch",
    )
    expect(
        "arith.addi" in branch_merge_text and "arith.subi" in branch_merge_text,
        "merged branch values should remain usable as ordinary runtime scalars after the conditional",
    )

    ast_if_side_effect_text = ast_if_side_effect_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(ast_if_side_effect_text, "AST-rewritten side-effect if specialization")
    expect(
        ast_if_side_effect_text.count("scf.if") == 1,
        "ast_rewrite=True Python if should lower to one scf.if for runtime conditions",
    )
    expect(
        "pto.barrier <PIPE_ALL>" in ast_if_side_effect_text,
        "ast_rewrite=True if body should lower into the scf.if then branch",
    )

    ast_if_merge_text = ast_if_merge_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(ast_if_merge_text, "AST-rewritten if-merge specialization")
    expect(
        re.search(r"scf\.if %\d+ -> \(i32, i32\)", ast_if_merge_text) is not None,
        "ast_rewrite=True Python if/else live-outs should lower to scf.if results",
    )
    expect(
        ast_if_merge_text.count("scf.yield") >= 2,
        "ast_rewrite=True Python if/else live-outs should materialize branch yields",
    )

    ast_if_old_value_merge_text = ast_if_old_value_merge_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(ast_if_old_value_merge_text, "AST-rewritten old-value if-merge specialization")
    expect(
        re.search(r"scf\.if %\d+ -> \(i32\)", ast_if_old_value_merge_text) is not None,
        "ast_rewrite=True one-sided Python if assignment should merge the old value through scf.if",
    )
    expect(
        ast_if_old_value_merge_text.count("scf.yield") >= 2,
        "ast_rewrite=True old-value if merge should yield from both branches",
    )

    ast_if_static_subscript_slot_merge_text = ast_if_static_subscript_slot_merge_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(
        ast_if_static_subscript_slot_merge_text,
        "AST-rewritten static subscript slot if-merge specialization",
    )
    expect(
        ast_if_static_subscript_slot_merge_text.count("scf.if") == 2,
        "AST-rewritten static subscript slot branch merge should preserve both runtime conditionals",
    )
    expect(
        "-> (i32)" in ast_if_static_subscript_slot_merge_text,
        "AST-rewritten static subscript slot merge should return the merged slot value",
    )
    expect(
        "pto.barrier <PIPE_ALL>" in ast_if_static_subscript_slot_merge_text,
        "static subscript slot values should remain usable after the branch merge",
    )

    ast_if_static_subscript_expression_merge_text = (
        ast_if_static_subscript_expression_merge_probe.compile(SLOT=1).mlir_text()
    )
    expect_parse_roundtrip_and_verify(
        ast_if_static_subscript_expression_merge_text,
        "AST-rewritten static subscript expression if-merge specialization",
    )
    expect(
        ast_if_static_subscript_expression_merge_text.count("scf.if") == 2,
        "static-expression and constexpr subscript indices should participate in branch merging",
    )

    ast_if_static_subscript_negative_index_merge_text = (
        ast_if_static_subscript_negative_index_merge_probe.compile().mlir_text()
    )
    expect_parse_roundtrip_and_verify(
        ast_if_static_subscript_negative_index_merge_text,
        "AST-rewritten negative static subscript if-merge specialization",
    )
    expect(
        ast_if_static_subscript_negative_index_merge_text.count("scf.if") == 2,
        "negative static subscript indices should participate in branch merging",
    )

    ast_if_branch_local_temp_liveness_text = ast_if_branch_local_temp_liveness_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(
        ast_if_branch_local_temp_liveness_text,
        "AST-rewritten if branch-local temp liveness specialization",
    )
    expect(
        "old_tmp" not in ast_if_branch_local_temp_liveness_text,
        "branch-local temporaries should not be merged as old live-out values",
    )

    ast_nested_with_if_merge_text = ast_nested_with_if_merge_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(ast_nested_with_if_merge_text, "AST-rewritten nested with if-merge specialization")
    expect(
        re.search(r"scf\.if %\d+ -> \(i32\)", ast_nested_with_if_merge_text) is not None,
        "ast_rewrite=True nested with block should preserve outer live-out liveness for if merge",
    )

    ast_runtime_for_text = ast_runtime_for_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(ast_runtime_for_text, "AST-rewritten runtime for specialization")
    expect(
        ast_runtime_for_text.count("scf.for") == 1,
        "ast_rewrite=True Python range(...) should lower to one scf.for",
    )

    ast_nested_runtime_for_induction_scope_text = ast_nested_runtime_for_induction_scope_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(
        ast_nested_runtime_for_induction_scope_text,
        "AST-rewritten nested runtime induction scope specialization",
    )
    expect(
        ast_nested_runtime_for_induction_scope_text.count("scf.for") == 2,
        "nested runtime loop induction variables should remain local to their loops",
    )

    ast_nested_runtime_for_local_temp_scope_text = ast_nested_runtime_for_local_temp_scope_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(
        ast_nested_runtime_for_local_temp_scope_text,
        "AST-rewritten nested runtime local temp scope specialization",
    )
    expect(
        ast_nested_runtime_for_local_temp_scope_text.count("scf.for") == 3,
        "nested runtime loop temporaries consumed in-loop should not become live-outs",
    )

    ast_runtime_for_carry_text = ast_runtime_for_carry_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(ast_runtime_for_carry_text, "AST-rewritten runtime carry for specialization")
    expect(
        "iter_args(" in ast_runtime_for_carry_text and "scf.yield" in ast_runtime_for_carry_text,
        "ast_rewrite=True accumulator loops should lower through scf.for iter_args",
    )

    ast_runtime_for_augassign_carry_text = ast_runtime_for_augassign_carry_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(
        ast_runtime_for_augassign_carry_text,
        "AST-rewritten augassign runtime carry for specialization",
    )
    expect(
        "iter_args(" in ast_runtime_for_augassign_carry_text and "scf.yield" in ast_runtime_for_augassign_carry_text,
        "ast_rewrite=True accumulator loops using += should lower through scf.for iter_args",
    )

    ast_runtime_for_branch_local_temp_text = ast_runtime_for_branch_local_temp_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(
        ast_runtime_for_branch_local_temp_text,
        "AST-rewritten runtime for branch-local temp specialization",
    )
    expect(
        "iter_args(" not in ast_runtime_for_branch_local_temp_text,
        "branch-local temporaries should not be inferred as loop-carried state",
    )

    ast_runtime_ifexp_assign_text = ast_runtime_ifexp_assign_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(
        ast_runtime_ifexp_assign_text,
        "AST-rewritten runtime IfExp assignment specialization",
    )
    expect(
        ast_runtime_ifexp_assign_text.count("scf.for") == 1,
        "assign-form Python conditional expressions inside runtime loops should preserve the runtime loop",
    )
    expect(
        ast_runtime_ifexp_assign_text.count("scf.if") == 1,
        "assign-form Python conditional expressions should normalize through the existing AST if rewrite",
    )

    ast_runtime_ifexp_python_literal_assign_text = (
        ast_runtime_ifexp_python_literal_assign_probe.compile().mlir_text()
    )
    expect_parse_roundtrip_and_verify(
        ast_runtime_ifexp_python_literal_assign_text,
        "AST-rewritten runtime IfExp assignment should materialize opposite-branch Python literals",
    )
    expect(
        ast_runtime_ifexp_python_literal_assign_text.count("scf.for") == 1,
        "assign-form Python conditional expressions with opposite-branch literals should preserve the runtime loop",
    )
    expect(
        ast_runtime_ifexp_python_literal_assign_text.count("scf.if") == 1,
        "assign-form Python conditional expressions with opposite-branch literals should normalize through the existing AST if rewrite",
    )

    ast_runtime_for_sibling_iv_reuse_text = ast_runtime_for_sibling_iv_reuse_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(
        ast_runtime_for_sibling_iv_reuse_text,
        "AST-rewritten sibling runtime for IV reuse specialization",
    )
    expect(
        ast_runtime_for_sibling_iv_reuse_text.count("scf.for") >= 3,
        "reusing a sibling runtime loop IV name should not be misdiagnosed as loop-target live-out",
    )

    ast_runtime_for_static_range_name_reuse_text = ast_runtime_for_static_range_name_reuse_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(
        ast_runtime_for_static_range_name_reuse_text,
        "AST-rewritten runtime for static_range/list-comprehension name reuse specialization",
    )
    expect(
        ast_runtime_for_static_range_name_reuse_text.count("scf.for") == 1,
        "static_range and comprehension-local names should not be inferred as outer runtime loop carry state",
    )

    ast_runtime_for_static_slot_carry_text = ast_runtime_for_static_slot_carry_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(
        ast_runtime_for_static_slot_carry_text,
        "AST-rewritten runtime for static subscript slot carry specialization",
    )
    expect(
        ast_runtime_for_static_slot_carry_text.count("scf.for") == 1,
        "static subscript slot carry should preserve the authored runtime loop",
    )
    expect(
        "iter_args(" in ast_runtime_for_static_slot_carry_text
        and "scf.yield" in ast_runtime_for_static_slot_carry_text,
        "static subscript slot carry should lower through scf.for iter_args",
    )
    ast_runtime_for_static_slot_nested_vecscope_text = (
        ast_runtime_for_static_slot_nested_vecscope_probe.compile().mlir_text()
    )
    expect_parse_roundtrip_and_verify(
        ast_runtime_for_static_slot_nested_vecscope_text,
        "AST-rewritten nested vecscope runtime for static subscript slots",
    )
    expect(
        ast_runtime_for_static_slot_nested_vecscope_text.count("scf.for") == 2,
        "nested vecscope static subscript slots should preserve both authored runtime loops",
    )
    ast_runtime_for_static_slot_global_index_text = ast_runtime_for_static_slot_global_index_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(
        ast_runtime_for_static_slot_global_index_text,
        "AST-rewritten runtime for global static subscript slot carry specialization",
    )
    expect(
        ast_runtime_for_static_slot_global_index_text.count("scf.for") == 1,
        "module-global static slot indices should lower through the authored runtime loop",
    )
    ast_runtime_for_static_slot_binop_index_text = ast_runtime_for_static_slot_binop_index_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(
        ast_runtime_for_static_slot_binop_index_text,
        "AST-rewritten runtime for BinOp static subscript slot carry specialization",
    )
    expect(
        ast_runtime_for_static_slot_binop_index_text.count("scf.for") == 1,
        "BinOp static slot indices should lower through the authored runtime loop",
    )
    ast_runtime_for_static_slot_constexpr_index_text = (
        ast_runtime_for_static_slot_constexpr_index_probe.compile(SLOT=2).mlir_text()
    )
    expect_parse_roundtrip_and_verify(
        ast_runtime_for_static_slot_constexpr_index_text,
        "AST-rewritten runtime for constexpr static subscript slot carry specialization",
    )
    expect(
        ast_runtime_for_static_slot_constexpr_index_text.count("scf.for") == 1,
        "constexpr static slot indices should lower through the authored runtime loop",
    )
    expect_raises(
        PTODSLAstRewriteError,
        lambda: ast_runtime_for_dynamic_slot_store_error_probe.compile(),
        "simple_name[static_int_or_static_range_iv]",
    )
    expect_raises(
        PTODSLAstRewriteError,
        lambda: ast_runtime_for_complex_slot_store_error_probe.compile(),
        "simple_name[static_int_or_static_range_iv]",
    )

    ast_rewrite_disabled_nested_helper_python_control_text = (
        ast_rewrite_disabled_nested_helper_python_control_probe.compile().mlir_text()
    )
    expect_parse_roundtrip_and_verify(
        ast_rewrite_disabled_nested_helper_python_control_text,
        "AST rewrite disabled nested Python helper control specialization",
    )
    expect(
        "scf.if" not in ast_rewrite_disabled_nested_helper_python_control_text
        and "scf.for" not in ast_rewrite_disabled_nested_helper_python_control_text,
        "ast_rewrite=False should keep nested helper function bodies as trace-time Python",
    )
    expect(
        ast_rewrite_disabled_nested_helper_python_control_text.count("pto.barrier <PIPE_ALL>") == 2,
        "nested Python helper should keep trace-time if/range behavior",
    )

    ast_nested_helper_ast_rewrite_text = ast_nested_helper_ast_rewrite_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(
        ast_nested_helper_ast_rewrite_text,
        "AST-rewritten nested helper function specialization",
    )
    expect(
        ast_nested_helper_ast_rewrite_text.count("scf.for") == 1,
        "default AST rewrite should rewrite range(...) loops inside nested helpers",
    )
    expect(
        ast_nested_helper_ast_rewrite_text.count("scf.if") == 1,
        "default AST rewrite should rewrite runtime if statements inside nested helpers",
    )
    expect(
        "iter_args(" in ast_nested_helper_ast_rewrite_text and "scf.yield" in ast_nested_helper_ast_rewrite_text,
        "rewritten nested helpers should preserve loop-carried and branch live-out values",
    )

    ptodsl_func_call_text = ptodsl_func_call_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(ptodsl_func_call_text, "@pto.func helper call specialization")
    expect(
        re.search(r"func\.func @func_runtime_for_return_helper__ptodsl_[0-9a-f]+\(.*\) -> i32", ptodsl_func_call_text)
        is not None,
        "@pto.func helpers that return one runtime value should materialize a typed helper result",
    )
    expect(
        re.search(r"func\.func @func_multi_return_helper__ptodsl_[0-9a-f]+\(.*\) -> \(i32, i32\)", ptodsl_func_call_text)
        is not None,
        "@pto.func helpers should support multiple returned runtime values",
    )
    expect(
        ptodsl_func_call_text.count("scf.for") >= 1 and ptodsl_func_call_text.count("scf.if") >= 1,
        "@pto.func helper bodies should use native control-flow AST rewrite",
    )
    expect(
        len(re.findall(r"func\.func @func_void_helper__ptodsl_[0-9a-f]+", ptodsl_func_call_text)) == 1
        and len(re.findall(r"call @func_void_helper__ptodsl_[0-9a-f]+", ptodsl_func_call_text)) == 2,
        "repeated @pto.func calls should reuse one materialized helper artifact",
    )
    ptodsl_func_partition_metadata_text = ptodsl_func_partition_metadata_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(
        ptodsl_func_partition_metadata_text,
        "@pto.func partition metadata specialization",
    )
    expect(
        re.search(r"func\.func @func_partition_metadata_helper__ptodsl_[0-9a-f]+", ptodsl_func_partition_metadata_text)
        is not None
        and re.search(r"func\.func @func_partition_metadata_helper__ptodsl_[0-9a-f]+\(.*\) -> i32", ptodsl_func_partition_metadata_text)
        is not None
        and re.search(r"%c1_i32 = arith\.constant 1 : i32", ptodsl_func_partition_metadata_text) is not None,
        "@pto.func should preserve partition metadata across the helper boundary",
    )
    ptodsl_func_future_annotations_text = ptodsl_func_future_annotations_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(
        ptodsl_func_future_annotations_text,
        "@pto.func future annotations specialization",
    )
    expect(
        re.search(
            r"func\.func @func_future_i32_return_helper__ptodsl_[0-9a-f]+\(.*i32.*\) -> i32",
            ptodsl_func_future_annotations_text,
        )
        is not None,
        "@pto.func should resolve PEP 563 string return annotations to PTO result types",
    )
    expect(
        re.search(
            r"func\.func @func_future_i64_literal_helper__ptodsl_[0-9a-f]+\(.*i64.*\) -> i64",
            ptodsl_func_future_annotations_text,
        )
        is not None,
        "@pto.func should resolve PEP 563 string parameter annotations before materializing literals",
    )
    expect(
        re.search(
            r"func\.func @func_future_void_helper__ptodsl_[0-9a-f]+\(.*i32.*\) attributes",
            ptodsl_func_future_annotations_text,
        )
        is not None,
        "@pto.func should resolve PEP 563 -> None annotations as void helpers",
    )
    i32_cache_signature = cache_signature_atom(pto.i32)
    i64_cache_signature = cache_signature_atom(pto.i64)
    ptr_cache_signature = cache_signature_atom(pto.ptr(pto.f32, "gm"))
    expect(
        i32_cache_signature != i64_cache_signature,
        "dtype cache signatures should distinguish different MLIR scalar types",
    )
    expect(
        "0x" not in repr(i32_cache_signature)
        and "0x" not in repr(ptr_cache_signature)
        and "function" not in repr(i32_cache_signature),
        "dtype cache signatures should not include Python factory reprs or memory addresses",
    )
    first_stable_symbol = ptodsl_func_stable_symbol_from_subprocess()
    second_stable_symbol = ptodsl_func_stable_symbol_from_subprocess()
    expect(
        first_stable_symbol == second_stable_symbol,
        "@pto.func helper symbols should stay stable across Python processes",
    )
    ptodsl_func_constexpr_text = ptodsl_func_constexpr_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(
        ptodsl_func_constexpr_text,
        "@pto.func const_expr specialization",
    )
    constexpr_helper_names = re.findall(
        r"func\.func @(func_constexpr_static_helper__ptodsl_[0-9a-f]+)\(%arg0: i32\) -> i32",
        ptodsl_func_constexpr_text,
    )
    expect(
        len(set(constexpr_helper_names)) == 2,
        "@pto.func const_expr parameters should specialize helpers without entering the runtime ABI",
    )
    expect(
        ptodsl_func_constexpr_text.count("call @func_constexpr_static_helper__ptodsl_") == 2
        and ptodsl_func_constexpr_text.count("arith.addi") >= 6,
        "@pto.func const_expr values should remain available for static_range unrolling",
    )
    expect_raises(
        TypeError,
        lambda: ptodsl_func_constexpr_runtime_value_probe.compile().mlir_text(),
        "const_expr parameter 'BLOCK' expects a compile-time Python value",
    )
    ptodsl_func_traced_argument_coercion_text = (
        ptodsl_func_traced_argument_coercion_probe.compile().mlir_text()
    )
    expect_parse_roundtrip_and_verify(
        ptodsl_func_traced_argument_coercion_text,
        "@pto.func traced argument annotation coercion",
    )
    expect(
        "arith.trunci" in ptodsl_func_traced_argument_coercion_text
        and re.search(
            r"func\.func @func_i32_argument_helper__ptodsl_[0-9a-f]+\(%arg0: i32\)",
            ptodsl_func_traced_argument_coercion_text,
        )
        is not None,
        "@pto.func should adapt traced SSA arguments to their declared parameter type",
    )
    expect_raises(
        TypeError,
        lambda: ptodsl_func_traced_argument_type_error_probe.compile().mlir_text(),
        "@pto.func parameter 'value' cannot coerce",
    )
    expect_raises(
        TypeError,
        lambda: ptodsl_func_runtime_closure_capture_probe.compile().mlir_text(),
        "captures runtime value 'value'",
    )
    expect_raises(
        PTODSLAstRewriteError,
        lambda: ptodsl_func_if_early_return_probe.compile().mlir_text(),
        "return/yield inside rewritten if branches",
    )
    expect_raises(
        PTODSLAstRewriteError,
        lambda: ptodsl_func_for_early_return_probe.compile().mlir_text(),
        "return/yield inside rewritten for-loop bodies",
    )
    expect_raises(
        PTODSLAstRewriteError,
        lambda: ptodsl_func_if_yield_probe.compile().mlir_text(),
        "return/yield inside rewritten if branches",
    )
    expect_raises(
        TypeError,
        lambda: pto.func(lambda value: value),
        "must explicitly declare return types",
    )

    ast_nested_helper_freevar_if_merge_text = ast_nested_helper_freevar_if_merge_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(
        ast_nested_helper_freevar_if_merge_text,
        "AST-rewritten nested helper freevar if-merge specialization",
    )
    expect(
        re.search(r"scf\.if %\d+ -> \(i32\)", ast_nested_helper_freevar_if_merge_text) is not None,
        "nested helper free variables should keep outer branch assignments live for SSA merge",
    )

    ast_nested_helper_name_store_liveness_text = ast_nested_helper_name_store_liveness_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(
        ast_nested_helper_name_store_liveness_text,
        "AST-rewritten nested helper name-store liveness specialization",
    )
    expect(
        re.search(r"scf\.if %\d+ ->", ast_nested_helper_name_store_liveness_text) is None,
        "nested function definitions should store their function name and kill earlier liveness",
    )

    ast_control_flow_equiv_explicit_text = ast_control_flow_equiv_explicit_probe.compile().mlir_text()
    ast_control_flow_equiv_native_text = ast_control_flow_equiv_native_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(
        ast_control_flow_equiv_explicit_text,
        "explicit control-flow integration specialization",
    )
    expect_parse_roundtrip_and_verify(
        ast_control_flow_equiv_native_text,
        "native AST-rewritten control-flow integration specialization",
    )
    expect(
        mlir_op_sequence(ast_control_flow_equiv_native_text) == mlir_op_sequence(ast_control_flow_equiv_explicit_text),
        "native Python for/if rewrite should emit the same operation sequence as explicit pto.for_/pto.if_",
    )
    for pattern in ("scf.for", "scf.if", "iter_args(", "scf.yield", "arith.addi"):
        expect(
            ast_control_flow_equiv_native_text.count(pattern) == ast_control_flow_equiv_explicit_text.count(pattern),
            f"native AST rewrite should match explicit control-flow count for {pattern}",
        )

    ast_closure_kernel_text = ast_closure_kernel_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(ast_closure_kernel_text, "AST-rewritten closure kernel specialization")
    expect(
        ast_closure_kernel_text.count("pto.barrier <PIPE_ALL>") == 3,
        "ast_rewrite=True factory kernels should preserve captured Python closure values",
    )

    ast_rebound_closure_kernel_text = ast_rebound_closure_kernel_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(
        ast_rebound_closure_kernel_text,
        "AST-rewritten rebound closure kernel specialization",
    )
    expect(
        ast_rebound_closure_kernel_text.count("pto.barrier <PIPE_ALL>") == 4,
        "ast_rewrite=True should read nonlocal closure values at compile time",
    )

    ast_mutable_closure_cache_first = ast_mutable_closure_cache_kernel_probe.compile()
    ast_mutable_closure_cache_first_text = ast_mutable_closure_cache_first.mlir_text()
    expect_parse_roundtrip_and_verify(
        ast_mutable_closure_cache_first_text,
        "AST-rewritten mutable closure cache first specialization",
    )
    expect(
        ast_mutable_closure_cache_first_text.count("pto.barrier <PIPE_ALL>") == 2,
        "first mutable closure specialization should use the initial nonlocal value",
    )
    set_ast_mutable_closure_cache_limit(4)
    ast_mutable_closure_cache_second = ast_mutable_closure_cache_kernel_probe.compile()
    ast_mutable_closure_cache_second_text = ast_mutable_closure_cache_second.mlir_text()
    expect_parse_roundtrip_and_verify(
        ast_mutable_closure_cache_second_text,
        "AST-rewritten mutable closure cache second specialization",
    )
    expect(
        ast_mutable_closure_cache_second is not ast_mutable_closure_cache_first,
        "AST rewrite closure changes should participate in the specialization cache key",
    )
    expect(
        ast_mutable_closure_cache_second_text.count("pto.barrier <PIPE_ALL>") == 2,
        "changed mutable closure specialization should keep the function's captured nonlocal value stable",
    )

    ast_signature_closure_default_text = ast_signature_closure_default_kernel_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(
        ast_signature_closure_default_text,
        "AST-rewritten signature closure default specialization",
    )
    expect(
        ast_signature_closure_default_text.count("pto.barrier <PIPE_ALL>") == 2,
        "ast_rewrite=True should resolve closure names used by signature defaults",
    )

    ast_rebound_subkernel_text = ast_rebound_subkernel_probe.compile(TRACE_TOKEN=1).mlir_text()
    expect_parse_roundtrip_and_verify(
        ast_rebound_subkernel_text,
        "AST-rewritten rebound subkernel specialization",
    )
    expect(
        ast_rebound_subkernel_text.count("pto.simt_launch @tileop_noop_simt_probe__simt_") == 4,
        "named subkernels should read nonlocal closure values when traced",
    )

    sourceless_ast_rewrite_text = sourceless_ast_rewrite_kernel_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(
        sourceless_ast_rewrite_text,
        "source-less AST rewrite fallback specialization",
    )
    expect(
        sourceless_ast_rewrite_text.count("pto.barrier <PIPE_ALL>") == 1,
        "source-less kernels should fall back to original trace-time Python execution",
    )

    sourceless_subkernel_text = sourceless_subkernel_entry_probe.compile(TRACE_TOKEN=1).mlir_text()
    expect_parse_roundtrip_and_verify(
        sourceless_subkernel_text,
        "source-less subkernel AST rewrite fallback specialization",
    )
    expect(
        sourceless_subkernel_text.count("pto.simt_launch @tileop_noop_simt_probe__simt_") == 1,
        "source-less subkernels should fall back to original trace-time Python execution",
    )

    sourceless_ptodsl_func_text = sourceless_ptodsl_func_probe.compile(TRACE_TOKEN=1).mlir_text()
    expect_parse_roundtrip_and_verify(
        sourceless_ptodsl_func_text,
        "source-less @pto.func AST rewrite fallback specialization",
    )
    expect(
        sourceless_ptodsl_func_text.count("pto.barrier <PIPE_ALL>") == 1,
        "source-less @pto.func helpers should fall back to original trace-time Python execution",
    )

    ast_python_bool_guard_enabled_text = ast_python_bool_guard_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(
        ast_python_bool_guard_enabled_text,
        "AST Python bool guard enabled specialization",
    )
    expect(
        "scf.if" not in ast_python_bool_guard_enabled_text,
        "Python bool guards should remain trace-time branches under ast_rewrite=True",
    )
    expect(
        ast_python_bool_guard_enabled_text.count("pto.barrier <PIPE_ALL>") == 2,
        "Python bool guards should execute enabled trace-time branches",
    )
    ast_python_bool_guard_disabled_text = ast_python_bool_guard_probe.compile(
        BLOCK=64,
        ENABLE=False,
    ).mlir_text()
    expect_parse_roundtrip_and_verify(
        ast_python_bool_guard_disabled_text,
        "AST Python bool guard disabled specialization",
    )
    expect(
        "scf.if" not in ast_python_bool_guard_disabled_text,
        "disabled Python bool guards should not lower to runtime branches",
    )
    expect(
        "pto.barrier <PIPE_ALL>" not in ast_python_bool_guard_disabled_text,
        "disabled Python bool guards should skip their trace-time branches",
    )

    ast_static_enabled_text = ast_static_control_flow_probe.compile(ENABLE=True).mlir_text()
    expect_parse_roundtrip_and_verify(ast_static_enabled_text, "AST static control-flow enabled specialization")
    expect(
        "scf.if" not in ast_static_enabled_text and "scf.for" not in ast_static_enabled_text,
        "pto.const_expr/pto.static_range should keep compile-time control flow under ast_rewrite=True",
    )
    expect(
        ast_static_enabled_text.count("pto.barrier <PIPE_ALL>") == 2,
        "pto.static_range(2) should unroll at trace time under ast_rewrite=True",
    )
    ast_static_range_loop_target_live_after_text = ast_static_range_loop_target_live_after_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(
        ast_static_range_loop_target_live_after_text,
        "AST static_range loop target live-after specialization",
    )
    expect(
        "scf.for" not in ast_static_range_loop_target_live_after_text,
        "pto.static_range(...) should keep Python loop semantics under ast_rewrite=True",
    )
    expect(
        ast_static_range_loop_target_live_after_text.count("pto.barrier <PIPE_ALL>") == 3,
        "pto.static_range(...) should allow the Python loop target to remain live after unrolling",
    )
    ast_static_range_break_continue_text = ast_static_range_break_continue_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(
        ast_static_range_break_continue_text,
        "AST static_range break/continue specialization",
    )
    expect(
        "scf.for" not in ast_static_range_break_continue_text,
        "pto.static_range(...) break/continue should stay in trace-time Python control flow",
    )
    expect(
        ast_static_range_break_continue_text.count("pto.barrier <PIPE_ALL>") == 4,
        "pto.static_range(...) should preserve Python break/continue behavior under ast_rewrite=True",
    )
    ast_static_disabled_text = ast_static_control_flow_probe.compile(ENABLE=False).mlir_text()
    expect(
        "pto.barrier <PIPE_ALL>" not in ast_static_disabled_text,
        "pto.const_expr(False) should skip the trace-time branch under ast_rewrite=True",
    )

    runtime_scalar_text = runtime_scalar_operator_probe.compile(BLOCK=8).mlir_text()
    expect_parse_roundtrip_and_verify(runtime_scalar_text, "runtime scalar operator specialization")
    expect("arith.index_cast" in runtime_scalar_text, "mixed i64/index runtime arithmetic should materialize index_cast")
    expect("arith.floordivsi" in runtime_scalar_text, "runtime // should lower to arith.floordivsi")
    expect("arith.remsi" in runtime_scalar_text, "runtime % should lower to arith.remsi")
    expect("arith.addf" in runtime_scalar_text, "runtime float + should lower to arith.addf")
    expect("arith.mulf" in runtime_scalar_text, "runtime float * should lower to arith.mulf")
    expect("arith.subf" in runtime_scalar_text, "runtime float - should lower to arith.subf")
    expect("arith.divf" in runtime_scalar_text, "runtime float / should lower to arith.divf")
    expect("arith.maximumf" in runtime_scalar_text, "scalar.max(float, float) should lower to arith.maximumf")
    expect("arith.minimumf" in runtime_scalar_text, "scalar.min(float, float) should lower to arith.minimumf")
    expect("math.exp" in runtime_scalar_text, "scalar.exp(...) should lower to math.exp")
    expect("math.log" in runtime_scalar_text, "scalar.log(...) should lower to math.log")
    expect("math.sqrt" in runtime_scalar_text, "scalar.sqrt(...) should lower to math.sqrt")
    expect("math.absf" in runtime_scalar_text, "scalar.abs(float) should lower to math.absf")
    expect("arith.cmpf ogt" in runtime_scalar_text, "float runtime '>' should lower to arith.cmpf ogt")
    expect("arith.cmpf oeq" in runtime_scalar_text, "float runtime '==' should lower to arith.cmpf oeq")
    expect("arith.cmpf oge" in runtime_scalar_text, "float runtime '>=' should lower to arith.cmpf oge")
    expect("arith.cmpf ole" in runtime_scalar_text, "float runtime '<=' should lower to arith.cmpf ole")
    expect("arith.andi" in runtime_scalar_text, "i1 conjunction from native '&' should lower to arith.andi")

    host_runtime_scalar_entry_text = host_runtime_scalar_entry_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(
        host_runtime_scalar_entry_text,
        "host runtime scalar entry specialization",
    )
    expect(
        "func.func @host_runtime_scalar_entry_probe" in host_runtime_scalar_entry_text,
        "host runtime scalar entry probe should compile into a launchable kernel",
    )
    expect(
        host_runtime_scalar_entry_probe.compile() is host_runtime_scalar_entry_probe.compile(),
        "kernels without constexpr parameters should still reuse one compiled specialization",
    )
    expect_raises(
        TypeError,
        lambda: host_runtime_scalar_entry_probe.compile(limit=4),
        "unknown @pto.jit constexpr parameter(s): limit",
    )
    expect_raises(
        TypeError,
        lambda: host_runtime_scalar_entry_probe.compile(alpha=1.0),
        "unknown @pto.jit constexpr parameter(s): alpha",
    )
    expect(
        "i32" in host_runtime_scalar_entry_text and "f32" in host_runtime_scalar_entry_text,
        "host runtime scalar entry probe should preserve scalar ABI argument types in MLIR",
    )

    signed_integer_scalar_text = signed_integer_scalar_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(signed_integer_scalar_text, "signed integer scalar specialization")
    expect(
        "builtin.unrealized_conversion_cast" in signed_integer_scalar_text,
        "signed/unsigned scalar lowering should bridge signless arith values through unrealized_conversion_cast",
    )
    expect("arith.addi" in signed_integer_scalar_text, "signed/unsigned scalar addition should lower through arith.addi")
    expect("arith.maxsi" in signed_integer_scalar_text, "signed scalar max should lower through arith.maxsi")
    expect("arith.minsi" in signed_integer_scalar_text, "signed scalar min should lower through arith.minsi")
    expect("arith.maxui" in signed_integer_scalar_text, "unsigned scalar max should lower through arith.maxui")
    expect("arith.minui" in signed_integer_scalar_text, "unsigned scalar min should lower through arith.minui")
    expect("math.absi" in signed_integer_scalar_text, "signed scalar abs should lower through math.absi")
    expect("arith.cmpi sgt" in signed_integer_scalar_text, "signed scalar cmp should preserve signed predicate")
    expect("arith.cmpi ugt" in signed_integer_scalar_text, "unsigned scalar cmp should preserve unsigned predicate")
    expect("pto.store" in runtime_scalar_text, "scalar.store(...) should lower to pto.store")

    tile_slice_text = tile_slice_surface_probe.compile(BLOCK=128).mlir_text()
    expect_parse_roundtrip_and_verify(tile_slice_text, "tile slice surface specialization")
    expect("memref.subview" not in tile_slice_text, "tile[row, col:] should no longer lower through memref.subview")
    expect("pto.tile_buf_addr" in tile_slice_text, "tile[row, col:] should materialize a tile address pointer")
    expect(
        "pto.vlds" in tile_slice_text and "!pto.ptr<f32, ub>" in tile_slice_text,
        "vlds(tile[row, col:]) should lower against the pointer slice view",
    )
    expect(
        "pto.vsts" in tile_slice_text and "!pto.ptr<f32, ub>" in tile_slice_text,
        "vsts(vec, tile[row, col:], mask) should lower against the pointer slice view",
    )

    tile_slice_1d_text = tile_slice_1d_surface_probe.compile(BLOCK=128).mlir_text()
    expect_parse_roundtrip_and_verify(tile_slice_1d_text, "1D tile slice surface specialization")
    expect("memref.subview" not in tile_slice_1d_text, "tile[start:] should lower without memref.subview")
    expect("pto.vldas" in tile_slice_1d_text, "vldas(tile[start:]) should lower against the 1D slice view")
    expect("pto.vldus" in tile_slice_1d_text, "vldus(tile[start:], align) should lower against the 1D slice view")
    expect("pto.vsts" in tile_slice_1d_text, "vsts(vec, tile[start:], mask) should lower against the 1D slice view")

    integer_loop_text = integer_loop_bound_probe.compile(BLOCK=8).mlir_text()
    expect_parse_roundtrip_and_verify(integer_loop_text, "integer loop bound specialization")
    expect(
        integer_loop_text.count("arith.index_cast") >= 2,
        "integer runtime loop bounds should be normalized to index with arith.index_cast",
    )
    expect(
        integer_loop_text.count("scf.for") == 2,
        "integer loop bound probe should still lower nested authored loops to scf.for",
    )

    scalar_pointer_offset_text = scalar_pointer_offset_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(scalar_pointer_offset_text, "scalar pointer offset specialization")
    expect(
        re.search(r"pto\.store %c1_i32, %\d+\[%c1\]", scalar_pointer_offset_text) is not None,
        "scalar.store(ptr, 1) should lower as element offset 1",
    )
    expect(
        re.search(r"pto\.store %c2_i32, %\d+\[%c2\]", scalar_pointer_offset_text) is not None,
        "scalar.store(ptr + 2) should lower as element offset 2",
    )
    expect(
        re.search(r"pto\.load %\d+\[%c1(?:_\d+)?\]", scalar_pointer_offset_text) is not None,
        "scalar.load(ptr, 1) should lower as element offset 1",
    )
    expect(
        re.search(r"pto\.load %\d+\[%c2(?:_\d+)?\]", scalar_pointer_offset_text) is not None,
        "scalar.load(ptr + 2) should lower as element offset 2",
    )

    scalar_contiguous_text = scalar_contiguous_vector_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(scalar_contiguous_text, "scalar contiguous vector specialization")
    expect("llvm.load" in scalar_contiguous_text, "scalar.load(..., contiguous=N) should lower to llvm.load")
    expect("llvm.store" in scalar_contiguous_text, "scalar.store(vector, ...) should lower to llvm.store")
    expect("vector<4xf32>" in scalar_contiguous_text, "contiguous=4 over f32 should produce vector<4xf32>")
    expect("llvm.insertelement" in scalar_contiguous_text, "pto.Vec(..., init=scalar) should broadcast with insertelement")
    expect("arith.mulf" in scalar_contiguous_text, "VecValue multiplication should lower to arith.mulf")
    expect(
        "pto.load" not in scalar_contiguous_text and "pto.store" not in scalar_contiguous_text,
        "contiguous vector memory access should not lower through scalar pto.load/store",
    )
    expect_raises(
        ValueError,
        lambda: scalar_contiguous_width_mismatch_probe.compile(),
        "does not match vector size",
    )
    expect_raises(
        TypeError,
        lambda: scalar_contiguous_scalar_store_probe.compile(),
        "scalar.store(scalar, ..., contiguous=N) is not supported",
    )

    per_element_vec_text = per_element_vec_f32_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(per_element_vec_text, "per-element Vec init= specialization")
    expect("vector<2xf32>" in per_element_vec_text, "pto.Vec(..., init=...) over f32 should produce vector<2xf32>")
    # NOTE: the exact `count("llvm.insertelement" / "llvm.store") == N` assertions in this
    # section are white-box snapshot checks of the traced MLIR. Unrelated pipeline changes
    # (canonicalization, constant folding of the undef/insertelement chain, etc.) can shift
    # the counts without any behavioral regression; update the expected numbers when that
    # happens. The behavioral guards are the distinct-SSA-value check below, the absence of
    # numeric conversion ops, and the dsl-st golden cases on A5.
    expect(
        per_element_vec_text.count("llvm.insertelement") == 2,
        "pto.Vec(..., init=...) should emit exactly two llvm.insertelement ops",
    )
    inserted_values = re.findall(
        r"llvm\.insertelement\s+(%[\w.\-]+)",
        per_element_vec_text,
    )
    expect(
        len(inserted_values) == 2 and inserted_values[0] != inserted_values[1],
        "pto.Vec(..., init=(v0, v1)) should insert two distinct SSA values, not broadcast",
    )
    expect(
        per_element_vec_text.count("llvm.store") == 1,
        "scalar.store(vector, ...) built from init= should emit exactly one llvm.store",
    )
    expect(
        "pto.store" not in per_element_vec_text,
        "per-element Vec store should not lower through scalar pto.store",
    )

    per_element_vec_i32_text = per_element_vec_i32_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(per_element_vec_i32_text, "per-element Vec init= i32 specialization")
    per_element_vec_ui32_text = per_element_vec_ui32_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(per_element_vec_ui32_text, "per-element Vec init= ui32 specialization")
    for int_label, int_text in (("i32", per_element_vec_i32_text), ("ui32", per_element_vec_ui32_text)):
        expect(
            int_text.count("llvm.insertelement") == 2,
            f"pto.Vec(pto.{int_label}, 2, init=...) should emit exactly two llvm.insertelement ops",
        )
        expect(
            int_text.count("llvm.store") == 1,
            f"pto.Vec(pto.{int_label}, 2, init=...) store should emit exactly one llvm.store",
        )
        expect(
            "arith.fpext" not in int_text
            and "arith.fptosi" not in int_text
            and "arith.sitofp" not in int_text
            and "arith.uitofp" not in int_text
            and "arith.fptrunc" not in int_text,
            f"pto.Vec(pto.{int_label}, 2, init=...) should keep element bit patterns without scalar conversions",
        )
    expect(
        "vector<2xi32>" in per_element_vec_ui32_text,
        "pto.Vec(pto.ui32, 2, init=...) should build a signless vector<2xi32> keeping the bit pattern",
    )

    vec_literals_text = vec_init_literals_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(vec_literals_text, "per-element Vec init= Python literal specialization")
    expect("vector<2xi32>" in vec_literals_text, "si32/ui32 Python literals should build signless vector<2xi32>")
    expect(
        vec_literals_text.count("llvm.insertelement") == 4,
        "two two-element Vec init= builders over literals should emit exactly four llvm.insertelement ops",
    )
    expect(
        vec_literals_text.count("llvm.store") == 2,
        "each literal-built vector should be stored with exactly one llvm.store",
    )
    expect(
        "pto.store" not in vec_literals_text,
        "literal-built vector stores should not lower through scalar pto.store",
    )
    expect(
        "arith.constant -1" in vec_literals_text,
        "a negative Python literal should keep the two-complement bit pattern",
    )
    expect(
        "arith.fpext" not in vec_literals_text
        and "arith.fptosi" not in vec_literals_text
        and "arith.sitofp" not in vec_literals_text
        and "arith.uitofp" not in vec_literals_text
        and "arith.trunci" not in vec_literals_text
        and "arith.extsi" not in vec_literals_text
        and "arith.extui" not in vec_literals_text,
        "Python literal elements should reach insertelement without any numeric widening/truncation casts",
    )

    expect_raises(
        TypeError,
        lambda: vec_init_mapping_probe.compile(),
        "expects a scalar, a builtin vector, or a sequence of scalars",
    )

    vec_broadcast_unsigned_text = vec_broadcast_unsigned_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(
        vec_broadcast_unsigned_text,
        "pto.Vec ui32 init= broadcast specialization",
    )
    expect(
        "vector<2xi32>" in vec_broadcast_unsigned_text,
        "pto.Vec(pto.ui32, 2, init=scalar) should build a signless vector<2xi32> instead of failing verification",
    )
    expect(
        vec_broadcast_unsigned_text.count("llvm.insertelement") == 2
        and vec_broadcast_unsigned_text.count("llvm.store") == 1,
        "unsigned broadcast should lower through two insertelements and one llvm.store",
    )

    vec_16b_text = vec_init_16b_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(vec_16b_text, "per-element Vec init= 16-bit specialization")
    for vec_type, label in (("vector<2xf16>", "f16 x 2"), ("vector<4xf16>", "f16 x 4"), ("vector<4xi16>", "i16 x 4")):
        expect(
            vec_type in vec_16b_text,
            f"pto.Vec(..., init=...) should build {vec_type} for {label} packs",
        )
    # Snapshot-style count (see the NOTE above the f32 probe assertions): 2+4+4+4 packed
    # elements across the four 16-bit stores. Update if the pipeline folds the chain.
    expect(
        vec_16b_text.count("llvm.insertelement") == 14,
        "16-bit packs (2+4+4+4 elements) should emit exactly fourteen llvm.insertelement ops",
    )
    expect(
        vec_16b_text.count("llvm.store") == 4,
        "each 16-bit vector should be stored with exactly one llvm.store",
    )
    expect(
        "pto.store" not in vec_16b_text,
        "16-bit vector stores should not lower through scalar pto.store",
    )
    expect(
        "arith.fpext" not in vec_16b_text
        and "arith.fptosi" not in vec_16b_text
        and "arith.sitofp" not in vec_16b_text
        and "arith.uitofp" not in vec_16b_text
        and "arith.trunci" not in vec_16b_text
        and "arith.extsi" not in vec_16b_text
        and "arith.extui" not in vec_16b_text,
        "16-bit literal elements should reach insertelement without any numeric widening or truncation casts",
    )

    expect_raises(
        ValueError,
        lambda: vec_init_size_mismatch_probe.compile(),
        "expects exactly 2 element(s), got 3",
    )
    expect_raises(
        TypeError,
        lambda: vec_init_str_probe.compile(),
        "expects a scalar, a builtin vector, or a sequence of scalars",
    )

    vec_arith_text = scalar_contiguous_vector_arith_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(vec_arith_text, "scalar contiguous vector arithmetic specialization")
    expect(
        "pto.simt_launch" in vec_arith_text,
        "VecValue arithmetic probe should lower through a SIMT launch",
    )
    expect("arith.addf" in vec_arith_text, "VecValue addition should lower to arith.addf")
    expect("arith.subf" in vec_arith_text, "VecValue subtraction should lower to arith.subf")
    expect("arith.mulf" in vec_arith_text, "VecValue multiplication should lower to arith.mulf")
    expect("vector<4xf32>" in vec_arith_text, "vector arithmetic should keep vector<4xf32> type")
    expect(
        vec_arith_text.count("arith.addf") == 1,
        "VecValue __add__ should emit exactly one arith.addf",
    )
    expect(
        vec_arith_text.count("arith.subf") == 1,
        "VecValue __sub__ should emit exactly one arith.subf",
    )
    expect(
        vec_arith_text.count("arith.mulf") == 1,
        "VecValue __mul__ should emit exactly one arith.mulf",
    )

    addptr_surface_text = addptr_surface_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(addptr_surface_text, "addptr surface specialization")
    expect(
        addptr_surface_text.count("pto.addptr") == 2,
        "addptr(...) should lower one PTO addptr op per authored pointer-advance call",
    )
    expect(
        "arith.index_cast" in addptr_surface_text,
        "addptr(ptr, i32-value) should coerce integer runtime scalars to index",
    )
    expect(
        re.search(r"pto\.addptr %\d+, %c2(?:_\d+)?", addptr_surface_text) is not None,
        "addptr(ptr, 2) should accept a Python int offset and lower it as an index element offset",
    )
    expect(
        re.search(r"pto\.addptr %\d+, %\d+", addptr_surface_text) is not None,
        "addptr(ptr, pto.i32(...)) should accept integer runtime scalars as element offsets",
    )

    simt_pointer_offset_text = simt_pointer_offset_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(simt_pointer_offset_text, "simt pointer offset specialization")
    expect(
        re.search(r"call @simt_pointer_offset_helper__simt_\d+", simt_pointer_offset_text) is not None,
        "@pto.simt pointer helper should lower to a helper func.call",
    )
    expect(
        re.search(r"pto\.store %c9_i32, %(?:arg0|\d+)\[%c1(?:_\d+)?\]", simt_pointer_offset_text) is not None,
        "ptr+offset sugar inside @pto.simt helpers should lower as address offsets, not scalar add",
    )
    expect(
        re.search(r"pto\.load %\d+\[%c1(?:_\d+)?\]", simt_pointer_offset_text) is not None,
        "@pto.simt pointer helper probe should preserve ptr+offset load syntax on the caller side",
    )
    simt_reserved_buffer_peer_text = simt_reserved_buffer_peer_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(
        simt_reserved_buffer_peer_text,
        "simt reserved-buffer peer specialization",
    )
    expect(
        re.search(
            r"pto\.import_reserved_buffer\{[^}]*peer_func = @simt_reserved_buffer_peer__simt_\d+",
            simt_reserved_buffer_peer_text,
        ) is not None,
        "import_reserved_buffer(peer_func=@pto.simt helper) should reference the materialized helper symbol",
    )
    expect_raises(
        RuntimeError,
        lambda: simt_reserved_buffer_ambiguous_peer_probe.compile().mlir_text(),
        "multiple specializations",
    )

    scalar_store_coercion_text = scalar_store_element_coercion_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(scalar_store_coercion_text, "scalar store coercion specialization")
    expect(
        scalar_store_coercion_text.count("arith.index_cast") >= 2,
        "scalar.store(...) should coerce index runtime values to the destination integer element type",
    )
    expect(
        "arith.trunci" in scalar_store_coercion_text,
        "scalar.store(...) should coerce wider integer runtime values down to the destination element type",
    )
    expect(
        scalar_store_coercion_text.count("pto.store") == 4,
        "scalar.store(...) coercion probe should still lower to four pto.store operations",
    )
    expect(
        re.search(r"pto\.store %\w+, %\d+\[%c0(?:_\d+)?\]", scalar_store_coercion_text) is not None,
        "scalar.store(index, i32_ptr) should preserve explicit target coercion at offset 0",
    )

    shared_index_coercion_text = shared_index_coercion_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(shared_index_coercion_text, "shared index coercion specialization")
    expect(
        shared_index_coercion_text.count("arith.index_cast") >= 3,
        "shared index coercion should cast one i32 value through loop bound, step, addptr, and event id paths",
    )
    expect("scf.for" in shared_index_coercion_text, "shared index coercion should lower i32 loop bounds to scf.for")
    expect("pto.addptr" in shared_index_coercion_text, "shared index coercion should lower i32 pointer offsets")
    expect(
        "pto.wait_flag_dyn" in shared_index_coercion_text,
        "shared index coercion should lower i32 event ids to dynamic wait_flag",
    )

    public_surface_text = public_surface_exports_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(public_surface_text, "public surface export specialization")
    explicit_mx_scale_staging_text = explicit_mx_scale_staging_surface_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(
        explicit_mx_scale_staging_text,
        "explicit MX scale staging specialization",
    )
    explicit_fp4_s4_staging_text = explicit_fp4_s4_staging_surface_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(
        explicit_fp4_s4_staging_text,
        "explicit FP4 S4 staging specialization",
    )
    acc_store_pre_quant_text = acc_store_pre_quant_surface_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(acc_store_pre_quant_text, "acc-store pre_quant public mode specialization")
    expect(
        "pre_quant(%" in acc_store_pre_quant_text and "mode = f32_bf16" in acc_store_pre_quant_text,
        "mte_l0c_gm pre_quant should preserve the f32_bf16 public mode",
    )
    expect_raises(
        ValueError,
        lambda: acc_store_bad_pre_quant_mode_probe.compile().mlir_text(),
        "unsupported acc store pre_quant mode",
    )
    expect_raises(
        TypeError,
        lambda: acc_store_bad_pre_quant_payload_probe.compile().mlir_text(),
        "acc store scalar pre_quant payload must be an f16, bf16, or f32 scalar",
    )
    compile_time_query_text = compile_time_query_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(compile_time_query_text, "compile-time query specialization")
    eager_scalar_text = eager_scalar_constructor_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(eager_scalar_text, "eager scalar constructor specialization")
    low_precision_storage_text = low_precision_storage_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(low_precision_storage_text, "low-precision storage specialization")
    pointer_vlds_text = pointer_vlds_inference_probe.compile(BLOCK=128).mlir_text()
    expect_parse_roundtrip_and_verify(pointer_vlds_text, "pointer vlds inference specialization")
    mask_bitcast_text = public_mask_bitcast_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(mask_bitcast_text, "public mask bitcast specialization")
    mask_surface_text = public_mask_surface_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(mask_surface_text, "public mask surface specialization")
    sync_surface_text = public_sync_surface_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(sync_surface_text, "public sync surface specialization")
    trap_surface_text = public_trap_surface_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(trap_surface_text, "public trap surface specialization")
    dynamic_buf_sync_text = public_dynamic_buf_sync_surface_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(dynamic_buf_sync_text, "dynamic buf sync surface specialization")
    explicit_runtime_index_bitwise_event_text = explicit_runtime_index_bitwise_event_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(
        explicit_runtime_index_bitwise_event_text,
        "explicit runtime index bitwise event specialization",
    )
    explicit_runtime_index_integer_bitwise_event_text = (
        explicit_runtime_index_integer_bitwise_event_probe.compile().mlir_text()
    )
    expect_parse_roundtrip_and_verify(
        explicit_runtime_index_integer_bitwise_event_text,
        "explicit runtime index/integer bitwise event specialization",
    )
    ast_runtime_index_bitwise_event_text = ast_runtime_index_bitwise_event_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(
        ast_runtime_index_bitwise_event_text,
        "AST runtime index bitwise event specialization",
    )
    data_movement_surface_text = public_data_movement_surface_probe.compile().mlir_text()
    ui8_pad_surface_text = ui8_pad_surface_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(ui8_pad_surface_text, "ui8 pad value specialization")
    expect_parse_roundtrip_and_verify(data_movement_surface_text, "public data movement surface specialization")
    vector_conversion_surface_text = public_vector_conversion_surface_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(vector_conversion_surface_text, "public vector conversion surface specialization")
    low_precision_memory_surface_text = low_precision_vector_memory_surface_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(low_precision_memory_surface_text, "low-precision vector memory surface specialization")
    low_precision_vcvt_surface_text = low_precision_vcvt_surface_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(low_precision_vcvt_surface_text, "low-precision vcvt surface specialization")
    vdup_surface_text = vdup_surface_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(vdup_surface_text, "public vdup surface specialization")
    vecscope_surface_text = vecscope_surface_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(vecscope_surface_text, "public vecscope surface specialization")
    vmulscvt_surface_text = vmulscvt_surface_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(vmulscvt_surface_text, "public vmulscvt surface specialization")
    vmula_surface_text = vmula_surface_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(vmula_surface_text, "public vmula surface specialization")
    vmadd_surface_text = vmadd_surface_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(vmadd_surface_text, "public vmadd surface specialization")
    vsstb_post_update_surface_text = vsstb_post_update_surface_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(vsstb_post_update_surface_text, "vsstb post-update surface specialization")
    vmi_wrapper_dispatch_text = vmi_wrapper_dispatch_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(vmi_wrapper_dispatch_text, "public VMI wrapper dispatch specialization")
    vmi_vhist_signless_source_text = vmi_vhist_signless_source_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(
        vmi_vhist_signless_source_text,
        "public VMI vdhist/vchist signless-source specialization",
    )
    expect(
        "pto.vmi.vdhist" in vmi_vhist_signless_source_text
        and "pto.vmi.vchist" in vmi_vhist_signless_source_text,
        "signless-source vdhist/vchist probe should still emit both histogram ops",
    )
    expect(
        "!pto.vmi.vreg<256xi8" in vmi_vhist_signless_source_text,
        "vdhist/vchist probe with dtype=pto.i8 source should surface signless i8 lanes in IR",
    )
    expect(
        "!pto.vmi.vreg<256xui16" in vmi_vhist_signless_source_text,
        "vdhist/vchist acc/result must remain ui16 in emitted IR",
    )
    vmi_vhist_signless_acc_text = vmi_vhist_signless_acc_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(
        vmi_vhist_signless_acc_text,
        "public VMI vdhist/vchist signless-acc specialization",
    )
    expect(
        "pto.vmi.vdhist" in vmi_vhist_signless_acc_text
        and "pto.vmi.vchist" in vmi_vhist_signless_acc_text,
        "signless-acc vdhist/vchist probe should emit both histogram ops",
    )
    expect(
        "!pto.vmi.vreg<256xi16" in vmi_vhist_signless_acc_text,
        "vdhist/vchist probe with dtype=pto.i16 acc should surface signless i16 lanes in IR",
    )
    expect(
        "!pto.vmi.vreg<256xui16" in vmi_vhist_signless_acc_text,
        "vdhist/vchist result must always be ui16, even when acc is signless i16",
    )
    vmi_vhist_bad_acc_error = expect_raises(
        TypeError,
        vmi_vhist_bad_acc_probe.compile,
        "requires acc element type to be ui16 or i16",
    )
    expect(
        "pto.vmi.vdhist(...)" in str(vmi_vhist_bad_acc_error),
        "vdhist bad-acc rejection should carry the pto.vmi.vdhist call-site context",
    )
    vmi_brc_vload_text = vmi_brc_vload_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(vmi_brc_vload_text, "public VMI brc vload specialization")
    expect(
        'dist_mode = "brc"' in vmi_brc_vload_text,
        "pto.vmi.vload should preserve the authored brc dist_mode without requiring a stride operand",
    )
    vmi_group_brc_vload_text = vmi_group_brc_vload_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(vmi_group_brc_vload_text, "public VMI grouped brc vload specialization")
    expect(
        'dist_mode = "brc"' in vmi_group_brc_vload_text and "group = 8" in vmi_group_brc_vload_text,
        "pto.vmi.vload should allow the grouped brc form exposed by the VMI IR contract",
    )
    vmi_block_stride_memory_text = vmi_block_stride_memory_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(
        vmi_block_stride_memory_text, "public VMI block-stride memory specialization"
    )
    expect(
        vmi_block_stride_memory_text.count("pto.vmi.vload") == 1
        and vmi_block_stride_memory_text.count("pto.vmi.vstore") == 1,
        "VMI block-stride memory forms should compile with block_stride alone",
    )
    expect(
        "repeat_stride" not in inspect.signature(pto.vmi.vload).parameters
        and "repeat_stride" not in inspect.signature(pto.vmi.vstore).parameters,
        "VMI load/store Python signatures must not expose repeat_stride",
    )
    fixed_width_integer_text = fixed_width_integer_specialization_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(fixed_width_integer_text, "fixed-width integer specialization")
    with mock.patch.object(vmi_namespace._pto, "vmi_vadd", None):
        missing_vmi_binding = expect_raises(
            NotImplementedError,
            vmi_missing_binding_probe.compile,
            "pto.vmi.vadd",
        )
    expect(
        "Rebuild PTO Python bindings" in str(missing_vmi_binding),
        "missing generated VMI bindings should diagnose rebuild guidance",
    )
    masked_vcvt_error = expect_raises(
        NotImplementedError,
        vmi_masked_vcvt_probe.compile,
        "pto.vmi.vcvt",
    )
    expect(
        "masked form" in str(masked_vcvt_error),
        "unsupported VMI backend/binding forms should diagnose the unavailable feature",
    )
    invalid_rounding_vcvt_error = expect_raises(
        ValueError,
        vmi_invalid_rounding_vcvt_probe.compile,
        "rounding",
    )
    expect(
        "expected one of A, H, R, Z" in str(invalid_rounding_vcvt_error),
        "pto.vmi.vcvt should reject unsupported VMI rounding tokens before IR verification",
    )
    vmi_round_r_vcvt_text = vmi_round_r_vcvt_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(vmi_round_r_vcvt_text, "VMI R-rounding vcvt specialization")
    expect(
        'rounding = "R"' in vmi_round_r_vcvt_text,
        "pto.vmi.vcvt should preserve the authored R rounding token for fp32->fp8",
    )
    vmi_bf16x2_to_f4x2_text = vmi_bf16x2_to_f4x2_vcvt_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(
        vmi_bf16x2_to_f4x2_text,
        "VMI bf16x2-to-f4x2 vcvt specialization",
    )
    expect(
        "!pto.vmi.vreg<256xbf16>" in vmi_bf16x2_to_f4x2_text,
        "VMI bf16x2 vcvt probe should load the source as 256 scalar bf16 lanes",
    )
    expect(
        "!pto.vmi.vreg<128x!pto.bf16x2>" in vmi_bf16x2_to_f4x2_text,
        "VMI vinterpret_cast should form 128 logical bf16x2 lanes",
    )
    expect(
        "!pto.vmi.vreg<128x!pto.f4E1M2x2>" in vmi_bf16x2_to_f4x2_text
        and "!pto.vmi.vreg<128x!pto.f4E2M1x2>" in vmi_bf16x2_to_f4x2_text,
        "VMI bf16x2 vcvt should preserve both packed fp4 result element types",
    )
    expect(
        vmi_bf16x2_to_f4x2_text.count('rounding = "R"') >= 2,
        "omitted rounding for bf16x2-to-f4x2 should materialize deterministic R rounding",
    )
    for rounding in ("R", "A", "F", "Z", "C"):
        expect(
            f'rounding = "{rounding}"' in vmi_bf16x2_to_f4x2_text,
            f"bf16x2-to-f4x2 should accept and preserve {rounding} rounding",
        )
    expect(
        'saturate =' not in vmi_bf16x2_to_f4x2_text,
        "bf16x2-to-f4x2 VMI vcvt must not emit a saturate attribute",
    )
    vmi_bf16x2_to_f4x2_vstore_text = vmi_bf16x2_to_f4x2_vstore_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(
        vmi_bf16x2_to_f4x2_vstore_text,
        "VMI bf16x2-to-f4x2 vcvt/vstore specialization",
    )
    expect(
        vmi_bf16x2_to_f4x2_vstore_text.count("pto.vmi.vstore") == 2,
        "bf16x2-to-f4x2 conversion results should be storable through VMI vstore",
    )
    expect(
        "pto.vmi.vstore" in vmi_bf16x2_to_f4x2_vstore_text
        and "f4E1M2x2" in vmi_bf16x2_to_f4x2_vstore_text
        and "f4E2M1x2" in vmi_bf16x2_to_f4x2_vstore_text,
        "VMI vstore should preserve both packed fp4 destination element types",
    )
    expect(
        "!pto.bf16x2" in vmi_bf16x2_to_f4x2_vstore_text
        and "!pto.f4E1M2x2" in vmi_bf16x2_to_f4x2_vstore_text
        and "!pto.f4E2M1x2" in vmi_bf16x2_to_f4x2_vstore_text,
        "size=128 should preserve the packed bf16x2 and fp4x2 element types",
    )
    expect(
        vmi_bf16x2_to_f4x2_vstore_text.count('rounding = "R"') == 2,
        "bf16x2-to-f4x2 vstore results should use explicit default R rounding",
    )
    expect(
        "saturate =" not in vmi_bf16x2_to_f4x2_vstore_text,
        "bf16x2-to-f4x2 vcvt/vstore must not emit a saturate attribute",
    )
    expect_raises(
        ValueError,
        vmi_bf16x2_to_f4x2_invalid_rounding_probe.compile,
        "expected one of A, C, F, R, Z",
    )
    for invalid_probe in (
        vmi_bf16x2_to_f4x2_sat_probe,
        vmi_bf16x2_to_f4x2_nosat_probe,
    ):
        expect_raises(
            ValueError,
            invalid_probe.compile,
            "does not support saturate for bf16x2",
        )
    for invalid_probe in (
        vmi_bf16x2_unsupported_pair_probe,
    ):
        expect_raises(
            TypeError,
            invalid_probe.compile,
            "supports bf16x2 only for bf16x2 <->",
        )
    vmi_f4x2_to_bf16x2_text = vmi_f4x2_to_bf16x2_vcvt_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(
        vmi_f4x2_to_bf16x2_text,
        "VMI f4x2-to-bf16x2 vcvt specialization",
    )
    expect(
        "!pto.vmi.vreg<128x!pto.f4E1M2x2>" in vmi_f4x2_to_bf16x2_text,
        "VMI f4x2 vcvt probe should load the source as 128 packed f4 lanes",
    )
    expect(
        "!pto.vmi.vreg<128x!pto.bf16x2>" in vmi_f4x2_to_bf16x2_text,
        "VMI f4x2-to-bf16x2 vcvt should widen to 128 logical bf16x2 lanes",
    )
    expect(
        'rounding =' not in vmi_f4x2_to_bf16x2_text,
        "f4x2-to-bf16x2 VMI vcvt must not emit a rounding attribute",
    )
    expect(
        "saturate =" not in vmi_f4x2_to_bf16x2_text,
        "f4x2-to-bf16x2 VMI vcvt must not emit a saturate attribute",
    )
    expect_raises(
        ValueError,
        vmi_f4x2_to_bf16x2_rounding_probe.compile,
        "does not support rounding for",
    )
    expect_raises(
        ValueError,
        vmi_f4x2_to_bf16x2_sat_probe.compile,
        "does not support saturate for",
    )

    expected_vmi_ops = [
        "pto.vmi.vload",
        "pto.vmi.vstore",
        "pto.vmi.vsstb",
        "pto.vmi.vci",
        "pto.vmi.vadd",
        "pto.vmi.vaddc",
        "pto.vmi.vaddcs",
        "pto.vmi.vsub",
        "pto.vmi.vmul",
        "pto.vmi.vdiv",
        "pto.vmi.vmax",
        "pto.vmi.vmin",
        "pto.vmi.vabs",
        "pto.vmi.vneg",
        "pto.vmi.vrelu",
        "pto.vmi.vexp",
        "pto.vmi.vln",
        "pto.vmi.vsqrt",
        "pto.vmi.vand",
        "pto.vmi.vor",
        "pto.vmi.vxor",
        "pto.vmi.vnot",
        "pto.vmi.vshl",
        "pto.vmi.vshr",
        "pto.vmi.vadds",
        "pto.vmi.vmuls",
        "pto.vmi.vmaxs",
        "pto.vmi.vmins",
        "pto.vmi.vshls",
        "pto.vmi.vshrs",
        "pto.vmi.vcmp",
        "pto.vmi.vcmps",
        "pto.vmi.vsel",
        "pto.vmi.vselr",
        "pto.vmi.vbrc",
        "pto.vmi.vcadd",
        "pto.vmi.vcmax",
        "pto.vmi.vcmin",
        "pto.vmi.vcvt",
        "pto.vmi.vinterpret_cast",
        "pto.vmi.vexpdif",
        "pto.vmi.vaxpy",
        "pto.vmi.vlrelu",
        "pto.vmi.vprelu",
        "pto.vmi.vmull",
        "pto.vmi.vmula",
        "pto.vmi.vchist",
        "pto.vmi.vdhist",
        "pto.vmi.vgather",
        "pto.vmi.vgatherb",
        "pto.vmi.vscatter",
        "pto.vmi.create_mask",
        "pto.vmi.create_group_mask",
        "pto.vmi.vintlv",
        "pto.vmi.vdintlv",
    ]
    for op_name in expected_vmi_ops:
        expect(
            op_name in vmi_wrapper_dispatch_text,
            f"representative {op_name} wrapper dispatch should emit the matching generated VMI op",
        )
    for op_name, expected_count in (("vcadd", 2), ("vcmax", 3), ("vcmin", 2)):
        expect(
            vmi_wrapper_dispatch_text.count(f"pto.vmi.{op_name}") == expected_count,
            f"pto.vmi.{op_name} should emit both omitted-group and explicit-group probes",
        )
    expect(
        vmi_wrapper_dispatch_text.count("pto.vmi.vexpdif") == 2,
        "VMI wrapper dispatch should cover both f32 and f16 vexpdif forms",
    )
    expect(
        "!pto.vmi.vreg<128xf32>" in vmi_wrapper_dispatch_text,
        "f16 vexpdif should produce a logical f32 result",
    )
    expect(
        vmi_wrapper_dispatch_text.count("group = 1") >= 6,
        "VMI reductions should make the omitted group equivalent to group=1",
    )
    expect(
        vmi_wrapper_dispatch_text.count("pto.vmi.vload") == 9,
        "vmi wrapper dispatch probe should lower nine explicit VMI loads",
    )
    expect(
        "pto.backend = \"vpto\"" in vmi_wrapper_dispatch_text,
        "VMI public surface probe should compile through the VMI/VPTO backend partition",
    )
    expect(
        "!pto.vmi.vreg<64xf32>" in vmi_wrapper_dispatch_text,
        "PTODSL VMI compile probes should materialize logical VMI vector result types in MLIR",
    )
    expect(
        "!pto.vmi.vreg<1xf32>" in vmi_wrapper_dispatch_text,
        "PTODSL VMI reduction probes should materialize reduced logical VMI vector result types in MLIR",
    )
    expect(
        "!pto.vmi.vreg<64xf16>" in vmi_wrapper_dispatch_text,
        "PTODSL VMI conversion probes should materialize converted logical VMI vector result types in MLIR",
    )
    for rounding in ("R", "A", "H", "Z"):
        expect(
            f'rounding = "{rounding}"' in vmi_wrapper_dispatch_text,
            f"PTODSL vcvt should preserve canonical {rounding} rounding",
        )
    expect(
        vmi_wrapper_dispatch_text.count('saturate = "SAT"') >= 5,
        "PTODSL vcvt should preserve explicit SAT and default omitted narrowing saturation to SAT",
    )
    expect(
        "!pto.vmi.vreg<64xi32>" in vmi_wrapper_dispatch_text,
        "PTODSL VMI index/reinterpret probes should materialize integer logical VMI vector result types in MLIR",
    )
    expect(
        "!pto.vmi.vreg<128xf16>" in vmi_wrapper_dispatch_text,
        "PTODSL vinterpret_cast should infer lane count by conserving total bits",
    )
    expect(
        "!pto.vmi.mask<64xpred>" in vmi_wrapper_dispatch_text,
        "PTODSL VMI compare and prefix mask probes should materialize logical VMI mask types in MLIR",
    )
    expect(
        "!pto.vmi.mask<64xpred>" in vmi_wrapper_dispatch_text,
        "PTODSL grouped mask probes should materialize grouped logical VMI mask types in MLIR",
    )
    expect(
        "reassoc" in vmi_wrapper_dispatch_text,
        "PTODSL VMI vcadd should preserve the reassoc attr in MLIR",
    )
    expect("pto.mte_gm_ub" in public_surface_text, "mte_load(...) should lower to pto.mte_gm_ub")
    expect("pto.mte_ub_gm" in public_surface_text, "mte_store(...) should lower to pto.mte_ub_gm")
    expect(
        re.search(r"pto\.mte_ub_gm [^\n]+ nburst\([^)]+\) l2_cache_ctl\(%c0[^)\n]*\)", public_surface_text) is not None,
        "mte_store(...) should pass default l2_cache_ctl through the pto.mte_ub_gm l2_cache_ctl group",
    )
    expect(public_surface_text.count("pto.mem_bar") >= 1, "mem_bar(...) should still lower explicit memory barriers")
    expect("pto.barrier <PIPE_ALL>" in public_surface_text, "pipe_barrier(Pipe.ALL) should lower to pto.barrier")
    expect("pto.trap" in trap_surface_text, "pto.trap() should lower to the public pto.trap operation")
    expect("pto.vexp" in public_surface_text, "vexp(...) should lower to pto.vexp")
    expect("pto.vcgmax" in public_surface_text, "vcgmax(...) should lower to pto.vcgmax")
    expect("pto.vcgadd" in public_surface_text, "vcgadd(...) should lower to pto.vcgadd")
    expect("pto.vsqz" in public_surface_text, "vsqz(...) should lower to pto.vsqz")
    expect("pto.vadds" in public_surface_text, "vsubs(...) should lower via scalar negation plus pto.vadds")
    expect("pto.mte_l1_l0a" in public_surface_text, "mte_l1_l0a(...) should lower to pto.mte_l1_l0a")
    expect("start(" not in public_surface_text, "mte_l1_l0a/l0b start_row/start_col should lower as operands")
    expect('pto.get_buf "PIPE_V", 0, 0' in sync_surface_text, 'get_buf(Pipe.V, 0) should lower to pto.get_buf with PIPE_V')
    expect('pto.rls_buf "PIPE_MTE2", 1, 2' in sync_surface_text, 'rls_buf(Pipe.MTE2, 1, 2) should lower to pto.rls_buf with PIPE_MTE2')
    expect("pto.get_buf_dyn" in dynamic_buf_sync_text, "get_buf_dyn should lower to pto.get_buf_dyn")
    expect("pto.rls_buf_dyn" in dynamic_buf_sync_text, "rls_buf_dyn should lower to pto.rls_buf_dyn")
    expect(
        'pto.get_buf_dyn "PIPE_V"' in dynamic_buf_sync_text,
        'get_buf_dyn(Pipe.V, const_buf_id) should lower to pto.get_buf_dyn with PIPE_V',
    )
    expect(
        'pto.rls_buf_dyn "PIPE_MTE2"' in dynamic_buf_sync_text,
        'rls_buf_dyn(Pipe.MTE2, const_buf_id, 2) should lower to pto.rls_buf_dyn with PIPE_MTE2',
    )
    expect(
        dynamic_buf_sync_text.count('pto.get_buf_dyn "PIPE_V"') == 1,
        "constant SSA buf-id get_buf_dyn should preserve PIPE_V",
    )
    expect(
        dynamic_buf_sync_text.count('pto.get_buf_dyn "PIPE_MTE2"') == 1,
        "loop get_buf_dyn should preserve PIPE_MTE2",
    )
    expect(
        dynamic_buf_sync_text.count('pto.rls_buf_dyn "PIPE_MTE2"') == 2,
        "constant and loop rls_buf_dyn calls should preserve PIPE_MTE2",
    )
    expect(
        "arith.andi" in dynamic_buf_sync_text,
        "iter & 1 in pto.for_ should lower to arith.andi for dynamic buf_id computation",
    )
    expect("pto.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]" in sync_surface_text, "set_flag(..., event_id=0) should lower static event ids to pto.set_flag")
    expect("pto.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]" in sync_surface_text, "wait_flag(..., event_id=0) should lower static event ids to pto.wait_flag")
    expect("pto.set_flag_dyn[<PIPE_V>, <PIPE_MTE3>, %c3]" in sync_surface_text, "set_flag(..., event_id=dynamic_event) should lower runtime event ids to pto.set_flag_dyn")
    expect("pto.wait_flag_dyn[<PIPE_V>, <PIPE_MTE3>, %c3]" in sync_surface_text, "wait_flag(..., event_id=dynamic_event) should lower runtime event ids to pto.wait_flag_dyn")
    expect("arith.andi" in explicit_runtime_index_bitwise_event_text, "explicit pto.for_ index & event id should lower to arith.andi")
    expect("arith.ori" in explicit_runtime_index_bitwise_event_text, "explicit pto.for_ index | event id should lower to arith.ori")
    expect("arith.xori" in explicit_runtime_index_bitwise_event_text, "explicit pto.for_ index ^ event id should lower to arith.xori")
    expect(
        re.search(r"arith\.andi .* : index", explicit_runtime_index_bitwise_event_text) is not None,
        "index & literal event id should stay in the index type domain",
    )
    expect(
        "arith.index_cast" not in explicit_runtime_index_bitwise_event_text,
        "index/literal bitwise event ids should not lower through fixed-width integer casts",
    )
    expect(
        explicit_runtime_index_bitwise_event_text.count("pto.wait_flag_dyn") == 3,
        "explicit pto.for_ index bitwise event id should lower to pto.wait_flag_dyn",
    )
    expect(
        explicit_runtime_index_bitwise_event_text.count("pto.set_flag_dyn") == 2,
        "explicit pto.for_ index bitwise event id should lower to pto.set_flag_dyn",
    )
    expect(
        explicit_runtime_index_integer_bitwise_event_text.count("arith.index_cast") >= 2,
        "index/integer bitwise event ids should coerce integer runtime scalars to index",
    )
    expect(
        explicit_runtime_index_integer_bitwise_event_text.count("pto.wait_flag_dyn") == 2,
        "index/integer bitwise event ids should lower waits to pto.wait_flag_dyn",
    )
    expect(
        explicit_runtime_index_integer_bitwise_event_text.count("pto.set_flag_dyn") == 1,
        "index/integer bitwise event ids should lower sets to pto.set_flag_dyn",
    )
    expect(
        "arith.andi" in ast_runtime_index_bitwise_event_text,
        "AST rewritten range loop index & event id should lower to arith.andi",
    )
    expect(
        ast_runtime_index_bitwise_event_text.count("pto.wait_flag_dyn") == 1,
        "AST rewritten range loop index bitwise event id should lower to pto.wait_flag_dyn",
    )
    expect(
        "pto.set_cross_block <PIPE_FIX>, 0" in sync_surface_text,
        "set_cross_block(Pipe.FIX, 0) should lower to pto.set_cross_block",
    )
    expect(
        "pto.wait_cross_block <PIPE_FIX>, 0" in sync_surface_text,
        "wait_cross_block(Pipe.FIX, 0) should lower to pto.wait_cross_block",
    )
    expect(
        "pto.set_intra_block <PIPE_MTE3>, %c3" in sync_surface_text,
        "set_intra_block(Pipe.MTE3, dynamic_event) should lower to pto.set_intra_block",
    )
    expect(
        "pto.set_intra_block <PIPE_FIX>, 4" in sync_surface_text,
        "set_intra_block(Pipe.FIX, 4) should preserve physical event ids",
    )
    expect(
        "pto.wait_intra_block <PIPE_V>, %c3" in sync_surface_text,
        "wait_intra_block(Pipe.V, dynamic_event) should lower to pto.wait_intra_block",
    )
    expect(
        "pto.wait_intra_block <PIPE_FIX>, 20" in sync_surface_text,
        "wait_intra_block(Pipe.FIX, 20) should preserve physical event ids",
    )
    expect(
        "pto.wait_intra_block <PIPE_MTE3>, %c3" in sync_surface_text,
        "wait_intra_block(Pipe.MTE3, dynamic_event) should lower to pto.wait_intra_block",
    )
    expect(
        "pto.wait_intra_block <PIPE_MTE3>, 31" in sync_surface_text,
        "wait_intra_block(Pipe.MTE3, 31) should preserve static physical event ids",
    )
    expect(data_movement_surface_text.count("pto.mte_gm_ub") == 2, "public grouped GM->UB wrappers should lower to pto.mte_gm_ub")
    expect("pto.mte_ub_gm" in data_movement_surface_text, "public grouped UB->GM wrapper should lower to pto.mte_ub_gm")
    expect(
        data_movement_surface_text.count("pto.set_store_atomic_cfg") == 2,
        "set_store_atomic_cfg should lower the requested config and reset",
    )
    for atomic_api in (
        "set_atomic_s32",
        "set_atomic_add",
        "set_atomic_max",
        "set_atomic_min",
        "set_atomic_f32",
        "set_atomic_f16",
        "set_atomic_bf16",
        "set_atomic_s16",
        "set_atomic_s8",
        "set_atomic_none",
    ):
        expect(
            f"pto.{atomic_api}" in data_movement_surface_text,
            f"{atomic_api} should lower through the explicit PTODSL surface",
        )
    expect(
        re.search(r"pto\.mte_ub_gm [^\n]+ nburst\([^)]+\) l2_cache_ctl\(%c7[^)\n]*\)", data_movement_surface_text) is not None,
        "mte_ub_gm(..., l2_cache='nared') should map to the l2_cache_ctl group",
    )
    expect(
        re.search(r"pto\.mte_ub_gm [^\n]+ nburst\([^)]+\) l2_cache_ctl\(%c15[^)\n]*\)", data_movement_surface_text) is not None,
        "mte_ub_gm(..., l2_cache='wtsred') should map to the l2_cache_ctl group",
    )
    expect(
        re.search(r"pto\.mte_gm_ub [^\n]+ pad\([^)]*\) [^\n]*pad i8", ui8_pad_surface_text) is not None,
        "ui8 pad value should be normalized to a signless i8 scalar for SET.MOV.PAD.VAL encoding (issue #1245)",
    )
    expect("pto.mte_ub_ub" in data_movement_surface_text, "public grouped UB->UB wrapper should lower to pto.mte_ub_ub")
    expect("pto.mte_ub_l1" in data_movement_surface_text, "public grouped UB->L1 wrapper should lower to pto.mte_ub_l1")
    expect("pto.mte_gm_l1" in data_movement_surface_text, "public grouped GM->L1 wrapper should lower to pto.mte_gm_l1")
    expect(
        "pto.raw_fill_l1" in data_movement_surface_text,
        "public L1 raw-fill wrapper should lower to pto.raw_fill_l1",
    )
    expect("pto.mte_l1_ub" in data_movement_surface_text, "public grouped L1->UB wrapper should lower to pto.mte_l1_ub")
    expect("pto.mte_gm_l1_frac" in data_movement_surface_text, "public GM->L1 frac wrapper should lower to pto.mte_gm_l1_frac")
    expect("pto.mte_l1_bt" in data_movement_surface_text, "public L1->BT wrapper should lower to pto.mte_l1_bt")
    expect("pto.mte_l1_fb" in data_movement_surface_text, "public L1->FB wrapper should lower to pto.mte_l1_fb")
    expect("pto.vldas" in data_movement_surface_text, "vldas(...) should lower to pto.vldas")
    expect("pto.vldus" in data_movement_surface_text, "vldus(...) should lower to pto.vldus")
    expect("pto.vldsx2" in data_movement_surface_text, "vldsx2(...) should lower to pto.vldsx2")
    expect("pto.vstur" in data_movement_surface_text, "vstur(...) should lower to pto.vstur")
    expect("pto.vstus" in data_movement_surface_text, "vstus(...) should lower to pto.vstus")
    expect("pto.vstsx2" in data_movement_surface_text, "vstsx2(...) should lower to pto.vstsx2")
    expect("pto.vgather2" in data_movement_surface_text, "vgather2(...) should lower to pto.vgather2")
    expect("pto.vgather2_bc" in data_movement_surface_text, "vgather2_bc(...) should lower to pto.vgather2_bc")
    expect("pto.vgatherb" in data_movement_surface_text, "vgatherb(...) should lower to pto.vgatherb")
    expect("pto.vscatter" in data_movement_surface_text, "vscatter(...) should lower to pto.vscatter")
    expect("pto.vsldb" in data_movement_surface_text, "vsldb(...) should lower to pto.vsldb")
    expect("pto.vsstb" in data_movement_surface_text, "vsstb(...) should lower to pto.vsstb")
    expect(
        re.search(r"arith\.index_cast %[a-zA-Z0-9_]+ : index to i16", fixed_width_integer_text) is not None,
        "vsldb/vsstb i16 operands should accept runtime index values through shared scalar adaptation",
    )
    expect(
        fixed_width_integer_text.count("pto.mte_gm_l1_frac") == 3,
        "mte_gm_l1_frac ctrl[1] should preserve bool, 0, and 1 i1 coercion cases",
    )
    expect(
        fixed_width_integer_text.count("i1") >= 3,
        "mte_gm_l1_frac ctrl[1] should materialize bool/0/1 as i1 values",
    )
    expect("pto.vstar" in data_movement_surface_text, "vstar(...) should lower to pto.vstar")
    expect("pto.vstas" in data_movement_surface_text, "vstas(...) should lower to pto.vstas")
    expect("pto.vlds" in vector_conversion_surface_text, "vlds(..., post_update=ON) should lower through pto.vlds on the current VPTO Python surface")
    expect("-> !pto.vreg<64xf32>, !pto.ptr<f32, ub>" in vector_conversion_surface_text, "vlds(..., post_update=ON) should request the updated source pointer result")
    expect("pto.vcvt" in vector_conversion_surface_text, "vcvt(...) should lower to pto.vcvt")
    expect('rnd = "R"' in vector_conversion_surface_text, "vcvt(..., rnd=VcvtRoundMode.R) should preserve the authored rounding attr")
    expect('sat = "SAT"' in vector_conversion_surface_text, "vcvt(..., sat=VcvtSatMode.SAT) should preserve the authored saturation attr")
    expect('part = "EVEN"' in vector_conversion_surface_text, "vcvt(..., part=VcvtPartMode.EVEN) should preserve the authored part attr")
    expect("pto.vsts" in vector_conversion_surface_text, "vsts(..., post_update=ON) should lower through pto.vsts on the current VPTO Python surface")
    expect(vector_conversion_surface_text.count("-> !pto.ptr<f16, ub>") >= 1, "vsts(..., post_update=ON) should request the updated destination pointer result")
    expect('dist = "PK_B32"' in vector_conversion_surface_text, "vsts(..., dist=VStoreDist.PK_B32) should preserve the authored store distribution")
    expect("pto.vpack" in vector_conversion_surface_text, "vpack(...) should lower to pto.vpack")
    expect("!pto.vreg<128xui16>" in vector_conversion_surface_text, "vpack(i32/u32 -> u16) should infer the unsigned packed result type")
    expect("!pto.vreg<256xf8E4M3FN>" in low_precision_memory_surface_text, "vlds/vsts should support f8e4m3 vreg storage")
    expect("!pto.vreg<256x!pto.hif8>" in low_precision_memory_surface_text, "vlds/vsts should support hif8 vreg storage")
    expect("!pto.ptr<f8E4M3FN, ub>" in low_precision_memory_surface_text, "low-precision f8 pointers should lower as UB pointers")
    expect("!pto.vreg<256xf8E4M3FN>" in low_precision_vcvt_surface_text, "vcvt(f32 -> f8e4m3) should infer the packed low-precision result type")
    expect("!pto.vreg<256x!pto.hif8>" in low_precision_vcvt_surface_text, "vcvt(f32 -> hif8) should infer the packed HiF8 result type")
    expect("!pto.vreg<256x!pto.f4E1M2x2>" in low_precision_vcvt_surface_text, "vcvt(bf16 -> f4e1m2x2) should infer the packed 4-bit result type")
    expect('rnd = "H"' in low_precision_vcvt_surface_text, "vcvt(..., rnd=VcvtRoundMode.H) should preserve the H rounding token")
    expect(low_precision_vcvt_surface_text.count('part = "P0"') >= 6, "low-precision packed vcvt forms should preserve P0 part selectors")
    expect(vdup_surface_text.count("pto.vdup") == 3, "vdup(...) should lower once per authored scalar/vector duplication")
    expect("f32, !pto.mask<b32> -> !pto.vreg<64xf32>" in vdup_surface_text, "vdup(scalar_f32, mask_b32) should infer an f32 vector result type")
    expect(vdup_surface_text.count('position = "LOWEST"') >= 1, "vdup(vec, mask) should default position to LOWEST")
    expect('position = "HIGHEST"' in vdup_surface_text, "vdup(vec, mask, PositionMode.HIGHEST) should preserve the authored position")
    expect(vecscope_surface_text.count("pto.vecscope") == 1, "pto.vecscope() should lower exactly one vecscope region")
    expect("pto.vlds" in vecscope_surface_text, "public vecscope body should allow vector loads")
    expect("pto.vsts" in vecscope_surface_text, "public vecscope body should allow vector stores")
    expect("pto.vmulscvt" in vmulscvt_surface_text, "vmulscvt(...) should lower to pto.vmulscvt")
    expect('\"A\"' in vmulscvt_surface_text, "vmulscvt(..., rnd=VcvtRoundMode.A) should preserve the authored round token")
    expect('\"EVEN\"' in vmulscvt_surface_text, "vmulscvt(..., part=PartMode.EVEN) should preserve the authored part token")
    expect("!pto.vreg<128xf16>" in vmulscvt_surface_text, "vmulscvt(f32 -> f16) should infer the packed f16 result type")
    expect("pto.vmula" in vmula_surface_text, "vmula(...) should lower to pto.vmula")
    expect("!pto.vreg<64xf32>" in vmula_surface_text, "vmula(f32, f32, f32) should infer the f32 vector result type")
    expect("pto.vmadd" in vmadd_surface_text, "vmadd(...) should lower to pto.vmadd")
    expect("!pto.vreg<64xf32>" in vmadd_surface_text, "vmadd(f32, f32, f32) should infer the f32 vector result type")
    expect("pto.vsstb" in vsstb_post_update_surface_text, "vsstb(..., post_update=ON) should still lower through pto.vsstb on the current VPTO IR")
    expect("-> !pto.ptr<f32, ub>" in vsstb_post_update_surface_text, "vsstb(..., post_update=ON) should request the updated destination pointer result")
    expect("pto.mte_l1_l0b" in public_surface_text, "mte_l1_l0b(...) should lower to pto.mte_l1_l0b")
    expect(
        public_surface_text.count("pto.mte_l1_l0a") >= 2,
        "explicit mte_l1_l0a(...) should remain on the public wrapper op",
    )
    expect(
        public_surface_text.count("pto.mte_l1_l0b") >= 2,
        "explicit mte_l1_l0b(...) should remain on the public wrapper op",
    )
    expect(
        explicit_fp4_s4_staging_text.count("pto.mte_l1_l0a") == 1,
        "FP4 explicit mte_l1_l0a(...) should trace through pto.mte_l1_l0a",
    )
    expect(
        explicit_fp4_s4_staging_text.count("pto.mte_l1_l0b") == 1,
        "FP4 explicit mte_l1_l0b(...) should trace through pto.mte_l1_l0b",
    )
    expect(
        "pto.load_cbuf_to_ca_s4" not in explicit_fp4_s4_staging_text and
        "pto.load_cbuf_to_cb_s4" not in explicit_fp4_s4_staging_text,
        "PTODSL must not expose internal raw S4 load operations",
    )
    explicit_ca_controls = (
        r"pto\.mte_l1_l0a .*%c4_i64(?:_\d+)?, %c0_i64(?:_\d+)?, "
        r"%c4_i64(?:_\d+)?, %c16_i64(?:_\d+)?, "
        r"%c16_i64(?:_\d+)?, %c4_i64(?:_\d+)? \{transpose = true\}"
    )
    expect(
        re.search(explicit_ca_controls, public_surface_text) is not None,
        "explicit mte_l1_l0a controls should remain on the public wrapper",
    )
    expect("pto.mte_l1_l0a_mx" in public_surface_text, "mte_l1_l0a_mx(...) should lower to pto.mte_l1_l0a_mx")
    expect("pto.mte_l1_l0b_mx" in public_surface_text, "mte_l1_l0b_mx(...) should lower to pto.mte_l1_l0b_mx")
    expect(
        explicit_mx_scale_staging_text.count("!pto.ptr<!pto.f8E8M0, l1>") >= 2,
        "explicit MX staging should preserve E8M0 L1 source pointers",
    )
    for op_name in ("mte_l1_l0a_mx", "mte_l1_l0b_mx"):
        expect(
            re.search(
                rf"pto\.{op_name} %\d+, %\d+, %c3_i64(?:_\d+)?, %c5_i64(?:_\d+)?, "
                r"%c16_i64(?:_\d+)?, %c2_i64(?:_\d+)?, %c8_i64(?:_\d+)?, %c2_i64(?:_\d+)",
                explicit_mx_scale_staging_text,
            ) is not None,
            f"{op_name}(...) should preserve every explicit MX control operand",
        )
    expect("pto.tmatmul.mx" in public_surface_text, "pto.tile.matmul_mx should lower to pto.tmatmul.mx")
    expect("pto.tmatmul.mx.acc" in public_surface_text, "pto.tile.matmul_mx_acc should lower to pto.tmatmul.mx.acc")
    expect("pto.tmatmul.mx.bias" in public_surface_text, "pto.tile.matmul_mx_bias should lower to pto.tmatmul.mx.bias")
    expect("pto.tgemv.mx" in public_surface_text, "pto.tile.gemv_mx should lower to pto.tgemv.mx")
    expect("pto.tgemv.mx.acc" in public_surface_text, "pto.tile.gemv_mx_acc should lower to pto.tgemv.mx.acc")
    expect("pto.tgemv.mx.bias" in public_surface_text, "pto.tile.gemv_mx_bias should lower to pto.tgemv.mx.bias")
    expect("pto.mte_l0c_l1" in public_surface_text, "mte_l0c_l1(...) should lower to pto.mte_l0c_l1")
    expect("pto.mte_l0c_gm" in public_surface_text, "mte_l0c_gm(...) should lower to pto.mte_l0c_gm")
    expect(public_surface_text.count("pto.mte_l0c_ub") >= 2, "mte_l0c_ub(...) should lower sub-block and split modes")
    expect("pto.mad" in public_surface_text, "mad(...) should lower to pto.mad")
    expect("unit_flag(check_only)" in public_surface_text, "mad unit_flag option should lower to PTO unit_flag clause")
    expect("disable_gemv" in public_surface_text, "mad disable_gemv option should lower to PTO disable_gemv clause")
    expect("tf32_mode(round_even)" in public_surface_text, "mad tf32_mode option should lower to PTO tf32_mode clause")
    expect("n_dir" in public_surface_text, "mad n_dir option should lower to PTO n_dir clause")
    expect("pto.mad_acc" in public_surface_text, "mad_acc(...) should lower to pto.mad_acc")
    expect("pto.mad_bias" in public_surface_text, "mad_bias(...) should lower to pto.mad_bias")
    expect("pto.mad_mx" in public_surface_text, "mad_mx(...) should lower to pto.mad_mx")
    expect("pto.mad_mx_acc" in public_surface_text, "mad_mx_acc(...) should lower to pto.mad_mx_acc")
    expect("pto.mad_mx_bias" in public_surface_text, "mad_mx_bias(...) should lower to pto.mad_mx_bias")
    expect("!pto.tile_buf<vec, 128x64xf8E4M3FN>" in low_precision_storage_text, "low-precision tile allocation should preserve float8 element types in MLIR")
    expect("!pto.tile_buf<vec, 64x64x!pto.hif8>" in low_precision_storage_text, "low-precision tile allocation should preserve HiF8 element types in MLIR")
    expect("pto.vlds" in pointer_vlds_text, "vlds(ptr, offset) should still lower to pto.vlds")
    expect("!pto.vreg<64xf32>" in pointer_vlds_text, "vlds(ptr, offset) should infer the result vreg type from the pointer element type")
    expect('dist = "BRC_B32"' in pointer_vlds_text, 'vlds(ptr, offset, dist="BRC_B32") should lower the authored load distribution')
    expect("pto.vbitcast" in pointer_vlds_text, "vbitcast(...) should lower to pto.vbitcast")
    expect("!pto.vreg<128xf16>" in pointer_vlds_text, "vbitcast(vec, pto.f16) should preserve the 256-byte payload while adjusting the lane count")
    expect(mask_bitcast_text.count("pto.pbitcast") == 2, "pbitcast(...) should lower to pto.pbitcast for each authored mask reinterpretation")
    expect("!pto.mask<b16>" in mask_bitcast_text, "pbitcast(mask, pto.mask_b16) should materialize the requested result mask type")
    expect("!pto.mask<b32>" in mask_bitcast_text, "pbitcast(mask, pto.mask_b32) should materialize the requested result mask type")
    expect_raises(
        TypeError,
        lambda: dynamic_buf_invalid_type_probe.compile(),
        "get_buf",
    )
    expect_raises(
        TypeError,
        lambda: dynamic_rls_buf_invalid_type_probe.compile(),
        "rls_buf",
    )
    expect_raises(
        ValueError,
        lambda: vmulscvt_surface_invalid_attr_probe.compile(),
        "vmulscvt(..., rnd=...) currently only supports A",
    )
    expect_raises(
        TypeError,
        lambda: pto.vmulscvt(None, None, None, rnd=pto.VcvtRoundMode.A, part=pto.PartMode.EVEN, sat=pto.VcvtSatMode.SAT),
        "got an unexpected keyword argument 'sat'",
    )
    expect_raises(
        TypeError,
        lambda: vcvt_surface_invalid_dtype_pair_probe.compile(),
        "vcvt(src, to_dtype, mask) currently does not support the dtype pair f32 -> u16",
    )
    expect_raises(
        TypeError,
        lambda: low_precision_vadd_invalid_probe.compile(),
        "does not support low-precision vreg elements yet",
    )
    expect_raises(
        TypeError,
        lambda: low_precision_vexp_invalid_probe.compile(),
        "does not support low-precision vreg elements yet",
    )
    expect_raises(
        TypeError,
        lambda: low_precision_vmuls_invalid_probe.compile(),
        "does not support low-precision vreg elements yet",
    )
    expect_raises(
        TypeError,
        lambda: low_precision_vsel_invalid_probe.compile(),
        "does not support low-precision vreg elements yet",
    )
    expect_raises(
        TypeError,
        lambda: low_precision_vmula_invalid_probe.compile(),
        "does not support low-precision vreg elements yet",
    )
    expect_raises(
        TypeError,
        lambda: low_precision_vmadd_invalid_probe.compile(),
        "does not support low-precision vreg elements yet",
    )
    expect_raises(
        TypeError,
        lambda: vcvt_low_precision_invalid_dtype_pair_probe.compile(),
        "vcvt(src, to_dtype, mask) currently does not support the dtype pair f8e4m3 -> bf16",
    )
    expect_raises(
        TypeError,
        lambda: vcvt_low_precision_invalid_dtype_pair_probe_2.compile(),
        "vcvt(src, to_dtype, mask) currently does not support the dtype pair f32 -> f4e1m2x2",
    )
    expect_raises(
        ValueError,
        lambda: vcvt_low_precision_invalid_part_probe.compile(),
        "part must be P0, P1, P2, or P3",
    )
    expect_raises(
        ValueError,
        lambda: vcvt_low_precision_missing_attr_probe.compile(),
        "requires sat for dtype pair f32 -> f8e4m3",
    )
    expect_raises(
        TypeError,
        lambda: vmulscvt_surface_invalid_dtype_pair_probe.compile(),
        "vmulscvt(src, scalar, mask) currently only supports the dtype pair f32 -> f16",
    )
    expect_raises(
        TypeError,
        lambda: vpack_surface_invalid_shape_probe.compile(),
        "vpack(src, part) currently supports only the source/result shape pairs s32/u32 -> u16 and s16/u16 -> u8",
    )
    expect_raises(
        TypeError,
        lambda: vdup_surface_invalid_scalar_position_probe.compile(),
        "position is only valid for vector input",
    )
    expect_raises(
        ValueError,
        lambda: vpack_surface_invalid_part_probe.compile(),
        "vpack(src, part) does not support part",
    )
    expect("pto.pset_b8" in mask_surface_text, "pset_b8(...) should lower to pto.pset_b8")
    expect("pto.pset_b16" in mask_surface_text, "pset_b16(...) should lower to pto.pset_b16")
    expect("pto.pset_b32" in mask_surface_text, "pset_b32(...) should lower to pto.pset_b32")
    expect("pto.pge_b8" in mask_surface_text, "pge_b8(...) should lower to pto.pge_b8")
    expect("pto.pge_b16" in mask_surface_text, "pge_b16(...) should lower to pto.pge_b16")
    expect("pto.pge_b32" in mask_surface_text, "pge_b32(...) should lower to pto.pge_b32")
    expect("pto.plt_b8" in mask_surface_text, "plt_b8(...) should lower to pto.plt_b8")
    expect("pto.plt_b16" in mask_surface_text, "plt_b16(...) should lower to pto.plt_b16")
    expect(mask_surface_text.count("pto.plt_b32") >= 1, "plt_b32(...) should still lower to pto.plt_b32")
    expect("pto.pand" in mask_surface_text, "pand(...) should lower to pto.pand")
    expect("pto.por" in mask_surface_text, "por(...) should lower to pto.por")
    expect("pto.pxor" in mask_surface_text, "pxor(...) should lower to pto.pxor")
    expect("pto.pnot" in mask_surface_text, "pnot(...) should lower to pto.pnot")
    expect("pto.psel" in mask_surface_text, "psel(...) should lower to pto.psel")
    expect("pto.ppack" in mask_surface_text, "ppack(...) should lower to pto.ppack")
    expect("pto.punpack" in mask_surface_text, "punpack(...) should lower to pto.punpack")
    expect(
        mask_surface_text.count('"HIGHER"') >= 2,
        "ppack/punpack should accept PredicatePart.HIGHER and preserve it in MLIR",
    )
    expect(
        '!pto.mask<b32> -> !pto.mask<b16>' in mask_surface_text,
        "ppack(..., to_type=pto.mask_b16) should materialize the requested narrowed result mask type",
    )
    expect(
        '!pto.mask<b16> -> !pto.mask<b32>' in mask_surface_text,
        "punpack(..., to_type=pto.mask_b32) should materialize the requested widened result mask type",
    )
    expect("pto.pintlv_b8" in mask_surface_text, "pintlv_b8(...) should lower to pto.pintlv_b8")
    expect("pto.pintlv_b16" in mask_surface_text, "pintlv_b16(...) should lower to pto.pintlv_b16")
    expect("pto.pintlv_b32" in mask_surface_text, "pintlv_b32(...) should lower to pto.pintlv_b32")
    expect("pto.pdintlv_b8" in mask_surface_text, "pdintlv_b8(...) should lower to pto.pdintlv_b8")
    expect("pto.pdintlv_b16" in mask_surface_text, "pdintlv_b16(...) should lower to pto.pdintlv_b16")
    expect("pto.pdintlv_b32" in mask_surface_text, "pdintlv_b32(...) should lower to pto.pdintlv_b32")
    expect("pto.vcmp" in mask_surface_text, "vcmp(...) should lower to pto.vcmp")
    expect("pto.vcmps" in mask_surface_text, "vcmps(...) should lower to pto.vcmps")
    expect('pto.vcmp %' in mask_surface_text and ', "eq"' in mask_surface_text, "vcmp(..., pto.CmpMode.EQ) should normalize to the compare attribute spelling")
    expect('pto.vcmps %' in mask_surface_text and ', "gt"' in mask_surface_text, "vcmps(..., pto.CmpMode.GT) should normalize to the compare attribute spelling")
    expect(mask_surface_text.count("pto.plds") == 3, "plds(...) should lower once per authored predicate load")
    expect(mask_surface_text.count("pto.psts") == 2, "psts(...) should lower once per authored predicate store")
    expect(mask_surface_text.count("pto.pstu") == 2, "pstu(...) should lower once per authored predicate unaligned store")
    expect(', "US"' in mask_surface_text, "plds(..., dist=pto.PredicateDist.US) should preserve the documented DIST token")
    expect(', "DS"' in mask_surface_text, "plds(..., dist=pto.PredicateDist.DS) should preserve the documented DIST token")
    expect(', "PK"' in mask_surface_text, "psts(..., dist=pto.PredicateDist.PK) should preserve the documented DIST token")
    expect("pto.init_align" in mask_surface_text, "init_align() should lower to pto.init_align")

    launch_handle = block64[1, None]
    expect(callable(launch_handle), "compiled[grid, stream] should return a launch callable")
    expect(hasattr(launch_handle, "__call__"), "launch handle should support __call__")
    source_launch_handle = source_native_build_compiled[1, None]
    expect(callable(source_launch_handle), "source-backed compiled[grid, stream] should return a launch callable")
    expect(hasattr(source_launch_handle, "__call__"), "source-backed launch handle should support __call__")

    print("ptodsl_jit_compile: PASS")
    os._exit(0)


_assert_ast_rewrite_nested_partial_assign_ssa_identity()


if __name__ == "__main__":
    main()
