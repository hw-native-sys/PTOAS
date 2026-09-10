# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Control-flow lowering engine for the PTODSL source-to-source AST rewrite."""

from __future__ import annotations

import ast
import copy
from dataclasses import dataclass

from ._ast_rewrite_errors import PTODSLAstRewriteError
from ._ast_rewrite_analysis import (
    _NameInfo,
    _SlotInfo,
    _SubscriptSlot,
    _definite_out_stmt,
    _drop_unreachable_tails,
    _is_pto_attr_call,
    _live_before_block,
    _live_before_stmt,
    _loop_control_flags,
    _loop_has_return,
    _name_info,
    _pto_attr,
    _range_triplet_and_hints,
    _read_before_assignment_names,
    _read_before_assignment_slots,
    _resolve_subscript_slots,
    _simple_name_targets,
    _slot_info,
    _slot_live_before_stmt,
    _try_eval_static_range,
)


def _flag_const(value):
    return ast.Call(
        func=_pto_attr("const"),
        args=[ast.Constant(1 if value else 0)],
        keywords=[ast.keyword(arg="dtype", value=_pto_attr("i1"))],
    )


def _name(name: str, ctx=ast.Load()):
    return ast.Name(id=name, ctx=ctx)


def _slot_subscript(slot: _SubscriptSlot, ctx=ast.Load()):
    return ast.Subscript(
        value=_name(slot.base),
        slice=ast.Constant(slot.index),
        ctx=ctx,
    )


def _map_subscript(map_name: str, index: int, ctx=ast.Load()):
    return ast.Subscript(
        value=_name(map_name),
        slice=ast.Constant(index),
        ctx=ctx,
    )


def _rewrite_static_range_for_body(node, static_env, static_iters, visit):
    """Rewrite body/orelse of a ``pto.static_range`` for under a temporary binding."""
    if not (_is_pto_attr_call(node.iter, "static_range") and isinstance(node.target, ast.Name)):
        return None
    values = _try_eval_static_range(node.iter, static_env, static_iters)
    if values is not None:
        old = static_iters.get(node.target.id)
        static_iters[node.target.id] = values
        try:
            node.body = [visit(stmt) for stmt in node.body]
        finally:
            if old is None:
                static_iters.pop(node.target.id, None)
            else:
                static_iters[node.target.id] = old
    else:
        node.body = [visit(stmt) for stmt in node.body]
    node.orelse = [visit(stmt) for stmt in node.orelse]
    return node


class _StaticRangeForRewriter(ast.NodeTransformer):
    """Lower ``pto.static_range`` loops under a shared lexical binding."""

    def __init__(self, static_env, static_iters=None):
        self._static_env = static_env
        self._static_iters = dict(static_iters or {})

    def visit_For(self, node):
        rewritten = _rewrite_static_range_for_body(
            node, self._static_env, self._static_iters, self.visit
        )
        if rewritten is None:
            return self.generic_visit(node)
        return rewritten


class _SlotCarryRewriter(_StaticRangeForRewriter):
    def __init__(self, slot_maps, static_env, static_iters=None):
        super().__init__(static_env, static_iters)
        self._slot_maps = slot_maps

    def visit_Subscript(self, node):
        slots = _resolve_subscript_slots(node, self._static_env, self._static_iters, require_static=False)
        if slots and len({slot.base for slot in slots}) == 1:
            base = next(iter(slots)).base
            if base in self._slot_maps and slots <= set(self._slot_maps[base]["slots"]):
                return ast.copy_location(
                    ast.Subscript(
                        value=_name(self._slot_maps[base]["map_name"]),
                        slice=copy.deepcopy(node.slice),
                        ctx=node.ctx,
                    ),
                    node,
                )
        return self.generic_visit(node)


class _SlotValueRewriter(_StaticRangeForRewriter):
    """Replace selected static list slots with scalar branch state names."""

    def __init__(self, slot_values, static_env, static_iters=None):
        super().__init__(static_env, static_iters)
        self._slot_values = dict(slot_values)

    def visit_Subscript(self, node):
        slots = _resolve_subscript_slots(node, self._static_env, self._static_iters, require_static=False)
        if len(slots) == 1:
            slot = next(iter(slots))
            value_name = self._slot_values.get(slot)
            if value_name is not None:
                return ast.copy_location(_name(value_name, node.ctx), node)
        return self.generic_visit(node)


class _ControlFlowExitVisitor(ast.NodeVisitor):
    def __init__(self, *, reject_bare_returns: bool):
        self.exit_node = None
        self._reject_bare_returns = reject_bare_returns

    def visit_Return(self, node):
        if self._reject_bare_returns:
            self.exit_node = node

    def visit_Yield(self, node):
        self.exit_node = node

    def visit_YieldFrom(self, node):
        self.exit_node = node

    def visit_FunctionDef(self, node):
        return

    def visit_AsyncFunctionDef(self, node):
        return

    def visit_Lambda(self, node):
        return

    def visit_ClassDef(self, node):
        return


def _reject_control_flow_exits(stmts, context: str, *, reject_bare_returns: bool):
    visitor = _ControlFlowExitVisitor(reject_bare_returns=reject_bare_returns)
    for stmt in stmts:
        visitor.visit(stmt)
        if visitor.exit_node is not None:
            raise PTODSLAstRewriteError(
                f"ast_rewrite=True does not support return/yield inside rewritten {context}; "
                "assign values to locals and return after the rewritten control flow"
            )


@dataclass
class _IfRewriteAnalysis:
    """Precomputed state for lowering one runtime ``if`` statement."""
    cond_name: str
    then_info: _NameInfo
    else_info: _NameInfo
    then_slot_info: _SlotInfo
    else_slot_info: _SlotInfo
    merge_slots: tuple
    merge_names: tuple
    old_value_names: dict
    control_state: object
    then_control: dict
    else_control: dict
    assigned_any: set


@dataclass
class _IfBranchPlan:
    """Generated names shared by the branch/merge lowering of one runtime if."""
    branch_name: str
    if_entry_names: dict
    slot_value_names: dict
    old_slot_value_names: dict


@dataclass
class _ForRangeSpec:
    """Range bounds plus hint keywords parsed from one runtime range-for."""
    start: object
    stop: object
    step: object
    hint_keywords: tuple


@dataclass
class _ForCarryPlan:
    """Reserved SSA names for the loop-carried names/slots of one plain for."""
    loop_carried: tuple
    loop_carried_slots: tuple
    slot_carry_names: dict
    slot_maps: dict


@dataclass
class _PlainForPlan:
    """Shared context for lowering one plain runtime for-loop with carries."""
    stmt: ast.stmt
    spec: _ForRangeSpec
    loop_name: str
    body: list
    static_iters: dict
    carry: _ForCarryPlan


@dataclass
class _ControlledForPlan:
    """Generated state names for one controlled (break/continue/else) for."""
    iv_name: str
    loop_carried: set
    skip_name: str
    did_break_name: str
    state_name: str
    state_names: tuple
    loop_name: str


@dataclass
class _WhilePlan:
    """Generated state names for one runtime while lowering."""
    loop_name: str
    carry_names: tuple
    active_name: str
    did_break_name: str
    state_name: str
    state_names: tuple
    controlled: bool


class _StateNameRewriter(ast.NodeTransformer):
    """Rewrite loop-state name loads to attribute loads off a state name."""

    def __init__(self, state_name, state_names):
        self._state_name = state_name
        self._state_names = frozenset(state_names)

    def visit_Name(self, node):
        if node.id in self._state_names and isinstance(node.ctx, ast.Load):
            return ast.copy_location(
                ast.Attribute(
                    value=_name(self._state_name), attr=node.id, ctx=ast.Load()
                ),
                node,
            )
        return node


class _WhileBoolLiteralRewriter(ast.NodeTransformer):
    """Materialize literal bool conditions as i1 constants for runtime while."""

    def visit_Constant(self, node):
        if isinstance(node.value, bool):
            return ast.copy_location(_flag_const(node.value), node)
        return node


class _ControlFlowRewriter:
    def __init__(
        self,
        static_env=None,
        *,
        section_uninitialized_aliases=None,
        reject_bare_returns: bool = False,
    ):
        self._static_env = dict(static_env or {})
        self._section_uninitialized_aliases = set(section_uninitialized_aliases or ())
        self._counter = 0
        # Each entry names the SSA flags used to emulate Python loop control
        # for the corresponding innermost runtime loop.
        self._loop_control_stack = []
        self._reject_bare_returns = reject_bare_returns

    @staticmethod
    def _build_dynamic_if_with_stmt(
        analysis, stmt, dynamic_then_body, dynamic_else_body, branch_name
    ):
        """Wrap the two executable branch bodies in the runtime ``pto.if_``."""
        with_stmt = ast.With(
            items=[
                ast.withitem(
                    context_expr=ast.Call(func=_pto_attr("if_"), args=[_name(analysis.cond_name)], keywords=[]),
                    optional_vars=_name(branch_name, ast.Store()),
                )
            ],
            body=[
                ast.With(
                    items=[
                        ast.withitem(
                            context_expr=ast.Attribute(
                                value=_name(branch_name),
                                attr="then_",
                                ctx=ast.Load(),
                            ),
                            optional_vars=None,
                        )
                    ],
                    body=dynamic_then_body or [ast.Pass()],
                    type_comment=None,
                )
            ],
            type_comment=None,
        )
        if stmt.orelse or dynamic_else_body:
            with_stmt.body.append(
                ast.With(
                    items=[
                        ast.withitem(
                            context_expr=ast.Attribute(
                                value=_name(branch_name),
                                attr="else_",
                                ctx=ast.Load(),
                            ),
                            optional_vars=None,
                        )
                    ],
                    body=dynamic_else_body or [ast.Pass()],
                    type_comment=None,
                )
            )
        return with_stmt

    @staticmethod
    def _assemble_dynamic_if_result(
        analysis,
        branch_plan,
        stmt,
        dynamic_body,
        trace_time_if,
    ):
        """Assemble the final statement list for a runtime ``if``."""
        result = [
            ast.Assign(
                targets=[_name(analysis.cond_name, ast.Store())],
                value=stmt.test,
            )
        ]
        result.extend(
            ast.Assign(
                targets=[_name(value_name, ast.Store())],
                value=_slot_subscript(slot),
            )
            for slot, value_name in branch_plan.slot_value_names.items()
        )
        result.extend(
            ast.Assign(
                targets=[_name(old_value_name, ast.Store())],
                value=_name(branch_plan.slot_value_names[slot]),
            )
            for slot, old_value_name in branch_plan.old_slot_value_names.items()
        )
        result.append(
            ast.copy_location(
                ast.If(
                    test=ast.Call(
                        func=_name("isinstance"),
                        args=[_name(analysis.cond_name), _name("bool")],
                        keywords=[],
                    ),
                    body=[trace_time_if],
                    orelse=dynamic_body,
                ),
                stmt,
            )
        )
        return result

    @staticmethod
    def _branch_assign(
        branch_name,
        names,
        *,
        entry_names,
        assigned_names,
        slot_value_names=None,
        old_slot_value_names=None,
        assigned_slots=None,
    ):
        slot_value_names = slot_value_names or {}
        old_slot_value_names = old_slot_value_names or {}
        assigned_slots = assigned_slots or set()
        keywords = [
            ast.keyword(
                arg=name,
                value=_name(name if name in assigned_names else entry_names[name]),
            )
            for name in names
        ]
        keywords.extend(
            ast.keyword(
                arg=value_name,
                value=_name(value_name if slot in assigned_slots else old_slot_value_names[slot]),
            )
            for slot, value_name in slot_value_names.items()
        )
        return ast.Expr(
            value=ast.Call(
                func=ast.Attribute(value=_name(branch_name), attr="assign", ctx=ast.Load()),
                args=[],
                keywords=keywords,
            )
        )

    @staticmethod
    def _validate_plain_for_step(step):
        """Reject a statically non-positive step for scf.for plain lowering."""
        # The plain path lowers to scf.for, whose control-flow lowering
        # compares the induction variable with the upper bound using a signed
        # less-than: a non-positive step would silently produce zero
        # iterations instead of Python range's descending iteration.  Only
        # loops with break/continue (the pto._while path) support negative
        # steps, so reject a constant non-positive step here.  Note that a
        # negative literal is a UnaryOp(USub, Constant), not a Constant, so
        # use literal_eval to see through it.  bool is an int subclass:
        # step=False (== 0) must be rejected like an explicit 0 (Python range
        # raises ValueError for it), while step=True (== 1) is legal Python -
        # normalize the literal to int 1 because downstream index coercion
        # rejects bool values.
        try:
            step_const = ast.literal_eval(step)
        except (ValueError, TypeError, SyntaxError):
            step_const = None
        if isinstance(step_const, bool):
            if not step_const:
                raise PTODSLAstRewriteError(
                    "ast_rewrite=True range(...) / pto.range(...) loops require a non-zero step; "
                    "got step=0 (Python range raises ValueError for a zero step)."
                )
            if isinstance(step, ast.Constant):
                step.value = 1
        elif isinstance(step_const, int) and step_const <= 0:
            if step_const == 0:
                raise PTODSLAstRewriteError(
                    "ast_rewrite=True range(...) / pto.range(...) loops require a non-zero step; "
                    "got step=0 (Python range raises ValueError for a zero step)."
                )
            raise PTODSLAstRewriteError(
                "ast_rewrite=True range(...) / pto.range(...) loops require a positive step; "
                f"got step={step_const}. Loops with break/continue support negative steps "
                "via the pto._while lowering; a dynamic step must be positive at runtime."
            )
        return step

    @staticmethod
    def _build_plain_for_result(stmt, spec, body):
        """Assemble the simple ``with pto.for_(...) as iv:`` lowering."""
        with_stmt = ast.With(
            items=[
                ast.withitem(
                    context_expr=ast.Call(
                        func=_pto_attr("for_"),
                        args=[spec.start, spec.stop],
                        keywords=[
                            ast.keyword(arg="step", value=spec.step),
                            *spec.hint_keywords,
                        ],
                    ),
                    optional_vars=_name(stmt.target.id, ast.Store()),
                )
            ],
            body=body or [ast.Pass()],
            type_comment=None,
        )
        return [ast.copy_location(with_stmt, stmt)]

    @staticmethod
    def _for_carry_slot_initializers(loop_carried_slots, slot_carry_names):
        """Read each carried slot's entering value into its carry temporary."""
        return [
            ast.Assign(
                targets=[_name(slot_carry_names[slot], ast.Store())],
                value=_slot_subscript(slot),
            )
            for slot in loop_carried_slots
        ]

    @staticmethod
    def _for_carry_prologue(plan):
        """Build the loop-entry prologue that unpacks iv/carry values."""
        carry = plan.carry
        prologue = [
            ast.Assign(
                targets=[_name(plan.stmt.target.id, ast.Store())],
                value=ast.Attribute(value=_name(plan.loop_name), attr="iv", ctx=ast.Load()),
            )
        ]
        prologue.extend(
            ast.Assign(
                targets=[_name(name, ast.Store())],
                value=ast.Attribute(value=_name(plan.loop_name), attr=name, ctx=ast.Load()),
            )
            for name in carry.loop_carried
        )
        prologue.extend(
            ast.Assign(
                targets=[_name(carry.slot_carry_names[slot], ast.Store())],
                value=ast.Attribute(
                    value=_name(plan.loop_name),
                    attr=carry.slot_carry_names[slot],
                    ctx=ast.Load(),
                ),
            )
            for slot in carry.loop_carried_slots
        )
        prologue.extend(
            ast.Assign(
                targets=[_name(data["map_name"], ast.Store())],
                value=ast.Dict(
                    keys=[ast.Constant(slot.index) for slot in data["slots"]],
                    values=[_name(carry.slot_carry_names[slot]) for slot in data["slots"]],
                ),
            )
            for data in carry.slot_maps.values()
        )
        return prologue

    @staticmethod
    def _extend_for_carry_finals(
        result, loop_name, loop_carried, loop_carried_slots, slot_carry_names
    ):
        """Append the pto.for_ ``final`` reads for carried names and slots."""
        for name in loop_carried:
            result.append(
                ast.Assign(
                    targets=[_name(name, ast.Store())],
                    value=ast.Call(
                        func=ast.Attribute(value=_name(loop_name), attr="final", ctx=ast.Load()),
                        args=[ast.Constant(name)],
                        keywords=[],
                    ),
                )
            )
        for slot in loop_carried_slots:
            result.append(
                ast.Assign(
                    targets=[_slot_subscript(slot, ast.Store())],
                    value=ast.Call(
                        func=ast.Attribute(value=_name(loop_name), attr="final", ctx=ast.Load()),
                        args=[ast.Constant(slot_carry_names[slot])],
                        keywords=[],
                    ),
                )
            )

    @staticmethod
    def _build_controlled_condition_fn(plan, stop, step):
        """Build the sign-aware range condition lambda for pto._while."""
        # Python range has direction-dependent bounds.  The sign-aware form
        # also handles a runtime step; a zero step is rejected by the runtime
        # range semantics before entering useful code in normal callers.
        iv_lt = ast.Compare(left=_name(plan.iv_name), ops=[ast.Lt()],
                            comparators=[copy.deepcopy(stop)])
        iv_gt = ast.Compare(left=_name(plan.iv_name), ops=[ast.Gt()],
                            comparators=[copy.deepcopy(stop)])
        if isinstance(step, ast.Constant) and isinstance(step.value, int):
            if step.value > 0:
                range_cond = iv_lt
            elif step.value < 0:
                range_cond = iv_gt
            else:
                raise PTODSLAstRewriteError(
                    "ast_rewrite=True runtime range loops do not support a zero step"
                )
        else:
            step_positive = ast.Compare(left=copy.deepcopy(step), ops=[ast.Gt()], comparators=[ast.Constant(0)])
            step_negative = ast.Compare(left=copy.deepcopy(step), ops=[ast.Lt()], comparators=[ast.Constant(0)])
            range_cond = ast.BinOp(
                left=ast.BinOp(left=step_positive, op=ast.BitAnd(), right=iv_lt),
                op=ast.BitOr(),
                right=ast.BinOp(left=step_negative, op=ast.BitAnd(), right=iv_gt),
            )
        condition = ast.BinOp(
            left=range_cond,
            op=ast.BitAnd(),
            right=ast.Compare(left=_name(plan.did_break_name), ops=[ast.Eq()],
                              comparators=[_flag_const(False)]),
        )
        condition = _StateNameRewriter(plan.state_name, plan.state_names).visit(condition)
        ast.fix_missing_locations(condition)
        return ast.Lambda(
            args=ast.arguments(posonlyargs=[], args=[ast.arg(arg=plan.state_name)], vararg=None,
                               kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]),
            body=condition,
        )

    @staticmethod
    def _build_controlled_for_with_stmt(plan, step, body):
        """Wrap the controlled loop body with its prologue and update call."""
        prologue = [
            ast.Assign(targets=[_name(name, ast.Store())],
                       value=ast.Attribute(value=_name(plan.loop_name), attr=name,
                                           ctx=ast.Load()))
            for name in plan.state_names
        ]
        # active is per-iteration execution state; did_break remains sticky.
        # The body is segmented by _guard_control_tail on the merged active
        # value, so no whole-body guard is needed here.
        prologue.append(ast.Assign(targets=[_name(plan.skip_name, ast.Store())],
                                   value=_flag_const(True)))
        updates = [
            ast.keyword(arg=plan.iv_name, value=ast.BinOp(
                left=_name(plan.iv_name), op=ast.Add(),
                right=copy.deepcopy(step),
            )),
            *[ast.keyword(arg=name, value=_name(name))
              for name in sorted(plan.loop_carried)],
            ast.keyword(arg=plan.skip_name, value=_name(plan.skip_name)),
            ast.keyword(arg=plan.did_break_name, value=_name(plan.did_break_name)),
        ]
        body.append(ast.Expr(value=ast.Call(
            func=ast.Attribute(value=_name(plan.loop_name), attr="update",
                               ctx=ast.Load()),
            args=[], keywords=updates)))
        return ast.With(
            items=[ast.withitem(context_expr=_name(plan.loop_name), optional_vars=None)],
            body=prologue + body,
            type_comment=None,
        )

    @staticmethod
    def _extend_controlled_loop_finals(result, plan, live_after):
        """Append the pto._while ``final`` reads for state carried past the loop."""
        result.extend(
            ast.Assign(
                targets=[_name(name, ast.Store())],
                value=ast.Call(
                    func=ast.Attribute(
                        value=_name(plan.loop_name), attr="final", ctx=ast.Load()
                    ),
                    args=[ast.Constant(name)],
                    keywords=[],
                ),
            )
            for name in plan.state_names
            if name in live_after or name in {plan.skip_name, plan.did_break_name}
        )

    @staticmethod
    def _build_while_condition_fn(stmt, state_name, state_names, did_break_name, controlled):
        """Build the state-name-rewritten condition lambda for pto._while."""
        condition = _StateNameRewriter(state_name, state_names).visit(copy.deepcopy(stmt.test))
        if controlled:
            condition = ast.BinOp(
                left=condition,
                op=ast.BitAnd(),
                right=ast.Compare(
                    left=_name(did_break_name), ops=[ast.Eq()],
                    comparators=[_flag_const(False)],
                ),
            )
            condition = _StateNameRewriter(state_name, state_names).visit(condition)
        condition = _WhileBoolLiteralRewriter().visit(condition)
        ast.fix_missing_locations(condition)
        return ast.Lambda(
            args=ast.arguments(
                posonlyargs=[], args=[ast.arg(arg=state_name)], vararg=None,
                kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]),
            body=condition,
        )

    @staticmethod
    def _while_initial_keywords(state_names, active_name, did_break_name):
        """Build the pto._while(...) initial values for each state name."""
        return [
            ast.keyword(
                arg=name,
                value=(_flag_const(True) if name == active_name else
                       _flag_const(False) if name == did_break_name else
                       _name(name)),
            ) for name in state_names
        ]

    @staticmethod
    def _build_while_with_stmt(loop_name, state_names, active_name, controlled, body):
        """Wrap the while body with its prologue, active reset, and update."""
        prologue = [
            ast.Assign(
                targets=[_name(name, ast.Store())],
                value=ast.Attribute(value=_name(loop_name), attr=name,
                                    ctx=ast.Load()),
            ) for name in state_names
        ]
        if controlled:
            # ``active`` is an iteration-local execution flag.  A continue
            # clears it for the remainder of this body, then the next body
            # entry re-enables it.  ``did_break`` is sticky across iterations.
            # The body is segmented by _guard_control_tail on the merged
            # active value, so no whole-body guard is needed here.
            prologue.append(ast.Assign(
                targets=[_name(active_name, ast.Store())], value=_flag_const(True)))
        update = ast.Expr(
            value=ast.Call(
                func=ast.Attribute(value=_name(loop_name), attr="update",
                                   ctx=ast.Load()),
                args=[], keywords=[
                    ast.keyword(arg=name, value=_name(name))
                    for name in state_names
                ],
            )
        )
        return ast.With(
            items=[ast.withitem(context_expr=_name(loop_name), optional_vars=None)],
            body=prologue + body + [update], type_comment=None,
        )

    @staticmethod
    def _extend_while_loop_finals(result, plan, live_after):
        """Append pto._while ``final`` reads for while state used after the loop."""
        result.extend(
            ast.Assign(
                targets=[_name(name, ast.Store())],
                value=ast.Call(
                    func=ast.Attribute(value=_name(plan.loop_name), attr="final",
                                       ctx=ast.Load()),
                    args=[ast.Constant(name)], keywords=[],
                ),
            ) for name in plan.carry_names if name in live_after
        )
        if plan.controlled:
            result.extend(
                ast.Assign(
                    targets=[_name(name, ast.Store())],
                    value=ast.Call(
                        func=ast.Attribute(value=_name(plan.loop_name), attr="final",
                                           ctx=ast.Load()),
                        args=[ast.Constant(name)], keywords=[]),
                ) for name in (plan.active_name, plan.did_break_name)
            )

    @staticmethod
    def _nested_block_entry_bindings(stmt, field, bound_before):
        """Return names bound on entry to a rewritten statement block field."""
        entry_bindings = set(bound_before or ())
        if field == "body" and isinstance(stmt, ast.With):
            # The with-as targets are definitely bound inside the body.
            for item in stmt.items:
                if item.optional_vars is not None:
                    entry_bindings |= _simple_name_targets(item.optional_vars)
        return entry_bindings

    def rewrite_block(
        self, stmts, *, live_after, live_after_slots=None, allow_loop_control=False,
        static_iters=None, bound_on_entry=None,
    ):
        rewritten_reversed = []
        live = set(live_after)
        live_slots = set(live_after_slots or ())
        static_iters = dict(static_iters or {})
        # Names definitely bound on entry to this block, plus the per-position
        # definite-in used to avoid turning unbound partial live-outs into loop
        # carries. Nested blocks inherit the definite-in of their enclosing
        # statement; the root block starts from the function parameters.
        definite_in = set(bound_on_entry or ())
        bound_before_by_stmt = {}
        for preceding in stmts:
            bound_before_by_stmt[id(preceding)] = set(definite_in)
            definite_in = _definite_out_stmt(preceding, definite_in, self._static_env, static_iters)
        control = self._loop_control_stack[-1] if self._loop_control_stack else None
        tail_assigned = set()
        tail_flags = {"break": False, "continue": False}
        for stmt in reversed(stmts):
            # Compute liveness from the authored AST before rewrite_stmt mutates
            # sibling statements in-place, otherwise later rewrites can pollute
            # earlier live-after analysis.
            live_before = _live_before_stmt(stmt, live)
            live_before_slots = _slot_live_before_stmt(stmt, live_slots, self._static_env, static_iters)
            stmt_stores = _name_info(stmt).stores
            stmt_flags = _loop_control_flags([stmt]) if control is not None else None
            rewritten = self.rewrite_stmt(
                stmt,
                live_after=live,
                live_after_slots=live_slots,
                allow_loop_control=allow_loop_control,
                static_iters=static_iters,
                bound_before=bound_before_by_stmt.get(id(stmt), set()),
            )
            # When this statement can stop the current iteration (top-level
            # break/continue, or a dynamic if containing one), the already-
            # rewritten tail must run only while the loop is still active.
            guarded_tail = self._guard_control_tail(
                stmt, control=control, tail=rewritten_reversed,
                tail_assigned=tail_assigned, tail_flags=tail_flags, live=live,
            )
            if guarded_tail is not None:
                # The tail now lives inside the guard: replace the
                # accumulator instead of prepending, or the tail nodes
                # would be shared by the guard and the block.
                rewritten_reversed = rewritten + guarded_tail
            else:
                rewritten_reversed[:0] = rewritten
            tail_assigned |= stmt_stores
            if stmt_flags is not None:
                tail_flags["break"] |= stmt_flags["break"]
                tail_flags["continue"] |= stmt_flags["continue"]
            live = live_before
            live_slots = live_before_slots
        return rewritten_reversed

    def rewrite_stmt(
        self, stmt, *, live_after, live_after_slots=None, allow_loop_control=False,
        static_iters=None, bound_before=None,
    ):
        live_after_slots = set(live_after_slots or ())
        static_iters = dict(static_iters or {})
        dispatched = self._dispatch_control_flow_stmt(
            stmt,
            live_after=live_after,
            live_after_slots=live_after_slots,
            allow_loop_control=allow_loop_control,
            static_iters=static_iters,
            bound_before=bound_before,
        )
        if dispatched is not None:
            return dispatched
        if isinstance(stmt, (ast.Break, ast.Continue)):
            return self._rewrite_loop_control_stmt(stmt, allow_loop_control=allow_loop_control)
        return [
            self._rewrite_nested(
                stmt,
                live_after=live_after,
                live_after_slots=live_after_slots,
                allow_loop_control=allow_loop_control,
                static_iters=static_iters,
                bound_before=bound_before,
            )
        ]

    def _fresh(self, prefix: str) -> str:
        value = f"__pto_ast_{prefix}_{self._counter}"
        self._counter += 1
        return value

    def _current_value(self, name, *, requires_pre_if_value=False):
        if requires_pre_if_value and name in self._section_uninitialized_aliases:
            raise PTODSLAstRewriteError(
                "ast_rewrite=True runtime if reads a section-local value before it is initialized; "
                f"initialize {name!r} before the conditional"
            )
        return _name(name)

    def _guard_control_tail(self, stmt, *, control, tail, tail_assigned, tail_flags, live,
                            forced_merge_names=()):
        """Guard the already-rewritten tail of a break/continue statement.

        When ``stmt`` can stop the current iteration (a top-level break/continue,
        or a dynamic if whose branches contain one, per ``_loop_control_flags``),
        every statement after it may only run while the loop is still active.
        The tail therefore executes under a ``pto.if_(active)`` guard whose
        condition is the merged active flag from the preceding statements, so
        statements after the transfer are skipped for that iteration.

        ``tail_assigned`` holds the names the tail assigns; names that are also
        live after the guard point are merged through the guard so later
        statements and the loop ``update`` keep consistent values.  Loop-carried
        names can look dead at the transfer point when their last authored read
        is before the break/continue, so ``forced_merge_names`` keeps tail
        assignments to those names from being trapped inside the guard region.
        ``tail_flags``
        records whether the tail itself contains control transfers: a tail that
        assigns ``active``/``did_break`` (they are not ``ast.Assign`` stores, so
        they never appear in ``tail_assigned``) must also merge them out, or the
        loop ``update`` would read flags whose SSA values are defined inside the
        guard region.  Controlled loops reject static subscript stores up front,
        so slots never need to participate here.  ``rewrite_block`` and
        ``_rewrite_loop_body`` share this segmentation; nested branch bodies pick
        up the current loop's control from the stack top automatically.

        Complexity note: a block with N control-transfer points produces N
        nested guards (each transfer's guard wraps the already-guarded tail
        behind it), with per-guard merge sets bounded by the names live at
        that point.  This nesting is inherent to predicating the tail: once an
        earlier transfer has fired, a later transfer statement must not even
        evaluate its condition, so the guards cannot be flattened into sibling
        regions without redesigning flag updates as data-flow (e.g.
        ``did_break |= cond & active``).  Typical kernels have very few
        transfer points per block, so the nesting is accepted deliberately
        rather than capped or merged.
        """
        if control is None or not tail:
            return None
        flags = _loop_control_flags([stmt])
        if not (flags["break"] or flags["continue"]):
            return None
        flag_names = set()
        if tail_flags["break"] or tail_flags["continue"]:
            flag_names.add(control["active"])
        if tail_flags["break"]:
            flag_names.add(control["did_break"])
        forced_merge_names = set(forced_merge_names)
        forced_or_live = set(live) | forced_merge_names
        value_merge_names = set(tail_assigned) & forced_or_live
        if value_merge_names:
            flag_names.add(control["active"])
        merge_names = sorted((value_merge_names | flag_names) & forced_or_live)
        return self._guard_block(
            _name(control["active"]),
            tail,
            merge_names=merge_names,
            assigned_names=set(tail_assigned) | flag_names,
        )

    def _rewrite_loop_body(self, stmts, *, live_after, live_after_slots=None, static_iters=None,
                           control=None, bound_on_entry=None, forced_tail_merge_names=()):
        """Rewrite loop statements while keeping each authored statement atomic.

        A rewritten dynamic ``if`` may contain several setup/branch/merge
        operations.  They must live in one ``scf.if`` region when the loop has
        been stopped by break/continue; wrapping each generated operation
        separately would create sibling-region SSA dominance violations.
        """
        rewritten_reversed = []
        live = set(live_after)
        if control is not None:
            live |= {control["active"], control["did_break"]}
        live_slots = set(live_after_slots or ())
        static_iters = dict(static_iters or {})
        definite_in = set(bound_on_entry or ())
        bound_before_by_stmt = {}
        for preceding in stmts:
            bound_before_by_stmt[id(preceding)] = set(definite_in)
            definite_in = _definite_out_stmt(preceding, definite_in, self._static_env, static_iters)
        tail_assigned = set()
        tail_flags = {"break": False, "continue": False}
        for stmt in reversed(stmts):
            live_before = _live_before_stmt(stmt, live)
            live_before_slots = _slot_live_before_stmt(stmt, live_slots, self._static_env, static_iters)
            rewrite_live = live
            if control is not None:
                rewrite_live = set(rewrite_live) | {control["active"], control["did_break"]}
            stmt_stores = _name_info(stmt).stores
            stmt_flags = _loop_control_flags([stmt]) if control is not None else None
            group = self.rewrite_stmt(
                stmt,
                live_after=rewrite_live,
                live_after_slots=live_slots,
                allow_loop_control=False,
                static_iters=static_iters,
                bound_before=bound_before_by_stmt.get(id(stmt), set()),
            )
            # Segment the already-rewritten tail on the merged active value
            # when this statement can stop the current iteration.
            guarded_tail = self._guard_control_tail(
                stmt, control=control, tail=rewritten_reversed,
                tail_assigned=tail_assigned, tail_flags=tail_flags, live=live,
                forced_merge_names=forced_tail_merge_names,
            )
            if guarded_tail is not None:
                # The tail now lives inside the guard: replace the
                # accumulator instead of prepending, or the tail nodes
                # would be shared by the guard and the block.
                rewritten_reversed = group + guarded_tail
            else:
                rewritten_reversed[:0] = group
            tail_assigned |= stmt_stores
            if stmt_flags is not None:
                tail_flags["break"] |= stmt_flags["break"]
                tail_flags["continue"] |= stmt_flags["continue"]
            live = set(live_before)
            if control is not None:
                live |= {control["active"], control["did_break"]}
            live_slots = live_before_slots
        return rewritten_reversed

    def _rewrite_nested(
        self, stmt, *, live_after, live_after_slots=None, allow_loop_control=False,
        static_iters=None, bound_before=None,
    ):
        live_after_slots = set(live_after_slots or ())
        static_iters = dict(static_iters or {})
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return self._rewrite_inner_function_def(stmt)
        if isinstance(stmt, (ast.Lambda, ast.ClassDef)):
            return stmt
        return self._rewrite_nested_statement_fields(
            stmt,
            live_after=live_after,
            live_after_slots=live_after_slots,
            allow_loop_control=allow_loop_control,
            static_iters=static_iters,
            bound_before=bound_before,
        )

    def _dispatch_control_flow_stmt(
        self, stmt, *, live_after, live_after_slots, allow_loop_control, static_iters,
        bound_before,
    ):
        """Forward If/For/While statements to their dedicated lowerings."""
        if isinstance(stmt, ast.If):
            return self._rewrite_if(
                stmt,
                live_after=live_after,
                live_after_slots=live_after_slots,
                allow_loop_control=allow_loop_control,
                static_iters=static_iters,
                bound_before=bound_before,
            )
        if isinstance(stmt, ast.For):
            return self._rewrite_for(
                stmt,
                live_after=live_after,
                live_after_slots=live_after_slots,
                allow_loop_control=allow_loop_control,
                static_iters=static_iters,
                bound_before=bound_before,
            )
        if isinstance(stmt, ast.While):
            return self._rewrite_while(
                stmt,
                live_after=live_after,
                live_after_slots=live_after_slots,
                allow_loop_control=allow_loop_control,
                static_iters=static_iters,
                bound_before=bound_before,
            )
        return None

    def _rewrite_loop_control_stmt(self, stmt, *, allow_loop_control):
        """Lower a top-level break/continue to flag assignments or reject it."""
        if self._loop_control_stack:
            control = self._loop_control_stack[-1]
            assigns = []
            if isinstance(stmt, ast.Break):
                assigns.append(ast.Assign(
                    targets=[_name(control["did_break"], ast.Store())],
                    value=_flag_const(True),
                ))
            assigns.append(ast.Assign(
                targets=[_name(control["active"], ast.Store())],
                value=_flag_const(False),
            ))
            return [ast.copy_location(item, stmt) for item in assigns]
        if allow_loop_control:
            return [stmt]
        raise PTODSLAstRewriteError(
            "ast_rewrite=True does not support break/continue in rewritten control flow"
        )

    def _rewrite_inner_function_def(self, stmt):
        """Rewrite the body of a nested function definition."""
        inner_params = set()
        for arg in stmt.args.posonlyargs + stmt.args.args + stmt.args.kwonlyargs:
            inner_params.add(arg.arg)
        stmt.body = self.rewrite_block(
            stmt.body,
            live_after=set(),
            live_after_slots=set(),
            allow_loop_control=False,
            static_iters={},
            bound_on_entry=inner_params,
        )
        return stmt

    def _rewrite_nested_statement_fields(
        self, stmt, *, live_after, live_after_slots, allow_loop_control, static_iters,
        bound_before,
    ):
        """Walk a non-control statement's nested blocks and expressions."""
        for field, value in ast.iter_fields(stmt):
            if field in {"body", "orelse", "finalbody"} and isinstance(value, list):
                setattr(
                    stmt,
                    field,
                    self.rewrite_block(
                        value,
                        live_after=live_after,
                        live_after_slots=live_after_slots,
                        allow_loop_control=allow_loop_control,
                        static_iters=static_iters,
                        bound_on_entry=self._nested_block_entry_bindings(
                            stmt, field, bound_before,
                        ),
                    ),
                )
            elif isinstance(value, ast.AST):
                self._rewrite_nested(
                    value,
                    live_after=live_after,
                    live_after_slots=live_after_slots,
                    allow_loop_control=allow_loop_control,
                    static_iters=static_iters,
                    bound_before=bound_before,
                )
            elif isinstance(value, list):
                self._rewrite_nested_field_items(
                    value,
                    live_after=live_after,
                    live_after_slots=live_after_slots,
                    allow_loop_control=allow_loop_control,
                    static_iters=static_iters,
                    bound_before=bound_before,
                )
        return stmt

    def _rewrite_nested_field_items(
        self, value, *, live_after, live_after_slots, allow_loop_control, static_iters,
        bound_before,
    ):
        """Rewrite the nested AST expressions carried by a statement field list."""
        for item in value:
            if isinstance(item, ast.AST):
                self._rewrite_nested(
                    item,
                    live_after=live_after,
                    live_after_slots=live_after_slots,
                    allow_loop_control=allow_loop_control,
                    static_iters=static_iters,
                    bound_before=bound_before,
                )

    def _rewrite_if(
        self, stmt, *, live_after, live_after_slots=None, allow_loop_control=False,
        static_iters=None, bound_before=None,
    ):
        live_after_slots = set(live_after_slots or ())
        static_iters = dict(static_iters or {})
        if _is_pto_attr_call(stmt.test, "const_expr"):
            return self._rewrite_const_expr_if(
                stmt,
                live_after=live_after,
                live_after_slots=live_after_slots,
                allow_loop_control=allow_loop_control,
                static_iters=static_iters,
                bound_before=bound_before,
            )
        return self._rewrite_dynamic_if(
            stmt,
            live_after=live_after,
            live_after_slots=live_after_slots,
            static_iters=static_iters,
            bound_before=bound_before,
        )

    def _rewrite_const_expr_if(
        self, stmt, *, live_after, live_after_slots, allow_loop_control, static_iters,
        bound_before,
    ):
        """Rewrite a compile-time ``pto.const_expr`` conditional in place."""
        stmt.body = self.rewrite_block(
            stmt.body,
            live_after=live_after,
            live_after_slots=live_after_slots,
            allow_loop_control=allow_loop_control,
            static_iters=static_iters,
            bound_on_entry=bound_before,
        )
        stmt.orelse = self.rewrite_block(
            stmt.orelse,
            live_after=live_after,
            live_after_slots=live_after_slots,
            allow_loop_control=allow_loop_control,
            static_iters=static_iters,
            bound_on_entry=bound_before,
        )
        return [stmt]

    def _rewrite_dynamic_if(self, stmt, *, live_after, live_after_slots, static_iters, bound_before):
        """Lower a runtime ``if`` into a traced branch and a BranchHandle merge."""
        _reject_control_flow_exits(
            stmt.body,
            "if branches",
            reject_bare_returns=self._reject_bare_returns,
        )
        _reject_control_flow_exits(
            stmt.orelse,
            "if branches",
            reject_bare_returns=self._reject_bare_returns,
        )
        analysis = self._analyze_if_rewrite(
            stmt, live_after=live_after, live_after_slots=live_after_slots,
            static_iters=static_iters,
        )
        then_body, else_body, trace_time_if, branch_name, if_entry_names = (
            self._prepare_if_branch_bodies(
                analysis, stmt, live_after=live_after, live_after_slots=live_after_slots,
                static_iters=static_iters, bound_before=bound_before,
            )
        )
        slot_value_names, old_slot_value_names = self._make_if_slot_value_names(
            analysis.merge_slots
        )
        branch_plan = _IfBranchPlan(
            branch_name=branch_name,
            if_entry_names=if_entry_names,
            slot_value_names=slot_value_names,
            old_slot_value_names=old_slot_value_names,
        )
        dynamic_then_body, dynamic_else_body = self._restore_and_rewrite_if_branch_bodies(
            analysis, branch_plan, then_body, else_body, static_iters,
        )
        dynamic_then_body, dynamic_else_body = self._append_dynamic_if_branch_merges(
            analysis, branch_plan, stmt, dynamic_then_body, dynamic_else_body,
        )
        ast.fix_missing_locations(trace_time_if)
        with_stmt = self._build_dynamic_if_with_stmt(
            analysis, stmt, dynamic_then_body, dynamic_else_body, branch_plan.branch_name
        )
        dynamic_body = self._build_dynamic_if_body(
            analysis, branch_plan.if_entry_names, with_stmt,
            branch_plan.branch_name, branch_plan.slot_value_names,
        )
        return self._assemble_dynamic_if_result(
            analysis, branch_plan, stmt, dynamic_body, trace_time_if
        )

    def _analyze_if_rewrite(self, stmt, *, live_after, live_after_slots, static_iters):
        """Compute liveness/merge state needed to lower a runtime ``if``."""
        cond_name = self._fresh("cond")
        then_info = _name_info(stmt.body)
        else_info = _name_info(stmt.orelse)
        then_slot_info = _slot_info(stmt.body, self._static_env, static_iters)
        else_slot_info = _slot_info(stmt.orelse, self._static_env, static_iters)
        assigned_slots = (
            then_slot_info.stores
            | else_slot_info.stores
        )
        merge_slots = tuple(sorted(live_after_slots & assigned_slots))
        assigned_any = then_info.stores | else_info.stores
        control_state = self._loop_control_stack[-1] if self._loop_control_stack else None
        then_control = _loop_control_flags(stmt.body) if control_state else {"break": False, "continue": False}
        else_control = _loop_control_flags(stmt.orelse) if control_state else {"break": False, "continue": False}
        if control_state:
            branch_breaks = then_control["break"] or else_control["break"]
            branch_exits = branch_breaks or then_control["continue"] or else_control["continue"]
            if branch_exits:
                assigned_any.add(control_state["active"])
            if branch_breaks:
                assigned_any.add(control_state["did_break"])
        merge_names = tuple(sorted(live_after & assigned_any))
        old_value_names = {
            name: self._fresh(f"old_{name}")
            for name in merge_names
            if name not in then_info.stores or name not in else_info.stores
        }
        return _IfRewriteAnalysis(
            cond_name=cond_name,
            then_info=then_info,
            else_info=else_info,
            then_slot_info=then_slot_info,
            else_slot_info=else_slot_info,
            merge_slots=merge_slots,
            merge_names=merge_names,
            old_value_names=old_value_names,
            control_state=control_state,
            then_control=then_control,
            else_control=else_control,
            assigned_any=assigned_any,
        )

    def _prepare_if_branch_bodies(self, analysis, stmt, *, live_after, live_after_slots, static_iters, bound_before):
        """Rewrite both branch blocks and build the trace-time ``if`` skeleton."""
        branch_live_after = set(live_after) | set(analysis.merge_names)
        branch_live_after_slots = set(live_after_slots) | set(analysis.merge_slots)
        # Variables not assigned on one side need their entering value at branch
        # merge time, and variables read inside a branch need a clean entering
        # environment. Snapshot the entering SSA value of exactly those names and
        # restore the Python binding at the top of each dynamic branch so sibling
        # branch tracing never observes the other branch's rebindings.
        branch_entry_names = set(analysis.old_value_names) | (
            analysis.assigned_any
            & (
                _live_before_block(stmt.body, branch_live_after)
                | _live_before_block(stmt.orelse, branch_live_after)
            )
        )
        if_entry_names = {
            name: self._fresh(f"if_entry_{name}") for name in sorted(branch_entry_names)
        }
        then_body = self.rewrite_block(
            stmt.body,
            live_after=branch_live_after,
            live_after_slots=branch_live_after_slots,
            allow_loop_control=False,
            static_iters=static_iters,
            bound_on_entry=bound_before,
        )
        else_body = self.rewrite_block(
            stmt.orelse,
            live_after=branch_live_after,
            live_after_slots=branch_live_after_slots,
            allow_loop_control=False,
            static_iters=static_iters,
            bound_on_entry=bound_before,
        )
        trace_time_if = ast.If(
            test=_name(analysis.cond_name),
            body=copy.deepcopy(then_body) or [ast.Pass()],
            orelse=copy.deepcopy(else_body) or [ast.Pass()],
        )
        branch_name = self._fresh("br")
        return then_body, else_body, trace_time_if, branch_name, if_entry_names

    def _make_if_slot_value_names(self, merge_slots):
        """Reserve branch-field names for merged static subscript values."""
        slot_value_names = {
            # BranchHandle deliberately rejects private attribute names. Keep
            # the generated branch field public while retaining a unique
            # compiler-generated local name for the rewritten slot value.
            slot: (
                f"pto_ast_slot_{slot.base}_"
                f"{'neg' if slot.index < 0 else ''}{abs(slot.index)}_{self._counter}"
            )
            for slot in merge_slots
        }
        self._counter += len(slot_value_names)
        old_slot_value_names = {
            slot: self._fresh(
                f"old_slot_{slot.base}_"
                f"{'neg' if slot.index < 0 else ''}{abs(slot.index)}"
            )
            for slot in merge_slots
        }
        return slot_value_names, old_slot_value_names

    def _restore_and_rewrite_if_branch_bodies(
        self,
        analysis,
        branch_plan,
        then_body,
        else_body,
        static_iters,
    ):
        """Prefix both executable branches with entry restores and slot rewrites."""
        dynamic_then_body = copy.deepcopy(then_body)
        dynamic_else_body = copy.deepcopy(else_body)
        if branch_plan.if_entry_names or branch_plan.old_slot_value_names:
            # The restore assignments below emit no IR: they only reset the
            # trace-time Python bindings so every dynamic branch starts from the
            # same entering SSA environment. Slot temporaries are shared between
            # sibling branches as well, so restore them from their entry copies
            # too; otherwise a nested conditional inside one branch captures the
            # other branch's slot value.
            entry_restores = [
                ast.Assign(
                    targets=[_name(name, ast.Store())],
                    value=_name(branch_plan.if_entry_names[name]),
                )
                for name in sorted(branch_plan.if_entry_names)
            ] + [
                ast.Assign(
                    targets=[_name(branch_plan.slot_value_names[slot], ast.Store())],
                    value=_name(branch_plan.old_slot_value_names[slot]),
                )
                for slot in sorted(analysis.merge_slots)
            ]
            dynamic_then_body = copy.deepcopy(entry_restores) + dynamic_then_body
            dynamic_else_body = copy.deepcopy(entry_restores) + dynamic_else_body
        if branch_plan.slot_value_names:
            dynamic_then_body = [
                _SlotValueRewriter(branch_plan.slot_value_names, self._static_env, static_iters).visit(item)
                for item in dynamic_then_body
            ]
            dynamic_else_body = [
                _SlotValueRewriter(branch_plan.slot_value_names, self._static_env, static_iters).visit(item)
                for item in dynamic_else_body
            ]
        return dynamic_then_body, dynamic_else_body

    def _append_dynamic_if_branch_merges(
        self,
        analysis,
        branch_plan,
        stmt,
        dynamic_then_body,
        dynamic_else_body,
    ):
        """Append per-branch merge assignments and fix generated locations."""
        if analysis.merge_names or branch_plan.slot_value_names:
            then_assigned = set(analysis.then_info.stores)
            else_assigned = set(analysis.else_info.stores)
            if analysis.control_state:
                if analysis.then_control["break"] or analysis.then_control["continue"]:
                    then_assigned.add(analysis.control_state["active"])
                if analysis.else_control["break"] or analysis.else_control["continue"]:
                    else_assigned.add(analysis.control_state["active"])
                if analysis.then_control["break"]:
                    then_assigned.add(analysis.control_state["did_break"])
                if analysis.else_control["break"]:
                    else_assigned.add(analysis.control_state["did_break"])
            dynamic_then_body.append(
                self._branch_assign(
                    branch_plan.branch_name,
                    analysis.merge_names,
                    entry_names=branch_plan.if_entry_names,
                    assigned_names=then_assigned,
                    slot_value_names=branch_plan.slot_value_names,
                    old_slot_value_names=branch_plan.old_slot_value_names,
                    assigned_slots=analysis.then_slot_info.stores,
                )
            )
            dynamic_else_body.append(
                self._branch_assign(
                    branch_plan.branch_name,
                    analysis.merge_names,
                    entry_names=branch_plan.if_entry_names,
                    assigned_names=else_assigned,
                    slot_value_names=branch_plan.slot_value_names,
                    old_slot_value_names=branch_plan.old_slot_value_names,
                    assigned_slots=analysis.else_slot_info.stores,
                )
            )
        for node in dynamic_then_body + dynamic_else_body:
            ast.fix_missing_locations(ast.copy_location(node, stmt))
        return dynamic_then_body, dynamic_else_body

    def _build_dynamic_if_body(
        self,
        analysis,
        if_entry_names,
        with_stmt,
        branch_name,
        slot_value_names,
    ):
        """Build the runtime (non-trace-time) body of the lowered conditional."""
        dynamic_body = [
            ast.Assign(
                targets=[_name(if_entry_names[name], ast.Store())],
                value=self._current_value(name, requires_pre_if_value=True),
            )
            for name in sorted(if_entry_names)
        ]
        dynamic_body.append(with_stmt)
        dynamic_body.extend(
            ast.Assign(
                targets=[_name(name, ast.Store())],
                value=ast.Call(
                    func=ast.Attribute(
                        value=_name(branch_name),
                        attr="get",
                        ctx=ast.Load(),
                    ),
                    args=[ast.Constant(value=name)],
                    keywords=[],
                ),
            )
            for name in analysis.merge_names
        )
        dynamic_body.extend(
            ast.Assign(
                targets=[_slot_subscript(slot, ast.Store())],
                value=ast.Attribute(
                    value=_name(branch_name),
                    attr=slot_value_names[slot],
                    ctx=ast.Load(),
                ),
            )
            for slot in analysis.merge_slots
        )
        return dynamic_body

    def _rewrite_for(
        self, stmt, *, live_after, live_after_slots=None, allow_loop_control=False,
        static_iters=None, bound_before=None,
    ):
        live_after_slots = set(live_after_slots or ())
        static_iters = dict(static_iters or {})
        # Drop statically dead tails of unconditional break/continue before
        # any analysis or tracing (they are unreachable in Python semantics).
        stmt.body = _drop_unreachable_tails(stmt.body)
        if _is_pto_attr_call(stmt.iter, "static_range"):
            return self._rewrite_static_range_for(
                stmt,
                live_after=live_after,
                live_after_slots=live_after_slots,
                static_iters=static_iters,
                bound_before=bound_before,
            )
        control = _loop_control_flags(stmt.body)
        if stmt.orelse or control["break"] or control["continue"]:
            return self._rewrite_controlled_for(
                stmt,
                live_after=live_after,
                live_after_slots=live_after_slots,
                static_iters=static_iters,
                bound_before=bound_before,
            )
        if not isinstance(stmt.target, ast.Name):
            raise PTODSLAstRewriteError("ast_rewrite=True runtime for-loops require a simple name target")
        _reject_control_flow_exits(
            stmt.body,
            "for-loop bodies",
            reject_bare_returns=self._reject_bare_returns,
        )
        if stmt.target.id in live_after:
            raise PTODSLAstRewriteError(
                "ast_rewrite=True runtime for-loops cannot expose the loop induction variable outside the loop yet; "
                f"use explicit pto.for_(...) for {stmt.target.id!r}"
            )
        return self._rewrite_plain_for(
            stmt,
            live_after=live_after,
            live_after_slots=live_after_slots,
            static_iters=static_iters,
            bound_before=bound_before,
        )

    def _rewrite_static_range_for(self, stmt, *, live_after, live_after_slots, static_iters, bound_before):
        """Rewrite a compile-time ``pto.static_range`` loop in place."""
        next_static_iters = dict(static_iters)
        if isinstance(stmt.target, ast.Name):
            values = _try_eval_static_range(stmt.iter, self._static_env, static_iters)
            if values is not None:
                next_static_iters[stmt.target.id] = values
        saved_control_stack = self._loop_control_stack
        self._loop_control_stack = []
        name_target = stmt.target.id if isinstance(stmt.target, ast.Name) else None
        entry_with_iv = set(bound_before or ())
        if name_target is not None:
            entry_with_iv.add(name_target)
        orelse_with_iv = set(bound_before or ())
        # The induction target is only definitely bound in the else clause
        # when the static range is known non-empty; an empty static range
        # jumps straight to the else clause with the target unbound.
        if name_target is not None and values is not None and len(values) > 0:
            orelse_with_iv.add(name_target)
        try:
            stmt.body = self.rewrite_block(
                stmt.body,
                live_after=live_after,
                live_after_slots=live_after_slots,
                allow_loop_control=True,
                static_iters=next_static_iters,
                bound_on_entry=entry_with_iv,
            )
            stmt.orelse = self.rewrite_block(
                stmt.orelse,
                live_after=live_after,
                live_after_slots=live_after_slots,
                allow_loop_control=True,
                static_iters=static_iters,
                bound_on_entry=orelse_with_iv,
            )
        finally:
            self._loop_control_stack = saved_control_stack
        return [stmt]

    def _rewrite_plain_for(self, stmt, *, live_after, live_after_slots, static_iters, bound_before):
        """Lower a plain runtime range-for (no transfers/else) to pto.for_."""
        start, stop, step, hint_keywords = _range_triplet_and_hints(stmt.iter)
        step = self._validate_plain_for_step(step)
        loop_carried, loop_carried_slots = self._analyze_plain_for_carries(
            stmt, live_after=live_after, live_after_slots=live_after_slots,
            static_iters=static_iters, bound_before=bound_before,
        )
        loop_name = self._fresh("loop")
        body = self._rewrite_plain_for_body(
            stmt, loop_carried, loop_carried_slots,
            live_after=live_after, live_after_slots=live_after_slots,
            static_iters=static_iters, bound_before=bound_before,
        )
        slot_carry_names, slot_maps = self._reserve_for_slot_names(loop_carried_slots)
        spec = _ForRangeSpec(start=start, stop=stop, step=step, hint_keywords=hint_keywords)
        if loop_carried or loop_carried_slots:
            return self._build_carried_for_result(
                _PlainForPlan(
                    stmt=stmt,
                    spec=spec,
                    loop_name=loop_name,
                    body=body,
                    static_iters=static_iters,
                    carry=_ForCarryPlan(
                        loop_carried=loop_carried,
                        loop_carried_slots=loop_carried_slots,
                        slot_carry_names=slot_carry_names,
                        slot_maps=slot_maps,
                    ),
                )
            )
        return self._build_plain_for_result(stmt, spec, body)

    def _analyze_plain_for_carries(self, stmt, *, live_after, live_after_slots, static_iters, bound_before):
        """Compute the loop-carried names and slots of a plain runtime for."""
        body_info = _name_info(stmt.body)
        body_slot_info = _slot_info(stmt.body, self._static_env, static_iters)
        if body_slot_info.invalid_stores:
            raise PTODSLAstRewriteError(body_slot_info.invalid_stores[0])
        assigned_live_after = body_info.stores & set(live_after)
        assigned_slots_live_after = body_slot_info.stores & set(live_after_slots)
        # Seed backward liveness with the body stores that are live after the
        # loop so that a partial assignment whose default path keeps the previous
        # value is recognized as a live-in and therefore a loop-carried iter_arg.
        # Only the implicit (default-path) reads are additionally gated on a
        # definite binding before the loop: carrying an unbound name would read
        # an unbound Python local when the loop is entered, instead of the clear
        # last-iteration-only diagnostics below. The same gate applies to static
        # subscript slots, keyed on the definite binding of the slot base list.
        reads_before_raw = _read_before_assignment_names(stmt.body)
        reads_before = _read_before_assignment_names(stmt.body, live_after=assigned_live_after)
        implicit_reads = reads_before - reads_before_raw
        safe_reads = reads_before_raw | {
            name for name in implicit_reads if name in (bound_before or set())
        }
        # The induction variable is re-bound at the top of every iteration and
        # must never be inferred as a carried name, otherwise the carry init
        # would read it before the loop (unbound) or shadow the fresh binding.
        loop_carried = tuple(
            name
            for name in sorted(body_info.stores & safe_reads)
            if name != stmt.target.id
        )
        slot_reads_raw = _read_before_assignment_slots(stmt.body, self._static_env, static_iters)
        slot_reads = _read_before_assignment_slots(
            stmt.body, self._static_env, static_iters, live_after=assigned_slots_live_after
        )
        slot_implicit = slot_reads - slot_reads_raw
        safe_slots = slot_reads_raw | {
            slot for slot in slot_implicit if slot.base in (bound_before or set())
        }
        loop_carried_slots = tuple(sorted(body_slot_info.stores & safe_slots))
        unsupported_last_values = sorted(assigned_live_after - set(loop_carried))
        if unsupported_last_values:
            raise PTODSLAstRewriteError(
                "ast_rewrite=True runtime for-loops cannot expose last-iteration-only values yet; "
                f"use explicit pto.for_(...).carry(...) for {unsupported_last_values}"
            )
        unsupported_last_slots = sorted(assigned_slots_live_after - set(loop_carried_slots))
        if unsupported_last_slots:
            slots = [slot.display for slot in unsupported_last_slots]
            raise PTODSLAstRewriteError(
                "ast_rewrite=True runtime for-loops cannot expose last-iteration-only static subscript values yet; "
                f"use explicit scalar temporaries for {slots}"
            )
        return loop_carried, loop_carried_slots

    def _rewrite_plain_for_body(
        self, stmt, loop_carried, loop_carried_slots, *, live_after, live_after_slots,
        static_iters, bound_before,
    ):
        """Rewrite the plain for body with the loop-carried names live."""
        loop_live_after = set(live_after) | set(loop_carried)
        loop_live_after_slots = set(live_after_slots) | set(loop_carried_slots)
        return self.rewrite_block(
            stmt.body,
            live_after=loop_live_after,
            live_after_slots=loop_live_after_slots,
            static_iters=static_iters,
            bound_on_entry=(
                set(bound_before or ())
                | {stmt.target.id}
                | set(loop_carried)
                | {slot.base for slot in loop_carried_slots}
            ),
        )

    def _reserve_for_slot_names(self, loop_carried_slots):
        """Reserve carry/map names for each carried static subscript slot."""
        slot_carry_names = {
            slot: self._fresh(f"slot_{slot.base}_{slot.index}")
            for slot in loop_carried_slots
        }
        slot_maps = {}
        for slot in loop_carried_slots:
            slot_maps.setdefault(slot.base, {"map_name": self._fresh(f"slot_{slot.base}_map"), "slots": []})
            slot_maps[slot.base]["slots"].append(slot)
        for data in slot_maps.values():
            data["slots"] = tuple(sorted(data["slots"]))
        return slot_carry_names, slot_maps

    def _build_carried_for_result(self, plan):
        """Assemble the carry() lowering for a plain runtime for-loop."""
        carry = plan.carry
        slot_initializers = self._for_carry_slot_initializers(
            carry.loop_carried_slots, carry.slot_carry_names
        )
        spec = plan.spec
        setup = ast.Assign(
            targets=[_name(plan.loop_name, ast.Store())],
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.Call(
                        func=_pto_attr("for_"),
                        args=[spec.start, spec.stop],
                        keywords=[
                            ast.keyword(arg="step", value=spec.step),
                            *spec.hint_keywords,
                        ],
                    ),
                    attr="carry",
                    ctx=ast.Load(),
                ),
                args=[],
                keywords=self._carry_keywords(carry, from_current=True),
            ),
        )
        prologue = self._for_carry_prologue(plan)
        composed_body = self._compose_for_carry_body(plan, prologue)
        with_stmt = ast.With(
            items=[ast.withitem(context_expr=_name(plan.loop_name), optional_vars=None)],
            body=composed_body or [ast.Pass()],
            type_comment=None,
        )
        result = slot_initializers + [
            ast.copy_location(setup, plan.stmt),
            ast.copy_location(with_stmt, plan.stmt),
        ]
        self._extend_for_carry_finals(
            result, plan.loop_name, carry.loop_carried, carry.loop_carried_slots,
            carry.slot_carry_names,
        )
        return result

    def _carry_keywords(self, carry, *, from_current):
        """Build the keyword list shared by ``carry()`` and ``update()`` calls."""
        pairs = [
            (name, self._current_value(name) if from_current else _name(name))
            for name in carry.loop_carried
        ]
        pairs.extend(
            (carry.slot_carry_names[slot], _name(carry.slot_carry_names[slot]))
            for slot in carry.loop_carried_slots
        )
        return [ast.keyword(arg=name, value=value) for name, value in pairs]

    def _compose_for_carry_body(self, plan, prologue):
        """Compose prologue + rewritten body + slot epilogue + update call."""
        carry = plan.carry
        body = plan.body
        if carry.loop_carried_slots:
            body = [
                _SlotCarryRewriter(
                    carry.slot_maps, self._static_env, plan.static_iters
                ).visit(item)
                for item in body
            ]
        slot_epilogue = [
            ast.Assign(
                targets=[_name(carry.slot_carry_names[slot], ast.Store())],
                value=_map_subscript(
                    carry.slot_maps[slot.base]["map_name"], slot.index
                ),
            )
            for slot in carry.loop_carried_slots
        ]
        return prologue + body + [
            *slot_epilogue,
            ast.Expr(
                value=ast.Call(
                    func=ast.Attribute(
                        value=_name(plan.loop_name), attr="update", ctx=ast.Load()
                    ),
                    args=[],
                    keywords=self._carry_keywords(carry, from_current=False),
                )
            )
        ]

    def _guard_block(self, condition, body, *, merge_names=(), assigned_names=()):
        """Guard lowered statements and merge values needed after the guard."""
        branch_name = self._fresh("condition_guard")
        merge_names = tuple(merge_names)
        assigned_names = set(assigned_names)
        old_names = {name: self._fresh(f"guard_old_{name}") for name in merge_names}
        prefix = [ast.Assign(
            targets=[_name(old_names[name], ast.Store())], value=_name(name))
            for name in merge_names]
        then_body = list(body) + [self._branch_assign(
            branch_name, merge_names, entry_names=old_names,
            assigned_names=assigned_names)] if merge_names else list(body)
        else_body = [self._branch_assign(
            branch_name, merge_names, entry_names=old_names,
            assigned_names=set())] if merge_names else [ast.Pass()]
        result = prefix + [ast.With(
            items=[ast.withitem(
                context_expr=ast.Call(func=_pto_attr("if_"), args=[condition], keywords=[]),
                optional_vars=_name(branch_name, ast.Store()),
            )],
            body=[ast.With(
                items=[ast.withitem(
                    context_expr=ast.Attribute(value=_name(branch_name), attr="then_", ctx=ast.Load()),
                    optional_vars=None,
                )],
                body=then_body or [ast.Pass()],
                type_comment=None,
            )],
            type_comment=None,
        )]
        if merge_names:
            result[-1].body.append(ast.With(
                items=[ast.withitem(
                    context_expr=ast.Attribute(value=_name(branch_name), attr="else_", ctx=ast.Load()),
                    optional_vars=None,
                )],
                body=else_body,
                type_comment=None,
            ))
            result.extend(ast.Assign(
                targets=[_name(name, ast.Store())],
                value=ast.Call(func=ast.Attribute(value=_name(branch_name), attr="get", ctx=ast.Load()),
                               args=[ast.Constant(name)], keywords=[]),
            ) for name in merge_names)
        return result

    def _rewrite_controlled_for(self, stmt, *, live_after, live_after_slots=None, static_iters=None, bound_before=None):
        """Lower range-for with transfers or else clauses through scf.while."""
        start, stop, step, loop_carried = self._analyze_controlled_for_loop(
            stmt, live_after=live_after, static_iters=static_iters,
            bound_before=bound_before,
        )
        iv_name = stmt.target.id
        skip_name, did_break_name, loop_name, state_name = self._fresh_controlled_loop_names()
        state_names = (iv_name,) + tuple(sorted(loop_carried)) + (skip_name, did_break_name)
        plan = _ControlledForPlan(
            iv_name=iv_name,
            loop_carried=loop_carried,
            skip_name=skip_name,
            did_break_name=did_break_name,
            state_name=state_name,
            state_names=state_names,
            loop_name=loop_name,
        )
        condition_fn = self._build_controlled_condition_fn(plan, stop, step)
        initial_values = [
            copy.deepcopy(start),
            *[_name(name) for name in sorted(loop_carried)],
            _flag_const(True),
            _flag_const(False),
        ]
        setup = ast.Assign(
            targets=[_name(plan.loop_name, ast.Store())],
            value=ast.Call(
                func=_pto_attr("_while"),
                args=[condition_fn],
                keywords=[
                    ast.keyword(arg=name, value=value)
                    for name, value in zip(plan.state_names, initial_values)
                ],
            ),
        )
        body = self._rewrite_controlled_loop_body(
            stmt, plan.skip_name, plan.did_break_name, plan.iv_name, plan.loop_carried,
            live_after=live_after, static_iters=static_iters,
            bound_before=bound_before,
        )
        with_stmt = self._build_controlled_for_with_stmt(plan, step, body)
        result = [ast.copy_location(setup, stmt), ast.copy_location(with_stmt, stmt)]
        self._extend_controlled_loop_finals(result, plan, live_after)
        if stmt.orelse:
            self._extend_controlled_loop_else(
                result, stmt, did_break_name, live_after=live_after,
                live_after_slots=live_after_slots, static_iters=static_iters,
                bound_before=bound_before,
            )
        return result

    def _analyze_controlled_for_loop(self, stmt, *, live_after, static_iters, bound_before):
        """Compute the loop-carried names for a controlled runtime for-loop."""
        if not isinstance(stmt.target, ast.Name):
            raise PTODSLAstRewriteError(
                "ast_rewrite=True runtime for-loops with break/continue require a simple name target"
            )
        if _loop_has_return(stmt.body):
            raise PTODSLAstRewriteError(
                "ast_rewrite=True does not support dynamic return inside runtime for"
            )
        start, stop, step, hint_keywords = _range_triplet_and_hints(stmt.iter)
        if hint_keywords:
            raise PTODSLAstRewriteError(
                "ast_rewrite=True pto.range(...) unroll hints are not supported on loops with "
                "break/continue or else clauses (these lower through pto._while)"
            )
        body_info = _name_info(stmt.body)
        body_slot_info = _slot_info(stmt.body, self._static_env, static_iters)
        if body_slot_info.invalid_stores:
            raise PTODSLAstRewriteError(body_slot_info.invalid_stores[0])
        if body_slot_info.stores:
            raise PTODSLAstRewriteError(
                "ast_rewrite=True runtime for break/continue does not support static subscript carries yet"
            )
        reads_before_raw = _read_before_assignment_names(stmt.body)
        assigned_live_after = body_info.stores & set(live_after)
        reads_before = _read_before_assignment_names(stmt.body, live_after=assigned_live_after)
        implicit_reads = (reads_before - reads_before_raw) | assigned_live_after
        safe_reads = reads_before_raw | {
            name for name in implicit_reads if name in (bound_before or set())
        }
        loop_carried = set(body_info.stores & safe_reads)
        loop_carried.discard(stmt.target.id)
        unsupported_last = sorted((body_info.stores & set(live_after)) - loop_carried)
        if unsupported_last:
            raise PTODSLAstRewriteError(
                "ast_rewrite=True runtime for-loops cannot expose last-iteration-only values yet; "
                f"use explicit pto.for_(...).carry(...) for {unsupported_last}"
            )
        return start, stop, step, loop_carried

    def _fresh_controlled_loop_names(self):
        """Reserve the flag/loop/state names for one pto._while lowering."""
        skip_name = self._fresh("loop_active")
        did_break_name = self._fresh("loop_did_break")
        loop_name = self._fresh("loop")
        state_name = self._fresh("for_state")
        return skip_name, did_break_name, loop_name, state_name

    def _rewrite_controlled_loop_body(
        self, stmt, skip_name, did_break_name, iv_name, loop_carried, *, live_after,
        static_iters, bound_before,
    ):
        """Rewrite the controlled loop body under the active/did_break flags."""
        self._loop_control_stack.append({"active": skip_name, "did_break": did_break_name})
        try:
            return self._rewrite_loop_body(
                stmt.body,
                live_after=(set(live_after) | loop_carried | {iv_name} |
                            {skip_name, did_break_name}),
                live_after_slots=set(),
                control={"active": skip_name, "did_break": did_break_name},
                static_iters=static_iters,
                bound_on_entry=set(bound_before or ()) | set(loop_carried) | {iv_name},
                forced_tail_merge_names=loop_carried,
            )
        finally:
            self._loop_control_stack.pop()

    def _extend_controlled_loop_else(
        self, result, stmt, did_break_name, *, live_after, live_after_slots, static_iters,
        bound_before,
    ):
        """Lower the normal-completion ``else`` clause under a no-break guard."""
        else_body = self.rewrite_block(
            stmt.orelse,
            live_after=live_after,
            live_after_slots=live_after_slots,
            allow_loop_control=False,
            static_iters=static_iters,
            bound_on_entry=bound_before,
        )
        else_info = _name_info(stmt.orelse)
        else_merge_names = tuple(sorted(else_info.stores & set(live_after)))
        result.extend(self._guard_block(
            ast.Compare(left=_name(did_break_name), ops=[ast.Eq()], comparators=[_flag_const(False)]),
            else_body,
            merge_names=else_merge_names,
            assigned_names=else_info.stores,
        ))

    def _rewrite_while(self, stmt, *, live_after, live_after_slots=None,
                       allow_loop_control=False, static_iters=None, bound_before=None):
        """Lower runtime ``while`` using named state and explicit control flags."""
        carry_names, control = self._analyze_while_loop(
            stmt, live_after=live_after, static_iters=static_iters,
            bound_before=bound_before,
        )
        active_name, did_break_name, loop_name, state_name = self._fresh_while_names()
        controlled = bool(stmt.orelse or control["break"] or control["continue"])
        state_names = carry_names + ((active_name, did_break_name) if controlled else ())
        plan = _WhilePlan(
            loop_name=loop_name,
            carry_names=carry_names,
            active_name=active_name,
            did_break_name=did_break_name,
            state_name=state_name,
            state_names=state_names,
            controlled=controlled,
        )
        condition_fn = self._build_while_condition_fn(
            stmt, plan.state_name, plan.state_names, plan.did_break_name, plan.controlled
        )
        setup = ast.Assign(
            targets=[_name(plan.loop_name, ast.Store())],
            value=ast.Call(
                func=_pto_attr("_while"),
                args=[condition_fn],
                keywords=self._while_initial_keywords(
                    plan.state_names, plan.active_name, plan.did_break_name
                ),
            ),
        )
        body = self._rewrite_while_loop_body(
            stmt, plan.active_name, plan.did_break_name, plan.controlled, plan.carry_names,
            live_after=live_after, live_after_slots=live_after_slots,
            static_iters=static_iters, bound_before=bound_before,
        )
        with_stmt = self._build_while_with_stmt(
            plan.loop_name, plan.state_names, plan.active_name, plan.controlled, body
        )
        result = [ast.copy_location(setup, stmt), ast.copy_location(with_stmt, stmt)]
        self._extend_while_loop_finals(result, plan, live_after)
        if stmt.orelse:
            self._extend_controlled_loop_else(
                result, stmt, did_break_name, live_after=live_after,
                live_after_slots=live_after_slots, static_iters=static_iters,
                bound_before=bound_before,
            )
        return result

    def _analyze_while_loop(self, stmt, *, live_after, static_iters, bound_before):
        """Compute loop-carried names and control flags for a runtime while."""
        # Drop statically dead tails of unconditional break/continue before
        # any analysis or tracing (they are unreachable in Python semantics),
        # so dead names never enter the carry computation or get traced.
        stmt.body = _drop_unreachable_tails(stmt.body)
        if _loop_has_return(stmt.body):
            raise PTODSLAstRewriteError(
                "ast_rewrite=True does not support dynamic return inside runtime while"
            )

        body_info = _name_info(stmt.body)
        body_slot_info = _slot_info(stmt.body, self._static_env, static_iters)
        else_slot_info = _slot_info(stmt.orelse, self._static_env, static_iters)
        if body_slot_info.invalid_stores or else_slot_info.invalid_stores:
            raise PTODSLAstRewriteError(
                (body_slot_info.invalid_stores or else_slot_info.invalid_stores)[0]
            )
        if body_slot_info.stores or else_slot_info.stores:
            raise PTODSLAstRewriteError(
                "ast_rewrite=True runtime while does not support static subscript carries yet"
            )
        test_info = _name_info(stmt.test)
        control = _loop_control_flags(stmt.body)
        # A loop-carried name must be one whose pre-iteration value is needed:
        # it is read by the test before each iteration, it is read before any
        # assignment inside the body (so it depends on the previous iteration's
        # value), or it is an assigned live-out whose partial-assignment default
        # path keeps the entering value.  Seed backward liveness with assigned
        # live-outs so nested conditionals expose that implicit read, then gate
        # only those implicit reads on a definite binding before the loop.  An
        # unbound implicit read must take the same last-iteration-only diagnostic
        # as runtime for-loops instead of being evaluated by pto._while(...)
        # setup.  This mirrors _rewrite_for while retaining explicit reads from
        # the authored body and test.
        assigned_live_after = body_info.stores & set(live_after)
        reads_before_raw = _read_before_assignment_names(stmt.body)
        reads_before = _read_before_assignment_names(stmt.body, live_after=assigned_live_after)
        implicit_reads = reads_before - reads_before_raw
        safe_reads = reads_before_raw | {
            name for name in implicit_reads if name in (bound_before or set())
        }
        required_carries = test_info.loads | safe_reads
        # A controlled loop (break/continue/else) may legitimately need no
        # user-level carry: its test reads only loop-invariant names and every
        # body store is a loop-local temporary.  The active/did_break control
        # flags then provide the loop-carried state themselves.  Only a
        # completely store-less body is rejected, since it cannot carry the
        # control transfer at all.
        controlled = control["break"] or control["continue"] or bool(stmt.orelse)
        if controlled and not body_info.stores:
            raise PTODSLAstRewriteError(
                "ast_rewrite=True runtime while break/continue requires explicit control-state lowering"
            )
        carry_names = tuple(sorted(body_info.stores & required_carries))
        unsupported_last_values = sorted(assigned_live_after - set(carry_names))
        if unsupported_last_values:
            raise PTODSLAstRewriteError(
                "ast_rewrite=True runtime while cannot expose last-iteration-only values yet; "
                f"use explicit pto.while_(...) state for {unsupported_last_values}"
            )
        if not carry_names and not (control["break"] or control["continue"] or stmt.orelse):
            raise PTODSLAstRewriteError(
                "ast_rewrite=True runtime while requires at least one loop-carried value"
            )
        return carry_names, control

    def _fresh_while_names(self):
        """Reserve the flag/loop/state names for one while lowering."""
        active_name = self._fresh("while_active")
        did_break_name = self._fresh("while_did_break")
        loop_name = self._fresh("while")
        state_name = self._fresh("while_state")
        return active_name, did_break_name, loop_name, state_name

    def _rewrite_while_loop_body(
        self, stmt, active_name, did_break_name, controlled, carry_names, *, live_after,
        live_after_slots, static_iters, bound_before,
    ):
        """Rewrite the while body under the (optional) control flags."""
        if controlled:
            self._loop_control_stack.append({"active": active_name, "did_break": did_break_name})
        try:
            return self._rewrite_loop_body(
                stmt.body,
                live_after=(set(live_after) | set(carry_names) |
                            ({active_name, did_break_name} if controlled else set())),
                live_after_slots=set(live_after_slots or ()),
                control=(
                    {"active": active_name, "did_break": did_break_name}
                    if controlled else None
                ),
                static_iters=static_iters,
                bound_on_entry=set(bound_before or ()) | set(carry_names),
                forced_tail_merge_names=carry_names,
            )
        finally:
            if controlled:
                self._loop_control_stack.pop()



__all__ = [
    "_flag_const",
    "_name",
    "_slot_subscript",
    "_map_subscript",
    "_SlotCarryRewriter",
    "_SlotValueRewriter",
    "_ControlFlowExitVisitor",
    "_reject_control_flow_exits",
    "_IfRewriteAnalysis",
    "_StateNameRewriter",
    "_WhileBoolLiteralRewriter",
    "_ControlFlowRewriter",
]
