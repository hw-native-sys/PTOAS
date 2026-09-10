# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Static analysis helpers for the PTODSL source-to-source AST rewrite."""

from __future__ import annotations

import ast
from dataclasses import dataclass

from ._ast_rewrite_errors import (
    PTODSLAstRewriteError,
    _MISSING_GLOBAL,
)


@dataclass(frozen=True)
class _NameInfo:
    loads: set[str]
    stores: set[str]


@dataclass(frozen=True, order=True)
class _SubscriptSlot:
    base: str
    index: int

    @property
    def display(self) -> str:
        return f"{self.base}[{self.index}]"


@dataclass(frozen=True)
class _SlotInfo:
    loads: set[_SubscriptSlot]
    stores: set[_SubscriptSlot]
    invalid_stores: tuple[str, ...] = ()


class _NameInfoVisitor(ast.NodeVisitor):
    def __init__(self):
        self.loads = set()
        self.stores = set()

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            self.loads.add(node.id)
        elif isinstance(node.ctx, (ast.Store, ast.Del)):
            self.stores.add(node.id)

    def visit_AugAssign(self, node):
        self._visit_augassign_target_load(node.target)
        self.visit(node.value)
        self.visit(node.target)

    def visit_For(self, node):
        self.visit(node.iter)
        bound = _target_stores(node.target)
        body_info = _name_info(node.body)
        orelse_info = _name_info(node.orelse)
        self.loads.update((body_info.loads | orelse_info.loads) - bound)
        self.stores.update((body_info.stores | orelse_info.stores) - bound)

    def visit_While(self, node):
        self.visit(node.test)
        body_info = _name_info(node.body)
        orelse_info = _name_info(node.orelse)
        self.loads.update(body_info.loads | orelse_info.loads)
        self.stores.update(body_info.stores | orelse_info.stores)

    def visit_FunctionDef(self, node):
        self.stores.add(node.name)
        for decorator in node.decorator_list:
            self.visit(decorator)
        self._visit_arguments_defaults(node.args)
        self.loads.update(_function_free_vars(node))

    def visit_AsyncFunctionDef(self, node):
        self.visit_FunctionDef(node)

    def visit_Lambda(self, node):
        self._visit_arguments_defaults(node.args)
        self.loads.update(_lambda_free_vars(node))

    def visit_ClassDef(self, node):
        self.stores.add(node.name)
        for decorator in node.decorator_list:
            self.visit(decorator)
        for base in node.bases:
            self.visit(base)
        for keyword in node.keywords:
            self.visit(keyword.value)
        self.loads.update(_class_body_free_vars(node))

    def visit_ListComp(self, node):
        self._visit_comprehension(node.generators, (node.elt,))

    def visit_SetComp(self, node):
        self._visit_comprehension(node.generators, (node.elt,))

    def visit_GeneratorExp(self, node):
        self._visit_comprehension(node.generators, (node.elt,))

    def visit_DictComp(self, node):
        self._visit_comprehension(node.generators, (node.key, node.value))

    def _visit_augassign_target_load(self, node):
        if isinstance(node, ast.Name):
            self.loads.add(node.id)
            return
        if isinstance(node, (ast.Attribute, ast.Subscript)):
            self.visit(node.value)
            return
        self.visit(node)

    def _visit_arguments_defaults(self, args):
        for default in args.defaults:
            self.visit(default)
        for default in args.kw_defaults:
            if default is not None:
                self.visit(default)

    def _visit_comprehension(self, generators, result_nodes):
        bound = set()
        for generator in generators:
            self._visit_comprehension_expr(generator.iter, bound)
            bound |= _target_stores(generator.target)
            for if_node in generator.ifs:
                self._visit_comprehension_expr(if_node, bound)
        for result_node in result_nodes:
            self._visit_comprehension_expr(result_node, bound)

    def _visit_comprehension_expr(self, node, bound):
        info = _name_info(node)
        self.loads.update(info.loads - set(bound))
        self.stores.update(info.stores - set(bound))


def _name_info(node) -> _NameInfo:
    visitor = _NameInfoVisitor()
    if isinstance(node, list):
        for item in node:
            visitor.visit(item)
    else:
        visitor.visit(node)
    return _NameInfo(visitor.loads, visitor.stores)


class _SlotInfoVisitor(ast.NodeVisitor):
    def __init__(self, static_env, static_iters=None):
        self._static_env = static_env
        self._static_iters = dict(static_iters or {})
        self.loads = set()
        self.stores = set()
        self.invalid_stores = []

    def visit_Subscript(self, node):
        if isinstance(node.ctx, ast.Load):
            self.loads.update(
                _resolve_subscript_slots(
                    node, self._static_env, self._static_iters, require_static=False
                )
            )
            return
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            slots = _resolve_subscript_slots(node, self._static_env, self._static_iters, require_static=True)
            if slots:
                self.stores.update(slots)
            else:
                self.invalid_stores.append(_unsupported_subscript_store_message(node))
            return
        self.generic_visit(node)

    def visit_AugAssign(self, node):
        if isinstance(node.target, ast.Subscript):
            slots = _resolve_subscript_slots(node.target, self._static_env, self._static_iters, require_static=True)
            if slots:
                self.loads.update(slots)
                self.stores.update(slots)
            else:
                self.invalid_stores.append(_unsupported_subscript_store_message(node.target))
        else:
            self.visit(node.target)
        self.visit(node.value)

    def visit_For(self, node):
        if _is_pto_attr_call(node.iter, "static_range") and isinstance(node.target, ast.Name):
            values = _try_eval_static_range(node.iter, self._static_env, self._static_iters)
            if values is None:
                for stmt in node.body:
                    self.visit(stmt)
                for stmt in node.orelse:
                    self.visit(stmt)
                return
            old = self._static_iters.get(node.target.id)
            self._static_iters[node.target.id] = values
            try:
                for stmt in node.body:
                    self.visit(stmt)
            finally:
                if old is None:
                    self._static_iters.pop(node.target.id, None)
                else:
                    self._static_iters[node.target.id] = old
            for stmt in node.orelse:
                self.visit(stmt)
            return
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        return

    def visit_AsyncFunctionDef(self, node):
        return

    def visit_Lambda(self, node):
        return

    def visit_ClassDef(self, node):
        return


def _slot_info(node, static_env, static_iters=None) -> _SlotInfo:
    visitor = _SlotInfoVisitor(static_env, static_iters)
    if isinstance(node, list):
        for item in node:
            visitor.visit(item)
    else:
        visitor.visit(node)
    return _SlotInfo(visitor.loads, visitor.stores, tuple(visitor.invalid_stores))


def _slot_live_before_block(stmts, live_after, static_env, static_iters=None) -> set[_SubscriptSlot]:
    live = set(live_after)
    for stmt in reversed(stmts):
        live = _slot_live_before_stmt(stmt, live, static_env, static_iters or {})
    return live


def _slot_live_before_stmt(stmt, live_after, static_env, static_iters) -> set[_SubscriptSlot]:
    if isinstance(stmt, (ast.With, ast.AsyncWith)):
        live = _slot_live_before_block(stmt.body, live_after, static_env, static_iters)
        # Python evaluates with-items from left to right and binds each
        # optional_vars immediately.  Reverse the sequence for liveness so a
        # context expression can use a binding produced by an earlier item
        # without incorrectly turning that use into a live-in.
        for item in reversed(stmt.items):
            if item.optional_vars is not None:
                live = _kill_slots_for_with_target(
                    live, item.optional_vars, static_env, static_iters
                )
            live |= _slot_info(item.context_expr, static_env, static_iters).loads
        return live
    if isinstance(stmt, ast.If):
        test_info = _slot_info(stmt.test, static_env, static_iters)
        return (
            set(test_info.loads)
            | _slot_live_before_block(stmt.body, live_after, static_env, static_iters)
            | _slot_live_before_block(stmt.orelse, live_after, static_env, static_iters)
        )
    if isinstance(stmt, ast.For):
        if _is_pto_attr_call(stmt.iter, "static_range") and isinstance(stmt.target, ast.Name):
            values = _try_eval_static_range(stmt.iter, static_env, static_iters)
            if values is not None:
                next_static_iters = dict(static_iters)
                next_static_iters[stmt.target.id] = values
                return (
                    _slot_live_before_block(stmt.body, live_after, static_env, next_static_iters)
                    | _slot_live_before_block(stmt.orelse, live_after, static_env, static_iters)
                )
        iter_info = _slot_info(stmt.iter, static_env, static_iters)
        body_info = _slot_info(stmt.body, static_env, static_iters)
        orelse_info = _slot_info(stmt.orelse, static_env, static_iters)
        assigned = body_info.stores | orelse_info.stores
        return (
            (set(live_after) - assigned)
            | set(iter_info.loads)
            | _slot_live_before_block(stmt.body, set(), static_env, static_iters)
            | _slot_live_before_block(stmt.orelse, set(), static_env, static_iters)
        )
    info = _slot_info(stmt, static_env, static_iters)
    live = _kill_slots_for_assigned_bases(live_after, stmt)
    return (set(live) - info.stores) | info.loads


def _read_before_assignment_slots(stmts, static_env, static_iters=None, live_after=None) -> set[_SubscriptSlot]:
    return _slot_live_before_block(stmts, set(live_after or ()), static_env, static_iters)


def _kill_slots_for_assigned_bases(slots, stmt) -> set[_SubscriptSlot]:
    assigned_bases = _assigned_name_targets(stmt)
    if not assigned_bases:
        return set(slots)
    return {
        slot
        for slot in slots
        if slot.base not in assigned_bases
    }


def _kill_slots_for_with_target(
    slots, target, static_env, static_iters
) -> set[_SubscriptSlot]:
    target_info = _slot_info(target, static_env, static_iters)
    bound_bases = _simple_name_targets(target)
    dynamic_subscript_bases = set()
    for subscript in _target_subscripts(target):
        if not _resolve_subscript_slots(
            subscript, static_env, static_iters, require_static=True
        ) and isinstance(subscript.value, ast.Name):
            dynamic_subscript_bases.add(subscript.value.id)
    killed_bases = bound_bases | dynamic_subscript_bases
    return {
        slot
        for slot in slots
        if slot.base not in killed_bases and slot not in target_info.stores
    }


def _target_subscripts(target):
    if isinstance(target, ast.Subscript):
        yield target
        return
    if isinstance(target, (ast.Tuple, ast.List)):
        for element in target.elts:
            yield from _target_subscripts(element)
        return
    if isinstance(target, ast.Starred):
        yield from _target_subscripts(target.value)


def _assigned_name_targets(stmt) -> set[str]:
    if isinstance(stmt, ast.Assign):
        names = set()
        for target in stmt.targets:
            names.update(_simple_name_targets(target))
        return names
    if isinstance(stmt, ast.AnnAssign):
        return _simple_name_targets(stmt.target)
    if isinstance(stmt, (ast.For, ast.AsyncFor)):
        return _simple_name_targets(stmt.target)
    if isinstance(stmt, (ast.With, ast.AsyncWith)):
        names = set()
        for item in stmt.items:
            if item.optional_vars is not None:
                names.update(_simple_name_targets(item.optional_vars))
        return names
    return set()


def _simple_name_targets(target) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        names = set()
        for elt in target.elts:
            names.update(_simple_name_targets(elt))
        return names
    return set()


def _definite_stores(stmt) -> set[str]:
    """Names definitely (unconditionally) bound by one statement.

    Conservative on purpose: names assigned inside conditionals or loop bodies
    are not treated as definite, so implicit loop carries are only inferred for
    names that are provably bound before the loop statement.
    """
    if isinstance(stmt, ast.Assign):
        names = set()
        for target in stmt.targets:
            names.update(_simple_name_targets(target))
        return names
    if isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
        return _simple_name_targets(stmt.target)
    return set()


def _deleted_names(node) -> set[str]:
    """Names whose local bindings may be deleted while executing node."""
    deleted = set()

    class Visitor(ast.NodeVisitor):
        def visit_Delete(self, delete):
            for target in delete.targets:
                deleted.update(_simple_name_targets(target))

        def visit_FunctionDef(self, function):
            return

        def visit_AsyncFunctionDef(self, function):
            return

        def visit_Lambda(self, function):
            return

        def visit_ClassDef(self, class_def):
            return

    visitor = Visitor()
    if isinstance(node, list):
        for item in node:
            visitor.visit(item)
    else:
        visitor.visit(node)
    return deleted


def _definite_out_stmt(stmt, bound_in, static_env=None, static_iters=None) -> set[str]:
    """Definite-assignment dataflow through one statement (not a block)."""
    if isinstance(stmt, ast.Assign) or (isinstance(stmt, ast.AnnAssign) and stmt.value is not None):
        return set(bound_in) | _definite_stores(stmt)
    if isinstance(stmt, ast.Delete):
        out = set(bound_in)
        for target in stmt.targets:
            out -= _simple_name_targets(target)
        return out
    if isinstance(stmt, ast.If):
        return _definite_out_block(stmt.body, bound_in, static_env, static_iters) & _definite_out_block(
            stmt.orelse, bound_in, static_env, static_iters
        )
    if isinstance(stmt, ast.With):
        out = set(bound_in)
        for item in stmt.items:
            if item.optional_vars is not None:
                out |= _simple_name_targets(item.optional_vars)
        return _definite_out_block(stmt.body, out, static_env, static_iters)
    if isinstance(stmt, ast.For):
        return _definite_out_for(stmt, bound_in, static_env, static_iters)
    # Everything else does not definitely bind names: loop bodies may not run
    # and the loop variable is unbound for empty ranges.
    return set(bound_in)


def _definite_out_for(stmt, bound_in, static_env=None, static_iters=None) -> set[str]:
    """Definite-assignment flow for ``for`` statements.

    Runtime loop bodies or targets are not definite because the loop may not
    run. A known static_range, however, executes at trace time, so its body and
    normal-completion else clause update bindings exactly like a Python
    for-loop.
    """
    if not _is_pto_attr_call(stmt.iter, "static_range"):
        return set(bound_in)
    values = _try_eval_static_range(stmt.iter, static_env or {}, static_iters or {})
    if values is None:
        return set(bound_in)
    out = set(bound_in)
    if not values:
        return _definite_out_block(stmt.orelse, out, static_env, static_iters)

    target_names = _simple_name_targets(stmt.target)
    control = _loop_control_flags(stmt.body)
    if control["break"] or control["continue"]:
        # Do not credit body assignments that a transfer may skip.
        # The target is rebound before every entered iteration, but
        # a body deletion can still leave it (or another incoming
        # name) unbound on the final/breaking path.
        out |= target_names
        out -= _deleted_names(stmt.body)
        if control["break"]:
            # The else clause is optional when a break is possible:
            # keep no additions from it, and remove bindings it may
            # delete on the normal-completion path.
            return out - _deleted_names(stmt.orelse)
        return _definite_out_block(stmt.orelse, out, static_env, static_iters)

    loop_static_iters = dict(static_iters or {})
    for value in values:
        iteration_static_iters = dict(loop_static_iters)
        if isinstance(stmt.target, ast.Name):
            iteration_static_iters[stmt.target.id] = (value,)
        out |= target_names
        out = _definite_out_block(stmt.body, out, static_env, iteration_static_iters)
    if isinstance(stmt.target, ast.Name):
        loop_static_iters[stmt.target.id] = (values[-1],)
    return _definite_out_block(stmt.orelse, out, static_env, loop_static_iters)


def _definite_out_block(stmts, bound_in, static_env=None, static_iters=None) -> set[str]:
    out = set(bound_in)
    for stmt in stmts:
        out = _definite_out_stmt(stmt, out, static_env, static_iters)
    return out


def _resolve_subscript_slots(node, static_env, static_iters, *, require_static) -> set[_SubscriptSlot]:
    if not isinstance(node.value, ast.Name):
        return set()
    index_values = _static_index_values(node.slice, static_env, static_iters)
    if index_values is None:
        return set()
    return {
        _SubscriptSlot(node.value.id, index)
        for index in index_values
    }


def _static_index_values(node, static_env, static_iters):
    try:
        return _eval_static_int_values(node, static_env, static_iters)
    except PTODSLAstRewriteError:
        return None


def _unsupported_subscript_store_message(node) -> str:
    try:
        text = ast.unparse(node)
    except Exception:
        text = "<subscript>"
    return (
        "ast_rewrite=True only supports static subscript carry stores of the form "
        f"simple_name[static_int_or_static_range_iv]; got {text!r}"
    )


def _try_eval_static_range(call, static_env, static_iters=None):
    if not _is_pto_attr_call(call, "static_range") or call.keywords:
        return None
    try:
        values = [_eval_static_int(arg, static_env, static_iters) for arg in call.args]
    except PTODSLAstRewriteError:
        return None
    if len(values) == 1:
        return tuple(range(values[0]))
    if len(values) == 2:
        return tuple(range(values[0], values[1]))
    if len(values) == 3:
        return tuple(range(values[0], values[1], values[2]))
    return None


def _eval_static_int(node, static_env, static_iters=None) -> int:
    values = _eval_static_int_values(node, static_env, static_iters or {})
    if len(values) != 1:
        raise PTODSLAstRewriteError("static integer expression must resolve to one value")
    return values[0]


def _eval_static_int_values(node, static_env, static_iters) -> tuple[int, ...]:
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
        return (node.value,)
    if isinstance(node, ast.Name):
        if node.id in static_iters:
            return tuple(static_iters[node.id])
        value = static_env.get(node.id, _MISSING_GLOBAL)
        if isinstance(value, int) and not isinstance(value, bool):
            return (value,)
        raise PTODSLAstRewriteError("static value is not an integer")
    if isinstance(node, ast.UnaryOp):
        return _eval_static_unary_values(node, static_env, static_iters)
    if isinstance(node, ast.BinOp):
        return _eval_static_binop_values(node, static_env, static_iters)
    raise PTODSLAstRewriteError("unsupported static integer expression")


def _eval_static_unary_values(node, static_env, static_iters) -> tuple[int, ...]:
    """Evaluate a unary static integer expression to all its possible values."""
    if isinstance(node.op, ast.UAdd):
        return tuple(+value for value in _eval_static_int_values(node.operand, static_env, static_iters))
    if isinstance(node.op, ast.USub):
        return tuple(-value for value in _eval_static_int_values(node.operand, static_env, static_iters))
    raise PTODSLAstRewriteError("unsupported static integer expression")


def _eval_static_binop_values(node, static_env, static_iters) -> tuple[int, ...]:
    """Evaluate a binary static integer expression over its value cross-product."""
    lhs_values = _eval_static_int_values(node.left, static_env, static_iters)
    rhs_values = _eval_static_int_values(node.right, static_env, static_iters)
    values = []
    seen = set()
    for lhs in lhs_values:
        for rhs in rhs_values:
            value = _eval_static_binop_scalar(node.op, lhs, rhs)
            if value not in seen:
                seen.add(value)
                values.append(value)
    return tuple(values)


def _eval_static_binop_scalar(op, lhs, rhs) -> int:
    """Apply one static integer binary operator to two scalar values."""
    if isinstance(op, ast.Add):
        return lhs + rhs
    if isinstance(op, ast.Sub):
        return lhs - rhs
    if isinstance(op, ast.Mult):
        return lhs * rhs
    if isinstance(op, ast.FloorDiv):
        return lhs // rhs
    if isinstance(op, ast.Mod):
        return lhs % rhs
    raise PTODSLAstRewriteError("unsupported static integer expression")


class _ScopeBindingVisitor(ast.NodeVisitor):
    def __init__(self):
        self.stores = set()
        self.globals = set()
        self.nonlocals = set()

    def visit_Name(self, node):
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self.stores.add(node.id)

    def visit_FunctionDef(self, node):
        self.stores.add(node.name)

    def visit_AsyncFunctionDef(self, node):
        self.visit_FunctionDef(node)

    def visit_Lambda(self, node):
        return

    def visit_ClassDef(self, node):
        self.stores.add(node.name)

    def visit_ListComp(self, node):
        self._visit_comprehension(node.generators, (node.elt,))

    def visit_SetComp(self, node):
        self._visit_comprehension(node.generators, (node.elt,))

    def visit_GeneratorExp(self, node):
        self._visit_comprehension(node.generators, (node.elt,))

    def visit_DictComp(self, node):
        self._visit_comprehension(node.generators, (node.key, node.value))

    def visit_Global(self, node):
        self.globals.update(node.names)

    def visit_Nonlocal(self, node):
        self.nonlocals.update(node.names)

    def visit_Import(self, node):
        for alias in node.names:
            self.stores.add(alias.asname or alias.name.split(".", 1)[0])

    def visit_ImportFrom(self, node):
        for alias in node.names:
            if alias.name == "*":
                continue
            self.stores.add(alias.asname or alias.name)

    def _visit_comprehension(self, generators, result_nodes):
        for generator in generators:
            self.visit(generator.iter)
            for if_node in generator.ifs:
                self.visit(if_node)
        for result_node in result_nodes:
            self.visit(result_node)


def _argument_names(args) -> set[str]:
    names = {arg.arg for arg in list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs)}
    if args.vararg is not None:
        names.add(args.vararg.arg)
    if args.kwarg is not None:
        names.add(args.kwarg.arg)
    return names


def _local_bindings(stmts) -> tuple[set[str], set[str]]:
    visitor = _ScopeBindingVisitor()
    for stmt in stmts:
        visitor.visit(stmt)
    local_stores = visitor.stores - visitor.globals - visitor.nonlocals
    return local_stores, visitor.globals


def _function_free_vars(node) -> set[str]:
    local_stores, globals_declared = _local_bindings(node.body)
    bound = _argument_names(node.args) | local_stores
    body_info = _name_info(node.body)
    return body_info.loads - bound - globals_declared


def _lambda_free_vars(node) -> set[str]:
    bound = _argument_names(node.args)
    body_info = _name_info(node.body)
    return body_info.loads - bound


def _class_body_free_vars(node) -> set[str]:
    local_stores, globals_declared = _local_bindings(node.body)
    body_info = _name_info(node.body)
    return body_info.loads - local_stores - globals_declared


def _target_stores(node) -> set[str]:
    return _name_info(node).stores


def _live_before_block(stmts, live_after) -> set[str]:
    live = set(live_after)
    for stmt in reversed(stmts):
        live = _live_before_stmt(stmt, live)
    return live


def _live_before_stmt(stmt, live_after) -> set[str]:
    if isinstance(stmt, (ast.With, ast.AsyncWith)):
        context_loads = set()
        bound = set()
        for item in stmt.items:
            context_loads |= _name_info(item.context_expr).loads
            if item.optional_vars is not None:
                bound |= _target_stores(item.optional_vars)
        return context_loads | (_live_before_block(stmt.body, live_after) - bound)
    if isinstance(stmt, ast.If):
        test_info = _name_info(stmt.test)
        return (
            set(test_info.loads)
            | _live_before_block(stmt.body, live_after)
            | _live_before_block(stmt.orelse, live_after)
        )
    if isinstance(stmt, ast.For):
        iter_info = _name_info(stmt.iter)
        target_stores = _target_stores(stmt.target)
        body_info = _name_info(stmt.body)
        orelse_info = _name_info(stmt.orelse)
        assigned = target_stores | body_info.stores | orelse_info.stores
        return (
            (set(live_after) - assigned)
            | set(iter_info.loads)
            | (_live_before_block(stmt.body, set()) - target_stores)
            | _live_before_block(stmt.orelse, set())
        )
    if isinstance(stmt, ast.While):
        test_info = _name_info(stmt.test)
        body_info = _name_info(stmt.body)
        else_info = _name_info(stmt.orelse)
        assigned = body_info.stores | else_info.stores
        loop_live = set(live_after) | set(test_info.loads) | set(body_info.loads)
        return (
            (set(live_after) - assigned)
            | set(test_info.loads)
            | (_live_before_block(stmt.body, loop_live) - assigned)
            | _live_before_block(stmt.orelse, set(live_after))
        )
    info = _name_info(stmt)
    return (set(live_after) - info.stores) | info.loads


def _read_before_assignment_names(stmts, live_after=None):
    """Return the names read (or implicitly relied on) before assignment.

    live_after seeds backward liveness. Without a seed only explicit reads are
    visible; with a seed, the partial-assignment default paths of nested
    conditionals surface their implicit reads of the entering value.
    """
    return _live_before_block(stmts, set(live_after or ()))


def _is_pto_attr_call(node, name: str) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == name
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "pto"
    )


def _is_range_call(node) -> bool:
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "range"


def _is_pto_range_call(node) -> bool:
    return _is_pto_attr_call(node, "range")


_UNROLL_HINT_KWARGS = ("unroll", "unroll_factor")


def _range_triplet_and_hints(call):
    """Return (start, stop, step, hint_keywords) for a loop iterable call.

    Accepts plain ``range(...)`` (no keyword arguments, as before) and the
    ``pto.range(...)`` marker, which additionally takes the ``unroll`` /
    ``unroll_factor`` hint keywords forwarded to the generated ``pto.for_``.
    """
    if _is_range_call(call):
        if call.keywords:
            raise PTODSLAstRewriteError("ast_rewrite=True range(...) loops do not support keyword arguments")
        hint_keywords = []
    elif _is_pto_range_call(call):
        hint_keywords = []
        for keyword in call.keywords:
            if keyword.arg not in _UNROLL_HINT_KWARGS:
                raise PTODSLAstRewriteError(
                    "ast_rewrite=True pto.range(...) loops only support the "
                    f"'unroll' and 'unroll_factor' keyword arguments; got {keyword.arg!r}"
                )
            hint_keywords.append(keyword)
    else:
        raise PTODSLAstRewriteError(
            "ast_rewrite=True only rewrites for-loops over range(...) or pto.range(...)"
        )
    args = call.args
    if len(args) == 1:
        return ast.Constant(0), args[0], ast.Constant(1), hint_keywords
    if len(args) == 2:
        return args[0], args[1], ast.Constant(1), hint_keywords
    if len(args) == 3:
        return args[0], args[1], args[2], hint_keywords
    raise PTODSLAstRewriteError("ast_rewrite=True range(...) loops require 1 to 3 arguments")


def _pto_attr(name: str, ctx=ast.Load()):
    return ast.Attribute(value=ast.Name(id="pto", ctx=ast.Load()), attr=name, ctx=ctx)


def _loop_control_flags(stmts):
    """Return break/continue presence for the current loop only.

    ``ast.walk`` is deliberately not used here: a control transfer in a
    nested loop belongs to that nested loop, not to the enclosing loop.
    """
    result = {"break": False, "continue": False}

    class Visitor(ast.NodeVisitor):
        def visit_For(self, node):
            return

        def visit_While(self, node):
            return

        def visit_Break(self, node):
            result["break"] = True

        def visit_Continue(self, node):
            result["continue"] = True

    visitor = Visitor()
    for stmt in stmts:
        visitor.visit(stmt)
    return result


def _stmt_always_transfers(stmt):
    """True when *stmt* cannot complete normally in the current iteration.

    A bare break/continue, or a compound statement whose every path exits
    the iteration: an ``if`` whose both branches always transfer, a ``with``
    whose body always transfers, or a ``try`` whose body and caught handlers
    transfer (or whose ``finally`` transfers).  Nested loops belong to
    themselves: a break inside one says nothing about the outer iteration.
    """
    if isinstance(stmt, (ast.Break, ast.Continue)):
        return True
    if isinstance(stmt, ast.If):
        return (
            bool(stmt.orelse)
            and _block_always_transfers(stmt.body)
            and _block_always_transfers(stmt.orelse)
        )
    if isinstance(stmt, (ast.With, ast.AsyncWith)):
        return _block_always_transfers(stmt.body)
    if isinstance(stmt, ast.Try) or type(stmt).__name__ == "TryStar":
        # A transfer in the try body does not cover an exception path caught
        # by a handler.  A finally transfer, on the other hand, overrides
        # every normal or exceptional path through the statement.
        if _block_always_transfers(stmt.finalbody):
            return True
        return (
            _block_always_transfers(stmt.body)
            and all(_block_always_transfers(handler.body) for handler in stmt.handlers)
        )
    return False


def _block_always_transfers(stmts):
    # _drop_unreachable_tails has already removed everything after the first
    # guaranteed transfer in each child block.  Looking only at the final
    # reachable statement keeps this helper correct when called independently
    # and avoids treating an earlier conditional transfer as unconditional.
    return bool(stmts) and _stmt_always_transfers(stmts[-1])


def _drop_unreachable_tails(stmts):
    """Truncate each statement list after a guaranteed transfer and recurse.

    Statements following a statement that always exits the current iteration
    (bare break/continue, both-branches-transfer if, transfer-bodied
    with/try) are unreachable in Python semantics.  Dropping them before
    name/slot analysis keeps dead names out of the carry computation and,
    more importantly, keeps the tracer from executing bodies that reference
    locals Python itself would never bind (``while ...: break; dead = dead +
    1`` is legal Python yet raised UnboundLocalError when the dead tail was
    still analyzed/traced).

    The recursion stops at nested loops' own statements only in the sense of
    ownership: their bodies are cleaned too, since each list is truncated at
    its own control transfers.
    """
    cleaned = []
    for stmt in stmts:
        cleaned.append(stmt)
        for field in ("body", "orelse", "finalbody"):
            value = getattr(stmt, field, None)
            if isinstance(value, list) and value and isinstance(value[0], ast.stmt):
                setattr(stmt, field, _drop_unreachable_tails(value))
        for handler in getattr(stmt, "handlers", []):
            handler.body = _drop_unreachable_tails(handler.body)
        if _stmt_always_transfers(stmt):
            return cleaned
    return cleaned


def _loop_has_return(stmts):
    """Check returns in the current loop body, excluding nested functions."""
    class Visitor(ast.NodeVisitor):
        found = False

        def visit_FunctionDef(self, node):
            return

        def visit_AsyncFunctionDef(self, node):
            return

        def visit_Lambda(self, node):
            return

        def visit_Return(self, node):
            self.found = True

    visitor = Visitor()
    for stmt in stmts:
        visitor.visit(stmt)
    return visitor.found


__all__ = [
    "_NameInfo",
    "_SubscriptSlot",
    "_SlotInfo",
    "_NameInfoVisitor",
    "_name_info",
    "_SlotInfoVisitor",
    "_slot_info",
    "_slot_live_before_block",
    "_slot_live_before_stmt",
    "_read_before_assignment_slots",
    "_kill_slots_for_assigned_bases",
    "_kill_slots_for_with_target",
    "_target_subscripts",
    "_assigned_name_targets",
    "_simple_name_targets",
    "_definite_stores",
    "_deleted_names",
    "_definite_out_stmt",
    "_definite_out_for",
    "_definite_out_block",
    "_resolve_subscript_slots",
    "_static_index_values",
    "_unsupported_subscript_store_message",
    "_try_eval_static_range",
    "_eval_static_int",
    "_eval_static_int_values",
    "_eval_static_unary_values",
    "_eval_static_binop_values",
    "_eval_static_binop_scalar",
    "_ScopeBindingVisitor",
    "_argument_names",
    "_local_bindings",
    "_function_free_vars",
    "_lambda_free_vars",
    "_class_body_free_vars",
    "_target_stores",
    "_live_before_block",
    "_live_before_stmt",
    "_read_before_assignment_names",
    "_is_pto_attr_call",
    "_is_range_call",
    "_is_pto_range_call",
    "_UNROLL_HINT_KWARGS",
    "_range_triplet_and_hints",
    "_pto_attr",
    "_loop_control_flags",
    "_stmt_always_transfers",
    "_block_always_transfers",
    "_drop_unreachable_tails",
    "_loop_has_return",
]
