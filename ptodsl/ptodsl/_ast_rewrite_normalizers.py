# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Lexical normalizers for the PTODSL source-to-source AST rewrite."""

from __future__ import annotations

import ast
import copy

from ._ast_rewrite_errors import (
    PTODSLAstRewriteError,
    _MISSING_GLOBAL,
)
from ._ast_rewrite_analysis import (
    _is_pto_attr_call,
    _name_info,
    _pto_attr,
    _target_stores,
)
from ._ast_rewrite_lowering import _name


class _SectionLexicalRewriter(ast.NodeTransformer):
    """Give ``with pto.section(...)`` a lexical, closure-like name scope."""

    def __init__(self):
        super().__init__()
        self._counter = 0
        self._env = {}
        self._local_names = set()
        self._known_bindings = set()
        self._section_outer_bindings = None
        self.section_entry_bindings = {}
        self.section_uninitialized_aliases = set()

    @staticmethod
    def _is_section_with(node):
        return isinstance(node, ast.With) and any(
            _is_pto_attr_call(item.context_expr, "section") for item in node.items
        )

    def visit_With(self, node):
        if not self._is_section_with(node):
            return self.generic_visit(node)
        node.items = [self.visit(item) for item in node.items]
        node.body = self._visit_section_body(node.body)
        return node

    def visit_Assign(self, node):
        node.value = self.visit(node.value)
        targets = set()
        for target in node.targets:
            targets |= self._target_names(target)
        self._activate_targets(targets)
        node.targets = [self.visit(target) for target in node.targets]
        if self._section_outer_bindings is None:
            self._known_bindings.update(targets)
        return node

    def visit_AnnAssign(self, node):
        if node.value is not None:
            node.value = self.visit(node.value)
        self._activate_targets(self._target_names(node.target))
        node.target = self.visit(node.target)
        if self._section_outer_bindings is None:
            self._known_bindings.update(self._target_names(node.target))
        return node

    def visit_AugAssign(self, node):
        if isinstance(node.target, ast.Name) and node.target.id in self._local_names:
            name = node.target.id
            if name in self._env:
                node.target.id = self._env[name]
            node.value = self.visit(node.value)
            self._activate_targets({name})
            node.target.id = self._env[name]
            if self._section_outer_bindings is None:
                self._known_bindings.add(name)
            return node
        return self.generic_visit(node)

    def visit_For(self, node):
        node.iter = self.visit(node.iter)
        self._activate_targets(self._target_names(node.target))
        node.target = self.visit(node.target)
        if self._section_outer_bindings is None:
            self._known_bindings.update(self._target_names(node.target))
        node.body, body_env = self._visit_block(node.body, self._env)
        self._env.update(body_env)
        node.orelse, else_env = self._visit_block(node.orelse, self._env)
        self._env.update(else_env)
        return node

    def visit_If(self, node):
        node.test = self.visit(node.test)
        # Both branches of a runtime conditional share one authored binding.
        # Any future env-forking visitor must apply the same invariant: reserve
        # common targets before visiting either branch. For section-local
        # bindings this prevents the branch merge from creating two aliases.
        common_targets = _name_info(node.body).stores & _name_info(node.orelse).stores
        self._activate_targets(common_targets)
        entry_env = dict(self._env)
        node.body, body_env = self._visit_block(node.body, entry_env)
        node.orelse, else_env = self._visit_block(node.orelse, entry_env)
        entry_aliases = set(entry_env.values())
        branch_only_aliases = set(body_env.values()) ^ set(else_env.values())
        self.section_uninitialized_aliases.update(
            alias
            for alias in branch_only_aliases - entry_aliases
            if alias not in self.section_entry_bindings
        )
        self._env.update(body_env)
        self._env.update(else_env)
        return node

    def visit_Name(self, node):
        alias = self._env.get(node.id)
        if alias is not None:
            node.id = alias
        return node

    def _fresh_alias(self, name):
        alias = f"__pto_section_{self._counter}_{name}"
        self._counter += 1
        return alias

    def _target_names(self, target):
        return _target_stores(target)

    def _activate_targets(self, targets):
        for name in targets & self._local_names:
            if name not in self._env:
                self._env[name] = self._fresh_alias(name)
            alias = self._env[name]
            if self._section_outer_bindings is not None and name in self._section_outer_bindings:
                self.section_entry_bindings.setdefault(alias, name)

    def _visit_block(self, stmts, env=None):
        old_env = self._env
        if env is not None:
            self._env = dict(env)
        try:
            result = [self.visit(stmt) for stmt in stmts]
            return result, dict(self._env)
        finally:
            self._env = old_env

    def _visit_section_body(self, stmts):
        old_env = self._env
        old_names = self._local_names
        old_outer_bindings = self._section_outer_bindings
        entry_binding_count = len(self.section_entry_bindings)
        self._env = {}
        self._local_names = _name_info(stmts).stores
        self._section_outer_bindings = set(self._known_bindings)
        try:
            body = [self.visit(stmt) for stmt in stmts]
            # Materialize outer values under their section-local aliases before
            # any runtime control flow. Subsequent branch merges can then read
            # the alias at the current program point instead of always falling
            # back to the section entry value.
            entry_bindings = list(self.section_entry_bindings.items())[entry_binding_count:]
            initializers = [
                ast.Assign(
                    targets=[_name(alias, ast.Store())],
                    value=_name(outer_name),
                )
                for alias, outer_name in entry_bindings
            ]
            return initializers + body
        finally:
            self._env = old_env
            self._local_names = old_names
            self._section_outer_bindings = old_outer_bindings



def _find_function_def(tree, name: str):
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if len(matches) != 1:
        return None
    return matches[0]




def _temporarily_bind_globals(globals_ns, bindings):
    restored = {}
    for name, value in bindings.items():
        restored[name] = globals_ns.get(name, _MISSING_GLOBAL)
        globals_ns[name] = value
    return restored


def _restore_globals(globals_ns, restored):
    for name, value in restored.items():
        if value is _MISSING_GLOBAL:
            globals_ns.pop(name, None)
        else:
            globals_ns[name] = value


def _inject_closure_defaults(function_def, closure_bindings):
    if not closure_bindings:
        return
    existing = set()
    for args in (
        function_def.args.posonlyargs,
        function_def.args.args,
        function_def.args.kwonlyargs,
    ):
        for arg in args:
            existing.add(arg.arg)
    if function_def.args.vararg is not None:
        existing.add(function_def.args.vararg.arg)
    if function_def.args.kwarg is not None:
        existing.add(function_def.args.kwarg.arg)

    for name in closure_bindings:
        if name in existing:
            continue
        function_def.args.kwonlyargs.append(ast.arg(arg=name))
        function_def.args.kw_defaults.append(ast.Constant(None))


def _sanitize_signature_for_exec(function_def):
    args = function_def.args
    args.defaults = [ast.Constant(None) for _ in args.defaults]
    args.kw_defaults = [
        ast.Constant(None) if default is not None else None
        for default in args.kw_defaults
    ]
    for arg in (
        list(args.posonlyargs)
        + list(args.args)
        + list(args.kwonlyargs)
    ):
        arg.annotation = None
    if args.vararg is not None:
        args.vararg.annotation = None
    if args.kwarg is not None:
        args.kwarg.annotation = None
    function_def.returns = None


def _is_normalizable_ifexp_assign_target(node) -> bool:
    return isinstance(node, ast.Name)


class _ConditionalExpressionNormalizer(ast.NodeTransformer):
    """Normalize assign-form ``IfExp`` into statement ``if`` before rewrite."""

    def visit_Assign(self, node):
        node = self.generic_visit(node)
        if not isinstance(node.value, ast.IfExp):
            return node
        if not node.targets or not all(_is_normalizable_ifexp_assign_target(target) for target in node.targets):
            return node
        return self._normalize_ifexp_assignment(node, node.value)

    def visit_AnnAssign(self, node):
        node = self.generic_visit(node)
        if node.value is None or not isinstance(node.value, ast.IfExp):
            return node
        if not _is_normalizable_ifexp_assign_target(node.target):
            return node
        return self._normalize_ifexp_assignment(node, node.value)

    def _normalize_ifexp_assignment(self, stmt, value):
        then_stmt = copy.deepcopy(stmt)
        then_stmt.value = value.body
        else_stmt = copy.deepcopy(stmt)
        else_stmt.value = value.orelse
        if_stmt = ast.If(
            test=value.test,
            body=[then_stmt],
            orelse=[else_stmt],
        )
        return ast.copy_location(self.generic_visit(if_stmt), stmt)


def _zero_arg_lambda(body):
    return ast.Lambda(
        args=ast.arguments(
            posonlyargs=[],
            args=[],
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[],
        ),
        body=body,
    )


class _BoolOpRewriter(ast.NodeTransformer):
    """Rewrite Python ``and``/``or`` into device-side short-circuit helpers.

    Python ``and``/``or`` are not overloadable operators: evaluating ``a and b``
    forces a truth check on ``a``, which calls ``__bool__`` on a PTODSL runtime
    value during tracing and raises.  This pass replaces every ``ast.BoolOp``
    with a right-nested lazy helper call so the RHS is only traced inside a
    device-side ``scf.if`` region::

        a and b and c  ->  pto._short_circuit_and(a, lambda: pto._short_circuit_and(b, lambda: c))
        a or  b or  c  ->  pto._short_circuit_or(a,  lambda: pto._short_circuit_or(b,  lambda: c))

    The transformation is purely syntactic and composes with every expression
    context: assignments, call arguments, ``return``, and ``if``/``while``
    conditions.  Nested function bodies are rewritten as well, mirroring the
    control-flow rewriter.  Statically-known ``bool``/``int`` operands are
    short-circuited at trace time by the helpers themselves, so the RHS is not
    traced at all when Python semantics would skip it.
    """

    @staticmethod
    def _reject_rhs_assignment_expressions(node):
        """Reject walrus expressions that would move into a generated lambda."""
        if any(
            isinstance(child, ast.NamedExpr)
            for value in node.values[1:]
            for child in ast.walk(value)
        ):
            raise PTODSLAstRewriteError(
                "ast_rewrite=True cannot rewrite an and/or expression containing "
                "an assignment expression (walrus operator) on the RHS; the "
                "assignment would bind inside the generated helper lambda"
            )

    def visit_BoolOp(self, node):
        self._reject_rhs_assignment_expressions(node)
        node = self.generic_visit(node)
        if isinstance(node.op, ast.And):
            helper = "_short_circuit_and"
        elif isinstance(node.op, ast.Or):
            helper = "_short_circuit_or"
        else:
            return node
        values = list(node.values)
        if len(values) < 2:
            return values[0] if values else node
        result = values[-1]
        for value in reversed(values[:-1]):
            result = ast.Call(
                func=_pto_attr(helper),
                args=[value, _zero_arg_lambda(result)],
                keywords=[],
            )
        return result


__all__ = [
    "_SectionLexicalRewriter",
    "_find_function_def",
    "_temporarily_bind_globals",
    "_restore_globals",
    "_inject_closure_defaults",
    "_sanitize_signature_for_exec",
    "_is_normalizable_ifexp_assign_target",
    "_ConditionalExpressionNormalizer",
    "_zero_arg_lambda",
    "_BoolOpRewriter",
]
