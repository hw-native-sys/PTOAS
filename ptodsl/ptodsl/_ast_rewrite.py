# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Source-to-source AST rewrite for ``@pto.jit(ast_rewrite=True)``."""

from __future__ import annotations

import ast
import copy
import inspect
import textwrap
from dataclasses import dataclass


class PTODSLAstRewriteError(SyntaxError):
    """Raised when AST rewrite sees unsupported Python control flow."""


def rewrite_jit_function(
    fn,
    *,
    static_bindings=None,
    rewrite_control_flow=True,
    reject_bare_returns: bool = False,
):
    """Return a function with PTODSL lexical sections lowered safely.

    ``pto.section`` is a physical SSA region, not a Python ``with`` hint.  The
    section body therefore gets a small source-level lexical rewrite even when
    the optional control-flow rewrite is disabled.  This keeps Python's
    function-local assignment rules from leaking a section-local SSA value into
    a sibling physical section.
    ``reject_bare_returns`` controls whether ``return`` inside rewritten
    control flow is rejected. ``@pto.jit`` keeps the historical behavior, while
    ``@pto.func`` enables this because helper bodies must keep one helper ABI.
    """
    try:
        source = inspect.getsource(fn)
        _, source_start_index = inspect.findsource(fn)
        source_start_line = source_start_index + 1
    except (OSError, TypeError) as exc:
        # Dynamically-created functions from exec/REPL/notebook contexts may
        # not have retrievable source. Keep existing tracing behavior for those
        # functions instead of making default-on AST rewrite a compatibility
        # break; native Python runtime control flow will still fail during
        # tracing with the usual PTODSL diagnostics.
        _ = exc
        return fn

    tree = ast.parse(textwrap.dedent(source))
    function_def = _find_function_def(tree, fn.__name__)
    if function_def is None:
        return fn

    # Preserve the first decorator line as the rewritten function's co_firstlineno.
    function_first_lineno = min(
        [function_def.lineno]
        + [decorator.lineno for decorator in function_def.decorator_list]
    )
    function_def.decorator_list = []
    closure_vars = inspect.getclosurevars(fn)
    static_env = dict(fn.__globals__)
    static_env.update(closure_vars.nonlocals)
    static_env.update(static_bindings or {})
    _inject_closure_defaults(function_def, closure_vars.nonlocals)
    _sanitize_signature_for_exec(function_def)
    function_def = _ConditionalExpressionNormalizer().visit(function_def)
    section_rewriter = _SectionLexicalRewriter()
    function_def = section_rewriter.visit(function_def)
    if rewrite_control_flow:
        rewriter = _ControlFlowRewriter(
            static_env,
            section_uninitialized_aliases=section_rewriter.section_uninitialized_aliases,
            reject_bare_returns=reject_bare_returns,
        )
        entry_params = set()
        for _arg in (
            list(function_def.args.posonlyargs)
            + list(function_def.args.args)
            + list(function_def.args.kwonlyargs)
        ):
            entry_params.add(_arg.arg)
        function_def.body = rewriter.rewrite_block(
            function_def.body, live_after=set(), bound_on_entry=entry_params
        )
    function_def.lineno = function_first_lineno
    tree = ast.Module(body=[function_def], type_ignores=[])
    ast.fix_missing_locations(tree)
    ast.increment_lineno(tree, source_start_line - 1)

    locals_ns = {}
    try:
        source_file = inspect.getsourcefile(fn)
    except (OSError, TypeError):
        source_file = None
    code = compile(tree, source_file or "<ptodsl-ast-rewrite>", "exec")
    globals_ns = fn.__globals__
    restored_globals = _temporarily_bind_globals(globals_ns, closure_vars.nonlocals)
    try:
        exec(code, globals_ns, locals_ns)
    finally:
        _restore_globals(globals_ns, restored_globals)
    rewritten = locals_ns[function_def.name]
    rewritten.__defaults__ = fn.__defaults__
    rewritten_kwdefaults = dict(rewritten.__kwdefaults__ or {})
    rewritten_kwdefaults.update(closure_vars.nonlocals)
    if fn.__kwdefaults__:
        rewritten_kwdefaults.update(fn.__kwdefaults__)
    rewritten.__kwdefaults__ = rewritten_kwdefaults
    rewritten.__annotations__ = dict(getattr(fn, "__annotations__", {}))
    rewritten.__doc__ = fn.__doc__
    rewritten.__module__ = fn.__module__
    rewritten.__qualname__ = fn.__qualname__
    return rewritten


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


def _find_function_def(tree, name: str):
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if len(matches) != 1:
        return None
    return matches[0]


_MISSING_GLOBAL = object()


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
    existing = {
        arg.arg
        for arg in (
            list(function_def.args.posonlyargs)
            + list(function_def.args.args)
            + list(function_def.args.kwonlyargs)
        )
    }
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

    def _visit_augassign_target_load(self, node):
        if isinstance(node, ast.Name):
            self.loads.add(node.id)
            return
        if isinstance(node, (ast.Attribute, ast.Subscript)):
            self.visit(node.value)
            return
        self.visit(node)

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
            self.loads.update(_resolve_subscript_slots(node, self._static_env, self._static_iters, require_static=False))
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
        # Runtime loop bodies or targets are not definite because the loop may
        # not run. A known static_range, however, executes at trace time, so its
        # body and normal-completion else clause update bindings exactly like a
        # Python for-loop.
        if _is_pto_attr_call(stmt.iter, "static_range"):
            values = _try_eval_static_range(stmt.iter, static_env or {}, static_iters or {})
            if values is not None:
                out = set(bound_in)
                if not values:
                    return _definite_out_block(
                        stmt.orelse, out, static_env, static_iters
                    )

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
                    return _definite_out_block(
                        stmt.orelse, out, static_env, static_iters
                    )

                loop_static_iters = dict(static_iters or {})
                for value in values:
                    iteration_static_iters = dict(loop_static_iters)
                    if isinstance(stmt.target, ast.Name):
                        iteration_static_iters[stmt.target.id] = (value,)
                    out |= target_names
                    out = _definite_out_block(
                        stmt.body, out, static_env, iteration_static_iters
                    )
                if isinstance(stmt.target, ast.Name):
                    loop_static_iters[stmt.target.id] = (values[-1],)
                return _definite_out_block(
                    stmt.orelse, out, static_env, loop_static_iters
                )
        return set(bound_in)
    # Everything else does not definitely bind names: loop bodies may not run
    # and the loop variable is unbound for empty ranges.
    return set(bound_in)


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
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
        return tuple(+value for value in _eval_static_int_values(node.operand, static_env, static_iters))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return tuple(-value for value in _eval_static_int_values(node.operand, static_env, static_iters))
    if isinstance(node, ast.BinOp):
        lhs_values = _eval_static_int_values(node.left, static_env, static_iters)
        rhs_values = _eval_static_int_values(node.right, static_env, static_iters)
        values = []
        seen = set()
        for lhs in lhs_values:
            for rhs in rhs_values:
                if isinstance(node.op, ast.Add):
                    value = lhs + rhs
                elif isinstance(node.op, ast.Sub):
                    value = lhs - rhs
                elif isinstance(node.op, ast.Mult):
                    value = lhs * rhs
                elif isinstance(node.op, ast.FloorDiv):
                    value = lhs // rhs
                elif isinstance(node.op, ast.Mod):
                    value = lhs % rhs
                else:
                    raise PTODSLAstRewriteError("unsupported static integer expression")
                if value not in seen:
                    seen.add(value)
                    values.append(value)
        return tuple(values)
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


class _SlotCarryRewriter(ast.NodeTransformer):
    def __init__(self, slot_maps, static_env, static_iters=None):
        self._slot_maps = slot_maps
        self._static_env = static_env
        self._static_iters = dict(static_iters or {})

    def visit_For(self, node):
        if _is_pto_attr_call(node.iter, "static_range") and isinstance(node.target, ast.Name):
            values = _try_eval_static_range(node.iter, self._static_env, self._static_iters)
            old = self._static_iters.get(node.target.id)
            if values is not None:
                self._static_iters[node.target.id] = values
            try:
                node.body = [self.visit(stmt) for stmt in node.body]
            finally:
                if values is not None:
                    if old is None:
                        self._static_iters.pop(node.target.id, None)
                    else:
                        self._static_iters[node.target.id] = old
            node.orelse = [self.visit(stmt) for stmt in node.orelse]
            return node
        return self.generic_visit(node)

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


class _SlotValueRewriter(ast.NodeTransformer):
    """Replace selected static list slots with scalar branch state names."""

    def __init__(self, slot_values, static_env, static_iters=None):
        self._slot_values = dict(slot_values)
        self._static_env = static_env
        self._static_iters = dict(static_iters or {})

    def visit_For(self, node):
        if _is_pto_attr_call(node.iter, "static_range") and isinstance(node.target, ast.Name):
            values = _try_eval_static_range(node.iter, self._static_env, self._static_iters)
            old = self._static_iters.get(node.target.id)
            if values is not None:
                self._static_iters[node.target.id] = values
            try:
                node.body = [self.visit(stmt) for stmt in node.body]
            finally:
                if values is not None:
                    if old is None:
                        self._static_iters.pop(node.target.id, None)
                    else:
                        self._static_iters[node.target.id] = old
            node.orelse = [self.visit(stmt) for stmt in node.orelse]
            return node
        return self.generic_visit(node)

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

    def rewrite_block(self, stmts, *, live_after, live_after_slots=None, allow_loop_control=False, static_iters=None, bound_on_entry=None):
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

    def rewrite_stmt(self, stmt, *, live_after, live_after_slots=None, allow_loop_control=False, static_iters=None, bound_before=None):
        live_after_slots = set(live_after_slots or ())
        static_iters = dict(static_iters or {})
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
        if isinstance(stmt, (ast.Break, ast.Continue)):
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
            raise PTODSLAstRewriteError("ast_rewrite=True does not support break/continue in rewritten control flow")
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

    def _rewrite_nested(self, stmt, *, live_after, live_after_slots=None, allow_loop_control=False, static_iters=None, bound_before=None):
        live_after_slots = set(live_after_slots or ())
        static_iters = dict(static_iters or {})
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
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
        if isinstance(stmt, (ast.Lambda, ast.ClassDef)):
            return stmt
        for field, value in ast.iter_fields(stmt):
            if field in {"body", "orelse", "finalbody"} and isinstance(value, list):
                entry_bindings = set(bound_before or ())
                if field == "body" and isinstance(stmt, ast.With):
                    # The with-as targets are definitely bound inside the body.
                    for item in stmt.items:
                        if item.optional_vars is not None:
                            entry_bindings |= _simple_name_targets(item.optional_vars)
                setattr(
                    stmt,
                    field,
                    self.rewrite_block(
                        value,
                        live_after=live_after,
                        live_after_slots=live_after_slots,
                        allow_loop_control=allow_loop_control,
                        static_iters=static_iters,
                        bound_on_entry=entry_bindings,
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
        return stmt

    def _rewrite_if(self, stmt, *, live_after, live_after_slots=None, allow_loop_control=False, static_iters=None, bound_before=None):
        live_after_slots = set(live_after_slots or ())
        static_iters = dict(static_iters or {})
        if _is_pto_attr_call(stmt.test, "const_expr"):
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
            if then_control["break"] or then_control["continue"] or else_control["break"] or else_control["continue"]:
                assigned_any.add(control_state["active"])
            if then_control["break"] or else_control["break"]:
                assigned_any.add(control_state["did_break"])
        merge_names = tuple(sorted(live_after & assigned_any))
        old_value_names = {
            name: self._fresh(f"old_{name}")
            for name in merge_names
            if name not in then_info.stores or name not in else_info.stores
        }

        branch_live_after = set(live_after) | set(merge_names)
        branch_live_after_slots = set(live_after_slots) | set(merge_slots)
        # Variables not assigned on one side need their entering value at branch
        # merge time, and variables read inside a branch need a clean entering
        # environment. Snapshot the entering SSA value of exactly those names and
        # restore the Python binding at the top of each dynamic branch so sibling
        # branch tracing never observes the other branch's rebindings.
        branch_entry_names = set(old_value_names) | (
            assigned_any
            & (
                _live_before_block(stmt.body, branch_live_after)
                | _live_before_block(stmt.orelse, branch_live_after)
            )
        )
        if_entry_names = {
            name: self._fresh(f'if_entry_{name}') for name in sorted(branch_entry_names)
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
            test=_name(cond_name),
            body=copy.deepcopy(then_body) or [ast.Pass()],
            orelse=copy.deepcopy(else_body) or [ast.Pass()],
        )
        branch_name = self._fresh("br")

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
        dynamic_then_body = copy.deepcopy(then_body)
        dynamic_else_body = copy.deepcopy(else_body)
        if if_entry_names or old_slot_value_names:
            # The restore assignments below emit no IR: they only reset the
            # trace-time Python bindings so every dynamic branch starts from the
            # same entering SSA environment. Slot temporaries are shared between
            # sibling branches as well, so restore them from their entry copies
            # too; otherwise a nested conditional inside one branch captures the
            # other branch's slot value.
            entry_restores = [
                ast.Assign(
                    targets=[_name(name, ast.Store())],
                    value=_name(if_entry_names[name]),
                )
                for name in sorted(if_entry_names)
            ] + [
                ast.Assign(
                    targets=[_name(slot_value_names[slot], ast.Store())],
                    value=_name(old_slot_value_names[slot]),
                )
                for slot in sorted(merge_slots)
            ]
            dynamic_then_body = copy.deepcopy(entry_restores) + dynamic_then_body
            dynamic_else_body = copy.deepcopy(entry_restores) + dynamic_else_body
        if slot_value_names:
            dynamic_then_body = [
                _SlotValueRewriter(slot_value_names, self._static_env, static_iters).visit(stmt)
                for stmt in dynamic_then_body
            ]
            dynamic_else_body = [
                _SlotValueRewriter(slot_value_names, self._static_env, static_iters).visit(stmt)
                for stmt in dynamic_else_body
            ]
        if merge_names or slot_value_names:
            then_assigned = set(then_info.stores)
            else_assigned = set(else_info.stores)
            if control_state:
                if then_control["break"] or then_control["continue"]:
                    then_assigned.add(control_state["active"])
                if else_control["break"] or else_control["continue"]:
                    else_assigned.add(control_state["active"])
                if then_control["break"]:
                    then_assigned.add(control_state["did_break"])
                if else_control["break"]:
                    else_assigned.add(control_state["did_break"])
            dynamic_then_body.append(
                self._branch_assign(
                    branch_name,
                    merge_names,
                    entry_names=if_entry_names,
                    assigned_names=then_assigned,
                    slot_value_names=slot_value_names,
                    old_slot_value_names=old_slot_value_names,
                    assigned_slots=then_slot_info.stores,
                )
            )
            dynamic_else_body.append(
                self._branch_assign(
                    branch_name,
                    merge_names,
                    entry_names=if_entry_names,
                    assigned_names=else_assigned,
                    slot_value_names=slot_value_names,
                    old_slot_value_names=old_slot_value_names,
                    assigned_slots=else_slot_info.stores,
                )
            )
        for node in dynamic_then_body + dynamic_else_body:
            ast.fix_missing_locations(ast.copy_location(node, stmt))
        ast.fix_missing_locations(trace_time_if)

        with_stmt = ast.With(
            items=[
                ast.withitem(
                    context_expr=ast.Call(func=_pto_attr("if_"), args=[_name(cond_name)], keywords=[]),
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
            for name in merge_names
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
            for slot in merge_slots
        )

        result = [
            ast.Assign(
                targets=[_name(cond_name, ast.Store())],
                value=stmt.test,
            )
        ]
        result.extend(
            ast.Assign(
                targets=[_name(value_name, ast.Store())],
                value=_slot_subscript(slot),
            )
            for slot, value_name in slot_value_names.items()
        )
        result.extend(
            ast.Assign(
                targets=[_name(old_value_name, ast.Store())],
                value=_name(slot_value_names[slot]),
            )
            for slot, old_value_name in old_slot_value_names.items()
        )
        result.append(
            ast.copy_location(
                ast.If(
                    test=ast.Call(
                        func=_name("isinstance"),
                        args=[_name(cond_name), _name("bool")],
                        keywords=[],
                    ),
                    body=[trace_time_if],
                    orelse=dynamic_body,
                ),
                stmt,
            )
        )
        return result

    def _branch_assign(
        self,
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

    def _rewrite_for(self, stmt, *, live_after, live_after_slots=None, allow_loop_control=False, static_iters=None, bound_before=None):
        live_after_slots = set(live_after_slots or ())
        static_iters = dict(static_iters or {})
        # Drop statically dead tails of unconditional break/continue before
        # any analysis or tracing (they are unreachable in Python semantics).
        stmt.body = _drop_unreachable_tails(stmt.body)
        if _is_pto_attr_call(stmt.iter, "static_range"):
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

        start, stop, step, hint_keywords = _range_triplet_and_hints(stmt.iter)
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
        body_info = _name_info(stmt.body)
        body_slot_info = _slot_info(stmt.body, self._static_env, static_iters)
        if body_slot_info.invalid_stores:
            raise PTODSLAstRewriteError(body_slot_info.invalid_stores[0])
        slot_reads_before = _read_before_assignment_slots(stmt.body, self._static_env, static_iters)
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

        loop_name = self._fresh("loop")
        loop_live_after = set(live_after) | set(loop_carried)
        loop_live_after_slots = set(live_after_slots) | set(loop_carried_slots)
        body = self.rewrite_block(
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

        if loop_carried or loop_carried_slots:
            slot_initializers = [
                ast.Assign(
                    targets=[_name(slot_carry_names[slot], ast.Store())],
                    value=_slot_subscript(slot),
                )
                for slot in loop_carried_slots
            ]
            setup = ast.Assign(
                targets=[_name(loop_name, ast.Store())],
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Call(
                            func=_pto_attr("for_"),
                            args=[start, stop],
                            keywords=[ast.keyword(arg="step", value=step), *hint_keywords],
                        ),
                        attr="carry",
                        ctx=ast.Load(),
                    ),
                    args=[],
                    keywords=[
                        ast.keyword(
                            arg=name,
                            value=self._current_value(name),
                        )
                        for name in loop_carried
                    ] + [
                        ast.keyword(arg=slot_carry_names[slot], value=_name(slot_carry_names[slot]))
                        for slot in loop_carried_slots
                    ],
                ),
            )
            prologue = [
                ast.Assign(
                    targets=[_name(stmt.target.id, ast.Store())],
                    value=ast.Attribute(value=_name(loop_name), attr="iv", ctx=ast.Load()),
                )
            ]
            prologue.extend(
                ast.Assign(
                    targets=[_name(name, ast.Store())],
                    value=ast.Attribute(value=_name(loop_name), attr=name, ctx=ast.Load()),
                )
                for name in loop_carried
            )
            prologue.extend(
                ast.Assign(
                    targets=[_name(slot_carry_names[slot], ast.Store())],
                    value=ast.Attribute(value=_name(loop_name), attr=slot_carry_names[slot], ctx=ast.Load()),
                )
                for slot in loop_carried_slots
            )
            prologue.extend(
                ast.Assign(
                    targets=[_name(data["map_name"], ast.Store())],
                    value=ast.Dict(
                        keys=[ast.Constant(slot.index) for slot in data["slots"]],
                        values=[_name(slot_carry_names[slot]) for slot in data["slots"]],
                    ),
                )
                for data in slot_maps.values()
            )
            if loop_carried_slots:
                body = [
                    _SlotCarryRewriter(slot_maps, self._static_env, static_iters).visit(stmt)
                    for stmt in body
                ]
            slot_epilogue = [
                ast.Assign(
                    targets=[_name(slot_carry_names[slot], ast.Store())],
                    value=_map_subscript(slot_maps[slot.base]["map_name"], slot.index),
                )
                for slot in loop_carried_slots
            ]
            body = prologue + body + [
                *slot_epilogue,
                ast.Expr(
                    value=ast.Call(
                        func=ast.Attribute(value=_name(loop_name), attr="update", ctx=ast.Load()),
                        args=[],
                        keywords=[
                            ast.keyword(arg=name, value=_name(name))
                            for name in loop_carried
                        ] + [
                            ast.keyword(arg=slot_carry_names[slot], value=_name(slot_carry_names[slot]))
                            for slot in loop_carried_slots
                        ],
                    )
                )
            ]
            with_stmt = ast.With(
                items=[ast.withitem(context_expr=_name(loop_name), optional_vars=None)],
                body=body or [ast.Pass()],
                type_comment=None,
            )
            result = slot_initializers + [ast.copy_location(setup, stmt), ast.copy_location(with_stmt, stmt)]
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
            return result

        with_stmt = ast.With(
            items=[
                ast.withitem(
                    context_expr=ast.Call(
                        func=_pto_attr("for_"),
                        args=[start, stop],
                        keywords=[ast.keyword(arg="step", value=step), *hint_keywords],
                    ),
                    optional_vars=_name(stmt.target.id, ast.Store()),
                )
            ],
            body=body or [ast.Pass()],
            type_comment=None,
        )
        return [ast.copy_location(with_stmt, stmt)]

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

        iv_name = stmt.target.id
        skip_name = self._fresh("loop_active")
        did_break_name = self._fresh("loop_did_break")
        loop_name = self._fresh("loop")
        state_names = (iv_name,) + tuple(sorted(loop_carried)) + (skip_name, did_break_name)
        state_name = self._fresh("for_state")

        class _StateRewriter(ast.NodeTransformer):
            def visit_Name(inner, node):
                if node.id in state_names and isinstance(node.ctx, ast.Load):
                    return ast.copy_location(ast.Attribute(
                        value=_name(state_name), attr=node.id, ctx=ast.Load()), node)
                return node

        # Python range has direction-dependent bounds.  The sign-aware form
        # also handles a runtime step; a zero step is rejected by the runtime
        # range semantics before entering useful code in normal callers.
        iv_lt = ast.Compare(left=_name(iv_name), ops=[ast.Lt()], comparators=[copy.deepcopy(stop)])
        iv_gt = ast.Compare(left=_name(iv_name), ops=[ast.Gt()], comparators=[copy.deepcopy(stop)])
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
            right=ast.Compare(left=_name(did_break_name), ops=[ast.Eq()], comparators=[_flag_const(False)]),
        )
        condition = _StateRewriter().visit(condition)
        ast.fix_missing_locations(condition)
        condition_fn = ast.Lambda(
            args=ast.arguments(posonlyargs=[], args=[ast.arg(arg=state_name)], vararg=None,
                               kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]),
            body=condition,
        )
        initial_values = [
            copy.deepcopy(start),
            *[_name(name) for name in sorted(loop_carried)],
            _flag_const(True),
            _flag_const(False),
        ]
        setup = ast.Assign(
            targets=[_name(loop_name, ast.Store())],
            value=ast.Call(func=_pto_attr("_while"), args=[condition_fn], keywords=[
                ast.keyword(arg=name, value=value)
                for name, value in zip(state_names, initial_values)
            ]),
        )

        self._loop_control_stack.append({"active": skip_name, "did_break": did_break_name})
        try:
            body = self._rewrite_loop_body(
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
        prologue = [
            ast.Assign(targets=[_name(name, ast.Store())],
                       value=ast.Attribute(value=_name(loop_name), attr=name, ctx=ast.Load()))
            for name in state_names
        ]
        # active is per-iteration execution state; did_break remains sticky.
        # The body is segmented by _guard_control_tail on the merged active
        # value, so no whole-body guard is needed here.
        prologue.append(ast.Assign(targets=[_name(skip_name, ast.Store())], value=_flag_const(True)))
        updates = [
            ast.keyword(arg=iv_name, value=ast.BinOp(left=_name(iv_name), op=ast.Add(), right=copy.deepcopy(step))),
            *[ast.keyword(arg=name, value=_name(name)) for name in sorted(loop_carried)],
            ast.keyword(arg=skip_name, value=_name(skip_name)),
            ast.keyword(arg=did_break_name, value=_name(did_break_name)),
        ]
        body.append(ast.Expr(value=ast.Call(
            func=ast.Attribute(value=_name(loop_name), attr="update", ctx=ast.Load()),
            args=[], keywords=updates)))
        with_stmt = ast.With(
            items=[ast.withitem(context_expr=_name(loop_name), optional_vars=None)],
            body=prologue + body,
            type_comment=None,
        )
        result = [ast.copy_location(setup, stmt), ast.copy_location(with_stmt, stmt)]
        result.extend(ast.Assign(
            targets=[_name(name, ast.Store())],
            value=ast.Call(func=ast.Attribute(value=_name(loop_name), attr="final", ctx=ast.Load()),
                           args=[ast.Constant(name)], keywords=[]),
        ) for name in state_names if name in live_after or name in {skip_name, did_break_name})
        if stmt.orelse:
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
        return result

    def _rewrite_while(self, stmt, *, live_after, live_after_slots=None,
                       allow_loop_control=False, static_iters=None, bound_before=None):
        """Lower runtime ``while`` using named state and explicit control flags."""
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
        if (control["break"] or control["continue"] or stmt.orelse) and not body_info.stores:
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

        active_name = self._fresh("while_active")
        did_break_name = self._fresh("while_did_break")
        controlled = bool(stmt.orelse or control["break"] or control["continue"])
        state_names = carry_names + ((active_name, did_break_name) if controlled else ())
        loop_name = self._fresh("while")
        state_name = self._fresh("while_state")

        class _ConditionStateRewriter(ast.NodeTransformer):
            def visit_Name(inner, node):
                if node.id in state_names and isinstance(node.ctx, ast.Load):
                    return ast.copy_location(
                        ast.Attribute(value=_name(state_name), attr=node.id,
                                      ctx=ast.Load()), node)
                return node

        class _ConditionLiteralRewriter(ast.NodeTransformer):
            # A literal test condition (``while True:`` / ``while False:``)
            # must be materialized as an i1 constant, not a Python bool: the
            # controlled condition is combined with ``and``, whose runtime
            # scalar operator would otherwise reject the bool literal.
            def visit_Constant(inner, node):
                if isinstance(node.value, bool):
                    return ast.copy_location(_flag_const(node.value), node)
                return node

        condition = _ConditionStateRewriter().visit(copy.deepcopy(stmt.test))
        if controlled:
            condition = ast.BinOp(
                left=condition,
                op=ast.BitAnd(),
                right=ast.Compare(
                    left=_name(did_break_name), ops=[ast.Eq()],
                    comparators=[_flag_const(False)],
                ),
            )
            condition = _ConditionStateRewriter().visit(condition)
        condition = _ConditionLiteralRewriter().visit(condition)
        ast.fix_missing_locations(condition)
        condition_fn = ast.Lambda(
            args=ast.arguments(
                posonlyargs=[], args=[ast.arg(arg=state_name)], vararg=None,
                kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]),
            body=condition,
        )
        setup = ast.Assign(
            targets=[_name(loop_name, ast.Store())],
            value=ast.Call(
                func=_pto_attr("_while"), args=[condition_fn], keywords=[
                    ast.keyword(
                        arg=name,
                        value=(_flag_const(True) if name == active_name else
                               _flag_const(False) if name == did_break_name else
                               _name(name)),
                    ) for name in state_names
                ],
            ),
        )

        if controlled:
            self._loop_control_stack.append({"active": active_name, "did_break": did_break_name})
        try:
            body = self._rewrite_loop_body(
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
        with_stmt = ast.With(
            items=[ast.withitem(context_expr=_name(loop_name), optional_vars=None)],
            body=prologue + body + [update], type_comment=None,
        )
        result = [ast.copy_location(setup, stmt),
                  ast.copy_location(with_stmt, stmt)]
        result.extend(
            ast.Assign(
                targets=[_name(name, ast.Store())],
                value=ast.Call(
                    func=ast.Attribute(value=_name(loop_name), attr="final",
                                       ctx=ast.Load()),
                    args=[ast.Constant(name)], keywords=[],
                ),
            ) for name in carry_names if name in live_after
        )
        if controlled:
            result.extend(
                ast.Assign(
                    targets=[_name(name, ast.Store())],
                    value=ast.Call(
                        func=ast.Attribute(value=_name(loop_name), attr="final", ctx=ast.Load()),
                        args=[ast.Constant(name)], keywords=[]),
                ) for name in (active_name, did_break_name)
            )
        if stmt.orelse:
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
        return result


__all__ = [
    "PTODSLAstRewriteError",
    "rewrite_jit_function",
]
