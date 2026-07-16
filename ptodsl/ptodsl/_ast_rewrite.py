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


def rewrite_jit_function(fn):
    """Return a function whose Python if/for control flow lowers to PTODSL APIs."""
    try:
        source = inspect.getsource(fn)
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

    function_def.decorator_list = []
    closure_vars = inspect.getclosurevars(fn)
    _inject_closure_defaults(function_def, closure_vars.nonlocals)
    _sanitize_signature_for_exec(function_def)
    rewriter = _ControlFlowRewriter()
    function_def.body = rewriter.rewrite_block(function_def.body, live_after=set())
    tree = ast.Module(body=[function_def], type_ignores=[])
    ast.fix_missing_locations(tree)

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


@dataclass(frozen=True)
class _NameInfo:
    loads: set[str]
    stores: set[str]


class _NameInfoVisitor(ast.NodeVisitor):
    def __init__(self):
        self.loads = set()
        self.stores = set()
        self._comprehension_bindings = set()

    def visit_Name(self, node):
        if node.id in self._comprehension_bindings:
            return
        if isinstance(node.ctx, ast.Load):
            self.loads.add(node.id)
        elif isinstance(node.ctx, (ast.Store, ast.Del)):
            self.stores.add(node.id)

    def visit_AugAssign(self, node):
        self._visit_augassign_target_load(node.target)
        self.visit(node.value)
        self.visit(node.target)

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
        self._add_loads(_function_free_vars(node))

    def visit_AsyncFunctionDef(self, node):
        self.visit_FunctionDef(node)

    def visit_Lambda(self, node):
        self._visit_arguments_defaults(node.args)
        self._add_loads(_lambda_free_vars(node))

    def visit_ClassDef(self, node):
        self.stores.add(node.name)
        for decorator in node.decorator_list:
            self.visit(decorator)
        for base in node.bases:
            self.visit(base)
        for keyword in node.keywords:
            self.visit(keyword.value)
        self._add_loads(_class_body_free_vars(node))

    def _visit_arguments_defaults(self, args):
        for default in args.defaults:
            self.visit(default)
        for default in args.kw_defaults:
            if default is not None:
                self.visit(default)

    def _add_loads(self, names):
        self.loads.update(name for name in names if name not in self._comprehension_bindings)

    def _visit_comprehension(self, generators, result_nodes):
        saved_bindings = self._comprehension_bindings
        self._comprehension_bindings = set(saved_bindings)
        try:
            for generator in generators:
                # The iterator is evaluated before this generator target is
                # bound. Later generators and the result run in the nested
                # comprehension scope.
                self.visit(generator.iter)
                self._comprehension_bindings.update(_target_binding_names(generator.target))
                self.visit(generator.target)
                for condition in generator.ifs:
                    self.visit(condition)
            for result_node in result_nodes:
                self.visit(result_node)
        finally:
            self._comprehension_bindings = saved_bindings

    def visit_ListComp(self, node):
        self._visit_comprehension(node.generators, [node.elt])

    def visit_SetComp(self, node):
        self._visit_comprehension(node.generators, [node.elt])

    def visit_DictComp(self, node):
        self._visit_comprehension(node.generators, [node.key, node.value])

    def visit_GeneratorExp(self, node):
        self._visit_comprehension(node.generators, [node.elt])


def _name_info(node) -> _NameInfo:
    visitor = _NameInfoVisitor()
    if isinstance(node, list):
        for item in node:
            visitor.visit(item)
    else:
        visitor.visit(node)
    return _NameInfo(visitor.loads, visitor.stores)


class _ScopeBindingVisitor(ast.NodeVisitor):
    def __init__(self):
        self.stores = set()
        self.globals = set()
        self.nonlocals = set()
        self._comprehension_bindings = set()

    def visit_Name(self, node):
        if node.id not in self._comprehension_bindings and isinstance(node.ctx, (ast.Store, ast.Del)):
            self.stores.add(node.id)

    def visit_FunctionDef(self, node):
        self.stores.add(node.name)

    def visit_AsyncFunctionDef(self, node):
        self.visit_FunctionDef(node)

    def visit_Lambda(self, node):
        return

    def visit_ClassDef(self, node):
        self.stores.add(node.name)

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
        saved_bindings = self._comprehension_bindings
        self._comprehension_bindings = set(saved_bindings)
        try:
            for generator in generators:
                self.visit(generator.iter)
                self._comprehension_bindings.update(_target_binding_names(generator.target))
                self.visit(generator.target)
                for condition in generator.ifs:
                    self.visit(condition)
            for result_node in result_nodes:
                self.visit(result_node)
        finally:
            self._comprehension_bindings = saved_bindings

    def visit_ListComp(self, node):
        self._visit_comprehension(node.generators, [node.elt])

    def visit_SetComp(self, node):
        self._visit_comprehension(node.generators, [node.elt])

    def visit_DictComp(self, node):
        self._visit_comprehension(node.generators, [node.key, node.value])

    def visit_GeneratorExp(self, node):
        self._visit_comprehension(node.generators, [node.elt])


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


def _target_binding_names(node) -> set[str]:
    if isinstance(node, ast.Name):
        return {node.id}
    if isinstance(node, ast.Starred):
        return _target_binding_names(node.value)
    if isinstance(node, (ast.List, ast.Tuple)):
        names = set()
        for element in node.elts:
            names.update(_target_binding_names(element))
        return names
    return set()


def _live_before_block(stmts, live_after) -> set[str]:
    live = set(live_after)
    for stmt in reversed(stmts):
        live = _live_before_stmt(stmt, live)
    return live


def _live_before_stmt(stmt, live_after) -> set[str]:
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
    info = _name_info(stmt)
    return (set(live_after) - info.stores) | info.loads


def _read_before_assignment_names(stmts):
    return _live_before_block(stmts, set())


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


def _range_triplet(call):
    if not _is_range_call(call):
        raise PTODSLAstRewriteError("ast_rewrite=True only rewrites for-loops over range(...)")
    if call.keywords:
        raise PTODSLAstRewriteError("ast_rewrite=True range(...) loops do not support keyword arguments")
    args = call.args
    if len(args) == 1:
        return ast.Constant(0), args[0], ast.Constant(1)
    if len(args) == 2:
        return args[0], args[1], ast.Constant(1)
    if len(args) == 3:
        return args[0], args[1], args[2]
    raise PTODSLAstRewriteError("ast_rewrite=True range(...) loops require 1 to 3 arguments")


def _pto_attr(name: str, ctx=ast.Load()):
    return ast.Attribute(value=ast.Name(id="pto", ctx=ast.Load()), attr=name, ctx=ctx)


def _name(name: str, ctx=ast.Load()):
    return ast.Name(id=name, ctx=ctx)


class _ControlFlowRewriter:
    def __init__(self):
        self._counter = 0

    def _fresh(self, prefix: str) -> str:
        value = f"__pto_ast_{prefix}_{self._counter}"
        self._counter += 1
        return value

    def rewrite_block(self, stmts, *, live_after, allow_loop_control=False):
        # Analyze the original source block before any nested rewrite replaces
        # a statement with generated AST. Generated carry/branch scaffolding
        # has different bindings from the authored Python source.
        rewrite_plan = []
        live = set(live_after)
        for stmt in reversed(stmts):
            rewrite_plan.append((stmt, set(live)))
            live = _live_before_stmt(stmt, live)

        rewritten_reversed = []
        for stmt, source_live_after in rewrite_plan:
            rewritten = self.rewrite_stmt(
                stmt,
                live_after=source_live_after,
                allow_loop_control=allow_loop_control,
            )
            rewritten_reversed[:0] = rewritten
        return rewritten_reversed

    def rewrite_stmt(self, stmt, *, live_after, allow_loop_control=False):
        if isinstance(stmt, ast.If):
            return self._rewrite_if(
                stmt,
                live_after=live_after,
                allow_loop_control=allow_loop_control,
            )
        if isinstance(stmt, ast.For):
            return self._rewrite_for(
                stmt,
                live_after=live_after,
                allow_loop_control=allow_loop_control,
            )
        if isinstance(stmt, (ast.Assign, ast.AnnAssign)) and isinstance(stmt.value, ast.IfExp):
            return self._rewrite_ifexp_assignment(
                stmt,
                live_after=live_after,
                allow_loop_control=allow_loop_control,
            )
        if isinstance(stmt, (ast.Break, ast.Continue)):
            if allow_loop_control:
                return [stmt]
            raise PTODSLAstRewriteError("ast_rewrite=True does not support break/continue in rewritten control flow")
        return [
            self._rewrite_nested(
                stmt,
                live_after=live_after,
                allow_loop_control=allow_loop_control,
            )
        ]

    def _rewrite_nested(self, stmt, *, live_after, allow_loop_control=False):
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            stmt.body = self.rewrite_block(
                stmt.body,
                live_after=set(),
                allow_loop_control=False,
            )
            return stmt
        if isinstance(stmt, (ast.Lambda, ast.ClassDef)):
            return stmt
        for field, value in ast.iter_fields(stmt):
            if field in {"body", "orelse", "finalbody"} and isinstance(value, list):
                setattr(
                    stmt,
                    field,
                    self.rewrite_block(
                        value,
                        live_after=live_after,
                        allow_loop_control=allow_loop_control,
                    ),
                )
            elif isinstance(value, ast.AST):
                self._rewrite_nested(
                    value,
                    live_after=live_after,
                    allow_loop_control=allow_loop_control,
                )
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST):
                        self._rewrite_nested(
                            item,
                            live_after=live_after,
                            allow_loop_control=allow_loop_control,
                        )
        return stmt

    def _rewrite_if(self, stmt, *, live_after, allow_loop_control=False):
        if _is_pto_attr_call(stmt.test, "const_expr"):
            stmt.body = self.rewrite_block(
                stmt.body,
                live_after=live_after,
                allow_loop_control=allow_loop_control,
            )
            stmt.orelse = self.rewrite_block(
                stmt.orelse,
                live_after=live_after,
                allow_loop_control=allow_loop_control,
            )
            return [stmt]

        cond_name = self._fresh("cond")
        then_info = _name_info(stmt.body)
        else_info = _name_info(stmt.orelse)
        assigned_any = then_info.stores | else_info.stores
        merge_names = tuple(sorted(live_after & assigned_any))
        old_value_names = {
            name: self._fresh(f"old_{name}")
            for name in merge_names
            if name not in then_info.stores or name not in else_info.stores
        }

        branch_live_after = set(live_after) | set(merge_names)
        then_body = self.rewrite_block(
            stmt.body,
            live_after=branch_live_after,
            allow_loop_control=False,
        )
        else_body = self.rewrite_block(
            stmt.orelse,
            live_after=branch_live_after,
            allow_loop_control=False,
        )
        trace_time_if = ast.If(
            test=_name(cond_name),
            body=copy.deepcopy(then_body) or [ast.Pass()],
            orelse=copy.deepcopy(else_body) or [ast.Pass()],
        )
        branch_name = self._fresh("br")

        dynamic_then_body = copy.deepcopy(then_body)
        dynamic_else_body = copy.deepcopy(else_body)
        if merge_names:
            dynamic_then_body.append(
                self._branch_assign(
                    branch_name,
                    merge_names,
                    old_value_names=old_value_names,
                    assigned_names=then_info.stores,
                )
            )
            dynamic_else_body.append(
                self._branch_assign(
                    branch_name,
                    merge_names,
                    old_value_names=old_value_names,
                    assigned_names=else_info.stores,
                )
            )

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
        dynamic_body = [with_stmt]
        dynamic_body.extend(
            ast.Assign(
                targets=[_name(name, ast.Store())],
                value=ast.Attribute(value=_name(branch_name), attr=name, ctx=ast.Load()),
            )
            for name in merge_names
        )

        result = [
            ast.Assign(
                targets=[_name(cond_name, ast.Store())],
                value=stmt.test,
            )
        ]
        result.extend(
            ast.Assign(
                targets=[_name(old_name, ast.Store())],
                value=_name(name),
            )
            for name, old_name in old_value_names.items()
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

    def _rewrite_ifexp_assignment(self, stmt, *, live_after, allow_loop_control=False):
        ifexp = stmt.value
        then_stmt = copy.deepcopy(stmt)
        then_stmt.value = copy.deepcopy(ifexp.body)
        else_stmt = copy.deepcopy(stmt)
        else_stmt.value = copy.deepcopy(ifexp.orelse)
        conditional = ast.copy_location(
            ast.If(
                test=copy.deepcopy(ifexp.test),
                body=[then_stmt],
                orelse=[else_stmt],
            ),
            stmt,
        )
        return self._rewrite_if(
            conditional,
            live_after=live_after,
            allow_loop_control=allow_loop_control,
        )

    def _branch_assign(self, branch_name, names, *, old_value_names, assigned_names):
        return ast.Expr(
            value=ast.Call(
                func=ast.Attribute(value=_name(branch_name), attr="assign", ctx=ast.Load()),
                args=[],
                keywords=[
                    ast.keyword(
                        arg=name,
                        value=_name(name if name in assigned_names else old_value_names[name]),
                    )
                    for name in names
                ],
            )
        )

    def _rewrite_for(self, stmt, *, live_after, allow_loop_control=False):
        if _is_pto_attr_call(stmt.iter, "static_range"):
            stmt.body = self.rewrite_block(
                stmt.body,
                live_after=live_after,
                allow_loop_control=True,
            )
            stmt.orelse = self.rewrite_block(
                stmt.orelse,
                live_after=live_after,
                allow_loop_control=allow_loop_control,
            )
            return [stmt]

        if stmt.orelse:
            raise PTODSLAstRewriteError("ast_rewrite=True does not support for-else on runtime loops")
        if not isinstance(stmt.target, ast.Name):
            raise PTODSLAstRewriteError("ast_rewrite=True runtime for-loops require a simple name target")
        if stmt.target.id in live_after:
            raise PTODSLAstRewriteError(
                "ast_rewrite=True runtime for-loops cannot expose the loop induction variable outside the loop yet; "
                f"use explicit pto.for_(...) for {stmt.target.id!r}"
            )

        start, stop, step = _range_triplet(stmt.iter)
        body_info = _name_info(stmt.body)
        reads_before = _read_before_assignment_names(stmt.body)
        assigned_live_after = body_info.stores & set(live_after)
        loop_carried = tuple(sorted(body_info.stores & reads_before))
        unsupported_last_values = sorted(assigned_live_after - set(loop_carried))
        if unsupported_last_values:
            raise PTODSLAstRewriteError(
                "ast_rewrite=True runtime for-loops cannot expose last-iteration-only values yet; "
                f"use explicit pto.for_(...).carry(...) for {unsupported_last_values}"
            )

        loop_name = self._fresh("loop")
        loop_live_after = set(live_after) | set(loop_carried)
        body = self.rewrite_block(stmt.body, live_after=loop_live_after)

        if loop_carried:
            setup = ast.Assign(
                targets=[_name(loop_name, ast.Store())],
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Call(
                            func=_pto_attr("for_"),
                            args=[start, stop],
                            keywords=[ast.keyword(arg="step", value=step)],
                        ),
                        attr="carry",
                        ctx=ast.Load(),
                    ),
                    args=[],
                    keywords=[
                        ast.keyword(arg=name, value=_name(name))
                        for name in loop_carried
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
            body = prologue + body + [
                ast.Expr(
                    value=ast.Call(
                        func=ast.Attribute(value=_name(loop_name), attr="update", ctx=ast.Load()),
                        args=[],
                        keywords=[
                            ast.keyword(arg=name, value=_name(name))
                            for name in loop_carried
                        ],
                    )
                )
            ]
            with_stmt = ast.With(
                items=[ast.withitem(context_expr=_name(loop_name), optional_vars=None)],
                body=body or [ast.Pass()],
                type_comment=None,
            )
            result = [ast.copy_location(setup, stmt), ast.copy_location(with_stmt, stmt)]
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
            return result

        with_stmt = ast.With(
            items=[
                ast.withitem(
                    context_expr=ast.Call(
                        func=_pto_attr("for_"),
                        args=[start, stop],
                        keywords=[ast.keyword(arg="step", value=step)],
                    ),
                    optional_vars=_name(stmt.target.id, ast.Store()),
                )
            ],
            body=body or [ast.Pass()],
            type_comment=None,
        )
        return [ast.copy_location(with_stmt, stmt)]


__all__ = [
    "PTODSLAstRewriteError",
    "rewrite_jit_function",
]
