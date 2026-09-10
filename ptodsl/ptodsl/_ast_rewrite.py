# coding=utf-8
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
import inspect
import textwrap

from ._ast_rewrite_errors import PTODSLAstRewriteError
from ._ast_rewrite_analysis import (
    _SubscriptSlot,
    _drop_unreachable_tails,
    _read_before_assignment_slots,
)
from ._ast_rewrite_normalizers import (
    _BoolOpRewriter,
    _ConditionalExpressionNormalizer,
    _SectionLexicalRewriter,
    _find_function_def,
    _inject_closure_defaults,
    _restore_globals,
    _sanitize_signature_for_exec,
    _temporarily_bind_globals,
)
from ._ast_rewrite_lowering import _ControlFlowRewriter


def rewrite_jit_function(
    fn,
    *,
    static_bindings=None,
    rewrite_control_flow=True,
    reject_bare_returns: bool = False,
):
    """Return a function with PTODSL lexical sections lowered safely.

    ``rewrite_control_flow`` controls both runtime statement rewriting and the
    ``and``/``or`` short-circuit rewrite.  ``pto.section`` is a physical SSA
    region, not a Python ``with`` hint.  Its body therefore gets a small
    source-level lexical rewrite even when
    the optional control-flow rewrite is disabled.  This keeps Python's
    function-local assignment rules from leaking a section-local SSA value into
    a sibling physical section.
    ``reject_bare_returns`` controls whether ``return`` inside rewritten
    control flow is rejected. ``@pto.jit`` keeps the historical behavior, while
    ``@pto.func`` enables this because helper bodies must keep one helper ABI.
    """
    source_info = _fetch_function_source(fn)
    if source_info is None:
        return fn
    source, source_start_line = source_info
    tree = ast.parse(textwrap.dedent(source))
    function_def = _find_function_def(tree, fn.__name__)
    if function_def is None:
        return fn
    function_def, static_env, closure_vars, section_rewriter, function_first_lineno = (
        _normalize_function_ast(function_def, fn, static_bindings, rewrite_control_flow)
    )
    if rewrite_control_flow:
        function_def = _rewrite_body_control_flow(
            function_def, static_env, section_rewriter, reject_bare_returns
        )
    rewritten = _exec_rewritten_function(
        function_def, fn, source_start_line, function_first_lineno, closure_vars
    )
    return _apply_function_metadata(fn, rewritten, closure_vars)


def _fetch_function_source(fn):
    """Return ``(source, source_start_line)`` for ``fn``, or ``None`` when the
    source is not retrievable (dynamically-created functions from
    exec/REPL/notebook contexts keep the existing tracing behavior instead of
    making default-on AST rewrite a compatibility break)."""
    try:
        source = inspect.getsource(fn)
        _, source_start_index = inspect.findsource(fn)
        return source, source_start_index + 1
    except (OSError, TypeError):
        return None


def _normalize_function_ast(function_def, fn, static_bindings, rewrite_control_flow):
    """Prepare the function AST for exec: decorators, closures, and the
    lexical section / conditional-expression normalizers."""
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
    if rewrite_control_flow:
        function_def = _BoolOpRewriter().visit(function_def)
    function_def = _ConditionalExpressionNormalizer().visit(function_def)
    section_rewriter = _SectionLexicalRewriter()
    function_def = section_rewriter.visit(function_def)
    return (
        function_def,
        static_env,
        closure_vars,
        section_rewriter,
        function_first_lineno,
    )


def _rewrite_body_control_flow(function_def, static_env, section_rewriter, reject_bare_returns):
    """Lower the whole function body through the control-flow rewriter."""
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
    return function_def


def _exec_rewritten_function(
    function_def, fn, source_start_line, function_first_lineno, closure_vars
):
    """Compile and exec the rewritten module, returning the new function."""
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
    if function_def.name not in locals_ns:
        raise KeyError(
            f"exec of the rewritten function did not bind {function_def.name!r}"
        )
    return locals_ns[function_def.name]


def _apply_function_metadata(fn, rewritten, closure_vars):
    """Copy defaults, annotations, and identity attributes onto the result."""
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


__all__ = [
    "PTODSLAstRewriteError",
    "rewrite_jit_function",
]
