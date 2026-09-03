# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
Control-flow context managers for PTO kernels.

All CMs work with the current MLIR insertion point; no context threading needed.

Public API
──────────
``section(kind)``         – ``pto.section.cube/vector { … }``
``vecscope()``            – ``pto.vecscope { … }``
``for_(lo, hi, step)``
                          – ``scf.for`` with optional named carry state via ``.carry(...)``
                          and an optional loop-unroll hint (``unroll=`` / ``unroll_factor=``)
``if_(cond)``             – ``scf.if`` via explicit branch handle + automatic named merge
``yield_(*vals)``         – ``scf.yield``
``static_range(...)``     – trace-time ``range(...)`` escape hatch for AST rewrite
``range(...)``            – AST-rewrite marker carrying loop-unroll hints for native ``for``
``const_expr(value)``     – trace-time ``if`` escape hatch for AST rewrite
``_short_circuit_and/or``  – internal device-side ``and``/``or`` helpers used by the AST rewriter
"""

import builtins
import warnings

from ._diagnostics import explicit_mode_required_with_context_error
from ._runtime_index_ops import coerce_runtime_index
from ._scalar_adaptation import coerce_integer_like, coerce_runtime_i1_value, coerce_runtime_integer_to_i1
from ._scalar_coercion import coerce_scalar_to_type
from ._surface_types import const_expr
from ._tracing.active import current_session, require_active_session
from ._tracing.control_flow import apply_unroll_hint, normalize_unroll_hint
from ._surface_values import (
    RuntimeValue,
    unwrap_surface_value,
    wrap_like_surface_value,
    wrap_surface_value,
)
from ._types import _StructDescriptor

from ptoas.mlir.dialects import arith, pto as _pto, scf
from ptoas.mlir.ir import IndexType, InsertionPoint, IntegerType


# ── vecscope ──────────────────────────────────────────────────────────────────

def _require_explicit_mode(surface: str):
    session = current_session()
    if session is None:
        return
    current_module_spec = getattr(session, "current_function_module_spec", session.module_spec)
    if getattr(current_module_spec, "mode", None) != "explicit":
        raise explicit_mode_required_with_context_error(surface, current_module_spec)

class _VecScopeCM:
    """Context manager for ``pto.vecscope { … }``."""

    def __enter__(self):
        _require_explicit_mode("pto.vecscope()")
        self._op = _pto.VecScopeOp()
        self._block = self._op.body.blocks.append()
        self._ip = InsertionPoint(self._block)
        self._ip.__enter__()
        return None

    def __exit__(self, *exc):
        self._ip.__exit__(*exc)


def vecscope() -> _VecScopeCM:
    """Return a context manager that emits ``pto.vecscope { … }``."""
    return _VecScopeCM()


class _SectionCM:
    """Context manager for an explicit cube or vector physical section."""

    def __init__(self, kind: str):
        if not isinstance(kind, str):
            raise TypeError("pto.section(kind) expects 'cube' or 'vector'")
        if kind not in {"cube", "vector"}:
            raise ValueError("pto.section(kind) expects 'cube' or 'vector'")
        self._kind = kind
        self._cm = None

    def __enter__(self):
        _require_explicit_mode("pto.section()")
        session = require_active_session("pto.section")
        self._cm = session.enter_physical_section(self._kind)
        self._cm.__enter__()
        return None

    def __exit__(self, *exc):
        return self._cm.__exit__(*exc)


def section(kind: str) -> _SectionCM:
    """Assign enclosed operations to one core; the context binds no ``as`` value."""
    return _SectionCM(kind)


# ── compile-time control-flow helpers ─────────────────────────────────────────

def static_range(*args):
    """Return ``range(*args)`` for trace-time unrolling under AST rewrite."""
    return builtins.range(*args)


class _PtoRangeMarker:
    """Marker for ``for i in pto.range(...)`` loops under AST rewrite.

    The AST rewriter recognizes ``pto.range(...)`` as a native-``for`` loop
    iterable that may carry a loop-unroll hint and rewrites the whole loop to
    ``pto.for_(..., unroll=..., unroll_factor=...)``.  The call itself is
    never evaluated in that flow; reaching this marker at runtime means the
    enclosing kernel was not AST-rewritten (or the marker was used outside a
    ``for`` statement).
    """

    def __init__(self, args, kwargs):
        self._args = args
        self._kwargs = kwargs

    def __iter__(self):
        raise RuntimeError(
            "pto.range(...) may only be used as the iterable of a 'for' loop "
            "inside an AST-rewritten kernel; use 'with pto.for_(...)' for "
            "explicitly authored loops"
        )


def range(*args, **kwargs):
    """AST-rewrite marker: ``for i in pto.range(start, stop[, step], unroll=...)``.

    Accepts the same positional start/stop/step arguments as the builtin
    ``range`` plus optional ``unroll=`` / ``unroll_factor=`` hints with the
    same semantics as ``pto.for_``.  The enclosing loop must be processed by
    the PTODSL AST rewriter; the marker itself is not a runtime iterable.
    """
    return _PtoRangeMarker(args, kwargs)

# ── for_ ──────────────────────────────────────────────────────────────────────

class LoopHandle:
    """
    Internal handle for a lowered ``scf.for`` loop.

    Attributes used by the control-flow implementation::

        loop.iv         – induction variable
        loop.iter_args  – tuple of inner (mutable) SSA values
        loop.results    – tuple of ForOp results (after loop exit)
    """

    def __init__(self, for_op, *, iter_arg_templates=()):
        self._op = for_op
        self._iter_arg_templates = tuple(iter_arg_templates)

    @property
    def iv(self):
        return wrap_surface_value(self._op.induction_variable)

    @property
    def iter_args(self):
        return tuple(
            wrap_like_surface_value(template, value)
            for template, value in zip(self._iter_arg_templates, self._op.inner_iter_args)
        )

    @property
    def results(self):
        return tuple(
            wrap_like_surface_value(template, value)
            for template, value in zip(self._iter_arg_templates, self._op.results)
        )


class _ForCM:
    def __init__(self, start, stop, step, iter_args, unroll=None, unroll_factor=None):
        self._start = start
        self._stop = stop
        self._step = step
        self._unroll = unroll
        self._unroll_factor = unroll_factor
        self._iter_arg_templates = tuple(iter_args) if iter_args is not None else ()
        self._iter_args = [unwrap_surface_value(value) for value in self._iter_arg_templates]
        self._for_op = None
        self._ip = None

    def __enter__(self):
        self._for_op = scf.ForOp(
            _coerce_index(self._start),
            _coerce_index(self._stop),
            _coerce_index(self._step),
            self._iter_args if self._iter_args else None,
        )
        apply_unroll_hint(self._for_op, self._unroll, self._unroll_factor)
        self._ip = InsertionPoint(self._for_op.body)
        self._ip.__enter__()
        if not self._iter_args:
            return wrap_surface_value(self._for_op.induction_variable)
        return LoopHandle(self._for_op, iter_arg_templates=self._iter_arg_templates)

    def __exit__(self, *exc):
        if not self._iter_args:
            scf.YieldOp([])
        self._ip.__exit__(*exc)


def for_(start, stop, *, step, unroll=None, unroll_factor=None):
    """
    ``scf.for`` context manager.

    Yields the induction variable; ``scf.yield`` is inserted automatically::

        with pto.for_(c0, c16, step=c1) as i:
            ...

    Named carry state is expressed with ``.carry(...)``::

        loop = pto.for_(c0, c128, step=c64).carry(acc=tile)
        with loop:
            cur = loop.acc
            loop.update(acc=cur)
        out = loop.final("acc")

    An optional loop-unroll hint is forwarded to the compiler::

        with pto.for_(c0, c16, step=c1, unroll="enable") as i:
            ...

    ``unroll="enable"`` keeps the loop and forwards
    ``llvm.loop.unroll.enable`` metadata, letting the compiler's cost model
    decide whether and how to unroll (equivalent to a no-factor
    ``#pragma unroll``).  ``unroll="full"`` asks PTOAS to unroll the loop
    completely when the trip count is a compile-time constant (otherwise the
    hint is dropped with a remark).  ``unroll_factor=N`` asks PTOAS to
    unroll by ``N`` (an epilogue loop handles the remainder; dynamic upper
    bounds are supported).  The two arguments are mutually exclusive.
    Loops without a hint are unchanged.
    """
    normalize_unroll_hint(unroll, unroll_factor, context="pto.for_(...)")
    return _ForBuilder(start, stop, step, unroll=unroll, unroll_factor=unroll_factor)


class _WhileStateView:
    def __init__(self, names, values, templates):
        self._values = {
            name: wrap_like_surface_value(template, value)
            for name, template, value in zip(names, templates, values)
        }

    def __getattr__(self, name):
        try:
            return self._values[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


class _WhileCM:
    """Internal builder for one ``scf.while`` with named loop-carried state."""

    def __init__(self, condition, state_items):
        self._condition = condition
        self._state_items = tuple(state_items)
        self._state_names = tuple(name for name, _ in self._state_items)
        self._templates = tuple(value for _, value in self._state_items)
        self._inits = [_materialize_carry_init(value) for value in self._templates]
        self._op = None
        self._ip = None
        self._yield_values = None
        self._after_state = None

    def __enter__(self):
        result_types = [value.type for value in self._inits]
        self._op = scf.WhileOp(result_types, self._inits)
        before = self._op.before.blocks.append(*result_types)
        with InsertionPoint(before):
            state = _WhileStateView(self._state_names, before.arguments, self._templates)
            condition = unwrap_surface_value(self._condition(state))
            scf.ConditionOp(condition, list(before.arguments))
        after = self._op.after.blocks.append(*result_types)
        self._after_state = _WhileStateView(self._state_names, after.arguments, self._templates)
        self._ip = InsertionPoint(after)
        self._ip.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if exc_type is None:
                if self._yield_values is None:
                    self.update(**{name: getattr(self._after_state, name) for name in self._state_names})
                scf.YieldOp(self._yield_values)
        finally:
            self._ip.__exit__(exc_type, exc, tb)

    def __getattr__(self, name):
        if name in self._state_names:
            return getattr(self._after_state, name)
        raise AttributeError(name)

    def update(self, **kwargs):
        missing = [name for name in self._state_names if name not in kwargs]
        extra = [name for name in kwargs if name not in self._state_names]
        if missing or extra:
            pieces = []
            if missing:
                pieces.append(f"missing: {', '.join(missing)}")
            if extra:
                pieces.append(f"unexpected: {', '.join(extra)}")
            raise RuntimeError("while.update(...) must match carry names exactly; " + "; ".join(pieces))
        if self._yield_values is not None:
            raise RuntimeError("while.update(...) may only be called once per loop body")
        self._yield_values = [unwrap_surface_value(kwargs[name]) for name in self._state_names]

    def final(self, name):
        try:
            index = self._state_names.index(name)
        except ValueError as exc:
            raise RuntimeError(f"unknown while carry state {name!r}") from exc
        return wrap_like_surface_value(self._templates[index], self._op.results[index])


class _WhileBuilder:
    def __init__(self, condition, state_items):
        self._condition = condition
        self._state_items = tuple(state_items)

    def __enter__(self):
        self._cm = _WhileCM(self._condition, self._state_items)
        return self._cm.__enter__()

    def __exit__(self, *exc):
        return self._cm.__exit__(*exc)

    def __getattr__(self, name):
        # AST-rewritten loops use hygienic state names for hidden control
        # flags.  They are private by convention, but still valid named
        # carries and must be visible through the loop handle.
        cm = self.__dict__.get("_cm")
        if cm is None:
            raise AttributeError(name)
        if name.startswith("_") and name not in getattr(cm, "_state_names", ()):
            raise AttributeError(name)
        return getattr(cm, name)


def _while(condition, **kwargs):
    if not callable(condition):
        raise TypeError("internal while builder expects condition(state)")
    if not kwargs:
        raise ValueError("internal while builder requires at least one named carry value")
    return _WhileBuilder(condition, tuple(kwargs.items()))


def while_(condition, **kwargs):
    warnings.warn(
        "pto.while_ is a low-level compatibility API; prefer native Python while",
        DeprecationWarning,
        stacklevel=2,
    )
    return _while(condition, **kwargs)


class _CarryLoopStateView:
    def __init__(self, names, values):
        self._names = tuple(names)
        self._values = dict(zip(self._names, values))

    def __getattr__(self, name):
        try:
            return self._values[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


class _CarryForCM(_ForCM):
    def __init__(self, start, stop, step, state_items, unroll=None, unroll_factor=None):
        self._state_items = tuple(state_items)
        self._state_names = tuple(name for name, _ in self._state_items)
        self._state_templates = tuple(value for _, value in self._state_items)
        self._session = None
        self._session_frame = None
        super().__init__(start, stop, step, self._state_templates,
                         unroll=unroll, unroll_factor=unroll_factor)
        self._yield_values = None
        self._entered = False

    def __enter__(self):
        self._session = current_session()
        if self._session is not None:
            self._session_frame = self._session.begin_carry_loop(
                self._start,
                self._stop,
                self._step,
                self._state_items,
                unroll=self._unroll,
                unroll_factor=self._unroll_factor,
            )
            self._for_op = self._session_frame.for_op
            handle = LoopHandle(self._for_op, iter_arg_templates=self._state_templates)
        else:
            handle = super().__enter__()
        self._entered = True
        self._yield_values = None
        self._loop_handle = handle
        self._state = _CarryLoopStateView(self._state_names, handle.iter_args)
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if self._session_frame is not None:
                self._session.finish_carry_loop(self._session_frame, exc_type, exc, tb)
                return None
            if exc_type is None:
                if self._yield_values is None:
                    raise RuntimeError(
                        "pto.for_(...).carry(...) requires loop.update(...) before leaving the loop body"
                    )
                scf.YieldOp(self._yield_values)
            return super().__exit__(exc_type, exc, tb)
        finally:
            self._entered = False
            self._session = None
            self._session_frame = None

    @property
    def iv(self):
        if not self._entered:
            raise RuntimeError("loop.iv is only available inside an active carry loop body")
        return self._loop_handle.iv

    def __getattr__(self, name):
        if name in self._state_names:
            if not self._entered:
                raise RuntimeError(f"loop.{name} is only available inside an active carry loop body")
            return getattr(self._state, name)
        raise AttributeError(name)

    def update(self, **kwargs):
        if not self._entered:
            raise RuntimeError("loop.update(...) may only be called inside the loop body")
        if self._session_frame is not None:
            self._session.update_carry_loop(self._session_frame, **kwargs)
            return
        missing = [name for name in self._state_names if name not in kwargs]
        extra = [name for name in kwargs if name not in self._state_names]
        if missing or extra:
            pieces = []
            if missing:
                pieces.append(f"missing: {', '.join(missing)}")
            if extra:
                pieces.append(f"unexpected: {', '.join(extra)}")
            raise RuntimeError("loop.update(...) must match carry names exactly; " + "; ".join(pieces))
        if self._yield_values is not None:
            raise RuntimeError("loop.update(...) may only be called once per loop body")
        self._yield_values = [
            unwrap_surface_value(kwargs[name])
            for name in self._state_names
        ]

    def final(self, name):
        if self._for_op is None:
            raise RuntimeError("loop.final(...) is only available after the loop has been built")
        try:
            index = self._state_names.index(name)
        except ValueError as exc:
            raise RuntimeError(
                f"loop.final(...) requested unknown carry state '{name}'; "
                f"expected one of: {', '.join(self._state_names)}"
            ) from exc
        return wrap_like_surface_value(self._state_templates[index], self._for_op.results[index])


class _ForBuilder:
    def __init__(self, start, stop, step, unroll=None, unroll_factor=None):
        self._start = start
        self._stop = stop
        self._step = step
        self._unroll = unroll
        self._unroll_factor = unroll_factor

    def __enter__(self):
        self._cm = _ForCM(self._start, self._stop, self._step, None,
                          unroll=self._unroll, unroll_factor=self._unroll_factor)
        return self._cm.__enter__()

    def __exit__(self, *exc):
        return self._cm.__exit__(*exc)

    def carry(self, **kwargs):
        if not kwargs:
            raise ValueError("carry(...) requires at least one named loop-carried value")
        for name, value in kwargs.items():
            if not isinstance(name, str) or not name:
                raise TypeError("carry(...) names must be non-empty strings")
            if isinstance(value, _StructDescriptor):
                raise TypeError(
                    "pto.for_(...).carry(...) does not accept pto.struct_type(...) descriptors; "
                    "declare the struct outside the loop and mutate it in place inside the loop body"
                )
        return _CarryForCM(self._start, self._stop, self._step, tuple(kwargs.items()),
                           unroll=self._unroll, unroll_factor=self._unroll_factor)


def _coerce_index(value):
    raw_value = unwrap_surface_value(value)
    return coerce_runtime_index(raw_value, context="pto.for_(...) loop bound")


def _materialize_carry_init(value):
    raw_value = unwrap_surface_value(value)
    if isinstance(raw_value, bool):
        raise TypeError("while loop-carried values do not accept Python bool literals")
    if isinstance(raw_value, int):
        from ptoas.mlir.ir import IntegerType

        return arith.ConstantOp(IntegerType.get_signless(32), raw_value).result
    if not hasattr(raw_value, "type"):
        raise TypeError("while loop-carried values must be typed SSA values")
    return raw_value


# ── if_ ───────────────────────────────────────────────────────────────────────

def _find_parent_block(op_view):
    """Return the block that directly contains *op_view*."""
    parent_op = op_view.operation.parent
    if parent_op is None:
        raise RuntimeError("unable to locate the parent block for pto.if_(...)")
    for region in parent_op.regions:
        for block in region.blocks:
            for candidate in block.operations:
                if candidate.operation is op_view.operation:
                    return block
    raise RuntimeError("unable to locate the parent block for pto.if_(...)")


def _move_block_ops(src_block, dst_block, *, yield_values):
    """Move all non-terminator ops from *src_block* into *dst_block* and yield."""
    with InsertionPoint(dst_block):
        terminator = scf.YieldOp(list(yield_values))
    yield_anchor = terminator.operation.opview
    for op in list(src_block.operations):
        if op.operation.name == "scf.yield":
            continue
        op.move_before(yield_anchor)


def _validate_matching_branch_names(then_assignment, else_assignment):
    """Validate merge names and return the authored result order."""
    then_names = set(then_assignment["raw_values"])
    else_names = set(else_assignment["raw_values"])
    if then_names == else_names:
        return then_assignment["order"]

    pieces = []
    missing_in_else = sorted(then_names - else_names)
    missing_in_then = sorted(else_names - then_names)
    if missing_in_else:
        pieces.append(f"missing in else: {', '.join(missing_in_else)}")
    if missing_in_then:
        pieces.append(f"missing in then: {', '.join(missing_in_then)}")
    raise RuntimeError(
        "br.assign(...) names must match across branches; " + "; ".join(pieces)
    )


def _resolved_branch_values(name, then_assignment, else_assignment, *, then_block,
                            else_block, short_circuit_merge=False):
    """Reconcile one pair of branch values and require matching result types."""
    then_value, else_value = _reconcile_branch_assignment_values(
        name,
        then_assignment["raw_values"][name],
        else_assignment["raw_values"][name],
        then_block=then_block,
        else_block=else_block,
        short_circuit_merge=short_circuit_merge,
    )
    if then_value.type != else_value.type:
        raise RuntimeError(
            f"br.assign(...) type mismatch for '{name}': "
            f"then branch yields {then_value.type}, else branch yields {else_value.type}"
        )
    return then_value, else_value


class _IfBranchCM:
    """Enters the insertion point of one branch block for ``with br.then_:`` style."""

    def __init__(self, owner, branch_name, block):
        self._owner = owner
        self._branch_name = branch_name
        self._block = block
        self._ip = None

    def __enter__(self):
        self._owner._enter_branch(self._branch_name)
        self._ip = InsertionPoint(self._block)
        self._ip.__enter__()

    def __exit__(self, *exc):
        try:
            self._ip.__exit__(*exc)
        finally:
            self._owner._leave_branch(self._branch_name)


class BranchHandle:
    """
    Handle for one authored ``pto.if_(...)`` branch pair.

    Usage::

        with pto.if_(cond) as br:
            with br.then_:
                br.assign(val=x)
            with br.else_:
                br.assign(val=y)
        out = br.val
    """

    def __init__(self, owner):
        self._owner = owner
        self.then_ = _IfBranchCM(owner, "then", owner._tmp_if.then_block)
        self.else_ = _IfBranchCM(owner, "else", owner._tmp_if.else_block)

    def assign(self, **kwargs):
        self._owner._assign_branch_values(kwargs)

    def get(self, name: str):
        """Return a merged value by name, including generated names.

        The AST rewriter uses generated names such as ``__pto_section_0_x``
        for lexical section bindings.  Keep those names out of the attribute
        namespace, but provide an explicit lookup path for the rewriter.
        """
        if not isinstance(name, str):
            raise TypeError("br.get(...) expects a string name")
        return self._owner._get_merged_value(name)

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return self._owner._get_merged_value(name)


class _IfCM:
    def __init__(self, cond, *, short_circuit_merge: bool = False):
        self._cond = cond
        self._short_circuit_merge = short_circuit_merge
        self._cond_value = None
        self._tmp_if = None
        self._parent_block = None
        self._active_branch = None
        self._branch_closed = {"then": False, "else": False}
        self._branch_entered = {"then": False, "else": False}
        self._branch_assignments = {"then": None, "else": None}
        self._merged_values = None
        self._finalized = False
        self._handle = None

    def __enter__(self):
        self._cond_value = unwrap_surface_value(self._cond)
        if not _is_i1_type(self._cond_value.type) and _is_integer_like_type(self._cond_value.type):
            self._cond_value = coerce_runtime_integer_to_i1(
                self._cond_value,
                context="pto.if_(...) condition",
            )
        self._tmp_if = scf.IfOp(self._cond_value, hasElse=True)
        self._parent_block = _find_parent_block(self._tmp_if)
        self._handle = BranchHandle(self)
        return self._handle

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None:
            self._erase_tmp_if()
            return None
        try:
            self._finalize()
        except Exception:
            self._erase_tmp_if()
            raise
        return None

    def _enter_branch(self, branch_name):
        if self._finalized:
            raise RuntimeError("pto.if_(...) branches are no longer available after the conditional closes")
        if self._active_branch is not None:
            raise RuntimeError(
                "pto.if_(...) does not support nested branch entry; close the current "
                f"br.{self._active_branch}_ block before entering br.{branch_name}_"
            )
        if self._branch_closed[branch_name]:
            raise RuntimeError(f"br.{branch_name}_ may only be entered once per pto.if_(...)")
        self._active_branch = branch_name
        self._branch_entered[branch_name] = True

    def _leave_branch(self, branch_name):
        if self._active_branch == branch_name:
            self._active_branch = None
        self._branch_closed[branch_name] = True

    def _assign_branch_values(self, kwargs):
        if self._active_branch is None:
            raise RuntimeError("br.assign(...) may only be used inside br.then_ or br.else_")
        if not kwargs:
            raise ValueError("br.assign(...) requires at least one named value")
        branch_name = self._active_branch
        if self._branch_assignments[branch_name] is not None:
            raise RuntimeError(f"br.{branch_name}_ may call br.assign(...) at most once")
        raw_values = {}
        templates = {}
        order = tuple(kwargs.keys())
        for name, value in kwargs.items():
            raw_value = unwrap_surface_value(value)
            if not hasattr(raw_value, "type") and not _is_branch_assign_literal(raw_value):
                raise TypeError(
                    "br.assign(...) expects PTO runtime values, authored surface values, "
                    "or Python scalar literals that can be inferred from the opposite branch; "
                    f"'{name}' received {type(value).__name__}"
                )
            raw_values[name] = raw_value
            templates[name] = value
        self._branch_assignments[branch_name] = {
            "order": order,
            "raw_values": raw_values,
            "templates": templates,
        }

    def _get_merged_value(self, name):
        if not self._finalized:
            raise RuntimeError(f"br.{name} is only available after the pto.if_(...) block closes")
        if self._merged_values is None or name not in self._merged_values:
            expected = ()
            if self._merged_values:
                expected = tuple(self._merged_values.keys())
            if expected:
                raise AttributeError(
                    f"br.{name} was not assigned by this conditional; "
                    f"expected one of: {', '.join(expected)}"
                )
            raise AttributeError(f"br.{name} was not assigned by this conditional")
        return self._merged_values[name]

    def _finalize(self):
        self._validate_no_stray_ops()
        if not any(self._branch_entered.values()):
            raise RuntimeError(
                "pto.if_(...) requires at least one explicit branch block; "
                "use 'with br.then_:' and optionally 'with br.else_:'"
            )
        merge_spec = self._validate_merge_spec()
        if merge_spec is None:
            self._finalize_side_effect_if()
        else:
            self._finalize_merged_if(merge_spec)
        self._finalized = True

    def _validate_no_stray_ops(self):
        parent_ops = list(self._parent_block.operations)
        if not parent_ops or parent_ops[-1].operation is not self._tmp_if.operation:
            raise RuntimeError(
                "pto.if_(...) body may only contain explicit 'with br.then_:' / "
                "'with br.else_:' blocks; PTODSL found operations emitted directly "
                "in the outer if body"
            )

    def _validate_merge_spec(self):
        then_assignment = self._branch_assignments["then"]
        else_assignment = self._branch_assignments["else"]
        if then_assignment is None and else_assignment is None:
            return None
        if then_assignment is None or else_assignment is None:
            raise RuntimeError(
                "automatic branch merge requires both br.then_ and br.else_ to call br.assign(...)"
            )

        order = _validate_matching_branch_names(then_assignment, else_assignment)
        resolved_then_values = {}
        resolved_else_values = {}
        result_types = []
        for name in order:
            then_value, else_value = _resolved_branch_values(
                name,
                then_assignment,
                else_assignment,
                then_block=self._tmp_if.then_block,
                else_block=self._tmp_if.else_block,
                short_circuit_merge=self._short_circuit_merge,
            )
            resolved_then_values[name] = then_value
            resolved_else_values[name] = else_value
            result_types.append(then_value.type)

        return {
            "order": order,
            "result_types": result_types,
            "then": {
                **then_assignment,
                "raw_values": resolved_then_values,
            },
            "else": {
                **else_assignment,
                "raw_values": resolved_else_values,
            },
        }

    def _finalize_side_effect_if(self):
        has_else = self._branch_entered["else"]
        final_if = scf.IfOp(self._cond_value, hasElse=has_else)
        _move_block_ops(self._tmp_if.then_block, final_if.then_block, yield_values=[])
        if has_else:
            _move_block_ops(self._tmp_if.else_block, final_if.else_block, yield_values=[])
        self._merged_values = {}
        self._tmp_if.erase()
        self._tmp_if = final_if

    def _finalize_merged_if(self, merge_spec):
        final_if = scf.IfOp(self._cond_value, merge_spec["result_types"], hasElse=True)
        then_yield_values = [
            merge_spec["then"]["raw_values"][name]
            for name in merge_spec["order"]
        ]
        else_yield_values = [
            merge_spec["else"]["raw_values"][name]
            for name in merge_spec["order"]
        ]
        _move_block_ops(self._tmp_if.then_block, final_if.then_block, yield_values=then_yield_values)
        _move_block_ops(self._tmp_if.else_block, final_if.else_block, yield_values=else_yield_values)

        merged = {}
        for name, template, result in zip(
            merge_spec["order"],
            (merge_spec["then"]["templates"][name] for name in merge_spec["order"]),
            final_if.results,
        ):
            merged[name] = wrap_like_surface_value(template, result)
        self._merged_values = merged
        self._tmp_if.erase()
        self._tmp_if = final_if

    def _erase_tmp_if(self):
        if self._tmp_if is None:
            return
        try:
            self._tmp_if.erase()
        except Exception:
            pass
        finally:
            self._tmp_if = None


def if_(cond) -> _IfCM:
    """
    ``scf.if`` context manager with explicit branch handles.

    Side-effect-only form::

        with pto.if_(has_rows) as br:
            with br.then_:
                ...

    Automatic named merge form::

        with pto.if_(has_chunk) as br:
            with br.then_:
                br.assign(x=a)
            with br.else_:
                br.assign(x=b)
        x = br.x
    """
    return _IfCM(cond)


# ── short-circuit and/or (AST-rewrite targets) ──────────────────────────────

def _branch_rhs_value(value, kind):
    """Validate a short-circuit RHS evaluated inside a runtime branch.

    A branch may only yield a PTO runtime scalar value, an i1-materializable
    bool literal, or a Python integer literal.  Python float literals are
    intentionally excluded because the short-circuit merge has no type anchor
    for a floating-point result; static float operands still use native Python
    short-circuiting before this helper is called.
    """
    if isinstance(value, (bool, int, RuntimeValue)):
        return value
    raise TypeError(
        "pto short-circuit " + kind + " RHS must be a PTO runtime scalar value or a "
        "Python bool/int literal; Python float literals are not supported in "
        "runtime branch merges, got " + type(value).__name__,
    )


def _materialize_bool_branch_value(value, *, context):
    """Materialize a Python ``bool`` operand to a signless ``i1`` constant.

    Python ``and``/``or`` may legally return a plain ``bool`` literal from one
    branch (``flag and True``).  Branch assignment needs a typed value when the
    opposite branch is typed, so literals are materialized here before
    ``br.assign``; all runtime operands pass through unchanged.
    """
    if isinstance(value, bool):
        return coerce_runtime_i1_value(value, context=context)
    return value


def _short_circuit_and(lhs, rhs_fn):
    """Internal device-side ``lhs and rhs()`` with Python short-circuit semantics.

    Emits a result-bearing ``scf.if``: when ``lhs`` is truthy the RHS lambda is
    traced inside the ``then`` region, otherwise the unmodified ``lhs`` operand
    is the result.  Non-runtime operands (plain Python values, including static
    floats) keep native Python truthiness and short-circuit at trace time
    without tracing the RHS.  Runtime left operands use non-zero truthiness for
    the control condition; the branch merge keeps Python operand semantics, so
    an ``i1`` next to an integer-like value widens to that integer's 0/1 value
    instead of collapsing the integer to a boolean.  Incompatible branch types
    keep their existing diagnostics.
    """
    if not callable(rhs_fn):
        raise TypeError("pto._short_circuit_and expects a zero-argument callable RHS")
    if not hasattr(lhs, "type"):
        return rhs_fn() if lhs else lhs
    raw_lhs = unwrap_surface_value(lhs)
    cond = coerce_runtime_i1_value(raw_lhs, context="pto short-circuit and condition")
    with _IfCM(cond, short_circuit_merge=True) as br:
        with br.then_:
            br.assign(value=_materialize_bool_branch_value(
                _branch_rhs_value(rhs_fn(), "and"), context="pto short-circuit and RHS"))
        with br.else_:
            br.assign(value=raw_lhs)
    return br.value


def _short_circuit_or(lhs, rhs_fn):
    """Internal device-side ``lhs or rhs()`` with Python short-circuit semantics.

    When ``lhs`` is truthy the unmodified ``lhs`` operand is the result and the
    RHS lambda is never traced; otherwise the RHS runs inside the ``else``
    region.  Non-runtime operands (plain Python values, including static
    floats) keep native Python truthiness and short-circuit at trace time.
    Runtime left operands use non-zero truthiness for the control condition;
    the branch merge keeps Python operand semantics, so an ``i1`` next to an
    integer-like value widens to that integer's 0/1 value instead of
    collapsing the integer to a boolean.  Incompatible branch types keep their
    existing diagnostics.
    """
    if not callable(rhs_fn):
        raise TypeError("pto._short_circuit_or expects a zero-argument callable RHS")
    if not hasattr(lhs, "type"):
        return lhs if lhs else rhs_fn()
    raw_lhs = unwrap_surface_value(lhs)
    cond = coerce_runtime_i1_value(raw_lhs, context="pto short-circuit or condition")
    with _IfCM(cond, short_circuit_merge=True) as br:
        with br.then_:
            br.assign(value=raw_lhs)
        with br.else_:
            br.assign(value=_materialize_bool_branch_value(
                _branch_rhs_value(rhs_fn(), "or"), context="pto short-circuit or RHS"))
    return br.value


def _is_branch_assign_literal(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_i1_type(type_obj) -> bool:
    return IntegerType.isinstance(type_obj) and IntegerType(type_obj).width == 1


def _is_integer_like_type(type_obj) -> bool:
    return IndexType.isinstance(type_obj) or IntegerType.isinstance(type_obj)


def _coerce_integer_to_i1_at(value, *, block, context):
    with InsertionPoint(block):
        return coerce_runtime_integer_to_i1(value, context=context)


def _coerce_i1_to_integer_at(value, *, block, target_type, context):
    """Widen an i1 branch value to *target_type* inside *block*.

    Fixed-width integer targets use ``arith.extui``, which zero-extends the
    0/1 truth value.  ``index`` is not a valid extui result type and a direct
    ``arith.index_cast`` from ``i1`` sign-extends (``1`` becomes ``-1``), so
    the truth value is first zero-extended to ``i32`` and then index-cast,
    keeping it bit-exact.
    """
    with InsertionPoint(block):
        if IndexType.isinstance(target_type):
            i32 = IntegerType.get_signless(32)
            widened = coerce_integer_like(value, i32)
            return arith.IndexCastOp(IndexType.get(), widened).result
        return coerce_integer_like(value, target_type)


def _reconcile_branch_assignment_values(
    name,
    then_value,
    else_value,
    *,
    then_block=None,
    else_block=None,
    short_circuit_merge: bool = False,
):
    """Reconcile one named value across both branches of ``br.assign``.

    The default merge is predicate-oriented (``pto.if_`` semantics): an
    integer-like operand next to an ``i1`` is normalized to its non-zero i1.
    ``short_circuit_merge=True`` instead keeps Python operand semantics for
    ``and``/``or``: when one branch yields an ``i1`` and the other an
    integer-like value, the integer type wins and the ``i1`` side is widened
    to its 0/1 integer value.
    """
    then_is_typed = hasattr(then_value, "type")
    else_is_typed = hasattr(else_value, "type")

    if short_circuit_merge and (then_is_typed != else_is_typed):
        # A Python bool literal against a typed branch materializes to i1
        # first; the typed-vs-typed rules below then pick the result type.
        if then_is_typed and isinstance(else_value, bool):
            else_value = coerce_runtime_i1_value(
                else_value,
                context=f"br.assign(...) else branch value for '{name}'",
            )
            else_is_typed = True
        elif else_is_typed and isinstance(then_value, bool):
            then_value = coerce_runtime_i1_value(
                then_value,
                context=f"br.assign(...) then branch value for '{name}'",
            )
            then_is_typed = True
        elif then_is_typed and isinstance(else_value, int) and _is_i1_type(then_value.type):
            raise TypeError(
                "short-circuit merge cannot infer an integer width for the Python "
                f"literal {else_value!r} against an i1 branch value; materialize it "
                "explicitly with pto.const(..., dtype=...)",
            )
        elif else_is_typed and isinstance(then_value, int) and _is_i1_type(else_value.type):
            raise TypeError(
                "short-circuit merge cannot infer an integer width for the Python "
                f"literal {then_value!r} against an i1 branch value; materialize it "
                "explicitly with pto.const(..., dtype=...)",
            )

    if then_is_typed and else_is_typed:
        then_is_i1 = _is_i1_type(then_value.type)
        else_is_i1 = _is_i1_type(else_value.type)
        if then_is_i1 != else_is_i1:
            if then_is_i1 and _is_integer_like_type(else_value.type):
                if short_circuit_merge:
                    then_value = _coerce_i1_to_integer_at(
                        then_value,
                        block=then_block,
                        target_type=else_value.type,
                        context=f"br.assign(...) then branch value for '{name}'",
                    )
                else:
                    else_value = _coerce_integer_to_i1_at(
                        else_value,
                        block=else_block,
                        context=f"br.assign(...) else branch value for '{name}'",
                    )
            elif else_is_i1 and _is_integer_like_type(then_value.type):
                if short_circuit_merge:
                    else_value = _coerce_i1_to_integer_at(
                        else_value,
                        block=else_block,
                        target_type=then_value.type,
                        context=f"br.assign(...) else branch value for '{name}'",
                    )
                else:
                    then_value = _coerce_integer_to_i1_at(
                        then_value,
                        block=then_block,
                        context=f"br.assign(...) then branch value for '{name}'",
                    )
        return then_value, else_value
    if then_is_typed:
        return then_value, coerce_scalar_to_type(
            else_value,
            then_value.type,
            context=f"br.assign(...) else branch value for '{name}'",
        )
    if else_is_typed:
        return coerce_scalar_to_type(
            then_value,
            else_value.type,
            context=f"br.assign(...) then branch value for '{name}'",
        ), else_value

    raise TypeError(
        "br.assign(...) cannot infer a PTO type when both branches provide only "
        f"Python literals for '{name}'; materialize one side explicitly with pto.const(...)"
    )


# ── yield_ ────────────────────────────────────────────────────────────────────

def yield_(*vals):
    """Emit ``scf.yield`` with the given values."""
    scf.YieldOp([unwrap_surface_value(value) for value in vals])


__all__ = [
    "section", "vecscope", "static_range", "range", "const_expr", "LoopHandle", "BranchHandle",
    "for_", "while_", "_while", "if_", "yield_",
]
