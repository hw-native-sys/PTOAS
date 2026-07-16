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
``vecscope()``            – ``pto.vecscope { … }``
``for_(lo, hi, step)``
                          – ``scf.for`` with optional named carry state via ``.carry(...)``
``if_(cond)``             – ``scf.if`` via explicit branch handle + automatic named merge
``yield_(*vals)``         – ``scf.yield``
``static_range(...)``     – trace-time ``range(...)`` escape hatch for AST rewrite
``const_expr(value)``     – trace-time ``if`` escape hatch for AST rewrite
"""

from ._bootstrap import make_context  # noqa: F401
from ._runtime_index_ops import coerce_runtime_index
from ._scalar_coercion import coerce_scalar_to_type
from ._surface_types import const_expr
from ._tracing.active import current_session
from ._surface_values import unwrap_surface_value, wrap_like_surface_value, wrap_surface_value

from mlir.dialects import pto as _pto, scf
from mlir.ir import InsertionPoint


# ── vecscope ──────────────────────────────────────────────────────────────────

class _VecScopeCM:
    """Context manager for ``pto.vecscope { … }``."""

    def __enter__(self):
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


# ── compile-time control-flow helpers ─────────────────────────────────────────

def static_range(*args):
    """Return ``range(*args)`` for trace-time unrolling under AST rewrite."""
    return range(*args)

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
    def __init__(self, start, stop, step, iter_args):
        self._start = start
        self._stop = stop
        self._step = step
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
        self._ip = InsertionPoint(self._for_op.body)
        self._ip.__enter__()
        if not self._iter_args:
            return wrap_surface_value(self._for_op.induction_variable)
        return LoopHandle(self._for_op, iter_arg_templates=self._iter_arg_templates)

    def __exit__(self, *exc):
        if not self._iter_args:
            scf.YieldOp([])
        self._ip.__exit__(*exc)


def for_(start, stop, *, step):
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
    """
    return _ForBuilder(start, stop, step)


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
    def __init__(self, start, stop, step, state_items):
        self._state_items = tuple(state_items)
        self._state_names = tuple(name for name, _ in self._state_items)
        self._state_templates = tuple(value for _, value in self._state_items)
        self._session = None
        self._session_frame = None
        super().__init__(start, stop, step, self._state_templates)
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
    def __init__(self, start, stop, step):
        self._start = start
        self._stop = stop
        self._step = step

    def __enter__(self):
        self._cm = _ForCM(self._start, self._stop, self._step, None)
        return self._cm.__enter__()

    def __exit__(self, *exc):
        return self._cm.__exit__(*exc)

    def carry(self, **kwargs):
        if not kwargs:
            raise ValueError("carry(...) requires at least one named loop-carried value")
        for name in kwargs:
            if not isinstance(name, str) or not name:
                raise TypeError("carry(...) names must be non-empty strings")
        return _CarryForCM(self._start, self._stop, self._step, tuple(kwargs.items()))


def _coerce_index(value):
    raw_value = unwrap_surface_value(value)
    return coerce_runtime_index(raw_value, context="pto.for_(...) loop bound")


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

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return self._owner._get_merged_value(name)


class _IfCM:
    def __init__(self, cond):
        self._cond = cond
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

        then_names = set(then_assignment["raw_values"].keys())
        else_names = set(else_assignment["raw_values"].keys())
        if then_names != else_names:
            missing_in_else = sorted(then_names - else_names)
            missing_in_then = sorted(else_names - then_names)
            pieces = []
            if missing_in_else:
                pieces.append(f"missing in else: {', '.join(missing_in_else)}")
            if missing_in_then:
                pieces.append(f"missing in then: {', '.join(missing_in_then)}")
            raise RuntimeError("br.assign(...) names must match across branches; " + "; ".join(pieces))

        order = then_assignment["order"]
        result_types = []
        result_templates = {}
        for name in order:
            then_value = then_assignment["raw_values"][name]
            else_value = else_assignment["raw_values"][name]
            then_type = getattr(then_value, "type", None)
            else_type = getattr(else_value, "type", None)
            then_template = then_assignment["templates"][name]

            if then_type is None and else_type is None:
                raise TypeError(
                    "br.assign(...) requires at least one PTO runtime value to infer the result type; "
                    f"'{name}' received {type(then_value).__name__} and {type(else_value).__name__}"
                )
            if then_type is None:
                then_value = self._materialize_branch_literal(
                    "then",
                    name,
                    then_value,
                    else_type,
                )
                then_assignment["raw_values"][name] = then_value
                then_type = then_value.type
                then_template = else_assignment["templates"][name]
            if else_type is None:
                else_value = self._materialize_branch_literal(
                    "else",
                    name,
                    else_value,
                    then_type,
                )
                else_assignment["raw_values"][name] = else_value
                else_type = else_value.type
            if then_value.type != else_value.type:
                raise RuntimeError(
                    f"br.assign(...) type mismatch for '{name}': "
                    f"then branch yields {then_value.type}, else branch yields {else_value.type}"
                )
            result_types.append(then_value.type)
            result_templates[name] = then_template

        return {
            "order": order,
            "result_types": result_types,
            "result_templates": result_templates,
            "then": then_assignment,
            "else": else_assignment,
        }

    def _materialize_branch_literal(self, branch_name, name, value, target_type):
        block = self._tmp_if.then_block if branch_name == "then" else self._tmp_if.else_block
        with InsertionPoint(block):
            return coerce_scalar_to_type(
                value,
                target_type,
                context=f"br.assign(...) value '{name}'",
            )

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
            (merge_spec["result_templates"][name] for name in merge_spec["order"]),
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


# ── yield_ ────────────────────────────────────────────────────────────────────

def yield_(*vals):
    """Emit ``scf.yield`` with the given values."""
    scf.YieldOp([unwrap_surface_value(value) for value in vals])


__all__ = [
    "vecscope", "static_range", "const_expr", "LoopHandle", "BranchHandle",
    "for_", "if_", "yield_",
]
