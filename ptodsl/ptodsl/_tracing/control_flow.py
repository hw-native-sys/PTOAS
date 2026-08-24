# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Tracing-time helpers for structured PTODSL control-flow lowering."""

from __future__ import annotations

from dataclasses import dataclass

from .._runtime_index_ops import coerce_runtime_index
from .._surface_values import unwrap_surface_value
from .._types import _is_struct_type

from ptoas.mlir.dialects import arith
from ptoas.mlir.dialects import scf
from ptoas.mlir.ir import InsertionPoint, IntegerAttr, IntegerType, StringAttr


# ── loop-unroll hints ─────────────────────────────────────────────────────────

_UNROLL_HINT_VALUES = ("full",)

# pto.unroll_factor is encoded as a signless i32 attribute and read back as a
# signed value by the backend, so the factor must fit in [1, INT32_MAX].
_MAX_UNROLL_FACTOR = 2**31 - 1


def normalize_unroll_hint(unroll, unroll_factor, *, context="pto.for_(...)"):
    """Validate one loop-unroll hint pair and return it unchanged.

    ``unroll`` must be "full" (the only supported value); ``unroll_factor``
    must be a positive Python int that fits the signless i32 attribute
    encoding (<= 2**31 - 1).  The two are mutually exclusive.
    """
    if unroll is not None:
        if not isinstance(unroll, str) or unroll not in _UNROLL_HINT_VALUES:
            raise ValueError(
                f"{context} unroll= expects one of {_UNROLL_HINT_VALUES}, got {unroll!r}"
            )
    if unroll_factor is not None:
        if unroll is not None:
            raise ValueError(
                f"{context} unroll= and unroll_factor= are mutually exclusive"
            )
        if isinstance(unroll_factor, bool) or not isinstance(unroll_factor, int):
            raise TypeError(
                f"{context} unroll_factor= expects a positive Python int, got "
                f"{type(unroll_factor).__name__}; dynamic factors are not supported"
            )
        if unroll_factor < 1:
            raise ValueError(
                f"{context} unroll_factor= must be a positive integer, got {unroll_factor}"
            )
        if unroll_factor > _MAX_UNROLL_FACTOR:
            raise ValueError(
                f"{context} unroll_factor= must fit a signless i32 attribute "
                f"(<= {_MAX_UNROLL_FACTOR}), got {unroll_factor}"
            )
    return unroll, unroll_factor


def apply_unroll_hint(for_op, unroll, unroll_factor):
    """Attach a validated unroll hint to one ``scf.for`` as discardable attrs."""
    if unroll is not None:
        for_op.operation.attributes["pto.unroll"] = StringAttr.get(unroll)
    if unroll_factor is not None:
        for_op.operation.attributes["pto.unroll_factor"] = IntegerAttr.get(
            IntegerType.get_signless(32), unroll_factor
        )


@dataclass
class CarryLoopFrame:
    """Active loop-carry lowering frame for one authored ``pto.for_().carry()``."""

    for_op: object
    insertion_point: InsertionPoint
    state_names: tuple[str, ...]
    state_templates: tuple[object, ...]
    yielded: bool = False


def build_carry_loop_frame(start, stop, step, state_items, *, unroll=None, unroll_factor=None) -> CarryLoopFrame:
    """Materialize one ``scf.for`` carry loop and enter its body insertion point."""
    state_items = tuple(state_items)
    state_names = tuple(name for name, _ in state_items)
    state_templates = tuple(value for _, value in state_items)
    iter_args = [_materialize_carry_init(value) for value in state_templates]
    for_op = scf.ForOp(
        _coerce_index(start),
        _coerce_index(stop),
        _coerce_index(step),
        iter_args,
    )
    apply_unroll_hint(for_op, unroll, unroll_factor)
    insertion_point = InsertionPoint(for_op.body)
    insertion_point.__enter__()
    return CarryLoopFrame(
        for_op=for_op,
        insertion_point=insertion_point,
        state_names=state_names,
        state_templates=state_templates,
    )


def yield_carry_loop_state(frame: CarryLoopFrame, **kwargs) -> None:
    """Validate one ``loop.update(...)`` call and emit the matching ``scf.yield``."""
    missing = [name for name in frame.state_names if name not in kwargs]
    extra = [name for name in kwargs if name not in frame.state_names]
    if missing or extra:
        pieces = []
        if missing:
            pieces.append(f"missing: {', '.join(missing)}")
        if extra:
            pieces.append(f"unexpected: {', '.join(extra)}")
        raise RuntimeError("loop.update(...) must match carry names exactly; " + "; ".join(pieces))
    if frame.yielded:
        raise RuntimeError("loop.update(...) may only be called once per loop body")
    scf.YieldOp([unwrap_surface_value(kwargs[name]) for name in frame.state_names])
    frame.yielded = True


def finish_carry_loop_frame(frame: CarryLoopFrame, exc_type, exc, tb) -> None:
    """Leave one active carry-loop frame and close its insertion point."""
    try:
        if exc_type is None and not frame.yielded:
            raise RuntimeError(
                "pto.for_(...).carry(...) requires loop.update(...) before leaving the loop body"
            )
    finally:
        frame.insertion_point.__exit__(exc_type, exc, tb)


def _coerce_index(value):
    raw_value = unwrap_surface_value(value)
    return coerce_runtime_index(raw_value, context="pto.for_(...).carry(...) loop bound")


def _materialize_carry_init(value):
    raw_value = unwrap_surface_value(value)
    if _is_struct_type(getattr(raw_value, "type", None)):
        raise TypeError(
            "pto.for_(...).carry(...) does not accept stack-local struct values; "
            "declare the struct outside the loop and mutate it in place inside the loop body"
        )
    if isinstance(raw_value, bool):
        raise TypeError("pto.for_(...).carry(...) does not accept bool loop-carried values")
    if isinstance(raw_value, int):
        return arith.ConstantOp(IntegerType.get_signless(32), raw_value).result
    return raw_value


__all__ = [
    "CarryLoopFrame",
    "apply_unroll_hint",
    "build_carry_loop_frame",
    "normalize_unroll_hint",
    "yield_carry_loop_state",
    "finish_carry_loop_frame",
]
