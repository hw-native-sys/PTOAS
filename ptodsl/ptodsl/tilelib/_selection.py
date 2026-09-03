# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""PTODSL TileLib candidate discovery, validation, and selection."""

from __future__ import annotations

from . import constraints as _constraints
from . import registry as _registry
from ._template_package import load_template
from .metadata import ScalarSpec, ScalarType, TileSpec, VectorSpec, ViewSpec


def _operand_config(spec, index):
    config = spec.get("config") or {}
    if not isinstance(config, dict):
        raise TypeError(f"operand_specs[{index}].config must be an object")
    return config


def _build_scalar_spec(name, spec, index):
    try:
        return ScalarSpec(dtype=ScalarType(spec["dtype"]), value=spec.get("value"))
    except KeyError as exc:
        raise ValueError(
            f"scalar operand {index} ({name!r}) is missing {exc.args[0]!r}"
        ) from exc


def _build_vector_spec(name, spec, index):
    try:
        return VectorSpec(shape=tuple(spec["shape"]), dtype=ScalarType(spec["dtype"]))
    except KeyError as exc:
        raise ValueError(
            f"vector operand {index} ({name!r}) is missing {exc.args[0]!r}"
        ) from exc


def _build_view_spec(name, spec, index):
    config = _operand_config(spec, index)
    try:
        strides = spec.get("strides")
        return ViewSpec(
            shape=tuple(spec["shape"]),
            dtype=ScalarType(spec["dtype"]),
            memory_space=spec.get("memory_space", "gm"),
            strides=tuple(strides) if strides is not None else None,
            layout=config.get("layout"),
        )
    except KeyError as exc:
        raise ValueError(
            f"view operand {index} ({name!r}) is missing {exc.args[0]!r}"
        ) from exc


def _build_tile_spec(name, spec, index):
    config = _operand_config(spec, index)
    try:
        shape = tuple(spec["shape"])
        dtype = ScalarType(spec["dtype"])
    except KeyError as exc:
        raise ValueError(
            f"tile operand {index} ({name!r}) is missing {exc.args[0]!r}"
        ) from exc
    valid_shape = spec.get("valid_shape")
    s_fractal_size = config.get("s_fractal_size", 512)
    if s_fractal_size == 0:
        s_fractal_size = 512
    return TileSpec(
        shape=shape,
        dtype=dtype,
        memory_space=spec.get("memory_space", "ub"),
        valid_shape=tuple(valid_shape) if valid_shape is not None else None,
        b_layout=config.get("b_layout", "row_major"),
        s_layout=config.get("s_layout", "none_box"),
        s_fractal_size=s_fractal_size,
        pad_value=spec.get("pad_value", config.get("pad_value", "Null")),
        compact_mode=config.get("compact_mode", "null"),
    )


def _build_tile_specs(descriptor, operand_specs: list) -> dict:
    """Map positional compiler operands onto a template's parameter names."""
    if not isinstance(operand_specs, list):
        raise TypeError("operand_specs must be a list")
    if len(operand_specs) != len(descriptor.param_names):
        raise ValueError(
            f"template {descriptor.name!r} expects {len(descriptor.param_names)} "
            f"operands, got {len(operand_specs)}"
        )

    specs = {}
    for index, (name, spec) in enumerate(zip(descriptor.param_names, operand_specs)):
        if not isinstance(spec, dict):
            raise TypeError(f"operand_specs[{index}] must be an object")

        kind = spec.get("kind")
        if kind == "scalar":
            specs[name] = _build_scalar_spec(name, spec, index)
            continue
        if kind == "vector":
            specs[name] = _build_vector_spec(name, spec, index)
            continue
        if kind == "view":
            specs[name] = _build_view_spec(name, spec, index)
            continue

        if kind != "tile":
            raise NotImplementedError(
                "PTODSL TileLib currently supports tile, scalar, view, "
                f"and vector operands; "
                f"operand {index} ({name!r}) has kind {kind!r}"
            )

        specs[name] = _build_tile_spec(name, spec, index)
    return specs


def _constraint_name(predicate) -> str:
    return getattr(predicate, "__name__", repr(predicate))


def _metadata_value(value):
    if callable(value):
        return {"callable": _constraint_name(value)}
    return value


def _metadata_for_descriptor(descriptor, constraint_context: dict) -> dict:
    metadata = descriptor.metadata
    if callable(metadata.Tail):
        has_tail = _constraints.passes((metadata.Tail,), constraint_context)
    else:
        has_tail = bool(metadata.Tail)
    return {
        "op": metadata.op,
        "target": metadata.target,
        "name": metadata.name,
        "dtypes": [list(signature) for signature in metadata.dtypes],
        "layouts": list(metadata.layouts),
        "memory_spaces": list(metadata.memory_spaces),
        "constraints": [
            _constraint_name(predicate) for predicate in metadata.constraints
        ],
        "priority": metadata.priority,
        "fusible": metadata.fusible,
        "loop_depth": metadata.loop_depth,
        "id": metadata.id,
        "Tail": _metadata_value(metadata.Tail),
        "has_tail": has_tail,
        "is_post_update": metadata.is_post_update,
        "iteration_axis": metadata.iteration_axis,
        "op_engine": metadata.op_engine,
        "op_class": metadata.op_class,
        "tags": list(metadata.tags),
    }


def _registered_candidates(target: str, op: str) -> list:
    # Import only this op's template module. Registration happens as an import
    # side effect and repeated requests are no-ops because the loader is cached.
    load_template(op, target)
    candidates = _registry.default_registry().lookup(op, target)
    if not candidates:
        raise _registry.NoMatchingTemplate(
            f"no template registered for op={op!r} target={target!r}"
        )
    return candidates


def _legal_candidate_specs(
    target: str,
    op: str,
    operand_specs: list,
    context_attrs: dict | None = None,
) -> list:
    """Return legal ``(descriptor, specs)`` pairs for this concrete request.

    Different template versions may have different parameter counts/order.  The
    wire operands are positional, so bind them against each descriptor before
    asking the Python constraint legalizer.
    """
    evaluated = []
    for descriptor in _registered_candidates(target, op):
        try:
            specs = _build_tile_specs(descriptor, operand_specs)
        except Exception as exc:
            evaluated.append((descriptor, None, f"operand binding failed: {exc}"))
            continue

        legality = _constraints.evaluate_candidate(
            descriptor,
            specs,
            target,
            op,
            context_attrs,
        )
        evaluated.append(
            (
                descriptor,
                specs,
                legality.reason if not legality.legal else None,
            )
        )

    legal = [
        (descriptor, specs)
        for descriptor, specs, reason in evaluated
        if specs is not None and reason is None
    ]
    if not legal:
        reasons = "; ".join(
            f"{descriptor.name}: {reason}"
            for descriptor, _, reason in evaluated
        )
        raise _registry.NoMatchingTemplate(
            f"no legal template for op={op!r} target={target!r}; {reasons}"
        )

    legal.sort(key=lambda pair: _registry.candidate_sort_key(pair[0]))
    return legal


def _require_unambiguous_top_candidate(target: str, op: str, legal: list) -> None:
    if len(legal) < 2:
        return

    top_priority = legal[0][0].metadata.priority
    winners = [
        descriptor
        for descriptor, _ in legal
        if descriptor.metadata.priority == top_priority
    ]
    if len(winners) > 1:
        names = ", ".join(descriptor.name for descriptor in winners)
        raise _registry.AmbiguousTemplate(
            f"multiple templates tie at priority {top_priority} for op={op!r} "
            f"target={target!r}: {names}; assign distinct priorities or make "
            "their constraints mutually exclusive"
        )


def _select_descriptor_and_specs(
    target: str,
    op: str,
    operand_specs: list,
    context_attrs: dict | None = None,
    candidate_id: str | None = None,
):
    legal = _legal_candidate_specs(target, op, operand_specs, context_attrs)
    if candidate_id:
        for descriptor, specs in legal:
            if descriptor.name == candidate_id:
                return descriptor, specs
        legal_names = ", ".join(descriptor.name for descriptor, _ in legal)
        raise _registry.NoMatchingTemplate(
            f"candidate {candidate_id!r} is not a legal template for op={op!r} "
            f"target={target!r}; legal candidates: {legal_names}"
        )

    if len(legal) == 1:
        return legal[0]

    _require_unambiguous_top_candidate(target, op, legal)
    return legal[0]


def metadata_request(
    target: str,
    op: str,
    operand_specs: list,
    context_attrs: dict | None = None,
) -> dict:
    """Return every legal candidate in deterministic selection order."""
    legal = _legal_candidate_specs(target, op, operand_specs, context_attrs)
    _require_unambiguous_top_candidate(target, op, legal)
    return {
        "target": target,
        "op": op,
        "candidates": [
            _metadata_for_descriptor(
                descriptor,
                {
                    **_constraints.build_context(specs, target, op),
                    **(context_attrs or {}),
                },
            )
            for descriptor, specs in legal
        ],
    }


__all__ = ["metadata_request"]
