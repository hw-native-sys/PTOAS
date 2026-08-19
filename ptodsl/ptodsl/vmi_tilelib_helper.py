# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Instantiate a PTODSL VMI TileLib candidate for ``ExpandTileOp``."""

from __future__ import annotations

import argparse
import importlib
import json
import sys

from ._tile_template_tracing import (
    TileSpec,
    bf16,
    f16,
    f32,
    i8,
    i16,
    i32,
)
from .tilelib import registry as _tilelib_registry
from .tilelib import constraints as _tilelib_constraints
from .tilelib.metadata import ScalarSpec, ScalarType as MetadataScalarType
from .tilelib.registry import TileTemplateRegistry


_DTYPE_MAP = {
    "f32": f32,
    "f16": f16,
    "bf16": bf16,
    "i32": i32,
    "i16": i16,
    "i8": i8,
}


def _normalize_op_name(op_name: str) -> str:
    return op_name[4:] if op_name.startswith("pto.") else op_name


def _qualify_op_name(op_name: str) -> str:
    return op_name if op_name.startswith("pto.") else f"pto.{op_name}"


def _parse_operand_specs(spec_text: str) -> list[dict]:
    try:
        raw_specs = json.loads(spec_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid operand-specs JSON: {exc}") from exc
    if not isinstance(raw_specs, list) or not raw_specs:
        raise ValueError("operand-specs must be a non-empty JSON array")
    return raw_specs


def _parse_context_attrs(spec_text: str | None) -> dict[str, object]:
    if not spec_text:
        return {}
    try:
        attrs = json.loads(spec_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid context-attrs JSON: {exc}") from exc
    if not isinstance(attrs, dict):
        raise ValueError("context-attrs must be a JSON object")
    return attrs


def _parse_dtype(raw: dict, index: int):
    dtype_name = raw.get("dtype")
    dtype = _DTYPE_MAP.get(dtype_name)
    if dtype is None:
        raise ValueError(f"operand-specs[{index}] has unsupported dtype {dtype_name!r}")
    return dtype


def _parse_parameter_spec(raw: dict, index: int):
    if not isinstance(raw, dict):
        raise ValueError(f"operand-specs[{index}] must be an object")
    kind = raw.get("kind")
    if kind == "scalar":
        return _parse_dtype(raw, index)
    if kind != "tile":
        raise ValueError(
            f"operand-specs[{index}] must be a tile or scalar for the PTODSL VMI provider"
        )

    dtype = _parse_dtype(raw, index)
    shape = raw.get("shape")
    if not isinstance(shape, list) or len(shape) != 2:
        raise ValueError(f"operand-specs[{index}] requires a static rank-2 shape")
    try:
        parsed_shape = tuple(int(dim) for dim in shape)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"operand-specs[{index}] shape must contain integers") from exc

    valid_shape = raw.get("valid_shape")
    parsed_valid_shape = parsed_shape
    if valid_shape is not None:
        if not isinstance(valid_shape, list) or len(valid_shape) != 2:
            raise ValueError(f"operand-specs[{index}] valid_shape must be rank-2")
        if any(dim is None for dim in valid_shape):
            raise ValueError(
                "initial PTODSL VMI provider does not support dynamic valid_shape"
            )
        parsed_valid_shape = tuple(int(dim) for dim in valid_shape)
        # pto-isa invariant: ValidRow <= alignRow (physical). valid_shape may be
        # smaller than physical shape (e.g. RowPlusOne: valid=(128,64),
        # shape=(129,64)) — the +1 padding band lives only in UB, never GM.
        if parsed_valid_shape[0] > parsed_shape[0] or parsed_valid_shape[1] > parsed_shape[1]:
            raise ValueError(
                "initial PTODSL VMI provider requires valid_shape to not exceed "
                f"physical shape {parsed_shape}; operand-specs[{index}] has "
                f"valid_shape={list(parsed_valid_shape)}"
            )

    memory_space = raw.get("memory_space", "ub")
    if memory_space != "ub":
        raise ValueError(
            f"initial PTODSL VMI provider supports only UB tiles, got {memory_space!r}"
        )
    b_layout, compact_mode = _parse_tile_config(raw.get("config"), index)
    return TileSpec(
        parsed_shape, dtype, memory_space="ub", b_layout=b_layout,
        valid_shape=parsed_valid_shape, compact_mode=compact_mode,
    )


def _parse_legality_parameter_spec(raw: dict, index: int):
    parsed = _parse_parameter_spec(raw, index)
    if raw.get("kind") == "scalar":
        return ScalarSpec(MetadataScalarType(parsed.name), raw.get("value"))
    return parsed


def _parse_tile_config(config: object, index: int) -> tuple[str, str]:
    """Return ``(b_layout, compact_mode)`` from the operand config object.

    compact_mode: "normal" (default, includes null/0/1) or "row_plus_one" (2).
    """
    if config is None:
        return ("row_major", "normal")
    if not isinstance(config, dict):
        raise ValueError(f"operand-specs[{index}] config must be an object")
    allowed_s_layouts = {"none_box", "row_major"}
    s_layout = config.get("s_layout", "none_box")
    if s_layout not in allowed_s_layouts:
        raise ValueError(
            "initial PTODSL VMI provider supports only none_box or row_major "
            f"secondary layouts; operand-specs[{index}] has s_layout={s_layout!r}"
        )
    expected = {
        "s_fractal_size": 512,
        "pad_value": "0x0",
    }
    for key, expected_value in expected.items():
        value = config.get(key, expected_value)
        if key == "pad_value" and isinstance(value, str):
            value = value.lower()
        if value != expected_value:
            raise ValueError(
                "initial PTODSL VMI provider supports only the default secondary layout; "
                f"operand-specs[{index}] has {key}={config.get(key)!r}"
            )
    b_layout = config.get("b_layout", "row_major")
    if b_layout not in {"row_major", "col_major"}:
        raise ValueError(
            "initial PTODSL VMI provider supports row-major or col-major tiles; "
            f"operand-specs[{index}] has b_layout={b_layout!r}"
        )
    # compact_mode: ExpandTileOp emits it as an int (0/1=Normal, 2=RowPlusOne).
    compact_int = config.get("compact_mode", 1)
    try:
        compact_int = int(compact_int)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"operand-specs[{index}] compact_mode must be an int, got {compact_int!r}"
        ) from exc
    if compact_int in (0, 1):
        compact_mode = "normal"
    elif compact_int == 2:
        compact_mode = "row_plus_one"
    else:
        raise ValueError(
            f"operand-specs[{index}] compact_mode must be 0/1 (Normal) or 2 "
            f"(RowPlusOne), got {compact_int}"
        )
    return (b_layout, compact_mode)


def _is_vmi_candidate(descriptor) -> bool:
    if getattr(descriptor, "ir_level", None) == "vmi":
        return True
    metadata = getattr(descriptor, "metadata", None)
    return metadata is not None and "vmi" in getattr(metadata, "tags", ())


def _find_candidates(module, *, target: str, op_name: str) -> list:
    # Out-of-tree tests/providers may still expose only a module-local
    # VMI_TILELIB_REGISTRY.  Prefer that local registry so global built-in VMI
    # candidates do not hide provider-specific ambiguity.
    legacy_registry = getattr(module, "VMI_TILELIB_REGISTRY", None)
    if (
        module.__name__
        not in {"ptodsl.vmi_tilelib"}
        and isinstance(legacy_registry, TileTemplateRegistry)
    ):
        normalized_op = _normalize_op_name(op_name)
        return legacy_registry.lookup(normalized_op, target)

    # Importing the provider module registers its VMI descriptors into the
    # ordinary PTODSL TileLib registry.
    _ = module
    qualified_op = _qualify_op_name(op_name)
    candidates = [
        descriptor
        for descriptor in _tilelib_registry.default_registry().lookup(
            qualified_op, target
        )
        if _is_vmi_candidate(descriptor)
    ]
    if candidates:
        return candidates

    if isinstance(legacy_registry, TileTemplateRegistry):
        normalized_op = _normalize_op_name(op_name)
        return legacy_registry.lookup(normalized_op, target)
    return []


def instantiate_candidate(
    *,
    target: str,
    op_name: str,
    operand_specs: list[dict],
    provider_module: str,
    context_attrs: dict[str, object] | None = None,
):
    module = importlib.import_module(provider_module)
    qualified_op = _qualify_op_name(op_name)
    candidates = _find_candidates(module, target=target, op_name=qualified_op)
    if not candidates:
        raise LookupError(
            f"no PTODSL VMI candidate for target={target!r}, op={qualified_op!r} "
            f"in module {provider_module!r}"
        )

    legal = []
    rejected = []
    for candidate in candidates:
        parameters = tuple(candidate.param_names)
        if len(parameters) != len(operand_specs):
            rejected.append(
                f"{candidate.name}: expects {len(parameters)} operands, "
                f"got {len(operand_specs)}"
            )
            continue
        render_specs = {
            name: _parse_parameter_spec(raw_spec, index)
            for index, (name, raw_spec) in enumerate(zip(parameters, operand_specs))
        }
        legality_specs = {
            name: _parse_legality_parameter_spec(raw_spec, index)
            for index, (name, raw_spec) in enumerate(zip(parameters, operand_specs))
        }
        result = _tilelib_constraints.evaluate_candidate(
            candidate,
            legality_specs,
            target,
            candidate.op,
            context_attrs,
        )
        if result.legal:
            legal.append((candidate, render_specs))
        else:
            rejected.append(f"{candidate.name}: {result.reason}")

    if not legal:
        reasons = "; ".join(rejected)
        raise LookupError(
            f"no legal PTODSL VMI candidate for target={target!r}, "
            f"op={qualified_op!r} in module {provider_module!r}; {reasons}"
        )

    legal.sort(key=lambda item: item[0].metadata.priority, reverse=True)
    top_priority = legal[0][0].metadata.priority
    winners = [item for item in legal if item[0].metadata.priority == top_priority]
    if len(winners) != 1:
        names = ", ".join(candidate.name for candidate, _ in winners)
        raise LookupError(
            "RFC-mode PTODSL VMI provider requires exactly one canonical "
            f"candidate per (target, op); target={target!r}, op={qualified_op!r}, "
            f"found {len(winners)} legal top-priority candidates in module "
            f"{provider_module!r}: {names}"
        )

    candidate, parameter_specs = winners[0]
    return candidate.specialize(
        context_attrs=context_attrs or {},
        **parameter_specs,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PTODSL VMI TileLib expand helper")
    parser.add_argument("--target", default="a5")
    parser.add_argument("--op", required=True)
    parser.add_argument("--operand-specs", required=True)
    parser.add_argument("--context-attrs")
    parser.add_argument(
        "--provider-module", default="ptodsl.vmi_tilelib"
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Validate candidate availability without rendering MLIR",
    )
    args = parser.parse_args(argv)

    try:
        operand_specs = _parse_operand_specs(args.operand_specs)
        context_attrs = _parse_context_attrs(args.context_attrs)
        artifact = instantiate_candidate(
            target=args.target,
            op_name=args.op,
            operand_specs=operand_specs,
            provider_module=args.provider_module,
            context_attrs=context_attrs,
        )
        if args.metadata_only:
            sys.stdout.write(json.dumps({"candidate": artifact.descriptor.name}))
            return 0
        mlir_text = artifact.mlir_text()
    except Exception as exc:
        print(f"vmi_tilelib_helper: error: {exc}", file=sys.stderr)
        return 1

    sys.stdout.write(mlir_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
