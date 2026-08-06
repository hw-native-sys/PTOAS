# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Central candidate-legality evaluation for TileLib version selection.

Hard metadata (dtype signatures, layouts, and memory spaces) and custom
``constraints=[predicate, ...]`` are evaluated here. The registry only
discovers candidates, delegates legality to this module, and ranks the legal
results.

Custom predicates are called by **name-matching their parameters** against the
per-operand context — the same introspection convention as tilelang-dsl's
``_evaluate_constraints``. A predicate receives keys like ``src_shape`` /
``dst_valid_shape`` / ``src_config`` and returns a truthy value when legal.

``BLayout`` / ``SLayout`` mirror tilelang's enums so a copied predicate's
``cfg.b_layout != pto.BLayout.ROW_MAJOR`` comparison works unchanged (str enums compare equal
to the raw layout strings carried in operand specs).
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from enum import Enum

from .._types import _normalize_compact_mode
from .metadata import ScalarSpec, VectorSpec, ViewSpec


class BLayout(str, Enum):
    ROW_MAJOR = "row_major"
    COL_MAJOR = "col_major"


class SLayout(str, Enum):
    NONE_BOX = "none_box"
    ROW_MAJOR = "row_major"
    COL_MAJOR = "col_major"


@dataclass(frozen=True)
class CandidateLegality:
    """Result of evaluating one candidate against concrete operands."""

    legal: bool
    reason: str | None = None


@dataclass(frozen=True)
class _ConfigView:
    """The ``{name}_config`` object exposed to constraint predicates."""

    b_layout: str
    s_layout: str
    s_fractal_size: int | None
    compact_mode: str | int | None


def build_context(tile_specs: dict, target: str, op: str) -> dict:
    """Build the flat name-keyed context predicates are matched against."""
    context: dict = {"target": target, "op": op}
    operand_dtypes = []
    operand_kinds = []
    operand_memory_spaces = []
    operand_rows = []
    operand_cols = []
    operand_sizes = []
    operand_valid_rows = []
    operand_valid_cols = []
    operand_b_layouts = []
    operand_s_layouts = []
    operand_s_fractal_sizes = []
    operand_compact_modes = []
    for name, spec in tile_specs.items():
        dtype = spec.dtype.name
        operand_dtypes.append(dtype)
        context[f"{name}_dtype"] = dtype

        if isinstance(spec, ScalarSpec):
            operand_kinds.append("scalar")
            context[f"{name}_kind"] = "scalar"
            if hasattr(spec, "value"):
                context[f"{name}_value"] = spec.value
            continue

        if isinstance(spec, VectorSpec):
            operand_kinds.append("vector")
            shape = tuple(spec.shape)
            operand_sizes.append(_shape_size(shape))
            context[f"{name}_kind"] = "vector"
            context[f"{name}_shape"] = shape
            context[f"{name}_size"] = _shape_size(shape)
            continue

        if isinstance(spec, ViewSpec):
            operand_kinds.append("view")
            shape = tuple(spec.shape)
            memory_space = getattr(spec, "memory_space", "gm")
            operand_memory_spaces.append(memory_space)
            if _is_static_shape(shape):
                operand_sizes.append(_shape_size(shape))
            context[f"{name}_kind"] = "view"
            context[f"{name}_shape"] = shape
            context[f"{name}_strides"] = tuple(spec.strides) if spec.strides else None
            context[f"{name}_memory_space"] = memory_space
            context[f"{name}_layout"] = spec.layout
            if len(shape) == 2:
                context[f"{name}_rows"], context[f"{name}_cols"] = shape
                if all(isinstance(dim, int) for dim in shape):
                    operand_rows.append(shape[0])
                    operand_cols.append(shape[1])
            continue

        if not hasattr(spec, "shape"):
            operand_kinds.append(type(spec).__name__)
            context[f"{name}_kind"] = type(spec).__name__
            continue

        operand_kinds.append("tile")
        shape = tuple(spec.shape)
        valid = tuple(spec.valid_shape) if getattr(spec, "valid_shape", None) else shape
        memory_space = getattr(spec, "memory_space", "ub")
        b_layout = getattr(spec, "b_layout", "row_major")
        s_layout = getattr(spec, "s_layout", "none_box")
        s_fractal_size = getattr(spec, "s_fractal_size", None)
        compact_mode = getattr(spec, "compact_mode", None)
        operand_memory_spaces.append(memory_space)
        operand_sizes.append(_shape_size(shape))
        operand_b_layouts.append(b_layout)
        operand_s_layouts.append(s_layout)
        operand_s_fractal_sizes.append(s_fractal_size)
        operand_compact_modes.append(compact_mode)
        context[f"{name}_kind"] = "tile"
        context[f"{name}_shape"] = shape
        context[f"{name}_valid_shape"] = valid
        context[f"{name}_memory_space"] = memory_space
        context[f"{name}_s_fractal_size"] = s_fractal_size
        context[f"{name}_compact_mode"] = compact_mode
        context[f"{name}_config"] = _ConfigView(
            b_layout=b_layout,
            s_layout=s_layout,
            s_fractal_size=s_fractal_size,
            compact_mode=compact_mode,
        )
        if len(shape) == 2:
            context[f"{name}_rows"], context[f"{name}_cols"] = shape
            if len(valid) == 2:
                (
                    context[f"{name}_valid_rows"],
                    context[f"{name}_valid_cols"],
                ) = valid
            operand_rows.append(shape[0])
            operand_cols.append(shape[1])
            if len(valid) == 2:
                operand_valid_rows.append(valid[0])
                operand_valid_cols.append(valid[1])
            else:
                operand_valid_rows.append(None)
                operand_valid_cols.append(None)
    context["operand_dtypes"] = tuple(operand_dtypes)
    context["operand_kinds"] = tuple(operand_kinds)
    context["operand_memory_spaces"] = tuple(operand_memory_spaces)
    context["operand_rows"] = tuple(operand_rows)
    context["operand_cols"] = tuple(operand_cols)
    context["operand_sizes"] = tuple(operand_sizes)
    context["operand_valid_rows"] = tuple(operand_valid_rows)
    context["operand_valid_cols"] = tuple(operand_valid_cols)
    context["operand_b_layouts"] = tuple(operand_b_layouts)
    context["operand_s_layouts"] = tuple(operand_s_layouts)
    context["operand_s_fractal_sizes"] = tuple(operand_s_fractal_sizes)
    context["operand_compact_modes"] = tuple(operand_compact_modes)
    return context


def _shape_size(shape):
    size = 1
    for dim in shape:
        if not isinstance(dim, int):
            return None
        size *= dim
    return size


def _is_static_shape(shape):
    return all(isinstance(dim, int) for dim in shape)


def evaluate_candidate(
    descriptor,
    tile_specs: dict,
    target: str,
    op: str,
    context_attrs: dict | None = None,
) -> CandidateLegality:
    """Evaluate every hard legality rule for one template descriptor."""
    if descriptor.target != target or descriptor.op != op:
        return CandidateLegality(
            False,
            f"candidate targets op={descriptor.op!r} target={descriptor.target!r}",
        )

    missing = [name for name in descriptor.param_names if name not in tile_specs]
    if missing:
        return CandidateLegality(
            False,
            f"missing operand specifications for {', '.join(missing)}",
        )
    extra = [name for name in tile_specs if name not in descriptor.param_names]
    if extra:
        return CandidateLegality(
            False,
            f"unexpected operand specifications for {', '.join(extra)}",
        )

    ordered_specs = {
        name: tile_specs[name]
        for name in descriptor.param_names
    }
    context = build_context(ordered_specs, target, op)

    metadata = descriptor.metadata
    dtype_signature = context["operand_dtypes"]
    if metadata.dtypes and dtype_signature not in metadata.dtypes:
        return CandidateLegality(
            False,
            f"dtype signature {dtype_signature} is not supported",
        )

    if not _metadata_values_match(
        metadata.layouts,
        context["operand_b_layouts"],
    ):
        return CandidateLegality(
            False,
            f"block layouts {context['operand_b_layouts']} do not match "
            f"{metadata.layouts}",
        )

    if not _metadata_values_match(
        metadata.memory_spaces,
        context["operand_memory_spaces"],
    ):
        return CandidateLegality(
            False,
            f"memory spaces {context['operand_memory_spaces']} do not match "
            f"{metadata.memory_spaces}",
        )

    if context_attrs:
        for name, value in context_attrs.items():
            context.setdefault(name, value)

    if not passes(metadata.constraints, context):
        return CandidateLegality(False, "custom constraints are not satisfied")

    return CandidateLegality(True)


def _metadata_values_match(expected, actual) -> bool:
    """Match one metadata value for all operands, or one value per operand."""
    expected = tuple(_enum_value(value) for value in expected)
    actual = tuple(_enum_value(value) for value in actual)
    if not expected:
        return True
    if len(expected) == 1:
        return all(value == expected[0] for value in actual)
    return len(expected) == len(actual) and expected == actual


def _enum_value(value):
    return getattr(value, "value", value)


def check_type(expected):
    expected = tuple(expected)

    def _check_type(operand_dtypes, **_):
        return tuple(operand_dtypes) == expected

    return _check_type


def check_memory_space(expected):
    def _check_memory_space(operand_memory_spaces, **_):
        return all(space == expected for space in operand_memory_spaces)

    return _check_memory_space


def check_layout(expected):
    def _check_layout(operand_b_layouts, **_):
        return all(layout == expected for layout in operand_b_layouts)

    return _check_layout


def check_s_layout(expected):
    def _check_s_layout(operand_s_layouts, **_):
        return all(layout == expected for layout in operand_s_layouts)

    return _check_s_layout


def require_same_valid_shape(*operand_names):
    def _require_same_valid_shape(**context):
        shapes = [context.get(f"{name}_valid_shape") for name in operand_names]
        return (
            bool(shapes)
            and None not in shapes
            and all(shape == shapes[0] for shape in shapes[1:])
        )

    return _require_same_valid_shape


def require_valid_rows(operand_name, rows):
    def _require_valid_rows(**context):
        shape = context.get(f"{operand_name}_valid_shape")
        return shape is not None and shape[0] == rows
    return _require_valid_rows


def require_contiguous(required=True):
    def _require_contiguous(operand_rows, operand_cols, operand_valid_cols, **_):
        if not required:
            return True
        if (
            len(operand_valid_cols) != len(operand_cols)
            or None in operand_valid_cols
        ):
            return False
        full_cols = all(
            valid == cols
            for valid, cols in zip(operand_valid_cols, operand_cols)
        )
        single_row = all(rows == 1 for rows in operand_rows)
        return full_cols or single_row

    return _require_contiguous


def require_elementwise_1d(*operand_names, memory_spaces=("ub", "vec")):
    """Require ordinary element-wise operands to describe one flat range.

    The rule is intentionally limited to rank-2 local tiles with a row-major,
    unboxed, gap-free physical layout. All named tiles must have the same
    static logical valid shape. A range is flattenable when every tile uses
    its full physical column axis, or when the logical range occupies only the
    first row. Predicate and conversion representations need family-specific
    constraints in addition to, or instead of, this ordinary rule.
    """

    allowed_memory_spaces = frozenset(memory_spaces)

    def _require_elementwise_1d(**context):
        if not operand_names:
            return False

        shapes = []
        valid_shapes = []
        for name in operand_names:
            if not _is_flat_local_tile(context, name, allowed_memory_spaces):
                return False

            shape = context.get(f"{name}_shape")
            valid_shape = context.get(f"{name}_valid_shape")
            shapes.append(shape)
            valid_shapes.append(valid_shape)

        if any(valid_shape != valid_shapes[0] for valid_shape in valid_shapes[1:]):
            return False

        full_columns = all(
            valid_shape[1] == shape[1]
            for shape, valid_shape in zip(shapes, valid_shapes)
        )
        single_logical_row = valid_shapes[0][0] == 1
        return full_columns or single_logical_row

    return _require_elementwise_1d


def require_conversion_1d(
    source_operand="src",
    destination_operand="dst",
    *,
    source_elements_per_destination=1,
    memory_spaces=("ub", "vec"),
):
    """Require two typed conversion streams to be independently flattenable.

    Conversion source and destination tiles need not have the same element
    width. Their typed pointers still advance over one common logical range,
    provided each tile is gap-free and the valid shapes obey the conversion's
    element-count relationship. ``source_elements_per_destination`` models
    packed forms such as A5 BF16-to-FP4, where one destination storage element
    represents two source elements.

    Multi-row ranges must fill the physical column axis of both tiles. A
    single logical row is also legal because neither stream crosses a row
    boundary. Unknown layout or compact-mode metadata rejects the candidate.
    """

    allowed_memory_spaces = frozenset(memory_spaces)
    ratio = source_elements_per_destination

    def _require_conversion_1d(**context):
        if not isinstance(ratio, int) or ratio <= 0:
            return False

        operands = (source_operand, destination_operand)
        shapes = []
        valid_shapes = []
        for name in operands:
            if not _is_flat_local_tile(context, name, allowed_memory_spaces):
                return False

            shape = context.get(f"{name}_shape")
            valid_shape = context.get(f"{name}_valid_shape")
            shapes.append(shape)
            valid_shapes.append(valid_shape)

        src_shape, dst_shape = shapes
        src_valid, dst_valid = valid_shapes
        if src_shape[0] != dst_shape[0] or src_valid[0] != dst_valid[0]:
            return False
        if (
            src_shape[1] != dst_shape[1] * ratio
            or src_valid[1] != dst_valid[1] * ratio
        ):
            return False

        full_columns = (
            src_valid[1] == src_shape[1]
            and dst_valid[1] == dst_shape[1]
        )
        return full_columns or dst_valid[0] == 1

    return _require_conversion_1d


_PREDICATE_PACKING_LAYOUTS = {
    # A5 stores one predicate bit per compared element. 32-bit comparisons
    # combine two 64-lane masks before a 16-byte PK store; 16-bit comparisons
    # use one 128-lane 16-byte PK store; and 8-bit comparisons use one
    # 256-lane 32-byte NORM store.
    "f32": (128, 16),
    "i32": (128, 16),
    "f16": (128, 16),
    "i16": (128, 16),
    "i8": (256, 32),
    "ui8": (256, 32),
}


def require_predicate_compare_1d(
    *data_operand_names,
    predicate_operand="dst",
    flattened_destination=False,
    memory_spaces=("ub", "vec"),
):
    """Require an A5 compare and its packed predicate output to be flattenable.

    Data operands must describe the same static contiguous logical range. The
    predicate destination has a different representation: one bit per source
    element, written in complete dtype-dependent predicate-store blocks.

    A single logical row is flattenable when its destination row has enough
    physical bytes for the rounded store. Multiple rows normally require every
    source row to end on a predicate-store boundary and the destination row
    stride to equal the exact bytes produced for one row.

    ``flattened_destination`` models operations such as ``tcmps`` whose 1D
    form writes the packed predicate into one continuous destination prefix.
    It accepts either an exact packed row stride or a destination row with one
    predicate container element per physical data element. The latter is an
    explicit logical-range capacity contract, not a packed-byte requirement;
    arbitrary intermediate predicate-row padding remains a 2D fallback.
    """

    allowed_memory_spaces = frozenset(memory_spaces)

    def _require_predicate_compare_1d(**context):
        if not data_operand_names:
            return False

        data_shapes = []
        data_valid_shapes = []
        for name in data_operand_names:
            if not _is_flat_local_tile(context, name, allowed_memory_spaces):
                return False
            shape = context.get(f"{name}_shape")
            valid_shape = context.get(f"{name}_valid_shape")
            data_shapes.append(shape)
            data_valid_shapes.append(valid_shape)

        if any(
            valid_shape != data_valid_shapes[0]
            for valid_shape in data_valid_shapes[1:]
        ):
            return False

        predicate_name = predicate_operand
        if not _is_flat_local_tile(
            context,
            predicate_name,
            allowed_memory_spaces,
        ):
            return False

        predicate_shape = context.get(f"{predicate_name}_shape")
        predicate_valid_shape = context.get(f"{predicate_name}_valid_shape")
        valid_rows, valid_cols = data_valid_shapes[0]
        if predicate_valid_shape[0] != valid_rows:
            return False

        dtype = context.get(f"{data_operand_names[0]}_dtype")
        store_layout = _PREDICATE_PACKING_LAYOUTS.get(dtype)
        if store_layout is None:
            return False
        elements_per_store, bytes_per_store = store_layout

        predicate_dtype = context.get(f"{predicate_name}_dtype")
        predicate_bytewidth = _dtype_bytewidth(predicate_dtype)
        if predicate_bytewidth is None:
            return False
        predicate_row_bytes = predicate_shape[1] * predicate_bytewidth
        if predicate_row_bytes % 32 != 0:
            return False
        row_store_count = _ceil_div(valid_cols, elements_per_store)
        required_row_bytes = row_store_count * bytes_per_store
        if predicate_row_bytes < required_row_bytes:
            return False

        single_logical_row = valid_rows == 1
        if single_logical_row:
            required_data_row_elements = row_store_count * elements_per_store
            return all(
                shape[1] >= required_data_row_elements
                for shape in data_shapes
            )

        data_rows_are_contiguous = all(
            valid_shape[1] == shape[1]
            for shape, valid_shape in zip(data_shapes, data_valid_shapes)
        )
        required_logical_row_bytes = (
            max(shape[1] for shape in data_shapes) * predicate_bytewidth
        )
        destination_holds_logical_range = (
            predicate_row_bytes >= required_logical_row_bytes
        )
        if flattened_destination and destination_holds_logical_range:
            total_elements = valid_rows * valid_cols
            total_store_count = _ceil_div(
                total_elements,
                elements_per_store,
            )
            required_total_bytes = total_store_count * bytes_per_store
            predicate_total_bytes = (
                predicate_shape[0] * predicate_row_bytes
            )
            return (
                data_rows_are_contiguous
                and predicate_total_bytes >= required_total_bytes
            )

        rows_end_on_store_boundary = valid_cols % elements_per_store == 0
        predicate_rows_are_contiguous = predicate_row_bytes == required_row_bytes
        return (
            data_rows_are_contiguous
            and rows_end_on_store_boundary
            and predicate_rows_are_contiguous
        )

    return _require_predicate_compare_1d


def require_predicate_select_1d(
    predicate_operand,
    *data_operand_names,
    temporary_operand=None,
    memory_spaces=("ub", "vec"),
):
    """Require A5 packed-predicate select operands to be flattenable.

    Data operands must describe one static contiguous logical range. The mask
    is a byte-addressed packed predicate whose row capacity and stride are
    checked using the selected data dtype. A5 does not access the ABI
    temporary, but its tile metadata must still be complete and supported.
    """

    allowed_memory_spaces = frozenset(memory_spaces)

    def _require_predicate_select_1d(**context):
        if not data_operand_names:
            return False

        data_shapes = []
        data_valid_shapes = []
        for name in data_operand_names:
            if not _is_flat_local_tile(context, name, allowed_memory_spaces):
                return False
            data_shapes.append(context.get(f"{name}_shape"))
            data_valid_shapes.append(context.get(f"{name}_valid_shape"))

        if any(
            valid_shape != data_valid_shapes[0]
            for valid_shape in data_valid_shapes[1:]
        ):
            return False

        if not _is_flat_local_tile(
            context,
            predicate_operand,
            allowed_memory_spaces,
        ):
            return False
        if temporary_operand is not None and not _is_flat_local_tile(
            context,
            temporary_operand,
            allowed_memory_spaces,
        ):
            return False

        predicate_shape = context.get(f"{predicate_operand}_shape")
        predicate_valid_shape = context.get(
            f"{predicate_operand}_valid_shape"
        )
        valid_rows, valid_cols = data_valid_shapes[0]
        if predicate_valid_shape[0] != valid_rows:
            return False

        dtype = context.get(f"{data_operand_names[0]}_dtype")
        store_layout = _PREDICATE_PACKING_LAYOUTS.get(dtype)
        if store_layout is None:
            return False
        elements_per_store, bytes_per_store = store_layout

        predicate_dtype = context.get(f"{predicate_operand}_dtype")
        predicate_bytewidth = _dtype_bytewidth(predicate_dtype)
        if predicate_bytewidth is None:
            return False
        predicate_row_bytes = predicate_shape[1] * predicate_bytewidth
        if predicate_row_bytes % 32 != 0:
            return False

        row_store_count = _ceil_div(valid_cols, elements_per_store)
        required_predicate_row_bytes = row_store_count * bytes_per_store
        if predicate_row_bytes < required_predicate_row_bytes:
            return False

        single_logical_row = valid_rows == 1
        if single_logical_row:
            required_data_row_elements = row_store_count * elements_per_store
            return all(
                shape[1] >= required_data_row_elements
                for shape in data_shapes
            )

        data_rows_are_contiguous = all(
            valid_shape[1] == shape[1]
            for shape, valid_shape in zip(data_shapes, data_valid_shapes)
        )
        rows_end_on_store_boundary = valid_cols % elements_per_store == 0
        predicate_rows_are_contiguous = (
            predicate_row_bytes == required_predicate_row_bytes
        )
        return (
            data_rows_are_contiguous
            and rows_end_on_store_boundary
            and predicate_rows_are_contiguous
        )

    return _require_predicate_select_1d


def _is_flat_local_tile(context, name, allowed_memory_spaces) -> bool:
    if context.get(f"{name}_kind") != "tile":
        return False
    shape = context.get(f"{name}_shape")
    valid_shape = context.get(f"{name}_valid_shape")
    if not _is_static_rank2_shape(shape) or not _is_static_rank2_shape(
        valid_shape
    ):
        return False
    if any(valid > physical for valid, physical in zip(valid_shape, shape)):
        return False
    if context.get(f"{name}_memory_space") not in allowed_memory_spaces:
        return False

    config = context.get(f"{name}_config")
    return (
        config is not None
        and _enum_value(config.b_layout) == BLayout.ROW_MAJOR.value
        and _enum_value(config.s_layout) == SLayout.NONE_BOX.value
        and _has_gap_free_row_stride(config.compact_mode)
    )


def _dtype_bytewidth(dtype) -> int | None:
    widths = {
        "i8": 1,
        "ui8": 1,
        "i16": 2,
        "ui16": 2,
        "f16": 2,
        "bf16": 2,
        "i32": 4,
        "ui32": 4,
        "f32": 4,
    }
    return widths.get(dtype)


def _ceil_div(value, divisor):
    return (value + divisor - 1) // divisor


def _is_static_rank2_shape(shape) -> bool:
    return (
        isinstance(shape, tuple)
        and len(shape) == 2
        and all(isinstance(dim, int) and dim > 0 for dim in shape)
    )


def _has_gap_free_row_stride(compact_mode) -> bool:
    return _normalize_compact_mode(_enum_value(compact_mode)) in {
        0,
        1,
        "Null",
        "Normal",
    }


def passes(predicates, context: dict) -> bool:
    """Return True iff every predicate is satisfied for *context* (legality filter)."""
    for predicate in predicates:
        try:
            signature = inspect.signature(predicate)
        except (TypeError, ValueError):
            return False
        kwargs: dict = {}
        for parameter in signature.parameters.values():
            if parameter.kind == inspect.Parameter.VAR_KEYWORD:
                for key, value in context.items():
                    kwargs.setdefault(key, value)
                continue
            if parameter.kind == inspect.Parameter.VAR_POSITIONAL:
                continue
            if parameter.name in context:
                kwargs[parameter.name] = context[parameter.name]
            elif parameter.default is not inspect.Parameter.empty:
                continue
            else:
                # A required parameter we can't supply -> treat as not satisfiable.
                return False
        try:
            if not predicate(**kwargs):
                return False
        except Exception:
            return False
    return True


__all__ = [
    "BLayout",
    "CandidateLegality",
    "SLayout",
    "build_context",
    "check_layout",
    "check_memory_space",
    "check_s_layout",
    "check_type",
    "evaluate_candidate",
    "passes",
    "require_contiguous",
    "require_conversion_1d",
    "require_elementwise_1d",
    "require_predicate_compare_1d",
    "require_predicate_select_1d",
    "require_same_valid_shape",
    "require_valid_rows",
]
