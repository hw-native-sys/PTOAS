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
from dataclasses import dataclass, field
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


@dataclass
class _ContextAccumulators:
    kinds: list = field(default_factory=list)
    memory_spaces: list = field(default_factory=list)
    rows: list = field(default_factory=list)
    cols: list = field(default_factory=list)
    sizes: list = field(default_factory=list)
    valid_rows: list = field(default_factory=list)
    valid_cols: list = field(default_factory=list)
    b_layouts: list = field(default_factory=list)
    s_layouts: list = field(default_factory=list)
    s_fractal_sizes: list = field(default_factory=list)
    compact_modes: list = field(default_factory=list)


@dataclass(frozen=True)
class _PredicateStorage:
    shape: tuple
    row_bytes: int
    elements_per_store: int
    bytes_per_store: int


def _collect_flat_data_context(context, names, allowed_memory_spaces):
    if not names:
        return None
    shapes = []
    valid_shapes = []
    for name in names:
        if not _is_flat_local_tile(context, name, allowed_memory_spaces):
            return None
        shapes.append(context.get(f"{name}_shape"))
        valid_shapes.append(context.get(f"{name}_valid_shape"))
    if any(valid != valid_shapes[0] for valid in valid_shapes[1:]):
        return None
    return shapes, valid_shapes


def _predicate_storage_context(
    context,
    name,
    allowed_memory_spaces,
    valid_rows,
    valid_cols,
    data_dtype,
):
    if not _is_flat_local_tile(context, name, allowed_memory_spaces):
        return None
    shape = context.get(f"{name}_shape")
    valid_shape = context.get(f"{name}_valid_shape")
    if valid_shape[0] != valid_rows:
        return None
    packing = _PREDICATE_PACKING_LAYOUTS.get(data_dtype)
    if packing is None:
        return None
    elements_per_store, bytes_per_store = packing
    bytewidth = _dtype_bytewidth(context.get(f"{name}_dtype"))
    if bytewidth is None:
        return None
    row_bytes = shape[1] * bytewidth
    if row_bytes % 32 != 0:
        return None
    row_store_count = _ceil_div(valid_cols, elements_per_store)
    required_row_bytes = row_store_count * bytes_per_store
    if row_bytes < required_row_bytes:
        return None
    return _PredicateStorage(shape, row_bytes, elements_per_store, bytes_per_store)


def _data_rows_contiguous(shapes, valid_shapes):
    return all(
        valid_shape[1] == shape[1]
        for shape, valid_shape in zip(shapes, valid_shapes)
    )


def _single_row_has_capacity(shapes, storage, valid_cols):
    required_elements = _ceil_div(valid_cols, storage.elements_per_store) * storage.elements_per_store
    return all(shape[1] >= required_elements for shape in shapes)


def _predicate_rows_are_flattenable(shapes, valid_shapes, storage, valid_cols):
    if not _data_rows_contiguous(shapes, valid_shapes):
        return False
    row_store_count = _ceil_div(valid_cols, storage.elements_per_store)
    required_row_bytes = row_store_count * storage.bytes_per_store
    return (
        valid_cols % storage.elements_per_store == 0
        and storage.row_bytes == required_row_bytes
    )


def _record_rank2_context(name, shape, valid, context, accumulators, *, static_only=False):
    if len(shape) != 2:
        return
    context[f"{name}_rows"], context[f"{name}_cols"] = shape
    if valid is not None and len(valid) == 2:
        context[f"{name}_valid_rows"], context[f"{name}_valid_cols"] = valid
    if static_only and not all(isinstance(dim, int) for dim in shape):
        return
    accumulators.rows.append(shape[0])
    accumulators.cols.append(shape[1])
    if valid is not None and len(valid) == 2:
        accumulators.valid_rows.append(valid[0])
        accumulators.valid_cols.append(valid[1])
    elif valid is not None:
        accumulators.valid_rows.append(None)
        accumulators.valid_cols.append(None)


def _record_scalar_context(name, spec, context, accumulators):
    accumulators.kinds.append("scalar")
    context[f"{name}_kind"] = "scalar"
    if hasattr(spec, "value"):
        context[f"{name}_value"] = spec.value


def _record_vector_context(name, spec, context, accumulators):
    accumulators.kinds.append("vector")
    shape = tuple(spec.shape)
    size = _shape_size(shape)
    accumulators.sizes.append(size)
    context[f"{name}_kind"] = "vector"
    context[f"{name}_shape"] = shape
    context[f"{name}_size"] = size


def _record_view_context(name, spec, context, accumulators):
    accumulators.kinds.append("view")
    shape = tuple(spec.shape)
    memory_space = getattr(spec, "memory_space", "gm")
    accumulators.memory_spaces.append(memory_space)
    if _is_static_shape(shape):
        accumulators.sizes.append(_shape_size(shape))
    context.update(
        {
            f"{name}_kind": "view",
            f"{name}_shape": shape,
            f"{name}_strides": tuple(spec.strides) if spec.strides else None,
            f"{name}_memory_space": memory_space,
            f"{name}_layout": spec.layout,
        }
    )
    _record_rank2_context(name, shape, None, context, accumulators, static_only=True)


def _record_tile_context(name, spec, context, accumulators):
    accumulators.kinds.append("tile")
    shape = tuple(spec.shape)
    valid = tuple(spec.valid_shape) if getattr(spec, "valid_shape", None) else shape
    memory_space = getattr(spec, "memory_space", "ub")
    b_layout = getattr(spec, "b_layout", "row_major")
    s_layout = getattr(spec, "s_layout", "none_box")
    s_fractal_size = getattr(spec, "s_fractal_size", None)
    compact_mode = getattr(spec, "compact_mode", None)
    accumulators.memory_spaces.append(memory_space)
    accumulators.sizes.append(_shape_size(shape))
    accumulators.b_layouts.append(b_layout)
    accumulators.s_layouts.append(s_layout)
    accumulators.s_fractal_sizes.append(s_fractal_size)
    accumulators.compact_modes.append(compact_mode)
    context.update(
        {
            f"{name}_kind": "tile",
            f"{name}_shape": shape,
            f"{name}_valid_shape": valid,
            f"{name}_memory_space": memory_space,
            f"{name}_s_fractal_size": s_fractal_size,
            f"{name}_compact_mode": compact_mode,
            f"{name}_config": _ConfigView(
                b_layout=b_layout,
                s_layout=s_layout,
                s_fractal_size=s_fractal_size,
                compact_mode=compact_mode,
            ),
        }
    )
    _record_rank2_context(name, shape, valid, context, accumulators)


def _record_operand_context(name, spec, context, accumulators):
    if isinstance(spec, ScalarSpec):
        return _record_scalar_context(name, spec, context, accumulators)
    if isinstance(spec, VectorSpec):
        return _record_vector_context(name, spec, context, accumulators)
    if isinstance(spec, ViewSpec):
        return _record_view_context(name, spec, context, accumulators)
    if not hasattr(spec, "shape"):
        accumulators.kinds.append(type(spec).__name__)
        context[f"{name}_kind"] = type(spec).__name__
        return
    _record_tile_context(name, spec, context, accumulators)


def build_context(tile_specs: dict, target: str, op: str) -> dict:
    """Build the flat name-keyed context predicates are matched against."""
    context: dict = {"target": target, "op": op}
    operand_dtypes = []
    accumulators = _ContextAccumulators()
    for name, spec in tile_specs.items():
        dtype = spec.dtype.name
        operand_dtypes.append(dtype)
        context[f"{name}_dtype"] = dtype
        _record_operand_context(name, spec, context, accumulators)
    context["operand_dtypes"] = tuple(operand_dtypes)
    for field_name, values in (
        ("operand_kinds", accumulators.kinds),
        ("operand_memory_spaces", accumulators.memory_spaces),
        ("operand_rows", accumulators.rows),
        ("operand_cols", accumulators.cols),
        ("operand_sizes", accumulators.sizes),
        ("operand_valid_rows", accumulators.valid_rows),
        ("operand_valid_cols", accumulators.valid_cols),
        ("operand_b_layouts", accumulators.b_layouts),
        ("operand_s_layouts", accumulators.s_layouts),
        ("operand_s_fractal_sizes", accumulators.s_fractal_sizes),
        ("operand_compact_modes", accumulators.compact_modes),
    ):
        context[field_name] = tuple(values)
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


def _candidate_shape_error(descriptor, tile_specs):
    missing = [name for name in descriptor.param_names if name not in tile_specs]
    if missing:
        return f"missing operand specifications for {', '.join(missing)}"
    extra = [name for name in tile_specs if name not in descriptor.param_names]
    if extra:
        return f"unexpected operand specifications for {', '.join(extra)}"
    return None


def _candidate_metadata_error(metadata, context):
    dtype_signature = context["operand_dtypes"]
    if metadata.dtypes and dtype_signature not in metadata.dtypes:
        return f"dtype signature {dtype_signature} is not supported"
    if not _metadata_values_match(metadata.layouts, context["operand_b_layouts"]):
        return (
            f"block layouts {context['operand_b_layouts']} do not match "
            f"{metadata.layouts}"
        )
    if not _metadata_values_match(metadata.memory_spaces, context["operand_memory_spaces"]):
        return (
            f"memory spaces {context['operand_memory_spaces']} do not match "
            f"{metadata.memory_spaces}"
        )
    return None


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

    shape_error = _candidate_shape_error(descriptor, tile_specs)
    if shape_error:
        return CandidateLegality(False, shape_error)

    ordered_specs = {
        name: tile_specs[name]
        for name in descriptor.param_names
    }
    context = build_context(ordered_specs, target, op)

    metadata = descriptor.metadata
    metadata_error = _candidate_metadata_error(metadata, context)
    if metadata_error:
        return CandidateLegality(False, metadata_error)

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
        data_context = _collect_flat_data_context(
            context,
            data_operand_names,
            allowed_memory_spaces,
        )
        if data_context is None:
            return False
        data_shapes, data_valid_shapes = data_context
        valid_rows, valid_cols = data_valid_shapes[0]
        storage = _predicate_storage_context(
            context,
            predicate_operand,
            allowed_memory_spaces,
            valid_rows,
            valid_cols,
            context.get(f"{data_operand_names[0]}_dtype"),
        )
        if storage is None:
            return False
        if valid_rows == 1:
            return _single_row_has_capacity(data_shapes, storage, valid_cols)
        data_rows_are_contiguous = _data_rows_contiguous(data_shapes, data_valid_shapes)
        destination_holds_logical_range = storage.row_bytes >= max(
            shape[1] for shape in data_shapes
        ) * _dtype_bytewidth(context.get(f"{predicate_operand}_dtype"))
        if flattened_destination and destination_holds_logical_range:
            total_elements = valid_rows * valid_cols
            total_store_count = _ceil_div(total_elements, storage.elements_per_store)
            required_total_bytes = total_store_count * storage.bytes_per_store
            predicate_total_bytes = storage.shape[0] * storage.row_bytes
            return (
                data_rows_are_contiguous
                and predicate_total_bytes >= required_total_bytes
            )
        return _predicate_rows_are_flattenable(
            data_shapes,
            data_valid_shapes,
            storage,
            valid_cols,
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
        data_context = _collect_flat_data_context(
            context,
            data_operand_names,
            allowed_memory_spaces,
        )
        if data_context is None:
            return False
        data_shapes, data_valid_shapes = data_context
        if temporary_operand is not None and not _is_flat_local_tile(context, temporary_operand, allowed_memory_spaces):
            return False
        valid_rows, valid_cols = data_valid_shapes[0]
        storage = _predicate_storage_context(
            context,
            predicate_operand,
            allowed_memory_spaces,
            valid_rows,
            valid_cols,
            context.get(f"{data_operand_names[0]}_dtype"),
        )
        if storage is None:
            return False
        if valid_rows == 1:
            return _single_row_has_capacity(data_shapes, storage, valid_cols)
        return _predicate_rows_are_flattenable(
            data_shapes,
            data_valid_shapes,
            storage,
            valid_cols,
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
