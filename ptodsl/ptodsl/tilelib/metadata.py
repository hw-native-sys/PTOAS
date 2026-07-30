# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""TileLib template metadata + tile specialization specs.

``TemplateMetadata`` carries both the *hard constraints* used to decide whether a
template is legal for a concrete TileOp (op/target/dtypes/layouts/memory_spaces) and the
*selection hints* used to rank legal candidates (priority/fusible/tags).
"""

from __future__ import annotations

from dataclasses import dataclass

from .._types import (
    f4e1m2x2 as _f4e1m2x2,
    f4e2m1x2 as _f4e2m1x2,
    f8e4m3 as _f8e4m3,
    f8e5m2 as _f8e5m2,
    float16 as _float16,
    float32 as _float32,
    hif8 as _hif8,
    int8 as _int8,
    int16 as _int16,
    int32 as _int32,
    int64 as _int64,
    si8 as _si8,
    si16 as _si16,
    si32 as _si32,
    si64 as _si64,
    tile_buf_type as _tile_buf_type,
    tensor_view_type_from_dims as _tensor_view_type_from_dims,
    ui8 as _ui8,
    ui16 as _ui16,
    ui32 as _ui32,
    ui64 as _ui64,
    _resolve,
)

from ptoas.mlir.ir import Type


@dataclass(frozen=True)
class ScalarType:
    """Author-facing dtype tag (used to build entry tile_buf types at specialize time)."""

    name: str

    def __repr__(self) -> str:
        return self.name


f32 = ScalarType("f32")
f16 = ScalarType("f16")
bf16 = ScalarType("bf16")
i32 = ScalarType("i32")
i16 = ScalarType("i16")
i8 = ScalarType("i8")
i64 = ScalarType("i64")
si32 = ScalarType("si32")
si16 = ScalarType("si16")
si8 = ScalarType("si8")
si64 = ScalarType("si64")
ui32 = ScalarType("ui32")
ui16 = ScalarType("ui16")
ui8 = ScalarType("ui8")
ui64 = ScalarType("ui64")
f8e4m3 = ScalarType("f8e4m3")
f8e5m2 = ScalarType("f8e5m2")
hif8 = ScalarType("hif8")
f4e1m2x2 = ScalarType("f4e1m2x2")
f4e2m1x2 = ScalarType("f4e2m1x2")


def scalar_descriptor(dtype: ScalarType):
    """Map a TileLib ``ScalarType`` to a ptodsl ``_types`` dtype descriptor."""
    descriptors = {
        "f32": _float32,
        "f16": _float16,
        "bf16": Type.parse("bf16"),
        "i8": _int8,
        "i16": _int16,
        "i32": _int32,
        "i64": _int64,
        "si8": _si8,
        "si16": _si16,
        "si32": _si32,
        "si64": _si64,
        "ui8": _ui8,
        "ui16": _ui16,
        "ui32": _ui32,
        "ui64": _ui64,
        "f8e4m3": _f8e4m3,
        "f8e5m2": _f8e5m2,
        "hif8": _hif8,
        "f4e1m2x2": _f4e1m2x2,
        "f4e2m1x2": _f4e2m1x2,
    }
    descriptor = descriptors.get(dtype.name)
    if descriptor is None:
        raise ValueError(f"unsupported scalar dtype {dtype.name}")
    return descriptor


def _scalar_type_token(dtype: ScalarType) -> str:
    aliases = {
        "f8e4m3": "f8E4M3FN",
        "hif8": "!pto.hif8",
    }
    return aliases.get(dtype.name, dtype.name)


@dataclass(frozen=True)
class TileSpec:
    """Concrete specialization of one tile operand.

    Shape, valid shape, memory space, and the complete tile configuration are
    carried for both constraint evaluation (selection) and the rendered entry
    ``tile_buf`` type.
    """

    shape: tuple
    dtype: ScalarType
    memory_space: str = "ub"
    valid_shape: tuple | None = None
    b_layout: str = "row_major"
    s_layout: str = "none_box"
    s_fractal_size: int = 512
    pad_value: str = "Null"
    compact_mode: str | int = "null"

    def __post_init__(self):
        if len(self.shape) != 2:
            raise ValueError("TileSpec currently only supports rank-2 tile shapes")
        if any(not isinstance(dim, int) or dim <= 0 for dim in self.shape):
            raise ValueError("TileSpec.shape must contain positive integers")

    def mlir_type(self):
        rows, cols = self.shape
        valid_shape = self.valid_shape if self.valid_shape is not None else self.shape
        return _tile_buf_type(
            [rows, cols],
            scalar_descriptor(self.dtype),
            list(valid_shape),
            blayout=_layout_token(self.b_layout),
            address_space=self.memory_space,
            slayout=_layout_token(self.s_layout),
            fractal_size=self.s_fractal_size,
            pad=_pad_token(self.pad_value),
            compact_mode=self.compact_mode,
        )


def _layout_token(value: str) -> str:
    aliases = {
        "row_major": "RowMajor",
        "col_major": "ColMajor",
        "none_box": "NoneBox",
        "RowMajor": "RowMajor",
        "ColMajor": "ColMajor",
        "NoneBox": "NoneBox",
    }
    return aliases.get(str(value), str(value))


def _pad_token(value: str) -> str:
    aliases = {
        "null": "Null",
        "Null": "Null",
        "zero": "Zero",
        "Zero": "Zero",
        "0x0": "Null",
        "0x00": "Null",
        "0x1": "Zero",
        "0x01": "Zero",
        "0x2": "Max",
        "0x02": "Max",
        "0x3": "Min",
        "0x03": "Min",
        0: "Null",
        1: "Zero",
        2: "Max",
        3: "Min",
    }
    return aliases.get(value, aliases.get(str(value), str(value)))


@dataclass(frozen=True)
class ScalarSpec:
    """Concrete specialization of one scalar TileOp operand."""

    dtype: ScalarType
    value: object | None = None

    def mlir_type(self):
        return _resolve(scalar_descriptor(self.dtype))


@dataclass(frozen=True)
class ViewSpec:
    """Concrete specialization of one tensor-view TileOp operand."""

    shape: tuple
    dtype: ScalarType
    memory_space: str = "gm"
    strides: tuple | None = None
    layout: str | None = None

    def mlir_type(self):
        return _resolve(
            _tensor_view_type_from_dims(self.shape, scalar_descriptor(self.dtype))
        )


@dataclass(frozen=True)
class VectorSpec:
    """Concrete specialization of one builtin vector TileOp operand."""

    shape: tuple
    dtype: ScalarType

    def mlir_type(self):
        dims = "x".join(str(dim) for dim in self.shape)
        elem = _resolve(scalar_descriptor(self.dtype))
        return Type.parse(f"vector<{dims}x{elem}>")


@dataclass(frozen=True)
class TemplateMetadata:
    """Hard constraints + selection hints for one registered template version."""

    op: str
    target: str
    name: str
    # Hard constraints
    dtypes: tuple = ()          # tuple of per-operand dtype-name tuples, e.g. (("f32","f32","f32"),)
    # Empty means unrestricted. One value applies to every operand; otherwise
    # provide one value per template parameter.
    layouts: tuple = ()
    memory_spaces: tuple = ()
    # Hard constraints (legality predicates: callables matched by param name — see constraints.py)
    constraints: tuple = ()
    # Selection hints
    priority: int = 0
    fusible: bool = False
    loop_depth: int | None = None
    id: int | None = None
    Tail: object = None
    is_post_update: bool = False
    iteration_axis: str = "none"
    op_engine: str = "other"
    op_class: str = "other"
    tags: tuple = ()

    @staticmethod
    def _normalize_iteration_axis(value):
        allowed = {"none", "row", "column"}
        if value not in allowed:
            raise ValueError(
                f"unsupported iteration_axis {value!r}; expected one of {', '.join(sorted(allowed))}"
            )
        return value

    @staticmethod
    def _normalize_op_engine(value):
        allowed = {"vector", "cube", "other"}
        if value not in allowed:
            raise ValueError(
                f"unsupported op_engine {value!r}; expected one of {', '.join(sorted(allowed))}"
            )
        return value

    @staticmethod
    def _normalize_op_class(value):
        allowed = {"elementwise", "reduction", "broadcast", "movement", "other"}
        if value not in allowed:
            raise ValueError(
                f"unsupported op_class {value!r}; expected one of {', '.join(sorted(allowed))}"
            )
        return value

    @staticmethod
    def build(*, op, target, name, dtypes=(), layouts=(), memory_spaces=(),
              constraints=(), priority=0, fusible=False, loop_depth=None,
              id=None, Tail=None, is_post_update=False, iteration_axis="none",
              op_engine="other", op_class="other", tags=()):
        return TemplateMetadata(
            op=op,
            target=target,
            name=name,
            dtypes=tuple(tuple(sig) for sig in dtypes),
            layouts=tuple(layouts),
            memory_spaces=tuple(memory_spaces),
            constraints=tuple(constraints),
            priority=priority,
            fusible=fusible,
            loop_depth=loop_depth,
            id=id,
            Tail=Tail,
            is_post_update=bool(is_post_update),
            iteration_axis=TemplateMetadata._normalize_iteration_axis(iteration_axis),
            op_engine=TemplateMetadata._normalize_op_engine(op_engine),
            op_class=TemplateMetadata._normalize_op_class(op_class),
            tags=tuple(tags),
        )


__all__ = [
    "ScalarType",
    "TileSpec",
    "ScalarSpec",
    "ViewSpec",
    "VectorSpec",
    "TemplateMetadata",
    "scalar_descriptor",
    "f32",
    "f16",
    "bf16",
    "i32",
    "i16",
    "i8",
    "i64",
    "si32",
    "si16",
    "si8",
    "si64",
    "ui32",
    "ui16",
    "ui8",
    "ui64",
    "f8e4m3",
    "f8e5m2",
    "hif8",
    "f4e1m2x2",
    "f4e2m1x2",
]
