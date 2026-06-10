# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Tracing-time wrappers for authored PTODSL surface values."""

from __future__ import annotations

import re
from dataclasses import dataclass

from ._diagnostics import native_python_control_flow_error
from ._runtime_scalar_ops import emit_runtime_binary_op, emit_runtime_bitwise_op, emit_runtime_compare
from ._surface_types import PartitionTensorView, TensorView, Tile
from ._types import _normalize_address_space, _resolve, ptr

from mlir.dialects import arith
from mlir.dialects import memref
from mlir.dialects import pto as _pto
from mlir.ir import IndexType, IntegerAttr, IntegerType, MemRefType, ShapedType, StridedLayoutAttr, Type


def unwrap_surface_value(value):
    """Return the underlying MLIR SSA value for a surface wrapper."""
    return value.value if isinstance(value, _SurfaceValue) else value


def _unwrap_sequence(values):
    normalized = []
    interned_ints = {}
    for value in values:
        if isinstance(value, int):
            if value not in interned_ints:
                interned_ints[value] = _index_const(value)
            normalized.append(interned_ints[value])
        else:
            normalized.append(unwrap_surface_value(value))
    return normalized


def _normalize_index(value):
    return unwrap_surface_value(value)


def _index_const(value: int):
    return arith.ConstantOp(IndexType.get(), value).result


def _add_index(lhs, rhs):
    if isinstance(lhs, int) and lhs == 0:
        return _normalize_index(rhs)
    if isinstance(rhs, int) and rhs == 0:
        return _normalize_index(lhs)
    lhs = _normalize_index(lhs)
    rhs = _normalize_index(rhs)
    if isinstance(lhs, int) and isinstance(rhs, int):
        return lhs + rhs
    if isinstance(lhs, int):
        lhs = _index_const(lhs)
    if isinstance(rhs, int):
        rhs = _index_const(rhs)
    return arith.AddIOp(lhs, rhs).result


def _try_get_constant_index(value) -> int | None:
    """Return a compile-time index when *value* is a Python int or ``arith.constant``."""
    if isinstance(value, int):
        return value
    raw = unwrap_surface_value(value)
    owner = getattr(raw, "owner", None)
    if owner is None or not hasattr(owner, "operation"):
        return None
    if owner.operation.name != "arith.constant":
        return None
    attrs = owner.operation.attributes
    if "value" not in attrs:
        return None
    try:
        return IntegerAttr(attrs["value"]).value
    except Exception:
        return None


def _static_index_dims(values) -> tuple[int, ...] | None:
    """Return static index dimensions when every entry is known at trace time."""
    dims = []
    for value in values:
        dim = _try_get_constant_index(value)
        if dim is None:
            return None
        dims.append(dim)
    return tuple(dims)


def _maybe_cast_tensor_view_type(type_obj):
    try:
        return _pto.TensorViewType(type_obj)
    except Exception:
        return None


def _maybe_cast_partition_tensor_view_type(type_obj):
    try:
        return _pto.PartitionTensorViewType(type_obj)
    except Exception:
        return None


def _maybe_cast_tile_buf_type(type_obj):
    try:
        return _pto.TileBufType(type_obj)
    except Exception:
        return None


def wrap_surface_value(
    value,
    *,
    root_tensor_view=None,
    offsets=None,
    sizes=None,
    tile_metadata=None,
):
    """Wrap a raw MLIR value into the authored PTODSL surface type when needed."""
    if isinstance(value, _SurfaceValue):
        return value

    type_obj = value.type
    if _maybe_cast_tensor_view_type(type_obj) is not None:
        return TensorViewValue(value)
    if _maybe_cast_partition_tensor_view_type(type_obj) is not None:
        return PartitionTensorViewValue(
            value,
            root_tensor_view=root_tensor_view,
            offsets=offsets,
            sizes=sizes,
        )
    if _maybe_cast_tile_buf_type(type_obj) is not None:
        return TileValue(value, **(tile_metadata or {}))
    try:
        MemRefType(type_obj)
        return AddressValue(value)
    except Exception:
        pass
    return RuntimeValue(value)


class _SurfaceValue:
    """Base class for authored PTODSL values backed by one MLIR SSA value."""

    def __init__(self, value):
        self._value = value

    @property
    def value(self):
        return self._value

    @property
    def type(self):
        return self._value.type

    @property
    def surface_metadata(self):
        return None

    def __bool__(self):
        raise native_python_control_flow_error("if/while condition")

    def __iter__(self):
        raise native_python_control_flow_error("for-loop iteration")

    def __repr__(self):
        return repr(self._value)


class RuntimeValue(_SurfaceValue):
    """Generic authored runtime value wrapper with fail-fast Python misuse diagnostics."""

    def __index__(self):
        raise native_python_control_flow_error("range()/loop bound")

    def __int__(self):
        raise native_python_control_flow_error("int() coercion")

    def __add__(self, other):
        return wrap_surface_value(emit_runtime_binary_op("add", self.value, unwrap_surface_value(other)))

    def __radd__(self, other):
        return wrap_surface_value(emit_runtime_binary_op("add", unwrap_surface_value(other), self.value))

    def __sub__(self, other):
        return wrap_surface_value(emit_runtime_binary_op("sub", self.value, unwrap_surface_value(other)))

    def __rsub__(self, other):
        return wrap_surface_value(emit_runtime_binary_op("sub", unwrap_surface_value(other), self.value))

    def __mul__(self, other):
        return wrap_surface_value(emit_runtime_binary_op("mul", self.value, unwrap_surface_value(other)))

    def __rmul__(self, other):
        return wrap_surface_value(emit_runtime_binary_op("mul", unwrap_surface_value(other), self.value))

    def __truediv__(self, other):
        return wrap_surface_value(emit_runtime_binary_op("truediv", self.value, unwrap_surface_value(other)))

    def __rtruediv__(self, other):
        return wrap_surface_value(emit_runtime_binary_op("truediv", unwrap_surface_value(other), self.value))

    def __floordiv__(self, other):
        return wrap_surface_value(emit_runtime_binary_op("floordiv", self.value, unwrap_surface_value(other)))

    def __rfloordiv__(self, other):
        return wrap_surface_value(emit_runtime_binary_op("floordiv", unwrap_surface_value(other), self.value))

    def __mod__(self, other):
        return wrap_surface_value(emit_runtime_binary_op("mod", self.value, unwrap_surface_value(other)))

    def __rmod__(self, other):
        return wrap_surface_value(emit_runtime_binary_op("mod", unwrap_surface_value(other), self.value))

    def __lt__(self, other):
        return wrap_surface_value(emit_runtime_compare("lt", self.value, unwrap_surface_value(other)))

    def __le__(self, other):
        return wrap_surface_value(emit_runtime_compare("le", self.value, unwrap_surface_value(other)))

    def __gt__(self, other):
        return wrap_surface_value(emit_runtime_compare("gt", self.value, unwrap_surface_value(other)))

    def __ge__(self, other):
        return wrap_surface_value(emit_runtime_compare("ge", self.value, unwrap_surface_value(other)))

    def __eq__(self, other):
        return wrap_surface_value(emit_runtime_compare("eq", self.value, unwrap_surface_value(other)))

    def __ne__(self, other):
        return wrap_surface_value(emit_runtime_compare("ne", self.value, unwrap_surface_value(other)))

    def __and__(self, other):
        return wrap_surface_value(emit_runtime_bitwise_op("and", self.value, unwrap_surface_value(other)))

    def __rand__(self, other):
        return wrap_surface_value(emit_runtime_bitwise_op("and", unwrap_surface_value(other), self.value))

    def __or__(self, other):
        return wrap_surface_value(emit_runtime_bitwise_op("or", self.value, unwrap_surface_value(other)))

    def __ror__(self, other):
        return wrap_surface_value(emit_runtime_bitwise_op("or", unwrap_surface_value(other), self.value))

    def __xor__(self, other):
        return wrap_surface_value(emit_runtime_bitwise_op("xor", self.value, unwrap_surface_value(other)))

    def __rxor__(self, other):
        return wrap_surface_value(emit_runtime_bitwise_op("xor", unwrap_surface_value(other), self.value))


class MaskResultValue(_SurfaceValue):
    """Mask value that also supports `(mask, remained)` unpacking."""

    def __init__(self, mask_value, scalar_out):
        super().__init__(mask_value)
        self.scalar_out = wrap_surface_value(scalar_out)

    def __iter__(self):
        yield self
        yield self.scalar_out


class AddressValue(_SurfaceValue):
    """Author-facing address view backed by either a PTO ptr or a memref."""

    def __add__(self, offset):
        return AddressOffsetValue(self, offset)

    def __radd__(self, offset):
        return AddressOffsetValue(self, offset)


@dataclass(frozen=True)
class AddressOffsetValue:
    """Address view plus an element offset, used by scalar.load/store sugar."""

    base: AddressValue
    offset: object

    def __add__(self, other):
        return AddressOffsetValue(self.base, _add_index(self.offset, other))

    def __radd__(self, other):
        return AddressOffsetValue(self.base, _add_index(other, self.offset))

    def __bool__(self):
        raise native_python_control_flow_error("if/while condition")

    def __iter__(self):
        raise native_python_control_flow_error("for-loop iteration")


@dataclass(frozen=True)
class TileElementRef:
    """One logical tile element selected by tile[row, col] surface syntax."""

    tile: "TileValue"
    linear_offset: object

    def __bool__(self):
        raise native_python_control_flow_error("if/while condition")

    def __iter__(self):
        raise native_python_control_flow_error("for-loop iteration")


class TileSliceValue(_SurfaceValue):
    """Author-facing memref view produced by `tile[row, col:]` style indexing."""

    def __init__(self, value, *, tile: "TileValue", offsets, shape):
        super().__init__(value)
        self.tile = tile
        self.offsets = tuple(offsets)
        self.shape = tuple(shape)

    @property
    def surface_metadata(self):
        return {
            "tile": self.tile,
            "offsets": self.offsets,
            "shape": self.shape,
        }


class TensorViewValue(_SurfaceValue, TensorView):
    """Author-facing tensor-view descriptor value."""

    def __init__(self, value, *, shape=None, strides=None):
        super().__init__(value)
        self.shape = tuple(shape) if shape is not None else None
        self.strides = tuple(strides) if strides is not None else None

    @property
    def surface_metadata(self):
        return {
            "shape": self.shape,
            "strides": self.strides,
        }

    def as_ptr(self):
        from ._ops import as_ptr
        return as_ptr(self)


class PartitionTensorViewValue(_SurfaceValue, PartitionTensorView):
    """Author-facing partitioned tensor-view descriptor value."""

    def __init__(self, value, *, root_tensor_view=None, offsets=None, sizes=None):
        super().__init__(value)
        self.root_tensor_view = root_tensor_view
        self.offsets = tuple(offsets) if offsets is not None else None
        self.sizes = tuple(sizes) if sizes is not None else None
        self.shape = self.sizes
        self.strides = getattr(root_tensor_view, "strides", None)

    def as_ptr(self):
        from ._ops import as_ptr
        return as_ptr(self)


class _TileValidShapeView:
    """Tuple-like proxy that lowers `tile.valid_shape[i]` on demand."""

    def __init__(self, tile: "TileValue"):
        self._tile = tile
        self._cache: dict[int, object] = {}

    def __getitem__(self, index: int):
        logical_rank = len(self._tile.shape) if self._tile.shape is not None else 2
        allowed = {0} if logical_rank == 1 else {0, 1}
        if index not in allowed:
            if logical_rank == 1:
                raise IndexError("PTODSL rank-1 tile.valid_shape currently supports only index 0")
            raise IndexError("PTODSL tile.valid_shape currently supports indices 0 and 1")
        cached = self._cache.get(index)
        if cached is not None:
            return cached
        if self._tile.static_valid_shape is not None:
            dim = self._tile.static_valid_shape[index]
            if dim is not None:
                value = _index_const(dim) if isinstance(dim, int) else unwrap_surface_value(dim)
                value = wrap_surface_value(value)
                self._cache[index] = value
                return value
        try:
            if logical_rank == 1:
                value = wrap_surface_value(_pto.TileValidColsOp(self._tile.value).result)
            elif index == 0:
                value = wrap_surface_value(_pto.TileValidRowsOp(self._tile.value).result)
            else:
                value = wrap_surface_value(_pto.TileValidColsOp(self._tile.value).result)
        except Exception:
            static_dim = _fallback_static_valid_dim(self._tile.type, index)
            if static_dim is None:
                raise RuntimeError(
                    "tile.valid_shape could not be lowered because the current "
                    "Python bindings do not materialize pto.tile_valid_* and "
                    "the tile type does not carry a recoverable static bound"
                ) from None
            value = wrap_surface_value(_index_const(static_dim))
        self._cache[index] = value
        return value


class TileValue(_SurfaceValue, Tile):
    """Author-facing tile handle with surface-style accessors."""

    def __init__(
        self,
        value,
        *,
        shape=None,
        physical_shape=None,
        dtype=None,
        memory_space=None,
        valid_shape=None,
    ):
        super().__init__(value)
        parsed = parse_tile_type_metadata(value.type)
        self.shape = tuple(shape) if shape is not None else (
            parsed["shape_dims"] if parsed is not None else None
        )
        self.physical_shape = tuple(physical_shape) if physical_shape is not None else (
            tuple(shape) if shape is not None else (
                parsed["shape_dims"] if parsed is not None else None
            )
        )
        self.dtype = dtype if dtype is not None else (
            parsed["element_type"] if parsed is not None else None
        )
        self.memory_space = memory_space if memory_space is not None else (
            parsed["memory_space"] if parsed is not None else None
        )
        self.static_valid_shape = tuple(valid_shape) if valid_shape is not None else (
            parsed["valid_dims"] if parsed is not None else None
        )
        self._valid_shape = _TileValidShapeView(self)

    @property
    def valid_shape(self):
        return self._valid_shape

    @valid_shape.setter
    def valid_shape(self, dims):
        from ._ops import set_tile_valid_shape

        set_tile_valid_shape(self, dims)
        self.static_valid_shape = tuple(dims)
        self._valid_shape._cache.clear()

    @property
    def surface_metadata(self):
        return {
            "shape": self.shape,
            "physical_shape": self.physical_shape,
            "dtype": self.dtype,
            "memory_space": self.memory_space,
            "valid_shape": self.static_valid_shape,
        }

    def as_ptr(self):
        from ._ops import as_ptr
        return as_ptr(self)

    def fill(self, value):
        from ._ops import fill_tile
        fill_tile(self, value)

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        if self.shape is None:
            raise RuntimeError("tile indexing requires tile shape metadata")

        if _is_tile_slice_key(key, self.shape):
            return _materialize_tile_slice(self, key)

        if len(key) != len(self.shape):
            raise TypeError(
                f"tile indexing expects {len(self.shape)} indices, got {len(key)}"
            )
        linear_offset = 0
        stride = 1
        for index, dim in zip(reversed(key), reversed(self.shape)):
            linear_offset = _add_index(linear_offset, _mul_index(index, stride))
            if dim is None:
                raise RuntimeError("tile indexing requires static tile shape metadata")
            stride *= dim
        return TileElementRef(self, linear_offset)


@dataclass(frozen=True)
class PartitionSpec:
    """Logical authored partition metadata used to compose nested slices."""

    root_tensor_view: object
    offsets: tuple
    sizes: tuple


def wrap_like_surface_value(template, value):
    """Wrap *value* using the same authored surface contract as *template*."""
    if isinstance(template, PartitionTensorViewValue):
        return PartitionTensorViewValue(
            value,
            root_tensor_view=template.root_tensor_view,
            offsets=template.offsets,
            sizes=template.sizes,
        )
    if isinstance(template, TensorViewValue):
        return TensorViewValue(value, shape=template.shape, strides=template.strides)
    if isinstance(template, TileValue):
        return TileValue(value, **template.surface_metadata)
    if isinstance(template, AddressValue):
        return AddressValue(value)
    return wrap_surface_value(value)


def extract_partition_spec(source) -> PartitionSpec | None:
    """Return the root tensor-view + composed slice metadata when available."""
    if isinstance(source, PartitionTensorViewValue) and source.root_tensor_view is not None:
        return PartitionSpec(
            root_tensor_view=source.root_tensor_view,
            offsets=source.offsets or (),
            sizes=source.sizes or (),
        )
    if isinstance(source, TensorViewValue):
        return PartitionSpec(root_tensor_view=source, offsets=(), sizes=())
    return None


def compose_partition_spec(source, *, offsets, sizes) -> PartitionSpec | None:
    """Compose a nested `partition_view(...)` against an existing partition."""
    parent = extract_partition_spec(source)
    if parent is None:
        return None
    if isinstance(source, TensorViewValue):
        return PartitionSpec(
            root_tensor_view=source,
            offsets=tuple(offsets),
            sizes=tuple(sizes),
        )
    if parent.offsets and len(parent.offsets) != len(offsets):
        raise ValueError("nested partition_view rank mismatch")
    composed_offsets = tuple(
        _add_index(parent_offset, child_offset)
        for parent_offset, child_offset in zip(parent.offsets, offsets)
    )
    return PartitionSpec(
        root_tensor_view=parent.root_tensor_view,
        offsets=composed_offsets,
        sizes=tuple(sizes),
    )


def infer_ptr_type_from_surface_value(surface_value):
    """Infer a PTO pointer type for `as_ptr()` from the authored source value."""
    value_type = surface_value.type

    tv_type = _maybe_cast_tensor_view_type(value_type)
    if tv_type is not None:
        return _resolve(ptr(tv_type.element_type, "gm"))

    part_type = _maybe_cast_partition_tensor_view_type(value_type)
    if part_type is not None:
        return _resolve(ptr(part_type.element_type, "gm"))

    tile_type = _maybe_cast_tile_buf_type(value_type)
    if tile_type is None:
        raise TypeError("as_ptr() expects a Tile, TensorView, or PartitionTensorView surface value")

    memory_space = getattr(tile_type, "memory_space", None)
    parsed = None
    if memory_space is None:
        parsed = parse_tile_type_metadata(value_type)
        if parsed is None:
            raise RuntimeError("unable to infer tile pointer type: tile type is missing memory-space metadata")
        memory_space = parsed["memory_space"]

    space_enum = getattr(memory_space, "value", None)
    if space_enum is not None:
        space_enum = _normalize_address_space(_ADDRESS_SPACE_VALUE_TO_KEYWORD.get(space_enum))
    else:
        space_enum = _normalize_address_space(str(memory_space))
    if space_enum is None:
        raise RuntimeError("unable to infer tile pointer type: unsupported tile memory space")

    try:
        return _resolve(ptr(tile_type.element_type, space_enum))
    except TypeError as exc:
        if "storage-only low-precision type" not in str(exc):
            raise
        return _resolve_storage_tile_ptr_type(tile_type.element_type, space_enum)


def _resolve_storage_tile_ptr_type(element_type, space_enum):
    space_attr = _pto.AddressSpaceAttr.get(space_enum)
    try:
        return _pto.PtrType.get(element_type, memory_space=space_attr)
    except TypeError:
        ptr_get_impl = getattr(_pto, "_ptr_type_get_impl", None)
        if ptr_get_impl is None:
            raise
        if space_enum != _pto.AddressSpace.GM:
            return ptr_get_impl(element_type, space_attr)
        return _pto.PtrType.get(element_type)


def emit_as_ptr(surface_value):
    """Lower `as_ptr()` on a surface value to the appropriate PTO op."""
    value = unwrap_surface_value(surface_value)
    result_type = infer_address_type_from_surface_value(surface_value)

    if isinstance(surface_value, (TensorViewValue, PartitionTensorViewValue)):
        return AddressValue(_pto.TensorViewAddrOp(result_type, value).result)
    if isinstance(surface_value, TileValue):
        return AddressValue(_pto.TileBufAddrOp(result_type, value).result)
    raise TypeError("as_ptr() expects a Tile, TensorView, or PartitionTensorView surface value")


_TILE_TYPE_RE = re.compile(
    r"!pto\.tile_buf<(?P<space>[^,]+),\s*(?P<shape>[0-9?x\s]+)x(?P<elem>[^,>]+)(?:,\s*blayout=[^,>]+)?(?:,\s*slayout=[^,>]+)?(?:,\s*valid=(?P<valid>[^,>]+))?(?:,.*)?>"
)


_ADDRESS_SPACE_VALUE_TO_KEYWORD = {
    1: "gm",
    2: "mat",
    3: "left",
    4: "right",
    5: "acc",
    6: "vec",
    7: "bias",
    8: "scaling",
}


def _read_tile_type_metadata_from_binding(type_obj):
    required = ("shape", "element_type", "memory_space", "valid_shape")
    if not all(hasattr(type_obj, name) for name in required):
        return None

    memory_space_attr = type_obj.memory_space
    memory_space_value = getattr(memory_space_attr, "value", None)
    memory_space = _ADDRESS_SPACE_VALUE_TO_KEYWORD.get(memory_space_value)
    if memory_space is None:
        return None

    def _normalize_dims(seq):
        dims = []
        for dim in seq:
            dims.append(None if dim == ShapedType.get_dynamic_size() else int(dim))
        return tuple(dims)

    return {
        "memory_space": memory_space,
        "shape_dims": _normalize_dims(type_obj.shape),
        "element_type": type_obj.element_type,
        "valid_dims": _normalize_dims(type_obj.valid_shape),
    }


def _fallback_static_valid_dim(type_obj, index: int):
    parsed = parse_tile_type_metadata(type_obj)
    if parsed is None:
        return None
    shape_dims = parsed["shape_dims"]
    valid_dims = parsed["valid_dims"]
    if index >= len(shape_dims) or index >= len(valid_dims):
        return None
    valid_dim = valid_dims[index]
    if valid_dim is not None:
        return valid_dim
    return shape_dims[index]


def parse_tile_type_metadata(type_obj):
    bound = _read_tile_type_metadata_from_binding(type_obj)
    if bound is not None:
        return bound

    match = _TILE_TYPE_RE.match(str(type_obj))
    if match is None:
        return None
    shape_dims = [
        None if dim == "?" else int(dim)
        for dim in match.group("shape").split("x")
    ]
    valid_group = match.group("valid")
    if valid_group:
        valid_dims = [
            None if dim == "?" else int(dim)
            for dim in valid_group.split("x")
        ]
    else:
        valid_dims = list(shape_dims)
    return {
        "memory_space": match.group("space"),
        "shape_dims": tuple(shape_dims),
        "element_type": Type.parse(match.group("elem")),
        "valid_dims": tuple(valid_dims),
    }


def infer_tile_element_type(tile):
    """Recover the tile element type from authored metadata or type text."""
    if isinstance(tile, TileValue) and tile.dtype is not None:
        return _resolve(tile.dtype)
    parsed = parse_tile_type_metadata(tile.type if isinstance(tile, TileValue) else tile)
    if parsed is None:
        raise RuntimeError("unable to recover tile element type from tile surface value")
    return parsed["element_type"]


def infer_address_type_from_surface_value(surface_value):
    """Infer the concrete result type emitted by `as_ptr()`."""
    return infer_ptr_type_from_surface_value(surface_value)


def infer_memref_type_from_surface_value(surface_value):
    """Build a memref address-view type that preserves element/rank/address-space."""
    if isinstance(surface_value, TileSliceValue):
        return surface_value.type

    if isinstance(surface_value, TileValue):
        physical_shape = getattr(surface_value, "physical_shape", None)
        if physical_shape is not None and surface_value.dtype is not None and surface_value.memory_space is not None:
            space_enum = _normalize_address_space(surface_value.memory_space)
            if space_enum is None:
                raise RuntimeError("unsupported tile memory space for memref address view")
            return MemRefType.get(
                list(physical_shape),
                _resolve(surface_value.dtype),
                memory_space=_pto.AddressSpaceAttr.get(space_enum),
            )

    value_type = surface_value.type

    tv_type = _maybe_cast_tensor_view_type(value_type)
    if tv_type is not None:
        return MemRefType.get(
            [ShapedType.get_dynamic_size()] * tv_type.rank,
            tv_type.element_type,
            memory_space=_pto.AddressSpaceAttr.get(_pto.AddressSpace.GM),
        )

    part_type = _maybe_cast_partition_tensor_view_type(value_type)
    if part_type is not None:
        return MemRefType.get(
            [ShapedType.get_dynamic_size()] * part_type.rank,
            part_type.element_type,
            memory_space=_pto.AddressSpaceAttr.get(_pto.AddressSpace.GM),
        )

    tile_type = _maybe_cast_tile_buf_type(value_type)
    if tile_type is None:
        raise TypeError("memref address inference expects a Tile, TensorView, or PartitionTensorView")

    parsed = parse_tile_type_metadata(value_type)
    if parsed is None:
        raise RuntimeError("unable to recover tile memref shape/address-space")
    space_enum = _normalize_address_space(parsed["memory_space"])
    if space_enum is None:
        raise RuntimeError("unsupported tile memory space for memref address view")
    return MemRefType.get(
        list(parsed["shape_dims"]),
        parsed["element_type"],
        memory_space=_pto.AddressSpaceAttr.get(space_enum),
    )


def resolve_address_access(target, offset=None):
    """Normalize address/tile element sugar into `(buffer, index_offset)`."""
    if isinstance(target, TileElementRef):
        base = emit_as_ptr(target.tile)
        resolved_offset = target.linear_offset
    elif isinstance(target, AddressOffsetValue):
        base = target.base
        resolved_offset = target.offset
    elif isinstance(target, AddressValue):
        base = target
        resolved_offset = 0
    else:
        base = target
        resolved_offset = 0

    if offset is not None:
        resolved_offset = _add_index(resolved_offset, offset)

    return unwrap_surface_value(base), _coerce_index_value(resolved_offset)


def _is_tile_slice_key(key, shape):
    if len(shape) == 1:
        return len(key) == 1 and isinstance(key[0], slice)
    if len(shape) == 2:
        return len(key) == 2 and isinstance(key[1], slice)
    return False


def _materialize_tile_slice(tile: TileValue, key):
    rank = len(tile.shape)
    if rank == 1:
        start_slice = key[0]
        if start_slice.stop is not None or start_slice.step is not None:
            raise TypeError("tile[start:] only supports an open-ended slice")
        start = 0 if start_slice.start is None else start_slice.start
        return _build_tile_slice_view(
            tile,
            raw_offsets=[0, start],
            shape=[_dynamic_extent(tile.shape[0], start)],
        )

    row, col_slice = key
    if col_slice.stop is not None or col_slice.step is not None:
        raise TypeError("tile[row, col:] only supports an open-ended column slice")
    col = 0 if col_slice.start is None else col_slice.start
    return _build_tile_slice_view(
        tile,
        raw_offsets=[row, col],
        shape=[_dynamic_extent(tile.shape[1], col)],
    )


def _build_tile_slice_view(tile: TileValue, *, raw_offsets, shape):
    base_memref = _emit_tile_memref(tile)
    base_type = MemRefType(base_memref.type)
    rank = len(base_type.shape)
    offset_operands, static_offsets = _split_dynamic_index_operands(raw_offsets)
    shape_operands, static_shape = _split_dynamic_index_operands(shape)
    base_strides, base_offset = base_type.get_strides_and_offset()
    if rank == 1:
        slice_type = _make_strided_memref_type(
            [_static_extent_if_known(shape[0])],
            base_type.element_type,
            [base_strides[0]],
            base_type.memory_space,
            offset=_compose_static_subview_offset(base_offset, base_strides, raw_offsets),
        )
        slice_value = memref.SubViewOp(
            slice_type,
            base_memref,
            offset_operands,
            shape_operands,
            [],
            static_offsets,
            static_shape,
            [1],
        ).result
        return TileSliceValue(slice_value, tile=tile, offsets=tuple(raw_offsets), shape=shape)

    slice_type = _make_strided_memref_type(
        [_static_extent_if_known(shape[0])],
        base_type.element_type,
        [base_strides[1]],
        base_type.memory_space,
        offset=_compose_static_subview_offset(base_offset, base_strides, raw_offsets),
    )
    slice_value = memref.SubViewOp(
        slice_type,
        base_memref,
        offset_operands,
        shape_operands,
        [],
        static_offsets,
        [1, static_shape[0]],
        [1, 1],
    ).result
    return TileSliceValue(slice_value, tile=tile, offsets=tuple(raw_offsets), shape=shape)


def _emit_tile_memref(tile: TileValue):
    memref_type = infer_memref_type_from_surface_value(tile)
    return _pto.TileBufAddrOp(memref_type, tile.value).result


def _dynamic_extent(static_dim, start):
    if isinstance(start, int):
        return static_dim - start
    return arith.SubIOp(_index_const(static_dim), _coerce_index_value(start)).result


def _static_extent_if_known(extent):
    return extent if isinstance(extent, int) else ShapedType.get_dynamic_size()


def _static_index_attr(value):
    return value if isinstance(value, int) else ShapedType.get_dynamic_size()


def _split_dynamic_index_operands(values):
    operands = []
    static_attrs = []
    for value in values:
        if isinstance(value, int):
            static_attrs.append(value)
        else:
            operands.append(_coerce_index_value(value))
            static_attrs.append(ShapedType.get_dynamic_size())
    return operands, static_attrs


def _make_strided_memref_type_with_offset(shape, element_type, strides, memory_space, *, offset):
    return MemRefType.get(
        list(shape),
        element_type,
        StridedLayoutAttr.get(offset, list(strides)),
        memory_space,
    )


def _make_strided_memref_type(shape, element_type, strides, memory_space, *, offset=ShapedType.get_dynamic_size()):
    return _make_strided_memref_type_with_offset(
        shape,
        element_type,
        strides,
        memory_space,
        offset=offset,
    )


def _compose_static_subview_offset(base_offset, base_strides, raw_offsets):
    if base_offset == ShapedType.get_dynamic_size():
        return ShapedType.get_dynamic_size()

    linear_offset = base_offset
    for stride, authored_offset in zip(base_strides, raw_offsets):
        if not isinstance(authored_offset, int):
            return ShapedType.get_dynamic_size()
        linear_offset += stride * authored_offset
    return linear_offset


def _mul_index(lhs, rhs):
    lhs = _normalize_index(lhs)
    rhs = _normalize_index(rhs)
    if isinstance(lhs, int) and isinstance(rhs, int):
        return lhs * rhs
    if isinstance(lhs, int):
        lhs = _index_const(lhs)
    if isinstance(rhs, int):
        rhs = _index_const(rhs)
    return arith.MulIOp(lhs, rhs).result


def _coerce_index_value(value):
    value = _normalize_index(value)
    if isinstance(value, int):
        return _index_const(value)
    if IndexType.isinstance(value.type):
        return value
    if IntegerType.isinstance(value.type):
        return arith.IndexCastOp(IndexType.get(), value).result
    raise TypeError(f"expected an index-like value, got {value.type}")


__all__ = [
    "AddressOffsetValue",
    "AddressValue",
    "MaskResultValue",
    "PartitionSpec",
    "PartitionTensorViewValue",
    "RuntimeValue",
    "TileElementRef",
    "TileSliceValue",
    "TensorViewValue",
    "TileValue",
    "compose_partition_spec",
    "emit_as_ptr",
    "extract_partition_spec",
    "infer_tile_element_type",
    "infer_address_type_from_surface_value",
    "infer_memref_type_from_surface_value",
    "infer_ptr_type_from_surface_value",
    "parse_tile_type_metadata",
    "resolve_address_access",
    "unwrap_surface_value",
    "wrap_like_surface_value",
    "wrap_surface_value",
    "_unwrap_sequence",
]
