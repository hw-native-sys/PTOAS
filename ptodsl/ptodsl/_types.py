# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
Lazy MLIR type descriptors and eager type constructors.

Type descriptors (``_DType`` subclasses) can be created *before* any MLIR
Context exists – they only resolve to concrete ``ptoas.mlir.ir.Type`` objects when
``_resolve()`` is called inside an active context.  This lets users write::

    def softmax(arg0: pto.ptr(pto.float32, "GM"), ...):
        ...

where the annotation is evaluated at *import* time (no active context), and
the actual type is materialised later by the ``@pto.jit`` decorator.
"""

from ptoas.mlir.dialects import pto as _pto
from ptoas.mlir.dialects import arith
from ptoas.mlir.dialects.builtin import UnrealizedConversionCastOp
from ptoas.mlir.ir import (
    BF16Type,
    F16Type,
    F32Type,
    Float8E4M3FNType,
    Float8E5M2Type,
    FloatAttr,
    IndexType,
    IntegerType,
    ShapedType,
    Type,
    VectorType,
)

from ._context import make_context

# ── Address-space name → AddressSpace enum ───────────────────────────────────
_ADDR_SPACE = {
    "ub":  _pto.AddressSpace.VEC,  # UB == unified buffer == VEC in PTO
    "gm":  _pto.AddressSpace.GM,
    "vec": _pto.AddressSpace.VEC,
    "mat": _pto.AddressSpace.MAT,
    "left": _pto.AddressSpace.LEFT,
    "right": _pto.AddressSpace.RIGHT,
    "acc": _pto.AddressSpace.ACC,
    "bias": _pto.AddressSpace.BIAS,
    "scaling": _pto.AddressSpace.SCALING,
    "GM":  _pto.AddressSpace.GM,
    "UB":  _pto.AddressSpace.VEC,
    "VEC": _pto.AddressSpace.VEC,
    "MAT": _pto.AddressSpace.MAT,
    "LEFT": _pto.AddressSpace.LEFT,
    "RIGHT": _pto.AddressSpace.RIGHT,
    "ACC": _pto.AddressSpace.ACC,
    "BIAS": _pto.AddressSpace.BIAS,
    "SCALING": _pto.AddressSpace.SCALING,
}


# ── Lazy type descriptor base ─────────────────────────────────────────────────

class _DType:
    """Deferred MLIR type: only resolves inside an active MLIR context."""

    def __init__(self, factory):
        self._factory = factory

    def resolve(self) -> Type:
        return self._factory()

    def __call__(self, value):
        target_type = self.resolve()
        kind = _classify_scalar_type(target_type)
        if kind == "float":
            return arith.ConstantOp(target_type, _parse_float_attr(target_type, value)).result
        if kind == "integer":
            return _materialize_integer_literal(target_type, value)
        raise TypeError(f"unsupported eager constructor target type {target_type}")

    def __ptodsl_cache_signature__(self):
        with make_context():
            return ("dtype", str(self.resolve()))

    def __repr__(self):
        return f"<pto.dtype {self._factory}>"


class _PtrDescriptor(_DType):
    def __init__(self, elem, space: str):
        self._elem = elem
        self._space = space

    def resolve(self) -> Type:
        elem = _ensure_tensor_storage_dtype(self._elem, context="pto.ptr(...)")
        space_enum = _normalize_address_space(self._space)
        if space_enum is None:
            raise ValueError(
                f"Unknown address space '{self._space}'; "
                f"known: {list(_ADDR_SPACE)}"
            )
        space_attr = _pto.AddressSpaceAttr.get(space_enum)
        try:
            return _pto.PtrType.get(elem, memory_space=space_attr)
        except TypeError:
            ptr_get_impl = getattr(_pto, "_ptr_type_get_impl", None)
            if ptr_get_impl is None:
                raise
            if space_enum != _pto.AddressSpace.GM:
                raise TypeError(
                    "The current PTO Python bindings only expose the default-GM "
                    "PtrType builder. Non-GM pointer construction is not "
                    "available through ptodsl._types.ptr(...) yet."
                )
            return ptr_get_impl(elem)

    def __repr__(self):
        return f"<pto.ptr {self._elem} {self._space}>"


class _VRegDescriptor(_DType):
    def __init__(self, lanes: int, elem):
        self._lanes = lanes
        self._elem = elem

    def resolve(self) -> Type:
        elem = _ensure_tensor_storage_dtype(self._elem, context="pto.vreg_type(...)")
        vreg_type_cls = getattr(_pto, "VRegType", None)
        if vreg_type_cls is None:
            raise TypeError(
                "The current PTO Python bindings do not expose VRegType. "
                "Rebuild the PTO Python extension before using pto.vreg_type(...)."
            )
        return vreg_type_cls.get(self._lanes, elem)

    def __repr__(self):
        return f"<pto.vreg {self._lanes}x{self._elem}>"


class _MaskDescriptor(_DType):
    def __init__(self, bits: str):
        self._bits = bits

    def resolve(self) -> Type:
        mask_type_cls = getattr(_pto, "MaskType", None)
        if mask_type_cls is None:
            raise TypeError(
                "The current PTO Python bindings do not expose MaskType. "
                "Rebuild the PTO Python extension before using pto.mask_type(...)."
            )
        return mask_type_cls.get(self._bits)

    def __repr__(self):
        return f"<pto.mask {self._bits}>"

class _StructDescriptor(_DType):
    """Deferred ``!pto.struct<...>`` type assembled from scalar fields."""

    def __init__(self, field_descriptors):
        self._field_descriptors = tuple(field_descriptors)

    @property
    def field_descriptors(self):
        return self._field_descriptors

    def resolve(self) -> Type:
        struct_type_cls = getattr(_pto, "StructType", None)
        if struct_type_cls is None:
            raise TypeError(
                "The current PTO Python bindings do not expose StructType. "
                "Rebuild the PTO Python extension before using pto.struct_type(...)."
            )
        field_types = [
            _resolve_struct_field_type(field, context="pto.struct_type(...)")
            for field in self._field_descriptors
        ]
        return struct_type_cls.get(field_types)

    def __repr__(self):
        fields = ", ".join(repr(field) for field in self._field_descriptors)
        return f"<pto.struct {fields}>"

# Legal logical lane counts for VMI vreg/mask types on the formal PTODSL
# surface: compact/group-slot flows use the first four values, full logical
# vectors use 64/128/256.  The IR layer intentionally accepts additional
# positive lane counts for internal compatibility.
VMI_LANE_COUNTS = (1, 2, 4, 8, 64, 128, 256)

class _VMIVRegDescriptor(_DType):
    def __init__(self, lanes: int, elem):
        if lanes not in VMI_LANE_COUNTS:
            raise ValueError(
                "pto.vmi.vreg(...) requires lanes to be one of "
                "1, 2, 4, 8, 64, 128, 256"
            )
        self._lanes = lanes
        self._elem = elem

    def resolve(self) -> Type:
        elem = _ensure_tensor_storage_dtype(self._elem, context="pto.vmi.vreg(...)")
        vreg_type_cls = getattr(_pto, "VMIVRegType", None)
        if vreg_type_cls is None:
            raise TypeError(
                "The current PTO Python bindings do not expose VMIVRegType. "
                "Rebuild the PTO Python extension before using pto.vmi.vreg(...)."
            )
        return vreg_type_cls.get(self._lanes, elem)

    def __repr__(self):
        return f"<pto.vmi.vreg {self._lanes}x{self._elem}>"


class _VMIMaskDescriptor(_DType):
    def __init__(self, lanes: int):
        if lanes not in VMI_LANE_COUNTS:
            raise ValueError(
                "pto.vmi.mask(...) requires lanes to be one of "
                "1, 2, 4, 8, 64, 128, 256"
            )
        self._lanes = lanes
        self._granularity = "pred"

    def resolve(self) -> Type:
        mask_type_cls = getattr(_pto, "VMIMaskType", None)
        if mask_type_cls is None:
            raise TypeError(
                "The current PTO Python bindings do not expose VMIMaskType. "
                "Rebuild the PTO Python extension before using pto.vmi.mask(...)."
            )
        return mask_type_cls.get(self._lanes, self._granularity)

    def __repr__(self):
        return f"<pto.vmi.mask {self._lanes}x{self._granularity}>"

class _VecDescriptor(_DType):
    def __init__(self, elem, size: int):
        self._elem = elem
        self._size = _validate_vec_size(size, context="pto.Vec(...)")

    def resolve(self) -> Type:
        elem = _ensure_non_storage_only_dtype(self._elem, context="pto.Vec(...)")
        return VectorType.get([self._size], elem)

    @property
    def size(self) -> int:
        return self._size

    @property
    def elem(self):
        return self._elem

    def __repr__(self):
        return f"<pto.Vec {self._size}x{self._elem}>"


def _validate_vec_size(size: int, *, context: str) -> int:
    if isinstance(size, bool) or not isinstance(size, int):
        raise TypeError(f"{context} expects size to be a positive Python integer")
    if size <= 0:
        raise ValueError(f"{context} expects size to be positive")
    return size


def _resolve(dtype) -> Type:
    """Coerce a ``_DType`` descriptor or a concrete ``ptoas.mlir.ir.Type`` to a Type."""
    if isinstance(dtype, _DType):
        return dtype.resolve()
    return dtype  # already a ptoas.mlir.ir.Type


def _classify_scalar_type(type_obj):
    if F32Type.isinstance(type_obj) or F16Type.isinstance(type_obj) or BF16Type.isinstance(type_obj):
        return "float"
    if IndexType.isinstance(type_obj) or IntegerType.isinstance(type_obj):
        return "integer"
    return None


def _isinstance_pto_type(type_obj, type_name: str) -> bool:
    cls = getattr(_pto, type_name, None)
    if cls is None:
        return False
    try:
        return cls.isinstance(type_obj)
    except Exception:
        return False


def _is_struct_type(type_obj) -> bool:
    """Return whether *type_obj* is a PTO struct type without requiring new bindings."""
    return isinstance(type_obj, _StructDescriptor) or _isinstance_pto_type(type_obj, "StructType")


def _resolve_struct_field_type(field, *, context: str) -> Type:
    """Resolve one public PTODSL struct field to a scalar or nested struct type."""
    if isinstance(field, _StructDescriptor):
        return field.resolve()
    if isinstance(field, _DType):
        field_type = field.resolve()
    elif isinstance(field, Type):
        field_type = field
    else:
        raise TypeError(
            f"{context} field must be a PTODSL scalar dtype, an MLIR scalar type, "
            f"or another pto.struct_type(...); got {field!r}"
        )

    if _is_struct_type(field_type):
        return field_type
    if IntegerType.isinstance(field_type):
        width = IntegerType(field_type).width
        if width in (8, 16, 32, 64):
            return field_type
    if any(cls.isinstance(field_type) for cls in (F16Type, BF16Type, F32Type)):
        return field_type
    raise TypeError(
        f"{context} field type {field_type} is not supported; expected i8/i16/i32/i64, "
        "f16/bf16/f32, or a nested pto.struct_type(...)"
    )


def _classify_storage_dtype(type_obj):
    if _classify_scalar_type(type_obj) is not None:
        return "compute"
    if Float8E4M3FNType.isinstance(type_obj) or Float8E5M2Type.isinstance(type_obj):
        return "storage_only"
    if any(
        _isinstance_pto_type(type_obj, name)
        for name in (
            "BF16x2Type",
            "F8E8M0Type",
            "HiF8Type",
            "HiF8x2Type",
            "F4E1M2x2Type",
            "F4E2M1x2Type",
        )
    ):
        return "storage_only"
    if VectorType.isinstance(type_obj):
        vec_elem = VectorType(type_obj).element_type
        if _classify_scalar_type(vec_elem) is not None:
            return "compute"
        # fp8 vector types (e.g. vector<2/4/8xf8E4M3FN>) are storage_only
        if Float8E4M3FNType.isinstance(vec_elem) or Float8E5M2Type.isinstance(vec_elem):
            return "storage_only"
    return "other"


def _is_storage_only_dtype(type_obj):
    return _classify_storage_dtype(type_obj) == "storage_only"


def _is_storage_only_authored_dtype(dtype) -> bool:
    if isinstance(dtype, _DType):
        return dtype in _STORAGE_ONLY_DTYPE_DESCRIPTORS
    return _is_storage_only_dtype(_resolve(dtype))


def _ensure_tensor_storage_dtype(dtype, *, context: str):
    type_obj = _resolve(dtype)
    category = _classify_storage_dtype(type_obj)
    if category not in {"compute", "storage_only"}:
        raise TypeError(f"{context} does not support element type {type_obj}")
    return type_obj


def _ensure_non_storage_only_dtype(dtype, *, context: str):
    type_obj = _resolve(dtype)
    if _is_storage_only_dtype(type_obj):
        raise TypeError(
            f"{context} does not accept storage-only low-precision type {type_obj}; "
            "these dtypes are only supported in Tile / TensorView / PartitionTensorView construction"
        )
    return type_obj


def _ensure_non_storage_only_authored_dtype(dtype, *, context: str):
    if _is_storage_only_authored_dtype(dtype):
        raise TypeError(
            f"{context} does not accept storage-only low-precision types; "
            "these dtypes are only supported in Tile / TensorView / PartitionTensorView construction"
        )
    return dtype


def _integer_signedness(type_obj):
    if not IntegerType.isinstance(type_obj):
        raise TypeError(f"expected integer type, got {type_obj}")
    text = str(type_obj)
    if text.startswith("si"):
        return "signed"
    if text.startswith("ui"):
        return "unsigned"
    return "signless"


def _signless_integer_type(type_obj):
    if not IntegerType.isinstance(type_obj):
        raise TypeError(f"expected integer type, got {type_obj}")
    return IntegerType.get_signless(IntegerType(type_obj).width)


def _strip_integer_signedness(value):
    value_type = getattr(value, "type", None)
    if value_type is None or not IntegerType.isinstance(value_type):
        return value
    signless_type = _signless_integer_type(value_type)
    if value_type == signless_type:
        return value
    return UnrealizedConversionCastOp([signless_type], [value]).results[0]


def _restore_integer_signedness(value, target_type):
    if not IntegerType.isinstance(target_type):
        raise TypeError(f"expected integer target type, got {target_type}")
    signless_type = _signless_integer_type(target_type)
    if target_type == signless_type:
        return value
    return UnrealizedConversionCastOp([target_type], [value]).results[0]


def _materialize_integer_literal(target_type, value):
    if not IntegerType.isinstance(target_type):
        raise TypeError(f"unsupported eager integer constructor target type {target_type}")
    signless_type = _signless_integer_type(target_type)
    raw_value = _parse_integer_value(value, target_type=target_type)
    constant = arith.ConstantOp(signless_type, raw_value).result
    if target_type == signless_type:
        return constant
    return _restore_integer_signedness(constant, target_type)


def _parse_integer_value(value, *, target_type=None):
    if isinstance(value, bool):
        raise TypeError("eager scalar constructors do not accept bool values")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        text = value.strip()
        return _parse_integer_text(text)
    raise TypeError(f"cannot materialize {value!r} as an integer constant of type {target_type}")


def _parse_integer_text(text: str):
    if text.startswith(("0x", "0X", "-0x", "-0X")):
        return int(text, 16)
    return int(text, 0)


def _parse_float_attr(target_type, value):
    if isinstance(value, bool):
        raise TypeError("eager scalar constructors do not accept bool values")
    if isinstance(value, str):
        text = value.strip()
        lower = text.lower()
        if lower in {"inf", "+inf", "-inf", "nan"}:
            numeric = float(lower)
        elif text.startswith(("0x", "0X")):
            return _float_attr_from_bit_pattern(target_type, text)
        else:
            numeric = float(text)
    else:
        numeric = float(value)
    return FloatAttr.get(target_type, numeric)


def _float_attr_from_bit_pattern(target_type, text):
    import math
    import struct

    if F16Type.isinstance(target_type):
        bits = int(text, 16) & 0xFFFF
        as_bytes = bits.to_bytes(2, byteorder="little", signed=False)
        numeric = struct.unpack("<e", as_bytes)[0]
        if math.isnan(numeric):
            numeric = float("nan")
        return FloatAttr.get(target_type, numeric)
    if BF16Type.isinstance(target_type):
        bits = int(text, 16) & 0xFFFF
        as_bytes = bits.to_bytes(2, byteorder="little", signed=False) + b"\x00\x00"
        numeric = struct.unpack("<f", as_bytes)[0]
        if math.isnan(numeric):
            numeric = float("nan")
        return FloatAttr.get(target_type, numeric)
    if F32Type.isinstance(target_type):
        bits = int(text, 16) & 0xFFFFFFFF
        as_bytes = bits.to_bytes(4, byteorder="little", signed=False)
        numeric = struct.unpack("<f", as_bytes)[0]
        if math.isnan(numeric):
            numeric = float("nan")
        return FloatAttr.get(target_type, numeric)
    raise TypeError(f"bit-pattern float literals are not supported for {target_type}")


def _normalize_address_space(space):
    if isinstance(space, str):
        return _ADDR_SPACE.get(space)
    if isinstance(space, _pto.AddressSpace):
        return space
    return None


def _int_descriptor(width: int, signedness: str):
    if signedness == "signless":
        return _DType(lambda: IntegerType.get_signless(width))
    if signedness == "signed":
        return _DType(lambda: IntegerType.get_signed(width))
    if signedness == "unsigned":
        return _DType(lambda: IntegerType.get_unsigned(width))
    raise ValueError(f"unsupported integer signedness {signedness!r}")


# ── Scalar dtype singletons ───────────────────────────────────────────────────

float32 = _DType(F32Type.get)
float16 = _DType(F16Type.get)
bf16    = _DType(BF16Type.get)
f8e4m3  = _DType(Float8E4M3FNType.get)
f8e5m2  = _DType(Float8E5M2Type.get)
f8e8m0  = _DType(lambda: _pto.F8E8M0Type.get())
hif8    = _DType(lambda: _pto.HiF8Type.get())
f4e1m2x2 = _DType(lambda: _pto.F4E1M2x2Type.get())
f4e2m1x2 = _DType(lambda: _pto.F4E2M1x2Type.get())
_STORAGE_ONLY_DTYPE_DESCRIPTORS = (
    f8e4m3,
    f8e5m2,
    f8e8m0,
    hif8,
    f4e1m2x2,
    f4e2m1x2,
)
int1    = _int_descriptor(1, "signless")
int8    = _int_descriptor(8, "signless")
int16   = _int_descriptor(16, "signless")
int32   = _int_descriptor(32, "signless")
int64   = _int_descriptor(64, "signless")
si8     = _int_descriptor(8, "signed")
si16    = _int_descriptor(16, "signed")
si32    = _int_descriptor(32, "signed")
si64    = _int_descriptor(64, "signed")
ui8     = _int_descriptor(8, "unsigned")
ui16    = _int_descriptor(16, "unsigned")
ui32    = _int_descriptor(32, "unsigned")
ui64    = _int_descriptor(64, "unsigned")
index   = _DType(IndexType.get)

# ── Packed vector type descriptors ──────────────────────────────────────────

f16x2  = _DType(lambda: VectorType.get([2], F16Type.get()))
bf16x2 = _DType(lambda: VectorType.get([2], BF16Type.get()))
f32x2  = _DType(lambda: VectorType.get([2], F32Type.get()))
f8e4m3x2 = _DType(lambda: VectorType.get([2], Float8E4M3FNType.get()))
f8e4m3x4 = _DType(lambda: VectorType.get([4], Float8E4M3FNType.get()))
f8e4m3x8 = _DType(lambda: VectorType.get([8], Float8E4M3FNType.get()))
f8e5m2x2 = _DType(lambda: VectorType.get([2], Float8E5M2Type.get()))
f8e5m2x4 = _DType(lambda: VectorType.get([4], Float8E5M2Type.get()))
f8e5m2x8 = _DType(lambda: VectorType.get([8], Float8E5M2Type.get()))
hif8x2 = _DType(lambda: _pto.HiF8x2Type.get())
i8x2   = _DType(lambda: VectorType.get([2], IntegerType.get_signless(8)))
i16x2  = _DType(lambda: VectorType.get([2], IntegerType.get_signless(16)))
i32x2  = _DType(lambda: VectorType.get([2], IntegerType.get_signless(32)))

# ``pto.bf16x2`` is the existing builtin/SIMT vector carrier.  VMI uses a
# distinct packed carrier type so that its logical lane count remains the
# number of bf16 pairs rather than the number of scalar bf16 lanes.
_vmi_bf16x2 = _DType(lambda: _pto.BF16x2Type.get())


# ── Type constructor functions ────────────────────────────────────────────────

def ptr(elem, space: str = "ub") -> _PtrDescriptor:
    """Return a lazy descriptor for ``!pto.ptr<elem, space>``."""
    return _PtrDescriptor(elem, space)


def vreg_type(lanes: int, elem) -> _VRegDescriptor:
    """Return a lazy descriptor for ``!pto.vreg<lanesxelem>``."""
    return _VRegDescriptor(lanes, elem)


def vec_type(elem, size: int) -> _VecDescriptor:
    """Return a lazy descriptor for builtin ``vector<size x elem>`` values."""
    return _VecDescriptor(elem, size)


def mask_type(bits: str = "b32") -> _MaskDescriptor:
    """Return a lazy descriptor for ``!pto.mask<bits>``."""
    return _MaskDescriptor(bits)


def struct_type(*field_types) -> _StructDescriptor:
    """Return a lazy descriptor for ``!pto.struct<field_types...>``."""
    if not field_types:
        raise ValueError("pto.struct_type(...) requires at least one field")
    return _StructDescriptor(field_types)


def vmi_vreg_type(lanes: int, elem) -> _VMIVRegDescriptor:
    """Return a lazy descriptor for ``!pto.vmi.vreg<lanesxelem>``."""
    return _VMIVRegDescriptor(lanes, elem)


def vmi_mask_type(lanes: int) -> _VMIMaskDescriptor:
    """Return a lazy descriptor for ``!pto.vmi.mask<lanesxpred>``."""
    return _VMIMaskDescriptor(lanes)


def tile_buf_type(shape, dtype, valid_shape=None, *,
                  blayout: str = "RowMajor",
                  address_space: str = "ub",
                  slayout: str = "NoneBox",
                  fractal_size: int = 512,
                  pad: str = "Null",
                  compact_mode: str | int = "Null") -> Type:
    """
    Construct a ``!pto.tile_buf<…>`` type via the Python bindings.

    ``valid_shape`` entries may be ``-1`` for dynamic (``?``) dimensions.
    When ``valid_shape`` is omitted, construct a tile type without a ``valid=``
    suffix and let the type's logical shape stand on its own.
    ``blayout="ColMajor"`` prints as ``blayout=col_major``.

    Requires an active MLIR context.
    """
    elem = _ensure_tensor_storage_dtype(dtype, context="pto.tile_buf_type(...)")
    space_enum = _normalize_address_space(address_space)
    if space_enum is None:
        raise ValueError(
            f"Unknown address_space '{address_space}'; known: {list(_ADDR_SPACE)}"
        )
    space_attr = _pto.AddressSpaceAttr.get(space_enum)
    cfg = _pto.TileBufConfigAttr.get(
        _pto.BLayoutAttr.get(getattr(_pto.BLayout, blayout)),
        _pto.SLayoutAttr.get(getattr(_pto.SLayout, slayout)),
        fractal_size,
        _pto.PadValueAttr.get(getattr(_pto.PadValue, pad)),
        compact_mode=_compact_mode_token(compact_mode),
    )
    if valid_shape is None and cfg is None:
        return _pto.TileBufType.get(shape, elem, space_attr)
    if valid_shape is None:
        return _pto.TileBufType.get(shape, elem, space_attr, config=cfg)
    if cfg is None:
        return _pto.TileBufType.get(shape, elem, space_attr, valid_shape=valid_shape)
    return _pto.TileBufType.get(shape, elem, space_attr, valid_shape=valid_shape, config=cfg)


def _normalize_compact_mode(value: str | int):
    if isinstance(value, int):
        return value
    aliases = {
        "null": "Null",
        "normal": "Normal",
        "row_plus_one": "RowPlusOne",
        "Null": "Null",
        "Normal": "Normal",
        "RowPlusOne": "RowPlusOne",
    }
    return aliases.get(str(value), str(value))


def _compact_mode_token(value: str | int):
    token = _normalize_compact_mode(value)
    if isinstance(token, int):
        return token
    try:
        return getattr(_pto.CompactMode, token)
    except AttributeError as exc:
        raise ValueError(f"Unknown compact_mode {value!r}") from exc


def tensor_view_type(rank: int, elem) -> Type:
    """``!pto.tensor_view<?x…xelem>`` with *rank* all-dynamic dims."""
    return _pto.TensorViewType.get(rank, _ensure_tensor_storage_dtype(elem, context="pto.tensor_view_type(...)"))


def tensor_view_type_from_dims(dims, elem) -> Type:
    """``!pto.tensor_view<d0x…xdN x elem>`` when every dimension is static."""
    resolved_elem = _ensure_tensor_storage_dtype(elem, context="pto.tensor_view_type_from_dims(...)")
    if all(isinstance(dim, int) for dim in dims):
        return _pto.TensorViewType.get(list(dims), resolved_elem)
    return tensor_view_type(len(dims), resolved_elem)


def part_tensor_view_type(rank: int, elem) -> Type:
    """``!pto.partition_tensor_view<?x…xelem>`` with *rank* all-dynamic dims."""
    kDynamic = ShapedType.get_dynamic_size()
    return _pto.PartitionTensorViewType.get(
        [kDynamic] * rank,
        _ensure_tensor_storage_dtype(elem, context="pto.part_tensor_view_type(...)"),
    )


def part_tensor_view_type_from_dims(dims, elem) -> Type:
    """``!pto.partition_tensor_view<d0x…xdN x elem>`` when every dimension is static."""
    resolved_elem = _ensure_tensor_storage_dtype(elem, context="pto.part_tensor_view_type_from_dims(...)")
    if all(isinstance(dim, int) for dim in dims):
        return _pto.PartitionTensorViewType.get(list(dims), resolved_elem)
    return part_tensor_view_type(len(dims), resolved_elem)


__all__ = [
    "_DType", "_resolve",
    "float32", "float16", "bf16",
    "f16x2", "bf16x2", "f32x2",
    "f8e4m3x2", "f8e4m3x4", "f8e4m3x8",
    "f8e5m2x2", "f8e5m2x4", "f8e5m2x8", "hif8x2",
    "i8x2", "i16x2", "i32x2",
    "f8e4m3", "f8e5m2", "f8e8m0", "hif8", "f4e1m2x2", "f4e2m1x2",
    "int1", "int8", "int16", "int32", "int64",
    "si8", "si16", "si32", "si64",
    "ui8", "ui16", "ui32", "ui64",
    "index",
    "ptr", "vreg_type", "vec_type", "mask_type", "struct_type",
    "vmi_vreg_type", "vmi_mask_type",
    "tile_buf_type", "tensor_view_type", "tensor_view_type_from_dims",
    "part_tensor_view_type", "part_tensor_view_type_from_dims",
]
