# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Shared private helpers for the PTODSL op modules: mode gates,
normalizers, shared emitters, and the coercion/type-inference hub."""

from functools import wraps
import warnings
from ._diagnostics import (
    PTODSLDeprecationWarning,
    explicit_mode_required_with_context_error,
    make_tensor_view_invalid_layout_error,
    make_tensor_view_missing_metadata_error,
    tile_row_alignment_error,
)
from ._host_tensors import resolve_tensor_data_entry
from ._scalar_coercion import coerce_scalar_to_type, materialize_scalar_literal
from ._scalar_adaptation import (
    classify_runtime_scalar_type,
    coerce_runtime_i1_value,
    coerce_runtime_index_value,
    coerce_runtime_integer_value,
)
from ._runtime_scalar_ops import emit_runtime_binary_op
from ._surface_values import (
    AllocatedBufferValue,
    MaskResultValue,
    PartitionTensorViewValue,
    TensorViewValue,
    TileSliceValue,
    TileValue,
    _coerce_index_value,
    _static_index_dims,
    _unwrap_sequence,
    compose_partition_spec,
    emit_as_ptr,
    infer_tile_element_type,
    is_runtime_scalar_ir_type,
    parse_tile_type_metadata,
    resolve_address_access,
    unwrap_surface_value,
    wrap_surface_value,
)
from ._types import (
    _is_struct_type,
    _isinstance_pto_type,
    _materialize_integer_literal,
    _normalize_address_space,
    _resolve,
    _strip_integer_signedness,
    mask_type,
    part_tensor_view_type,
    part_tensor_view_type_from_dims,
    ptr,
    tensor_view_type,
    tensor_view_type_from_dims,
    vreg_type,
)
from ptoas.mlir.dialects import arith, pto as _pto
from ptoas.mlir.ir import (
    Attribute,
    BF16Type,
    F16Type,
    F32Type,
    Float8E4M3FNType,
    Float8E5M2Type,
    FloatAttr,
    IndexType,
    IntegerAttr,
    IntegerType,
    MemRefType,
    Operation,
    Type,
    TypeAttr,
    UnitAttr,
    VectorType,
)



_PIPE_ALIASES = {
    "MTE1": "PIPE_MTE1",
    "MTE2": "PIPE_MTE2",
    "MTE3": "PIPE_MTE3",
    "MTE4": "PIPE_MTE4",
    "V":    "PIPE_V",
    "M":    "PIPE_M",
    "S":    "PIPE_S",
    "ALL":  "PIPE_ALL",
}


def _pipe_attr(name: str):
    if not isinstance(name, str):
        return _pto.PipeAttr.get(name)
    canonical = _PIPE_ALIASES.get(name, name)
    if not canonical.startswith("PIPE_"):
        canonical = "PIPE_" + canonical
    return _pto.PipeAttr.get(getattr(_pto.PIPE, canonical))


def _event_attr(event_id: int):
    return getattr(_pto, f"EVENT_ID{event_id}")


def _canonical_pipe_token(pipe):
    if isinstance(pipe, str):
        canonical = _PIPE_ALIASES.get(pipe, pipe)
        if not canonical.startswith("PIPE_"):
            canonical = "PIPE_" + canonical
        return canonical

    for canonical in (
        "PIPE_FIX", "PIPE_MTE1", "PIPE_MTE2", "PIPE_MTE3", "PIPE_MTE4",
        "PIPE_V", "PIPE_M", "PIPE_S", "PIPE_V2", "PIPE_ALL",
    ):
        pipe_attr = getattr(_pto.PIPE, canonical, None)
        if pipe_attr is not None and pipe == pipe_attr:
            return canonical
    return None


def _validate_static_event_id(event_id, *, context: str):
    if isinstance(event_id, bool):
        raise TypeError(f"{context} does not accept bool values")
    if isinstance(event_id, int) and not 0 <= event_id <= 7:
        raise ValueError(f"{context} expects static event_id in [0, 7], got {event_id}")


def _validate_static_buf_id(buf_id, *, context: str):
    if isinstance(buf_id, bool):
        raise TypeError(f"{context} does not accept bool values")
    if isinstance(buf_id, int) and not 0 <= buf_id <= 31:
        raise ValueError(f"{context} expects static buf_id in [0, 31], got {buf_id}")


def _validate_static_event_id_range(event_id, *, context: str, lo: int, hi: int, meaning: str = "event_id"):
    if isinstance(event_id, bool):
        raise TypeError(f"{context} does not accept bool values")
    if isinstance(event_id, int) and not lo <= event_id <= hi:
        raise ValueError(f"{context} expects static {meaning} in [{lo}, {hi}], got {event_id}")


def _validate_sync_pipe(pipe, *, context: str, allowed: tuple[str, ...]):
    canonical = _canonical_pipe_token(pipe)
    if canonical is None:
        raise TypeError(f"{context} expects a concrete Pipe value, got {pipe!r}")
    if canonical not in allowed:
        expected = ", ".join(f"<{name}>" for name in allowed)
        raise ValueError(f"{context} expects pipe to be one of {expected}, got <{canonical}>")
    return canonical


def _require_explicit_mode(surface: str):
    try:
        from ._tracing.active import current_session
        session = current_session()
    except Exception:
        session = None
    if session is None:
        return
    current_module_spec = getattr(session, "current_function_module_spec", session.module_spec)
    current_mode = getattr(current_module_spec, "mode", None)
    if current_mode != "explicit":
        raise explicit_mode_required_with_context_error(surface, current_module_spec)


def _current_target_arch():
    try:
        from ._tracing.active import current_session
        session = current_session()
    except Exception:
        return None
    if session is None:
        return None
    current_module_spec = getattr(session, "current_function_module_spec", session.module_spec)
    return getattr(current_module_spec, "target_arch", None)


def _require_target_arch(surface: str, allowed: set[str]):
    target = _current_target_arch()
    if target is None:
        return
    normalized = str(target).lower()
    if normalized not in allowed:
        expected = ", ".join(f"target='{name}'" for name in sorted(allowed))
        raise ValueError(f"{surface} is only supported for {expected}; got target={target!r}")


def _require_simt_subkernel(surface: str):
    try:
        from ._tracing.active import current_session
        session = current_session()
    except Exception:
        session = None
    frame = session.current_subkernel if session is not None else None
    if frame is not None and frame.role == "simt":
        return
    current = "top-level @pto.jit body" if frame is None else f"@pto.{frame.role}"
    raise RuntimeError(
        f"{surface} may only be used inside a @pto.simt helper or inline pto.simt() scope; "
        f"current scope is {current}."
    )


def _explicit_mode_only(surface: str):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            _require_explicit_mode(surface)
            return fn(*args, **kwargs)

        return wrapper

    return decorator


# ── Constants ────────────────────────────────────────────────────────────────

def castptr(int_addr, result_ptr_type):
    """``pto.castptr`` – cast an integer address to a typed PTO pointer."""
    return wrap_surface_value(
        _pto.CastPtrOp(_resolve(result_ptr_type), unwrap_surface_value(int_addr)).result
    )


def addptr(base_ptr, index_offset):
    """``pto.addptr`` – advance a pointer by an index offset."""
    return wrap_surface_value(
        _pto.AddPtrOp(
            unwrap_surface_value(base_ptr),
            _coerce_index(index_offset, context="addptr(ptr, offset)"),
        ).result
    )


# ── Vector load / store ───────────────────────────────────────────────────────

def vbitcast(vector_value, to_dtype):
    """``pto.vbitcast`` – reinterpret one vector register as a different element type."""
    target_elem = _resolve(to_dtype)
    target_type = _resolve(vreg_type(_elements_per_vreg(target_elem), target_elem))
    return wrap_surface_value(
        _pto.VbitcastOp(
            target_type,
            unwrap_surface_value(vector_value),
        ).result
    )

def _normalize_vcvt_round_mode(mode, *, context: str):
    token = mode
    if not isinstance(token, str):
        token = str(token)
        if "." in token:
            token = token.rsplit(".", 1)[-1]
    normalized = token.strip().upper()
    allowed = {"R", "A", "F", "C", "Z", "O", "H"}
    if normalized not in allowed:
        expected = ", ".join(sorted(allowed))
        raise ValueError(f"{context} does not support rnd {mode!r}; expected one of {expected}")
    return normalized

def _normalize_enum_attr(value, *, enum_cls, attr_cls, context: str):
    if value is None or isinstance(value, Attribute):
        return value
    token = value if isinstance(value, str) else getattr(value, "name", None)
    if token is not None:
        normalized = "".join(ch for ch in str(token) if ch.isalnum()).lower()
        candidates = [
            name for name in dir(enum_cls)
            if not name.startswith("_")
            and "".join(ch for ch in name if ch.isalnum()).lower() == normalized
        ]
        if not candidates:
            allowed = ", ".join(name for name in dir(enum_cls) if not name.startswith("_"))
            raise ValueError(f"{context} does not support {value!r}; expected one of {allowed}")
        value = getattr(enum_cls, candidates[0])
    return attr_cls.get(value)

_UNSET = object()


def _resolve_precision_argument(precision, legacy_precision, *, legacy_name: str, context: str):
    """Normalize a deprecated operation-specific precision keyword."""
    if legacy_precision is _UNSET:
        return precision
    if precision is not None:
        raise TypeError(f"{context} accepts either precision or {legacy_name}, not both")
    warnings.warn(
        f"{legacy_name} is deprecated; use precision instead",
        PTODSLDeprecationWarning,
        stacklevel=3,
    )
    return legacy_precision


def _normalize_vpack_part(part, *, context: str):
    token = part
    if not isinstance(token, str):
        token = str(token)
        if "." in token:
            token = token.rsplit(".", 1)[-1]
    normalized = token.strip().upper()
    allowed = {"LOWER", "HIGHER"}
    if normalized not in allowed:
        expected = ", ".join(sorted(allowed))
        raise ValueError(f"{context} does not support part {part!r}; expected one of {expected}")
    return normalized

_MASK_PATTERN_TOKENS = {
    "PAT_ALL",
    "PAT_ALLF",
    "PAT_H",
    "PAT_Q",
    "PAT_M3",
    "PAT_M4",
    *(f"PAT_VL{count}" for count in range(1, 129)),
}

_TILE_MASK_PATTERN_TOKENS = {
    "P0101",
    "P1010",
    "P0001",
    "P0010",
    "P0100",
    "P1000",
    "P1111",
}

_CMP_MODE_TOKENS = {"eq", "ne", "lt", "le", "gt", "ge"}
_PREDICATE_PART_TOKENS = {"LOWER", "HIGHER"}
_PREDICATE_LOAD_DIST_TOKENS = {"NORM", "US", "DS"}
_PREDICATE_STORE_DIST_TOKENS = {"NORM", "PK"}
_POST_UPDATE_TOKENS = {"NO_POST_UPDATE", "POST_UPDATE"}


def _normalize_mask_pattern(pattern):
    token = pattern
    if not isinstance(token, str):
        token = str(token)
        if "." in token:
            token = token.rsplit(".", 1)[-1]
    token = token.strip().upper()
    normalized = token if token.startswith("PAT_") else f"PAT_{token}"
    if normalized not in _MASK_PATTERN_TOKENS:
        raise ValueError(
            f"unsupported mask pattern {pattern!r}; expected one of PAT_ALL, PAT_ALLF, "
            "PAT_H, PAT_Q, PAT_VL1..PAT_VL128, PAT_M3, PAT_M4"
        )
    return normalized


def _normalize_tile_mask_pattern(pattern):
    token = pattern
    if not isinstance(token, str):
        token = str(token)
        if "." in token:
            token = token.rsplit(".", 1)[-1]
    token = token.strip().upper()
    if token not in _TILE_MASK_PATTERN_TOKENS:
        raise ValueError(
            f"unsupported tile mask pattern {pattern!r}; expected one of "
            "P0101, P1010, P0001, P0010, P0100, P1000, P1111"
        )
    return token


def _tile_mask_pattern_attr(pattern):
    token = _normalize_tile_mask_pattern(pattern)
    return _pto.MaskPatternAttr.get(getattr(_pto.MaskPattern, token))


def _normalize_cmp_mode(cmp_mode):
    token = cmp_mode
    if not isinstance(token, str):
        token = str(token)
        if "." in token:
            token = token.rsplit(".", 1)[-1]
    normalized = token.strip().lower()
    if normalized not in _CMP_MODE_TOKENS:
        raise ValueError(
            f"unsupported cmp_mode {cmp_mode!r}; expected one of EQ, NE, LT, LE, GT, GE"
        )
    return normalized


def _cmp_mode_attr(cmp_mode):
    return Attribute.parse(f"#pto<cmp {_normalize_cmp_mode(cmp_mode)}>")


def _normalize_predicate_part(part):
    token = part
    if not isinstance(token, str):
        token = str(token)
        if "." in token:
            token = token.rsplit(".", 1)[-1]
    normalized = token.strip().upper()
    if normalized not in _PREDICATE_PART_TOKENS:
        raise ValueError(f"unsupported predicate part {part!r}; expected LOWER or HIGHER")
    return normalized


def _normalize_predicate_dist(dist, *, allowed: set[str], context: str):
    token = dist
    if not isinstance(token, str):
        token = str(token)
        if "." in token:
            token = token.rsplit(".", 1)[-1]
    normalized = token.strip().upper()
    if normalized not in allowed:
        expected = ", ".join(sorted(allowed))
        raise ValueError(f"{context} does not support dist {dist!r}; expected one of {expected}")
    return normalized


def _normalize_post_update_mode(mode, *, context: str):
    token = mode
    if not isinstance(token, str):
        token = str(token)
        if "." in token:
            token = token.rsplit(".", 1)[-1]
    normalized = token.strip().upper()
    if normalized in {"OFF", "NO_POST_UPDATE"}:
        return "NO_POST_UPDATE"
    if normalized in {"ON", "POST_UPDATE"}:
        return "POST_UPDATE"
    expected = ", ".join(sorted(_POST_UPDATE_TOKENS))
    raise ValueError(f"{context} does not support mode {mode!r}; expected one of ON/OFF ({expected})")


def _mask_type_from_bits(mask_bits: int):
    return _resolve(mask_type(f"b{mask_bits}"))


def _resolve_mask_result_type(result_type, *, context: str):
    if result_type is None:
        return None
    resolved = _resolve(result_type)
    try:
        _pto.MaskType(resolved)
    except Exception as exc:
        raise TypeError(f"{context} expects to_type to resolve to a PTO mask type, got {resolved}") from exc
    return resolved


def _infer_mask_metadata(mask_value, *, context: str):
    raw_type = unwrap_surface_value(mask_value).type
    try:
        mask_ty = _pto.MaskType(raw_type)
    except Exception as exc:
        raise TypeError(f"{context} expects a PTO mask value, got {raw_type}") from exc
    granularity = mask_ty.granularity
    return int(granularity[1:]), raw_type


def _require_same_mask_types(values, *, context: str):
    raw_types = [unwrap_surface_value(value).type for value in values]
    first = raw_types[0]
    for other in raw_types[1:]:
        if other != first:
            raise TypeError(f"{context} expects masks of the same granularity, got {first} and {other}")
    return first


def _pointer_element_type(ptr_value, *, context: str):
    raw_type = unwrap_surface_value(ptr_value).type
    try:
        return _pto.PtrType(raw_type).element_type
    except Exception:
        try:
            return MemRefType(raw_type).element_type
        except Exception as exc:
            raise TypeError(f"{context} expects a PTO pointer or memref-backed address, got {raw_type}") from exc


def _coerce_index(value, *, context: str):
    raw_value = unwrap_surface_value(value)
    try:
        return coerce_runtime_index_value(raw_value, context=context)
    except TypeError as exc:
        if hasattr(raw_value, "type"):
            raise TypeError(f"{context} expects an index-like scalar, got {raw_value.type}") from exc
        raise


def init_align():
    """``pto.init_align`` – materialize the initial alignment state."""
    return wrap_surface_value(_pto.InitAlignOp(_pto.AlignType.get()).result)

def _emit_unary_vec_op(op_ctor, inp, mask):
    _reject_low_precision_vreg_operands(inp, context=f"pto.{_surface_name_for_op_ctor(op_ctor)}(...)")
    return wrap_surface_value(
        op_ctor(
            unwrap_surface_value(inp).type,
            unwrap_surface_value(inp),
            unwrap_surface_value(mask),
        ).result
    )


def _emit_binary_vec_op(op_ctor, lhs, rhs, mask):
    _reject_low_precision_vreg_operands(
        lhs,
        rhs,
        context=f"pto.{_surface_name_for_op_ctor(op_ctor)}(...)",
    )
    return wrap_surface_value(
        op_ctor(
            unwrap_surface_value(lhs).type,
            unwrap_surface_value(lhs),
            unwrap_surface_value(rhs),
            unwrap_surface_value(mask),
        ).result
    )


def _normalize_shift_count_vector(rhs, *, context: str):
    """Reinterpret an integer shift-count vector as its signed type."""
    raw_type = unwrap_surface_value(rhs).type
    # Preserve generated-wrapper dispatch tests, which intentionally use mock
    # values instead of MLIR SSA values. Real PTODSL values always carry Type.
    if not isinstance(raw_type, Type):
        return rhs
    _, element_type = _infer_vreg_metadata(rhs)
    if not IntegerType.isinstance(element_type):
        raise TypeError(f"{context} requires an integer shift-count vector")
    integer_type = IntegerType(element_type)
    signed_type = IntegerType.get_signed(integer_type.width)
    if element_type == signed_type:
        return rhs
    return vbitcast(rhs, signed_type)


def _emit_shift_vec_op(op_ctor, lhs, rhs, mask):
    context = f"pto.{_surface_name_for_op_ctor(op_ctor)}(...)"
    _reject_low_precision_vreg_operands(lhs, rhs, context=context)
    normalized_rhs = _normalize_shift_count_vector(rhs, context=context)
    return wrap_surface_value(
        op_ctor(
            unwrap_surface_value(lhs).type,
            unwrap_surface_value(lhs),
            unwrap_surface_value(normalized_rhs),
            unwrap_surface_value(mask),
        ).result
    )


def _emit_vec_scalar_masked_op(op_ctor, inp, scalar, mask, *, context: str):
    _reject_low_precision_vreg_operands(inp, context=f"pto.{context}(...)")
    scalar_value = _coerce_scalar_like_vector_element(inp, scalar, context=context)
    return wrap_surface_value(
        op_ctor(
            unwrap_surface_value(inp).type,
            unwrap_surface_value(inp),
            unwrap_surface_value(scalar_value),
            unwrap_surface_value(mask),
        ).result
    )

def _normalize_vdup_position_mode(position, *, context: str):
    token = position
    if not isinstance(token, str):
        token = str(token)
        if "." in token:
            token = token.rsplit(".", 1)[-1]
    normalized = token.strip().upper()
    allowed = {"LOWEST", "HIGHEST"}
    if normalized not in allowed:
        expected = ", ".join(sorted(allowed))
        raise ValueError(f"{context} does not support position {position!r}; expected one of {expected}")
    return normalized


def _normalize_acc_to_vec_mode(mode, *, context: str):
    if mode is None:
        return None
    if isinstance(mode, str):
        token = mode.strip().lower()
        aliases = {
            "vec0": "single_mode_vec0",
            "vec1": "single_mode_vec1",
            "split_m": "dual_mode_split_m",
            "split_n": "dual_mode_split_n",
            "single_mode_vec0": "single_mode_vec0",
            "single_mode_vec1": "single_mode_vec1",
            "dual_mode_split_m": "dual_mode_split_m",
            "dual_mode_split_n": "dual_mode_split_n",
        }
        normalized = aliases.get(token)
        if normalized is None:
            expected = ", ".join(sorted(aliases))
            raise ValueError(f"{context} expects mode to be one of {expected}, got {mode!r}")
        return Attribute.parse(f"#pto<acc_to_vec_mode {normalized}>")
    return mode

def as_ptr(value):
    """Materialize a typed pointer from a tile or tensor-view descriptor."""
    wrapped = wrap_surface_value(value)
    return emit_as_ptr(wrapped)


def _constant_like(value, mlir_type):
    value = unwrap_surface_value(value)
    if hasattr(value, "type"):
        return value
    if isinstance(value, float):
        return arith.ConstantOp(mlir_type, FloatAttr.get(mlir_type, value)).result
    if IntegerType.isinstance(mlir_type):
        return _materialize_integer_literal(mlir_type, value)
    return arith.ConstantOp(mlir_type, value).result


def _index_zero():
    return arith.ConstantOp(IndexType.get(), 0).result


def _tile_slice_linear_offset(tile_slice: TileSliceValue):
    offsets = tile_slice.offsets
    if len(offsets) == 1:
        return offsets[0]
    if len(offsets) != 2:
        raise RuntimeError("tile slice pointer lowering only supports rank-1 or rank-2 offsets")

    physical_shape = getattr(tile_slice.tile, "physical_shape", None)
    if physical_shape is None or len(physical_shape) != 2 or physical_shape[1] is None:
        raise RuntimeError("tile slice pointer lowering requires static physical column shape metadata")

    row, col = offsets
    stride = physical_shape[1]
    if isinstance(row, int) and isinstance(col, int):
        return row * stride + col

    row_value = _coerce_index(row, context="tile slice pointer lowering")
    row_stride = arith.MulIOp(row_value, arith.ConstantOp(IndexType.get(), stride).result).result
    col_value = _coerce_index(col, context="tile slice pointer lowering")
    return arith.AddIOp(row_stride, col_value).result


def _tile_slice_ptr(tile_slice: TileSliceValue):
    base_ptr = emit_as_ptr(tile_slice.tile)
    linear_offset = _tile_slice_linear_offset(tile_slice)
    if isinstance(linear_offset, int) and linear_offset == 0:
        return base_ptr
    return addptr(base_ptr, _coerce_index(linear_offset, context="tile slice pointer lowering"))


def _tile_slice_address(tile_slice: TileSliceValue):
    base_ptr = emit_as_ptr(tile_slice.tile)
    linear_offset = _tile_slice_linear_offset(tile_slice)
    return base_ptr, _coerce_index(linear_offset, context="tile slice pointer lowering")


def _infer_vreg_type_from_tile_slice(tile_slice: TileSliceValue):
    elem_type = infer_tile_element_type(tile_slice.tile)
    lanes = _elements_per_vreg(elem_type)
    return _resolve(vreg_type(lanes, elem_type))


def _infer_vreg_type_from_address_source(src_ptr):
    raw_source = unwrap_surface_value(src_ptr)
    source_type = raw_source.type
    try:
        elem_type = _pto.PtrType(source_type).element_type
    except Exception:
        try:
            elem_type = MemRefType(source_type).element_type
        except Exception as exc:
            raise TypeError(
                f"vlds(ptr, offset) cannot infer a vector-register type from source {source_type}; "
                "pass result_vreg_type= explicitly"
            ) from exc
    lanes = _elements_per_vreg(elem_type)
    return _resolve(vreg_type(lanes, elem_type))


def _elements_per_vreg(elem_type):
    try:
        bytewidth = _element_bytewidth(elem_type)
    except TypeError as exc:
        raise TypeError(f"vlds/vsts tile-slice sugar does not support element type {elem_type}")
    return 256 // bytewidth


def _infer_vreg_metadata(vector_value):
    raw_type = unwrap_surface_value(vector_value).type
    try:
        vreg_type = _pto.VRegType(raw_type)
        return vreg_type.lanes, vreg_type.element_type
    except Exception:
        text = str(raw_type)
        if not text.startswith("!pto.vreg<") or "x" not in text:
            raise TypeError(f"expected PTO vector-register type, got {raw_type}")
        body = text[len("!pto.vreg<"):-1]
        lanes_text, elem_text = body.split("x", 1)
        return int(lanes_text), Type.parse(elem_text)


def _surface_name_for_op_ctor(op_ctor) -> str:
    name = getattr(op_ctor, "__name__", "")
    if name.startswith("V") and name.endswith("Op"):
        return name[1:-2].lower()
    return name or "vector_op"


def _is_low_precision_elem_type(elem_type) -> bool:
    if Float8E4M3FNType.isinstance(elem_type) or Float8E5M2Type.isinstance(elem_type):
        return True
    return any(
        _isinstance_pto_type(elem_type, name)
        for name in ("HiF8Type", "F4E1M2x2Type", "F4E2M1x2Type")
    )


def _reject_low_precision_vreg(value, *, context: str) -> None:
    raw_value = unwrap_surface_value(value)
    try:
        _, elem_type = _infer_vreg_metadata(raw_value)
    except TypeError:
        return
    if _is_low_precision_elem_type(elem_type):
        raise TypeError(
            f"{context} does not support low-precision vreg elements yet; "
            "low-precision vregs are currently only supported on explicit memory/conversion paths such as "
            "vlds/vsts/vcvt/vmulscvt/vpack"
        )


def _reject_low_precision_vreg_operands(*values, context: str) -> None:
    for value in values:
        _reject_low_precision_vreg(value, context=context)


def _element_bytewidth(elem_type):
    if F32Type.isinstance(elem_type):
        return 4
    if any(cls.isinstance(elem_type) for cls in (F16Type, BF16Type)):
        return 2
    if Float8E4M3FNType.isinstance(elem_type) or Float8E5M2Type.isinstance(elem_type):
        return 1
    if any(_isinstance_pto_type(elem_type, name) for name in ("HiF8Type", "F4E1M2x2Type", "F4E2M1x2Type")):
        return 1
    if IntegerType.isinstance(elem_type):
        width = IntegerType(elem_type).width
        if width % 8 != 0:
            raise TypeError(f"unsupported sub-byte integer element type {elem_type}")
        return width // 8
    raise TypeError(f"unsupported element type {elem_type}")


def bytewidth(dtype):
    """Return the size in bytes of one element of *dtype*."""
    return _element_bytewidth(_resolve(dtype))


def elements_per_vreg(dtype):
    """Return how many elements of *dtype* fit in one 256-byte vector register."""
    return _elements_per_vreg(_resolve(dtype))


def _mask_bits_for_dtype(dtype):
    elem_type = _resolve(dtype)
    bytewidth = _element_bytewidth(elem_type)
    if bytewidth == 4:
        return 32
    if bytewidth == 2:
        return 16
    if bytewidth == 1:
        return 8
    raise TypeError(f"make_mask(...) does not support dtype {elem_type}")


def _pset_op_for_mask_bits(mask_bits: int):
    return {
        8: _pto.PsetB8Op,
        16: _pto.PsetB16Op,
        32: _pto.PsetB32Op,
    }[mask_bits]


def _plt_op_for_mask_bits(mask_bits: int):
    return {
        8: _pto.PltB8Op,
        16: _pto.PltB16Op,
        32: _pto.PltB32Op,
    }[mask_bits]


def _coerce_i32(value, *, context: str):
    raw_value = unwrap_surface_value(value)
    return coerce_runtime_integer_value(raw_value, IntegerType.get_signless(32), context=context)


def _coerce_i64(value, *, context: str):
    raw_value = unwrap_surface_value(value)
    return coerce_runtime_integer_value(raw_value, IntegerType.get_signless(64), context=context)


def _validate_raw_fill_l1_static_scalar(
        value, *, context: str, maximum=None, alignment=None):
    if isinstance(value, bool):
        raise TypeError(f"{context} does not accept bool values")
    if not isinstance(value, int):
        return
    if value < 0:
        raise ValueError(f"{context} expects a non-negative static value, got {value}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{context} expects a static value in [0, {maximum}], got {value}")
    if alignment is not None and value % alignment != 0:
        raise ValueError(f"{context} expects a static value aligned to {alignment} bytes, got {value}")


def _validate_raw_fill_l1_fill_word_bits(fill_word_bits, *, context: str):
    if isinstance(fill_word_bits, bool) or not isinstance(fill_word_bits, int):
        raise TypeError(f"{context} expects a static int fill_word_bits, got {fill_word_bits!r}")
    if fill_word_bits not in (16, 32):
        raise ValueError(f"{context} expects fill_word_bits 16 or 32, got {fill_word_bits}")


def _coerce_i1(value, *, context: str):
    raw_value = unwrap_surface_value(value)
    return coerce_runtime_i1_value(raw_value, context=context)


def _i64_zero():
    return arith.ConstantOp(IntegerType.get_signless(64), 0).result


def _coerce_scalar_like_vector_element(vector_value, scalar_value, *, context: str):
    _, elem_type = _infer_vreg_metadata(vector_value)
    return coerce_scalar_to_type(scalar_value, elem_type, context=f"{context}(...)")


def _negate_runtime_scalar(value):
    raw_value = unwrap_surface_value(value)
    kind = classify_runtime_scalar_type(raw_value.type)
    zero = materialize_scalar_literal(
        0.0 if kind == "float" else 0, raw_value.type,
        context="_negate_runtime_scalar(...)")
    return emit_runtime_binary_op("sub", zero, raw_value)


def _mul_bytes(value, elem_type):
    factor = _element_bytewidth(_resolve(elem_type))
    raw_value = unwrap_surface_value(value)
    if isinstance(raw_value, int):
        return raw_value * factor
    return emit_runtime_binary_op("mul", raw_value, factor)


def _membar_attr(kind: str):
    normalized = str(kind)
    supported = {
        "VV_ALL",
        "VST_VLD",
        "VLD_VST",
        "VST_VST",
        "VS_ALL",
        "VST_LD",
        "VLD_ST",
        "VST_ST",
        "SV_ALL",
        "ST_VLD",
        "LD_VST",
        "ST_VST",
        "SS_ALL",
        "ST_LD",
        "LD_ST",
        "ST_ST",
    }
    if normalized not in supported:
        raise ValueError(f"unsupported mem_bar kind {kind!r}")
    return Attribute.parse(f"#pto.membar<{normalized}>")


def _normalize_token(value, *, context: str):
    token = getattr(value, "value", value)
    if not isinstance(token, str):
        token = str(token)
        if "." in token:
            token = token.rsplit(".", 1)[-1]
    return token.strip().lower()


def _normalize_sat_mode(sat, *, context: str, allow_preserve_nan: bool):
    normalized = _normalize_token(sat, context=context)
    aliases = {
        "on": "sat",
        "off": "nosat",
        "preserve_nan": "sat_preserve_nan",
        "sat": "sat",
        "nosat": "nosat",
        "sat_preserve_nan": "sat_preserve_nan",
        "sat(preserve_nan)": "sat_preserve_nan",
    }
    token = aliases.get(normalized)
    if token is None:
        expected = "on/off" + ("/preserve_nan" if allow_preserve_nan else "")
        raise ValueError(f"{context} does not support {sat!r}; expected {expected}")
    if token == "sat_preserve_nan" and not allow_preserve_nan:
        raise ValueError(f"{context} does not support preserve_nan saturation")
    return token


def _enum_attr(kind, value, *, supported: set[str], context: str):
    normalized = _normalize_token(value, context=context)
    if normalized not in supported:
        expected = ", ".join(sorted(supported))
        raise ValueError(f"{context} does not support {value!r}; expected one of {expected}")
    return Attribute.parse(f"#pto<{kind} {normalized}>")

_SIGNEDNESS_TOKENS = {"signed", "unsigned"}
_L1_CACHE_TOKENS = {"cache", "uncache"}
_LD_L2_CACHE_TOKENS = {
    "nmfv", "nmlv", "nmprs", "nmpref",
    "nakeep", "naclean", "nadrop",
    "idsfv", "idslv", "idsprs", "idspref",
    "exfv", "exlv", "exprs", "expref",
}
_ST_L2_CACHE_TOKENS = {
    "nmfv", "nmlv", "nmprs", "nmred",
    "naci", "napw", "napi", "nared",
    "wbhfv", "wbhlv", "wbhprs", "wbhred",
    "wtsfv", "wtslv", "wtsprs", "wtsred",
}
_ST_L2_CACHE_CONTROL_VALUES = {
    "nmfv": 0,
    "nmlv": 1,
    "nmprs": 2,
    "nmred": 3,
    "naci": 4,
    "napw": 5,
    "napi": 6,
    "nared": 7,
    "wbhfv": 8,
    "wbhlv": 9,
    "wbhprs": 10,
    "wbhred": 11,
    "wtsfv": 12,
    "wtslv": 13,
    "wtsprs": 14,
    "wtsred": 15,
}
_ROUNDING_TOKENS = {"r", "a", "f", "c", "z", "o", "h"}
_SATURATION_TOKENS = {"sat", "nosat"}


def _optional_signedness_attr(signedness, *, context: str):
    if signedness is None:
        return None
    return _simt_enum_attr("signedness", signedness, supported=_SIGNEDNESS_TOKENS, context=context)


def _required_signedness_attr(signedness, *, context: str):
    if signedness is None:
        raise TypeError(f"{context} requires signedness='signed' or 'unsigned'")
    return _optional_signedness_attr(signedness, context=context)


def _l1_cache_attr(value, *, context: str):
    return _simt_enum_attr("l1cache", value, supported=_L1_CACHE_TOKENS, context=context)


def _ld_l2_cache_attr(value, *, context: str):
    return _simt_enum_attr("ld_l2cache", value, supported=_LD_L2_CACHE_TOKENS, context=context)


def _st_l2_cache_attr(value, *, context: str):
    return _simt_enum_attr("st_l2cache", value, supported=_ST_L2_CACHE_TOKENS, context=context)


def _normalize_mte_store_l2_cache(l2_cache, *, context: str):
    token = _normalize_token(l2_cache, context=context)
    if token not in _ST_L2_CACHE_CONTROL_VALUES:
        expected = ", ".join(sorted(_ST_L2_CACHE_CONTROL_VALUES))
        raise ValueError(f"{context} does not support {l2_cache!r}; expected one of {expected}")
    return _ST_L2_CACHE_CONTROL_VALUES[token]


def _rounding_attr(value, *, context: str):
    return _simt_enum_attr("rounding", value, supported=_ROUNDING_TOKENS, context=context)


def _saturation_attr(value, *, context: str):
    normalized = _normalize_token(value, context=context)
    aliases = {"on": "sat", "off": "nosat", "sat": "sat", "nosat": "nosat"}
    token = aliases.get(normalized)
    if token is None:
        expected = ", ".join(sorted((*_SATURATION_TOKENS, "on", "off")))
        raise ValueError(f"{context} does not support {value!r}; expected one of {expected}")
    return _simt_enum_attr("saturation", token, supported=_SATURATION_TOKENS, context=context)


def _simt_enum_attr(kind, value, *, supported: set[str], context: str):
    normalized = _normalize_token(value, context=context)
    if normalized not in supported:
        expected = ", ".join(sorted(supported))
        raise ValueError(f"{context} does not support {value!r}; expected one of {expected}")
    return Attribute.parse(f"#pto.{kind}<{normalized}>")


def _coerce_i32_operand(value, *, context: str):
    return coerce_scalar_to_type(value, IntegerType.get_signless(32), context=context)


def _same_type_unary(op_cls, value):
    return wrap_surface_value(op_cls(unwrap_surface_value(value)).result)


def _same_type_binary(op_cls, lhs, rhs, *, context: str):
    raw_lhs = unwrap_surface_value(lhs)
    raw_rhs = coerce_scalar_to_type(rhs, raw_lhs.type, context=context)
    return wrap_surface_value(op_cls(raw_lhs, raw_rhs).result)


def _same_type_ternary(op_cls, lhs, rhs, acc, *, context: str):
    raw_lhs = unwrap_surface_value(lhs)
    raw_rhs = coerce_scalar_to_type(rhs, raw_lhs.type, context=context)
    raw_acc = coerce_scalar_to_type(acc, raw_lhs.type, context=context)
    return wrap_surface_value(op_cls(raw_lhs, raw_rhs, raw_acc).result)


def _validate_redux_signedness(value_type, signedness, *, require_for_integer: bool, context: str):
    if IntegerType.isinstance(value_type):
        if require_for_integer and signedness is None:
            raise TypeError(f"{context} requires signedness='signed' or 'unsigned' for integer values")
        return
    if signedness is not None:
        raise TypeError(f"{context} does not accept signedness for floating-point values")


def _validate_integer_signedness_only(value_type, signedness, *, context: str):
    if signedness is not None and not IntegerType.isinstance(value_type):
        raise TypeError(f"{context} does not accept signedness for non-integer values")


def _validate_convert_signedness(src_type, dst_type, signedness, *, context: str):
    src_int = IntegerType.isinstance(src_type)
    dst_int = IntegerType.isinstance(dst_type)
    if src_int and dst_int:
        raise TypeError(f"{context} does not support integer-to-integer conversion")
    if src_int or dst_int:
        if signedness is None:
            raise TypeError(
                f"{context} requires signedness='signed' or 'unsigned' "
                "when converting to or from integer types")
        return
    if signedness is not None:
        raise TypeError(f"{context} does not accept signedness for floating-point or packed conversion")
