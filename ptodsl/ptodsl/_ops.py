# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
PTO operation wrappers.

Every function in this module emits one or more MLIR operations at the
active insertion point and returns the primary SSA result(s).

Design rules:
- Vector math ops infer the result type from the first operand's type.
- ``vlds(tile[row, col:])`` and ``vlds(ptr, offset)`` infer the result
  ``vreg`` type from the source element type. ``vbrc_load`` still requires an
  explicit result ``vreg`` type because broadcast widths are authored
  explicitly in the current surface.
- ``make_tensor_view`` infers the TensorViewType from ``len(shape)`` and the
  pointer's element type.
- ``partition_view`` infers the PartitionTensorViewType from the source type.
"""

from functools import wraps

from ._bootstrap import make_context  # noqa: F401 – ensure MLIR on sys.path
from ._diagnostics import (
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
    parse_tile_type_metadata,
    resolve_address_access,
    unwrap_surface_value,
    wrap_surface_value,
)
from ._types import (
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

from mlir.dialects import arith, pto as _pto
from mlir.ir import (
    Attribute,
    BF16Type,
    F16Type,
    F32Type,
    Float8E4M3FNType,
    Float8E5M2Type,
    FloatAttr,
    IndexType,
    IntegerType,
    MemRefType,
    Operation,
    Type,
    TypeAttr,
    UnitAttr,
)

# Pipe name shorthands → canonical PIPE_* names
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

def const(value: int, *, dtype=None):
    """
    Emit an ``arith.constant``.

    ``dtype`` is a ``_DType`` descriptor or a concrete ``mlir.ir.Type``.
    Defaults to ``index`` when omitted.
    """
    from ._types import index as _idx_dtype
    mlir_type = _resolve(dtype) if dtype is not None else _resolve(_idx_dtype)
    if any(cls.isinstance(mlir_type) for cls in (F16Type, BF16Type, F32Type)):
        return wrap_surface_value(arith.ConstantOp(mlir_type, FloatAttr.get(mlir_type, value)).result)
    if IntegerType.isinstance(mlir_type):
        return wrap_surface_value(_materialize_integer_literal(mlir_type, value))
    return wrap_surface_value(arith.ConstantOp(mlir_type, value).result)


def get_op_attr(name: str, default=None):
    """Return a TileLib render-time op attribute supplied by the C++ bridge."""
    from ._tracing.active import current_runtime

    runtime = current_runtime()
    attrs = getattr(runtime, "context_attrs", None)
    if not attrs:
        return default
    return attrs.get(name, default)


# ── Pointer ops ───────────────────────────────────────────────────────────────

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

_VLOAD_DIST_TOKENS = {
    "NORM",
    "UNPK_B8", "UNPK_B16", "UNPK_B32",
    "BRC_B8", "BRC_B16", "BRC_B32", "BRC_BLK",
    "US_B8", "US_B16",
    "DS_B8", "DS_B16",
    # Extra load distributions already supported by the backend lowering.
    "E2B_B16", "E2B_B32",
    "UNPK4",
    "SPLT4CHN",
}


def vlds(src_ptr, offset=None, result_vreg_type=None, *, dist=None, post_update="OFF"):
    """``pto.vlds`` – vector load from a tile slice or from *src_ptr* at *offset*."""
    post_mode = _normalize_post_update_mode(post_update, context="vlds(..., post_update=...)")
    if isinstance(src_ptr, TileSliceValue):
        if offset is not None or result_vreg_type is not None:
            raise TypeError("vlds(tile[row, col:]) infers its memref slice and vreg type; do not pass offset/result_vreg_type")
        if post_mode != "NO_POST_UPDATE":
            raise TypeError("vlds(tile[...], post_update=...) only supports post_update=PostUpdate.OFF; use the pointer form for stateful loads")
        kwargs = {}
        if dist is not None:
            kwargs["dist"] = _normalize_dist_token(
                dist,
                allowed=_VLOAD_DIST_TOKENS,
                context="vlds(..., dist)",
            )
        raw_source = unwrap_surface_value(src_ptr)
        return wrap_surface_value(
            _pto.VldsOp(
                _infer_vreg_type_from_tile_slice(src_ptr),
                None,
                raw_source,
                _index_zero(),
                **kwargs,
            ).result
        )

    if offset is None:
        raise TypeError("vlds(ptr, offset, result_vreg_type=None) requires an explicit offset")
    if result_vreg_type is None:
        result_vreg_type = _infer_vreg_type_from_address_source(src_ptr)
    kwargs = {}
    if dist is not None:
        kwargs["dist"] = _normalize_dist_token(
            dist,
            allowed=_VLOAD_DIST_TOKENS,
            context="vlds(..., dist)",
        )
    raw_source = unwrap_surface_value(src_ptr)
    if post_mode == "POST_UPDATE":
        post_ctor = getattr(_pto, "VldsPostOp", None)
        if post_ctor is not None:
            op = post_ctor(
                _resolve(result_vreg_type),
                raw_source.type,
                raw_source,
                _coerce_index(offset, context="vlds(ptr, offset)"),
                **kwargs,
            )
            return wrap_surface_value(op.result), wrap_surface_value(op.updated_source)
        op = _pto.VldsOp(
            _resolve(result_vreg_type),
            raw_source.type,
            raw_source,
            _coerce_index(offset, context="vlds(ptr, offset)"),
            **kwargs,
        )
        return wrap_surface_value(op.result), wrap_surface_value(op.updated_base)
    return wrap_surface_value(
        _pto.VldsOp(
            _resolve(result_vreg_type),
            None,
            raw_source,
            _coerce_index(offset, context="vlds(ptr, offset)"),
            **kwargs,
        ).result
    )


def vldas(source):
    """``pto.vldas`` – prime alignment state for a following unaligned load stream."""
    if isinstance(source, TileSliceValue):
        source = _tile_slice_ptr(source)
    return wrap_surface_value(
        _pto.VldasOp(
            _pto.AlignType.get(),
            unwrap_surface_value(source),
        ).result
    )


def vldus(source, align):
    """``pto.vldus`` – unaligned vector load threaded through alignment state."""
    result_type = (
        _infer_vreg_type_from_tile_slice(source)
        if isinstance(source, TileSliceValue)
        else _infer_vreg_type_from_address_source(source)
    )
    if isinstance(source, TileSliceValue):
        source = _tile_slice_ptr(source)
    op = _pto.VldusOp(
        result_type,
        _pto.AlignType.get(),
        unwrap_surface_value(source),
        unwrap_surface_value(align),
    )
    return wrap_surface_value(op.result), wrap_surface_value(op.updated_align)


_DEINTERLEAVE_DIST_TOKENS = {"DINTLV_B8", "DINTLV_B16", "DINTLV_B32", "BDINTLV"}
_INTERLEAVE_DIST_TOKENS = {"INTLV_B8", "INTLV_B16", "INTLV_B32"}
_VSTORE_DIST_TOKENS = {
    "NORM_B8", "NORM_B16", "NORM_B32",
    "1PT_B8", "1PT_B16", "1PT_B32",
    "PK_B16", "PK_B32", "PK_B64", "PK4_B32",
    "MRG4CHN_B8", "MRG2CHN_B8", "MRG2CHN_B16",
}


def _normalize_dist_token(dist, *, allowed: set[str], context: str):
    token = dist
    if not isinstance(token, str):
        token = str(token)
        if "." in token:
            token = token.rsplit(".", 1)[-1]
    normalized = token.strip().upper()
    if normalized.startswith("_"):
        normalized = normalized[1:]
    if normalized not in allowed:
        expected = ", ".join(sorted(allowed))
        raise ValueError(f"{context} does not support dist {dist!r}; expected one of {expected}")
    return normalized


def vldsx2(source, offset_or_dist, dist=None):
    """``pto.vldsx2`` – dual vector load with deinterleave."""
    if isinstance(source, TileSliceValue):
        if dist is not None:
            raise TypeError("vldsx2(tile[row, col:], dist) does not accept a separate offset argument")
        result_type = _infer_vreg_type_from_tile_slice(source)
        op = _pto.Vldsx2Op(
            result_type,
            result_type,
            unwrap_surface_value(source),
            _index_zero(),
            _normalize_dist_token(
                offset_or_dist,
                allowed=_DEINTERLEAVE_DIST_TOKENS,
                context="vldsx2(..., dist)",
            ),
        )
        return wrap_surface_value(op.low), wrap_surface_value(op.high)

    if dist is None:
        raise TypeError("vldsx2(ptr, offset, dist) requires an explicit offset and dist")
    result_type = _infer_vreg_type_from_address_source(source)
    op = _pto.Vldsx2Op(
        result_type,
        result_type,
        unwrap_surface_value(source),
        _coerce_index(offset_or_dist, context="vldsx2(ptr, offset, dist)"),
        _normalize_dist_token(
            dist,
            allowed=_DEINTERLEAVE_DIST_TOKENS,
            context="vldsx2(..., dist)",
        ),
    )
    return wrap_surface_value(op.low), wrap_surface_value(op.high)


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


def pbitcast(mask_value, to_type):
    """``pto.pbitcast`` – reinterpret one mask register at a different granularity."""
    return wrap_surface_value(
        _pto.PbitcastOp(
            _resolve(to_type),
            unwrap_surface_value(mask_value),
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


def _normalize_vcvt_sat_mode(mode, *, context: str):
    token = mode
    if not isinstance(token, str):
        token = str(token)
        if "." in token:
            token = token.rsplit(".", 1)[-1]
    normalized = token.strip().upper()
    if normalized in {"SAT", "RS_ENABLE"}:
        return "SAT"
    if normalized in {"NOSAT", "RS_DISABLE"}:
        return "NOSAT"
    raise ValueError(f"{context} does not support sat {mode!r}; expected SAT/NOSAT")


def _normalize_vcvt_part_mode(mode, *, context: str):
    token = mode
    if not isinstance(token, str):
        token = str(token)
        if "." in token:
            token = token.rsplit(".", 1)[-1]
    normalized = token.strip().upper()
    allowed = {"EVEN", "ODD", "P0", "P1", "P2", "P3"}
    if normalized not in allowed:
        expected = ", ".join(sorted(allowed))
        raise ValueError(f"{context} does not support part {mode!r}; expected one of {expected}")
    return normalized


def _normalize_enum_attr(value, *, enum_cls, attr_cls, context: str):
    if value is None or isinstance(value, Attribute):
        return value
    if isinstance(value, str):
        token = value.strip().upper()
        try:
            value = getattr(enum_cls, token)
        except AttributeError as exc:
            allowed = ", ".join(name for name in dir(enum_cls) if name.isupper())
            raise ValueError(f"{context} does not support {value!r}; expected one of {allowed}") from exc
    return attr_cls.get(value)


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


def _classify_vcvt_elem_kind(elem_type):
    if Float8E4M3FNType.isinstance(elem_type):
        return "f8e4m3"
    if Float8E5M2Type.isinstance(elem_type):
        return "f8e5m2"
    if _isinstance_pto_type(elem_type, "HiF8Type"):
        return "hif8"
    if _isinstance_pto_type(elem_type, "F4E1M2x2Type"):
        return "f4e1m2x2"
    if _isinstance_pto_type(elem_type, "F4E2M1x2Type"):
        return "f4e2m1x2"
    if F16Type.isinstance(elem_type):
        return "f16"
    if BF16Type.isinstance(elem_type):
        return "bf16"
    if F32Type.isinstance(elem_type):
        return "f32"
    if IntegerType.isinstance(elem_type):
        int_type = IntegerType(elem_type)
        width = int_type.width
        if width == 8:
            return "u8" if int_type.is_unsigned else "s8"
        if width == 16:
            return "u16" if int_type.is_unsigned else "s16"
        if width == 32:
            return "u32" if int_type.is_unsigned else "s32"
        if width == 64 and not int_type.is_unsigned:
            return "s64"
    return None


def _vcvt_contract(requires_rnd, requires_sat, requires_part, *, part_family=None, allowed_rnd=None):
    return {
        "requires_rnd": requires_rnd,
        "requires_sat": requires_sat,
        "requires_part": requires_part,
        "part_family": part_family,
        "allowed_rnd": set(allowed_rnd) if allowed_rnd is not None else None,
    }


_VCVT_CONTRACTS = {
    ("f32", "f8e4m3"): _vcvt_contract(True, True, True, part_family="packed4", allowed_rnd="RAHZ"),
    ("f32", "f8e5m2"): _vcvt_contract(True, True, True, part_family="packed4", allowed_rnd="RAHZ"),
    ("f32", "hif8"): _vcvt_contract(True, True, True, part_family="packed4", allowed_rnd="AH"),
    ("f32", "f16"): _vcvt_contract(True, True, True),
    ("f32", "bf16"): _vcvt_contract(True, True, True),
    ("f32", "s16"): _vcvt_contract(True, True, True),
    ("f32", "s64"): _vcvt_contract(True, True, True),
    ("f32", "s32"): _vcvt_contract(True, True, False),
    ("f16", "f8e4m3"): _vcvt_contract(True, True, True, allowed_rnd="RAFZC"),
    ("f16", "f8e5m2"): _vcvt_contract(True, True, True, allowed_rnd="RAFZC"),
    ("f16", "hif8"): _vcvt_contract(True, True, True, allowed_rnd="AH"),
    ("f16", "f32"): _vcvt_contract(False, False, True),
    ("f16", "s32"): _vcvt_contract(True, False, True),
    ("f16", "s16"): _vcvt_contract(True, True, False),
    ("f16", "s8"): _vcvt_contract(True, True, True),
    ("f16", "u8"): _vcvt_contract(True, True, True),
    ("bf16", "f8e4m3"): _vcvt_contract(True, True, True, allowed_rnd="RAFZC"),
    ("bf16", "f8e5m2"): _vcvt_contract(True, True, True, allowed_rnd="RAFZC"),
    ("bf16", "f4e1m2x2"): _vcvt_contract(True, False, True, part_family="packed4", allowed_rnd="RAFZC"),
    ("bf16", "f4e2m1x2"): _vcvt_contract(True, False, True, part_family="packed4", allowed_rnd="RAFZC"),
    ("bf16", "f16"): _vcvt_contract(True, True, False),
    ("bf16", "f32"): _vcvt_contract(False, False, True),
    ("bf16", "s32"): _vcvt_contract(True, True, True),
    ("u8", "f16"): _vcvt_contract(False, False, True),
    ("u8", "u16"): _vcvt_contract(False, False, True),
    ("u8", "u32"): _vcvt_contract(False, False, True),
    ("s8", "f16"): _vcvt_contract(False, False, True),
    ("s8", "s16"): _vcvt_contract(False, False, True),
    ("s8", "s32"): _vcvt_contract(False, False, True),
    ("u16", "u8"): _vcvt_contract(False, True, True),
    ("u16", "u32"): _vcvt_contract(False, False, True),
    ("s16", "f16"): _vcvt_contract(True, False, False),
    ("s16", "f32"): _vcvt_contract(False, False, True),
    ("s16", "u32"): _vcvt_contract(False, False, True),
    ("s16", "s32"): _vcvt_contract(False, False, True),
    ("s16", "u8"): _vcvt_contract(False, True, True),
    ("u32", "u8"): _vcvt_contract(False, True, True),
    ("u32", "u16"): _vcvt_contract(False, True, True),
    ("u32", "s16"): _vcvt_contract(False, True, True),
    ("s32", "f32"): _vcvt_contract(True, False, False),
    ("s32", "u8"): _vcvt_contract(False, True, True),
    ("s32", "u16"): _vcvt_contract(False, True, True),
    ("s32", "s16"): _vcvt_contract(False, True, True),
    ("s32", "s64"): _vcvt_contract(False, False, True),
    ("s64", "f32"): _vcvt_contract(True, False, True),
    ("s64", "s32"): _vcvt_contract(False, True, True),
    ("f8e4m3", "f32"): _vcvt_contract(False, False, True, part_family="packed4"),
    ("f8e5m2", "f32"): _vcvt_contract(False, False, True, part_family="packed4"),
    ("hif8", "f32"): _vcvt_contract(False, False, True, part_family="packed4"),
    ("f4e1m2x2", "bf16"): _vcvt_contract(False, False, True, part_family="packed4"),
    ("f4e2m1x2", "bf16"): _vcvt_contract(False, False, True, part_family="packed4"),
}


_VCVT_ELEM_BITS = {
    "f4e1m2x2": 8,
    "f4e2m1x2": 8,
    "f8e4m3": 8,
    "f8e5m2": 8,
    "hif8": 8,
    "u8": 8,
    "s8": 8,
    "f16": 16,
    "bf16": 16,
    "u16": 16,
    "s16": 16,
    "f32": 32,
    "u32": 32,
    "s32": 32,
    "s64": 64,
}


def _infer_vcvt_part_family(src_kind, result_kind):
    src_bits = _VCVT_ELEM_BITS.get(src_kind)
    result_bits = _VCVT_ELEM_BITS.get(result_kind)
    if src_bits is None or result_bits is None:
        return None
    larger = max(src_bits, result_bits)
    smaller = min(src_bits, result_bits)
    if larger == smaller * 2:
        return "even_odd"
    if larger == smaller * 4:
        return "packed4"
    return None


def _validate_vcvt_attrs(src_kind, result_kind, contract, *, rnd, sat, part, context: str):
    if rnd is None:
        if contract["requires_rnd"]:
            raise ValueError(f"{context} requires rnd for dtype pair {src_kind} -> {result_kind}")
    elif not contract["requires_rnd"]:
        raise ValueError(f"{context} does not support rnd for dtype pair {src_kind} -> {result_kind}")

    if sat is None:
        if contract["requires_sat"]:
            raise ValueError(f"{context} requires sat for dtype pair {src_kind} -> {result_kind}")
    elif not contract["requires_sat"]:
        raise ValueError(f"{context} does not support sat for dtype pair {src_kind} -> {result_kind}")

    if part is None:
        if contract["requires_part"]:
            raise ValueError(f"{context} requires part for dtype pair {src_kind} -> {result_kind}")
    elif not contract["requires_part"]:
        raise ValueError(f"{context} does not support part for dtype pair {src_kind} -> {result_kind}")

    allowed_rnd = contract["allowed_rnd"]
    if rnd is not None and allowed_rnd is not None and rnd not in allowed_rnd:
        expected = ", ".join(sorted(allowed_rnd))
        raise ValueError(
            f"{context} does not support rnd {rnd!r} for dtype pair "
            f"{src_kind} -> {result_kind}; expected one of {expected}"
        )

    if part is None:
        return
    part_family = contract["part_family"] or _infer_vcvt_part_family(src_kind, result_kind)
    if part_family == "even_odd":
        if part not in {"EVEN", "ODD"}:
            raise ValueError(
                f"{context} part must be EVEN or ODD for dtype pair "
                f"{src_kind} -> {result_kind}"
            )
    elif part_family == "packed4":
        if part not in {"P0", "P1", "P2", "P3"}:
            raise ValueError(
                f"{context} part must be P0, P1, P2, or P3 for dtype pair "
                f"{src_kind} -> {result_kind}"
            )
    elif part_family is None:
        raise ValueError(f"{context} part is not supported for dtype pair {src_kind} -> {result_kind}")


def _validate_vcvt_dtype_pair(src, result_dtype, *, rnd=None, sat=None, part=None, context: str):
    _, src_elem_type = _infer_vreg_metadata(src)
    resolved_result_dtype = _resolve(result_dtype)
    src_kind = _classify_vcvt_elem_kind(src_elem_type)
    result_kind = _classify_vcvt_elem_kind(resolved_result_dtype)
    if src_kind is None or result_kind is None:
        raise TypeError(
            f"{context} does not support source/result element types "
            f"{src_elem_type} -> {resolved_result_dtype}"
        )
    contract = _VCVT_CONTRACTS.get((src_kind, result_kind))
    if contract is None:
        raise TypeError(
            f"{context} currently does not support the dtype pair "
            f"{src_kind} -> {result_kind}"
        )
    _validate_vcvt_attrs(src_kind, result_kind, contract, rnd=rnd, sat=sat, part=part, context=context)
    return resolved_result_dtype


def _infer_result_vreg_type_for_element_dtype(src, result_dtype, *, rnd=None, sat=None, part=None, context: str):
    resolved_type = _validate_vcvt_dtype_pair(
        src,
        result_dtype,
        rnd=rnd,
        sat=sat,
        part=part,
        context=context,
    )
    try:
        _pto.VRegType(resolved_type)
        return resolved_type
    except Exception:
        pass
    lanes, src_elem_type = _infer_vreg_metadata(src)
    total_bytes = lanes * _element_bytewidth(src_elem_type)
    result_elem_bytes = _element_bytewidth(resolved_type)
    if total_bytes % result_elem_bytes != 0:
        raise TypeError(
            f"{context} cannot infer a result vreg type from {unwrap_surface_value(src).type} -> {resolved_type}; "
            "the total vector payload is not evenly divisible by the target element width"
        )
    return _resolve(vreg_type(total_bytes // result_elem_bytes, resolved_type))


def _infer_vpack_result_type(src):
    lanes, elem_type = _infer_vreg_metadata(src)
    if not IntegerType.isinstance(elem_type):
        raise TypeError(f"vpack(src, part) expects an integer source vreg, got {elem_type}")
    src_int_type = IntegerType(elem_type)
    src_width = src_int_type.width
    if src_width not in {16, 32}:
        raise TypeError(
            "vpack(src, part) currently supports only the source/result shape pairs "
            "s32/u32 -> u16 and s16/u16 -> u8"
        )
    result_elem_type = IntegerType.get_unsigned(src_width // 2)
    result_type = _resolve(vreg_type(lanes * 2, result_elem_type))
    result_vreg_type = _pto.VRegType(result_type)
    if result_vreg_type.element_count != lanes * 2:
        raise TypeError(
            "vpack(src, part) requires the packed result lane count to be twice "
            "the source lane count"
        )
    result_int_type = IntegerType(result_vreg_type.element_type)
    if result_int_type.width * 2 != src_width:
        raise TypeError(
            "vpack(src, part) requires the packed result element width to be half "
            "the source element width"
        )
    if not result_int_type.is_unsigned:
        raise TypeError("vpack(src, part) requires an unsigned packed result element type")
    if not ((src_width == 32 and result_int_type.width == 16) or
            (src_width == 16 and result_int_type.width == 8)):
        raise TypeError(
            "vpack(src, part) currently supports only the source/result shape pairs "
            "s32/u32 -> u16 and s16/u16 -> u8"
        )
    return result_type


def vcvt(src, to_dtype, mask, *, rnd=None, sat=None, part=None):
    """``pto.vcvt`` – explicit vector type conversion."""
    kwargs = {}
    if rnd is not None:
        kwargs["rnd"] = _normalize_vcvt_round_mode(rnd, context="vcvt(..., rnd=...)")
    if sat is not None:
        kwargs["sat"] = _normalize_vcvt_sat_mode(sat, context="vcvt(..., sat=...)")
    if part is not None:
        kwargs["part"] = _normalize_vcvt_part_mode(part, context="vcvt(..., part=...)")
    return wrap_surface_value(
        _pto.VcvtOp(
            _infer_result_vreg_type_for_element_dtype(
                src,
                to_dtype,
                rnd=kwargs.get("rnd"),
                sat=kwargs.get("sat"),
                part=kwargs.get("part"),
                context="vcvt(src, to_dtype, mask)",
            ),
            unwrap_surface_value(src),
            unwrap_surface_value(mask),
            **kwargs,
        ).result
    )


def vpack(src, part):
    """``pto.vpack`` – narrow-pack one vector half into an unsigned result vector."""
    return wrap_surface_value(
        _pto.VpackOp(
            _infer_vpack_result_type(src),
            unwrap_surface_value(src),
            _normalize_vpack_part(part, context="vpack(src, part)"),
        ).result
    )


def vmulscvt(src, scalar, mask, *, rnd, part):
    """``pto.vmulscvt`` – explicit fused mul+convert micro-op."""
    op_ctor = getattr(_pto, "VmulscvtOp", None)
    if op_ctor is None:
        pto_module_path = getattr(_pto, "__file__", "<unknown>")
        raise NotImplementedError(
            "pto.vmulscvt(...) is not available in the current PTO build: "
            f"the loaded Python bindings at {pto_module_path} do not expose a VPTO vmulscvt op yet"
        )
    round_mode = _normalize_vcvt_round_mode(rnd, context="vmulscvt(..., rnd=...)")
    if round_mode != "A":
        raise ValueError("vmulscvt(..., rnd=...) currently only supports A on the current PTO backend")
    lanes, elem_type = _infer_vreg_metadata(src)
    if not F32Type.isinstance(elem_type):
        raise TypeError(
            "vmulscvt(src, scalar, mask) currently only supports the dtype pair "
            f"f32 -> f16; got source element type {elem_type}"
        )
    result_type = _resolve(vreg_type(lanes * 2, F16Type.get()))
    scalar_value = _coerce_scalar_like_vector_element(src, scalar, context="vmulscvt")
    return wrap_surface_value(
        op_ctor(
            result_type,
            unwrap_surface_value(src),
            unwrap_surface_value(scalar_value),
            unwrap_surface_value(mask),
            round_mode,
            _normalize_vcvt_part_mode(part, context="vmulscvt(..., part=...)"),
        ).result
    )


def vsts(val, dst_ptr, offset, mask=None, *, dist=None, post_update="OFF"):
    """``pto.vsts`` – vector store to a tile slice or to *dst_ptr* at *offset*."""
    post_mode = _normalize_post_update_mode(post_update, context="vsts(..., post_update=...)")
    if isinstance(dst_ptr, TileSliceValue):
        if mask is not None:
            raise TypeError("vsts(vec, tile[row, col:], mask) does not accept a separate offset argument")
        if post_mode != "NO_POST_UPDATE":
            raise TypeError("vsts(vec, tile[...], post_update=...) only supports post_update=PostUpdate.OFF; use the pointer form for stateful stores")
        kwargs = {}
        if dist is not None:
            kwargs["dist"] = _normalize_dist_token(
                dist,
                allowed=_VSTORE_DIST_TOKENS,
                context="vsts(..., dist)",
            )
        raw_destination = unwrap_surface_value(dst_ptr)
        _pto.VstsOp(
            None,
            unwrap_surface_value(val),
            raw_destination,
            _index_zero(),
            unwrap_surface_value(offset),
            **kwargs,
        )
        return

    if mask is None:
        raise TypeError("vsts(vec, ptr, offset, mask) requires an explicit mask")
    kwargs = {}
    if dist is not None:
        kwargs["dist"] = _normalize_dist_token(
            dist,
            allowed=_VSTORE_DIST_TOKENS,
            context="vsts(..., dist)",
        )
    if post_mode == "POST_UPDATE":
        raw_destination = unwrap_surface_value(dst_ptr)
        post_ctor = getattr(_pto, "VstsPostOp", None)
        if post_ctor is not None:
            op = post_ctor(
                raw_destination.type,
                unwrap_surface_value(val),
                raw_destination,
                _coerce_index(offset, context="vsts(ptr, offset, mask)"),
                unwrap_surface_value(mask),
                **kwargs,
            )
            return wrap_surface_value(op.updated_destination)
        op = _pto.VstsOp(
            raw_destination.type,
            unwrap_surface_value(val),
            raw_destination,
            _coerce_index(offset, context="vsts(ptr, offset, mask)"),
            unwrap_surface_value(mask),
            **kwargs,
        )
        return wrap_surface_value(op.updated_base)
    _pto.VstsOp(
        None,
        unwrap_surface_value(val),
        unwrap_surface_value(dst_ptr),
        _coerce_index(offset, context="vsts(ptr, offset, mask)"),
        unwrap_surface_value(mask),
        **kwargs,
    )


def vstsx2(low, high, dst_ptr, offset_or_dist, dist_or_mask=None, mask=None):
    """``pto.vstsx2`` – dual interleaving vector store."""
    if isinstance(dst_ptr, TileSliceValue):
        if mask is not None:
            raise TypeError("vstsx2(low, high, tile[row, col:], dist, mask) does not accept a separate offset argument")
        _pto.Vstsx2Op(
            unwrap_surface_value(low),
            unwrap_surface_value(high),
            unwrap_surface_value(dst_ptr),
            _index_zero(),
            _normalize_dist_token(
                offset_or_dist,
                allowed=_INTERLEAVE_DIST_TOKENS,
                context="vstsx2(..., dist)",
            ),
            unwrap_surface_value(dist_or_mask),
        )
        return

    if mask is None:
        raise TypeError("vstsx2(low, high, ptr, offset, dist, mask) requires an explicit offset, dist, and mask")
    _pto.Vstsx2Op(
        unwrap_surface_value(low),
        unwrap_surface_value(high),
        unwrap_surface_value(dst_ptr),
        _coerce_index(offset_or_dist, context="vstsx2(ptr, offset, dist, mask)"),
        _normalize_dist_token(
            dist_or_mask,
            allowed=_INTERLEAVE_DIST_TOKENS,
            context="vstsx2(..., dist)",
        ),
        unwrap_surface_value(mask),
    )


def vgather2(buf, offsets, mask, result_vreg_type=None):
    """``pto.vgather2`` – indexed gather from UB."""
    rt = result_vreg_type if result_vreg_type is not None else _infer_vreg_type_from_address_source(buf)
    return wrap_surface_value(
        _pto.Vgather2Op(
            _resolve(rt),
            unwrap_surface_value(buf),
            unwrap_surface_value(offsets),
            unwrap_surface_value(mask),
        ).result
    )


def vgather2_bc(buf, offsets, mask, result_vreg_type=None):
    """``pto.vgather2_bc`` – indexed gather from UB with masked zero-fill."""
    rt = result_vreg_type if result_vreg_type is not None else _infer_vreg_type_from_address_source(buf)
    return wrap_surface_value(
        _pto.Vgather2BcOp(
            _resolve(rt),
            unwrap_surface_value(buf),
            unwrap_surface_value(offsets),
            unwrap_surface_value(mask),
        ).result
    )


def vgatherb(buf, offsets, mask, result_vreg_type=None):
    """``pto.vgatherb`` – block gather from UB using byte offsets."""
    rt = result_vreg_type if result_vreg_type is not None else _infer_vreg_type_from_address_source(buf)
    return wrap_surface_value(
        _pto.VgatherbOp(
            _resolve(rt),
            unwrap_surface_value(buf),
            unwrap_surface_value(offsets),
            unwrap_surface_value(mask),
        ).result
    )


def vscatter(value, destination, offsets, mask):
    """``pto.vscatter`` – indexed scatter to UB."""
    _pto.VscatterOp(
        unwrap_surface_value(value),
        unwrap_surface_value(destination),
        unwrap_surface_value(offsets),
        unwrap_surface_value(mask),
    )


def _coerce_i16(value, *, context: str):
    raw_value = unwrap_surface_value(value)
    return coerce_runtime_integer_value(raw_value, IntegerType.get_signless(16), context=context)


def vsldb(source, block_stride, repeat_stride, mask):
    """``pto.vsldb`` – block-strided load."""
    result_type = (
        _infer_vreg_type_from_tile_slice(source)
        if isinstance(source, TileSliceValue)
        else _infer_vreg_type_from_address_source(source)
    )
    return wrap_surface_value(
        _pto.VsldbOp(
            result_type,
            unwrap_surface_value(source),
            _coerce_i16(block_stride, context="vsldb(..., block_stride, repeat_stride, mask)"),
            _coerce_i16(repeat_stride, context="vsldb(..., block_stride, repeat_stride, mask)"),
            unwrap_surface_value(mask),
        ).result
    )


def vsstb(value, destination, block_stride, repeat_stride, mask, *, post_update="OFF"):
    """``pto.vsstb`` – block-strided store."""
    post_mode = _normalize_post_update_mode(post_update, context="vsstb(..., post_update=...)")
    if post_mode == "POST_UPDATE":
        raw_destination = unwrap_surface_value(destination)
        op = _pto.VsstbOp(
            raw_destination.type,
            unwrap_surface_value(value),
            raw_destination,
            _coerce_i16(block_stride, context="vsstb(..., block_stride, repeat_stride, mask)"),
            _coerce_i16(repeat_stride, context="vsstb(..., block_stride, repeat_stride, mask)"),
            unwrap_surface_value(mask),
        )
        return wrap_surface_value(op.updated_base)
    _pto.VsstbOp(
        None,
        unwrap_surface_value(value),
        unwrap_surface_value(destination),
        _coerce_i16(block_stride, context="vsstb(..., block_stride, repeat_stride, mask)"),
        _coerce_i16(repeat_stride, context="vsstb(..., block_stride, repeat_stride, mask)"),
        unwrap_surface_value(mask),
    )


# ── Mask / predicate ops ──────────────────────────────────────────────────────

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


def _plt_impl(mask_bits: int, scalar):
    plt_op = _plt_op_for_mask_bits(mask_bits)(
        _mask_type_from_bits(mask_bits),
        IntegerType.get_signless(32),
        _coerce_i32(scalar, context=f"plt_b{mask_bits}(scalar)"),
    )
    return wrap_surface_value(plt_op.mask), wrap_surface_value(plt_op.scalar_out)


def plt_b8(scalar):
    """``pto.plt_b8`` – predicate-load from a 32-bit scalar into a b8 mask."""
    return _plt_impl(8, scalar)


def plt_b16(scalar):
    """``pto.plt_b16`` – predicate-load from a 32-bit scalar into a b16 mask."""
    return _plt_impl(16, scalar)


def plt_b32(scalar):
    """
    ``pto.plt_b32`` – predicate-load from a 32-bit scalar.

    Returns ``(mask_value, scalar_out)``.  ``scalar_out`` is often unused
    and can be discarded with ``_``.
    """
    return _plt_impl(32, scalar)


def _pset_impl(mask_bits: int, pattern):
    return wrap_surface_value(
        _pset_op_for_mask_bits(mask_bits)(
            _mask_type_from_bits(mask_bits),
            _normalize_mask_pattern(pattern),
        ).result
    )


def pset_b8(pattern):
    """``pto.pset_b8(pattern)`` → ``!pto.mask<b8>``."""
    return _pset_impl(8, pattern)


def pset_b16(pattern):
    """``pto.pset_b16(pattern)`` → ``!pto.mask<b16>``."""
    return _pset_impl(16, pattern)


def pset_b32(pattern):
    """``pto.pset_b32(pattern)`` → ``!pto.mask<b32>``."""
    return _pset_impl(32, pattern)


def _pge_op_for_mask_bits(mask_bits: int):
    return {
        8: _pto.PgeB8Op,
        16: _pto.PgeB16Op,
        32: _pto.PgeB32Op,
    }[mask_bits]


def _pge_impl(mask_bits: int, pattern):
    return wrap_surface_value(
        _pge_op_for_mask_bits(mask_bits)(
            _mask_type_from_bits(mask_bits),
            _normalize_mask_pattern(pattern),
        ).result
    )


def pge_b8(pattern):
    """``pto.pge_b8(pattern)`` → ``!pto.mask<b8>``."""
    return _pge_impl(8, pattern)


def pge_b16(pattern):
    """``pto.pge_b16(pattern)`` → ``!pto.mask<b16>``."""
    return _pge_impl(16, pattern)


def pge_b32(pattern):
    """``pto.pge_b32(pattern)`` → ``!pto.mask<b32>``."""
    return _pge_impl(32, pattern)


def pand(src0, src1, mask):
    """``pto.pand`` – gated mask AND."""
    result_type = _require_same_mask_types((src0, src1, mask), context="pand(src0, src1, mask)")
    return wrap_surface_value(
        _pto.PandOp(
            result_type,
            unwrap_surface_value(src0),
            unwrap_surface_value(src1),
            unwrap_surface_value(mask),
        ).result
    )


def por(src0, src1, mask):
    """``pto.por`` – gated mask OR."""
    result_type = _require_same_mask_types((src0, src1, mask), context="por(src0, src1, mask)")
    return wrap_surface_value(
        _pto.PorOp(
            result_type,
            unwrap_surface_value(src0),
            unwrap_surface_value(src1),
            unwrap_surface_value(mask),
        ).result
    )


def pxor(src0, src1, mask):
    """``pto.pxor`` – gated mask XOR."""
    result_type = _require_same_mask_types((src0, src1, mask), context="pxor(src0, src1, mask)")
    return wrap_surface_value(
        _pto.PxorOp(
            result_type,
            unwrap_surface_value(src0),
            unwrap_surface_value(src1),
            unwrap_surface_value(mask),
        ).result
    )


def pnot(src, mask):
    """``pto.pnot`` – gated mask NOT."""
    result_type = _require_same_mask_types((src, mask), context="pnot(src, mask)")
    return wrap_surface_value(
        _pto.PnotOp(
            result_type,
            unwrap_surface_value(src),
            unwrap_surface_value(mask),
        ).result
    )


def psel(src0, src1, sel):
    """``pto.psel`` – per-lane mask select."""
    result_type = _require_same_mask_types((src0, src1, sel), context="psel(src0, src1, sel)")
    return wrap_surface_value(
        _pto.PselOp(
            result_type,
            unwrap_surface_value(src0),
            unwrap_surface_value(src1),
            unwrap_surface_value(sel),
        ).result
    )


def ppack(mask_value, part, to_type=None):
    """``pto.ppack`` – pack predicate bits into the selected half."""
    _, inferred_type = _infer_mask_metadata(mask_value, context="ppack(mask, part)")
    result_type = _resolve_mask_result_type(to_type, context="ppack(mask, part, to_type=...)")
    if result_type is None:
        result_type = inferred_type
    return wrap_surface_value(
        _pto.PpackOp(
            result_type,
            unwrap_surface_value(mask_value),
            _normalize_predicate_part(part),
        ).result
    )


def punpack(mask_value, part, to_type=None):
    """``pto.punpack`` – unpack predicate bits from the selected half."""
    _, inferred_type = _infer_mask_metadata(mask_value, context="punpack(mask, part)")
    result_type = _resolve_mask_result_type(to_type, context="punpack(mask, part, to_type=...)")
    if result_type is None:
        result_type = inferred_type
    return wrap_surface_value(
        _pto.PunpackOp(
            result_type,
            unwrap_surface_value(mask_value),
            _normalize_predicate_part(part),
        ).result
    )


def _pintlv_op_for_mask_bits(mask_bits: int):
    return {
        8: _pto.PintlvB8Op,
        16: _pto.PintlvB16Op,
        32: _pto.PintlvB32Op,
    }[mask_bits]


def _pdintlv_op_for_mask_bits(mask_bits: int):
    return {
        8: _pto.PdintlvB8Op,
        16: _pto.PdintlvB16Op,
        32: _pto.PdintlvB32Op,
    }[mask_bits]


def _mask_pair_op(op_resolver, lhs, rhs, *, expected_mask_bits: int, context: str):
    mask_bits, result_type = _infer_mask_metadata(lhs, context=context)
    if mask_bits != expected_mask_bits:
        raise TypeError(f"{context} expects mask_b{expected_mask_bits} operands, got mask_b{mask_bits}")
    _require_same_mask_types((lhs, rhs), context=context)
    op = op_resolver(mask_bits)(
        result_type,
        result_type,
        unwrap_surface_value(lhs),
        unwrap_surface_value(rhs),
    )
    return wrap_surface_value(op.low), wrap_surface_value(op.high)


def pintlv_b8(lhs, rhs):
    """``pto.pintlv_b8`` – interleave two b8 masks."""
    return _mask_pair_op(
        _pintlv_op_for_mask_bits,
        lhs,
        rhs,
        expected_mask_bits=8,
        context="pintlv_b8(lhs, rhs)",
    )


def pintlv_b16(lhs, rhs):
    """``pto.pintlv_b16`` – interleave two b16 masks."""
    return _mask_pair_op(
        _pintlv_op_for_mask_bits,
        lhs,
        rhs,
        expected_mask_bits=16,
        context="pintlv_b16(lhs, rhs)",
    )


def pintlv_b32(lhs, rhs):
    """``pto.pintlv_b32`` – interleave two b32 masks."""
    return _mask_pair_op(
        _pintlv_op_for_mask_bits,
        lhs,
        rhs,
        expected_mask_bits=32,
        context="pintlv_b32(lhs, rhs)",
    )


def pdintlv_b8(lhs, rhs):
    """``pto.pdintlv_b8`` – deinterleave two b8 masks."""
    return _mask_pair_op(
        _pdintlv_op_for_mask_bits,
        lhs,
        rhs,
        expected_mask_bits=8,
        context="pdintlv_b8(lhs, rhs)",
    )


def pdintlv_b16(lhs, rhs):
    """``pto.pdintlv_b16`` – deinterleave two b16 masks."""
    return _mask_pair_op(
        _pdintlv_op_for_mask_bits,
        lhs,
        rhs,
        expected_mask_bits=16,
        context="pdintlv_b16(lhs, rhs)",
    )


def pdintlv_b32(lhs, rhs):
    """``pto.pdintlv_b32`` – deinterleave two b32 masks."""
    return _mask_pair_op(
        _pdintlv_op_for_mask_bits,
        lhs,
        rhs,
        expected_mask_bits=32,
        context="pdintlv_b32(lhs, rhs)",
    )


def vcmp(src0, src1, seed_mask, cmp_mode):
    """``pto.vcmp`` – vector/vector comparison producing a predicate mask."""
    _, elem_type = _infer_vreg_metadata(src0)
    result_type = _mask_type_from_bits(_mask_bits_for_dtype(elem_type))
    seed_type = unwrap_surface_value(seed_mask).type
    if seed_type != result_type:
        raise TypeError(
            f"vcmp(src0, src1, seed_mask, cmp_mode) expects seed_mask {result_type}, got {seed_type}"
        )
    return wrap_surface_value(
        _pto.VcmpOp(
            result_type,
            unwrap_surface_value(src0),
            unwrap_surface_value(src1),
            unwrap_surface_value(seed_mask),
            _normalize_cmp_mode(cmp_mode),
        ).result
    )


def vcmps(src, scalar, seed_mask, cmp_mode):
    """``pto.vcmps`` – vector/scalar comparison producing a predicate mask."""
    _, elem_type = _infer_vreg_metadata(src)
    result_type = _mask_type_from_bits(_mask_bits_for_dtype(elem_type))
    seed_type = unwrap_surface_value(seed_mask).type
    if seed_type != result_type:
        raise TypeError(
            f"vcmps(src, scalar, seed_mask, cmp_mode) expects seed_mask {result_type}, got {seed_type}"
        )
    scalar_value = _coerce_scalar_like_vector_element(src, scalar, context="vcmps")
    return wrap_surface_value(
        _pto.VcmpsOp(
            result_type,
            unwrap_surface_value(src),
            unwrap_surface_value(scalar_value),
            unwrap_surface_value(seed_mask),
            _normalize_cmp_mode(cmp_mode),
        ).result
    )


def plds(buf, offset, *, dist="NORM"):
    """``pto.plds`` – load a predicate mask from UB memory."""
    elem_type = _pointer_element_type(buf, context="plds(buf, offset)")
    result_type = _mask_type_from_bits(_mask_bits_for_dtype(elem_type))
    return wrap_surface_value(
        _pto.PldsOp(
            result_type,
            unwrap_surface_value(buf),
            _coerce_index(offset, context="plds(buf, offset)"),
            _normalize_predicate_dist(
                dist,
                allowed=_PREDICATE_LOAD_DIST_TOKENS,
                context="plds(..., dist)",
            ),
        ).result
    )


def psts(mask_value, buf, offset, *, dist="NORM"):
    """``pto.psts`` – store a predicate mask to UB memory."""
    _infer_mask_metadata(mask_value, context="psts(mask, buf, offset)")
    _pto.PstsOp(
        unwrap_surface_value(mask_value),
        unwrap_surface_value(buf),
        _coerce_index(offset, context="psts(mask, buf, offset)"),
        _normalize_predicate_dist(
            dist,
            allowed=_PREDICATE_STORE_DIST_TOKENS,
            context="psts(..., dist)",
        ),
    )


def pstu(align_in, mask_value, buf):
    """``pto.pstu`` – unaligned predicate store with threaded alignment state."""
    mask_bits, _ = _infer_mask_metadata(mask_value, context="pstu(align_in, mask, buf)")
    if mask_bits not in {16, 32}:
        raise TypeError("pstu(align_in, mask, buf) currently supports only mask_b16 and mask_b32")
    elem_type = _pointer_element_type(buf, context="pstu(align_in, mask, buf)")
    expected_bytes = mask_bits // 8
    actual_bytes = _element_bytewidth(elem_type)
    if actual_bytes != expected_bytes:
        raise TypeError(
            f"pstu(align_in, mask, buf) expects a {expected_bytes}-byte pointer element for mask_b{mask_bits}, "
            f"got {elem_type}"
        )
    align_type = _pto.AlignType.get()
    base_type = unwrap_surface_value(buf).type
    op = _pto.PstuOp(
        align_type,
        base_type,
        unwrap_surface_value(align_in),
        unwrap_surface_value(mask_value),
        unwrap_surface_value(buf),
    )
    return wrap_surface_value(op.align_out), wrap_surface_value(op.base_out)


def vstar(align, destination):
    """``pto.vstar`` – flush alignment-buffered tail bytes to the destination base."""
    _pto.VstarOp(
        unwrap_surface_value(align),
        unwrap_surface_value(destination),
    )


def vstas(align, destination, offset):
    """``pto.vstas`` – flush alignment-buffered tail bytes with an explicit offset."""
    _pto.VstasOp(
        unwrap_surface_value(align),
        unwrap_surface_value(destination),
        _coerce_i32(offset, context="vstas(align, destination, offset)"),
    )


def vstur(align_in, value, base, mode="NO_POST_UPDATE"):
    """``pto.vstur`` – unaligned vector store that updates only alignment state."""
    return wrap_surface_value(
        _pto.VsturOp(
            _pto.AlignType.get(),
            unwrap_surface_value(align_in),
            unwrap_surface_value(value),
            unwrap_surface_value(base),
            _normalize_post_update_mode(mode, context="vstur(..., mode)"),
        ).align_out
    )


def vstus(align_in, offset, value, base):
    """``pto.vstus`` – scalar-offset unaligned vector store that updates alignment state."""
    return wrap_surface_value(
        _pto.VstusOp(
            _pto.AlignType.get(),
            unwrap_surface_value(align_in),
            _coerce_i32(offset, context="vstus(align, offset, value, base)"),
            unwrap_surface_value(value),
            unwrap_surface_value(base),
        ).align_out
    )


# ── Vector math (result type inferred from first operand) ─────────────────────

def vbr(value):
    """``pto.vbr`` – broadcast one scalar value to all vector lanes."""
    raw_value = unwrap_surface_value(value)
    if isinstance(raw_value, bool):
        raise TypeError("vbr(value) does not accept bool values")

    if hasattr(raw_value, "type"):
        scalar_kind = classify_runtime_scalar_type(raw_value.type)
        if scalar_kind == "index":
            raise TypeError("vbr(value) does not support index scalars")
        scalar_value = raw_value
        elem_type = raw_value.type
    else:
        if isinstance(raw_value, float):
            elem_type = F32Type.get()
        elif isinstance(raw_value, int):
            elem_type = IntegerType.get_signless(32)
        else:
            raise TypeError("vbr(value) expects a runtime scalar or one Python int/float literal")
        scalar_value = materialize_scalar_literal(raw_value, elem_type, context="vbr(value)")

    try:
        result_type = _resolve(vreg_type(_elements_per_vreg(elem_type), elem_type))
    except TypeError as exc:
        raise TypeError(f"vbr(value) does not support scalar type {elem_type}") from exc

    return wrap_surface_value(_pto.VbrOp(result_type, scalar_value).result)


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


def vadd(lhs, rhs, mask, result_type=None):
    """``pto.vadd`` – element-wise add."""
    _reject_low_precision_vreg_operands(lhs, rhs, context="pto.vadd(...)")
    rt = result_type if result_type is not None else lhs.type
    return wrap_surface_value(
        _pto.VaddOp(
            _resolve(rt),
            unwrap_surface_value(lhs),
            unwrap_surface_value(rhs),
            unwrap_surface_value(mask),
        ).result
    )


def vsub(lhs, rhs, mask):
    """``pto.vsub`` – element-wise subtract."""
    return _emit_binary_vec_op(_pto.VsubOp, lhs, rhs, mask)


def vmul(lhs, rhs, mask):
    """``pto.vmul`` – element-wise multiply."""
    return _emit_binary_vec_op(_pto.VmulOp, lhs, rhs, mask)


def vmax(lhs, rhs, mask):
    """``pto.vmax`` – element-wise maximum."""
    return _emit_binary_vec_op(_pto.VmaxOp, lhs, rhs, mask)


def vmin(lhs, rhs, mask):
    """``pto.vmin`` – element-wise minimum."""
    return _emit_binary_vec_op(_pto.VminOp, lhs, rhs, mask)


def vand(lhs, rhs, mask):
    """``pto.vand`` – element-wise bitwise and."""
    return _emit_binary_vec_op(_pto.VandOp, lhs, rhs, mask)


def vor(lhs, rhs, mask):
    """``pto.vor`` – element-wise bitwise or."""
    return _emit_binary_vec_op(_pto.VorOp, lhs, rhs, mask)


def vxor(lhs, rhs, mask):
    """``pto.vxor`` – element-wise bitwise xor."""
    return _emit_binary_vec_op(_pto.VxorOp, lhs, rhs, mask)


def vdiv(lhs, rhs, mask):
    """``pto.vdiv`` – element-wise divide."""
    return _emit_binary_vec_op(_pto.VdivOp, lhs, rhs, mask)


def vtrc(inp, mask, *, rnd="Z"):
    """``pto.vtrc`` – truncate/round vector lanes according to *rnd*."""
    _reject_low_precision_vreg_operands(inp, context="pto.vtrc(...)")
    return wrap_surface_value(
        _pto.VtrcOp(
            unwrap_surface_value(inp).type,
            unwrap_surface_value(inp),
            unwrap_surface_value(mask),
            _normalize_vcvt_round_mode(rnd, context="vtrc(..., rnd=...)"),
        ).result
    )


def vshl(lhs, rhs, mask):
    """``pto.vshl`` – element-wise shift left."""
    return _emit_binary_vec_op(_pto.VshlOp, lhs, rhs, mask)


def vshr(lhs, rhs, mask):
    """``pto.vshr`` – element-wise shift right."""
    return _emit_binary_vec_op(_pto.VshrOp, lhs, rhs, mask)


def vcmax(v, mask):
    """``pto.vcmax`` – cross-lane maximum reduction."""
    return _emit_unary_vec_op(_pto.VcmaxOp, v, mask)


def vcadd(v, mask):
    """``pto.vcadd`` – cross-lane add (sum reduction)."""
    _reject_low_precision_vreg_operands(v, context="pto.vcadd(...)")
    raw_v = unwrap_surface_value(v)
    input_type = _pto.VRegType(raw_v.type)
    elem_type = input_type.element_type
    result_elem_type = elem_type
    result_lanes = input_type.element_count
    if IntegerType.isinstance(elem_type):
        int_type = IntegerType(elem_type)
        if int_type.width == 8:
            if int_type.is_unsigned:
                result_elem_type = IntegerType.get_unsigned(16)
            elif int_type.is_signed:
                result_elem_type = IntegerType.get_signed(16)
            else:
                result_elem_type = IntegerType.get_signless(16)
            result_lanes = input_type.element_count // 2
        elif int_type.width == 16:
            if int_type.is_unsigned:
                result_elem_type = IntegerType.get_unsigned(32)
            elif int_type.is_signed:
                result_elem_type = IntegerType.get_signed(32)
            else:
                result_elem_type = IntegerType.get_signless(32)
            result_lanes = input_type.element_count // 2
    result_type = _resolve(vreg_type(result_lanes, result_elem_type))
    return wrap_surface_value(
        _pto.VcaddOp(
            result_type,
            raw_v,
            unwrap_surface_value(mask),
        ).result
    )


def vcmin(v, mask):
    """``pto.vcmin`` – cross-lane minimum reduction."""
    return _emit_unary_vec_op(_pto.VcminOp, v, mask)


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


def _mask_granularity_bits(mask_value, *, context: str) -> int:
    mask_bits, _ = _infer_mask_metadata(mask_value, context=context)
    return mask_bits


def _infer_vdup_scalar_result_type(input_value, mask_value, *, context: str):
    scalar_raw = unwrap_surface_value(input_value)
    scalar_type = scalar_raw.type
    mask_bits = _mask_granularity_bits(mask_value, context=context)
    if IntegerType.isinstance(scalar_type):
        scalar_type = _strip_integer_signedness(scalar_raw)
        scalar_width = IntegerType(scalar_type.type).width
        if scalar_width != mask_bits:
            raise TypeError(
                f"{context} expects scalar input width {scalar_width} to match mask granularity b{mask_bits}"
            )
        element_type = scalar_type.type
    elif F16Type.isinstance(scalar_type) or BF16Type.isinstance(scalar_type):
        if mask_bits != 16:
            raise TypeError(f"{context} expects f16/bf16 scalar input to pair with mask_b16, got mask_b{mask_bits}")
        element_type = scalar_type
    elif F32Type.isinstance(scalar_type):
        if mask_bits != 32:
            raise TypeError(f"{context} expects f32 scalar input to pair with mask_b32, got mask_b{mask_bits}")
        element_type = scalar_type
    else:
        raise TypeError(
            f"{context} only supports scalar input types i8/i16/i32, si8/si16/si32, ui8/ui16/ui32, f16, bf16, and f32; got {scalar_type}"
        )
    return _resolve(vreg_type(_elements_per_vreg(element_type), element_type))


def _coerce_vdup_scalar_input(input_value, mask_value, *, context: str):
    raw_input = unwrap_surface_value(input_value)
    if hasattr(raw_input, "type"):
        return raw_input

    mask_bits = _mask_granularity_bits(mask_value, context=context)
    if isinstance(raw_input, bool):
        raise TypeError(f"{context} does not accept bool literals")
    if isinstance(raw_input, float):
        if mask_bits == 16:
            target_type = F16Type.get()
        elif mask_bits == 32:
            target_type = F32Type.get()
        else:
            raise TypeError(f"{context} cannot materialize a float literal for mask_b{mask_bits}")
        return coerce_scalar_to_type(raw_input, target_type, context=context)
    return coerce_scalar_to_type(raw_input, IntegerType.get_signless(mask_bits), context=context)


def vdup(input_value, mask, position=None):
    """``pto.vdup`` – duplicate a scalar or selected vector lane into active lanes."""
    raw_input = unwrap_surface_value(input_value)
    try:
        _pto.VRegType(raw_input.type)
        _reject_low_precision_vreg_operands(input_value, context="pto.vdup(vec, mask, position=...)")
        result_type = raw_input.type
        normalized_position = (
            _normalize_vdup_position_mode(position, context="vdup(vec, mask, position=...)")
            if position is not None
            else "LOWEST"
        )
    except Exception:
        if position is not None:
            raise TypeError("vdup(scalar, mask, position=...) does not support position; position is only valid for vector input")
        raw_input = _coerce_vdup_scalar_input(input_value, mask, context="vdup(scalar, mask)")
        result_type = _infer_vdup_scalar_result_type(raw_input, mask, context="vdup(scalar, mask)")
        normalized_position = None
    return wrap_surface_value(
        _pto.VdupOp(
            result_type,
            raw_input,
            unwrap_surface_value(mask),
            position=normalized_position,
        ).result
    )


def vln(inp, mask):
    """``pto.vln`` – element-wise natural logarithm."""
    return _emit_unary_vec_op(_pto.VlnOp, inp, mask)


def vsqrt(inp, mask):
    """``pto.vsqrt`` – element-wise square root."""
    return _emit_unary_vec_op(_pto.VsqrtOp, inp, mask)


def vabs(inp, mask):
    """``pto.vabs`` – element-wise absolute value."""
    return _emit_unary_vec_op(_pto.VabsOp, inp, mask)


def vneg(inp, mask):
    """``pto.vneg`` – element-wise negation."""
    return _emit_unary_vec_op(_pto.VnegOp, inp, mask)


def vrelu(inp, mask):
    """``pto.vrelu`` – element-wise ReLU."""
    return _emit_unary_vec_op(_pto.VreluOp, inp, mask)


def vnot(inp, mask):
    """``pto.vnot`` – element-wise bitwise/logical not."""
    return _emit_unary_vec_op(_pto.VnotOp, inp, mask)


def vexpdif(inp, ref, mask, part: str = "ODD"):
    """``pto.vexpdif`` – ``exp(inp - ref)`` selecting ODD or EVEN lanes."""
    _reject_low_precision_vreg_operands(inp, ref, context="pto.vexpdif(...)")
    return wrap_surface_value(
        _pto.VexpdifOp(
            unwrap_surface_value(inp).type,
            unwrap_surface_value(inp),
            unwrap_surface_value(ref),
            unwrap_surface_value(mask),
            part,
        ).result
    )


def vexp(inp, mask):
    """``pto.vexp`` – element-wise exponential."""
    return _emit_unary_vec_op(_pto.VexpOp, inp, mask)


def vrec(inp, mask):
    """``pto.vrec`` – reciprocal, surfaced as ``1 / inp``."""
    zero_vec = vmuls(inp, 0, mask)
    one_vec = vadds(zero_vec, 1, mask)
    return vdiv(one_vec, inp, mask)


def vrsqrt(inp, mask):
    """``pto.vrsqrt`` – inverse square root, surfaced as ``1 / sqrt(inp)``."""
    sqrt_vec = vsqrt(inp, mask)
    return vrec(sqrt_vec, mask)


def vcgmax(v, mask):
    """``pto.vcgmax`` – group maximum reduction."""
    _reject_low_precision_vreg_operands(v, context="pto.vcgmax(...)")
    return wrap_surface_value(
        _pto.VcgmaxOp(
            unwrap_surface_value(v).type,
            unwrap_surface_value(v),
            unwrap_surface_value(mask),
        ).result
    )


def vcgadd(v, mask):
    """``pto.vcgadd`` – group sum reduction."""
    _reject_low_precision_vreg_operands(v, context="pto.vcgadd(...)")
    return wrap_surface_value(
        _pto.VcgaddOp(
            unwrap_surface_value(v).type,
            unwrap_surface_value(v),
            unwrap_surface_value(mask),
        ).result
    )


def vcgmin(v, mask):
    """``pto.vcgmin`` – group minimum reduction."""
    _reject_low_precision_vreg_operands(v, context="pto.vcgmin(...)")
    return wrap_surface_value(
        _pto.VcgminOp(
            unwrap_surface_value(v).type,
            unwrap_surface_value(v),
            unwrap_surface_value(mask),
        ).result
    )


def vcpadd(v, mask):
    """``pto.vcpadd`` – inclusive prefix sum."""
    return _emit_unary_vec_op(_pto.VcpaddOp, v, mask)


def vprelu(lhs, rhs, mask):
    """``pto.vprelu`` – vector parametric ReLU."""
    return _emit_binary_vec_op(_pto.VpreluOp, lhs, rhs, mask)


def vintlv(lhs, rhs):
    """``pto.vintlv`` – interleave two vectors and return the low/high pair."""
    _reject_low_precision_vreg_operands(lhs, rhs, context="pto.vintlv(...)")
    low, high = _pto.VintlvOp(
        unwrap_surface_value(lhs).type,
        unwrap_surface_value(rhs).type,
        unwrap_surface_value(lhs),
        unwrap_surface_value(rhs),
    ).results
    return wrap_surface_value(low), wrap_surface_value(high)


def vdintlv(lhs, rhs):
    """``pto.vdintlv`` – deinterleave two vectors and return the low/high pair."""
    _reject_low_precision_vreg_operands(lhs, rhs, context="pto.vdintlv(...)")
    low, high = _pto.VdintlvOp(
        unwrap_surface_value(lhs).type,
        unwrap_surface_value(rhs).type,
        unwrap_surface_value(lhs),
        unwrap_surface_value(rhs),
    ).results
    return wrap_surface_value(low), wrap_surface_value(high)


def vselr(src0, src1):
    """``pto.vselr`` – vector select/reorder helper."""
    _reject_low_precision_vreg_operands(src0, src1, context="pto.vselr(...)")
    return wrap_surface_value(
        _pto.VselrOp(
            unwrap_surface_value(src0).type,
            unwrap_surface_value(src0),
            unwrap_surface_value(src1),
        ).result
    )


def vaddc(lhs, rhs, mask):
    """``pto.vaddc`` – vector add with carry-out predicate."""
    _reject_low_precision_vreg_operands(lhs, rhs, context="pto.vaddc(...)")
    carry_type = unwrap_surface_value(mask).type
    result, carry = _pto.VaddcOp(
        unwrap_surface_value(lhs).type,
        carry_type,
        unwrap_surface_value(lhs),
        unwrap_surface_value(rhs),
        unwrap_surface_value(mask),
    ).results
    return wrap_surface_value(result), wrap_surface_value(carry)


def vaddcs(lhs, rhs, carry_in, mask):
    """``pto.vaddcs`` – vector add with carry-in and carry-out."""
    _reject_low_precision_vreg_operands(lhs, rhs, context="pto.vaddcs(...)")
    carry_type = unwrap_surface_value(carry_in).type
    result, carry = _pto.VaddcsOp(
        unwrap_surface_value(lhs).type,
        carry_type,
        unwrap_surface_value(lhs),
        unwrap_surface_value(rhs),
        unwrap_surface_value(carry_in),
        unwrap_surface_value(mask),
    ).results
    return wrap_surface_value(result), wrap_surface_value(carry)


def vmull(lhs, rhs, mask):
    """``pto.vmull`` – widening vector multiply returning low/high vectors."""
    _reject_low_precision_vreg_operands(lhs, rhs, context="pto.vmull(...)")
    low, high = _pto.VmullOp(
        unwrap_surface_value(lhs).type,
        unwrap_surface_value(rhs).type,
        unwrap_surface_value(lhs),
        unwrap_surface_value(rhs),
        unwrap_surface_value(mask),
    ).results
    return wrap_surface_value(low), wrap_surface_value(high)


def vbitsort(destination, source, indices, repeat_times):
    """``pto.vbitsort`` – bitonic-sort vector tile primitive."""
    _pto.VbitsortOp(
        unwrap_surface_value(destination),
        unwrap_surface_value(source),
        unwrap_surface_value(indices),
        _coerce_index(repeat_times, context="vbitsort(repeat_times)"),
    )


def vmrgsort4(destination, source0, source1, source2, source3, count, config):
    """``pto.vmrgsort4`` – four-way merge-sort primitive."""
    _pto.Vmrgsort4Op(
        unwrap_surface_value(destination),
        unwrap_surface_value(source0),
        unwrap_surface_value(source1),
        unwrap_surface_value(source2),
        unwrap_surface_value(source3),
        _coerce_i64(count, context="vmrgsort4(count)"),
        _coerce_i64(config, context="vmrgsort4(config)"),
    )


def load_scalar(ptr_value, offset=0, result_type=None):
    """``pto.load_scalar`` – load one scalar from a pointer-like value."""
    if result_type is None:
        result_type = _pointer_element_type(ptr_value, context="load_scalar(ptr)")
    return wrap_surface_value(
        _pto.LoadScalarOp(
            _resolve(result_type),
            unwrap_surface_value(ptr_value),
            _coerce_index(offset, context="load_scalar(offset)"),
        ).value
    )


def store_scalar(ptr_value, offset, value):
    """``pto.store_scalar`` – store one scalar to a pointer-like value."""
    _pto.StoreScalarOp(
        unwrap_surface_value(ptr_value),
        _coerce_index(offset, context="store_scalar(offset)"),
        unwrap_surface_value(value),
    )


def vadds(inp, scalar, mask):
    """``pto.vadds`` – vector plus scalar under mask."""
    return _emit_vec_scalar_masked_op(_pto.VaddsOp, inp, scalar, mask, context="vadds")


def vsubs(inp, scalar, mask):
    """``pto.vsubs`` – vector minus scalar under mask."""
    raw_scalar = _coerce_scalar_like_vector_element(inp, scalar, context="vsubs")
    neg_scalar = _negate_runtime_scalar(raw_scalar)
    return wrap_surface_value(
        _pto.VaddsOp(
            unwrap_surface_value(inp).type,
            unwrap_surface_value(inp),
            neg_scalar,
            unwrap_surface_value(mask),
        ).result
    )


def vmuls(inp, scalar, mask):
    """``pto.vmuls`` – vector times scalar under mask."""
    return _emit_vec_scalar_masked_op(_pto.VmulsOp, inp, scalar, mask, context="vmuls")


def vmaxs(inp, scalar, mask):
    """``pto.vmaxs`` – vector/scalar maximum under mask."""
    return _emit_vec_scalar_masked_op(_pto.VmaxsOp, inp, scalar, mask, context="vmaxs")


def vmins(inp, scalar, mask):
    """``pto.vmins`` – vector/scalar minimum under mask."""
    return _emit_vec_scalar_masked_op(_pto.VminsOp, inp, scalar, mask, context="vmins")


def vlrelu(inp, alpha, mask):
    """``pto.vlrelu`` – vector leaky ReLU under mask."""
    return _emit_vec_scalar_masked_op(_pto.VlreluOp, inp, alpha, mask, context="vlrelu")


def vshrs(inp, scalar, mask):
    """``pto.vshrs`` – vector shift-right by a uniform ``i16`` amount."""
    _reject_low_precision_vreg_operands(inp, context="pto.vshrs(...)")
    i16 = IntegerType.get_signless(16)
    raw = unwrap_surface_value(scalar)
    if hasattr(raw, "type"):
        scalar_value = coerce_scalar_to_type(raw, i16, context="vshrs")
    else:
        scalar_value = materialize_scalar_literal(int(raw), i16, context="vshrs")
    return wrap_surface_value(
        _pto.VshrsOp(
            unwrap_surface_value(inp).type,
            unwrap_surface_value(inp),
            unwrap_surface_value(scalar_value),
            unwrap_surface_value(mask),
        ).result
    )


def vshls(inp, scalar, mask):
    """``pto.vshls`` – vector shift-left by a uniform ``i16`` amount."""
    _reject_low_precision_vreg_operands(inp, context="pto.vshls(...)")
    i16 = IntegerType.get_signless(16)
    raw = unwrap_surface_value(scalar)
    if hasattr(raw, "type"):
        scalar_value = coerce_scalar_to_type(raw, i16, context="vshls")
    else:
        scalar_value = materialize_scalar_literal(int(raw), i16, context="vshls")
    return wrap_surface_value(
        _pto.VshlsOp(
            unwrap_surface_value(inp).type,
            unwrap_surface_value(inp),
            unwrap_surface_value(scalar_value),
            unwrap_surface_value(mask),
        ).result
    )


def vands(inp, scalar, mask):
    """``pto.vands`` – vector AND scalar (emulated via ``vand`` + ``vbr``)."""
    return vand(
        inp,
        vbr(_coerce_scalar_like_vector_element(inp, scalar, context="vands")),
        mask,
    )


def vors(inp, scalar, mask):
    """``pto.vors`` – vector OR scalar (emulated via ``vor`` + ``vbr``)."""
    return vor(
        inp,
        vbr(_coerce_scalar_like_vector_element(inp, scalar, context="vors")),
        mask,
    )


def vxors(inp, scalar, mask):
    """``pto.vxors`` – vector XOR scalar (emulated via ``vxor`` + ``vbr``)."""
    return vxor(
        inp,
        vbr(_coerce_scalar_like_vector_element(inp, scalar, context="vxors")),
        mask,
    )


def vaddrelu(lhs, rhs, mask):
    """``pto.vaddrelu`` – add, then apply ReLU."""
    return vrelu(vadd(lhs, rhs, mask), mask)


def vsubrelu(lhs, rhs, mask):
    """``pto.vsubrelu`` – subtract, then apply ReLU."""
    return vrelu(vsub(lhs, rhs, mask), mask)


def vaxpy(alpha, x, y, mask):
    """``pto.vaxpy`` – fused ``alpha * x + y``."""
    _reject_low_precision_vreg_operands(x, y, context="pto.vaxpy(...)")
    alpha_value = _coerce_scalar_like_vector_element(x, alpha, context="vaxpy")
    return wrap_surface_value(
        _pto.VaxpyOp(
            unwrap_surface_value(x).type,
            unwrap_surface_value(x),
            unwrap_surface_value(y),
            unwrap_surface_value(alpha_value),
            unwrap_surface_value(mask),
        ).result
    )


def vmula(acc, lhs, rhs, mask):
    """``pto.vmula`` – fused ``acc + lhs * rhs`` under mask."""
    _reject_low_precision_vreg_operands(acc, lhs, rhs, context="pto.vmula(...)")
    return wrap_surface_value(
        _pto.VmulaOp(
            unwrap_surface_value(acc).type,
            unwrap_surface_value(acc),
            unwrap_surface_value(lhs),
            unwrap_surface_value(rhs),
            unwrap_surface_value(mask),
        ).result
    )

def vmadd(acc, lhs, rhs, mask):
    """``pto.vmadd`` – fused multiply-add: ``acc * lhs + rhs`` (single rounding)."""
    _reject_low_precision_vreg_operands(acc, lhs, rhs, context="pto.vmadd(...)")
    return wrap_surface_value(
        _pto.VmaddOp(
            unwrap_surface_value(acc).type,
            unwrap_surface_value(acc),
            unwrap_surface_value(lhs),
            unwrap_surface_value(rhs),
            unwrap_surface_value(mask),
        ).result
    )

def vci(base, order=None):
    """``pto.vci`` – generate lane indices from a scalar base."""
    raw_base = unwrap_surface_value(base)
    if hasattr(raw_base, "type"):
        scalar_value = raw_base
        elem_type = raw_base.type
    elif isinstance(raw_base, int):
        elem_type = IntegerType.get_signless(32)
        scalar_value = materialize_scalar_literal(raw_base, elem_type, context="vci(base)")
    else:
        raise TypeError("vci(base) expects a runtime scalar or Python int")

    result_type = _resolve(vreg_type(_elements_per_vreg(elem_type), elem_type))
    kwargs = {}
    if order is not None:
        kwargs["order"] = order
    return wrap_surface_value(_pto.VciOp(result_type, scalar_value, **kwargs).result)


def vsel(true_v, false_v, mask):
    """``pto.vsel`` – element-wise select under a predicate mask."""
    _reject_low_precision_vreg_operands(true_v, false_v, context="pto.vsel(...)")
    return wrap_surface_value(
        _pto.VselOp(
            unwrap_surface_value(true_v).type,
            unwrap_surface_value(true_v),
            unwrap_surface_value(false_v),
            unwrap_surface_value(mask),
        ).result
    )


# ── Tile-domain operations ────────────────────────────────────────────────────

def _coerce_tensor_view_layout_attr(layout):
    if layout is None:
        return None
    if isinstance(layout, str):
        canonical = layout.upper()
        if canonical not in {"ND", "DN", "NZ"}:
            raise make_tensor_view_invalid_layout_error(layout)
        return _pto.LayoutAttr.get(getattr(_pto.Layout, canonical))
    if isinstance(layout, Attribute):
        return layout
    try:
        return _pto.LayoutAttr.get(layout)
    except Exception as exc:  # pragma: no cover - defensive pybind fallback
        raise make_tensor_view_invalid_layout_error(layout) from exc


def make_tensor_view(ptr, *, shape=None, strides=None, layout=None):
    """
    ``pto.make_tensor_view`` – wrap a pointer as a tensor view.

    Type is inferred: rank from ``len(shape)``, element type from ``ptr``.
    """
    if shape is None or strides is None:
        raise make_tensor_view_missing_metadata_error(ptr)
    ptr = resolve_tensor_data_entry(ptr)
    rank = len(shape)
    raw_ptr = unwrap_surface_value(ptr)
    elem = _pto.PtrType(raw_ptr.type).element_type
    normalized_shape = [
        _coerce_index(dim, context="make_tensor_view(shape=...)")
        for dim in shape
    ]
    normalized_strides = [
        _coerce_index(dim, context="make_tensor_view(strides=...)")
        for dim in strides
    ]
    static_dims = _static_index_dims(normalized_shape)
    tv_type = (
        tensor_view_type_from_dims(static_dims, elem)
        if static_dims is not None
        else tensor_view_type(rank, elem)
    )
    layout_attr = _coerce_tensor_view_layout_attr(layout)
    value = _pto.MakeTensorViewOp(
        tv_type,
        raw_ptr,
        _unwrap_sequence(normalized_shape),
        _unwrap_sequence(normalized_strides),
        layout=layout_attr,
    ).result
    return TensorViewValue(value, shape=tuple(shape), strides=tuple(strides))


def _normalize_static_tile_shape(shape):
    static_shape = []
    for dim in shape:
        if isinstance(dim, bool) or not isinstance(dim, int):
            raise TypeError(
                "alloc_tile(shape=...) currently requires a static physical tile shape. "
                "Use constexpr/static integers for shape and place runtime metadata in valid_shape."
            )
        static_shape.append(dim)
    return tuple(static_shape)


def _authored_tile_physical_shape(shape):
    if len(shape) == 1:
        return (1, shape[0])
    return tuple(shape)


def _split_valid_shape(shape, valid_shape):
    logical_rank = len(shape)
    if valid_shape is None:
        return _authored_tile_physical_shape(shape), None, None, tuple(shape)

    if len(valid_shape) != logical_rank:
        raise TypeError(
            f"alloc_tile(valid_shape=...) rank mismatch: expected {logical_rank} dims, got {len(valid_shape)}"
        )

    surface_valid_shape = []
    if logical_rank == 1:
        dim = valid_shape[0]
        surface_valid_shape.append(dim)
        if isinstance(dim, bool):
            raise TypeError("alloc_tile(valid_shape=...) does not accept bool dimensions")
        if isinstance(dim, int):
            return (1, dim), None, None, tuple(surface_valid_shape)
        return (-1, -1), 1, dim, tuple(surface_valid_shape)

    type_valid_shape = []
    valid_row = None
    valid_col = None
    for index, dim in enumerate(valid_shape):
        surface_valid_shape.append(dim)
        if isinstance(dim, bool):
            raise TypeError("alloc_tile(valid_shape=...) does not accept bool dimensions")
        if isinstance(dim, int):
            type_valid_shape.append(dim)
            continue
        type_valid_shape.append(-1)
        if index == 0:
            valid_row = dim
            continue
        if index == 1:
            valid_col = dim
            continue
        raise TypeError(
            "alloc_tile(valid_shape=...) currently only supports dynamic runtime metadata "
            "for the first two dimensions"
        )
    return tuple(type_valid_shape), valid_row, valid_col, tuple(surface_valid_shape)


def _uses_row_major_none_box_layout(blayout, slayout) -> bool:
    return str(blayout).lower() == "rowmajor" and str(slayout).lower() == "nonebox"


def _validate_authored_tile_row_alignment(shape, dtype, *, blayout, slayout):
    if not _uses_row_major_none_box_layout(blayout, slayout):
        return
    if not shape:
        return
    elem_bytewidth = _element_bytewidth(_resolve(dtype))
    row_bytes = shape[-1] * elem_bytewidth
    required_alignment = 32
    if row_bytes % required_alignment == 0:
        return
    raise tile_row_alignment_error(
        shape=shape,
        dtype=str(_resolve(dtype)),
        row_bytes=row_bytes,
        required_alignment=required_alignment,
    )


def partition_view(tv, *, offsets, sizes):
    """
    ``pto.partition_view`` – slice a tensor view.

    Type is inferred from the source tensor-view type.
    """
    spec = compose_partition_spec(tv, offsets=offsets, sizes=sizes)
    if spec is not None:
        source = spec.root_tensor_view
        offsets = spec.offsets
        sizes = spec.sizes
    else:
        source = tv

    raw_source = unwrap_surface_value(source)
    src_type = _pto.TensorViewType(raw_source.type)
    rank = src_type.rank
    elem = src_type.element_type
    normalized_offsets = [
        _coerce_index(offset, context="partition_view(offsets=...)")
        for offset in offsets
    ]
    normalized_sizes = [
        _coerce_index(size, context="partition_view(sizes=...)")
        for size in sizes
    ]
    static_dims = _static_index_dims(normalized_sizes)
    ptv_type = (
        part_tensor_view_type_from_dims(static_dims, elem)
        if static_dims is not None
        else part_tensor_view_type(rank, elem)
    )
    value = _pto.PartitionViewOp(
        ptv_type,
        raw_source,
        _unwrap_sequence(normalized_offsets),
        _unwrap_sequence(normalized_sizes),
    ).result
    return wrap_surface_value(
        value,
        root_tensor_view=source if spec is None else spec.root_tensor_view,
        offsets=tuple(offsets),
        sizes=tuple(sizes),
    )


def _tile_logical_rank(tile, *, context: str) -> int:
    shape = getattr(tile, "shape", None)
    if shape is not None:
        return len(shape)
    parsed = parse_tile_type_metadata(unwrap_surface_value(tile).type)
    if parsed is not None:
        return len(parsed["shape_dims"])
    raise TypeError(f"{context} requires tile shape metadata to infer sizes")


def _source_view_rank(tv, *, context: str) -> int:
    shape = getattr(tv, "shape", None)
    if shape is not None:
        return len(shape)
    raw_type = unwrap_surface_value(tv).type
    try:
        return _pto.TensorViewType(raw_type).rank
    except Exception:
        try:
            return _pto.PartitionTensorViewType(raw_type).rank
        except Exception as exc:
            raise TypeError(f"{context} expects a tensor view or partition tensor view, got {raw_type}") from exc


def _is_partition_tensor_view(value) -> bool:
    try:
        _pto.PartitionTensorViewType(unwrap_surface_value(value).type)
        return True
    except Exception:
        return False


def _normalize_transfer_offsets(tv, *, offsets, context: str):
    if offsets is None:
        return [0] * _source_view_rank(tv, context=context)
    if isinstance(offsets, tuple):
        return list(offsets)
    if isinstance(offsets, list):
        return offsets
    return [offsets]


def _normalize_transfer_sizes(sizes):
    if isinstance(sizes, tuple):
        return list(sizes)
    if isinstance(sizes, list):
        return sizes
    return [sizes]


def _infer_tile_transfer_sizes(tile, *, context: str):
    valid_shape = getattr(tile, "valid_shape", None)
    if valid_shape is None:
        raise TypeError(f"{context} requires tile valid_shape metadata to infer sizes")
    sizes = []
    for index in range(_tile_logical_rank(tile, context=context)):
        try:
            dim = valid_shape[index]
        except Exception as exc:
            raise TypeError(
                f"{context} could not read tile.valid_shape[{index}] to infer sizes; "
                "pass sizes= explicitly"
            ) from exc
        if dim is None:
            raise ValueError(
                f"{context} cannot infer partition sizes because tile.valid_shape[{index}] is None; "
                "pass sizes= explicitly"
            )
        sizes.append(dim)
    return sizes


def _tile_transfer_partition(tv, tile, *, offsets=None, sizes=None, context: str):
    normalized_offsets = _normalize_transfer_offsets(
        tv,
        offsets=offsets,
        context=context,
    )
    normalized_sizes = (
        _infer_tile_transfer_sizes(tile, context=context)
        if sizes is None
        else _normalize_transfer_sizes(sizes)
    )
    if len(normalized_offsets) != len(normalized_sizes):
        if sizes is None:
            raise ValueError(
                f"{context} cannot infer partition sizes for rank-{len(normalized_offsets)} view "
                f"from rank-{len(normalized_sizes)} tile; pass sizes= explicitly"
            )
        raise ValueError(
            f"{context} expects offset rank and sizes rank to match, got "
            f"{len(normalized_offsets)} and {len(normalized_sizes)}"
        )
    return partition_view(tv, offsets=normalized_offsets, sizes=normalized_sizes)


def alloc_buffer(shape, dtype, **kwargs):
    """
    Allocate explicit-body scratch storage and return an address-like value.

    The allocation emits an LLVM stack allocation in the surrounding explicit
    kernel or helper body. UB scratch uses explicit ``pto.castptr`` /
    ``pto.addptr`` pointer authoring and an appropriate host launch wrapper.
    """
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(
            f"pto.alloc_buffer(...) does not accept keyword argument(s): {unexpected}. "
            "It only allocates explicit-body local buffers; author UB scratch explicitly with "
            "pto.castptr/pto.addptr and pass the dynamic UB byte count at launch."
        )
    _require_explicit_mode("pto.alloc_buffer(...)")
    element_type = _resolve(dtype)
    element_count = _static_alloc_buffer_element_count(shape)
    elem_bytes = _element_bytewidth(element_type)
    byte_size = element_count * elem_bytes

    return _alloc_local_buffer(
        shape,
        dtype,
        element_type,
        element_count,
        byte_size,
    )


def _static_alloc_buffer_element_count(shape):
    if isinstance(shape, int):
        dims = (shape,)
    elif isinstance(shape, (list, tuple)):
        dims = tuple(shape)
    else:
        raise TypeError("pto.alloc_buffer(shape, ...) expects an int or a tuple/list of static dimensions")
    if not dims:
        raise ValueError("pto.alloc_buffer(shape, ...) expects at least one dimension")
    count = 1
    for dim in dims:
        raw_dim = unwrap_surface_value(dim)
        if isinstance(raw_dim, bool):
            raise TypeError("pto.alloc_buffer(shape, ...) does not accept bool dimensions")
        if not isinstance(raw_dim, int):
            raise TypeError(
                "pto.alloc_buffer(shape, ...) requires static integer dimensions; "
                f"got {getattr(raw_dim, 'type', type(raw_dim).__name__)}"
            )
        if raw_dim <= 0:
            raise ValueError(f"pto.alloc_buffer(shape, ...) dimensions must be positive, got {raw_dim}")
        count *= raw_dim
    return count


def _alloc_local_buffer(shape, dtype, element_type, element_count, byte_size):
    i32 = IntegerType.get_signless(32)
    count = _materialize_integer_literal(i32, element_count)
    llvm_ptr_type = Type.parse("!llvm.ptr")
    attributes = {
        "elem_type": TypeAttr.get(element_type),
    }
    if _is_persistent_alloc_buffer_candidate():
        attributes["pto.persistent"] = UnitAttr.get()
    alloca = Operation.create(
        "llvm.alloca",
        results=[llvm_ptr_type],
        operands=[count],
        attributes=attributes,
    ).results[0]
    return AllocatedBufferValue(
        alloca,
        shape=_normalize_alloc_buffer_shape_metadata(shape),
        dtype=dtype,
        element_type=element_type,
        element_count=element_count,
        byte_size=byte_size,
    )


def _is_persistent_alloc_buffer_candidate() -> bool:
    try:
        from ._tracing.active import current_session
        session = current_session()
    except Exception:
        session = None
    if session is None:
        return False
    if getattr(session.module_spec, "entry", False) is not True:
        return False
    if session.current_function is not session.entry_function:
        return False
    return session.current_subkernel is None


def _normalize_alloc_buffer_shape_metadata(shape):
    if isinstance(shape, int):
        return (shape,)
    return tuple(unwrap_surface_value(dim) for dim in shape)


def alloc_tile(
    tile_type=None,
    *,
    shape=None,
    dtype=None,
    memory_space="ub",
    valid_shape=None,
    blayout: str = "RowMajor",
    slayout: str = "NoneBox",
    fractal_size: int = 512,
    pad: str = "Null",
    addr=None,
    valid_row=None,
    valid_col=None,
):
    """
    ``pto.alloc_tile``.

    Accepts either the authored surface form:

    ``alloc_tile(shape=[...], dtype=..., memory_space=..., valid_shape=..., addr=...)``

    or the low-level explicit-type form:

    ``alloc_tile(tile_type, addr=..., valid_row=..., valid_col=...)``.
    """
    if tile_type is not None and shape is not None:
        raise TypeError("alloc_tile() accepts either tile_type or shape=/dtype=, not both")

    if tile_type is None:
        if shape is None or dtype is None:
            raise TypeError("alloc_tile() requires either tile_type or both shape= and dtype=")
        if valid_row is not None or valid_col is not None:
            raise TypeError(
                "alloc_tile(shape=..., dtype=...) uses the authored surface form; "
                "use valid_shape=... instead of valid_row=/valid_col="
            )
        logical_shape = _normalize_static_tile_shape(shape)
        physical_shape = _authored_tile_physical_shape(logical_shape)
        _validate_authored_tile_row_alignment(physical_shape, dtype, blayout=blayout, slayout=slayout)
        type_valid_shape, valid_row, valid_col, surface_valid_shape = _split_valid_shape(logical_shape, valid_shape)
        from ._types import tile_buf_type
        tile_type = tile_buf_type(
            physical_shape,
            dtype,
            type_valid_shape,
            blayout=blayout,
            address_space=memory_space,
            slayout=slayout,
            fractal_size=fractal_size,
            pad=pad,
        )
        shape = logical_shape
    else:
        physical_shape = None
        surface_valid_shape = None

    value = _pto.AllocTileOp(
        _resolve(tile_type),
        addr=_coerce_i64(addr, context="alloc_tile(addr)") if addr is not None else None,
        valid_row=_coerce_index(valid_row, context="alloc_tile(valid_row)") if valid_row is not None else None,
        valid_col=_coerce_index(valid_col, context="alloc_tile(valid_col)") if valid_col is not None else None,
    ).result
    if tile_type is not None and (valid_row is not None or valid_col is not None):
        parsed_tile_type = parse_tile_type_metadata(_resolve(tile_type))
        rank = len(shape) if shape is not None else len(parsed_tile_type["shape_dims"])
        surface_valid_shape = [None] * rank
        if rank >= 1:
            surface_valid_shape[0] = valid_row
        if rank >= 2:
            surface_valid_shape[1] = valid_col
        surface_valid_shape = tuple(surface_valid_shape)
    return wrap_surface_value(
        value,
        tile_metadata={
            "shape": shape,
            "physical_shape": physical_shape,
            "dtype": dtype,
            "memory_space": memory_space,
            "valid_shape": surface_valid_shape,
        },
    )


def set_tile_valid_shape(tile, valid_shape):
    """Update the runtime valid-shape metadata of an authored dynamic tile."""
    parsed_tile_type = parse_tile_type_metadata(unwrap_surface_value(tile).type)
    if parsed_tile_type is None:
        raise TypeError("tile.valid_shape assignment expects a tile_buf-backed value")
    if len(parsed_tile_type["shape_dims"]) != 2:
        raise TypeError("tile.valid_shape assignment currently only supports rank-2 tiles")
    logical_rank = len(tile.shape) if getattr(tile, "shape", None) is not None else 2
    if logical_rank == 1:
        if len(valid_shape) != 1:
            raise TypeError("rank-1 tile.valid_shape assignment expects exactly one dimension")
        if parsed_tile_type["valid_dims"] != (None, None):
            raise TypeError(
                "rank-1 tile.valid_shape assignment requires a tile allocated with "
                "valid_shape=[...] so the physical valid row/col metadata remain dynamic"
            )
        valid_row = _coerce_index_value(1)
        valid_col = _coerce_index(valid_shape[0], context="tile.valid_shape assignment")
    else:
        if len(valid_shape) != 2:
            raise TypeError("tile.valid_shape assignment currently expects exactly two dimensions")
        if parsed_tile_type["valid_dims"] != (None, None):
            raise TypeError(
                "tile.valid_shape assignment requires a tile allocated with fully dynamic "
                "valid_shape=[..., ...]"
            )
        valid_row = _coerce_index(valid_shape[0], context="tile.valid_shape assignment")
        valid_col = _coerce_index(valid_shape[1], context="tile.valid_shape assignment")
    _pto.SetValidShapeOp(
        unwrap_surface_value(tile),
        valid_row,
        valid_col,
    )


def tload(part, tile):
    """``pto.tload ins(part) outs(tile)``."""
    _pto.TLoadOp(None, unwrap_surface_value(part), unwrap_surface_value(tile))


def tstore(tile, part):
    """``pto.tstore ins(tile) outs(part)``."""
    _pto.TStoreOp(None, unwrap_surface_value(tile), unwrap_surface_value(part))


def tmov(src, dst, *, mode=None):
    """``pto.tmov ins(src) outs(dst)`` – move data between tile domains."""
    kwargs = {}
    if mode is not None:
        kwargs["accToVecMode"] = _normalize_acc_to_vec_mode(mode, context="tmov(..., mode=...)")
    _pto.TMovOp(None, unwrap_surface_value(src), unwrap_surface_value(dst), **kwargs)


def ttrans(src, tmp, dst):
    """``pto.ttrans ins(src, tmp) outs(dst)`` – tile transpose (DPS)."""
    _pto.ttrans(
        unwrap_surface_value(src),
        unwrap_surface_value(tmp),
        unwrap_surface_value(dst),
    )


def textract(src, dst, index_row, index_col):
    """``pto.textract ins(src, index_row, index_col) outs(dst)``."""
    _pto.TExtractOp(
        unwrap_surface_value(src),
        _coerce_index(index_row, context="textract(index_row)"),
        _coerce_index(index_col, context="textract(index_col)"),
        unwrap_surface_value(dst),
    )


def tinsert(src, dst, index_row, index_col):
    """``pto.tinsert ins(src, index_row, index_col) outs(dst)``."""
    _pto.TInsertOp(
        unwrap_surface_value(src),
        _coerce_index(index_row, context="tinsert(index_row)"),
        _coerce_index(index_col, context="tinsert(index_col)"),
        unwrap_surface_value(dst),
    )


def tmatmul(lhs, rhs, dst):
    """``pto.tmatmul ins(lhs, rhs) outs(dst)``."""
    _pto.TMatmulOp(
        None,
        unwrap_surface_value(lhs),
        unwrap_surface_value(rhs),
        unwrap_surface_value(dst),
    )


def tmatmul_acc(acc_in, lhs, rhs, dst):
    """``pto.tmatmul.acc ins(acc_in, lhs, rhs) outs(dst)``."""
    _pto.TMatmulAccOp(
        None,
        unwrap_surface_value(acc_in),
        unwrap_surface_value(lhs),
        unwrap_surface_value(rhs),
        unwrap_surface_value(dst),
    )


def tmatmul_mx(lhs, lhs_scale, rhs, rhs_scale, dst, *, acc_phase=None):
    """``pto.tmatmul.mx ins(lhs, lhs_scale, rhs, rhs_scale) outs(dst)``."""
    _pto.TMatmulMxOp(
        None,
        unwrap_surface_value(lhs),
        unwrap_surface_value(lhs_scale),
        unwrap_surface_value(rhs),
        unwrap_surface_value(rhs_scale),
        unwrap_surface_value(dst),
        accPhase=acc_phase,
    )


def tmatmul_mx_acc(acc_in, lhs, lhs_scale, rhs, rhs_scale, dst, *, acc_phase=None):
    """``pto.tmatmul.mx.acc ins(acc_in, lhs, lhs_scale, rhs, rhs_scale) outs(dst)``."""
    _pto.TMatmulMxAccOp(
        None,
        unwrap_surface_value(acc_in),
        unwrap_surface_value(lhs),
        unwrap_surface_value(lhs_scale),
        unwrap_surface_value(rhs),
        unwrap_surface_value(rhs_scale),
        unwrap_surface_value(dst),
        accPhase=acc_phase,
    )


def tmatmul_mx_bias(lhs, lhs_scale, rhs, rhs_scale, bias, dst):
    """``pto.tmatmul.mx.bias ins(lhs, lhs_scale, rhs, rhs_scale, bias) outs(dst)``."""
    _pto.TMatmulMxBiasOp(
        None,
        unwrap_surface_value(lhs),
        unwrap_surface_value(lhs_scale),
        unwrap_surface_value(rhs),
        unwrap_surface_value(rhs_scale),
        unwrap_surface_value(bias),
        unwrap_surface_value(dst),
    )


def tgemv_mx(lhs, lhs_scale, rhs, rhs_scale, dst, *, acc_phase=None):
    """``pto.tgemv.mx ins(lhs, lhs_scale, rhs, rhs_scale) outs(dst)``."""
    _pto.TGemvMxOp(
        None,
        unwrap_surface_value(lhs),
        unwrap_surface_value(lhs_scale),
        unwrap_surface_value(rhs),
        unwrap_surface_value(rhs_scale),
        unwrap_surface_value(dst),
        accPhase=acc_phase,
    )


def tgemv_mx_acc(acc_in, lhs, lhs_scale, rhs, rhs_scale, dst, *, acc_phase=None):
    """``pto.tgemv.mx.acc ins(acc_in, lhs, lhs_scale, rhs, rhs_scale) outs(dst)``."""
    _pto.TGemvMxAccOp(
        None,
        unwrap_surface_value(acc_in),
        unwrap_surface_value(lhs),
        unwrap_surface_value(lhs_scale),
        unwrap_surface_value(rhs),
        unwrap_surface_value(rhs_scale),
        unwrap_surface_value(dst),
        accPhase=acc_phase,
    )


def tgemv_mx_bias(lhs, lhs_scale, rhs, rhs_scale, bias, dst):
    """``pto.tgemv.mx.bias ins(lhs, lhs_scale, rhs, rhs_scale, bias) outs(dst)``."""
    _pto.TGemvMxBiasOp(
        None,
        unwrap_surface_value(lhs),
        unwrap_surface_value(lhs_scale),
        unwrap_surface_value(rhs),
        unwrap_surface_value(rhs_scale),
        unwrap_surface_value(bias),
        unwrap_surface_value(dst),
    )


def _coerce_tile_scalar_operand(tile, scalar, *, context: str):
    return _constant_like(scalar, infer_tile_element_type(wrap_surface_value(tile)))


def tadd(src0, src1, dst):
    """``pto.tadd ins(src0, src1) outs(dst)``."""
    _pto.tadd(
        unwrap_surface_value(src0),
        unwrap_surface_value(src1),
        unwrap_surface_value(dst),
    )


def taddrelu(src0, src1, dst):
    """``pto.taddrelu ins(src0, src1) outs(dst)``."""
    _require_target_arch("pto.tile.addrelu", {"a2", "a3"})
    _pto.taddrelu(
        unwrap_surface_value(src0),
        unwrap_surface_value(src1),
        unwrap_surface_value(dst),
    )


def tsub(src0, src1, dst):
    """``pto.tsub ins(src0, src1) outs(dst)``."""
    _pto.tsub(
        unwrap_surface_value(src0),
        unwrap_surface_value(src1),
        unwrap_surface_value(dst),
    )


def tmul(src0, src1, dst):
    """``pto.tmul ins(src0, src1) outs(dst)``."""
    _pto.tmul(
        unwrap_surface_value(src0),
        unwrap_surface_value(src1),
        unwrap_surface_value(dst),
    )


def tdiv(src0, src1, dst, *, div_precision=None):
    """``pto.tdiv ins(src0, src1) outs(dst)``."""
    _pto.tdiv(
        unwrap_surface_value(src0),
        unwrap_surface_value(src1),
        unwrap_surface_value(dst),
        precision_type=div_precision,
    )


def tmax(src0, src1, dst):
    """``pto.tmax ins(src0, src1) outs(dst)``."""
    _pto.tmax(
        unwrap_surface_value(src0),
        unwrap_surface_value(src1),
        unwrap_surface_value(dst),
    )


def tmin(src0, src1, dst):
    """``pto.tmin ins(src0, src1) outs(dst)``."""
    _pto.tmin(
        unwrap_surface_value(src0),
        unwrap_surface_value(src1),
        unwrap_surface_value(dst),
    )


def tadds(src, scalar, dst):
    """``pto.tadds ins(src, scalar) outs(dst)``."""
    _pto.tadds(
        unwrap_surface_value(src),
        _coerce_tile_scalar_operand(src, scalar, context="tadds"),
        unwrap_surface_value(dst),
    )


def tsubs(src, scalar, dst):
    """``pto.tsubs ins(src, scalar) outs(dst)``."""
    _pto.tsubs(
        unwrap_surface_value(src),
        _coerce_tile_scalar_operand(src, scalar, context="tsubs"),
        unwrap_surface_value(dst),
    )


def tmuls(src, scalar, dst):
    """``pto.tmuls ins(src, scalar) outs(dst)``."""
    _pto.tmuls(
        unwrap_surface_value(src),
        _coerce_tile_scalar_operand(src, scalar, context="tmuls"),
        unwrap_surface_value(dst),
    )


def tdivs(src, scalar, dst, *, div_precision=None):
    """``pto.tdivs ins(src, scalar) outs(dst)``."""
    _pto.tdivs(
        unwrap_surface_value(src),
        _coerce_tile_scalar_operand(src, scalar, context="tdivs"),
        unwrap_surface_value(dst),
        precision_type=div_precision,
    )


def tmaxs(src, scalar, dst):
    """``pto.tmaxs ins(src, scalar) outs(dst)``."""
    _pto.tmaxs(
        unwrap_surface_value(src),
        _coerce_tile_scalar_operand(src, scalar, context="tmaxs"),
        unwrap_surface_value(dst),
    )


def tmins(src, scalar, dst):
    """``pto.tmins ins(src, scalar) outs(dst)``."""
    _pto.tmins(
        unwrap_surface_value(src),
        _coerce_tile_scalar_operand(src, scalar, context="tmins"),
        unwrap_surface_value(dst),
    )


def texp(src, dst, *, exp_precision=None):
    """``pto.texp ins(src) outs(dst)``."""
    _pto.texp(
        unwrap_surface_value(src),
        unwrap_surface_value(dst),
        precision_type=exp_precision,
    )


def tlog(src, dst, *, log_precision=None):
    """``pto.tlog ins(src) outs(dst)``."""
    _pto.tlog(
        unwrap_surface_value(src),
        unwrap_surface_value(dst),
        precision_type=log_precision,
    )


def tsqrt(src, dst, *, sqrt_precision=None):
    """``pto.tsqrt ins(src) outs(dst)``."""
    _pto.tsqrt(
        unwrap_surface_value(src),
        unwrap_surface_value(dst),
        precision_type=sqrt_precision,
    )


def trsqrt(src, dst, *, tmp=None, rsqrt_precision=None):
    """``pto.trsqrt ins(src, tmp?) outs(dst)``."""
    _pto.trsqrt(
        unwrap_surface_value(src),
        unwrap_surface_value(dst),
        tmp=None if tmp is None else unwrap_surface_value(tmp),
        precision_type=rsqrt_precision,
    )


def trecip(src, dst, *, recip_precision=None):
    """``pto.trecip ins(src) outs(dst)``."""
    _pto.trecip(
        unwrap_surface_value(src),
        unwrap_surface_value(dst),
        precision_type=recip_precision,
    )


def tabs(src, dst):
    """``pto.tabs ins(src) outs(dst)``."""
    _pto.tabs(
        unwrap_surface_value(src),
        unwrap_surface_value(dst),
    )


def tneg(src, dst):
    """``pto.tneg ins(src) outs(dst)``."""
    _pto.tneg(
        unwrap_surface_value(src),
        unwrap_surface_value(dst),
    )


def trelu(src, dst):
    """``pto.trelu ins(src) outs(dst)``."""
    _pto.trelu(
        unwrap_surface_value(src),
        unwrap_surface_value(dst),
    )


def tlrelu(src, slope, dst):
    """``pto.tlrelu ins(src, slope) outs(dst)``."""
    _pto.tlrelu(
        unwrap_surface_value(src),
        _coerce_tile_scalar_operand(src, slope, context="tlrelu"),
        unwrap_surface_value(dst),
    )


def trowsum(src, tmp, dst):
    """``pto.trowsum ins(src, tmp) outs(dst)``."""
    _pto.trowsum(
        unwrap_surface_value(src),
        unwrap_surface_value(tmp),
        unwrap_surface_value(dst),
    )


def trowmax(src, tmp, dst):
    """``pto.trowmax ins(src, tmp) outs(dst)``."""
    _pto.trowmax(
        unwrap_surface_value(src),
        unwrap_surface_value(tmp),
        unwrap_surface_value(dst),
    )


def trowmin(src, tmp, dst):
    """``pto.trowmin ins(src, tmp) outs(dst)``."""
    _pto.trowmin(
        unwrap_surface_value(src),
        unwrap_surface_value(tmp),
        unwrap_surface_value(dst),
    )


def trowprod(src, tmp, dst):
    """``pto.trowprod ins(src, tmp) outs(dst)``."""
    _pto.trowprod(
        unwrap_surface_value(src),
        unwrap_surface_value(tmp),
        unwrap_surface_value(dst),
    )


def trowargmax(src, tmp, dst):
    """``pto.trowargmax ins(src, tmp) outs(dst)``."""
    _pto.trowargmax(
        unwrap_surface_value(src),
        unwrap_surface_value(tmp),
        unwrap_surface_value(dst),
    )


def trowargmin(src, tmp, dst):
    """``pto.trowargmin ins(src, tmp) outs(dst)``."""
    _pto.trowargmin(
        unwrap_surface_value(src),
        unwrap_surface_value(tmp),
        unwrap_surface_value(dst),
    )


def tcolsum(src, dst, *, tmp=None, is_binary=None):
    """``pto.tcolsum ins(src, tmp?) outs(dst)``."""
    _pto.tcolsum(
        unwrap_surface_value(src),
        unwrap_surface_value(dst),
        tmp=None if tmp is None else unwrap_surface_value(tmp),
        is_binary=is_binary,
    )


def tcolmax(src, dst):
    """``pto.tcolmax ins(src) outs(dst)``."""
    _pto.tcolmax(
        unwrap_surface_value(src),
        unwrap_surface_value(dst),
    )


def tcolmin(src, dst):
    """``pto.tcolmin ins(src) outs(dst)``."""
    _pto.tcolmin(
        unwrap_surface_value(src),
        unwrap_surface_value(dst),
    )


def tcolprod(src, dst):
    """``pto.tcolprod ins(src) outs(dst)``."""
    _pto.tcolprod(
        unwrap_surface_value(src),
        unwrap_surface_value(dst),
    )


def tcolargmax(src, tmp, dst):
    """``pto.tcolargmax ins(src, tmp) outs(dst)``."""
    _pto.tcolargmax(
        unwrap_surface_value(src),
        unwrap_surface_value(tmp),
        unwrap_surface_value(dst),
    )


def tcolargmin(src, tmp, dst):
    """``pto.tcolargmin ins(src, tmp) outs(dst)``."""
    _pto.tcolargmin(
        unwrap_surface_value(src),
        unwrap_surface_value(tmp),
        unwrap_surface_value(dst),
    )


def tcmp(src0, src1, dst, *, cmp_mode=None):
    """``pto.tcmp ins(src0, src1) outs(dst)``."""
    _pto.tcmp(
        unwrap_surface_value(src0),
        unwrap_surface_value(src1),
        unwrap_surface_value(dst),
        cmp_mode=None if cmp_mode is None else _cmp_mode_attr(cmp_mode),
    )


def tcmps(src, scalar, dst, *, cmp_mode=None):
    """``pto.tcmps ins(src, scalar) outs(dst)``."""
    _pto.tcmps(
        unwrap_surface_value(src),
        _coerce_tile_scalar_operand(src, scalar, context="tcmps"),
        unwrap_surface_value(dst),
        cmp_mode=None if cmp_mode is None else _cmp_mode_attr(cmp_mode),
    )


def texpands(scalar, dst):
    """``pto.texpands ins(scalar) outs(dst)``."""
    _pto.texpands(
        _coerce_tile_scalar_operand(dst, scalar, context="texpands"),
        unwrap_surface_value(dst),
    )


def _tile_numel(shape, *, context: str):
    numel = 1
    for dim in shape:
        if isinstance(dim, bool) or not isinstance(dim, int):
            raise TypeError(f"{context} currently requires a static shape")
        numel *= dim
    return numel


def treshape(src, *, shape, dtype=None, blayout=None):
    """``pto.treshape ins(src) -> result``."""
    src_value = unwrap_surface_value(src)
    src_shape = getattr(src, "shape", None)
    src_dtype = getattr(src, "dtype", None)
    src_memory_space = getattr(src, "memory_space", None)
    src_metadata = parse_tile_type_metadata(src_value.type)
    if src_shape is None and src_metadata is not None:
        src_shape = tuple(src_metadata["shape_dims"])
    if src_dtype is None and src_metadata is not None:
        src_dtype = src_metadata["element_type"]
    if src_memory_space is None and src_metadata is not None:
        src_memory_space = src_metadata["memory_space"]
    if src_shape is None or src_dtype is None or src_memory_space is None:
        raise TypeError("treshape(...) expects a tile_buf-backed Tile value")

    result_shape = _normalize_static_tile_shape(shape)
    result_dtype = dtype if dtype is not None else src_dtype
    result_blayout = blayout if blayout is not None else "RowMajor"

    src_numel = _tile_numel(src_shape, context="treshape(src, shape=...) source")
    dst_numel = _tile_numel(result_shape, context="treshape(src, shape=...) result")
    src_bytes = src_numel * _element_bytewidth(_resolve(src_dtype))
    dst_bytes = dst_numel * _element_bytewidth(_resolve(result_dtype))
    if src_bytes != dst_bytes:
        raise ValueError(
            "treshape(src, shape=..., dtype=...) requires source and result to have the same total byte size"
        )

    result_memory_space = src_memory_space
    result_physical_shape = _authored_tile_physical_shape(result_shape)
    _validate_authored_tile_row_alignment(result_physical_shape, result_dtype, blayout=result_blayout, slayout="NoneBox")

    from ._types import tile_buf_type

    result_type = tile_buf_type(
        result_physical_shape,
        result_dtype,
        blayout=result_blayout,
        address_space=result_memory_space,
    )
    value = _pto.treshape(result_type, src_value)
    return wrap_surface_value(
        value,
        tile_metadata={
            "shape": result_shape,
            "physical_shape": result_physical_shape,
            "dtype": result_dtype,
            "memory_space": result_memory_space,
        },
    )


def trowexpand(src, dst):
    """``pto.trowexpand ins(src) outs(dst)``."""
    _pto.trowexpand(
        unwrap_surface_value(src),
        unwrap_surface_value(dst),
    )


def tcolexpand(src, dst):
    """``pto.tcolexpand ins(src) outs(dst)``."""
    _pto.tcolexpand(
        unwrap_surface_value(src),
        unwrap_surface_value(dst),
    )


def trowexpandadd(src0, src1, dst):
    """``pto.trowexpandadd ins(src0, src1) outs(dst)``."""
    _pto.trowexpandadd(
        unwrap_surface_value(src0),
        unwrap_surface_value(src1),
        unwrap_surface_value(dst),
    )


def trowexpandsub(src0, src1, dst, *, tmp=None):
    """``pto.trowexpandsub ins(src0, src1, tmp?) outs(dst)``."""
    _pto.trowexpandsub(
        unwrap_surface_value(src0),
        unwrap_surface_value(src1),
        unwrap_surface_value(dst),
        tmp=None if tmp is None else unwrap_surface_value(tmp),
    )


def trowexpandmul(src0, src1, dst, *, tmp=None):
    """``pto.trowexpandmul ins(src0, src1, tmp?) outs(dst)``."""
    _pto.trowexpandmul(
        unwrap_surface_value(src0),
        unwrap_surface_value(src1),
        unwrap_surface_value(dst),
        tmp=None if tmp is None else unwrap_surface_value(tmp),
    )


def trowexpanddiv(src0, src1, dst, *, tmp=None, div_precision=None):
    """``pto.trowexpanddiv ins(src0, src1, tmp?) outs(dst)``."""
    _pto.trowexpanddiv(
        unwrap_surface_value(src0),
        unwrap_surface_value(src1),
        unwrap_surface_value(dst),
        tmp=None if tmp is None else unwrap_surface_value(tmp),
        precision_type=div_precision,
    )


def trowexpandmax(src0, src1, dst, *, tmp=None):
    """``pto.trowexpandmax ins(src0, src1, tmp?) outs(dst)``."""
    _pto.trowexpandmax(
        unwrap_surface_value(src0),
        unwrap_surface_value(src1),
        unwrap_surface_value(dst),
        tmp=None if tmp is None else unwrap_surface_value(tmp),
    )


def trowexpandmin(src0, src1, dst, *, tmp=None):
    """``pto.trowexpandmin ins(src0, src1, tmp?) outs(dst)``."""
    _pto.trowexpandmin(
        unwrap_surface_value(src0),
        unwrap_surface_value(src1),
        unwrap_surface_value(dst),
        tmp=None if tmp is None else unwrap_surface_value(tmp),
    )


def trowexpandexpdif(src0, src1, dst, *, tmp=None):
    """``pto.trowexpandexpdif ins(src0, src1, tmp?) outs(dst)``."""
    _pto.trowexpandexpdif(
        unwrap_surface_value(src0),
        unwrap_surface_value(src1),
        unwrap_surface_value(dst),
        tmp=None if tmp is None else unwrap_surface_value(tmp),
    )


def tcolexpandadd(src0, src1, dst):
    """``pto.tcolexpandadd ins(src0, src1) outs(dst)``."""
    _pto.tcolexpandadd(
        unwrap_surface_value(src0),
        unwrap_surface_value(src1),
        unwrap_surface_value(dst),
    )


def tcolexpandsub(src0, src1, dst):
    """``pto.tcolexpandsub ins(src0, src1) outs(dst)``."""
    _pto.tcolexpandsub(
        unwrap_surface_value(src0),
        unwrap_surface_value(src1),
        unwrap_surface_value(dst),
    )


def tcolexpandmul(src0, src1, dst):
    """``pto.tcolexpandmul ins(src0, src1) outs(dst)``."""
    _pto.tcolexpandmul(
        unwrap_surface_value(src0),
        unwrap_surface_value(src1),
        unwrap_surface_value(dst),
    )


def tcolexpanddiv(src0, src1, dst, *, div_precision=None):
    """``pto.tcolexpanddiv ins(src0, src1) outs(dst)``."""
    _pto.tcolexpanddiv(
        unwrap_surface_value(src0),
        unwrap_surface_value(src1),
        unwrap_surface_value(dst),
        precision_type=div_precision,
    )


def tcolexpandmax(src0, src1, dst):
    """``pto.tcolexpandmax ins(src0, src1) outs(dst)``."""
    _pto.tcolexpandmax(
        unwrap_surface_value(src0),
        unwrap_surface_value(src1),
        unwrap_surface_value(dst),
    )


def tcolexpandmin(src0, src1, dst):
    """``pto.tcolexpandmin ins(src0, src1) outs(dst)``."""
    _pto.tcolexpandmin(
        unwrap_surface_value(src0),
        unwrap_surface_value(src1),
        unwrap_surface_value(dst),
    )


def tcolexpandexpdif(src0, src1, dst):
    """``pto.tcolexpandexpdif ins(src0, src1) outs(dst)``."""
    _pto.tcolexpandexpdif(
        unwrap_surface_value(src0),
        unwrap_surface_value(src1),
        unwrap_surface_value(dst),
    )


def _resolve_selection_tmp(dst, tmp, *, context: str):
    if tmp is not None:
        return tmp

    session = None
    try:
        from ._tracing.active import current_session
        session = current_session()
    except Exception:
        session = None

    if session is not None and getattr(session.module_spec, "target_arch", None) == "a5":
        return dst

    return alloc_tile(tile_type=unwrap_surface_value(dst).type)


def tsort32(src, idx, dst, *, tmp=None):
    """``pto.tsort32 ins(src, idx, tmp?) outs(dst)``."""
    _pto.tsort32(
        unwrap_surface_value(src),
        unwrap_surface_value(idx),
        unwrap_surface_value(dst),
        tmp=None if tmp is None else unwrap_surface_value(tmp),
    )


def _unwrap_optional_integer(value):
    if value is None:
        return None
    if isinstance(value, int):
        value = const(value, dtype=IntegerType.get_signless(32))
    return unwrap_surface_value(value)


def tmrgsort(src, dst, block_len=None, *, tmp=None, excuted=None, exhausted=None):
    """``pto.tmrgsort`` tile merge-sort wrapper.

    Format 1 uses ``tmrgsort(src, dst, block_len)``.  Format 2 can pass
    ``src`` and ``dst`` as sequences and provide ``tmp`` plus ``excuted``.
    """
    srcs = src if isinstance(src, (list, tuple)) else [src]
    dsts = dst if isinstance(dst, (list, tuple)) else [dst]
    _pto.tmrgsort(
        [unwrap_surface_value(value) for value in srcs],
        [unwrap_surface_value(value) for value in dsts],
        block_len=_unwrap_optional_integer(block_len),
        tmp=None if tmp is None else unwrap_surface_value(tmp),
        excuted=None if excuted is None else unwrap_surface_value(excuted),
        exhausted=exhausted,
    )


def tgather(
    src,
    dst,
    *,
    cdst=None,
    indices=None,
    tmp=None,
    k_value=None,
    mask_pattern=None,
    cmp_mode=None,
    offset=None,
):
    """``pto.tgather`` tile gather/select wrapper."""
    _pto.tgather(
        unwrap_surface_value(src),
        unwrap_surface_value(dst),
        cdst=None if cdst is None else unwrap_surface_value(cdst),
        indices=None if indices is None else unwrap_surface_value(indices),
        tmp=None if tmp is None else unwrap_surface_value(tmp),
        k_value=None if k_value is None else unwrap_surface_value(k_value),
        mask_pattern=None if mask_pattern is None else _tile_mask_pattern_attr(mask_pattern),
        cmp_mode=None if cmp_mode is None else _normalize_cmp_mode(cmp_mode),
        offset=offset,
    )


def tgatherb(src, offsets, dst):
    """``pto.tgatherb`` – tile gather using byte offsets (DPS)."""
    _pto.tgatherb(
        unwrap_surface_value(src),
        unwrap_surface_value(offsets),
        unwrap_surface_value(dst),
    )


def tsel(mask, src0, src1, dst, *, tmp=None):
    """``pto.tsel ins(mask, src0, src1, tmp) outs(dst)`` with synthesized scratch when omitted."""
    resolved_tmp = tmp if tmp is not None else _resolve_selection_tmp(dst, tmp, context="tsel")
    _pto.tsel(
        unwrap_surface_value(mask),
        unwrap_surface_value(src0),
        unwrap_surface_value(src1),
        unwrap_surface_value(resolved_tmp),
        unwrap_surface_value(dst),
    )


def tsels(mask, src, scalar, dst, *, tmp=None):
    """``pto.tsels ins(mask, src, tmp, scalar) outs(dst)`` with synthesized scratch when omitted."""
    resolved_tmp = tmp if tmp is not None else _resolve_selection_tmp(dst, tmp, context="tsels")
    _pto.tsels(
        unwrap_surface_value(mask),
        unwrap_surface_value(src),
        unwrap_surface_value(resolved_tmp),
        _coerce_tile_scalar_operand(src, scalar, context="tsels"),
        unwrap_surface_value(dst),
    )


def tcvt(src, dst, *, tmp=None, rmode=None, sat_mode=None):
    """``pto.tcvt ins(src) outs(dst)``.

    The ``tmp`` parameter is retained for backward compatibility but is not
    supported by the current PTO backend; passing a non-None value raises.
    """
    if tmp is not None:
        raise TypeError("pto.tile.cvt(..., tmp=...) is not supported by the current PTO Python bindings")
    _pto.tcvt(
        unwrap_surface_value(src),
        unwrap_surface_value(dst),
        rmode=_normalize_enum_attr(
            rmode,
            enum_cls=_pto.RoundMode,
            attr_cls=_pto.RoundModeAttr,
            context="tile.cvt(..., rmode=...)",
        ),
        sat_mode=_normalize_enum_attr(
            sat_mode,
            enum_cls=_pto.SaturationMode,
            attr_cls=_pto.SaturationModeAttr,
            context="tile.cvt(..., sat_mode=...)",
        ),
    )


def tnot(src, dst):
    """``pto.tnot ins(src) outs(dst)``."""
    _pto.tnot(
        unwrap_surface_value(src),
        unwrap_surface_value(dst),
    )


def tand(src0, src1, dst):
    """``pto.tand ins(src0, src1) outs(dst)``."""
    _pto.tand(
        unwrap_surface_value(src0),
        unwrap_surface_value(src1),
        unwrap_surface_value(dst),
    )


def tands(src, scalar, dst):
    """``pto.tands ins(src, scalar) outs(dst)``."""
    _pto.tands(
        unwrap_surface_value(src),
        _coerce_tile_scalar_operand(src, scalar, context="tands"),
        unwrap_surface_value(dst),
    )


def tor(src0, src1, dst):
    """``pto.tor ins(src0, src1) outs(dst)``."""
    _pto.tor(
        unwrap_surface_value(src0),
        unwrap_surface_value(src1),
        unwrap_surface_value(dst),
    )


def tors(src, scalar, dst):
    """``pto.tors ins(src, scalar) outs(dst)``."""
    _pto.tors(
        unwrap_surface_value(src),
        _coerce_tile_scalar_operand(src, scalar, context="tors"),
        unwrap_surface_value(dst),
    )


def txor(src0, src1, tmp, dst):
    """``pto.txor ins(src0, src1, tmp) outs(dst)``."""
    _pto.txor(
        unwrap_surface_value(src0),
        unwrap_surface_value(src1),
        unwrap_surface_value(tmp),
        unwrap_surface_value(dst),
    )


def txors(src, scalar, tmp, dst):
    """``pto.txors ins(src, scalar, tmp) outs(dst)``."""
    _pto.txors(
        unwrap_surface_value(src),
        _coerce_tile_scalar_operand(src, scalar, context="txors"),
        unwrap_surface_value(tmp),
        unwrap_surface_value(dst),
    )


def tshl(src0, src1, dst):
    """``pto.tshl ins(src0, src1) outs(dst)``."""
    _pto.tshl(
        unwrap_surface_value(src0),
        unwrap_surface_value(src1),
        unwrap_surface_value(dst),
    )


def tshls(src, scalar, dst):
    """``pto.tshls ins(src, scalar) outs(dst)``."""
    _pto.tshls(
        unwrap_surface_value(src),
        _coerce_tile_scalar_operand(src, scalar, context="tshls"),
        unwrap_surface_value(dst),
    )


def tshr(src0, src1, dst):
    """``pto.tshr ins(src0, src1) outs(dst)``."""
    _pto.tshr(
        unwrap_surface_value(src0),
        unwrap_surface_value(src1),
        unwrap_surface_value(dst),
    )


def tshrs(src, scalar, dst):
    """``pto.tshrs ins(src, scalar) outs(dst)``."""
    _pto.tshrs(
        unwrap_surface_value(src),
        _coerce_tile_scalar_operand(src, scalar, context="tshrs"),
        unwrap_surface_value(dst),
    )


def tpartadd(src0, src1, dst):
    """``pto.tpartadd ins(src0, src1) outs(dst)``."""
    _pto.tpartadd(
        unwrap_surface_value(src0),
        unwrap_surface_value(src1),
        unwrap_surface_value(dst),
    )


def tpartmul(src0, src1, dst):
    """``pto.tpartmul ins(src0, src1) outs(dst)``."""
    _pto.tpartmul(
        unwrap_surface_value(src0),
        unwrap_surface_value(src1),
        unwrap_surface_value(dst),
    )


def tpartmax(src0, src1, dst):
    """``pto.tpartmax ins(src0, src1) outs(dst)``."""
    _pto.tpartmax(
        unwrap_surface_value(src0),
        unwrap_surface_value(src1),
        unwrap_surface_value(dst),
    )


def tpartmin(src0, src1, dst):
    """``pto.tpartmin ins(src0, src1) outs(dst)``."""
    _pto.tpartmin(
        unwrap_surface_value(src0),
        unwrap_surface_value(src1),
        unwrap_surface_value(dst),
    )


def tfillpad(src, dst):
    """``pto.tfillpad ins(src) outs(dst)``."""
    _pto.tfillpad(
        unwrap_surface_value(src),
        unwrap_surface_value(dst),
    )


def tfillpad_expand(src, dst):
    """``pto.tfillpad_expand ins(src) outs(dst)``."""
    _pto.tfillpad_expand(
        unwrap_surface_value(src),
        unwrap_surface_value(dst),
    )


def tfillpad_inplace(src, dst):
    """``pto.tfillpad_inplace ins(src) outs(dst)``."""
    _pto.tfillpad_inplace(
        unwrap_surface_value(src),
        unwrap_surface_value(dst),
    )


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


def _infer_vreg_type_from_tile_slice(tile_slice: TileSliceValue):
    memref_type = MemRefType(tile_slice.type)
    elem_type = memref_type.element_type
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
    zero = materialize_scalar_literal(0.0 if kind == "float" else 0, raw_value.type, context="_negate_runtime_scalar(...)")
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


def _acc_store_ub_dst_mode_attr(mode):
    return _enum_attr(
        "acc_store_ub_dst_mode",
        mode,
        supported={"single", "split_m", "split_n"},
        context="mte_l0c_ub dst_mode",
    )


def _acc_store_unit_flag_attr(unit_flag):
    if unit_flag is None:
        return None
    return _enum_attr(
        "unit_flag_ctrl",
        unit_flag,
        supported={"check_only", "check_and_clear"},
        context="acc store unit_flag",
    )

_ACC_STORE_PRE_QUANT_MODES = frozenset({
    "no_convert",
    "f32_f16",
    "qf322hif8_pre_vec",
    "qf322hif8_pre_scalar",
    "qf322hif8_pre_hybrid_vec",
    "qf322hif8_pre_hybrid_scalar",
    "deqs32_int_vec",
    "deqs32_int_scalar",
    "req8_vec",
    "req8_scalar",
    "deqf16_vec",
    "deqf16_scalar",
    "qf322fp8_pre_vec",
    "qf322fp8_pre_scalar",
    "qf322f32_pre_vec",
    "qf322f32_pre_scalar",
    "f32_bf16",
    "qf162b8_pre_vec",
    "qf162b8_pre_scalar",
    "qf162s4_pre_vec",
    "qf162s4_pre_scalar",
    "req4_vec",
    "req4_scalar",
    "qf322b8_pre_vec",
    "qf322b8_pre_scalar",
    "qf322s4_pre_vec",
    "qf322s4_pre_scalar",
    "deqs16_vec",
    "deqs16_scalar",
    "qf162s16_pre_vec",
    "qf162s16_pre_scalar",
    "qf322f16_pre_vec",
    "qf322f16_pre_scalar",
    "qf322bf16_pre_vec",
    "qf322bf16_pre_scalar",
    "qs322bf16_pre_vec",
    "qs322bf16_pre_scalar",
})

_ACC_STORE_PRE_QUANT_SKIP_PAYLOAD_KIND_CHECK = frozenset({
    "no_convert",
})

_ACC_STORE_VECTOR_PRE_QUANT_MODES = frozenset({
    mode for mode in _ACC_STORE_PRE_QUANT_MODES if mode.endswith("_vec")
})


def _is_acc_store_float_scalar_payload(value) -> bool:
    raw_value = unwrap_surface_value(value)
    return hasattr(raw_value, "type") and any(
        cls.isinstance(raw_value.type) for cls in (F16Type, BF16Type, F32Type)
    )


def _is_acc_store_scaling_pointer_payload(value) -> bool:
    raw_value = unwrap_surface_value(value)
    if not hasattr(raw_value, "type"):
        return False
    try:
        ptr_type = _pto.PtrType(raw_value.type)
    except Exception:
        return False
    scaling_attr = _pto.AddressSpaceAttr.get(_pto.AddressSpace.SCALING)
    memory_space = getattr(ptr_type, "memory_space", None)
    if (
        memory_space != scaling_attr
        and getattr(memory_space, "value", None) != _pto.AddressSpace.SCALING
    ):
        return False
    return any(
        cls.isinstance(ptr_type.element_type)
        for cls in (F16Type, BF16Type, F32Type)
    )


def _acc_store_pre_quant(pre_quant):
    if pre_quant is None:
        return None, None
    if not isinstance(pre_quant, tuple) or len(pre_quant) != 2:
        raise TypeError("acc store pre_quant expects (payload, mode)")
    payload, mode = pre_quant
    normalized_mode = _normalize_token(mode, context="acc store pre_quant mode")
    if normalized_mode not in _ACC_STORE_PRE_QUANT_MODES:
        raise ValueError(f"unsupported acc store pre_quant mode: {normalized_mode}")
    if normalized_mode in _ACC_STORE_PRE_QUANT_SKIP_PAYLOAD_KIND_CHECK:
        pass
    elif normalized_mode in _ACC_STORE_VECTOR_PRE_QUANT_MODES:
        if not _is_acc_store_scaling_pointer_payload(payload):
            raise TypeError(
                "acc store vector pre_quant payload must be a scaling pointer "
                "with f16, bf16, or f32 elements"
            )
    else:
        if not _is_acc_store_float_scalar_payload(payload):
            raise TypeError(
                "acc store scalar pre_quant payload must be an f16, bf16, or f32 scalar"
            )
    return (
        unwrap_surface_value(payload),
        Attribute.parse(f"#pto<quant_pre_mode {normalized_mode}>"),
    )


def _acc_store_pre_relu(pre_relu):
    if pre_relu is None:
        return None, None, None
    if not isinstance(pre_relu, tuple) or len(pre_relu) != 3:
        raise TypeError("acc store pre_relu expects (mode, payload, clip)")
    mode, payload, clip = pre_relu
    return (
        None if payload is None else unwrap_surface_value(payload),
        Attribute.parse(f"#pto<relu_pre_mode {_normalize_token(mode, context='acc store pre_relu mode')}>"),
        None if clip is None else unwrap_surface_value(clip),
    )


def _acc_store_layout(layout):
    if layout is None:
        return None, None, None
    if isinstance(layout, tuple):
        if len(layout) != 2:
            raise TypeError("acc store layout tuple expects (mode, operand)")
        mode, operand = layout
        normalized = _normalize_token(mode, context="acc store layout")
        if normalized == "nz2dn":
            return Attribute.parse("#pto<acc_store_mode nz2dn>"), None, _coerce_i64(operand, context="acc store layout nz2dn")
        if normalized == "nz2nz":
            return Attribute.parse("#pto<acc_store_mode nz2nz>"), _coerce_i64(operand, context="acc store layout nz2nz"), None
        raise ValueError("acc store layout tuple only supports nz2dn or nz2nz")
    normalized = _normalize_token(layout, context="acc store layout")
    if normalized != "nz2nd":
        raise ValueError("acc store layout string only supports nz2nd; use (mode, operand) for nz2dn/nz2nz")
    return Attribute.parse("#pto<acc_store_mode nz2nd>"), None, None


def _acc_store_loop3(loop3):
    if loop3 is None:
        return None, None, None
    if not isinstance(loop3, tuple) or len(loop3) != 3:
        raise TypeError("acc store loop3 expects (count, src_stride, dst_stride)")
    count, src_stride, dst_stride = loop3
    return (
        _coerce_i64(count, context="acc store loop3 count"),
        _coerce_i64(src_stride, context="acc store loop3 src_stride"),
        _coerce_i64(dst_stride, context="acc store loop3 dst_stride"),
    )


def _acc_store_sat_attr(sat):
    if sat is None:
        return None
    return _enum_attr(
        "acc_store_sat_mode",
        _normalize_sat_mode(sat, context="acc store sat", allow_preserve_nan=True),
        supported={"sat", "nosat", "sat_preserve_nan"},
        context="acc store sat",
    )


def _acc_store_atomic_attrs(atomic):
    if atomic is None:
        return None, None
    if not isinstance(atomic, tuple) or len(atomic) != 2:
        raise TypeError("acc store atomic expects (type, op)")
    atomic_type, atomic_op = atomic
    return (
        _enum_attr(
            "acc_store_atomic_type",
            atomic_type,
            supported={"f32", "f16", "bf16", "s32", "s16", "s8"},
            context="acc store atomic type",
        ),
        _enum_attr(
            "acc_store_atomic_op",
            atomic_op,
            supported={"add", "max", "min"},
            context="acc store atomic op",
        ),
    )


def _acc_store_options(unit_flag=None, pre_quant=None, pre_relu=None, layout=None, loop3=None, sat=None, atomic=None):
    pre_quant_value, pre_quant_mode = _acc_store_pre_quant(pre_quant)
    pre_relu_value, pre_relu_mode, clip_value = _acc_store_pre_relu(pre_relu)
    mode, split, loop0_src_stride = _acc_store_layout(layout)
    loop3_count, loop3_src_stride, loop3_dst_stride = _acc_store_loop3(loop3)
    atomic_type, atomic_op = _acc_store_atomic_attrs(atomic)
    return {
        "pre_quant": pre_quant_value,
        "pre_relu": pre_relu_value,
        "clip_value": clip_value,
        "split": split,
        "loop0_src_stride": loop0_src_stride,
        "loop3_count": loop3_count,
        "loop3_src_stride": loop3_src_stride,
        "loop3_dst_stride": loop3_dst_stride,
        "mode": mode,
        "unit_flag": _acc_store_unit_flag_attr(unit_flag),
        "pre_quant_mode": pre_quant_mode,
        "pre_relu_mode": pre_relu_mode,
        "sat_mode": _acc_store_sat_attr(sat),
        "atomic_type": atomic_type,
        "atomic_op": atomic_op,
    }


def _normalize_ub_split(split):
    normalized = _normalize_token(split, context="mte_l0c_ub split")
    aliases = {
        "m": "split_m",
        "n": "split_n",
        "split_m": "split_m",
        "split_n": "split_n",
    }
    mode = aliases.get(normalized)
    if mode is None:
        raise ValueError("mte_l0c_ub split expects M or N")
    return mode


def _mte_l0c_ub_dst_mode(sub_blockid=0, *, split=None):
    if split is not None:
        token = getattr(sub_blockid, "value", sub_blockid)
        if token not in {0, None}:
            raise ValueError("mte_l0c_ub split cannot be combined with non-default sub_blockid")
        return _acc_store_ub_dst_mode_attr(_normalize_ub_split(split)), None
    token = getattr(sub_blockid, "value", sub_blockid)
    if isinstance(token, str):
        raise TypeError("mte_l0c_ub sub_blockid expects 0 or 1; use split='M' or split='N' for dual-destination stores")
    if isinstance(token, bool):
        raise TypeError("mte_l0c_ub sub_blockid bool is not supported; use sub-block 0/1 or split='M'/'N'")
    if isinstance(token, int) and token not in {0, 1}:
        raise ValueError("mte_l0c_ub sub_blockid constant must be 0 or 1")
    return _acc_store_ub_dst_mode_attr("single"), _coerce_i64(token, context="mte_l0c_ub sub_blockid")


def _cube_load_frac_mode_attr(mode):
    return _enum_attr(
        "cube_load_frac_mode",
        mode,
        supported={"nd2nz", "dn2nz"},
        context="mte_gm_l1_frac mode",
    )


def _normalize_pair(name, pair, *, context: str):
    if not isinstance(pair, tuple) or len(pair) != 2:
        raise TypeError(f"{context} expects {name}=(value0, value1)")
    first, second = pair
    return (
        _coerce_i64(first, context=f"{context} {name}[0]"),
        _coerce_i64(second, context=f"{context} {name}[1]"),
    )


def _normalize_frac_src_layout(src_layout, *, context: str):
    if not isinstance(src_layout, tuple) or len(src_layout) not in (1, 2):
        raise TypeError(f"{context} expects src_layout=(inner_stride,) or (inner_stride, outer_stride)")
    inner = _coerce_i64(src_layout[0], context=f"{context} src_layout[0]")
    outer = None
    if len(src_layout) == 2:
        outer = _coerce_i64(src_layout[1], context=f"{context} src_layout[1]")
    return inner, outer


def _normalize_frac_dst_group(dst_group, *, context: str):
    if not isinstance(dst_group, tuple) or len(dst_group) != 4:
        raise TypeError(f"{context} expects dst_group=(group_count, loop2_stride, loop3_stride, loop4_stride)")
    group_count, loop2_stride, loop3_stride, loop4_stride = dst_group
    return (
        _coerce_i64(group_count, context=f"{context} dst_group[0]"),
        _coerce_i64(loop2_stride, context=f"{context} dst_group[1]"),
        _coerce_i64(loop3_stride, context=f"{context} dst_group[2]"),
        _coerce_i64(loop4_stride, context=f"{context} dst_group[3]"),
    )


def _normalize_frac_ctrl(ctrl, *, context: str):
    if not isinstance(ctrl, tuple) or len(ctrl) != 2:
        raise TypeError(f"{context} expects ctrl=(l2_cache_ctrl, smallc0_en)")
    l2_cache_ctrl, smallc0_en = ctrl
    return (
        _coerce_i64(l2_cache_ctrl, context=f"{context} ctrl[0]"),
        _coerce_i1(smallc0_en, context=f"{context} ctrl[1]"),
    )


def _mad_unit_flag_attr(unit_flag):
    if unit_flag is None:
        return None
    return _enum_attr(
        "mad_unit_flag_mode",
        unit_flag,
        supported={"check_only", "check_and_set"},
        context="mad unit_flag",
    )


def _mad_sat_attr(sat):
    if sat is None:
        return None
    return _enum_attr(
        "mad_sat_mode",
        _normalize_sat_mode(sat, context="mad sat", allow_preserve_nan=False),
        supported={"sat", "nosat"},
        context="mad sat",
    )


def _tf32_mode_attr(tf32_mode):
    if tf32_mode is None:
        return None
    return _enum_attr(
        "tf32_mode",
        tf32_mode,
        supported={"round_even", "round_away"},
        context="mad tf32_mode",
    )


def _mad_options(unit_flag=None, disable_gemv=False, sat=None, tf32_mode=None, n_dir=False):
    if not isinstance(disable_gemv, bool):
        raise TypeError("mad disable_gemv expects bool")
    if not isinstance(n_dir, bool):
        raise TypeError("mad n_dir expects bool")
    return {
        "unit_flag_mode": _mad_unit_flag_attr(unit_flag),
        "disable_gemv": disable_gemv,
        "sat_mode": _mad_sat_attr(sat),
        "tf32_mode": _tf32_mode_attr(tf32_mode),
        "n_dir": n_dir,
    }


def _mad_mx_options(unit_flag=None, disable_gemv=False, sat=None, n_dir=False):
    return {
        key: value
        for key, value in _mad_options(
            unit_flag=unit_flag,
            disable_gemv=disable_gemv,
            sat=sat,
            tf32_mode=None,
            n_dir=n_dir,
        ).items()
        if key != "tf32_mode"
    }


def _infer_dma_partition_row_stride(partition: PartitionTensorViewValue):
    if partition.shape is None or partition.strides is None:
        raise TypeError("mte_load/mte_store require partition view shape/stride metadata")
    outer_dims = list(partition.shape[:-1])
    non_unit = [i for i, dim in enumerate(outer_dims) if dim != 1]
    if len(non_unit) > 1:
        raise TypeError(
            "mte_load/mte_store currently only support partitions with at most one non-unit "
            "dimension before the contiguous innermost dimension"
        )
    if not non_unit:
        return 1, 0
    dim_index = non_unit[0]
    return partition.shape[dim_index], partition.strides[dim_index]


def _infer_dma_tile_geometry(tile: TileValue):
    if tile.shape is None:
        raise TypeError("mte_load/mte_store require tile shape metadata")
    if len(tile.shape) == 1:
        valid_cols = tile.valid_shape[0]
        return 1, valid_cols, tile.shape[0]
    if len(tile.shape) == 2:
        return tile.valid_shape[0], tile.valid_shape[1], tile.shape[1]
    raise TypeError("mte_load/mte_store currently only support rank-1 or rank-2 tiles")


def _infer_dma_2d_copy_signature(partition, tile, *, direction: str):
    row_count, src_row_stride = _infer_dma_partition_row_stride(partition)
    tile_rows, valid_cols, physical_cols = _infer_dma_tile_geometry(tile)
    if direction == "gm_to_ub":
        return row_count, valid_cols, _mul_bytes(src_row_stride, infer_tile_element_type(tile)), physical_cols * _element_bytewidth(infer_tile_element_type(tile))
    return row_count, valid_cols, physical_cols * _element_bytewidth(infer_tile_element_type(tile)), _mul_bytes(src_row_stride, infer_tile_element_type(tile))


def fill_tile(tile, value):
    """Broadcast a scalar into an entire tile."""
    wrapped_tile = wrap_surface_value(tile)
    scalar_value = _constant_like(value, infer_tile_element_type(wrapped_tile))
    _pto.TExpandsOp(scalar_value, unwrap_surface_value(wrapped_tile))


def make_mask(dtype, value):
    """Create a predicate mask matching *dtype* granularity."""
    mask_bits = _mask_bits_for_dtype(dtype)
    result_type = _mask_type_from_bits(mask_bits)

    if isinstance(value, str):
        return wrap_surface_value(
            _pset_op_for_mask_bits(mask_bits)(result_type, _normalize_mask_pattern(value)).result
        )

    raw_value = unwrap_surface_value(value)
    authored_scalar_type = raw_value.type if hasattr(raw_value, "type") else IntegerType.get_signless(32)
    raw_value = _coerce_i32(raw_value, context="make_mask(..., value)")
    plt_op = _plt_op_for_mask_bits(mask_bits)(result_type, IntegerType.get_signless(32), raw_value)
    next_value = coerce_scalar_to_type(
        plt_op.scalar_out,
        authored_scalar_type,
        context="make_mask(..., value) result",
    )
    return MaskResultValue(plt_op.mask, next_value)


# ── Hardware / sync ───────────────────────────────────────────────────────────

def _require_pto_ptr_operand(value, *, context: str):
    raw_value = unwrap_surface_value(value)
    try:
        _pto.PtrType(raw_value.type)
    except Exception as exc:
        raise TypeError(f"{context} expects PTO ptr operands, got {raw_value.type}") from exc
    return raw_value


@_explicit_mode_only("pto.mte_load(...)")
def mte_load(source, destination, l2_cache_ctl, len_burst, *, nburst, loops=None, pad=None):
    """
    Ptr-based GM->UB DMA wrapper aligned with the underlying ``pto.dma_load`` surface.

    This wrapper intentionally accepts only explicit pointer operands. It does
    not infer burst shape or strides from TensorView / PartitionTensorView /
    Tile metadata.
    """
    n_burst, nburst_src_stride, nburst_dst_stride = _normalize_dma_group(
        "nburst",
        nburst,
        context="mte_load(...)",
    )
    loop_counts, loop_src_strides, loop_dst_strides = _normalize_dma_loops(
        loops,
        context="mte_load(...)",
    )
    pad_value, left_padding_count, right_padding_count = _normalize_dma_pad(
        pad,
        context="mte_load(...)",
    )
    _pto.MteGmUbOp(
        _require_pto_ptr_operand(source, context="mte_load(...)"),
        _require_pto_ptr_operand(destination, context="mte_load(...)"),
        _coerce_i64(l2_cache_ctl, context="mte_load l2_cache_ctl"),
        _coerce_i64(len_burst, context="mte_load len_burst"),
        n_burst,
        nburst_src_stride,
        nburst_dst_stride,
        loop_counts,
        loop_src_strides,
        loop_dst_strides,
        pad_value=pad_value,
        left_padding_count=left_padding_count,
        right_padding_count=right_padding_count,
    )


@_explicit_mode_only("pto.mte_store(...)")
def mte_store(
    source,
    destination,
    len_burst,
    *,
    nburst,
    loops=None,
    l2_cache="nmfv",
):
    """Ptr-based UB->GM DMA wrapper aligned with the underlying ``pto.dma_store`` surface."""
    n_burst, nburst_src_stride, nburst_dst_stride = _normalize_dma_group(
        "nburst",
        nburst,
        context="mte_store(...)",
    )
    loop_counts, loop_src_strides, loop_dst_strides = _normalize_dma_loops(
        loops,
        context="mte_store(...)",
    )
    _pto.MteUbGmOp(
        _require_pto_ptr_operand(source, context="mte_store(...)"),
        _require_pto_ptr_operand(destination, context="mte_store(...)"),
        _coerce_i64(len_burst, context="mte_store len_burst"),
        n_burst,
        nburst_src_stride,
        nburst_dst_stride,
        loop_counts,
        loop_src_strides,
        loop_dst_strides,
        l2_cache_ctl=_coerce_i64(
            _normalize_mte_store_l2_cache(l2_cache, context="mte_store(...) l2_cache"),
            context="mte_store l2 cache control",
        ),
    )


def _normalize_dma_group(name, triple, *, context: str):
    if not isinstance(triple, tuple) or len(triple) != 3:
        raise TypeError(f"{context} expects {name}=(count, src_stride, dst_stride)")
    count, src_stride, dst_stride = triple
    return (
        _coerce_i64(count, context=f"{context} {name}[0]"),
        _coerce_i64(src_stride, context=f"{context} {name}[1]"),
        _coerce_i64(dst_stride, context=f"{context} {name}[2]"),
    )


def _normalize_dma_loops(loops, *, context: str):
    if loops is None:
        return [], [], []
    if not isinstance(loops, (list, tuple)):
        raise TypeError(f"{context} expects loops to be a list[tuple[int, int, int]] or None")
    counts = []
    src_strides = []
    dst_strides = []
    for i, loop in enumerate(loops):
        count, src_stride, dst_stride = _normalize_dma_group(
            f"loops[{i}]",
            loop,
            context=context,
        )
        counts.append(count)
        src_strides.append(src_stride)
        dst_strides.append(dst_stride)
    return counts, src_strides, dst_strides


def _normalize_dma_pad(pad, *, context: str):
    if pad is None:
        return None, None, None
    if not isinstance(pad, tuple):
        raise TypeError(f"{context} expects pad to be tuple[ScalarType] or tuple[ScalarType, int, int]")
    if len(pad) == 1:
        pad_value = pad[0]
        left_count = 0
        right_count = 0
    elif len(pad) == 3:
        pad_value, left_count, right_count = pad
    else:
        raise TypeError(f"{context} expects pad to have length 1 or 3")
    return (
        materialize_scalar_literal(pad_value, F32Type.get(), context=f"{context} pad[0]")
        if not hasattr(pad_value, "type") else unwrap_surface_value(pad_value),
        _coerce_i64(left_count, context=f"{context} pad[1]"),
        _coerce_i64(right_count, context=f"{context} pad[2]"),
    )


@_explicit_mode_only("pto.mte_gm_ub(...)")
def mte_gm_ub(source, destination, l2_cache_ctl, len_burst, *, nburst, loops=None, pad=None):
    """``pto.mte_gm_ub`` – grouped GM-to-UB DMA surface."""
    n_burst, nburst_src_stride, nburst_dst_stride = _normalize_dma_group(
        "nburst",
        nburst,
        context="mte_gm_ub(...)",
    )
    loop_counts, loop_src_strides, loop_dst_strides = _normalize_dma_loops(
        loops,
        context="mte_gm_ub(...)",
    )
    pad_value, left_padding_count, right_padding_count = _normalize_dma_pad(
        pad,
        context="mte_gm_ub(...)",
    )
    _pto.MteGmUbOp(
        unwrap_surface_value(source),
        unwrap_surface_value(destination),
        _coerce_i64(l2_cache_ctl, context="mte_gm_ub l2_cache_ctl"),
        _coerce_i64(len_burst, context="mte_gm_ub len_burst"),
        n_burst,
        nburst_src_stride,
        nburst_dst_stride,
        loop_counts,
        loop_src_strides,
        loop_dst_strides,
        pad_value=pad_value,
        left_padding_count=left_padding_count,
        right_padding_count=right_padding_count,
    )


@_explicit_mode_only("pto.mte_ub_gm(...)")
def mte_ub_gm(
    source,
    destination,
    len_burst,
    *,
    nburst,
    loops=None,
    l2_cache="nmfv",
):
    """``pto.mte_ub_gm`` – grouped UB-to-GM DMA surface."""
    n_burst, nburst_src_stride, nburst_dst_stride = _normalize_dma_group(
        "nburst",
        nburst,
        context="mte_ub_gm(...)",
    )
    loop_counts, loop_src_strides, loop_dst_strides = _normalize_dma_loops(
        loops,
        context="mte_ub_gm(...)",
    )
    _pto.MteUbGmOp(
        unwrap_surface_value(source),
        unwrap_surface_value(destination),
        _coerce_i64(len_burst, context="mte_ub_gm len_burst"),
        n_burst,
        nburst_src_stride,
        nburst_dst_stride,
        loop_counts,
        loop_src_strides,
        loop_dst_strides,
        l2_cache_ctl=_coerce_i64(
            _normalize_mte_store_l2_cache(l2_cache, context="mte_ub_gm(...) l2_cache"),
            context="mte_ub_gm l2 cache control",
        ),
    )


@_explicit_mode_only("pto.mte_ub_ub(...)")
def mte_ub_ub(source, destination, len_burst, *, nburst):
    """``pto.mte_ub_ub`` – grouped UB-to-UB DMA surface."""
    n_burst, src_stride, dst_stride = _normalize_dma_group(
        "nburst",
        nburst,
        context="mte_ub_ub(...)",
    )
    _pto.MteUbUbOp(
        unwrap_surface_value(source),
        unwrap_surface_value(destination),
        n_burst,
        _coerce_i64(len_burst, context="mte_ub_ub len_burst"),
        src_stride,
        dst_stride,
    )


@_explicit_mode_only("pto.mte_ub_l1(...)")
def mte_ub_l1(source, destination, len_burst, *, nburst):
    """``pto.mte_ub_l1`` – grouped UB-to-L1 DMA surface."""
    n_burst, src_stride, dst_stride = _normalize_dma_group(
        "nburst",
        nburst,
        context="mte_ub_l1(...)",
    )
    _pto.MteUbL1Op(
        unwrap_surface_value(source),
        unwrap_surface_value(destination),
        n_burst,
        _coerce_i64(len_burst, context="mte_ub_l1 len_burst"),
        src_stride,
        dst_stride,
    )


@_explicit_mode_only("pto.mte_gm_l1(...)")
def mte_gm_l1(source, destination, len_burst, *, nburst, loops=None):
    """``pto.mte_gm_l1`` – grouped GM-to-L1/CBUF DMA surface."""
    n_burst, nburst_src_stride, nburst_dst_stride = _normalize_dma_group(
        "nburst",
        nburst,
        context="mte_gm_l1(...)",
    )
    loop_counts, loop_src_strides, loop_dst_strides = _normalize_dma_loops(
        loops,
        context="mte_gm_l1(...)",
    )
    _pto.MteGmL1Op(
        unwrap_surface_value(source),
        unwrap_surface_value(destination),
        _coerce_i64(len_burst, context="mte_gm_l1 len_burst"),
        n_burst,
        nburst_src_stride,
        nburst_dst_stride,
        loop_counts,
        loop_src_strides,
        loop_dst_strides,
    )


@_explicit_mode_only("pto.mte_l1_ub(...)")
def mte_l1_ub(source, destination, len_burst, *, nburst, loops=None):
    """``pto.mte_l1_ub`` – grouped L1/CBUF-to-UB DMA surface."""
    n_burst, nburst_src_stride, nburst_dst_stride = _normalize_dma_group(
        "nburst",
        nburst,
        context="mte_l1_ub(...)",
    )
    loop_counts, loop_src_strides, loop_dst_strides = _normalize_dma_loops(
        loops,
        context="mte_l1_ub(...)",
    )
    _pto.MteL1UbOp(
        unwrap_surface_value(source),
        unwrap_surface_value(destination),
        _coerce_i64(len_burst, context="mte_l1_ub len_burst"),
        n_burst,
        nburst_src_stride,
        nburst_dst_stride,
        loop_counts,
        loop_src_strides,
        loop_dst_strides,
    )


@_explicit_mode_only("pto.mte_gm_l1_frac(...)")
def mte_gm_l1_frac(source, destination, mode, *, shape, src_layout, dst_group, ctrl):
    """``pto.mte_gm_l1_frac`` – GM-to-L1 load with fractal layout conversion."""
    n_value, d_value = _normalize_pair("shape", shape, context="mte_gm_l1_frac(...)")
    src_inner_stride, src_outer_stride = _normalize_frac_src_layout(
        src_layout,
        context="mte_gm_l1_frac(...)",
    )
    group_count, dst_loop2_stride, dst_loop3_stride, dst_loop4_stride = _normalize_frac_dst_group(
        dst_group,
        context="mte_gm_l1_frac(...)",
    )
    l2_cache_ctrl, smallc0_en = _normalize_frac_ctrl(ctrl, context="mte_gm_l1_frac(...)")
    _pto.MteGmL1FracOp(
        unwrap_surface_value(source),
        unwrap_surface_value(destination),
        n_value,
        d_value,
        src_inner_stride,
        group_count,
        dst_loop2_stride,
        dst_loop3_stride,
        dst_loop4_stride,
        l2_cache_ctrl,
        smallc0_en,
        _cube_load_frac_mode_attr(mode),
        src_outer_stride=src_outer_stride,
    )


@_explicit_mode_only("pto.mte_l1_bt(...)")
def mte_l1_bt(source, destination, len_burst, *, nburst):
    """``pto.mte_l1_bt`` – grouped L1-to-BT auxiliary staging."""
    n_burst, nburst_src_gap, nburst_dst_gap = _normalize_dma_group(
        "nburst",
        nburst,
        context="mte_l1_bt(...)",
    )
    _pto.MteL1BtOp(
        unwrap_surface_value(source),
        unwrap_surface_value(destination),
        _coerce_i64(len_burst, context="mte_l1_bt len_burst"),
        n_burst,
        nburst_src_gap,
        nburst_dst_gap,
    )


@_explicit_mode_only("pto.mte_l1_fb(...)")
def mte_l1_fb(source, destination, len_burst, *, nburst):
    """``pto.mte_l1_fb`` – grouped L1-to-FB auxiliary staging."""
    n_burst, nburst_src_gap, nburst_dst_gap = _normalize_dma_group(
        "nburst",
        nburst,
        context="mte_l1_fb(...)",
    )
    _pto.MteL1FbOp(
        unwrap_surface_value(source),
        unwrap_surface_value(destination),
        _coerce_i64(len_burst, context="mte_l1_fb len_burst"),
        n_burst,
        nburst_src_gap,
        nburst_dst_gap,
    )


def mem_bar(barrier_type):
    """``pto.mem_bar`` with a small authored enum surface."""
    barrier_name = getattr(barrier_type, "value", barrier_type)
    _pto.MemBarOp(kind=_membar_attr(barrier_name))


@_explicit_mode_only("pto.mte_l1_l0a(...)")
def mte_l1_l0a(
    source,
    destination,
    m,
    k,
    *,
    start_row=0,
    start_col=0,
    transpose=False,
):
    """``pto.mte_l1_l0a`` – cube-side LEFT staging."""
    _pto.MteL1L0aOp(
        unwrap_surface_value(source),
        unwrap_surface_value(destination),
        _coerce_i64(m, context="mte_l1_l0a m"),
        _coerce_i64(k, context="mte_l1_l0a k"),
        _coerce_i64(start_row, context="mte_l1_l0a start_row"),
        _coerce_i64(start_col, context="mte_l1_l0a start_col"),
        transpose=transpose,
    )


@_explicit_mode_only("pto.mte_l1_l0b(...)")
def mte_l1_l0b(
    source,
    destination,
    k,
    n,
    *,
    start_row=0,
    start_col=0,
    transpose=False,
):
    """``pto.mte_l1_l0b`` – cube-side RIGHT staging."""
    _pto.MteL1L0bOp(
        unwrap_surface_value(source),
        unwrap_surface_value(destination),
        _coerce_i64(k, context="mte_l1_l0b k"),
        _coerce_i64(n, context="mte_l1_l0b n"),
        _coerce_i64(start_row, context="mte_l1_l0b start_row"),
        _coerce_i64(start_col, context="mte_l1_l0b start_col"),
        transpose=transpose,
    )


@_explicit_mode_only("pto.mte_l1_l0a_mx(...)")
def mte_l1_l0a_mx(
    source,
    destination,
    m,
    k,
    *,
    start_row=0,
    start_col=0,
):
    """``pto.mte_l1_l0a_mx`` – MX cube-side LEFT staging."""
    _pto.MteL1L0aMxOp(
        unwrap_surface_value(source),
        unwrap_surface_value(destination),
        _coerce_i64(m, context="mte_l1_l0a_mx m"),
        _coerce_i64(k, context="mte_l1_l0a_mx k"),
        _coerce_i64(start_row, context="mte_l1_l0a_mx start_row"),
        _coerce_i64(start_col, context="mte_l1_l0a_mx start_col"),
    )


@_explicit_mode_only("pto.mte_l1_l0b_mx(...)")
def mte_l1_l0b_mx(
    source,
    destination,
    k,
    n,
    *,
    start_row=0,
    start_col=0,
):
    """``pto.mte_l1_l0b_mx`` – MX cube-side RIGHT staging."""
    _pto.MteL1L0bMxOp(
        unwrap_surface_value(source),
        unwrap_surface_value(destination),
        _coerce_i64(k, context="mte_l1_l0b_mx k"),
        _coerce_i64(n, context="mte_l1_l0b_mx n"),
        _coerce_i64(start_row, context="mte_l1_l0b_mx start_row"),
        _coerce_i64(start_col, context="mte_l1_l0b_mx start_col"),
    )


@_explicit_mode_only("pto.mte_l0c_l1(...)")
def mte_l0c_l1(
    source,
    destination,
    m,
    n,
    src_stride,
    dst_stride,
    *,
    unit_flag=None,
    pre_quant=None,
    pre_relu=None,
    layout=None,
    loop3=None,
    sat=None,
):
    """``pto.mte_l0c_l1`` – ACC to L1 structured writeback."""
    options = _acc_store_options(
        unit_flag=unit_flag,
        pre_quant=pre_quant,
        pre_relu=pre_relu,
        layout=layout,
        loop3=loop3,
        sat=sat,
    )
    options.pop("atomic_type")
    options.pop("atomic_op")
    _pto.MteL0cL1Op(
        unwrap_surface_value(source),
        unwrap_surface_value(destination),
        _coerce_i64(m, context="mte_l0c_l1 m"),
        _coerce_i64(n, context="mte_l0c_l1 n"),
        _coerce_i64(src_stride, context="mte_l0c_l1 src_stride"),
        _coerce_i64(dst_stride, context="mte_l0c_l1 dst_stride"),
        **options,
    )


@_explicit_mode_only("pto.mte_l0c_gm(...)")
def mte_l0c_gm(
    source,
    destination,
    m,
    n,
    src_stride,
    dst_stride,
    sid,
    l2_cache_ctrl,
    *,
    unit_flag=None,
    pre_quant=None,
    pre_relu=None,
    layout=None,
    loop3=None,
    sat=None,
    atomic=None,
):
    """``pto.mte_l0c_gm`` – ACC to GM structured writeback."""
    _pto.MteL0cGmOp(
        unwrap_surface_value(source),
        unwrap_surface_value(destination),
        _coerce_i64(m, context="mte_l0c_gm m"),
        _coerce_i64(n, context="mte_l0c_gm n"),
        _coerce_i64(src_stride, context="mte_l0c_gm src_stride"),
        _coerce_i64(dst_stride, context="mte_l0c_gm dst_stride"),
        _coerce_i64(sid, context="mte_l0c_gm sid"),
        _coerce_i64(l2_cache_ctrl, context="mte_l0c_gm l2_cache_ctrl"),
        **_acc_store_options(
            unit_flag=unit_flag,
            pre_quant=pre_quant,
            pre_relu=pre_relu,
            layout=layout,
            loop3=loop3,
            sat=sat,
            atomic=atomic,
        ),
    )


@_explicit_mode_only("pto.mte_l0c_ub(...)")
def mte_l0c_ub(
    source,
    destination,
    m,
    n,
    src_stride,
    dst_stride,
    sub_blockid=0,
    *,
    split=None,
    unit_flag=None,
    pre_quant=None,
    pre_relu=None,
    layout=None,
    loop3=None,
    sat=None,
):
    """``pto.mte_l0c_ub`` – ACC to UB store."""
    dst_mode_attr, sub_blockid_value = _mte_l0c_ub_dst_mode(sub_blockid, split=split)
    options = _acc_store_options(
        unit_flag=unit_flag,
        pre_quant=pre_quant,
        pre_relu=pre_relu,
        layout=layout,
        loop3=loop3,
        sat=sat,
    )
    options.pop("atomic_type")
    options.pop("atomic_op")
    _pto.MteL0cUbOp(
        unwrap_surface_value(source),
        unwrap_surface_value(destination),
        _coerce_i64(m, context="mte_l0c_ub m"),
        _coerce_i64(n, context="mte_l0c_ub n"),
        _coerce_i64(src_stride, context="mte_l0c_ub src_stride"),
        _coerce_i64(dst_stride, context="mte_l0c_ub dst_stride"),
        dst_mode_attr,
        sub_blockid=sub_blockid_value,
        **options,
    )


def mad(lhs, rhs, dst, m, n, k, *, unit_flag=None, disable_gemv=False, sat=None, tf32_mode=None, n_dir=False):
    """``pto.mad`` – cube matmul accumulate."""
    _pto.MadOp(
        unwrap_surface_value(lhs),
        unwrap_surface_value(rhs),
        unwrap_surface_value(dst),
        _coerce_i64(m, context="mad m"),
        _coerce_i64(n, context="mad n"),
        _coerce_i64(k, context="mad k"),
        **_mad_options(
            unit_flag=unit_flag,
            disable_gemv=disable_gemv,
            sat=sat,
            tf32_mode=tf32_mode,
            n_dir=n_dir,
        ),
    )


def mad_acc(lhs, rhs, dst, m, n, k, *, unit_flag=None, disable_gemv=False, sat=None, tf32_mode=None, n_dir=False):
    """``pto.mad_acc`` – cube matmul accumulate into an existing accumulator."""
    _pto.MadAccOp(
        unwrap_surface_value(lhs),
        unwrap_surface_value(rhs),
        unwrap_surface_value(dst),
        _coerce_i64(m, context="mad_acc m"),
        _coerce_i64(n, context="mad_acc n"),
        _coerce_i64(k, context="mad_acc k"),
        **_mad_options(
            unit_flag=unit_flag,
            disable_gemv=disable_gemv,
            sat=sat,
            tf32_mode=tf32_mode,
            n_dir=n_dir,
        ),
    )


def mad_bias(lhs, rhs, dst, bias, m, n, k, *, unit_flag=None, disable_gemv=False, sat=None, tf32_mode=None, n_dir=False):
    """``pto.mad_bias`` – cube matmul initialized from a bias buffer."""
    _pto.MadBiasOp(
        unwrap_surface_value(lhs),
        unwrap_surface_value(rhs),
        unwrap_surface_value(dst),
        unwrap_surface_value(bias),
        _coerce_i64(m, context="mad_bias m"),
        _coerce_i64(n, context="mad_bias n"),
        _coerce_i64(k, context="mad_bias k"),
        **_mad_options(
            unit_flag=unit_flag,
            disable_gemv=disable_gemv,
            sat=sat,
            tf32_mode=tf32_mode,
            n_dir=n_dir,
        ),
    )


def mad_mx(lhs, rhs, dst, m, n, k, *, unit_flag=None, disable_gemv=False, sat=None, n_dir=False):
    """``pto.mad_mx`` – MX-format cube matmul."""
    _pto.MadMxOp(
        unwrap_surface_value(lhs),
        unwrap_surface_value(rhs),
        unwrap_surface_value(dst),
        _coerce_i64(m, context="mad_mx m"),
        _coerce_i64(n, context="mad_mx n"),
        _coerce_i64(k, context="mad_mx k"),
        **_mad_mx_options(
            unit_flag=unit_flag,
            disable_gemv=disable_gemv,
            sat=sat,
            n_dir=n_dir,
        ),
    )


def mad_mx_acc(lhs, rhs, dst, m, n, k, *, unit_flag=None, disable_gemv=False, sat=None, n_dir=False):
    """``pto.mad_mx_acc`` – MX-format cube matmul accumulate."""
    _pto.MadMxAccOp(
        unwrap_surface_value(lhs),
        unwrap_surface_value(rhs),
        unwrap_surface_value(dst),
        _coerce_i64(m, context="mad_mx_acc m"),
        _coerce_i64(n, context="mad_mx_acc n"),
        _coerce_i64(k, context="mad_mx_acc k"),
        **_mad_mx_options(
            unit_flag=unit_flag,
            disable_gemv=disable_gemv,
            sat=sat,
            n_dir=n_dir,
        ),
    )


def mad_mx_bias(lhs, rhs, dst, bias, m, n, k, *, unit_flag=None, disable_gemv=False, sat=None, n_dir=False):
    """``pto.mad_mx_bias`` – MX-format cube matmul initialized from a bias buffer."""
    _pto.MadMxBiasOp(
        unwrap_surface_value(lhs),
        unwrap_surface_value(rhs),
        unwrap_surface_value(dst),
        unwrap_surface_value(bias),
        _coerce_i64(m, context="mad_mx_bias m"),
        _coerce_i64(n, context="mad_mx_bias n"),
        _coerce_i64(k, context="mad_mx_bias k"),
        **_mad_mx_options(
            unit_flag=unit_flag,
            disable_gemv=disable_gemv,
            sat=sat,
            n_dir=n_dir,
        ),
    )

def get_block_idx():
    """``pto.get_block_idx`` → i64 block index."""
    return wrap_surface_value(_pto.GetBlockIdxOp().result)


def get_block_num():
    """``pto.get_block_num`` → i64 block count."""
    return wrap_surface_value(_pto.GetBlockNumOp().result)


def get_subblock_idx():
    """``pto.get_subblock_idx`` → i64 subblock index."""
    return wrap_surface_value(_pto.GetSubBlockIdxOp().result)


def get_subblock_num():
    """``pto.get_subblock_num`` → i64 subblock count."""
    return wrap_surface_value(_pto.GetSubBlockNumOp().result)


def store_vfsimt_info(dim_z, dim_y, dim_x):
    """``pto.store_vfsimt_info`` – configure the SIMT VF launch descriptor."""
    _pto.StoreVfSimtInfoOp(
        unwrap_surface_value(dim_z),
        unwrap_surface_value(dim_y),
        unwrap_surface_value(dim_x),
    )


def simt_launch(body, *args, dims=(1, 1, 1), **kwargs):
    """``pto.simt_launch`` – launch a ``@pto.simt`` helper with ``(x, y, z)`` dimensions."""
    spec = getattr(body, "spec", None)
    role = getattr(spec, "role", None)
    role_value = getattr(role, "value", role)
    if role_value != "simt":
        raise TypeError("pto.simt_launch(body, ...) expects body to be a @pto.simt-decorated function")

    body._validate_simt_launch_invocation(*args, **kwargs)

    from ._tracing.active import require_active_session
    session = require_active_session("pto.simt_launch")
    session.lower_simt_launch_subkernel(body, *args, dims=dims, **kwargs)


def get_tid_x():
    """``pto.get_tid_x`` → i32 SIMT lane X coordinate."""
    return wrap_surface_value(_pto.GetTidXOp().result)


def get_tid_y():
    """``pto.get_tid_y`` → i32 SIMT lane Y coordinate."""
    return wrap_surface_value(_pto.GetTidYOp().result)


def get_tid_z():
    """``pto.get_tid_z`` → i32 SIMT lane Z coordinate."""
    return wrap_surface_value(_pto.GetTidZOp().result)


def get_tid():
    """``pto.get_tid`` → ``(x, y, z)`` SIMT lane coordinates."""
    return get_tid_x(), get_tid_y(), get_tid_z()


def get_block_dim_x():
    """``pto.get_block_dim_x`` → i32 SIMT block X dimension."""
    return wrap_surface_value(_pto.GetBlockDimXOp().result)


def get_block_dim_y():
    """``pto.get_block_dim_y`` → i32 SIMT block Y dimension."""
    return wrap_surface_value(_pto.GetBlockDimYOp().result)


def get_block_dim_z():
    """``pto.get_block_dim_z`` → i32 SIMT block Z dimension."""
    return wrap_surface_value(_pto.GetBlockDimZOp().result)


def get_block_dim():
    """``pto.get_block_dim`` → ``(x, y, z)`` SIMT block dimensions."""
    return get_block_dim_x(), get_block_dim_y(), get_block_dim_z()


def get_grid_dim_x():
    """``pto.get_grid_dim_x`` → i32 SIMT grid X dimension."""
    return wrap_surface_value(_pto.GetGridDimXOp().result)


def get_grid_dim_y():
    """``pto.get_grid_dim_y`` → i32 SIMT grid Y dimension."""
    return wrap_surface_value(_pto.GetGridDimYOp().result)


def get_grid_dim_z():
    """``pto.get_grid_dim_z`` → i32 SIMT grid Z dimension."""
    return wrap_surface_value(_pto.GetGridDimZOp().result)


def get_grid_dim():
    """``pto.get_grid_dim`` → ``(x, y, z)`` SIMT grid dimensions."""
    return get_grid_dim_x(), get_grid_dim_y(), get_grid_dim_z()


def get_block_idx_x():
    """``pto.get_block_idx_x`` → i32 SIMT block X index."""
    return wrap_surface_value(_pto.GetBlockIdxXOp().result)


def get_block_idx_y():
    """``pto.get_block_idx_y`` → i32 SIMT block Y index."""
    return wrap_surface_value(_pto.GetBlockIdxYOp().result)


def get_block_idx_z():
    """``pto.get_block_idx_z`` → i32 SIMT block Z index."""
    return wrap_surface_value(_pto.GetBlockIdxZOp().result)


def get_veccoreid():
    """``pto.get_veccoreid`` → i32 SIMT vector-core id."""
    return wrap_surface_value(_pto.GetVecCoreIdOp().result)


def get_clock32():
    """``pto.get_clock32`` → i32 SIMT clock sample."""
    return wrap_surface_value(_pto.GetClock32Op().result)


def get_clock64():
    """``pto.get_clock64`` → i64 SIMT clock sample."""
    return wrap_surface_value(_pto.GetClock64Op().result)


def get_laneid():
    """``pto.get_laneid`` → i32 SIMT lane id."""
    return wrap_surface_value(_pto.GetLaneIdOp().result)


def get_lanemask_eq():
    """``pto.get_lanemask_eq`` → i32 SIMT lane equality mask."""
    return wrap_surface_value(_pto.GetLaneMaskEqOp().result)


def get_lanemask_le():
    """``pto.get_lanemask_le`` → i32 SIMT lane less-or-equal mask."""
    return wrap_surface_value(_pto.GetLaneMaskLeOp().result)


def get_lanemask_lt():
    """``pto.get_lanemask_lt`` → i32 SIMT lane less-than mask."""
    return wrap_surface_value(_pto.GetLaneMaskLtOp().result)


def get_lanemask_ge():
    """``pto.get_lanemask_ge`` → i32 SIMT lane greater-or-equal mask."""
    return wrap_surface_value(_pto.GetLaneMaskGeOp().result)


def get_lanemask_gt():
    """``pto.get_lanemask_gt`` → i32 SIMT lane greater-than mask."""
    return wrap_surface_value(_pto.GetLaneMaskGtOp().result)


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
            raise TypeError(f"{context} requires signedness='signed' or 'unsigned' when converting to or from integer types")
        return
    if signedness is not None:
        raise TypeError(f"{context} does not accept signedness for floating-point or packed conversion")


def vote_all(pred):
    """``pto.vote_all`` – SIMT all-lane predicate vote."""
    return wrap_surface_value(_pto.VoteAllOp(unwrap_surface_value(pred)).result)


def vote_any(pred):
    """``pto.vote_any`` – SIMT any-lane predicate vote."""
    return wrap_surface_value(_pto.VoteAnyOp(unwrap_surface_value(pred)).result)


def vote_uni(pred):
    """``pto.vote_uni`` – SIMT uniform-predicate vote."""
    return wrap_surface_value(_pto.VoteUniOp(unwrap_surface_value(pred)).result)


def vote_ballot(pred):
    """``pto.vote_ballot`` – SIMT ballot predicate vote."""
    return wrap_surface_value(_pto.VoteBallotOp(unwrap_surface_value(pred)).result)


def _validate_shuffle_width(width, *, context: str):
    if width not in (16, 32):
        raise ValueError(f"{context} expects width to be 16 or 32, got {width}")
    return width


def shuffle_idx(value, index, *, width=32):
    """``pto.shuffle_idx`` – read a payload from an absolute SIMT lane index."""
    return wrap_surface_value(_pto.ShuffleIdxOp(
        unwrap_surface_value(value),
        _coerce_i32_operand(index, context="shuffle_idx(..., index)"),
        width=_validate_shuffle_width(width, context="shuffle_idx(..., width)"),
    ).result)


def shuffle_up(value, offset, *, width=32):
    """``pto.shuffle_up`` – read a payload from a lower-index SIMT lane."""
    return wrap_surface_value(_pto.ShuffleUpOp(
        unwrap_surface_value(value),
        _coerce_i32_operand(offset, context="shuffle_up(..., offset)"),
        width=_validate_shuffle_width(width, context="shuffle_up(..., width)"),
    ).result)


def shuffle_down(value, offset, *, width=32):
    """``pto.shuffle_down`` – read a payload from a higher-index SIMT lane."""
    return wrap_surface_value(_pto.ShuffleDownOp(
        unwrap_surface_value(value),
        _coerce_i32_operand(offset, context="shuffle_down(..., offset)"),
        width=_validate_shuffle_width(width, context="shuffle_down(..., width)"),
    ).result)


def shuffle_bfly(value, mask, *, width=32):
    """``pto.shuffle_bfly`` – read a payload from a butterfly-selected SIMT lane."""
    return wrap_surface_value(_pto.ShuffleBflyOp(
        unwrap_surface_value(value),
        _coerce_i32_operand(mask, context="shuffle_bfly(..., mask)"),
        width=_validate_shuffle_width(width, context="shuffle_bfly(..., width)"),
    ).result)


def redux_add(value, *, signedness=None):
    """``pto.redux_add`` – SIMT lane sum reduction."""
    raw_value = unwrap_surface_value(value)
    _validate_redux_signedness(raw_value.type, signedness, require_for_integer=False, context="redux_add(value)")
    return wrap_surface_value(_pto.ReduxAddOp(
        raw_value,
        signedness=_optional_signedness_attr(signedness, context="redux_add(..., signedness)"),
    ).result)


def redux_max(value, *, signedness=None):
    """``pto.redux_max`` – SIMT lane max reduction."""
    raw_value = unwrap_surface_value(value)
    _validate_redux_signedness(raw_value.type, signedness, require_for_integer=True, context="redux_max(value)")
    return wrap_surface_value(_pto.ReduxMaxOp(
        raw_value,
        signedness=_optional_signedness_attr(signedness, context="redux_max(..., signedness)"),
    ).result)


def redux_min(value, *, signedness=None):
    """``pto.redux_min`` – SIMT lane min reduction."""
    raw_value = unwrap_surface_value(value)
    _validate_redux_signedness(raw_value.type, signedness, require_for_integer=True, context="redux_min(value)")
    return wrap_surface_value(_pto.ReduxMinOp(
        raw_value,
        signedness=_optional_signedness_attr(signedness, context="redux_min(..., signedness)"),
    ).result)


def ldg(ptr_or_ref, offset=None, *, l1cache="cache", l2cache="nmfv"):
    """``pto.ldg`` – scalar GM load with cache controls."""
    buffer_value, index_value = resolve_address_access(ptr_or_ref, offset)
    result_type = _pointer_element_type(buffer_value, context="ldg(ptr, offset)")
    return wrap_surface_value(_pto.PTOLdgOp(
        result_type,
        buffer_value,
        index_value,
        l1cache=_l1_cache_attr(l1cache, context="ldg(..., l1cache)"),
        l2cache=_ld_l2_cache_attr(l2cache, context="ldg(..., l2cache)"),
    ).value)


def stg(value, ptr_or_ref, offset=None, *, l1cache="cache", l2cache="nmfv"):
    """``pto.stg`` – scalar GM store with cache controls."""
    buffer_value, index_value = resolve_address_access(ptr_or_ref, offset)
    elem_type = _pointer_element_type(buffer_value, context="stg(value, ptr, offset)")
    _pto.PTOStgOp(
        buffer_value,
        index_value,
        coerce_scalar_to_type(value, elem_type, context="stg(value, ...)"),
        l1cache=_l1_cache_attr(l1cache, context="stg(..., l1cache)"),
        l2cache=_st_l2_cache_attr(l2cache, context="stg(..., l2cache)"),
    )


def _atomic_binary(op_cls, ptr, value, *, l2cache, signedness, context: str):
    raw_ptr = unwrap_surface_value(ptr)
    elem_type = _pointer_element_type(raw_ptr, context=context)
    _validate_integer_signedness_only(elem_type, signedness, context=context)
    raw_value = coerce_scalar_to_type(value, elem_type, context=context)
    return wrap_surface_value(op_cls(
        raw_value.type,
        raw_ptr,
        raw_value,
        l2cache=_st_l2_cache_attr(l2cache, context=f"{context} l2cache"),
        signedness=_optional_signedness_attr(signedness, context=f"{context} signedness"),
    ).old)


def atomic_exch(ptr, value, *, l2cache="nmfv", signedness=None):
    """``pto.atomic_exch`` – SIMT scalar atomic exchange."""
    return _atomic_binary(_pto.AtomicExchOp, ptr, value, l2cache=l2cache, signedness=signedness, context="atomic_exch(ptr, value)")


def atomic_add(ptr, value, *, l2cache="nmfv", signedness=None):
    """``pto.atomic_add`` – SIMT scalar atomic add."""
    return _atomic_binary(_pto.AtomicAddOp, ptr, value, l2cache=l2cache, signedness=signedness, context="atomic_add(ptr, value)")


def atomic_sub(ptr, value, *, l2cache="nmfv", signedness=None):
    """``pto.atomic_sub`` – SIMT scalar atomic subtract."""
    return _atomic_binary(_pto.AtomicSubOp, ptr, value, l2cache=l2cache, signedness=signedness, context="atomic_sub(ptr, value)")


def atomic_min(ptr, value, *, l2cache="nmfv", signedness=None):
    """``pto.atomic_min`` – SIMT scalar atomic min."""
    return _atomic_binary(_pto.AtomicMinOp, ptr, value, l2cache=l2cache, signedness=signedness, context="atomic_min(ptr, value)")


def atomic_max(ptr, value, *, l2cache="nmfv", signedness=None):
    """``pto.atomic_max`` – SIMT scalar atomic max."""
    return _atomic_binary(_pto.AtomicMaxOp, ptr, value, l2cache=l2cache, signedness=signedness, context="atomic_max(ptr, value)")


def atomic_and(ptr, value, *, l2cache="nmfv", signedness=None):
    """``pto.atomic_and`` – SIMT scalar atomic bitwise and."""
    return _atomic_binary(_pto.AtomicAndOp, ptr, value, l2cache=l2cache, signedness=signedness, context="atomic_and(ptr, value)")


def atomic_or(ptr, value, *, l2cache="nmfv", signedness=None):
    """``pto.atomic_or`` – SIMT scalar atomic bitwise or."""
    return _atomic_binary(_pto.AtomicOrOp, ptr, value, l2cache=l2cache, signedness=signedness, context="atomic_or(ptr, value)")


def atomic_xor(ptr, value, *, l2cache="nmfv", signedness=None):
    """``pto.atomic_xor`` – SIMT scalar atomic bitwise xor."""
    return _atomic_binary(_pto.AtomicXorOp, ptr, value, l2cache=l2cache, signedness=signedness, context="atomic_xor(ptr, value)")


def atomic_cas(ptr, compare, value, *, l2cache="nmfv", signedness=None):
    """``pto.atomic_cas`` – SIMT scalar atomic compare-and-swap."""
    raw_ptr = unwrap_surface_value(ptr)
    elem_type = _pointer_element_type(raw_ptr, context="atomic_cas(ptr, compare, value)")
    _validate_integer_signedness_only(elem_type, signedness, context="atomic_cas(ptr, compare, value)")
    raw_compare = coerce_scalar_to_type(compare, elem_type, context="atomic_cas(compare)")
    raw_value = coerce_scalar_to_type(value, elem_type, context="atomic_cas(value)")
    return wrap_surface_value(_pto.AtomicCasOp(
        raw_ptr,
        raw_compare,
        raw_value,
        l2cache=_st_l2_cache_attr(l2cache, context="atomic_cas(..., l2cache)"),
        signedness=_optional_signedness_attr(signedness, context="atomic_cas(..., signedness)"),
    ).old)


def prmt(lhs, rhs, selector):
    """``pto.prmt`` – SIMT scalar byte permutation."""
    return wrap_surface_value(_pto.PrmtOp(
        _coerce_i32_operand(lhs, context="prmt(lhs, ...)"),
        _coerce_i32_operand(rhs, context="prmt(..., rhs, ...)"),
        _coerce_i32_operand(selector, context="prmt(..., selector)"),
    ).result)


def mulhi(lhs, rhs, *, signedness):
    """``pto.mulhi`` – high half of an integer product."""
    raw_lhs = unwrap_surface_value(lhs)
    raw_rhs = coerce_scalar_to_type(rhs, raw_lhs.type, context="mulhi(lhs, rhs)")
    return wrap_surface_value(_pto.MulhiOp(
        raw_lhs,
        raw_rhs,
        _required_signedness_attr(signedness, context="mulhi(..., signedness)"),
    ).result)


def mul_i32toi64(lhs, rhs, *, signedness):
    """``pto.mul_i32toi64`` – widened i32 product."""
    return wrap_surface_value(_pto.MulI32ToI64Op(
        _coerce_i32_operand(lhs, context="mul_i32toi64(lhs, ...)"),
        _coerce_i32_operand(rhs, context="mul_i32toi64(..., rhs)"),
        _required_signedness_attr(signedness, context="mul_i32toi64(..., signedness)"),
    ).result)


def absf(value):
    """``pto.absf`` – SIMT floating absolute value."""
    return _same_type_unary(_pto.AbsFOp, value)


def sqrt(value):
    """``pto.sqrt`` – SIMT floating square root."""
    return _same_type_unary(_pto.SqrtOp, value)


def exp(value):
    """``pto.exp`` – SIMT floating exponential."""
    return _same_type_unary(_pto.ExpOp, value)


def log(value):
    """``pto.log`` – SIMT floating natural logarithm."""
    return _same_type_unary(_pto.LogOp, value)


def pow(lhs, rhs):
    """``pto.pow`` – SIMT floating power."""
    return _same_type_binary(_pto.PowOp, lhs, rhs, context="pow(lhs, rhs)")


def ceil(value):
    """``pto.ceil`` – SIMT floating ceil."""
    return _same_type_unary(_pto.CeilOp, value)


def floor(value):
    """``pto.floor`` – SIMT floating floor."""
    return _same_type_unary(_pto.FloorOp, value)


def rint(value):
    """``pto.rint`` – SIMT floating rint."""
    return _same_type_unary(_pto.RintOp, value)


def round(value):
    """``pto.round`` – SIMT floating round."""
    return _same_type_unary(_pto.RoundOp, value)


def fmin(lhs, rhs):
    """``pto.fmin`` – SIMT floating minimum."""
    return _same_type_binary(_pto.FMinOp, lhs, rhs, context="fmin(lhs, rhs)")


def fmax(lhs, rhs):
    """``pto.fmax`` – SIMT floating maximum."""
    return _same_type_binary(_pto.FMaxOp, lhs, rhs, context="fmax(lhs, rhs)")


def fma(lhs, rhs, acc):
    """``pto.fma`` – SIMT floating fused multiply-add."""
    return _same_type_ternary(_pto.FmaOp, lhs, rhs, acc, context="fma(lhs, rhs, acc)")


def convert(src, dst_type, *, rounding, saturation, signedness=None):
    """``pto.convert`` – SIMT scalar or packed conversion."""
    raw_src = unwrap_surface_value(src)
    raw_dst_type = _resolve(dst_type)
    _validate_convert_signedness(raw_src.type, raw_dst_type, signedness, context="convert(src, dst_type)")
    return wrap_surface_value(_pto.ConvertOp(
        raw_dst_type,
        raw_src,
        _rounding_attr(rounding, context="convert(..., rounding)"),
        _saturation_attr(saturation, context="convert(..., saturation)"),
        signedness=_optional_signedness_attr(signedness, context="convert(..., signedness)"),
    ).dst)


def syncthreads():
    """``pto.syncthreads`` – synchronize SIMT workitems."""
    _pto.SyncthreadsOp()


def threadfence():
    """``pto.threadfence`` – issue a SIMT workitem memory fence."""
    _pto.ThreadfenceOp()


def threadfence_block():
    """``pto.threadfence_block`` – issue a SIMT block-scoped memory fence."""
    _pto.ThreadfenceBlockOp()


def _slot_attr_value(slot, *, context: str):
    if not isinstance(slot, int) or isinstance(slot, bool):
        raise TypeError(f"{context} expects a non-negative Python int slot")
    if slot < 0:
        raise ValueError(f"{context} expects a non-negative slot, got {slot}")
    return slot


def keep(payload, *, slot):
    """``pto.keep`` – preserve a SIMT scalar payload in an explicit slot."""
    _pto.KeepOp(unwrap_surface_value(payload), _slot_attr_value(slot, context="keep(..., slot)"))


def resume(result_type, *, slot):
    """``pto.resume`` – restore a SIMT scalar payload from an explicit slot."""
    return wrap_surface_value(_pto.ResumeOp(
        _resolve(result_type),
        _slot_attr_value(slot, context="resume(..., slot)"),
    ).result)


def pipe_barrier(pipe):
    """``pto.pipe_barrier(pipe)`` – drain the specified hardware pipeline."""
    _pto.BarrierOp(_pipe_attr(pipe))


def get_buf(pipe, buf_id, mode=0):
    """``pto.get_buf(pipe, buf_id, mode=0)`` – acquire a buffer token.

    ``buf_id`` accepts a static integer (0–31) or a runtime index-like PTO scalar.
    """
    buf_id_op, is_static = _buf_id_operand(
        buf_id,
        context="get_buf(..., buf_id=...)",
    )
    if is_static:
        _pto.GetBufOp(_pipe_attr(pipe), buf_id_op, mode=mode)
        return
    _pto.GetBufDynOp(
        _pipe_attr(pipe),
        mode=mode,
        buf_id=buf_id_op,
    )


def rls_buf(pipe, buf_id, mode=0):
    """``pto.rls_buf(pipe, buf_id, mode=0)`` – release a buffer token.

    ``buf_id`` accepts a static integer (0–31) or a runtime index-like PTO scalar.
    """
    buf_id_op, is_static = _buf_id_operand(
        buf_id,
        context="rls_buf(..., buf_id=...)",
    )
    if is_static:
        _pto.RlsBufOp(_pipe_attr(pipe), buf_id_op, mode=mode)
        return
    _pto.RlsBufDynOp(
        _pipe_attr(pipe),
        mode=mode,
        buf_id=buf_id_op,
    )


def _buf_id_operand(buf_id, *, context: str):
    if isinstance(buf_id, int):
        _validate_static_buf_id(buf_id, context=context)
        return buf_id, True
    return _coerce_index(buf_id, context=context), False


def _sync_event_id_operand(event_id, *, context: str):
    _validate_static_event_id(event_id, context=context)
    return event_id if isinstance(event_id, int) else unwrap_surface_value(event_id)


def _sync_event_id_operand_in_range(event_id, *, context: str, lo: int, hi: int, meaning: str = "event_id"):
    _validate_static_event_id_range(event_id, context=context, lo=lo, hi=hi, meaning=meaning)
    return event_id if isinstance(event_id, int) else unwrap_surface_value(event_id)


def _flag_event_id_operand(event_id, *, context: str):
    if isinstance(event_id, int):
        _validate_static_event_id(event_id, context=context)
        return event_id, True
    return _coerce_index(event_id, context=context), False


def set_cross_flag(pipe, event_id):
    """``pto.set_cross_flag(pipe, event_id)`` – cross-core sync facade for ``pto.sync.set``."""
    _validate_sync_pipe(pipe, context="set_cross_flag(pipe, event_id)", allowed=("PIPE_FIX",))
    event_operand = _sync_event_id_operand(event_id, context="set_cross_flag(..., event_id=...)")
    _pto.sync_set(_pipe_attr(pipe), event_operand)


def wait_cross_flag(pipe, event_id):
    """``pto.wait_cross_flag(pipe, event_id)`` – cross-core sync facade for ``pto.sync.wait``."""
    _validate_sync_pipe(pipe, context="wait_cross_flag(pipe, event_id)", allowed=("PIPE_FIX",))
    event_operand = _sync_event_id_operand(event_id, context="wait_cross_flag(..., event_id=...)")
    _pto.sync_wait(_pipe_attr(pipe), event_operand)


def set_intra_flag(pipe, event_id):
    """``pto.set_intra_flag(pipe, event_id)`` – intra-block sync facade for ``pto.sync.set``."""
    _validate_sync_pipe(
        pipe,
        context="set_intra_flag(pipe, event_id)",
        allowed=("PIPE_FIX", "PIPE_MTE1", "PIPE_MTE2", "PIPE_MTE3", "PIPE_V", "PIPE_S"),
    )
    event_operand = _sync_event_id_operand_in_range(
        event_id,
        context="set_intra_flag(..., event_id=...)",
        lo=0,
        hi=31,
        meaning="physical event_id",
    )
    _pto.sync_set(_pipe_attr(pipe), event_operand)


def wait_intra_flag(pipe, event_id):
    """``pto.wait_intra_flag(pipe, event_id)`` – intra-block sync facade for ``pto.sync.wait``."""
    _validate_sync_pipe(
        pipe,
        context="wait_intra_flag(pipe, event_id)",
        allowed=("PIPE_FIX", "PIPE_MTE1", "PIPE_MTE2", "PIPE_MTE3", "PIPE_V", "PIPE_S"),
    )
    event_operand = _sync_event_id_operand_in_range(
        event_id,
        context="wait_intra_flag(..., event_id=...)",
        lo=0,
        hi=31,
        meaning="physical event_id",
    )
    _pto.sync_wait(_pipe_attr(pipe), event_operand)


def set_flag(src: str, dst: str, *, event_id: int = 0):
    """``pto.set_flag[src, dst, event_id]``.

    Accepts short pipe names (``"MTE2"``, ``"V"``, …) or full ``"PIPE_MTE2"``
    names.  Static ``event_id`` values in ``[0, 7]`` lower to ``pto.set_flag``;
    runtime index-like values lower to ``pto.set_flag_dyn``.
    """
    event_operand, is_static = _flag_event_id_operand(
        event_id,
        context="set_flag(..., event_id=...)",
    )
    if is_static:
        _pto.set_flag(_pipe_attr(src), _pipe_attr(dst), _event_attr(event_operand))
        return
    _pto.set_flag_dyn(_pipe_attr(src), _pipe_attr(dst), event_operand)


def wait_flag(src: str, dst: str, *, event_id: int = 0):
    """``pto.wait_flag[src, dst, event_id]``.

    Static ``event_id`` values in ``[0, 7]`` lower to ``pto.wait_flag``;
    runtime index-like values lower to ``pto.wait_flag_dyn``.
    """
    event_operand, is_static = _flag_event_id_operand(
        event_id,
        context="wait_flag(..., event_id=...)",
    )
    if is_static:
        _pto.wait_flag(_pipe_attr(src), _pipe_attr(dst), _event_attr(event_operand))
        return
    _pto.wait_flag_dyn(_pipe_attr(src), _pipe_attr(dst), event_operand)


def reserve_buffer(name, *, size, location, auto=True, base=None):
    """``pto.reserve_buffer(name, size, location, auto=True, base=None)``."""
    space = _normalize_address_space(location)
    if space not in (_pto.AddressSpace.VEC, _pto.AddressSpace.MAT):
        raise ValueError(
            "reserve_buffer(location=...) expects 'vec' or 'mat' address space"
        )
    op = _pto.ReserveBufferOp(
        name,
        size,
        _pto.AddressSpaceAttr.get(space),
        bool(auto),
        base=base,
    )
    return wrap_surface_value(op.result)


def import_reserved_buffer(name, *, peer_func):
    """``pto.import_reserved_buffer(name, peer_func=...)``."""
    if not isinstance(peer_func, str):
        spec = getattr(peer_func, "spec", None)
        role = getattr(spec, "role", None)
        role_value = getattr(role, "value", role)
        if role_value == "simt":
            from ._tracing.active import require_active_session
            session = require_active_session("pto.import_reserved_buffer")
            peer_func = session.resolve_simt_peer_symbol(peer_func)
        else:
            peer_func = getattr(spec, "symbol_name", None) \
                or getattr(peer_func, "__name__", None) \
                or str(peer_func)
    op = _pto.ImportReservedBufferOp(name, peer_func)
    return wrap_surface_value(op.result)


__all__ = [
    "const",
    "castptr", "addptr",
    "vlds", "vldas", "vldus", "vldsx2", "vsts", "vstsx2",
    "init_align",
    "plt_b8", "plt_b16", "plt_b32",
    "pset_b8", "pset_b16", "pset_b32",
    "pge_b8", "pge_b16", "pge_b32",
    "make_mask",
    "pand", "por", "pxor", "pnot", "psel",
    "pbitcast", "vcvt", "vpack", "vmulscvt", "ppack", "punpack",
    "pintlv_b8", "pintlv_b16", "pintlv_b32",
    "pdintlv_b8", "pdintlv_b16", "pdintlv_b32",
    "vintlv", "vdintlv",
    "vgather2", "vgather2_bc", "vgatherb", "vscatter", "vsldb", "vsstb",
    "vcmp", "vcmps",
    "plds", "psts", "pstu", "vstar", "vstas", "vstur", "vstus",
    "vbitcast",
    "vbr",
    "vadd", "vsub", "vmul", "vdiv", "vmax", "vmin",
    "vand", "vor", "vxor", "vshl", "vshr",
    "vcmax", "vcadd", "vcmin", "vdup", "vexpdif",
    "vexp", "vln", "vsqrt", "vabs", "vneg", "vrec", "vrsqrt", "vrelu", "vnot",
    "vcgmax", "vcgadd", "vcgmin", "vcpadd",
    "vadds", "vsubs", "vmuls", "vmaxs", "vmins", "vlrelu", "vshrs", "vshls", "vands", "vors", "vxors",
    "vaxpy", "vmula", "vci", "vaddrelu", "vsubrelu",
    "vsel",
    "make_tensor_view", "partition_view",
    "alloc_buffer", "alloc_tile",
    "tload", "tstore", "tmov", "tinsert",
    "tmatmul", "tmatmul_acc", "tmatmul_mx", "tmatmul_mx_acc", "tmatmul_mx_bias",
    "tgemv_mx", "tgemv_mx_acc", "tgemv_mx_bias",
    "tadd", "taddrelu", "tsub", "tmul", "tdiv", "tmax", "tmin",
    "tadds", "tsubs", "tmuls", "tdivs", "tmaxs", "tmins",
    "texp", "tlog", "tsqrt", "trsqrt", "trecip", "tabs", "tneg",
    "trelu", "tlrelu",
    "trowsum", "trowmax", "trowmin", "trowprod", "trowargmax", "trowargmin",
    "tcolsum", "tcolmax", "tcolmin", "tcolprod", "tcolargmax", "tcolargmin",
    "tcmp", "tcmps",
    "texpands", "treshape", "trowexpand", "tcolexpand",
    "trowexpandadd", "trowexpandsub", "trowexpandmul", "trowexpanddiv", "trowexpandmax", "trowexpandmin", "trowexpandexpdif",
    "tcolexpandadd", "tcolexpandsub", "tcolexpandmul", "tcolexpanddiv", "tcolexpandmax", "tcolexpandmin", "tcolexpandexpdif",
    "tsort32", "tmrgsort", "tgather",
    "tsel", "tsels", "tcvt",
    "tnot", "tand", "tands", "tor", "tors", "txor", "txors", "tshl", "tshls", "tshr", "tshrs",
    "tpartadd", "tpartmul", "tpartmax", "tpartmin",
    "tfillpad", "tfillpad_expand", "tfillpad_inplace",
    "as_ptr",
    "mte_load", "mte_store", "mte_gm_ub", "mte_ub_gm", "mte_ub_ub", "mte_ub_l1",
    "mte_gm_l1", "mte_l1_ub", "mte_gm_l1_frac", "mte_l1_bt", "mte_l1_fb", "mem_bar",
    "mte_l1_l0a", "mte_l1_l0b", "mte_l1_l0a_mx", "mte_l1_l0b_mx",
    "mte_l0c_l1", "mte_l0c_gm", "mte_l0c_ub",
    "mad", "mad_acc", "mad_bias", "mad_mx", "mad_mx_acc", "mad_mx_bias",
    "get_block_idx", "get_block_num", "get_subblock_idx", "get_subblock_num",
    "store_vfsimt_info", "simt_launch",
    "get_tid", "get_tid_x", "get_tid_y", "get_tid_z",
    "get_block_dim", "get_block_dim_x", "get_block_dim_y", "get_block_dim_z",
    "get_grid_dim", "get_grid_dim_x", "get_grid_dim_y", "get_grid_dim_z",
    "get_block_idx_x", "get_block_idx_y", "get_block_idx_z",
    "get_veccoreid", "get_clock32", "get_clock64",
    "get_laneid", "get_lanemask_eq", "get_lanemask_le", "get_lanemask_lt",
    "get_lanemask_ge", "get_lanemask_gt",
    "vote_all", "vote_any", "vote_uni", "vote_ballot",
    "shuffle_idx", "shuffle_up", "shuffle_down", "shuffle_bfly",
    "redux_add", "redux_max", "redux_min",
    "ldg", "stg",
    "atomic_exch", "atomic_add", "atomic_sub", "atomic_min", "atomic_max",
    "atomic_and", "atomic_or", "atomic_xor", "atomic_cas",
    "prmt", "mulhi", "mul_i32toi64",
    "absf", "sqrt", "exp", "log", "pow", "ceil", "floor", "rint", "round",
    "fmin", "fmax", "fma", "convert",
    "syncthreads", "threadfence", "threadfence_block", "keep", "resume",
    "pipe_barrier", "get_buf", "rls_buf",
    "set_cross_flag", "wait_cross_flag", "set_intra_flag", "wait_intra_flag",
    "set_flag", "wait_flag",
    "reserve_buffer", "import_reserved_buffer",
]
