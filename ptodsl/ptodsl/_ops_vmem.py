# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Vector memory ops: loads/stores, cvt/pack contracts, gather/scatter."""

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

from ._ops_common import (
    _coerce_index,
    _coerce_scalar_like_vector_element,
    _element_bytewidth,
    _infer_vreg_metadata,
    _infer_vreg_type_from_address_source,
    _infer_vreg_type_from_tile_slice,
    _normalize_post_update_mode,
    _normalize_vcvt_round_mode,
    _normalize_vpack_part,
    _tile_slice_address,
    _tile_slice_ptr,
)


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


def _vector_memory_dist_kwargs(dist, *, allowed: set[str], context: str):
    if dist is None:
        return {}
    return {
        "dist": _normalize_dist_token(
            dist,
            allowed=allowed,
            context=context,
        )
    }


def _vlds_tile_slice(src_ptr, *, dist, post_mode):
    if post_mode != "NO_POST_UPDATE":
        raise TypeError(
            "vlds(tile[...], post_update=...) only supports post_update=PostUpdate.OFF; "
            "use the pointer form for stateful loads"
        )
    source, source_offset = _tile_slice_address(src_ptr)
    return wrap_surface_value(
        _pto.VldsOp(
            _infer_vreg_type_from_tile_slice(src_ptr),
            None,
            unwrap_surface_value(source),
            source_offset,
            **_vector_memory_dist_kwargs(
                dist,
                allowed=_VLOAD_DIST_TOKENS,
                context="vlds(..., dist)",
            ),
        ).result
    )


def _vlds_pointer(src_ptr, offset, result_vreg_type, *, dist, post_mode):
    raw_source = unwrap_surface_value(src_ptr)
    result_type = _resolve(result_vreg_type)
    raw_offset = _coerce_index(offset, context="vlds(ptr, offset)")
    kwargs = _vector_memory_dist_kwargs(
        dist,
        allowed=_VLOAD_DIST_TOKENS,
        context="vlds(..., dist)",
    )
    if post_mode != "POST_UPDATE":
        return wrap_surface_value(
            _pto.VldsOp(result_type, None, raw_source, raw_offset, **kwargs).result
        )

    post_ctor = getattr(_pto, "VldsPostOp", None)
    if post_ctor is not None:
        op = post_ctor(result_type, raw_source.type, raw_source, raw_offset, **kwargs)
        return wrap_surface_value(op.result), wrap_surface_value(op.updated_source)
    op = _pto.VldsOp(result_type, raw_source.type, raw_source, raw_offset, **kwargs)
    return wrap_surface_value(op.result), wrap_surface_value(op.updated_base)


def vlds(src_ptr, offset=None, result_vreg_type=None, *, dist=None, post_update="OFF"):
    """``pto.vlds`` – vector load from a tile slice or from *src_ptr* at *offset*."""
    post_mode = _normalize_post_update_mode(post_update, context="vlds(..., post_update=...)")
    if isinstance(src_ptr, TileSliceValue):
        if offset is not None or result_vreg_type is not None:
            raise TypeError(
                "vlds(tile[row, col:]) infers its pointer slice and vreg type; "
                "do not pass offset/result_vreg_type")
        return _vlds_tile_slice(src_ptr, dist=dist, post_mode=post_mode)

    if offset is None:
        raise TypeError("vlds(ptr, offset, result_vreg_type=None) requires an explicit offset")
    if result_vreg_type is None:
        result_vreg_type = _infer_vreg_type_from_address_source(src_ptr)
    return _vlds_pointer(
        src_ptr,
        offset,
        result_vreg_type,
        dist=dist,
        post_mode=post_mode,
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
        None,
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


def vldsx2(source, offset_or_dist, dist=None, *, result_vreg_type=None):
    """``pto.vldsx2`` – dual vector load with deinterleave."""
    if isinstance(source, TileSliceValue):
        if dist is not None:
            raise TypeError("vldsx2(tile[row, col:], dist) does not accept a separate offset argument")
        result_type = _infer_vreg_type_from_tile_slice(source)
        source, source_offset = _tile_slice_address(source)
        op = _pto.Vldsx2Op(
            result_type,
            result_type,
            None,
            unwrap_surface_value(source),
            source_offset,
            _normalize_dist_token(
                offset_or_dist,
                allowed=_DEINTERLEAVE_DIST_TOKENS,
                context="vldsx2(..., dist)",
            ),
        )
        return wrap_surface_value(op.low), wrap_surface_value(op.high)

    if dist is None:
        raise TypeError("vldsx2(ptr, offset, dist) requires an explicit offset and dist")
    if result_vreg_type is not None:
        result_type = _resolve(result_vreg_type)
    else:
        result_type = _infer_vreg_type_from_address_source(source)
    op = _pto.Vldsx2Op(
        result_type,
        result_type,
        None,
        unwrap_surface_value(source),
        _coerce_index(offset_or_dist, context="vldsx2(ptr, offset, dist)"),
        _normalize_dist_token(
            dist,
            allowed=_DEINTERLEAVE_DIST_TOKENS,
            context="vldsx2(..., dist)",
        ),
    )
    return wrap_surface_value(op.low), wrap_surface_value(op.high)

def pbitcast(mask_value, to_type):
    """``pto.pbitcast`` – reinterpret one mask register at a different granularity."""
    return wrap_surface_value(
        _pto.PbitcastOp(
            _resolve(to_type),
            unwrap_surface_value(mask_value),
        ).result
    )

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





_VCVT_BUILTIN_FLOAT_KINDS = (
    (Float8E4M3FNType, "f8e4m3"),
    (Float8E5M2Type, "f8e5m2"),
    (F16Type, "f16"),
    (BF16Type, "bf16"),
    (F32Type, "f32"),
)
_VCVT_PTO_FLOAT_KINDS = (
    ("HiF8Type", "hif8"),
    ("F4E1M2x2Type", "f4e1m2x2"),
    ("F4E2M1x2Type", "f4e2m1x2"),
)


def _classify_vcvt_integer_kind(elem_type):
    if IntegerType.isinstance(elem_type):
        int_type = IntegerType(elem_type)
        if int_type.width in {8, 16, 32}:
            prefix = "u" if int_type.is_unsigned else "s"
            return f"{prefix}{int_type.width}"
        if int_type.width == 64 and not int_type.is_unsigned:
            return "s64"
    return None


def _classify_vcvt_elem_kind(elem_type):
    for type_cls, kind in _VCVT_BUILTIN_FLOAT_KINDS:
        if type_cls.isinstance(elem_type):
            return kind
    for type_name, kind in _VCVT_PTO_FLOAT_KINDS:
        if _isinstance_pto_type(elem_type, type_name):
            return kind
    return _classify_vcvt_integer_kind(elem_type)


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


def _vsts_tile_slice(val, dst_ptr, mask_value, *, extra_mask, dist, post_mode):
    if extra_mask is not None:
        raise TypeError("vsts(vec, tile[row, col:], mask) does not accept a separate offset argument")
    if post_mode != "NO_POST_UPDATE":
        raise TypeError(
            "vsts(vec, tile[...], post_update=...) only supports post_update=PostUpdate.OFF; "
            "use the pointer form for stateful stores"
        )
    destination, destination_offset = _tile_slice_address(dst_ptr)
    _pto.VstsOp(
        None,
        unwrap_surface_value(val),
        unwrap_surface_value(destination),
        destination_offset,
        unwrap_surface_value(mask_value),
        **_vector_memory_dist_kwargs(
            dist,
            allowed=_VSTORE_DIST_TOKENS,
            context="vsts(..., dist)",
        ),
    )


def _vsts_pointer(val, dst_ptr, offset, mask_value, *, dist, post_mode):
    raw_destination = unwrap_surface_value(dst_ptr)
    raw_value = unwrap_surface_value(val)
    raw_offset = _coerce_index(offset, context="vsts(ptr, offset, mask)")
    raw_mask = unwrap_surface_value(mask_value)
    kwargs = _vector_memory_dist_kwargs(
        dist,
        allowed=_VSTORE_DIST_TOKENS,
        context="vsts(..., dist)",
    )
    if post_mode != "POST_UPDATE":
        _pto.VstsOp(None, raw_value, raw_destination, raw_offset, raw_mask, **kwargs)
        return None

    post_ctor = getattr(_pto, "VstsPostOp", None)
    if post_ctor is not None:
        op = post_ctor(
            raw_destination.type,
            raw_value,
            raw_destination,
            raw_offset,
            raw_mask,
            **kwargs,
        )
        return wrap_surface_value(op.updated_destination)
    op = _pto.VstsOp(
        raw_destination.type,
        raw_value,
        raw_destination,
        raw_offset,
        raw_mask,
        **kwargs,
    )
    return wrap_surface_value(op.updated_base)


def vsts(val, dst_ptr, offset, mask=None, *, dist=None, post_update="OFF"):
    """``pto.vsts`` – vector store to a tile slice or to *dst_ptr* at *offset*."""
    post_mode = _normalize_post_update_mode(post_update, context="vsts(..., post_update=...)")
    if isinstance(dst_ptr, TileSliceValue):
        return _vsts_tile_slice(
            val,
            dst_ptr,
            offset,
            extra_mask=mask,
            dist=dist,
            post_mode=post_mode,
        )
    if mask is None:
        raise TypeError("vsts(vec, ptr, offset, mask) requires an explicit mask")
    return _vsts_pointer(val, dst_ptr, offset, mask, dist=dist, post_mode=post_mode)


def vstsx2(low, high, dst_ptr, offset_or_dist, dist_or_mask=None, mask=None):
    """``pto.vstsx2`` – dual interleaving vector store."""
    if isinstance(dst_ptr, TileSliceValue):
        if mask is not None:
            raise TypeError("vstsx2(low, high, tile[row, col:], dist, mask) does not accept a separate offset argument")
        destination, destination_offset = _tile_slice_address(dst_ptr)
        _pto.Vstsx2Op(
            unwrap_surface_value(low),
            unwrap_surface_value(high),
            unwrap_surface_value(destination),
            destination_offset,
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
    """Scatter b8/b16/b32 values to UB using width-matched integer offsets.

    The b8 form consumes 128 offsets and stores the 128 even-numbered bytes of
    its 256-lane value vector. The b16 and b32 forms consume one offset per
    value lane.
    """
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
    if isinstance(source, TileSliceValue):
        source = _tile_slice_ptr(source)
    return wrap_surface_value(
        _pto.VsldbOp(
            result_type,
            None,
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
