# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Public PTODSL namespace for formal VMI APIs."""

from __future__ import annotations

import inspect
from collections.abc import Sequence

from ptoas.mlir.dialects import pto as _pto
from ptoas.mlir.ir import (
    BF16Type,
    F16Type,
    F32Type,
    Float8E4M3FNType,
    Float8E5M2Type,
    IntegerType,
    MemRefType,
)

from ._scalar_coercion import coerce_scalar_to_type
from ._surface_values import _coerce_index_value, _try_get_constant_index, unwrap_surface_value, wrap_surface_value
from ._types import (
    _ensure_tensor_storage_dtype,
    _resolve,
    _vmi_bf16x2,
    vmi_mask_type,
    vmi_vreg_type,
)


class _UnspecifiedArgument:
    def __repr__(self) -> str:
        return "UNSPECIFIED"


_UNSPECIFIED = _UnspecifiedArgument()


def _missing_vmi_support_error(op_name: str) -> NotImplementedError:
    return NotImplementedError(
        f"{op_name} is not available in the current PTO Python bindings or "
        "backend support. Rebuild PTO Python bindings and update VMI support "
        "for this operation before using the PTODSL pto.vmi surface."
    )


def _unsupported_vmi_feature_error(op_name: str, feature: str) -> NotImplementedError:
    return NotImplementedError(
        f"{op_name} {feature} is not available in the current generated VMI "
        "binding/backend support; update the PTO bindings or use an explicit "
        "supported VMI form."
    )


def _generated(op_name: str):
    fn = getattr(_pto, f"vmi_{op_name}", None)
    if fn is None:
        raise _missing_vmi_support_error(f"pto.vmi.{op_name}")
    return fn


def _raw(value):
    return unwrap_surface_value(value)


def _raw_sequence(values):
    if _is_sequence(values):
        return [_raw(value) for value in values]
    return [_raw(values)]


def _is_sequence(value) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def _wrap_result(result):
    if hasattr(result, "type"):
        return wrap_surface_value(result)
    # An Operation with no results (e.g. store ops) — return it unwrapped.
    if hasattr(result, "operation") or str(type(result)) == "<class 'ptoas.mlir._mlir_libs._mlir.ir.Operation'>":
        return result
    try:
        count = len(result)
    except TypeError:
        count = None
    if count is not None:
        return tuple(wrap_surface_value(result[index]) for index in range(count))
    if _is_sequence(result) or hasattr(result, "__iter__"):
        return tuple(wrap_surface_value(value) for value in result)
    return wrap_surface_value(result)


def _type_of(value):
    return _raw(value).type


def _require_result_type(result_type, *, context: str):
    if result_type is None:
        raise TypeError(f"{context} requires explicit result_type")
    return _resolve(result_type)


def _as_vmi_vreg_type(type_obj, *, context: str):
    vreg_type_cls = getattr(_pto, "VMIVRegType", None)
    if vreg_type_cls is None:
        raise _missing_vmi_support_error("!pto.vmi.vreg")
    try:
        return vreg_type_cls(type_obj)
    except Exception as exc:
        raise TypeError(f"{context} expects a !pto.vmi.vreg value, got {type_obj}") from exc


def _vmi_element_type(type_obj, *, context: str):
    return _as_vmi_vreg_type(type_obj, context=context).element_type


def _as_vmi_mask_type(type_obj, *, context: str):
    mask_type_cls = getattr(_pto, "VMIMaskType", None)
    if mask_type_cls is None:
        raise _missing_vmi_support_error("!pto.vmi.mask")
    try:
        return mask_type_cls(type_obj)
    except Exception as exc:
        raise TypeError(f"{context} expects a !pto.vmi.mask value, got {type_obj}") from exc


def _vmi_mask_element_count(mask_type, *, context: str):
    for attr in ("element_count", "elementCount"):
        value = getattr(mask_type, attr, None)
        if value is not None:
            return int(value)
    getter = getattr(mask_type, "getElementCount", None)
    if callable(getter):
        return int(getter())
    raise TypeError(f"{context} could not determine VMI mask lane count from {mask_type}")


def _vmi_layout_attr(type_obj):
    for attr in ("layout", "layout_attr"):
        value = getattr(type_obj, attr, None)
        if value is not None:
            return value
    for getter_name in ("getLayout", "getLayoutAttr"):
        getter = getattr(type_obj, getter_name, None)
        if callable(getter):
            value = getter()
            if value is not None:
                return value
    return None


def _pointer_element_type(type_obj, *, context: str):
    ptr_type_cls = getattr(_pto, "PtrType", None)
    if ptr_type_cls is not None:
        try:
            return ptr_type_cls(type_obj).element_type
        except Exception:
            pass
    try:
        return MemRefType(type_obj).element_type
    except Exception as exc:
        raise TypeError(f"{context} expects a pointer or memref source, got {type_obj}") from exc


def _type_bit_width(type_obj, *, context: str):
    if IntegerType.isinstance(type_obj):
        return IntegerType(type_obj).width
    if Float8E4M3FNType.isinstance(type_obj) or Float8E5M2Type.isinstance(type_obj):
        return 8
    if F16Type.isinstance(type_obj) or BF16Type.isinstance(type_obj):
        return 16
    if F32Type.isinstance(type_obj):
        return 32
    raise TypeError(f"{context} does not support element type {type_obj}")


def _is_vmi_float_element_type(type_obj) -> bool:
    return any(
        cls.isinstance(type_obj)
        for cls in (BF16Type, F16Type, F32Type, Float8E4M3FNType, Float8E5M2Type)
    )


def _normalize_vmi_vcvt_rounding(mode, *, context: str):
    token = mode
    if not isinstance(token, str):
        token = str(token)
        if "." in token:
            token = token.rsplit(".", 1)[-1]
    normalized = token.strip().upper()
    allowed = {"R", "A", "H", "Z"}
    if normalized not in allowed:
        expected = ", ".join(sorted(allowed))
        raise ValueError(
            f"{context} does not support rounding {mode!r}; expected one of {expected}"
        )
    return normalized


def _derive_vcvt_result_type(source, to_dtype, *, context: str):
    if to_dtype is None:
        raise TypeError(f"{context} requires to_dtype")
    source_type = _as_vmi_vreg_type(_type_of(source), context=context)
    elem_type = _ensure_tensor_storage_dtype(to_dtype, context=context)
    return _pto.VMIVRegType.get(
        source_type.element_count,
        elem_type,
        layout=source_type.layout,
    )


# Legal VMI vreg lane counts (1, 2, 4, 8, 64, 128, 256). Used by
# _derive_vinterpret_cast_result_type to validate a derived target count.
_VMI_LEGAL_LANE_COUNTS = frozenset({1, 2, 4, 8, 64, 128, 256})


def _derive_vinterpret_cast_result_type(source, to_dtype, *, context: str):
    if to_dtype is None:
        raise TypeError(f"{context} requires to_dtype")
    source_type = _as_vmi_vreg_type(_type_of(source), context=context)
    source_elem_type = source_type.element_type
    target_elem_type = _ensure_tensor_storage_dtype(to_dtype, context=context)
    source_elem_bits = _type_bit_width(source_elem_type, context=context)
    target_elem_bits = _type_bit_width(target_elem_type, context=context)
    source_total_bits = source_type.element_count * source_elem_bits
    if target_elem_bits == 0 or source_total_bits % target_elem_bits != 0:
        raise TypeError(
            f"{context} requires the source bit count to be divisible by the "
            f"target element width; got {source_type} ({source_total_bits} bits) "
            f"-> {target_elem_type} ({target_elem_bits} bits)"
        )
    target_count = source_total_bits // target_elem_bits
    if target_count not in _VMI_LEGAL_LANE_COUNTS:
        raise ValueError(
            f"{context} derived lane count {target_count} is outside the legal "
            f"domain 1, 2, 4, 8, 64, 128, 256"
        )
    # A bit-reinterpretation that changes the lane count must not reuse the
    # source layout: group/slot counts are tied to the source lanes.
    layout = source_type.layout
    if layout is not None and target_count != source_type.element_count:
        raise TypeError(
            f"{context} cannot carry a layout across a lane-count change "
            f"({source_type.element_count} -> {target_count}); the source "
            "layout is tied to the original lanes"
        )
    return _pto.VMIVRegType.get(
        target_count,
        target_elem_type,
        layout=layout,
    )


def _check_vci_group_tiles_phys_vl(elem_type, size, group, *, context: str):
    """Validate that a grouped vci size tiles the physical VL.

    For group > 1, each group must occupy a whole number of physical vregs
    (64, 128, or 256 lanes per group).  group=1 is equivalent to ungrouped
    and is always legal regardless of tiling.
    """
    if group is None or group <= 1:
        return
    if size % group != 0:
        raise ValueError(
            f"{context}: size {size} must be divisible by group {group}"
        )
    group_size = size // group
    if group_size <= 0:
        raise ValueError(
            f"{context}: group_size must be positive; got size={size}, group={group}"
        )
    # Physical VL depends on the element width: i32/f32 = 64 lanes,
    # i16/f16 = 128 lanes, i8 = 256 lanes.
    if IntegerType.isinstance(elem_type):
        width = IntegerType(elem_type).width
    elif _is_vmi_float_element_type(elem_type):
        from ptoas.mlir.ir import FloatType
        if FloatType.isinstance(elem_type):
            width = FloatType(elem_type).width
        else:
            width = 32
    else:
        width = 32
    phys_vl = 64 if width >= 32 else (128 if width >= 16 else 256)
    # The backend accepts a group whose per-group lane count tiles the
    # physical VL in either direction: a small group within one VL
    # (group_size < phys_vl, phys_vl % group_size == 0) or a group spanning
    # several VLs (group_size > phys_vl, group_size % phys_vl == 0).
    if phys_vl % group_size != 0 and group_size % phys_vl != 0:
        raise ValueError(
            f"{context}: group_size {group_size} does not tile physical lanes "
            f"({phys_vl}) in either direction; size={size}, group={group}"
        )


def _derive_vbrc_result_type(value, size, *, context: str):
    if size is None:
        raise TypeError(f"{context} requires size")
    raw_value = _raw(value)
    if not hasattr(raw_value, "type"):
        raise TypeError(
            f"{context} requires a typed scalar such as pto.f32(0.0) or "
            "a VMI vector input; plain Python scalars are ambiguous"
        )
    value_type = raw_value.type
    if _is_vmi_vreg_type(value_type):
        elem_type = _vmi_element_type(value_type, context=context)
    else:
        elem_type = value_type
    return _pto.VMIVRegType.get(size, elem_type)


def _derive_vci_result_type(base, size, *, context: str):
    if size is None:
        raise TypeError(f"{context} requires size")
    raw_base = _raw(base)
    if not hasattr(raw_base, "type"):
        raise TypeError(
            f"{context} requires a typed scalar such as pto.i32(0) or "
            "pto.f32(0.0); plain Python scalars are ambiguous"
        )
    base_type = raw_base.type
    # VMI vreg elements must be 8/16/32-bit.  An MLIR index (64-bit, used
    # for loop induction variables) is narrowed to signless i32 — this is
    # the documented dynamic-base case.  Any other unsupported width
    # (e.g. an explicit i64) must raise a clear error rather than silently
    # truncating and changing the value.
    if _is_vmi_float_element_type(base_type):
        pass  # float types are fine
    elif IntegerType.isinstance(base_type):
        width = IntegerType(base_type).width
        if width not in (8, 16, 32):
            # MLIR index is 64-bit signless; narrow it to i32 (dynamic base).
            if str(base_type) == "index":
                base_type = IntegerType.get_signless(32)
            else:
                raise TypeError(
                    f"{context} requires an 8/16/32-bit integer or float base; "
                    f"got {base_type} ({width}-bit)"
                )
    elif str(base_type) == "index":
        base_type = IntegerType.get_signless(32)
    else:
        raise TypeError(
            f"{context} requires a typed 8/16/32-bit scalar base; got {base_type}"
        )
    return _pto.VMIVRegType.get(size, base_type)


def _derive_vmull_result_types(a, b, *, context: str):
    lhs_type = _as_vmi_vreg_type(_type_of(a), context=context)
    rhs_type = _as_vmi_vreg_type(_type_of(b), context=context)
    if lhs_type != rhs_type:
        raise TypeError(f"{context} requires a and b to have identical VMI vreg types")
    element_type = lhs_type.element_type
    if not IntegerType.isinstance(element_type):
        raise TypeError(f"{context} requires 32-bit integer vectors")
    integer_type = IntegerType(element_type)
    if integer_type.width != 32:
        raise TypeError(f"{context} requires 32-bit integer vectors")
    return lhs_type, rhs_type


def _derive_hist_result_type(acc, *, context: str):
    """acc must be 16-bit unsigned or signless integer; result is always ui16."""
    acc_type = _as_vmi_vreg_type(_type_of(acc), context=context)
    element_type = acc_type.element_type
    if not IntegerType.isinstance(element_type):
        raise TypeError(
            f"{context} requires acc element type to be ui16 or i16, "
            f"got {element_type}"
        )
    int_type = IntegerType(element_type)
    if int_type.width != 16 or int_type.is_signed:
        raise TypeError(
            f"{context} requires acc element type to be ui16 or i16, "
            f"got {element_type}"
        )
    return _pto.VMIVRegType.get(
        acc_type.element_count,
        IntegerType.get_unsigned(16),
        layout=acc_type.layout,
    )


def _derive_vgather_result_type(source, offsets, *, context: str):
    offsets_type = _as_vmi_vreg_type(_type_of(offsets), context=context)
    result_type = _pointer_element_type(_type_of(source), context=context)
    return _pto.VMIVRegType.get(
        offsets_type.element_count,
        result_type,
        layout=offsets_type.layout,
    )


def _derive_vgatherb_result_type(source, mask, *, context: str):
    mask_type = _as_vmi_mask_type(_type_of(mask), context=context)
    result_element_type = _pointer_element_type(_type_of(source), context=context)
    result_layout = _vmi_layout_attr(mask_type)
    return _pto.VMIVRegType.get(
        _vmi_mask_element_count(mask_type, context=context),
        result_element_type,
        layout=result_layout,
    )


def _derive_vmi_reduce_result_type(source, group, *, context: str):
    source_type = _as_vmi_vreg_type(_type_of(source), context=context)
    result_lanes = 1
    if group is not None:
        try:
            result_lanes = int(group)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{context} requires group to be an integer when provided") from exc
        if result_lanes <= 0:
            raise TypeError(f"{context} requires group to be positive, got {group!r}")
    return _pto.VMIVRegType.get(result_lanes, source_type.element_type)


def _coerce_scalar_like_vmi_element(vector_value, scalar_value, *, context: str):
    elem_type = _vmi_element_type(_type_of(vector_value), context=context)
    return coerce_scalar_to_type(scalar_value, elem_type, context=context)


def _variadic_mask(mask):
    if mask is None:
        return []
    return _raw_sequence(mask)


def _vstore_accepts_updated_base_arg(fn) -> bool:
    try:
        parameters = tuple(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        return False
    return bool(parameters) and parameters[0] == "updated_base"


def _emit_vstore_generated(
    *,
    updated_base,
    values,
    destination,
    offset,
    mask,
    stride,
    block_stride,
    dist_mode,
    group,
    pmode,
    loc,
    ip,
):
    fn = _generated("vstore")
    raw_stride = None if stride is None else _coerce_index_value(stride)
    raw_block_stride = _i16_value(block_stride, context="pto.vmi.vstore(block_stride)")
    kwargs = {
        "stride": raw_stride,
        "block_stride": raw_block_stride,
        "dist_mode": dist_mode,
        "group": group,
        "pmode": pmode,
        "loc": loc,
        "ip": ip,
    }
    if _vstore_accepts_updated_base_arg(fn):
        op = fn(
            updated_base,
            values,
            destination,
            offset,
            mask,
            **kwargs,
        )
        if updated_base is None:
            return op
        updated = getattr(op, "updated_base", None)
        if updated is not None:
            return updated
        results = getattr(op, "results", None)
        if results:
            return results[0]
        operation = getattr(op, "operation", None)
        if operation is not None and getattr(operation, "results", None):
            return operation.results[0]
        return op
    if updated_base is not None:
        raise NotImplementedError(
            "pto.vmi.vstore(..., post_update=True) requires generated VMI "
            "Python bindings with an updated_base result"
        )
    return fn(
        values,
        destination,
        offset,
        mask,
        **kwargs,
    )


def _required_mask(mask, *, context: str):
    if mask is None:
        raise TypeError(f"{context} requires a mask operand")
    return _raw(mask)


def _required_variadic_mask(mask, *, context: str):
    if mask is None:
        raise TypeError(f"{context} requires a mask operand")
    return _raw_sequence(mask)


def _i16_value(value, *, context: str):
    if value is None:
        return None
    return coerce_scalar_to_type(value, IntegerType.get_signless(16), context=context)


def _resolve_vmi_mask_type(size, *, context: str):
    if size is None:
        raise TypeError(f"{context} requires size")
    return _resolve(vmi_mask_type(size))


def _vmi_vreg_element_count(type_obj, *, context: str):
    vreg_type = _as_vmi_vreg_type(type_obj, context=context)
    for attr in ("element_count", "elementCount"):
        value = getattr(vreg_type, attr, None)
        if value is not None:
            return int(value)
    getter = getattr(vreg_type, "getElementCount", None)
    if callable(getter):
        return int(getter())
    raise TypeError(f"{context} could not determine VMI vector lane count from {type_obj}")


def _resolve_vmi_unpack_result_type(source, size, to_dtype, *, context: str):
    if to_dtype is None:
        raise TypeError(f'{context} requires to_dtype when dist_mode="unpack"')
    source_type = _pointer_element_type(_type_of(source), context=context)
    result_type = _ensure_tensor_storage_dtype(to_dtype, context=context)
    source_bits = _type_bit_width(source_type, context=context)
    result_bits = _type_bit_width(result_type, context=context)
    if source_bits * 2 != result_bits:
        raise TypeError(
            f"{context} requires unpack to widen by exactly one step; got "
            f"{source_type} -> {result_type}"
        )
    return _pto.VMIVRegType.get(size, result_type)


def _resolve_vmi_vload_result_types(source, size, *, dist_mode, to_dtype, context: str):
    if to_dtype is not None and dist_mode != "unpack":
        raise TypeError(f'{context} accepts to_dtype only when dist_mode="unpack"')
    if size is None:
        raise TypeError(f"{context} requires size")
    if dist_mode == "unpack":
        return [_resolve_vmi_unpack_result_type(source, size, to_dtype, context=context)]
    element_type = _pointer_element_type(_type_of(source), context=context)
    resolved = _pto.VMIVRegType.get(size, element_type)
    if dist_mode == "dintlv":
        return [resolved, resolved]
    return [resolved]


def _validate_vmi_load_modes(
    context: str,
    *,
    dist_mode,
    group,
    stride,
    block_stride,
    allow_group_brc: bool,
    allowed_dist_modes,
):
    if dist_mode is not None and dist_mode not in allowed_dist_modes:
        expected = ", ".join(repr(mode) for mode in sorted(allowed_dist_modes, key=str))
        raise TypeError(f"{context} does not support dist_mode={dist_mode!r}; expected one of {expected}")

    if group is not None:
        if dist_mode is not None and (not allow_group_brc or dist_mode != "brc"):
            raise TypeError(f"{context} does not allow dist_mode together with group")
        if block_stride is not None:
            raise TypeError(f"{context} does not allow block_stride together with group")
        if stride is None:
            raise TypeError(f"{context} with group=... requires stride")
        return

    if block_stride is not None:
        if dist_mode is not None:
            raise TypeError(f"{context} does not allow dist_mode together with block_stride")
        if stride is not None:
            raise TypeError(f"{context} does not allow stride together with block_stride")
        return

    if stride is not None:
        raise TypeError(f"{context} accepts stride only when group is provided")


def _call_value(op_name: str, *args, **kwargs):
    return _wrap_result(_generated(op_name)(*args, **kwargs))


# VMI binary ops for which swapping scalar/vector operands preserves
# semantics (a ⊕ b == b ⊕ a).  Non-commutative ops (vsub/vdiv/vshl/vshr)
# must NOT swap; their scalar-vector form must broadcast the scalar and
# keep the original operand order.
_VMI_COMMUTATIVE_BINARY_OPS = frozenset(
    {"vadd", "vmul", "vmax", "vmin", "vand", "vor", "vxor"}
)


def _emit_binary(op_name: str, lhs, rhs, mask=None, *, pmode=None, loc=None, ip=None):
    context = f"pto.vmi.{op_name}(...)"
    raw_lhs = _raw(lhs)
    raw_rhs = _raw(rhs)
    lhs_has_type = hasattr(raw_lhs, "type")
    rhs_has_type = hasattr(raw_rhs, "type")
    lhs_is_vreg = lhs_has_type and _is_vmi_vreg_type(raw_lhs.type)
    rhs_is_vreg = rhs_has_type and _is_vmi_vreg_type(raw_rhs.type)
    if lhs_is_vreg and not rhs_is_vreg:
        # vector-scalar: delegate to the vec-scalar variant (e.g. vadd -> vadds)
        vec_scalar_name = op_name + "s"
        if hasattr(_pto, f"vmi_{vec_scalar_name}"):
            return _emit_vec_scalar(
                vec_scalar_name, lhs, rhs, mask, pmode=pmode, loc=loc, ip=ip
            )
        # Fallback: broadcast the scalar into a constant vector
        vreg_type = raw_lhs.type
        coerced = _coerce_scalar_like_vmi_element(lhs, rhs, context=context)
        raw_rhs = _raw(
            _call_value(
                "vbrc",
                vreg_type,
                coerced,
                loc=loc,
                ip=ip,
            )
        )
    elif rhs_is_vreg and not lhs_is_vreg:
        vec_scalar_name = op_name + "s"
        if op_name in _VMI_COMMUTATIVE_BINARY_OPS:
            # Commutative: scalar-vector normalizes to vec-scalar by
            # swapping (e.g. vadd(1, vec) -> vadds(vec, 1)).
            if hasattr(_pto, f"vmi_{vec_scalar_name}"):
                return _emit_vec_scalar(
                    vec_scalar_name, rhs, lhs, mask, pmode=pmode, loc=loc, ip=ip
                )
        # Non-commutative (vsub/vdiv/vshl/vshr) or no vec-scalar builder:
        # broadcast the scalar into a vector and keep the original operand
        # order so the operation's semantics are preserved.
        vreg_type = raw_rhs.type
        coerced = _coerce_scalar_like_vmi_element(rhs, lhs, context=context)
        raw_lhs = _raw(
            _call_value(
                "vbrc",
                vreg_type,
                coerced,
                loc=loc,
                ip=ip,
            )
        )
    else:
        vreg_type = raw_lhs.type if lhs_has_type else _type_of(lhs)
    return _call_value(
        op_name,
        vreg_type,
        raw_lhs,
        raw_rhs,
        _variadic_mask(mask),
        pmode=pmode,
        loc=loc,
        ip=ip,
    )


def _emit_unary(op_name: str, source, mask=None, *, pmode=None, loc=None, ip=None):
    return _call_value(
        op_name,
        _type_of(source),
        _raw(source),
        _variadic_mask(mask),
        pmode=pmode,
        loc=loc,
        ip=ip,
    )


def _emit_vec_scalar(op_name: str, source, scalar, mask, *, pmode=None, loc=None, ip=None):
    context = f"pto.vmi.{op_name}(...)"
    if op_name in ("vshls", "vshrs"):
        # Shift count operands must be I16 per the VMI hardware contract.
        scalar = coerce_scalar_to_type(
            scalar, IntegerType.get_signless(16), context=context
        )
    else:
        scalar = _coerce_scalar_like_vmi_element(source, scalar, context=context)
    return _call_value(
        op_name,
        _type_of(source),
        _raw(source),
        scalar,
        _required_mask(mask, context=context),
        pmode=pmode,
        loc=loc,
        ip=ip,
    )


def _emit_reduce(
    op_name: str,
    source,
    mask,
    *,
    group=None,
    pmode=None,
    loc=None,
    ip=None,
    reassoc=_UNSPECIFIED,
):
    context = f"pto.vmi.{op_name}(...)"
    if group is not None:
        if isinstance(group, bool) or not isinstance(group, int):
            raise TypeError(f"{context} requires group to be a positive integer")
        if group <= 0:
            raise ValueError(f"{context} requires group to be positive, got {group!r}")
        # The result lane count equals the group value; it must be a legal
        # VMI vreg width.
        if group not in (1, 2, 4, 8, 32, 64, 128, 256):
            raise ValueError(
                f"{context} requires group to be one of 1, 2, 4, 8, 64, 128, 256; "
                f"got {group}"
            )
    if op_name == "vcadd":
        source_elem_type = _vmi_element_type(_type_of(source), context=context)
        if reassoc is _UNSPECIFIED:
            if _is_vmi_float_element_type(source_elem_type):
                raise TypeError(
                    f"{context} on floating-point vectors requires an explicit reassoc "
                    "argument; spell out reassoc=True or reassoc=False"
                )
        elif not isinstance(reassoc, bool):
            raise TypeError(
                f"{context} requires reassoc to be the Python boolean True or False; "
                f"received {reassoc!r}"
            )
    kwargs = {"group": group if group is not None else 1, "pmode": pmode, "loc": loc, "ip": ip}
    if reassoc is not _UNSPECIFIED:
        # reassoc is a UnitAttr on the C++ side: its presence (not its value)
        # signals that the user made an explicit choice.  Always materialise
        # the attribute when the user explicitly specified reassoc (True or
        # False); the verifier requires it for floating-point sources.
        kwargs["reassoc"] = True
    return _call_value(
        op_name,
        _derive_vmi_reduce_result_type(source, group, context=context),
        _raw(source),
        _required_mask(mask, context=context),
        **kwargs,
    )


class _VMINamespace:
    vreg = staticmethod(vmi_vreg_type)
    mask = staticmethod(vmi_mask_type)
    bf16x2 = _vmi_bf16x2

    @staticmethod
    def vload(
        source,
        offset,
        *,
        size,
        to_dtype=None,
        stride=None,
        block_stride=None,
        dist_mode=None,
        group=None,
        loc=None,
        ip=None,
    ):
        _validate_vmi_load_modes(
            "pto.vmi.vload(...)",
            dist_mode=dist_mode,
            group=group,
            stride=stride,
            block_stride=block_stride,
            allow_group_brc=True,
            allowed_dist_modes={None, "continuous", "dintlv", "unpack", "brc"},
        )
        result_types = _resolve_vmi_vload_result_types(
            source,
            size,
            dist_mode=dist_mode,
            to_dtype=to_dtype,
            context="pto.vmi.vload(...)",
        )
        return _call_value(
            "vload",
            result_types,
            _raw(source),
            _coerce_index_value(offset),
            stride=None if stride is None else _coerce_index_value(stride),
            block_stride=_i16_value(block_stride, context="pto.vmi.vload(block_stride)"),
            dist_mode=dist_mode,
            group=group,
            loc=loc,
            ip=ip,
        )

    @staticmethod
    def vstore(
        values,
        destination,
        offset,
        mask=None,
        *,
        stride=None,
        block_stride=None,
        dist_mode=None,
        group=None,
        pmode=None,
        post_update=False,
        loc=None,
        ip=None,
    ):
        _validate_vmi_load_modes(
            "pto.vmi.vstore(...)",
            dist_mode=dist_mode,
            group=group,
            stride=stride,
            block_stride=block_stride,
            allow_group_brc=False,
            allowed_dist_modes={None, "continuous", "dintlv"},
        )
        if group is not None and mask is not None:
            raise TypeError("pto.vmi.vstore(...) group mode does not take a mask operand")
        if dist_mode == "dintlv":
            if not _is_sequence(values) or len(values) != 2:
                raise TypeError('pto.vmi.vstore(...) with dist_mode="dintlv" requires an (even, odd) pair')
        elif _is_sequence(values):
            raise TypeError("pto.vmi.vstore(...) expects a single VMI vector unless dist_mode=\"dintlv\"")
dest = _raw(destination)
        if post_update:
            # Produce the updated_base result (block-stride mode only): the
            # generated vstore builder takes the result type as `updated_base`.
            # Returns the updated dst pointer Value.
            op = _emit_vstore_generated(
                updated_base=_type_of(dest),
                values=_raw_sequence(values),
                destination=dest,
                offset=_coerce_index_value(offset),
                mask=_variadic_mask(mask),
                stride=stride,
                block_stride=block_stride,
                dist_mode=dist_mode,
                group=group,
                pmode=pmode,
                loc=loc,
                ip=ip,
            )
            return _wrap_result(op)
        return _emit_vstore_generated(
            updated_base=None,
            values=_raw_sequence(values),
            destination=dest,
            offset=_coerce_index_value(offset),
            mask=_variadic_mask(mask),
            stride=stride,
            block_stride=block_stride,
            dist_mode=dist_mode,
            group=group,
            pmode=pmode,
            loc=loc,
            ip=ip,
        )

    @staticmethod
    def vsstb(value, destination, offset, block_stride, mask, *, pmode=None, loc=None, ip=None):
        context = "pto.vmi.vsstb(...)"
        return _call_value(
            "vsstb",
            _raw(value),
            _raw(destination),
            _raw(offset),
            _raw(block_stride),
            _required_mask(mask, context=context),
            pmode=pmode,
            loc=loc,
            ip=ip,
        )

    @staticmethod
    def vci(base, *, size, order=None, group=None, loc=None, ip=None):
        context = "pto.vmi.vci(...)"
        result_type = _derive_vci_result_type(base, size, context=context)
        _check_vci_group_tiles_phys_vl(
            result_type.element_type, size, group, context=context
        )
        base = coerce_scalar_to_type(
            base,
            _vmi_element_type(result_type, context=context),
            context="pto.vmi.vci(base)",
        )
        return _call_value("vci", result_type, base, order=order, group=group, loc=loc, ip=ip)

    vadd = staticmethod(lambda lhs, rhs, mask=None, **kw: _emit_binary("vadd", lhs, rhs, mask, **kw))

    @staticmethod
    def vaddc(lhs, rhs, mask, *, loc=None, ip=None):
        context = "pto.vmi.vaddc(...)"
        source_type = _as_vmi_vreg_type(_type_of(lhs), context=context)
        mask_type = _resolve_vmi_mask_type(source_type.element_count, context=context)
        return _call_value(
            "vaddc",
            source_type,
            mask_type,
            _raw(lhs),
            _raw(rhs),
            _required_mask(mask, context=context),
            loc=loc,
            ip=ip,
        )

    @staticmethod
    def vaddcs(lhs, rhs, carry_in, mask, *, loc=None, ip=None):
        context = "pto.vmi.vaddcs(...)"
        source_type = _as_vmi_vreg_type(_type_of(lhs), context=context)
        mask_type = _resolve_vmi_mask_type(source_type.element_count, context=context)
        return _call_value(
            "vaddcs",
            source_type,
            mask_type,
            _raw(lhs),
            _raw(rhs),
            _required_mask(carry_in, context=context),
            _required_mask(mask, context=context),
            loc=loc,
            ip=ip,
        )

    vsub = staticmethod(lambda lhs, rhs, mask=None, **kw: _emit_binary("vsub", lhs, rhs, mask, **kw))
    vmul = staticmethod(lambda lhs, rhs, mask=None, **kw: _emit_binary("vmul", lhs, rhs, mask, **kw))
    vdiv = staticmethod(lambda lhs, rhs, mask=None, **kw: _emit_binary("vdiv", lhs, rhs, mask, **kw))
    vmax = staticmethod(lambda lhs, rhs, mask=None, **kw: _emit_binary("vmax", lhs, rhs, mask, **kw))
    vmin = staticmethod(lambda lhs, rhs, mask=None, **kw: _emit_binary("vmin", lhs, rhs, mask, **kw))
    vand = staticmethod(lambda lhs, rhs, mask=None, **kw: _emit_binary("vand", lhs, rhs, mask, **kw))
    vor = staticmethod(lambda lhs, rhs, mask=None, **kw: _emit_binary("vor", lhs, rhs, mask, **kw))
    vxor = staticmethod(lambda lhs, rhs, mask=None, **kw: _emit_binary("vxor", lhs, rhs, mask, **kw))
    vshl = staticmethod(lambda lhs, rhs, mask=None, **kw: _emit_binary("vshl", lhs, rhs, mask, **kw))
    vshr = staticmethod(lambda lhs, rhs, mask=None, **kw: _emit_binary("vshr", lhs, rhs, mask, **kw))

    vabs = staticmethod(lambda source, mask=None, **kw: _emit_unary("vabs", source, mask, **kw))
    vneg = staticmethod(lambda source, mask=None, **kw: _emit_unary("vneg", source, mask, **kw))
    vrelu = staticmethod(lambda source, mask=None, **kw: _emit_unary("vrelu", source, mask, **kw))
    vexp = staticmethod(lambda source, mask=None, **kw: _emit_unary("vexp", source, mask, **kw))
    vln = staticmethod(lambda source, mask=None, **kw: _emit_unary("vln", source, mask, **kw))
    vsqrt = staticmethod(lambda source, mask=None, **kw: _emit_unary("vsqrt", source, mask, **kw))
    vnot = staticmethod(lambda source, mask=None, **kw: _emit_unary("vnot", source, mask, **kw))

    def _deprecated_vec_scalar(op_name, source, scalar, mask, **kw):
        import warnings
        from ._diagnostics import PTODSLDeprecationWarning
        warnings.warn(
            f"pto.vmi.{op_name}(vector, scalar, mask) is deprecated; use "
            f"pto.vmi.{op_name[:-1]}(vector, scalar, mask) instead",
            PTODSLDeprecationWarning,
            stacklevel=2,
        )
        return _emit_vec_scalar(op_name, source, scalar, mask, **kw)

    vadds = staticmethod(
        lambda source, scalar, mask, **kw: _VMINamespace._deprecated_vec_scalar(
            "vadds", source, scalar, mask, **kw
        )
    )
    vmuls = staticmethod(
        lambda source, scalar, mask, **kw: _VMINamespace._deprecated_vec_scalar(
            "vmuls", source, scalar, mask, **kw
        )
    )
    vmaxs = staticmethod(
        lambda source, scalar, mask, **kw: _VMINamespace._deprecated_vec_scalar(
            "vmaxs", source, scalar, mask, **kw
        )
    )
    vmins = staticmethod(
        lambda source, scalar, mask, **kw: _VMINamespace._deprecated_vec_scalar(
            "vmins", source, scalar, mask, **kw
        )
    )
    vshls = staticmethod(
        lambda source, scalar, mask, **kw: _VMINamespace._deprecated_vec_scalar(
            "vshls", source, scalar, mask, **kw
        )
    )
    vshrs = staticmethod(
        lambda source, scalar, mask, **kw: _VMINamespace._deprecated_vec_scalar(
            "vshrs", source, scalar, mask, **kw
        )
    )

    @staticmethod
    def vcmp(lhs, rhs, seed, cmp, *, pmode=None, loc=None, ip=None):
        return _call_value(
            "vcmp",
            _type_of(seed),
            _raw(lhs),
            _raw(rhs),
            _raw(seed),
            cmp,
            pmode=pmode,
            loc=loc,
            ip=ip,
        )

    @staticmethod
    def vcmps(source, scalar, seed, cmp, *, pmode=None, loc=None, ip=None):
        context = "pto.vmi.vcmps(...)"
        return _call_value(
            "vcmps",
            _type_of(seed),
            _raw(source),
            _coerce_scalar_like_vmi_element(source, scalar, context=context),
            _raw(seed),
            cmp,
            pmode=pmode,
            loc=loc,
            ip=ip,
        )

    @staticmethod
    def vsel(mask, true_value, false_value, *, pmode=None, loc=None, ip=None):
        return _call_value(
            "vsel",
            _type_of(true_value),
            _raw(mask),
            _raw(true_value),
            _raw(false_value),
            pmode=pmode,
            loc=loc,
            ip=ip,
        )

    @staticmethod
    def vselr(source, index, *, loc=None, ip=None):
        return _call_value(
            "vselr",
            _type_of(source),
            _raw(source),
            _raw(index),
            loc=loc,
            ip=ip,
        )

    @staticmethod
    def vbrc(value, *, size, group=None, loc=None, ip=None):
        context = "pto.vmi.vbrc(...)"
        result_type = _derive_vbrc_result_type(value, size, context=context)
        raw_value = _raw(value)
        if group is not None and (not hasattr(raw_value, "type") or not _is_vmi_vreg_type(raw_value.type)):
            raise TypeError(f"{context} with group=... requires a VMI vector input")
        if group is not None:
            if isinstance(group, bool) or not isinstance(group, int):
                raise TypeError(f"{context} requires group to be a positive Python integer")
            if group <= 0:
                raise ValueError(f"{context} requires group to be positive, got {group!r}")
            if not hasattr(raw_value, "type") or not _is_vmi_vreg_type(raw_value.type):
                raise TypeError(f"{context} with group=... requires a VMI vector input")
            value_lanes = _vmi_vreg_element_count(raw_value.type, context=context)
            if value_lanes != group:
                raise ValueError(
                    f"{context} with group=... requires the input lane count to match group; "
                    f"got {value_lanes} lanes for group={group}"
                )
        if not hasattr(raw_value, "type") or not _is_vmi_vreg_type(raw_value.type):
            raw_value = coerce_scalar_to_type(
                value,
                _vmi_element_type(result_type, context=context),
                context="pto.vmi.vbrc(value)",
            )
        return _call_value("vbrc", result_type, raw_value, group=group, loc=loc, ip=ip)

    vcadd = staticmethod(lambda source, mask, *, group=None, pmode=None, reassoc=_UNSPECIFIED, loc=None, ip=None: _emit_reduce("vcadd", source, mask, group=group, pmode=pmode, reassoc=reassoc, loc=loc, ip=ip))
    vcmax = staticmethod(lambda source, mask, *, group=None, pmode=None, loc=None, ip=None: _emit_reduce("vcmax", source, mask, group=group, pmode=pmode, loc=loc, ip=ip))
    vcmin = staticmethod(lambda source, mask, *, group=None, pmode=None, loc=None, ip=None: _emit_reduce("vcmin", source, mask, group=group, pmode=pmode, loc=loc, ip=ip))

    @staticmethod
    def vcvt(
        source,
        to_dtype=None,
        mask=None,
        *,
        rounding=None,
        saturate=None,
        pmode=None,
        loc=None,
        ip=None,
    ):
        if mask is not None:
            raise _unsupported_vmi_feature_error("pto.vmi.vcvt", "masked form")
        result_type = _derive_vcvt_result_type(source, to_dtype, context="pto.vmi.vcvt(...)")
        if rounding is not None:
            rounding = _normalize_vmi_vcvt_rounding(
                rounding,
                context="pto.vmi.vcvt(..., rounding=...)",
            )
        if saturate is None:
            # The VMI verifier requires explicit "SAT" or "NOSAT" for
            # narrowing and fp-to-int directions.  Default to "SAT" when
            # the user does not specify.
            src_bits = _type_bit_width(
                _as_vmi_vreg_type(_type_of(source), context="pto.vmi.vcvt(...)").element_type,
                context="pto.vmi.vcvt(...)",
            )
            dst_bits = _type_bit_width(
                result_type.element_type,
                context="pto.vmi.vcvt(...)",
            )
            src_is_fp = _is_vmi_float_element_type(
                _as_vmi_vreg_type(_type_of(source), context="pto.vmi.vcvt(...)").element_type
            )
            dst_is_fp = _is_vmi_float_element_type(result_type.element_type)
            if src_bits > dst_bits or (src_is_fp and not dst_is_fp):
                saturate = "SAT"
        return _call_value(
            "vcvt",
            result_type,
            _raw(source),
            rounding=rounding,
            saturate=saturate,
            pmode=pmode,
            loc=loc,
            ip=ip,
        )

    @staticmethod
    def vinterpret_cast(source, to_dtype=None, *, loc=None, ip=None):
        return _call_value(
            "vinterpret_cast",
            _derive_vinterpret_cast_result_type(
                source,
                to_dtype,
                context="pto.vmi.vinterpret_cast(...)",
            ),
            _raw(source),
            loc=loc,
            ip=ip,
        )

    @staticmethod
    def vexpdif(x, max_value, mask, *, pmode=None, loc=None, ip=None):
        return _call_value(
            "vexpdif",
            _type_of(max_value),
            _raw(x),
            _raw(max_value),
            _required_mask(mask, context="pto.vmi.vexpdif(...)"),
            pmode=pmode,
            loc=loc,
            ip=ip,
        )

    @staticmethod
    def vaxpy(x, acc, alpha, mask, *, pmode=None, loc=None, ip=None):
        context = "pto.vmi.vaxpy(...)"
        return _call_value(
            "vaxpy",
            _type_of(acc),
            _raw(x),
            _raw(acc),
            _coerce_scalar_like_vmi_element(x, alpha, context=context),
            _required_mask(mask, context=context),
            pmode=pmode,
            loc=loc,
            ip=ip,
        )

    @staticmethod
    def vlrelu(x, slope, mask, *, pmode=None, loc=None, ip=None):
        return _call_value(
            "vlrelu",
            _type_of(x),
            _raw(x),
            _coerce_scalar_like_vmi_element(x, slope, context="pto.vmi.vlrelu(...)"),
            _required_mask(mask, context="pto.vmi.vlrelu(...)"),
            pmode=pmode,
            loc=loc,
            ip=ip,
        )

    @staticmethod
    def vprelu(x, alpha, mask, *, pmode=None, loc=None, ip=None):
        return _call_value(
            "vprelu",
            _type_of(x),
            _raw(x),
            _raw(alpha),
            _required_mask(mask, context="pto.vmi.vprelu(...)"),
            pmode=pmode,
            loc=loc,
            ip=ip,
        )

    @staticmethod
    def vmull(a, b, mask, *, pmode=None, loc=None, ip=None):
        result_type = _type_of(a)
        return _call_value(
            "vmull",
            *_derive_vmull_result_types(a, b, context="pto.vmi.vmull(...)"),
            _raw(a),
            _raw(b),
            _required_mask(mask, context="pto.vmi.vmull(...)"),
            pmode=pmode,
            loc=loc,
            ip=ip,
        )

    @staticmethod
    def vmula(acc, lhs, rhs, mask, *, pmode=None, loc=None, ip=None):
        return _call_value(
            "vmula",
            _type_of(acc),
            _raw(acc),
            _raw(lhs),
            _raw(rhs),
            _required_variadic_mask(mask, context="pto.vmi.vmula(...)"),
            pmode=pmode,
            loc=loc,
            ip=ip,
        )

    @staticmethod
    def vdhist(acc, source, mask, *, loc=None, ip=None):
        return _call_value(
            "vdhist",
            _derive_hist_result_type(acc, context="pto.vmi.vdhist(...)"),
            _raw(acc),
            _raw(source),
            _required_mask(mask, context="pto.vmi.vdhist(...)"),
            loc=loc,
            ip=ip,
        )

    @staticmethod
    def vchist(acc, source, mask, *, loc=None, ip=None):
        return _call_value(
            "vchist",
            _derive_hist_result_type(acc, context="pto.vmi.vchist(...)"),
            _raw(acc),
            _raw(source),
            _required_mask(mask, context="pto.vmi.vchist(...)"),
            loc=loc,
            ip=ip,
        )

    @staticmethod
    def vgather(source, offsets, mask, *, pmode=None, loc=None, ip=None):
        return _call_value(
            "vgather",
            _derive_vgather_result_type(source, offsets, context="pto.vmi.vgather(...)"),
            _raw(source),
            _raw(offsets),
            _required_mask(mask, context="pto.vmi.vgather(...)"),
            pmode=pmode,
            loc=loc,
            ip=ip,
        )

    @staticmethod
    def vgatherb(source, offsets, mask, *, pmode=None, loc=None, ip=None):
        return _call_value(
            "vgatherb",
            _derive_vgatherb_result_type(source, mask, context="pto.vmi.vgatherb(...)"),
            _raw(source),
            _raw(offsets),
            _required_mask(mask, context="pto.vmi.vgatherb(...)"),
            pmode=pmode,
            loc=loc,
            ip=ip,
        )

    @staticmethod
    def vscatter(value, destination, offsets, mask, *, pmode=None, loc=None, ip=None):
        return _generated("vscatter")(
            _raw(value),
            _raw(destination),
            _raw(offsets),
            _required_mask(mask, context="pto.vmi.vscatter(...)"),
            pmode=pmode,
            loc=loc,
            ip=ip,
        )

    @staticmethod
    def create_mask(
        active_lanes,
        *,
        size,
        group=None,
        loc=None,
        ip=None,
    ):
        context = "pto.vmi.create_mask(...)"
        result_type = _resolve_vmi_mask_type(size, context=context)
        if group is None:
            return _call_value("create_mask", result_type, _coerce_index_value(active_lanes), loc=loc, ip=ip)
        if isinstance(group, bool) or not isinstance(group, int):
            raise TypeError(f"{context} requires group to be a positive Python integer")
        if group <= 0:
            raise ValueError(f"{context} requires group to be positive, got {group!r}")
        if size % group != 0:
            raise ValueError(f"{context} requires size to be divisible by group; got size={size!r}, group={group!r}")
        group_size = size // group
        active_lanes_const = _try_get_constant_index(active_lanes)
        if active_lanes_const is not None and active_lanes_const > group_size:
            raise ValueError(
                f"{context} requires active_lanes to be <= the inferred group_size; "
                f"got active_lanes={active_lanes_const!r}, group_size={group_size!r}"
            )
        return _call_value(
            "create_group_mask",
            result_type,
            _coerce_index_value(active_lanes),
            group,
            group_size,
            loc=loc,
            ip=ip,
        )

    @staticmethod
    def vintlv(lhs, rhs, mask, *, pmode=None, loc=None, ip=None):
        return _call_value(
            "vintlv",
            _type_of(lhs),
            _type_of(rhs),
            _raw(lhs),
            _raw(rhs),
            _required_mask(mask, context="pto.vmi.vintlv(...)"),
            pmode=pmode,
            loc=loc,
            ip=ip,
        )

    @staticmethod
    def vdintlv(lhs, rhs, mask, *, pmode=None, loc=None, ip=None):
        return _call_value(
            "vdintlv",
            _type_of(lhs),
            _type_of(rhs),
            _raw(lhs),
            _raw(rhs),
            _required_mask(mask, context="pto.vmi.vdintlv(...)"),
            pmode=pmode,
            loc=loc,
            ip=ip,
        )


def _is_vmi_vreg_type(type_obj) -> bool:
    vreg_type_cls = getattr(_pto, "VMIVRegType", None)
    if vreg_type_cls is None:
        return False
    try:
        return vreg_type_cls.isinstance(type_obj)
    except Exception:
        return False


vmi = _VMINamespace()

__all__ = ["vmi"]
