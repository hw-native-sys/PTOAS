# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Public PTODSL namespace for formal VMI APIs."""

from __future__ import annotations

from collections.abc import Sequence

from ptoas.mlir.dialects import pto as _pto
from ptoas.mlir.ir import (
    BF16Type,
    F16Type,
    F32Type,
    Float8E4M3FNType,
    Float8E5M2Type,
    IndexType,
    IntegerType,
    MemRefType,
    UnitAttr,
)

from ._scalar_coercion import coerce_scalar_to_type
from ._diagnostics import deprecated
from ._surface_values import _coerce_index_value, _try_get_constant_index, unwrap_surface_value, wrap_surface_value
from ._types import (
    VMI_LANE_COUNTS,
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
    if _isinstance_pto_type(type_obj, "BF16x2Type"):
        return 32
    if any(
        _isinstance_pto_type(type_obj, type_name)
        for type_name in ("F4E1M2x2Type", "F4E2M1x2Type")
    ):
        return 8
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
    ) or any(
        _isinstance_pto_type(type_obj, type_name)
        for type_name in ("BF16x2Type", "F4E1M2x2Type", "F4E2M1x2Type")
    )


def _isinstance_pto_type(type_obj, type_name: str) -> bool:
    type_cls = getattr(_pto, type_name, None)
    if type_cls is None:
        return False
    try:
        return type_cls.isinstance(type_obj)
    except Exception:
        return False


def _is_bf16x2_type(type_obj) -> bool:
    return _isinstance_pto_type(type_obj, "BF16x2Type")


def _is_f4x2_type(type_obj) -> bool:
    return any(
        _isinstance_pto_type(type_obj, type_name)
        for type_name in ("F4E1M2x2Type", "F4E2M1x2Type")
    )


def _validate_vmi_vcvt_bf16x2_pair(source_type, result_type, *, context: str) -> bool:
    is_supported_pair = (
        _is_bf16x2_type(source_type) and _is_f4x2_type(result_type)
    ) or (
        _is_f4x2_type(source_type) and _is_bf16x2_type(result_type)
    )
    if is_supported_pair:
        return True
    if _is_bf16x2_type(source_type) or _is_bf16x2_type(result_type):
        raise TypeError(
            f"{context} supports bf16x2 only for bf16x2 <-> "
            f"f4E1M2x2/f4E2M1x2 conversion; got {source_type} -> {result_type}"
        )
    return False


def _normalize_vmi_vcvt_rounding(mode, *, context: str, allowed=None):
    token = mode
    if not isinstance(token, str):
        token = str(token)
        if "." in token:
            token = token.rsplit(".", 1)[-1]
    normalized = token.strip().upper()
    allowed_modes = set(allowed or {"R", "A", "H", "Z"})
    if normalized not in allowed_modes:
        expected = ", ".join(sorted(allowed_modes))
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


def _check_vmi_lane_count(lanes: int, *, context: str) -> None:
    if lanes not in VMI_LANE_COUNTS:
        raise ValueError(
            f"{context} requires lanes to be one of 1, 2, 4, 8, 64, 128, 256; "
            f"got {lanes}"
        )


def _derive_vinterpret_cast_result_type(source, to_dtype, *, context: str):
    if to_dtype is None:
        raise TypeError(f"{context} requires to_dtype")
    source_type = _as_vmi_vreg_type(_type_of(source), context=context)
    source_lanes = source_type.element_count
    source_elem_type = source_type.element_type
    target_elem_type = _ensure_tensor_storage_dtype(to_dtype, context=context)
    source_bits = _type_bit_width(source_elem_type, context=context)
    target_bits = _type_bit_width(target_elem_type, context=context)
    total_bits = source_type.element_count * source_bits
    if total_bits % target_bits != 0:
        raise TypeError(
            f"{context} requires the source bit count to be divisible by the "
            f"target element width; got {source_type.element_count}x"
            f"{source_elem_type} -> {target_elem_type}"
        )
    target_lanes = total_bits // target_bits
    _check_vmi_lane_count(target_lanes, context=context)
    if target_lanes != source_lanes and source_type.layout is not None:
        raise TypeError(
            f"{context} cannot preserve the source layout across a lane-count "
            f"change ({source_type} -> {target_lanes}x{target_elem_type}); "
            "layouts are tied to the source lane count"
        )
    layout = source_type.layout if target_lanes == source_lanes else None
    return _pto.VMIVRegType.get(
        target_lanes,
        target_elem_type,
        layout=layout,
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
    elem_type = raw_base.type
    # Dynamic loop indices (TileLang T.serial / scf.for IVs) are MLIR index.
    # VCI requires an integer/float sreg element type; Ascend uses i32.
    # Coerce index → signless i32 so pto.vmi.vci(dynamic_base) lowers to
    # ``VCI Vd, Sn`` instead of failing ODS (index) or verify (i64).
    if IndexType.isinstance(elem_type):
        elem_type = IntegerType.get_signless(32)
    return _pto.VMIVRegType.get(size, elem_type)


def _physical_lanes_per_part(elem_type, *, context: str) -> int | None:
    """A5 256B physical VL lane count for VCI element types, else None."""
    if IntegerType.isinstance(elem_type):
        width = IntegerType(elem_type).width
        if width == 8:
            return 256
        if width == 16:
            return 128
        if width == 32:
            return 64
        return None
    # Float: f16→128, f32→64 (match getDataLanesPerPart).
    name = str(elem_type)
    if "f16" in name or "bf16" in name:
        return 128
    if "f32" in name:
        return 64
    return None


def _check_vci_group_tiles_phys_vl(elem_type, size, group, *, context: str) -> None:
    # One group is exactly the ordinary continuous iota, including tails that
    # do not tile physical VL (for example i32 size=100).
    if group == 1:
        return
    group_size = size // group
    phys = _physical_lanes_per_part(elem_type, context=context)
    if phys is None:
        return
    if group_size % phys != 0 and phys % group_size != 0:
        raise ValueError(
            f"{context} requires group_size ({group_size}) to divide or be a "
            f"multiple of physical lanes per part ({phys}) for element type "
            f"{elem_type}"
        )


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


def _derive_add_carry_result_types(lhs, rhs, mask, *, carry_in=None, context: str):
    lhs_type = _as_vmi_vreg_type(_type_of(lhs), context=context)
    rhs_type = _as_vmi_vreg_type(_type_of(rhs), context=context)
    if lhs_type != rhs_type:
        raise TypeError(f"{context} requires lhs and rhs to have identical VMI vreg types")
    element_type = lhs_type.element_type
    if not IntegerType.isinstance(element_type) or IntegerType(element_type).width != 32:
        raise TypeError(f"{context} requires 32-bit integer vectors")

    mask_type = _as_vmi_mask_type(_type_of(mask), context=context)
    if _vmi_mask_element_count(mask_type, context=context) != lhs_type.element_count:
        raise TypeError(f"{context} requires the mask lane count to match the data vectors")
    if carry_in is not None:
        carry_in_type = _as_vmi_mask_type(_type_of(carry_in), context=context)
        if carry_in_type != mask_type:
            raise TypeError(f"{context} requires carry_in and mask to have identical VMI mask types")
    return lhs_type, mask_type


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
        if result_lanes not in (1, 2, 4, 8):
            raise ValueError(
                f"{context} requires group to be one of 1, 2, 4, 8; "
                f"got {group!r}"
            )
        if source_type.element_count % result_lanes != 0:
            raise ValueError(
                f"{context} requires group to evenly divide the source lane "
                f"count; got group={result_lanes}, lanes={source_type.element_count}"
            )
    return _pto.VMIVRegType.get(result_lanes, source_type.element_type)


def _coerce_scalar_like_vmi_element(vector_value, scalar_value, *, context: str):
    elem_type = _vmi_element_type(_type_of(vector_value), context=context)
    return coerce_scalar_to_type(scalar_value, elem_type, context=context)


def _variadic_mask(mask):
    if mask is None:
        return []
    return _raw_sequence(mask)


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


def _emit_binary(op_name: str, lhs, rhs, mask=None, *, pmode=None, loc=None, ip=None):
    return _call_value(
        op_name,
        _type_of(lhs),
        _raw(lhs),
        _raw(rhs),
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
    scalar_value = (
        coerce_scalar_to_type(scalar, IntegerType.get_signless(16), context=context)
        if op_name in {"vshls", "vshrs"}
        else _coerce_scalar_like_vmi_element(source, scalar, context=context)
    )
    return _call_value(
        op_name,
        _type_of(source),
        _raw(source),
        scalar_value,
        _required_mask(mask, context=context),
        pmode=pmode,
        loc=loc,
        ip=ip,
    )


def _emit_binary_or_vec_scalar(
    binary_op_name: str,
    vec_scalar_op_name: str,
    lhs,
    rhs,
    mask=None,
    *,
    commutative=False,
    **kw,
):
    """Dispatch a VMI binary family from the operand kinds."""
    lhs_type = getattr(_raw(lhs), "type", None)
    rhs_type = getattr(_raw(rhs), "type", None)
    if rhs_type is not None and _is_vmi_vreg_type(rhs_type):
        if commutative and (lhs_type is None or not _is_vmi_vreg_type(lhs_type)):
            return _emit_vec_scalar(vec_scalar_op_name, rhs, lhs, mask, **kw)
        return _emit_binary(binary_op_name, lhs, rhs, mask, **kw)
    return _emit_vec_scalar(vec_scalar_op_name, lhs, rhs, mask, **kw)


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
    kwargs = {"group": group, "pmode": pmode, "loc": loc, "ip": ip}
    if reassoc is not _UNSPECIFIED:
        kwargs["reassoc"] = UnitAttr.get()
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
        return _generated("vstore")(
            _raw_sequence(values),
            _raw(destination),
            _coerce_index_value(offset),
            _variadic_mask(mask),
            stride=None if stride is None else _coerce_index_value(stride),
            block_stride=_i16_value(block_stride, context="pto.vmi.vstore(block_stride)"),
            dist_mode=dist_mode,
            group=group,
            pmode=pmode,
            loc=loc,
            ip=ip,
        )

    @staticmethod
    def vsstb(value, destination, offset, block_stride, mask, *, pmode=None, loc=None, ip=None):
        context = "pto.vmi.vsstb(...)"
        return _generated("vsstb")(
            _raw(value), _raw(destination), _coerce_index_value(offset),
            _i16_value(block_stride, context=f"{context} block_stride"),
            _required_mask(mask, context=context), pmode=pmode, loc=loc, ip=ip,
        )

    @staticmethod
    def vci(base, *, size, order=None, group=None, loc=None, ip=None):
        context = "pto.vmi.vci(...)"
        if group is not None:
            if isinstance(group, bool) or not isinstance(group, int):
                raise TypeError(f"{context} requires group to be a positive Python integer")
            if group <= 0:
                raise ValueError(f"{context} requires group to be positive, got {group!r}")
            if size % group != 0:
                raise ValueError(
                    f"{context} requires size divisible by group; got size={size!r}, group={group!r}"
                )
        result_type = _derive_vci_result_type(base, size, context=context)
        if group is not None:
            _check_vci_group_tiles_phys_vl(
                result_type.element_type, size, group, context=context
            )
        base = coerce_scalar_to_type(
            base,
            _vmi_element_type(result_type, context=context),
            context="pto.vmi.vci(base)",
        )
        return _call_value(
            "vci", result_type, base, order=order, group=group, loc=loc, ip=ip
        )

    @staticmethod
    def vadd(lhs, rhs, mask=None, **kw):
        """Emit VMI vector addition, selecting vector or scalar form by type."""
        return _emit_binary_or_vec_scalar("vadd", "vadds", lhs, rhs, mask, commutative=True, **kw)

    @staticmethod
    def vaddc(lhs, rhs, mask, *, loc=None, ip=None):
        """Emit a 32-bit integer add with per-lane carry output."""
        context = "pto.vmi.vaddc(...)"
        mask_value = _required_mask(mask, context=context)
        result_type, carry_type = _derive_add_carry_result_types(
            lhs, rhs, mask_value, context=context
        )
        return _call_value(
            "vaddc",
            result_type,
            carry_type,
            _raw(lhs),
            _raw(rhs),
            mask_value,
            loc=loc,
            ip=ip,
        )

    @staticmethod
    def vaddcs(lhs, rhs, carry_in, mask, *, loc=None, ip=None):
        """Emit a 32-bit integer add with carry input and carry output."""
        context = "pto.vmi.vaddcs(...)"
        mask_value = _required_mask(mask, context=context)
        result_type, carry_type = _derive_add_carry_result_types(
            lhs, rhs, mask_value, carry_in=carry_in, context=context
        )
        return _call_value(
            "vaddcs",
            result_type,
            carry_type,
            _raw(lhs),
            _raw(rhs),
            _raw(carry_in),
            mask_value,
            loc=loc,
            ip=ip,
        )

    vsub = staticmethod(lambda lhs, rhs, mask=None, **kw: _emit_binary("vsub", lhs, rhs, mask, **kw))

    @staticmethod
    def vmul(lhs, rhs, mask=None, **kw):
        """Emit VMI vector multiplication, selecting vector or scalar form by type."""
        return _emit_binary_or_vec_scalar("vmul", "vmuls", lhs, rhs, mask, commutative=True, **kw)

    vdiv = staticmethod(lambda lhs, rhs, mask=None, **kw: _emit_binary("vdiv", lhs, rhs, mask, **kw))

    @staticmethod
    def vmax(lhs, rhs, mask=None, **kw):
        """Emit VMI maximum, selecting vector or scalar form by type."""
        return _emit_binary_or_vec_scalar("vmax", "vmaxs", lhs, rhs, mask, commutative=True, **kw)

    @staticmethod
    def vmin(lhs, rhs, mask=None, **kw):
        """Emit VMI minimum, selecting vector or scalar form by type."""
        return _emit_binary_or_vec_scalar("vmin", "vmins", lhs, rhs, mask, commutative=True, **kw)

    vand = staticmethod(lambda lhs, rhs, mask=None, **kw: _emit_binary("vand", lhs, rhs, mask, **kw))
    vor = staticmethod(lambda lhs, rhs, mask=None, **kw: _emit_binary("vor", lhs, rhs, mask, **kw))
    vxor = staticmethod(lambda lhs, rhs, mask=None, **kw: _emit_binary("vxor", lhs, rhs, mask, **kw))

    @staticmethod
    def vshl(lhs, rhs, mask=None, **kw):
        """Emit VMI shift-left, selecting vector or scalar form by type."""
        return _emit_binary_or_vec_scalar("vshl", "vshls", lhs, rhs, mask, **kw)

    @staticmethod
    def vshr(lhs, rhs, mask=None, **kw):
        """Emit VMI shift-right, selecting vector or scalar form by type."""
        return _emit_binary_or_vec_scalar("vshr", "vshrs", lhs, rhs, mask, **kw)

    vabs = staticmethod(lambda source, mask=None, **kw: _emit_unary("vabs", source, mask, **kw))
    vneg = staticmethod(lambda source, mask=None, **kw: _emit_unary("vneg", source, mask, **kw))
    vrelu = staticmethod(lambda source, mask=None, **kw: _emit_unary("vrelu", source, mask, **kw))
    vexp = staticmethod(lambda source, mask=None, **kw: _emit_unary("vexp", source, mask, **kw))
    vln = staticmethod(lambda source, mask=None, **kw: _emit_unary("vln", source, mask, **kw))
    vsqrt = staticmethod(lambda source, mask=None, **kw: _emit_unary("vsqrt", source, mask, **kw))
    vnot = staticmethod(lambda source, mask=None, **kw: _emit_unary("vnot", source, mask, **kw))

    @staticmethod
    @deprecated("use pto.vmi.vadd(vector, scalar, mask) instead")
    def vadds(source, scalar, mask, **kw):
        """Deprecated VMI vector-scalar add compatibility entry point."""
        return _emit_vec_scalar("vadds", source, scalar, mask, **kw)

    @staticmethod
    @deprecated("use pto.vmi.vmul(vector, scalar, mask) instead")
    def vmuls(source, scalar, mask, **kw):
        return _emit_vec_scalar("vmuls", source, scalar, mask, **kw)

    @staticmethod
    @deprecated("use pto.vmi.vmax(vector, scalar, mask) instead")
    def vmaxs(source, scalar, mask, **kw):
        return _emit_vec_scalar("vmaxs", source, scalar, mask, **kw)

    @staticmethod
    @deprecated("use pto.vmi.vmin(vector, scalar, mask) instead")
    def vmins(source, scalar, mask, **kw):
        return _emit_vec_scalar("vmins", source, scalar, mask, **kw)

    @staticmethod
    @deprecated("use pto.vmi.vshl(vector, scalar, mask) instead")
    def vshls(source, scalar, mask, **kw):
        return _emit_vec_scalar("vshls", source, scalar, mask, **kw)

    @staticmethod
    @deprecated("use pto.vmi.vshr(vector, scalar, mask) instead")
    def vshrs(source, scalar, mask, **kw):
        return _emit_vec_scalar("vshrs", source, scalar, mask, **kw)

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

    vcadd = staticmethod(lambda source, mask, *, group=1, pmode=None, reassoc=_UNSPECIFIED, loc=None, ip=None: _emit_reduce("vcadd", source, mask, group=1 if group is None else group, pmode=pmode, reassoc=reassoc, loc=loc, ip=ip))
    vcmax = staticmethod(lambda source, mask, *, group=1, pmode=None, loc=None, ip=None: _emit_reduce("vcmax", source, mask, group=1 if group is None else group, pmode=pmode, loc=loc, ip=ip))
    vcmin = staticmethod(lambda source, mask, *, group=1, pmode=None, loc=None, ip=None: _emit_reduce("vcmin", source, mask, group=1 if group is None else group, pmode=pmode, loc=loc, ip=ip))

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
        source_type = _as_vmi_vreg_type(
            _type_of(source),
            context="pto.vmi.vcvt(...)",
        )
        is_bf16x2_pair = _validate_vmi_vcvt_bf16x2_pair(
            source_type.element_type,
            result_type.element_type,
            context="pto.vmi.vcvt(...)",
        )
        is_bf16x2_to_f4x2 = is_bf16x2_pair and _is_bf16x2_type(
            source_type.element_type
        )
        is_f4x2_to_bf16x2 = is_bf16x2_pair and _is_f4x2_type(
            source_type.element_type
        )
        if rounding is not None:
            if is_f4x2_to_bf16x2:
                raise ValueError(
                    "pto.vmi.vcvt(...) does not support rounding for "
                    "f4E1M2x2/f4E2M1x2 -> bf16x2 conversion"
                )
            rounding = _normalize_vmi_vcvt_rounding(
                rounding,
                context="pto.vmi.vcvt(..., rounding=...)",
                allowed={"R", "A", "F", "Z", "C"} if is_bf16x2_to_f4x2 else None,
            )
        elif is_bf16x2_to_f4x2:
            rounding = "R"
        if is_bf16x2_to_f4x2 and saturate is not None:
            raise ValueError(
                "pto.vmi.vcvt(...) does not support saturate for bf16x2 -> "
                "f4E1M2x2/f4E2M1x2 conversion"
            )
        if is_f4x2_to_bf16x2 and saturate is not None:
            raise ValueError(
                "pto.vmi.vcvt(...) does not support saturate for "
                "f4E1M2x2/f4E2M1x2 -> bf16x2 conversion"
            )
        if saturate is None:
            # The VMI verifier requires explicit "SAT" or "NOSAT" for
            # narrowing and fp-to-int directions.  Default to "SAT" when
            # the user does not specify.
            src_bits = _type_bit_width(source_type.element_type, context="pto.vmi.vcvt(...)")
            dst_bits = _type_bit_width(
                result_type.element_type,
                context="pto.vmi.vcvt(...)",
            )
            src_is_fp = _is_vmi_float_element_type(source_type.element_type)
            dst_is_fp = _is_vmi_float_element_type(result_type.element_type)
            if not is_bf16x2_pair and (src_bits > dst_bits or (src_is_fp and not dst_is_fp)):
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
        context = "pto.vmi.vexpdif(...)"
        x_type = _as_vmi_vreg_type(_type_of(x), context=context)
        max_type = _as_vmi_vreg_type(_type_of(max_value), context=context)
        if x_type != max_type:
            raise TypeError(
                f"{context} requires x and max_value to have identical VMI vreg types"
            )
        if not (
            F16Type.isinstance(x_type.element_type)
            or F32Type.isinstance(x_type.element_type)
        ):
            raise TypeError(f"{context} requires f16 or f32 input vectors")
        mask_type = _as_vmi_mask_type(_type_of(mask), context=context)
        if (
            _vmi_mask_element_count(mask_type, context=context)
            != x_type.element_count
        ):
            raise TypeError(f"{context} requires mask and input lane counts to match")
        result_type = _pto.VMIVRegType.get(
            x_type.element_count,
            F32Type.get(),
            layout=x_type.layout if F32Type.isinstance(x_type.element_type) else None,
        )
        return _call_value(
            "vexpdif",
            result_type,
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
