# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Predicate/mask ops: plt/pset/pge families, mask logic, interleave."""

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
    _PREDICATE_LOAD_DIST_TOKENS,
    _PREDICATE_STORE_DIST_TOKENS,
    _coerce_i32,
    _coerce_index,
    _coerce_scalar_like_vector_element,
    _element_bytewidth,
    _infer_mask_metadata,
    _infer_vreg_metadata,
    _mask_bits_for_dtype,
    _mask_type_from_bits,
    _normalize_cmp_mode,
    _normalize_mask_pattern,
    _normalize_post_update_mode,
    _normalize_predicate_dist,
    _normalize_predicate_part,
    _plt_op_for_mask_bits,
    _pointer_element_type,
    _pset_op_for_mask_bits,
    _require_same_mask_types,
    _resolve_mask_result_type,
)


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
            None,
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
        None,
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
        None,
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
            None,
            unwrap_surface_value(align_in),
            _coerce_i32(offset, context="vstus(align, offset, value, base)"),
            unwrap_surface_value(value),
            unwrap_surface_value(base),
        ).align_out
    )


# ── Vector math (result type inferred from first operand) ─────────────────────

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
