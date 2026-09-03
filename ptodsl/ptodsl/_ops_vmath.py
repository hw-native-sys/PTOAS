# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Vector math ops: arithmetic, comparison helpers, scalar load/store."""

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
    _coerce_i64,
    _coerce_index,
    _coerce_scalar_like_vector_element,
    _elements_per_vreg,
    _emit_binary_vec_op,
    _emit_shift_vec_op,
    _emit_unary_vec_op,
    _emit_vec_scalar_masked_op,
    _infer_mask_metadata,
    _negate_runtime_scalar,
    _normalize_vcvt_round_mode,
    _normalize_vdup_position_mode,
    _pointer_element_type,
    _reject_low_precision_vreg_operands,
)


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
    return _emit_shift_vec_op(_pto.VshlOp, lhs, rhs, mask)


def vshr(lhs, rhs, mask):
    """``pto.vshr`` – element-wise shift right."""
    return _emit_shift_vec_op(_pto.VshrOp, lhs, rhs, mask)


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

def _mask_granularity_bits(mask_value, *, context: str) -> int:
    mask_bits, _ = _infer_mask_metadata(mask_value, context=context)
    return mask_bits


def _infer_vdup_scalar_result_type(input_value, mask_value, *, context: str):
    scalar_raw = unwrap_surface_value(input_value)
    scalar_type = scalar_raw.type
    mask_bits = _mask_granularity_bits(mask_value, context=context)
    if IntegerType.isinstance(scalar_type):
        scalar_width = IntegerType(scalar_type).width
        if scalar_width != mask_bits:
            raise TypeError(
                f"{context} expects scalar input width {scalar_width} to match mask granularity b{mask_bits}"
            )
        element_type = scalar_type
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
            f"{context} only supports scalar input types i8/i16/i32, si8/si16/si32, "
            f"ui8/ui16/ui32, f16, bf16, and f32; got {scalar_type}"
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
            raise TypeError(
                "vdup(scalar, mask, position=...) does not support position; "
                "position is only valid for vector input")
        raw_input = _coerce_vdup_scalar_input(input_value, mask, context="vdup(scalar, mask)")
        result_type = _infer_vdup_scalar_result_type(raw_input, mask, context="vdup(scalar, mask)")
        result_element_type = _pto.VRegType(result_type).element_type
        raw_input = coerce_scalar_to_type(
            raw_input,
            result_element_type,
            context="vdup(scalar, mask)",
        )
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


def vsqz(inp, mask):
    """``pto.vsqz`` - compact active lanes to the front while preserving order."""
    return _emit_unary_vec_op(_pto.VsqzOp, inp, mask)


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


def chistv2(acc, source, mask, bin_val):
    """``pto.chistv2`` – accumulate 128-bin histogram from b8 source into b16 accumulator."""
    return wrap_surface_value(
        _pto.Chistv2Op(
            unwrap_surface_value(acc).type,
            unwrap_surface_value(acc),
            unwrap_surface_value(source),
            unwrap_surface_value(mask),
            unwrap_surface_value(bin_val),
        ).result
    )


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


def _resolve_l1_bypass_scalar_pointer(ptr_value, *, context: str):
    """Validate the GM integer pointer contract used by ``ld_dev``."""
    raw_ptr = unwrap_surface_value(ptr_value)
    try:
        ptr_type = _pto.PtrType(raw_ptr.type)
    except Exception as exc:
        raise TypeError(f"{context} requires a typed PTO pointer") from exc

    gm_space = _pto.AddressSpaceAttr.get(_pto.AddressSpace.GM)
    if ptr_type.memory_space != gm_space:
        raise TypeError(f"{context} requires a GM pointer, got {raw_ptr.type}")

    elem_type = ptr_type.element_type
    if not IntegerType.isinstance(elem_type) or IntegerType(elem_type).width not in (8, 16, 32, 64):
        raise TypeError(
            f"{context} supports only i8, i16, i32 or i64 pointer elements, "
            f"got {elem_type}"
        )
    return raw_ptr, elem_type


def _validate_scalar_l1_bypass(value, *, context: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{context} expects bypass_l1 to be a bool")
    return value


def load_scalar(ptr_value, offset=0, *, bypass_l1=False):
    """Load one scalar through the scalar pipeline.

    ``bypass_l1=True`` selects the AICore GM device-memory form and emits
    ``pto.ld_dev``.  The bypass form requires a GM pointer with an i8, i16,
    i32, or i64 element type and is valid only in an ordinary AICore entry.
    """
    bypass_l1 = _validate_scalar_l1_bypass(bypass_l1, context="load_scalar(...)")
    if bypass_l1:
        raw_ptr, elem_type = _resolve_l1_bypass_scalar_pointer(
            ptr_value, context="load_scalar(..., bypass_l1=True)"
        )
        return wrap_surface_value(
            _pto.PTOLdDevOp(
                elem_type,
                raw_ptr,
                _coerce_index(offset, context="load_scalar(offset)"),
            ).value
        )

    result_type = _pointer_element_type(ptr_value, context="load_scalar(ptr)")
    return wrap_surface_value(
        _pto.LoadScalarOp(
            _resolve(result_type),
            unwrap_surface_value(ptr_value),
            _coerce_index(offset, context="load_scalar(offset)"),
        ).value
    )


def store_scalar(ptr_value, offset, value, *, bypass_l1=False):
    """Store one scalar through the scalar pipeline.

    ``bypass_l1=True`` selects the AICore GM device-memory form and emits
    ``pto.st_dev``.  The bypass form requires a GM pointer with an i8, i16,
    i32, or i64 element type and is valid only in an ordinary AICore entry.
    """
    bypass_l1 = _validate_scalar_l1_bypass(bypass_l1, context="store_scalar(...)")
    if bypass_l1:
        raw_ptr, elem_type = _resolve_l1_bypass_scalar_pointer(
            ptr_value, context="store_scalar(..., bypass_l1=True)"
        )
        raw_value = coerce_scalar_to_type(
            value,
            elem_type,
            context="store_scalar(..., bypass_l1=True)",
        )
        _pto.PTOStDevOp(
            raw_value,
            raw_ptr,
            _coerce_index(offset, context="store_scalar(offset)"),
        )
        return

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
