# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""SIMT ops: index/dimension queries, votes/shuffles, atomics, scalar
math, synchronization, and pipe/buffer management."""

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
    _coerce_i32_operand,
    _coerce_index,
    _event_attr,
    _l1_cache_attr,
    _ld_l2_cache_attr,
    _optional_signedness_attr,
    _pipe_attr,
    _pointer_element_type,
    _required_signedness_attr,
    _rounding_attr,
    _same_type_binary,
    _same_type_ternary,
    _same_type_unary,
    _saturation_attr,
    _st_l2_cache_attr,
    _validate_convert_signedness,
    _validate_integer_signedness_only,
    _validate_redux_signedness,
    _validate_static_buf_id,
    _validate_static_event_id,
    _validate_static_event_id_range,
    _validate_sync_pipe,
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
    """``pto.stg`` – GM store with cache controls."""
    buffer_value, index_value = resolve_address_access(ptr_or_ref, offset)
    elem_type = _pointer_element_type(buffer_value, context="stg(value, ptr, offset)")
    raw_value = unwrap_surface_value(value)
    raw_value_type = getattr(raw_value, "type", None)
    if raw_value_type == elem_type:
        stored_value = raw_value
    elif VectorType.isinstance(elem_type):
        raise TypeError(
            f"stg(value, ...) vector value type must match destination element type: "
            f"got {raw_value_type if raw_value_type is not None else type(raw_value).__name__}, "
            f"expected {elem_type}"
        )
    else:
        stored_value = coerce_scalar_to_type(value, elem_type, context="stg(value, ...)")
    _pto.PTOStgOp(
        buffer_value,
        index_value,
        stored_value,
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
    return _atomic_binary(
        _pto.AtomicExchOp, ptr, value, l2cache=l2cache, signedness=signedness,
        context="atomic_exch(ptr, value)")


def atomic_add(ptr, value, *, l2cache="nmfv", signedness=None):
    """``pto.atomic_add`` – SIMT scalar atomic add."""
    return _atomic_binary(
        _pto.AtomicAddOp, ptr, value, l2cache=l2cache, signedness=signedness,
        context="atomic_add(ptr, value)")


def atomic_sub(ptr, value, *, l2cache="nmfv", signedness=None):
    """``pto.atomic_sub`` – SIMT scalar atomic subtract."""
    return _atomic_binary(
        _pto.AtomicSubOp, ptr, value, l2cache=l2cache, signedness=signedness,
        context="atomic_sub(ptr, value)")


def atomic_min(ptr, value, *, l2cache="nmfv", signedness=None):
    """``pto.atomic_min`` – SIMT scalar atomic min."""
    return _atomic_binary(
        _pto.AtomicMinOp, ptr, value, l2cache=l2cache, signedness=signedness,
        context="atomic_min(ptr, value)")


def atomic_max(ptr, value, *, l2cache="nmfv", signedness=None):
    """``pto.atomic_max`` – SIMT scalar atomic max."""
    return _atomic_binary(
        _pto.AtomicMaxOp, ptr, value, l2cache=l2cache, signedness=signedness,
        context="atomic_max(ptr, value)")


def atomic_and(ptr, value, *, l2cache="nmfv", signedness=None):
    """``pto.atomic_and`` – SIMT scalar atomic bitwise and."""
    return _atomic_binary(
        _pto.AtomicAndOp, ptr, value, l2cache=l2cache, signedness=signedness,
        context="atomic_and(ptr, value)")


def atomic_or(ptr, value, *, l2cache="nmfv", signedness=None):
    """``pto.atomic_or`` – SIMT scalar atomic bitwise or."""
    return _atomic_binary(
        _pto.AtomicOrOp, ptr, value, l2cache=l2cache, signedness=signedness,
        context="atomic_or(ptr, value)")


def atomic_xor(ptr, value, *, l2cache="nmfv", signedness=None):
    """``pto.atomic_xor`` – SIMT scalar atomic bitwise xor."""
    return _atomic_binary(
        _pto.AtomicXorOp, ptr, value, l2cache=l2cache, signedness=signedness,
        context="atomic_xor(ptr, value)")


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


def sin(value):
    """``pto.sin`` – A5 SIMT floating sine software-library hook."""
    return _same_type_unary(_pto.SinOp, value)


def cos(value):
    """``pto.cos`` – A5 SIMT floating cosine software-library hook."""
    return _same_type_unary(_pto.CosOp, value)


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


def trap():
    """``pto.trap`` – unconditionally terminate device-side execution."""
    _pto.TrapOp()


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


def set_cross_block(pipe, event_id):
    """``pto.set_cross_block(pipe, event_id)`` – emits ``pto.set_cross_block``."""
    _validate_sync_pipe(
        pipe,
        context="set_cross_block(pipe, event_id)",
        allowed=("PIPE_FIX", "PIPE_MTE1", "PIPE_MTE2", "PIPE_MTE3", "PIPE_V"),
    )
    event_operand = _sync_event_id_operand_in_range(
        event_id, context="set_cross_block(..., event_id=...)", lo=0, hi=15
    )
    _pto.set_cross_block(_pipe_attr(pipe), event_operand)


def wait_cross_block(pipe, event_id):
    """``pto.wait_cross_block(pipe, event_id)`` – emits ``pto.wait_cross_block``."""
    _validate_sync_pipe(
        pipe,
        context="wait_cross_block(pipe, event_id)",
        allowed=("PIPE_FIX", "PIPE_MTE1", "PIPE_MTE2", "PIPE_MTE3", "PIPE_V"),
    )
    event_operand = _sync_event_id_operand_in_range(
        event_id, context="wait_cross_block(..., event_id=...)", lo=0, hi=15
    )
    _pto.wait_cross_block(_pipe_attr(pipe), event_operand)


def set_intra_block(pipe, event_id):
    """``pto.set_intra_block(pipe, event_id)`` – emits ``pto.set_intra_block``."""
    _validate_sync_pipe(
        pipe,
        context="set_intra_block(pipe, event_id)",
        allowed=("PIPE_FIX", "PIPE_MTE1", "PIPE_MTE2", "PIPE_MTE3", "PIPE_V"),
    )
    event_operand = _sync_event_id_operand_in_range(
        event_id,
        context="set_intra_block(..., event_id=...)",
        lo=0,
        hi=31,
        meaning="physical event_id",
    )
    _pto.set_intra_block(_pipe_attr(pipe), event_operand)


def wait_intra_block(pipe, event_id):
    """``pto.wait_intra_block(pipe, event_id)`` – emits ``pto.wait_intra_block``."""
    _validate_sync_pipe(
        pipe,
        context="wait_intra_block(pipe, event_id)",
        allowed=("PIPE_FIX", "PIPE_MTE1", "PIPE_MTE2", "PIPE_MTE3", "PIPE_V"),
    )
    event_operand = _sync_event_id_operand_in_range(
        event_id,
        context="wait_intra_block(..., event_id=...)",
        lo=0,
        hi=31,
        meaning="physical event_id",
    )
    _pto.wait_intra_block(_pipe_attr(pipe), event_operand)


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
