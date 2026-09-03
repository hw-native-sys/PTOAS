# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Data-movement ops: MTE transfers, mad, accumulator-store attributes."""

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
    _coerce_i1,
    _coerce_i32,
    _coerce_i64,
    _constant_like,
    _element_bytewidth,
    _enum_attr,
    _explicit_mode_only,
    _membar_attr,
    _mul_bytes,
    _normalize_mte_store_l2_cache,
    _normalize_sat_mode,
    _normalize_token,
    _validate_raw_fill_l1_fill_word_bits,
    _validate_raw_fill_l1_static_scalar,
)



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
            return (
                Attribute.parse("#pto<acc_store_mode nz2dn>"),
                None,
                _coerce_i64(operand, context="acc store layout nz2dn"))
        if normalized == "nz2nz":
            return (
                Attribute.parse("#pto<acc_store_mode nz2nz>"),
                _coerce_i64(operand, context="acc store layout nz2nz"),
                None)
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
        return (
            row_count, valid_cols,
            _mul_bytes(src_row_stride, infer_tile_element_type(tile)),
            physical_cols * _element_bytewidth(infer_tile_element_type(tile)))
    return (
        row_count, valid_cols,
        physical_cols * _element_bytewidth(infer_tile_element_type(tile)),
        _mul_bytes(src_row_stride, infer_tile_element_type(tile)))


def fill_tile(tile, value):
    """Broadcast a scalar into an entire tile."""
    wrapped_tile = wrap_surface_value(tile)
    scalar_value = _constant_like(value, infer_tile_element_type(wrapped_tile))
    _pto.TExpandsOp(scalar_value, unwrap_surface_value(wrapped_tile))

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
    if hasattr(pad_value, "type"):
        # The pad value is encoded as a raw bit pattern into SET.MOV.PAD.VAL,
        # so signedness is irrelevant to the backend. Normalize explicit
        # signed/unsigned integer pads (ui8/si8/...) to the signless
        # counterpart the IR verifier accepts, preserving the bit pattern.
        pad_value = _strip_integer_signedness(unwrap_surface_value(pad_value))
    else:
        pad_value = materialize_scalar_literal(
            pad_value, F32Type.get(), context=f"{context} pad[0]"
        )
    return (
        pad_value,
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


@_explicit_mode_only("pto.set_store_atomic_cfg(...)")
def set_store_atomic_cfg(config):
    """Configure scalar ST atomic mode via ST_ATOMIC_CFG (SU path)."""
    _pto.SetStoreAtomicCfgOp(
        _coerce_i64(config, context="set_store_atomic_cfg config")
    )


def _nullary_atomic_config(op_cls, api_name):
    @_explicit_mode_only(f"pto.{api_name}(...)")
    def _impl():
        op_cls()

    _impl.__name__ = api_name
    _impl.__doc__ = (
        f"``pto.{api_name}`` – configure MTE/FIXP store-atomic CTRL state."
    )
    return _impl


set_atomic_add = _nullary_atomic_config(_pto.SetAtomicAddOp, "set_atomic_add")
set_atomic_max = _nullary_atomic_config(_pto.SetAtomicMaxOp, "set_atomic_max")
set_atomic_min = _nullary_atomic_config(_pto.SetAtomicMinOp, "set_atomic_min")
set_atomic_none = _nullary_atomic_config(_pto.SetAtomicNoneOp, "set_atomic_none")
set_atomic_f32 = _nullary_atomic_config(_pto.SetAtomicF32Op, "set_atomic_f32")
set_atomic_f16 = _nullary_atomic_config(_pto.SetAtomicF16Op, "set_atomic_f16")
set_atomic_bf16 = _nullary_atomic_config(_pto.SetAtomicBF16Op, "set_atomic_bf16")
set_atomic_s32 = _nullary_atomic_config(_pto.SetAtomicS32Op, "set_atomic_s32")
set_atomic_s16 = _nullary_atomic_config(_pto.SetAtomicS16Op, "set_atomic_s16")
set_atomic_s8 = _nullary_atomic_config(_pto.SetAtomicS8Op, "set_atomic_s8")


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


@_explicit_mode_only("pto.raw_fill_l1(...)")
def raw_fill_l1(dst, byte_offset, raw_value, *,
                repeat_times, block_num_32b, dst_gap_32b, fill_word_bits):
    """Fill a strided L1/MAT region with a raw 16-bit or 32-bit pattern."""
    _validate_raw_fill_l1_static_scalar(
        byte_offset, context="raw_fill_l1 byte_offset", alignment=32)
    _validate_raw_fill_l1_static_scalar(
        repeat_times, context="raw_fill_l1 repeat_times", maximum=32767)
    _validate_raw_fill_l1_static_scalar(
        block_num_32b, context="raw_fill_l1 block_num_32b", maximum=32767)
    _validate_raw_fill_l1_static_scalar(
        dst_gap_32b, context="raw_fill_l1 dst_gap_32b", maximum=32767)
    _validate_raw_fill_l1_fill_word_bits(
        fill_word_bits, context="raw_fill_l1 fill_word_bits")
    _pto.RawFillL1Op(
        unwrap_surface_value(dst),
        _coerce_i64(byte_offset, context="raw_fill_l1 byte_offset"),
        _coerce_i32(raw_value, context="raw_fill_l1 raw_value"),
        _coerce_i64(repeat_times, context="raw_fill_l1 repeat_times"),
        _coerce_i64(block_num_32b, context="raw_fill_l1 block_num_32b"),
        _coerce_i64(dst_gap_32b, context="raw_fill_l1 dst_gap_32b"),
        IntegerAttr.get(IntegerType.get_signless(64), fill_word_bits),
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


def _normalize_cube_transpose(value, *, context: str) -> bool:
    """Normalize the legacy BoolAttr-compatible 0/1 spelling."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    raise TypeError(f"{context} transpose expects bool or integer 0/1")


@_explicit_mode_only("pto.mte_l1_l0a(...)")
def mte_l1_l0a(
    source,
    destination,
    m=None,
    k=None,
    *,
    start_row=0,
    start_col=0,
    m_start=None,
    k_start=None,
    m_step=None,
    k_step=None,
    src_stride=None,
    dst_stride=None,
    transpose=False,
):
    """``pto.mte_l1_l0a`` – structured or explicit-control L1-to-L0A load.

    Use either the existing shape-derived ``m``/``k`` form or provide all six
    explicit L1-to-L0A controls. PTOAS selects the regular or packed-S4 raw
    load during wrapper expansion.
    """
    transpose = _normalize_cube_transpose(transpose, context="mte_l1_l0a")
    controls = (m_start, k_start, m_step, k_step, src_stride, dst_stride)
    has_explicit_controls = any(control is not None for control in controls)
    if has_explicit_controls:
        if m is not None or k is not None:
            raise TypeError(
                "mte_l1_l0a accepts either m/k or explicit controls, not both"
            )
        if start_row != 0 or start_col != 0:
            raise TypeError(
                "mte_l1_l0a start_row/start_col are unavailable with explicit controls"
            )
        if any(control is None for control in controls):
            raise TypeError(
                "mte_l1_l0a explicit controls require m_start, k_start, "
                "m_step, k_step, src_stride, and dst_stride"
            )
        _pto.MteL1L0aOp(
            unwrap_surface_value(source),
            unwrap_surface_value(destination),
            m_start=_coerce_i64(m_start, context="mte_l1_l0a m_start"),
            k_start=_coerce_i64(k_start, context="mte_l1_l0a k_start"),
            m_step=_coerce_i64(m_step, context="mte_l1_l0a m_step"),
            k_step=_coerce_i64(k_step, context="mte_l1_l0a k_step"),
            src_stride=_coerce_i64(src_stride, context="mte_l1_l0a src_stride"),
            dst_stride=_coerce_i64(dst_stride, context="mte_l1_l0a dst_stride"),
            transpose=transpose,
        )
        return
    if m is None or k is None:
        raise TypeError("mte_l1_l0a requires m and k without explicit controls")
    _pto.MteL1L0aOp(
        unwrap_surface_value(source),
        unwrap_surface_value(destination),
        m=_coerce_i64(m, context="mte_l1_l0a m"),
        k=_coerce_i64(k, context="mte_l1_l0a k"),
        start_row=_coerce_i64(start_row, context="mte_l1_l0a start_row"),
        start_col=_coerce_i64(start_col, context="mte_l1_l0a start_col"),
        transpose=transpose,
    )


@_explicit_mode_only("pto.mte_l1_l0b(...)")
def mte_l1_l0b(
    source,
    destination,
    k=None,
    n=None,
    *,
    start_row=0,
    start_col=0,
    m_start=None,
    k_start=None,
    m_step=None,
    k_step=None,
    src_stride=None,
    dst_stride=None,
    transpose=False,
):
    """``pto.mte_l1_l0b`` – structured or explicit-control L1-to-L0B load.

    Use either the existing shape-derived ``k``/``n`` form or provide all six
    explicit L1-to-L0B controls. PTOAS selects the regular or packed-S4 raw
    load during wrapper expansion.
    """
    transpose = _normalize_cube_transpose(transpose, context="mte_l1_l0b")
    controls = (m_start, k_start, m_step, k_step, src_stride, dst_stride)
    has_explicit_controls = any(control is not None for control in controls)
    if has_explicit_controls:
        if k is not None or n is not None:
            raise TypeError(
                "mte_l1_l0b accepts either k/n or explicit controls, not both"
            )
        if start_row != 0 or start_col != 0:
            raise TypeError(
                "mte_l1_l0b start_row/start_col are unavailable with explicit controls"
            )
        if any(control is None for control in controls):
            raise TypeError(
                "mte_l1_l0b explicit controls require m_start, k_start, "
                "m_step, k_step, src_stride, and dst_stride"
            )
        _pto.MteL1L0bOp(
            unwrap_surface_value(source),
            unwrap_surface_value(destination),
            m_start=_coerce_i64(m_start, context="mte_l1_l0b m_start"),
            k_start=_coerce_i64(k_start, context="mte_l1_l0b k_start"),
            m_step=_coerce_i64(m_step, context="mte_l1_l0b m_step"),
            k_step=_coerce_i64(k_step, context="mte_l1_l0b k_step"),
            src_stride=_coerce_i64(src_stride, context="mte_l1_l0b src_stride"),
            dst_stride=_coerce_i64(dst_stride, context="mte_l1_l0b dst_stride"),
            transpose=transpose,
        )
        return
    if k is None or n is None:
        raise TypeError("mte_l1_l0b requires k and n without explicit controls")
    _pto.MteL1L0bOp(
        unwrap_surface_value(source),
        unwrap_surface_value(destination),
        k=_coerce_i64(k, context="mte_l1_l0b k"),
        n=_coerce_i64(n, context="mte_l1_l0b n"),
        start_row=_coerce_i64(start_row, context="mte_l1_l0b start_row"),
        start_col=_coerce_i64(start_col, context="mte_l1_l0b start_col"),
        transpose=transpose,
    )


@_explicit_mode_only("pto.mte_l1_l0a_mx(...)")
def mte_l1_l0a_mx(
    source,
    destination,
    m=None,
    k=None,
    *,
    start_row=None,
    start_col=None,
    x_start=None,
    y_start=None,
    x_step=None,
    y_step=None,
    src_stride=None,
    dst_stride=None,
):
    """``pto.mte_l1_l0a_mx`` – MX cube-side LEFT staging.

    Use either the existing shape-derived ``m``/``k`` form or provide all six
    full MX operands. Both forms trace through the L1-to-L0A MX wrapper.
    """
    full_operands = (x_start, y_start, x_step, y_step, src_stride, dst_stride)
    has_full_operands = any(operand is not None for operand in full_operands)
    if has_full_operands:
        if m is not None or k is not None:
            raise TypeError(
                "mte_l1_l0a_mx accepts either m/k or full MX operands, not both"
            )
        # The legacy API defaulted these shape-only fields to zero. Continue
        # accepting explicit zero so existing full-MX callers remain valid.
        if ((start_row is not None and start_row != 0) or
                (start_col is not None and start_col != 0)):
            raise TypeError(
                "mte_l1_l0a_mx start_row/start_col are unavailable with full MX operands"
            )
        if any(operand is None for operand in full_operands):
            raise TypeError(
                "mte_l1_l0a_mx full MX operands require x_start, y_start, "
                "x_step, y_step, src_stride, and dst_stride"
            )
        _pto.MteL1L0aMxOp(
            unwrap_surface_value(source),
            unwrap_surface_value(destination),
            x_start=_coerce_i64(x_start, context="mte_l1_l0a_mx x_start"),
            y_start=_coerce_i64(y_start, context="mte_l1_l0a_mx y_start"),
            x_step=_coerce_i64(x_step, context="mte_l1_l0a_mx x_step"),
            y_step=_coerce_i64(y_step, context="mte_l1_l0a_mx y_step"),
            src_stride=_coerce_i64(src_stride, context="mte_l1_l0a_mx src_stride"),
            dst_stride=_coerce_i64(dst_stride, context="mte_l1_l0a_mx dst_stride"),
        )
        return
    if m is None or k is None:
        raise TypeError("mte_l1_l0a_mx requires m and k without full MX operands")
    if start_row is None:
        start_row = 0
    if start_col is None:
        start_col = 0
    _pto.MteL1L0aMxOp(
        unwrap_surface_value(source),
        unwrap_surface_value(destination),
        m=_coerce_i64(m, context="mte_l1_l0a_mx m"),
        k=_coerce_i64(k, context="mte_l1_l0a_mx k"),
        start_row=_coerce_i64(start_row, context="mte_l1_l0a_mx start_row"),
        start_col=_coerce_i64(start_col, context="mte_l1_l0a_mx start_col"),
    )


@_explicit_mode_only("pto.mte_l1_l0b_mx(...)")
def mte_l1_l0b_mx(
    source,
    destination,
    k=None,
    n=None,
    *,
    start_row=None,
    start_col=None,
    x_start=None,
    y_start=None,
    x_step=None,
    y_step=None,
    src_stride=None,
    dst_stride=None,
):
    """``pto.mte_l1_l0b_mx`` – MX cube-side RIGHT staging.

    Use either the existing shape-derived ``k``/``n`` form or provide all six
    full MX operands. Both forms trace through the L1-to-L0B MX wrapper.
    """
    full_operands = (x_start, y_start, x_step, y_step, src_stride, dst_stride)
    has_full_operands = any(operand is not None for operand in full_operands)
    if has_full_operands:
        if k is not None or n is not None:
            raise TypeError(
                "mte_l1_l0b_mx accepts either k/n or full MX operands, not both"
            )
        # The legacy API defaulted these shape-only fields to zero. Continue
        # accepting explicit zero so existing full-MX callers remain valid.
        if ((start_row is not None and start_row != 0) or
                (start_col is not None and start_col != 0)):
            raise TypeError(
                "mte_l1_l0b_mx start_row/start_col are unavailable with full MX operands"
            )
        if any(operand is None for operand in full_operands):
            raise TypeError(
                "mte_l1_l0b_mx full MX operands require x_start, y_start, "
                "x_step, y_step, src_stride, and dst_stride"
            )
        _pto.MteL1L0bMxOp(
            unwrap_surface_value(source),
            unwrap_surface_value(destination),
            x_start=_coerce_i64(x_start, context="mte_l1_l0b_mx x_start"),
            y_start=_coerce_i64(y_start, context="mte_l1_l0b_mx y_start"),
            x_step=_coerce_i64(x_step, context="mte_l1_l0b_mx x_step"),
            y_step=_coerce_i64(y_step, context="mte_l1_l0b_mx y_step"),
            src_stride=_coerce_i64(src_stride, context="mte_l1_l0b_mx src_stride"),
            dst_stride=_coerce_i64(dst_stride, context="mte_l1_l0b_mx dst_stride"),
        )
        return
    if k is None or n is None:
        raise TypeError("mte_l1_l0b_mx requires k and n without full MX operands")
    if start_row is None:
        start_row = 0
    if start_col is None:
        start_col = 0
    _pto.MteL1L0bMxOp(
        unwrap_surface_value(source),
        unwrap_surface_value(destination),
        k=_coerce_i64(k, context="mte_l1_l0b_mx k"),
        n=_coerce_i64(n, context="mte_l1_l0b_mx n"),
        start_row=_coerce_i64(start_row, context="mte_l1_l0b_mx start_row"),
        start_col=_coerce_i64(start_col, context="mte_l1_l0b_mx start_col"),
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


def mad_bias(lhs, rhs, dst, bias, m, n, k, *, unit_flag=None, disable_gemv=False,
             sat=None, tf32_mode=None, n_dir=False):
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
