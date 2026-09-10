# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""PTODSL TileLib templates for pto.tsels."""

from dataclasses import dataclass

from ptodsl import pto
from ptodsl._ast_rewrite import rewrite_jit_function
import ptodsl.tilelib as tilelib

from ._elementwise import (
    _emit_select_2d_f32_pair_tail,
    _f32_paired_cols,
    _f32_select_masks,
    traversal_metadata,
)


_DTYPES = [
    ("i8", "i8", "i8", "i8", "i8"),
    ("i16", "i8", "i8", "i8", "i8"),
    ("i32", "i8", "i8", "i8", "i8"),
    ("i8", "i16", "i16", "i16", "i16"),
    ("i16", "i16", "i16", "i16", "i16"),
    ("i32", "i16", "i16", "i16", "i16"),
    ("i8", "i32", "i32", "i32", "i32"),
    ("i16", "i32", "i32", "i32", "i32"),
    ("i32", "i32", "i32", "i32", "i32"),
    ("i8", "f32", "f32", "f32", "f32"),
    ("i16", "f32", "f32", "f32", "f32"),
    ("i32", "f32", "f32", "f32", "f32"),
    ("i8", "f16", "f16", "f16", "f16"),
    ("i16", "f16", "f16", "f16", "f16"),
    ("i32", "f16", "f16", "f16", "f16"),
]


def _tsels_shapes(
    mask_valid_shape=(),
    src_valid_shape=(),
    tmp_valid_shape=(),
    dst_valid_shape=(),
    **_,
):
    _ = mask_valid_shape, tmp_valid_shape
    return (
        len(src_valid_shape) == 2
        and tuple(src_valid_shape) == tuple(dst_valid_shape)
    )


def _scalar_vector(dtype, scalar):
    lanes = pto.elements_per_vreg(dtype)
    full_mask, _ = pto.make_mask(dtype, lanes)
    return pto.vdup(scalar, full_mask)


@dataclass(frozen=True)
class _TselsLinearOperands:
    """Trace-time operands for the linear scalar-select emitter."""

    mask_ptr: object
    src_ptr: object
    dst_ptr: object
    scalar_vec: object
    dtype: object
    total_elements: object


@rewrite_jit_function
def _emit_tsels_linear(operands):
    """Emit linear scalar-select stores for non-wide vector types."""
    mask_ptr = operands.mask_ptr
    src_ptr = operands.src_ptr
    dst_ptr = operands.dst_ptr
    scalar_vec = operands.scalar_vec
    dtype = operands.dtype
    total_elements = operands.total_elements
    lanes = pto.elements_per_vreg(dtype)
    remained = total_elements
    is_f16 = str(dtype) in {"f16", "i16"}
    for offset in range(0, total_elements, lanes):
        pred, remained = pto.make_mask(dtype, remained)
        dist = pto.PredicateDist.US if is_f16 else pto.PredicateDist.NORM
        select_mask = pto.plds(mask_ptr, offset // 8, dist=dist)
        if is_f16:
            select_mask = pto.pbitcast(select_mask, pto.mask_b16)
        value = pto.vlds(src_ptr, offset)
        pto.vsts(pto.vsel(value, scalar_vec, select_mask), dst_ptr, offset, pred)


@rewrite_jit_function
def _emit_tsels_1d(mask, src, scalar, dst):
    dtype = dst.dtype
    valid_rows, valid_cols = dst.valid_shape
    lanes = pto.elements_per_vreg(dtype)
    total_elements = valid_rows * valid_cols
    mask_ptr = pto.castptr(mask.as_ptr(), pto.ptr(pto.ui8, "ub"))
    src_ptr = src.as_ptr()
    dst_ptr = dst.as_ptr()
    scalar_vec = _scalar_vector(dtype, scalar)

    if str(dtype) in {"f32", "i32"}:
        full_mask_b16 = pto.pset_b16(pto.PAT.ALL)
        remained = total_elements
        for offset in range(0, total_elements, lanes * 2):
            raw_mask = pto.plds(
                mask_ptr,
                offset // 8,
                dist=pto.PredicateDist.US,
            )
            select_mask0, select_mask1 = _f32_select_masks(
                raw_mask,
                full_mask_b16,
            )
            pred0, remained = pto.make_mask(dtype, remained)
            pred1, remained = pto.make_mask(dtype, remained)
            value0 = pto.vlds(src_ptr, offset)
            value1 = pto.vlds(src_ptr, offset + lanes)
            selected0 = pto.vsel(value0, scalar_vec, select_mask0)
            selected1 = pto.vsel(value1, scalar_vec, select_mask1)
            pto.vsts(selected0, dst_ptr, offset, pred0)
            pto.vsts(selected1, dst_ptr, offset + lanes, pred1)
    else:
        operands = _TselsLinearOperands(
            mask_ptr,
            src_ptr,
            dst_ptr,
            scalar_vec,
            dtype,
            total_elements,
        )
        _emit_tsels_linear(operands)


def _emit_tsels_2d_f32(mask, src, scalar, dst):
    """Emit per-row paired + tail scalar-select stores for f32/i32 payloads."""
    dtype = dst.dtype
    valid_rows, valid_cols = dst.valid_shape
    lanes = pto.elements_per_vreg(dtype)
    mask_ptr = pto.castptr(mask.as_ptr(), pto.ptr(pto.ui8, "ub"))
    mask_stride = mask.shape[1] * pto.bytewidth(mask.dtype)
    scalar_vec = _scalar_vector(dtype, scalar)
    full_mask_b16 = pto.pset_b16(pto.PAT.ALL)
    paired_cols = _f32_paired_cols(valid_cols, lanes)

    def load_pair(row, col):
        return (
            pto.vlds(src[row, col:]),
            scalar_vec,
            pto.vlds(src[row, col + lanes:]),
            scalar_vec,
        )

    def load_tail(row, col):
        return (
            pto.vlds(src[row, col:]),
            scalar_vec,
        )

    _emit_select_2d_f32_pair_tail(
        mask_ptr,
        mask_stride,
        dtype,
        valid_rows,
        valid_cols,
        lanes,
        dst,
        full_mask_b16,
        paired_cols,
        load_pair,
        load_tail,
    )


@rewrite_jit_function
def _emit_tsels_2d_narrow(mask, src, scalar, dst, packed):
    """Emit per-row scalar-select stores for packed f16/i16 or byte i8 masks."""
    dtype = dst.dtype
    valid_rows, valid_cols = dst.valid_shape
    lanes = pto.elements_per_vreg(dtype)
    mask_ptr = pto.castptr(mask.as_ptr(), pto.ptr(pto.ui8, "ub"))
    mask_stride = mask.shape[1] * pto.bytewidth(mask.dtype)
    scalar_vec = _scalar_vector(dtype, scalar)
    dist = pto.PredicateDist.US if packed else pto.PredicateDist.NORM
    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, lanes):
            pred, remained = pto.make_mask(dtype, remained)
            mask_offset = row * mask_stride + col // 8
            select_mask = pto.plds(
                mask_ptr,
                mask_offset,
                dist=dist,
            )
            if packed:
                select_mask = pto.pbitcast(select_mask, pto.mask_b16)
            value = pto.vlds(src[row, col:])
            result = pto.vsel(value, scalar_vec, select_mask)
            pto.vsts(result, dst[row, col:], pred)


@rewrite_jit_function
def _emit_tsels_2d(mask, src, scalar, dst):
    dtype = dst.dtype
    if str(dtype) in {"f32", "i32"}:
        _emit_tsels_2d_f32(mask, src, scalar, dst)
    else:
        _emit_tsels_2d_narrow(mask, src, scalar, dst, str(dtype) in {"f16", "i16"})


def _register_tsels(*, name, traversal):
    constraints = [
        tilelib.check_memory_space("ub"),
        tilelib.check_layout("row_major"),
        tilelib.check_s_layout("none_box"),
        _tsels_shapes,
    ]
    loop_depth, priority, candidate_id = traversal_metadata(traversal)
    if traversal == "1d":
        constraints.append(
            tilelib.require_predicate_select_1d(
                "mask",
                "src",
                "dst",
                temporary_operand="tmp",
                memory_spaces=("ub",),
            )
        )

    @tilelib.tile_template(
        op="pto.tsels",
        target="a5",
        name=name,
        dtypes=_DTYPES,
        iteration_axis="none",
        op_engine="vector",
        op_class="other",
        constraints=constraints,
        priority=priority,
        id=candidate_id,
        loop_depth=loop_depth,
        is_post_update=False,
        tags=("select", "scalar", "predicate-load"),
    )
    def template(
        mask: pto.Tile,
        src: pto.Tile,
        tmp: pto.Tile,
        scalar,
        dst: pto.Tile,
    ):
        _ = tmp
        if traversal == "1d":
            _emit_tsels_1d(mask, src, scalar, dst)
        else:
            _emit_tsels_2d(mask, src, scalar, dst)

    return template


template_tsels = _register_tsels(
    name="template_tsels",
    traversal="2d",
)

template_tsels_1d = _register_tsels(
    name="template_tsels_1d",
    traversal="1d",
)
