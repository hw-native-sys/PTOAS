# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Shared PTODSL implementations for row arg-reductions."""

from ptodsl import pto
from ptodsl._ast_rewrite import rewrite_jit_function
import ptodsl.tilelib as tilelib

from ._common import ROW_REDUCTION_DTYPES, element_store_dist, is_bf16, scalar_literal
from ._arg_reduction_constraints import (
    ArgReductionConstraintContext,
    index_output_constraints,
    value_output_constraints,
)
from ._part import pad_max, pad_min


def _row_arg_index_constraints(**kwargs):
    context = ArgReductionConstraintContext.from_kwargs(kwargs)
    return index_output_constraints(context, "row", {"i32", "ui32"})


def _row_arg_value_constraints(**kwargs):
    context = ArgReductionConstraintContext.from_kwargs(kwargs)
    return value_output_constraints(context, "row", {"i32", "ui32"})


@rewrite_jit_function
def _update_row_arg_acc(val_acc, idx_acc, val, idx, value_one_mask, cmp_mode):
    cmp = pto.vcmp(val_acc, val, value_one_mask, cmp_mode)
    cmp_idx = pto.pbitcast(cmp, pto.mask_b32)
    return pto.vsel(val, val_acc, cmp), pto.vsel(idx, idx_acc, cmp_idx)


@rewrite_jit_function
def _emit_row_arg_value(dst_val, row, val_acc, widen, conversion_mask):
    dst_val_cols = dst_val.shape[1]
    dst_val_ptr = dst_val.as_ptr()
    dst_val_addr = pto.addptr(dst_val_ptr, row * dst_val_cols)
    val_one_mask, _ = pto.make_mask(dst_val.dtype, 1)
    value_to_store = val_acc
    if widen:
        value_to_store = pto.vcvt(
            val_acc,
            dst_val.dtype,
            conversion_mask,
            rnd=pto.VcvtRoundMode.R,
            sat=pto.VcvtSatMode.NOSAT,
            part=pto.VcvtPartMode.EVEN,
        )
    pto.vsts(
        value_to_store,
        dst_val_addr,
        0,
        val_one_mask,
        dist=element_store_dist(dst_val.dtype),
    )


@rewrite_jit_function
def _emit_row_arg(src, dst_idx, dst_val, reduce_op, cmp_mode):
    src_dtype = src.dtype
    idx_dtype = dst_idx.dtype
    valid_rows, valid_cols = src.valid_shape
    src_cols = src.shape[1]
    dst_idx_cols = dst_idx.shape[1]
    src_ptr = src.as_ptr()
    dst_idx_ptr = dst_idx.as_ptr()
    widen = is_bf16(src_dtype)
    reduction_dtype = pto.f32.resolve() if widen else src_dtype
    lanes = pto.elements_per_vreg(reduction_dtype)
    value_one_mask, _ = pto.make_mask(reduction_dtype, 1)
    idx_one_mask, _ = pto.make_mask(idx_dtype, 1)
    init_val = pad_min(reduction_dtype) if cmp_mode == "lt" else pad_max(reduction_dtype)
    if widen:
        conversion_mask = pto.make_mask(src_dtype, pto.PAT.ALL)

    for row in range(0, valid_rows, 1):
        remained = valid_cols
        val_acc = pto.vbr(init_val)
        idx_acc = pto.vbr(scalar_literal(idx_dtype, 0))
        zero_src = pto.vbr(scalar_literal(reduction_dtype, 0))

        for col in range(0, valid_cols, lanes):
            mask, remained = pto.make_mask(reduction_dtype, remained)
            src_addr = pto.addptr(src_ptr, row * src_cols + col)
            if widen:
                unpacked = pto.vlds(src_addr, 0, dist="UNPK_B16")
                value = pto.vcvt(
                    unpacked,
                    pto.f32,
                    conversion_mask,
                    part=pto.VcvtPartMode.EVEN,
                )
            else:
                value = pto.vlds(src_addr, 0)
            reduced = reduce_op(value, mask)
            val, idx_src = pto.vdintlv(reduced, zero_src)
            idx = pto.vadds(pto.vbitcast(idx_src, idx_dtype), col, idx_one_mask)
            val_acc, idx_acc = _update_row_arg_acc(
                val_acc, idx_acc, val, idx, value_one_mask, cmp_mode
            )

        dst_idx_addr = pto.addptr(dst_idx_ptr, row * dst_idx_cols)
        pto.vsts(
            idx_acc,
            dst_idx_addr,
            0,
            idx_one_mask,
            dist=element_store_dist(idx_dtype),
        )
        if pto.const_expr(dst_val is not None):
            _emit_row_arg_value(dst_val, row, val_acc, widen, value_one_mask)


def register_row_arg(*, op, name, reduce_op, cmp_mode):
    index_dtypes = [
        (dtype, dtype, index_dtype)
        for dtype in ROW_REDUCTION_DTYPES + ("bf16",)
        for index_dtype in ("i32", "ui32")
    ]
    value_dtypes = [
        (dtype, dtype, index_dtype, dtype)
        for dtype in ROW_REDUCTION_DTYPES + ("bf16",)
        for index_dtype in ("i32", "ui32")
    ]
    common_metadata = {
        "op": op,
        "target": "a5",
        "iteration_axis": "row",
        "op_engine": "vector",
        "op_class": "reduction",
        "loop_depth": 2,
        "is_post_update": False,
        "tags": ("reduction", "row", "arg"),
    }
    common_constraints = [
        tilelib.check_memory_space("ub"),
        tilelib.check_layout("row_major"),
        tilelib.check_s_layout("none_box"),
    ]

    @tilelib.tile_template(
        **common_metadata,
        name=f"{name}_index_only",
        dtypes=index_dtypes,
        constraints=[*common_constraints, _row_arg_index_constraints],
        id=1,
    )
    def template_index_only(src: pto.Tile, tmp: pto.Tile, dst_idx: pto.Tile):
        _ = tmp
        _emit_row_arg(src, dst_idx, None, reduce_op, cmp_mode)

    @tilelib.tile_template(
        **common_metadata,
        name=name,
        dtypes=value_dtypes,
        constraints=[*common_constraints, _row_arg_value_constraints],
        id=0,
    )
    def template(src: pto.Tile, tmp: pto.Tile, dst_idx: pto.Tile, dst_val: pto.Tile):
        _ = tmp
        _emit_row_arg(src, dst_idx, dst_val, reduce_op, cmp_mode)

    return template


__all__ = ["register_row_arg"]
