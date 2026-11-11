# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Shared column arg-reduction templates."""

from dataclasses import dataclass

from ptodsl import pto
from ptodsl._ast_rewrite import rewrite_jit_function
import ptodsl.tilelib as tilelib

from ._arg_reduction_constraints import (
    ArgReductionConstraintContext,
    index_output_constraints,
    value_output_constraints,
)


def _col_arg_index_constraints(**kwargs):
    context = ArgReductionConstraintContext.from_kwargs(kwargs)
    return index_output_constraints(context, "col", {"i32", "ui32"})


def _col_arg_value_constraints(**kwargs):
    context = ArgReductionConstraintContext.from_kwargs(kwargs)
    index_dtypes = (
        {"i16", "ui16"}
        if context.src_dtype in {"f16", "ui16"}
        else {"i32", "ui32"}
    )
    source_dtypes = {"f16", "f32", "ui16", "ui32"}
    return value_output_constraints(
        context, "col", index_dtypes, source_dtypes
    )


def _intermediate_value_dtype(dtype):
    if str(dtype) == "ui8":
        return pto.ui16
    return pto.i16


@dataclass(frozen=True)
class _I8Ui8IndexStore:
    """Trace-time invariants for one i8/ui8 index-store quarter."""

    dst: object
    lanes_i32: object
    full_mask_intermediate: object
    final_dtype: object


def _emit_i8_ui8_indexes(store, index_output, col_base, masks):
    """Emit interleaved 32-bit index stores for one i8/ui8 output quarter."""
    dst = store.dst
    lanes_i32 = store.lanes_i32
    full_mask_intermediate = store.full_mask_intermediate
    final_dtype = store.final_dtype
    output_even = pto.vcvt(
        index_output, final_dtype, full_mask_intermediate,
        part=pto.VcvtPartMode.EVEN,
    )
    output_odd = pto.vcvt(
        index_output, final_dtype, full_mask_intermediate,
        part=pto.VcvtPartMode.ODD,
    )
    output_0, output_1 = pto.vintlv(output_even, output_odd)
    output_0 = pto.vbitcast(output_0, store.dst.dtype)
    output_1 = pto.vbitcast(output_1, store.dst.dtype)
    pto.vsts(output_0, dst[0, col_base:], masks[0])
    pto.vsts(output_1, dst[0, col_base + lanes_i32:], masks[1])


def _i8_ui8_col_i32_masks(valid_cols, col):
    """Split one column's remaining i32 lanes into the four store masks."""
    remained = valid_cols - col
    mask_i32_0, remained = pto.make_mask(pto.i32, remained)
    mask_i32_1, remained = pto.make_mask(pto.i32, remained)
    mask_i32_2, remained = pto.make_mask(pto.i32, remained)
    mask_i32_3, _ = pto.make_mask(pto.i32, remained)
    return mask_i32_0, mask_i32_1, mask_i32_2, mask_i32_3


def _emit_i8_ui8_col_result(store, index_output_0, index_output_1, col_base, masks):
    """Emit two interleaved 32-bit index stores for one i8/ui8 column."""
    _emit_i8_ui8_indexes(
        store,
        index_output_0,
        col_base,
        masks[:2],
    )
    _emit_i8_ui8_indexes(
        store,
        index_output_1,
        col_base + 2 * store.lanes_i32,
        masks[2:],
    )


@rewrite_jit_function
def _emit_col_arg_i8_ui8(src, dst_idx, cmp_mode, reduce_op):
    src_valid_rows, src_valid_cols = src.valid_shape
    dtype = src.dtype
    intermediate_dtype = _intermediate_value_dtype(dtype)
    index_dtype = pto.i16
    final_dtype = pto.i32
    lanes_i8 = pto.elements_per_vreg(dtype)
    lanes_i32 = pto.elements_per_vreg(pto.i32)
    full_mask_i8 = pto.make_mask(dtype, pto.PAT.ALL)
    full_mask_intermediate = pto.make_mask(intermediate_dtype, pto.PAT.ALL)
    full_mask_index = pto.make_mask(index_dtype, pto.PAT.ALL)
    store = _I8Ui8IndexStore(dst_idx, lanes_i32, full_mask_intermediate, final_dtype)

    for col in range(0, src_valid_cols, lanes_i8):
        masks = _i8_ui8_col_i32_masks(src_valid_cols, col)

        index_old_even = pto.vdup(index_dtype(0), full_mask_index)
        index_old_odd = pto.vdup(index_dtype(0), full_mask_index)
        index_new_even = pto.vdup(index_dtype(0), full_mask_index)
        index_new_odd = pto.vdup(index_dtype(0), full_mask_index)

        vreg_old = pto.vlds(src[0, col:])
        vreg_old_even = pto.vcvt(vreg_old, intermediate_dtype, full_mask_i8, part=pto.VcvtPartMode.EVEN)
        vreg_old_odd = pto.vcvt(vreg_old, intermediate_dtype, full_mask_i8, part=pto.VcvtPartMode.ODD)

        for row in range(1, src_valid_rows, 1):
            index_new_even = pto.vadds(index_new_even, index_dtype(1), full_mask_index)
            index_new_odd = pto.vadds(index_new_odd, index_dtype(1), full_mask_index)
            vreg_new = pto.vlds(src[row, col:])
            vreg_new_even = pto.vcvt(vreg_new, intermediate_dtype, full_mask_i8, part=pto.VcvtPartMode.EVEN)
            vreg_new_odd = pto.vcvt(vreg_new, intermediate_dtype, full_mask_i8, part=pto.VcvtPartMode.ODD)

            select_even = pto.vcmp(vreg_new_even, vreg_old_even, full_mask_intermediate, cmp_mode)
            select_odd = pto.vcmp(vreg_new_odd, vreg_old_odd, full_mask_intermediate, cmp_mode)

            index_old_even = pto.vsel(index_new_even, index_old_even, select_even)
            index_old_odd = pto.vsel(index_new_odd, index_old_odd, select_odd)

            vreg_old_even = reduce_op(vreg_old_even, vreg_new_even, full_mask_intermediate)
            vreg_old_odd = reduce_op(vreg_old_odd, vreg_new_odd, full_mask_intermediate)

        index_output_0, index_output_1 = pto.vintlv(index_old_even, index_old_odd)
        _emit_i8_ui8_col_result(
            store,
            index_output_0,
            index_output_1,
            col,
            masks,
        )


@rewrite_jit_function
def _emit_col_arg_f32_ui32(src, dst_idx, dst_val, cmp_mode, reduce_op):
    src_valid_rows, src_valid_cols = src.valid_shape
    src_cols = src.shape[1]
    src_ptr = src.as_ptr()
    dst_idx_ptr = dst_idx.as_ptr()
    index_dtype = dst_idx.dtype
    index_zero = pto.ui32(0) if str(index_dtype) == "ui32" else pto.i32(0)
    index_one = pto.ui32(1) if str(index_dtype) == "ui32" else pto.i32(1)
    lanes = pto.elements_per_vreg(src.dtype)
    full_mask = pto.make_mask(src.dtype, pto.PAT.ALL)

    for col in range(0, src_valid_cols, lanes):
        remained = src_valid_cols - col
        mask, _ = pto.make_mask(src.dtype, remained)
        index_old = pto.vdup(index_zero, mask)
        index_new = pto.vdup(index_zero, mask)
        src_addr = pto.addptr(src_ptr, col)
        best_vals = pto.vlds(src_addr, 0)

        for row in range(1, src_valid_rows, 1):
            index_new = pto.vadds(index_new, index_one, mask)
            src_addr = pto.addptr(src_ptr, row * src_cols + col)
            new_vals = pto.vlds(src_addr, 0)
            select = pto.vcmp(new_vals, best_vals, full_mask, cmp_mode)
            index_old = pto.vsel(index_new, index_old, select)
            best_vals = reduce_op(best_vals, new_vals, mask)

        dst_idx_addr = pto.addptr(dst_idx_ptr, col)
        pto.vsts(index_old, dst_idx_addr, 0, mask)
        if pto.const_expr(dst_val is not None):
            dst_val_addr = pto.addptr(dst_val.as_ptr(), col)
            pto.vsts(best_vals, dst_val_addr, 0, mask)


@rewrite_jit_function
def _emit_col_arg_f16_ui16(src, dst_idx, dst_val, cmp_mode, reduce_op):
    src_valid_rows, src_valid_cols = src.valid_shape
    dtype = src.dtype
    lanes = pto.elements_per_vreg(dtype)
    lanes_i32 = pto.elements_per_vreg(pto.i32)
    full_mask = pto.make_mask(dtype, pto.PAT.ALL)

    for col in range(0, src_valid_cols, lanes):
        remained = src_valid_cols - col
        elem_mask, _ = pto.make_mask(dtype, remained)
        mask_i32_0, remained = pto.make_mask(pto.i32, remained)
        mask_i32_1, _ = pto.make_mask(pto.i32, remained)

        index_old = pto.vdup(pto.i16(0), elem_mask)
        index_new = pto.vdup(pto.i16(0), elem_mask)
        best_vals = pto.vlds(src[0, col:])

        for row in range(1, src_valid_rows, 1):
            index_new = pto.vadds(index_new, pto.i16(1), elem_mask)
            new_vals = pto.vlds(src[row, col:])
            select = pto.vcmp(new_vals, best_vals, full_mask, cmp_mode)
            index_old = pto.vsel(index_new, index_old, select)
            best_vals = reduce_op(best_vals, new_vals, elem_mask)

        if pto.const_expr(dst_val is None):
            index_even = pto.vcvt(
                index_old, pto.i32, full_mask, part=pto.VcvtPartMode.EVEN
            )
            index_odd = pto.vcvt(
                index_old, pto.i32, full_mask, part=pto.VcvtPartMode.ODD
            )
            index_lo, index_hi = pto.vintlv(index_even, index_odd)
            if str(dst_idx.dtype) == "ui32":
                index_lo = pto.vbitcast(index_lo, pto.ui32)
                index_hi = pto.vbitcast(index_hi, pto.ui32)
            pto.vsts(index_lo, dst_idx[0, col:], mask_i32_0)
            pto.vsts(index_hi, dst_idx[0, col + lanes_i32:], mask_i32_1)
        else:
            index_output = index_old
            if str(dst_idx.dtype) == "ui16":
                index_output = pto.vbitcast(index_output, pto.ui16)
            pto.vsts(index_output, dst_idx[0, col:], elem_mask)
            pto.vsts(best_vals, dst_val[0, col:], elem_mask)


def _register_col_arg_candidate(
    *,
    op,
    name,
    dtypes,
    candidate_id,
    constraints,
    emit,
    cmp_mode,
    reduce_op,
    with_value_output,
):
    metadata = {
        "op": op,
        "target": "a5",
        "name": name,
        "dtypes": dtypes,
        "iteration_axis": "column",
        "op_engine": "vector",
        "op_class": "reduction",
        "constraints": [constraints],
        "id": candidate_id,
        "loop_depth": 2,
        "is_post_update": False,
        "tags": ("reduction", "column", "arg"),
    }
    if with_value_output:
        @tilelib.tile_template(**metadata)
        def template(
            src: pto.Tile,
            tmp: pto.Tile,
            dst_idx: pto.Tile,
            dst_val: pto.Tile,
        ):
            _ = tmp
            emit(src, dst_idx, dst_val, cmp_mode, reduce_op)

        return template

    @tilelib.tile_template(**metadata)
    def template_index_only(src: pto.Tile, tmp: pto.Tile, dst_idx: pto.Tile):
        _ = tmp
        emit(src, dst_idx, None, cmp_mode, reduce_op)

    return template_index_only


def _emit_col_arg_i8_ui8_index(src, dst_idx, _dst_val, cmp_mode, reduce_op):
    _emit_col_arg_i8_ui8(src, dst_idx, cmp_mode, reduce_op)


_F32_UI32_INDEX_DTYPES = (
    ("f32", "f32", "i32"),
    ("f32", "f32", "ui32"),
    ("ui32", "ui32", "i32"),
    ("ui32", "ui32", "ui32"),
    ("i32", "i32", "i32"),
    ("i32", "i32", "ui32"),
)
_F32_UI32_VALUE_DTYPES = (
    ("f32", "f32", "i32", "f32"),
    ("f32", "f32", "ui32", "f32"),
    ("ui32", "ui32", "i32", "ui32"),
    ("ui32", "ui32", "ui32", "ui32"),
)
_F16_UI16_INDEX_DTYPES = (
    ("f16", "f16", "i32"),
    ("f16", "f16", "ui32"),
    ("ui16", "ui16", "i32"),
    ("ui16", "ui16", "ui32"),
    ("i16", "i16", "i32"),
    ("i16", "i16", "ui32"),
)
_F16_UI16_VALUE_DTYPES = (
    ("f16", "f16", "i16", "f16"),
    ("f16", "f16", "ui16", "f16"),
    ("ui16", "ui16", "i16", "ui16"),
    ("ui16", "ui16", "ui16", "ui16"),
)
_I8_UI8_INDEX_DTYPES = (
    ("i8", "i8", "i32"),
    ("i8", "i8", "ui32"),
    ("ui8", "ui8", "i32"),
    ("ui8", "ui8", "ui32"),
)


def _col_arg_candidate_specs(name):
    f16_ui16_name = name.replace("f32_to_i32", "f16_ui16")
    i8_ui8_name = name.replace("f32_to_i32", "i8_ui8_to_i32")
    return (
        (
            f"{name}_index_only", _F32_UI32_INDEX_DTYPES, 3,
            _col_arg_index_constraints, _emit_col_arg_f32_ui32, False,
        ),
        (
            name, _F32_UI32_VALUE_DTYPES, 0,
            _col_arg_value_constraints, _emit_col_arg_f32_ui32, True,
        ),
        (
            f"{f16_ui16_name}_index_only", _F16_UI16_INDEX_DTYPES, 4,
            _col_arg_index_constraints, _emit_col_arg_f16_ui16, False,
        ),
        (
            f16_ui16_name, _F16_UI16_VALUE_DTYPES, 1,
            _col_arg_value_constraints, _emit_col_arg_f16_ui16, True,
        ),
        (
            i8_ui8_name, _I8_UI8_INDEX_DTYPES, 2,
            _col_arg_index_constraints, _emit_col_arg_i8_ui8_index, False,
        ),
    )


def register_col_arg_template(*, op, name, cmp_mode, reduce_op):
    template = None
    for candidate_name, dtypes, candidate_id, constraint, emit, with_value_output in (
        _col_arg_candidate_specs(name)
    ):
        registered = _register_col_arg_candidate(
            op=op,
            name=candidate_name,
            dtypes=dtypes,
            candidate_id=candidate_id,
            constraints=constraint,
            emit=emit,
            cmp_mode=cmp_mode,
            reduce_op=reduce_op,
            with_value_output=with_value_output,
        )
        if candidate_name == name:
            template = registered
    return template


__all__ = ["register_col_arg_template"]
