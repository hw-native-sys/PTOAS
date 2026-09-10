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


def _constraints(src_config, tmp_config, dst_config, src_dtype, tmp_dtype, dst_dtype, dst_valid_shape, **_):
    return (
        src_config.b_layout == "row_major"
        and tmp_config.b_layout == "row_major"
        and dst_config.b_layout == "row_major"
        and src_config.s_layout == "none_box"
        and tmp_config.s_layout == "none_box"
        and dst_config.s_layout == "none_box"
        and src_dtype == tmp_dtype
        and dst_dtype == "i32"
        and dst_valid_shape[0] == 1
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
    """Emit the interleaved i32 index stores for one i8/ui8 output quarter."""
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
    output_0 = pto.vbitcast(output_0, pto.i32)
    output_1 = pto.vbitcast(output_1, pto.i32)
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
    """Emit the two interleaved i32 store quarters for one i8/ui8 column."""
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
def _emit_col_arg_i8_ui8(src, dst, cmp_mode, reduce_op):
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
    store = _I8Ui8IndexStore(dst, lanes_i32, full_mask_intermediate, final_dtype)

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


def register_col_arg_template(*, op, name, cmp_mode, reduce_op):
    @tilelib.tile_template(
        op=op,
        target="a5",
        name=name,
        dtypes=[("f32", "f32", "i32"), ("ui32", "ui32", "i32")],
        iteration_axis="column",
        op_engine="vector",
        op_class="reduction",
        constraints=[_constraints],
        id=0,
        loop_depth=2,
        is_post_update=False,
        tags=("reduction", "column", "arg"),
    )
    def template(src: pto.Tile, tmp: pto.Tile, dst: pto.Tile):
        _ = tmp
        src_valid_rows, src_valid_cols = src.valid_shape
        src_cols = src.shape[1]
        src_ptr = src.as_ptr()
        dst_ptr = dst.as_ptr()
        lanes = pto.elements_per_vreg(src.dtype)

        full_mask = pto.make_mask(src.dtype, pto.PAT.ALL)
        for col in range(0, src_valid_cols, lanes):
            remained = src_valid_cols - col
            mask, _ = pto.make_mask(src.dtype, remained)
            index_old = pto.vdup(pto.i32(0), mask)
            index_new = pto.vdup(pto.i32(0), mask)
            src_addr = pto.addptr(src_ptr, col)
            best_vals = pto.vlds(src_addr, 0)

            for row in range(1, src_valid_rows, 1):
                index_new = pto.vadds(index_new, pto.i32(1), mask)
                src_addr = pto.addptr(src_ptr, row * src_cols + col)
                new_vals = pto.vlds(src_addr, 0)
                select = pto.vcmp(new_vals, best_vals, full_mask, cmp_mode)
                index_old = pto.vsel(index_new, index_old, select)
                best_vals = reduce_op(best_vals, new_vals, mask)

            dst_addr = pto.addptr(dst_ptr, col)
            pto.vsts(index_old, dst_addr, 0, mask)

    @tilelib.tile_template(
        op=op,
        target="a5",
        name=name.replace("f32_to_i32", "f16_ui16_to_i32"),
        dtypes=[("f16", "f16", "i32"), ("ui16", "ui16", "i32")],
        iteration_axis="column",
        op_engine="vector",
        op_class="reduction",
        constraints=[_constraints],
        id=1,
        loop_depth=2,
        is_post_update=False,
        tags=("reduction", "column", "arg"),
    )
    def template_f16_ui16(src: pto.Tile, tmp: pto.Tile, dst: pto.Tile):
        _ = tmp
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

            index_even = pto.vcvt(index_old, pto.i32, full_mask, part=pto.VcvtPartMode.EVEN)
            index_odd = pto.vcvt(index_old, pto.i32, full_mask, part=pto.VcvtPartMode.ODD)
            index_lo, index_hi = pto.vintlv(index_even, index_odd)

            pto.vsts(index_lo, dst[0, col:], mask_i32_0)
            pto.vsts(index_hi, dst[0, col + lanes_i32:], mask_i32_1)

    @tilelib.tile_template(
        op=op,
        target="a5",
        name=name.replace("f32_to_i32", "i8_ui8_to_i32"),
        dtypes=[("i8", "i8", "i32"), ("ui8", "ui8", "i32")],
        iteration_axis="column",
        op_engine="vector",
        op_class="reduction",
        constraints=[_constraints],
        id=2,
        loop_depth=2,
        is_post_update=False,
        tags=("reduction", "column", "arg"),
    )
    def template_i8_ui8(src: pto.Tile, tmp: pto.Tile, dst: pto.Tile):
        _ = tmp
        _emit_col_arg_i8_ui8(src, dst, cmp_mode, reduce_op)

    return template


__all__ = ["register_col_arg_template"]
