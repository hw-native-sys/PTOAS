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
import ptodsl.tilelib as tilelib

from ._common import NUMERIC_DTYPES, element_store_dist
from ._part import pad_max, pad_min


def _single_output_col(dst_valid_shape=(), **_):
    return len(dst_valid_shape) == 2 and dst_valid_shape[1] == 1


def _scalar_literal(dtype, value):
    name = str(dtype)
    if name == "f32":
        return pto.f32(float(value))
    if name == "f16":
        return pto.f16(float(value))
    if name == "bf16":
        return pto.bf16(float(value))
    if name == "ui32":
        return pto.ui32(value)
    if name == "ui16":
        return pto.ui16(value)
    if name == "ui8":
        return pto.ui8(value)
    if name in {"i32", "si32"}:
        return pto.i32(value)
    if name in {"i16", "si16"}:
        return pto.i16(value)
    return pto.i8(value)


def register_row_arg(*, op, name, reduce_op, cmp_mode):
    @tilelib.tile_template(
        op=op,
        target="a5",
        name=name,
        dtypes=[(dtype, dtype, index_dtype) for dtype in NUMERIC_DTYPES for index_dtype in ("i32", "ui32")],
        iteration_axis="row",
        op_engine="vector",
        op_class="reduction",
        constraints=[
            tilelib.check_memory_space("ub"),
            tilelib.check_layout("row_major"),
            tilelib.check_s_layout("none_box"),
            _single_output_col,
        ],
        id=0,
        loop_depth=2,
        is_post_update=False,
        tags=("reduction", "row", "arg"),
    )
    def template(src: pto.Tile, tmp: pto.Tile, dst: pto.Tile):
        _ = tmp
        src_dtype = src.dtype
        idx_dtype = dst.dtype
        valid_rows, valid_cols = src.valid_shape
        src_cols = src.shape[1]
        dst_cols = dst.shape[1]
        src_ptr = src.as_ptr()
        dst_ptr = dst.as_ptr()
        lanes = pto.elements_per_vreg(src_dtype)
        src_one_mask, _ = pto.make_mask(src_dtype, 1)
        idx_one_mask, _ = pto.make_mask(idx_dtype, 1)
        init_val = pad_min(src_dtype) if cmp_mode == "lt" else pad_max(src_dtype)

        for row in range(0, valid_rows, 1):
            remained = valid_cols
            val_acc = pto.vbr(init_val)
            idx_acc = pto.vbr(_scalar_literal(idx_dtype, 0))
            zero_src = pto.vbr(_scalar_literal(src_dtype, 0))

            for col in range(0, valid_cols, lanes):
                mask, remained = pto.make_mask(src_dtype, remained)
                src_addr = pto.addptr(src_ptr, row * src_cols + col)
                value = pto.vlds(src_addr, 0)
                reduced = reduce_op(value, mask)
                val, idx_src = pto.vdintlv(reduced, zero_src)
                idx = pto.vbitcast(idx_src, idx_dtype)
                idx = pto.vadds(idx, col, idx_one_mask)
                cmp = pto.vcmp(val_acc, val, src_one_mask, cmp_mode)
                cmp_idx = pto.pbitcast(cmp, pto.mask_b32)
                val_acc = pto.vsel(val, val_acc, cmp)
                idx_acc = pto.vsel(idx, idx_acc, cmp_idx)

            dst_addr = pto.addptr(dst_ptr, row * dst_cols)
            pto.vsts(idx_acc, dst_addr, 0, idx_one_mask, dist=element_store_dist(idx_dtype))

    return template


__all__ = ["register_row_arg"]
