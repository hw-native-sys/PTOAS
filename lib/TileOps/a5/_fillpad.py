# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Shared basic fill-pad helpers."""

from ptodsl import pto, scalar
import ptodsl.tilelib as tilelib


_DTYPES = [
    ("f16", "f16"),
    ("bf16", "bf16"),
    ("f32", "f32"),
    ("ui8", "ui8"),
    ("si8", "si8"),
    ("i8", "i8"),
    ("ui16", "ui16"),
    ("si16", "si16"),
    ("i16", "i16"),
    ("ui32", "ui32"),
    ("si32", "si32"),
    ("i32", "i32"),
]


def _row_major(src_config, dst_config, src_dtype, dst_dtype, **_):
    return (
        src_config.b_layout == "row_major"
        and dst_config.b_layout == "row_major"
        and src_config.s_layout == "none_box"
        and dst_config.s_layout == "none_box"
        and src_dtype == dst_dtype
    )


def _zero(dtype):
    name = str(dtype)
    if name == "f32":
        return pto.f32(0.0)
    if name == "f16":
        return pto.f16(0.0)
    if name == "bf16":
        return pto.bf16(0.0)
    if name in {"ui32", "si32", "i32"}:
        return pto.i32(0)
    if name in {"ui16", "si16", "i16"}:
        return pto.i16(0)
    return pto.i8(0)


def _fill_scalar(dst):
    dtype = dst.dtype
    pad_value = str(getattr(dst, "pad_value", "Null")).lower()
    if str(dtype) == "f32" and pad_value in {"zero", "0x1", "0x01"}:
        return pto.f32(0.0)
    if pad_value in {"max", "0x2", "0x02"}:
        return _max(dtype)
    if pad_value in {"min", "0x3", "0x03"}:
        return _min(dtype)
    return _zero(dtype)


def _max(dtype):
    name = str(dtype)
    if name == "f32":
        return pto.f32(3.4028234663852886e38)
    if name == "f16":
        return pto.f16(65504.0)
    if name == "bf16":
        return pto.bf16(3.3895313892515355e38)
    if name == "ui32":
        return pto.i32(-1)
    if name in {"si32", "i32"}:
        return pto.i32(2147483647)
    if name == "ui16":
        return pto.i16(-1)
    if name in {"si16", "i16"}:
        return pto.i16(32767)
    if name == "ui8":
        return pto.i8(-1)
    return pto.i8(127)


def _min(dtype):
    name = str(dtype)
    if name == "f32":
        return pto.f32(-3.4028234663852886e38)
    if name == "f16":
        return pto.f16(-65504.0)
    if name == "bf16":
        return pto.bf16(-3.3895313892515355e38)
    if name in {"ui32", "ui16", "ui8"}:
        return _zero(dtype)
    if name in {"si32", "i32"}:
        return pto.i32(-2147483648)
    if name in {"si16", "i16"}:
        return pto.i16(-32768)
    return pto.i8(-128)


def _copy_region(src, dst, valid_rows, col_start, col_stop):
    dtype = dst.dtype
    lanes = pto.elements_per_vreg(dtype)
    with pto.for_(0, valid_rows, step=1) as row:
        remained = col_stop - col_start
        col_loop = pto.for_(col_start, col_stop, step=lanes).carry(remained=remained)
        with col_loop:
            col = col_loop.iv
            mask, remained = pto.make_mask(dtype, remained)
            data = pto.vlds(src[row, col:])
            pto.vsts(data, dst[row, col:], mask)
            col_loop.update(remained=remained)


def _scalar_tail_start(dst, lanes):
    if len(dst.shape) < 2:
        return None
    cols = dst.shape[1]
    if cols <= lanes or cols % lanes == 0:
        return None
    return (cols // lanes) * lanes


def _fill(dst, row_start, row_stop, col_start, col_stop, scalar_tail_start=None):
    dtype = dst.dtype
    lanes = pto.elements_per_vreg(dtype)
    cols = dst.shape[1]
    dst_ptr = dst.as_ptr()
    fill_scalar = _fill_scalar(dst)
    vector_col_stop = scalar_tail_start if scalar_tail_start is not None else col_stop
    with pto.for_(row_start, row_stop, step=1) as row:
        remained = vector_col_stop - col_start
        col_loop = pto.for_(col_start, vector_col_stop, step=lanes).carry(remained=remained)
        with col_loop:
            col = col_loop.iv
            mask, remained = pto.make_mask(dtype, remained)
            vec = pto.vdup(fill_scalar, mask)
            addr = pto.addptr(dst_ptr, row * cols + col)
            pto.vsts(vec, addr, 0, mask)
            col_loop.update(remained=remained)
        if scalar_tail_start is not None:
            with pto.for_(scalar_tail_start, col_stop, step=1) as col:
                scalar.store(fill_scalar, dst[row, col])


def _fill_inplace(dst, src_valid_rows, src_valid_cols, dst_valid_rows, dst_valid_cols):
    _fill(dst, 0, src_valid_rows, src_valid_cols, dst_valid_cols)
    _fill(dst, src_valid_rows, dst_valid_rows, 0, dst_valid_cols)


def register_fillpad(*, op, name, copy):
    @tilelib.tile_template(
        op=op,
        target="a5",
        name=name,
        dtypes=_DTYPES,
        iteration_axis="none",
        op_engine="other",
        op_class="movement",
        constraints=[_row_major],
        id=0,
        loop_depth=2,
        is_post_update=False,
        tags=("fillpad",),
    )
    def template(src: pto.Tile, dst: pto.Tile):
        src_valid_rows, src_valid_cols = src.valid_shape
        dst_valid_rows, dst_valid_cols = dst.valid_shape
        lanes = pto.elements_per_vreg(dst.dtype)
        aligned_cols = (src_valid_cols // lanes) * lanes
        if not copy:
            _fill_inplace(dst, src_valid_rows, src_valid_cols, dst_valid_rows, dst_valid_cols)
            return
        if copy:
            _copy_region(src, dst, src_valid_rows, 0, aligned_cols)
        fill_row_stop = dst_valid_rows if op == "pto.tfillpad_expand" else src_valid_rows
        scalar_tail_start = _scalar_tail_start(dst, lanes)
        _fill(dst, 0, fill_row_stop, aligned_cols, dst_valid_cols, scalar_tail_start=scalar_tail_start)
        if copy:
            _copy_region(src, dst, src_valid_rows, aligned_cols, src_valid_cols)
        _fill(dst, src_valid_rows, dst_valid_rows, 0, dst_valid_cols, scalar_tail_start=scalar_tail_start)

    return template


__all__ = ["register_fillpad"]
