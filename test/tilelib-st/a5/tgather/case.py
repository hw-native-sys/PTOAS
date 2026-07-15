#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# PTODSL rewrite of test/tilelang_st/npu/a5/src/st/testcase/tgather.


from pathlib import Path
import sys
import zlib

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import auto_main, golden_output_case
from ptodsl import pto


PTO_TO_NP_DTYPE = {
    pto.f32: np.float32,
    pto.f16: np.float16,
    pto.i32: np.int32,
    pto.i16: np.int16,
    pto.i8: np.int8,
    pto.ui32: np.uint32,
    pto.ui16: np.uint16,
    pto.ui8: np.uint8,
}


def npy_dtype(pto_type) -> np.dtype:
    return PTO_TO_NP_DTYPE[pto_type]


# Each case is (name, src_dtype, idx_dtype, src_shape, dst_shape).
# The indices tile has the same shape as dst and contains element
# indices into the flat src tile.
CASE_SHAPES = [
    ("f32_1x128_1x64", pto.f32, pto.i32, (1, 128), (1, 64)),
    ("f32_1x64_1x32", pto.f32, pto.i32, (1, 64), (1, 32)),
    ("f32_i32_32x1024_16x64", pto.f32, pto.i32, (32, 1024), (16, 64)),
    ("i32_i32_32x512_16x256", pto.i32, pto.i32, (32, 512), (16, 256)),
    ("f16_i16_16x1024_16x128", pto.f16, pto.i16, (16, 1024), (16, 128)),
    ("i16_i16_32x256_32x64", pto.i16, pto.i16, (32, 256), (32, 64)),
    ("i8_i16_16x128_16x64", pto.i8, pto.i16, (16, 128), (16, 64)),
    ("i8_ui16_16x128_16x64", pto.i8, pto.ui16, (16, 128), (16, 64)),
    ("ui8_ui16_16x128_16x64", pto.ui8, pto.ui16, (16, 128), (16, 64)),
]

# --- Masked gather cases (axis=0, row) ---
# Row: gather along columns within each row.
# src(R, C_src) -> dst(R, C_dst), C_src >= C_dst (src is expanded, dst is compact)
# (name, dtype, src_shape, dst_shape, pattern, axis)
MASK_CASES_ROW = [
    ("mask_row_f16_16x64_16x64_P1111",   pto.f16, (16, 64),  (16, 64), "P1111", "row"),
    ("mask_row_f32_16x64_16x64_P1111",   pto.f32, (16, 64),  (16, 64), "P1111", "row"),
    ("mask_row_i32_16x64_16x64_P1111",   pto.i32, (16, 64),  (16, 64), "P1111", "row"),
    ("mask_row_i16_16x64_16x64_P1111",   pto.i16, (16, 64),  (16, 64), "P1111", "row"),
    ("mask_row_f16_16x128_16x64_P1010",  pto.f16, (16, 128), (16, 64), "P1010", "row"),
    ("mask_row_f16_16x128_16x64_P0101",  pto.f16, (16, 128), (16, 64), "P0101", "row"),
    ("mask_row_f32_16x128_16x64_P1010",  pto.f32, (16, 128), (16, 64), "P1010", "row"),
    ("mask_row_f32_16x128_16x64_P0101",  pto.f32, (16, 128), (16, 64), "P0101", "row"),
    ("mask_row_i32_16x128_16x64_P1010",  pto.i32, (16, 128), (16, 64), "P1010", "row"),
    ("mask_row_i32_16x128_16x64_P0101",  pto.i32, (16, 128), (16, 64), "P0101", "row"),
    ("mask_row_i16_16x128_16x64_P1010",  pto.i16, (16, 128), (16, 64), "P1010", "row"),
    ("mask_row_i16_16x128_16x64_P0101",  pto.i16, (16, 128), (16, 64), "P0101", "row"),
    ("mask_row_f16_16x256_16x64_P1000",  pto.f16, (16, 256), (16, 64), "P1000", "row"),
    ("mask_row_f16_16x256_16x64_P0100",  pto.f16, (16, 256), (16, 64), "P0100", "row"),
    ("mask_row_f16_16x256_16x64_P0010",  pto.f16, (16, 256), (16, 64), "P0010", "row"),
    ("mask_row_f16_16x256_16x64_P0001",  pto.f16, (16, 256), (16, 64), "P0001", "row"),
    ("mask_row_f32_16x256_16x64_P1000",  pto.f32, (16, 256), (16, 64), "P1000", "row"),
    ("mask_row_f32_16x256_16x64_P0100",  pto.f32, (16, 256), (16, 64), "P0100", "row"),
    ("mask_row_f32_16x256_16x64_P0010",  pto.f32, (16, 256), (16, 64), "P0010", "row"),
    ("mask_row_f32_16x256_16x64_P0001",  pto.f32, (16, 256), (16, 64), "P0001", "row"),
    ("mask_row_i32_16x256_16x64_P1000",  pto.i32, (16, 256), (16, 64), "P1000", "row"),
    ("mask_row_i32_16x256_16x64_P0100",  pto.i32, (16, 256), (16, 64), "P0100", "row"),
    ("mask_row_i32_16x256_16x64_P0010",  pto.i32, (16, 256), (16, 64), "P0010", "row"),
    ("mask_row_i32_16x256_16x64_P0001",  pto.i32, (16, 256), (16, 64), "P0001", "row"),
    ("mask_row_i16_16x256_16x64_P1000",  pto.i16, (16, 256), (16, 64), "P1000", "row"),
    ("mask_row_i16_16x256_16x64_P0100",  pto.i16, (16, 256), (16, 64), "P0100", "row"),
    ("mask_row_i16_16x256_16x64_P0010",  pto.i16, (16, 256), (16, 64), "P0010", "row"),
    ("mask_row_i16_16x256_16x64_P0001",  pto.i16, (16, 256), (16, 64), "P0001", "row"),
]

# --- Masked gather cases (axis=1, col) ---
# Col: gather across rows, keeping columns intact.
# src(R_src, C) -> dst(R_dst, C), R_src >= R_dst (src is expanded, dst is compact)
# (name, dtype, src_shape, dst_shape, pattern, axis)
MASK_CASES_COL = [
    ("mask_col_f16_16x64_16x64_P1111",   pto.f16, (16, 64), (16, 64),  "P1111", "col"),
    ("mask_col_f32_16x64_16x64_P1111",   pto.f32, (16, 64), (16, 64),  "P1111", "col"),
    ("mask_col_i32_16x64_16x64_P1111",   pto.i32, (16, 64), (16, 64),  "P1111", "col"),
    ("mask_col_i16_16x64_16x64_P1111",   pto.i16, (16, 64), (16, 64),  "P1111", "col"),
    ("mask_col_f16_32x64_16x64_P1010",   pto.f16, (32, 64), (16, 64),  "P1010", "col"),
    ("mask_col_f16_32x64_16x64_P0101",   pto.f16, (32, 64), (16, 64),  "P0101", "col"),
    ("mask_col_f32_32x64_16x64_P1010",   pto.f32, (32, 64), (16, 64),  "P1010", "col"),
    ("mask_col_f32_32x64_16x64_P0101",   pto.f32, (32, 64), (16, 64),  "P0101", "col"),
    ("mask_col_i32_32x64_16x64_P1010",   pto.i32, (32, 64), (16, 64),  "P1010", "col"),
    ("mask_col_i32_32x64_16x64_P0101",   pto.i32, (32, 64), (16, 64),  "P0101", "col"),
    ("mask_col_i16_32x64_16x64_P1010",   pto.i16, (32, 64), (16, 64),  "P1010", "col"),
    ("mask_col_i16_32x64_16x64_P0101",   pto.i16, (32, 64), (16, 64),  "P0101", "col"),
    ("mask_col_f16_64x64_16x64_P1000",   pto.f16, (64, 64), (16, 64),  "P1000", "col"),
    ("mask_col_f16_64x64_16x64_P0100",   pto.f16, (64, 64), (16, 64),  "P0100", "col"),
    ("mask_col_f16_64x64_16x64_P0010",   pto.f16, (64, 64), (16, 64),  "P0010", "col"),
    ("mask_col_f16_64x64_16x64_P0001",   pto.f16, (64, 64), (16, 64),  "P0001", "col"),
    ("mask_col_f32_64x64_16x64_P1000",   pto.f32, (64, 64), (16, 64),  "P1000", "col"),
    ("mask_col_f32_64x64_16x64_P0100",   pto.f32, (64, 64), (16, 64),  "P0100", "col"),
    ("mask_col_f32_64x64_16x64_P0010",   pto.f32, (64, 64), (16, 64),  "P0010", "col"),
    ("mask_col_f32_64x64_16x64_P0001",   pto.f32, (64, 64), (16, 64),  "P0001", "col"),
    ("mask_col_i32_64x64_16x64_P1000",   pto.i32, (64, 64), (16, 64),  "P1000", "col"),
    ("mask_col_i32_64x64_16x64_P0100",   pto.i32, (64, 64), (16, 64),  "P0100", "col"),
    ("mask_col_i32_64x64_16x64_P0010",   pto.i32, (64, 64), (16, 64),  "P0010", "col"),
    ("mask_col_i32_64x64_16x64_P0001",   pto.i32, (64, 64), (16, 64),  "P0001", "col"),
    ("mask_col_i16_64x64_16x64_P1000",   pto.i16, (64, 64), (16, 64),  "P1000", "col"),
    ("mask_col_i16_64x64_16x64_P0100",   pto.i16, (64, 64), (16, 64),  "P0100", "col"),
    ("mask_col_i16_64x64_16x64_P0010",   pto.i16, (64, 64), (16, 64),  "P0010", "col"),
    ("mask_col_i16_64x64_16x64_P0001",   pto.i16, (64, 64), (16, 64),  "P0001", "col"),
]

MASK_CASES = MASK_CASES_ROW + MASK_CASES_COL


def _tgather_body(
    src_ptr,
    offset_ptr,
    dst_ptr,
    *,
    src_rows,
    src_cols,
    dst_rows,
    dst_cols,
    src_dtype,
    idx_dtype,
):
    """Shared kernel body for the tgather cases.

    Loads *src* and *offset* tiles from GM, performs gather using
    ``pto.tile.gather``, and stores *dst* back to GM.
    """

    src_view = pto.make_tensor_view(
        src_ptr, shape=[src_rows, src_cols], strides=[src_cols, 1]
    )
    indices_view = pto.make_tensor_view(
        offset_ptr, shape=[dst_rows, dst_cols], strides=[dst_cols, 1]
    )
    dst_view = pto.make_tensor_view(
        dst_ptr, shape=[dst_rows, dst_cols], strides=[dst_cols, 1]
    )

    src_tile = pto.alloc_tile(shape=[src_rows, src_cols], dtype=src_dtype)
    indices_tile = pto.alloc_tile(shape=[dst_rows, dst_cols], dtype=idx_dtype)
    dst_tile = pto.alloc_tile(shape=[dst_rows, dst_cols], dtype=src_dtype)

    pto.tile.load(src_view, src_tile)
    pto.tile.load(indices_view, indices_tile)
    pto.tile.gather(src_tile, dst_tile, indices=indices_tile)
    pto.tile.store(dst_tile, dst_view)


def _tgather_mask_body(src_ptr, dst_ptr, *, src_rows, src_cols,
                       dst_rows, dst_cols, dtype, pattern, axis):
    src_view = pto.make_tensor_view(src_ptr, shape=[src_rows, src_cols],
                                    strides=[src_cols, 1])
    dst_view = pto.make_tensor_view(dst_ptr, shape=[dst_rows, dst_cols],
                                    strides=[dst_cols, 1])

    src_tile = pto.alloc_tile(shape=[src_rows, src_cols], dtype=dtype)
    dst_tile = pto.alloc_tile(shape=[dst_rows, dst_cols], dtype=dtype)

    pto.tile.load(src_view, src_tile)
    pto.tile.gather(src_tile, dst_tile, mask_pattern=pattern, axis=axis)
    pto.tile.store(dst_tile, dst_view)


# One decorated kernel per case, each binding static shapes at definition time.
_tgather_kernels = {}
for _name, _src_dtype, _idx_dtype, _src_shape, _dst_shape in CASE_SHAPES:
    _sr, _sc = _src_shape
    _dr, _dc = _dst_shape

    def _make(
        sr=_sr,
        sc=_sc,
        dr=_dr,
        dc=_dc,
        sdt=_src_dtype,
        idt=_idx_dtype,
        kernel_name=f"tgather_{_name}",
    ):
        @pto.jit(name=kernel_name, target="a5")
        def _kernel(
            src_ptr: pto.ptr(sdt, "gm"),
            offset_ptr: pto.ptr(idt, "gm"),
            dst_ptr: pto.ptr(sdt, "gm"),
        ):
            _tgather_body(
                src_ptr,
                offset_ptr,
                dst_ptr,
                src_rows=sr,
                src_cols=sc,
                dst_rows=dr,
                dst_cols=dc,
                src_dtype=sdt,
                idx_dtype=idt,
            )

        return _kernel

    _tgather_kernels[_name] = _make()


_mask_kernels = {}
for _name, _dtype, _src_shape, _dst_shape, _pattern, _axis in MASK_CASES:
    _sr, _sc = _src_shape
    _dr, _dc = _dst_shape

    def _make_mask(sr=_sr, sc=_sc, dr=_dr, dc=_dc, dtype=_dtype,
                   pattern=_pattern, axis=_axis,
                   kernel_name=f"tgather_{_name}"):
        @pto.jit(name=kernel_name, target="a5")
        def _kernel(
            src_ptr: pto.ptr(dtype, "gm"),
            dst_ptr: pto.ptr(dtype, "gm"),
        ):
            _tgather_mask_body(
                src_ptr, dst_ptr,
                src_rows=sr, src_cols=sc, dst_rows=dr, dst_cols=dc,
                dtype=dtype, pattern=pattern, axis=axis,
            )
        return _kernel

    _mask_kernels[_name] = _make_mask()


def _make_inputs(name, src_dtype, idx_dtype, src_shape, dst_shape):
    src_np = npy_dtype(src_dtype)
    idx_np = npy_dtype(idx_dtype)
    rng_seed = zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF
    rng = np.random.RandomState(rng_seed)

    if np.issubdtype(src_np, np.floating):
        src = rng.uniform(-10.0, 10.0, size=src_shape).astype(src_np)
    else:
        src = rng.randint(-20, 20, size=src_shape).astype(src_np)

    num_src_elements = int(np.prod(src_shape))
    raw_offsets = rng.randint(0, num_src_elements, size=dst_shape).astype(idx_np)

    return [src, raw_offsets]


def _make_expected(src, offsets):
    flat_src = src.reshape(-1)
    dst = np.empty(offsets.shape, dtype=src.dtype)
    flat_offsets = offsets.ravel()
    flat_dst = dst.ravel()
    for i in range(len(flat_dst)):
        flat_dst[i] = flat_src[int(flat_offsets[i])]
    return dst


_ROW_PATTERN_PARAMS = {
    "P0101": (0, 2),
    "P1010": (1, 2),
    "P0001": (0, 4),
    "P0010": (1, 4),
    "P0100": (2, 4),
    "P1000": (3, 4),
    "P1111": (0, 1),
}

_COL_PATTERN_PARAMS = {
    "P0101": (0, 2),
    "P1010": (1, 2),
    "P0001": (0, 4),
    "P0010": (1, 4),
    "P0100": (2, 4),
    "P1000": (3, 4),
    "P1111": (0, 1),
}


def _gather_mask_row_golden(src, dst_rows, dst_cols, pattern):
    offset, stride = _ROW_PATTERN_PARAMS[pattern]
    dst = np.zeros((dst_rows, dst_cols), dtype=src.dtype)
    for i in range(dst_rows):
        for j in range(dst_cols):
            dst[i, j] = src[i, j * stride + offset]
    return dst


def _gather_mask_col_golden(src, dst_rows, dst_cols, pattern):
    start, stride = _COL_PATTERN_PARAMS[pattern]
    dst = np.zeros((dst_rows, dst_cols), dtype=src.dtype)
    for i in range(dst_rows):
        src_row = i * stride + start
        for j in range(dst_cols):
            dst[i, j] = src[src_row, j]
    return dst


def _make_mask_inputs(name, dtype, src_shape):
    np_dt = npy_dtype(dtype)
    rng = np.random.RandomState(zlib.crc32(name.encode()))
    src = rng.uniform(0, 100, src_shape).astype(np_dt)
    return [src]


CASES = []
for _name, _src_dtype, _idx_dtype, _src_shape, _dst_shape in CASE_SHAPES:
    CASES.append(
        golden_output_case(
            "tgather_" + _name,
            _tgather_kernels[_name],
            inputs=lambda _n=_name, _sd=_src_dtype, _id=_idx_dtype, _ss=_src_shape, _ds=_dst_shape: (
                _make_inputs(_n, _sd, _id, _ss, _ds)
            ),
            expected=_make_expected,
            rtol=1e-6,
            atol=1e-6,
        )
    )

for _name, _dtype, _src_shape, _dst_shape, _pattern, _axis in MASK_CASES:
    _golden = (_gather_mask_row_golden if _axis == "row"
               else _gather_mask_col_golden)
    CASES.append(
        golden_output_case(
            "tgather_" + _name,
            _mask_kernels[_name],
            inputs=lambda _n=_name, _d=_dtype, _ss=_src_shape:
                _make_mask_inputs(_n, _d, _ss),
            expected=lambda src, _dr=_dst_shape[0], _dc=_dst_shape[1], _p=_pattern, _g=_golden:
                _g(src, _dr, _dc, _p),
            rtol=1e-6,
            atol=1e-6,
        )
    )


auto_main(globals())
