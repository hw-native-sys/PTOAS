#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# PTODSL rewrite of pto-isa/tests/npu/a5/src/st/testcase/tscatter.
#
# TSCATTER scatters elements from a source tile into a destination tile.
# Two modes are supported:
#   - Indexed scatter: each element of src is written to dst at the position
#     specified by the corresponding index value.
#   - Masked scatter: src elements are placed into dst at positions determined
#     by a mask pattern (e.g., P0101 interleaves with zeros at 2x expansion).
#     Both row and col directions are tested.

from pathlib import Path
import sys
import zlib

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import auto_main, golden_output_case
from ptodsl import pto


PTO_TO_NP_DTYPE = {
    pto.f32:  np.float32,
    pto.f16:  np.float16,
    pto.i32:  np.int32,
    pto.i16:  np.int16,
    pto.i8:   np.int8,
    pto.ui32: np.uint32,
    pto.ui16: np.uint16,
    pto.ui8:  np.uint8,
}


def npy_dtype(pto_type) -> np.dtype:
    return PTO_TO_NP_DTYPE[pto_type]


# --- Indexed scatter cases ---
# (name, src_dtype, idx_dtype, src_shape, idx_shape)
INDEX_CASES = [
    ("i8_uint16_256x32_256x32",    pto.i8,   pto.ui16, (256, 32), (256, 32)),
    ("ui8_int16_256x32_256x32",    pto.ui8,  pto.i16,  (256, 32), (256, 32)),
    ("i32_uint32_31x128_31x128",   pto.i32,  pto.ui32, (31, 128), (31, 128)),
    ("ui32_int32_32x64_32x64",     pto.ui32, pto.i32,  (32, 64),  (32, 64)),
    ("f32_uint32_7x448_7x448",     pto.f32,  pto.ui32,  (7, 448),  (7, 448)),
    ("f32_uint32_32x64_32x64",     pto.f32,  pto.ui32, (32, 64),  (32, 64)),
    ("i16_uint16_15x192_15x192",   pto.i16,  pto.ui16, (15, 192), (15, 192)),
    ("ui16_int16_63x64_63x64",     pto.ui16, pto.i16,  (63, 64),  (63, 64)),
    ("f16_uint16_63x64_63x64",     pto.f16,  pto.ui16, (63, 64),  (63, 64)),
]

# --- Masked scatter cases (axis=0, row) ---
# Row: scatter along columns within each row.
# src(R, C_src) -> dst(R, C_dst), C_dst >= C_src
# (name, dtype, src_shape, dst_shape, pattern, axis)
MASK_CASES_ROW = [
    ("mask_row_f16_16x64_16x64_P1111",   pto.f16, (16, 64), (16, 64),  "P1111", "row"),
    ("mask_row_f32_16x64_16x64_P1111",   pto.f32, (16, 64), (16, 64),  "P1111", "row"),
    ("mask_row_i32_16x64_16x64_P1111",   pto.i32, (16, 64), (16, 64),  "P1111", "row"),
    ("mask_row_i16_16x64_16x64_P1111",   pto.i16, (16, 64), (16, 64),  "P1111", "row"),
    ("mask_row_f16_16x64_16x128_P1010",  pto.f16, (16, 64), (16, 128), "P1010", "row"),
    ("mask_row_f16_16x64_16x128_P0101",  pto.f16, (16, 64), (16, 128), "P0101", "row"),
    ("mask_row_f32_16x64_16x128_P1010",  pto.f32, (16, 64), (16, 128), "P1010", "row"),
    ("mask_row_f32_16x64_16x128_P0101",  pto.f32, (16, 64), (16, 128), "P0101", "row"),
    ("mask_row_i32_16x64_16x128_P1010",  pto.i32, (16, 64), (16, 128), "P1010", "row"),
    ("mask_row_i32_16x64_16x128_P0101",  pto.i32, (16, 64), (16, 128), "P0101", "row"),
    ("mask_row_i16_16x64_16x128_P1010",  pto.i16, (16, 64), (16, 128), "P1010", "row"),
    ("mask_row_i16_16x64_16x128_P0101",  pto.i16, (16, 64), (16, 128), "P0101", "row"),
    ("mask_row_f16_16x64_16x256_P1000",  pto.f16, (16, 64), (16, 256), "P1000", "row"),
    ("mask_row_f16_16x64_16x256_P0100",  pto.f16, (16, 64), (16, 256), "P0100", "row"),
    ("mask_row_f16_16x64_16x256_P0010",  pto.f16, (16, 64), (16, 256), "P0010", "row"),
    ("mask_row_f16_16x64_16x256_P0001",  pto.f16, (16, 64), (16, 256), "P0001", "row"),
    ("mask_row_f32_16x64_16x256_P1000",  pto.f32, (16, 64), (16, 256), "P1000", "row"),
    ("mask_row_f32_16x64_16x256_P0100",  pto.f32, (16, 64), (16, 256), "P0100", "row"),
    ("mask_row_f32_16x64_16x256_P0010",  pto.f32, (16, 64), (16, 256), "P0010", "row"),
    ("mask_row_f32_16x64_16x256_P0001",  pto.f32, (16, 64), (16, 256), "P0001", "row"),
    ("mask_row_i32_16x64_16x256_P1000",  pto.i32, (16, 64), (16, 256), "P1000", "row"),
    ("mask_row_i32_16x64_16x256_P0100",  pto.i32, (16, 64), (16, 256), "P0100", "row"),
    ("mask_row_i32_16x64_16x256_P0010",  pto.i32, (16, 64), (16, 256), "P0010", "row"),
    ("mask_row_i32_16x64_16x256_P0001",  pto.i32, (16, 64), (16, 256), "P0001", "row"),
    ("mask_row_i16_16x64_16x256_P1000",  pto.i16, (16, 64), (16, 256), "P1000", "row"),
    ("mask_row_i16_16x64_16x256_P0100",  pto.i16, (16, 64), (16, 256), "P0100", "row"),
    ("mask_row_i16_16x64_16x256_P0010",  pto.i16, (16, 64), (16, 256), "P0010", "row"),
    ("mask_row_i16_16x64_16x256_P0001",  pto.i16, (16, 64), (16, 256), "P0001", "row"),
]

# --- Masked scatter cases (axis=1, col) ---
# Col: scatter across rows, keeping columns intact.
# src(R_src, C) -> dst(R_dst, C), R_dst >= R_src
# (name, dtype, src_shape, dst_shape, pattern, axis)
MASK_CASES_COL = [
    ("mask_col_f16_16x64_16x64_P1111",   pto.f16, (16, 64), (16, 64),  "P1111", "col"),
    ("mask_col_f32_16x64_16x64_P1111",   pto.f32, (16, 64), (16, 64),  "P1111", "col"),
    ("mask_col_i32_16x64_16x64_P1111",   pto.i32, (16, 64), (16, 64),  "P1111", "col"),
    ("mask_col_i16_16x64_16x64_P1111",   pto.i16, (16, 64), (16, 64),  "P1111", "col"),
    ("mask_col_f16_16x64_32x64_P1010",   pto.f16, (16, 64), (32, 64),  "P1010", "col"),
    ("mask_col_f16_16x64_32x64_P0101",   pto.f16, (16, 64), (32, 64),  "P0101", "col"),
    ("mask_col_f32_16x64_32x64_P1010",   pto.f32, (16, 64), (32, 64),  "P1010", "col"),
    ("mask_col_f32_16x64_32x64_P0101",   pto.f32, (16, 64), (32, 64),  "P0101", "col"),
    ("mask_col_i32_16x64_32x64_P1010",   pto.i32, (16, 64), (32, 64),  "P1010", "col"),
    ("mask_col_i32_16x64_32x64_P0101",   pto.i32, (16, 64), (32, 64),  "P0101", "col"),
    ("mask_col_i16_16x64_32x64_P1010",   pto.i16, (16, 64), (32, 64),  "P1010", "col"),
    ("mask_col_i16_16x64_32x64_P0101",   pto.i16, (16, 64), (32, 64),  "P0101", "col"),
    ("mask_col_f16_16x64_64x64_P1000",   pto.f16, (16, 64), (64, 64),  "P1000", "col"),
    ("mask_col_f16_16x64_64x64_P0100",   pto.f16, (16, 64), (64, 64),  "P0100", "col"),
    ("mask_col_f16_16x64_64x64_P0010",   pto.f16, (16, 64), (64, 64),  "P0010", "col"),
    ("mask_col_f16_16x64_64x64_P0001",   pto.f16, (16, 64), (64, 64),  "P0001", "col"),
    ("mask_col_f32_16x64_64x64_P1000",   pto.f32, (16, 64), (64, 64),  "P1000", "col"),
    ("mask_col_f32_16x64_64x64_P0100",   pto.f32, (16, 64), (64, 64),  "P0100", "col"),
    ("mask_col_f32_16x64_64x64_P0010",   pto.f32, (16, 64), (64, 64),  "P0010", "col"),
    ("mask_col_f32_16x64_64x64_P0001",   pto.f32, (16, 64), (64, 64),  "P0001", "col"),
    ("mask_col_i32_16x64_64x64_P1000",   pto.i32, (16, 64), (64, 64),  "P1000", "col"),
    ("mask_col_i32_16x64_64x64_P0100",   pto.i32, (16, 64), (64, 64),  "P0100", "col"),
    ("mask_col_i32_16x64_64x64_P0010",   pto.i32, (16, 64), (64, 64),  "P0010", "col"),
    ("mask_col_i32_16x64_64x64_P0001",   pto.i32, (16, 64), (64, 64),  "P0001", "col"),
    ("mask_col_i16_16x64_64x64_P1000",   pto.i16, (16, 64), (64, 64),  "P1000", "col"),
    ("mask_col_i16_16x64_64x64_P0100",   pto.i16, (16, 64), (64, 64),  "P0100", "col"),
    ("mask_col_i16_16x64_64x64_P0010",   pto.i16, (16, 64), (64, 64),  "P0010", "col"),
    ("mask_col_i16_16x64_64x64_P0001",   pto.i16, (16, 64), (64, 64),  "P0001", "col"),
]

MASK_CASES = MASK_CASES_ROW + MASK_CASES_COL


# ---------------------------------------------------------------------------
# Kernel bodies
# ---------------------------------------------------------------------------

def _tscatter_index_body(src_ptr, idx_ptr, dst_ptr, *, src_rows, src_cols,
                         idx_rows, idx_cols, dtype, idx_dtype):
    src_view = pto.make_tensor_view(src_ptr, shape=[src_rows, src_cols],
                                    strides=[src_cols, 1])
    idx_view = pto.make_tensor_view(idx_ptr, shape=[idx_rows, idx_cols],
                                    strides=[idx_cols, 1])
    dst_view = pto.make_tensor_view(dst_ptr, shape=[src_rows, src_cols],
                                    strides=[src_cols, 1])

    src_tile = pto.alloc_tile(shape=[src_rows, src_cols], dtype=dtype)
    idx_tile = pto.alloc_tile(shape=[idx_rows, idx_cols], dtype=idx_dtype)
    dst_tile = pto.alloc_tile(shape=[src_rows, src_cols], dtype=dtype)

    pto.tile.load(src_view, src_tile)
    pto.tile.load(idx_view, idx_tile)
    pto.tile.scatter(src_tile, dst_tile, indexes=idx_tile)
    pto.tile.store(dst_tile, dst_view)


def _tscatter_mask_body(src_ptr, dst_ptr, *, src_rows, src_cols,
                        dst_rows, dst_cols, dtype, pattern, axis):
    src_view = pto.make_tensor_view(src_ptr, shape=[src_rows, src_cols],
                                    strides=[src_cols, 1])
    dst_view = pto.make_tensor_view(dst_ptr, shape=[dst_rows, dst_cols],
                                    strides=[dst_cols, 1])

    src_tile = pto.alloc_tile(shape=[src_rows, src_cols], dtype=dtype)
    dst_tile = pto.alloc_tile(shape=[dst_rows, dst_cols], dtype=dtype)

    pto.tile.load(src_view, src_tile)
    pto.tile.scatter(src_tile, dst_tile, mask_pattern=pattern,
                     axis=axis)
    pto.tile.store(dst_tile, dst_view)


# ---------------------------------------------------------------------------
# One @pto.jit kernel per case variant
# ---------------------------------------------------------------------------

_index_kernels = {}
for _name, _src_dtype, _idx_dtype, _src_shape, _idx_shape in INDEX_CASES:
    _sr, _sc = _src_shape
    _ir, _ic = _idx_shape

    def _make_idx(sr=_sr, sc=_sc, ir=_ir, ic=_ic, src_dtype=_src_dtype,
                  idx_dtype=_idx_dtype, kernel_name=f"tscatter_{_name}"):
        @pto.jit(name=kernel_name, target="a5")
        def _kernel(
            src_ptr: pto.ptr(src_dtype, "gm"),
            idx_ptr: pto.ptr(idx_dtype, "gm"),
            dst_ptr: pto.ptr(src_dtype, "gm"),
        ):
            _tscatter_index_body(
                src_ptr, idx_ptr, dst_ptr,
                src_rows=sr, src_cols=sc, idx_rows=ir, idx_cols=ic,
                dtype=src_dtype, idx_dtype=idx_dtype,
            )
        return _kernel

    _index_kernels[_name] = _make_idx()


_mask_kernels = {}
for _name, _dtype, _src_shape, _dst_shape, _pattern, _axis in MASK_CASES:
    _sr, _sc = _src_shape
    _dr, _dc = _dst_shape

    def _make_mask(sr=_sr, sc=_sc, dr=_dr, dc=_dc, dtype=_dtype,
                   pattern=_pattern, axis=_axis,
                   kernel_name=f"tscatter_{_name}"):
        @pto.jit(name=kernel_name, target="a5")
        def _kernel(
            src_ptr: pto.ptr(dtype, "gm"),
            dst_ptr: pto.ptr(dtype, "gm"),
        ):
            _tscatter_mask_body(
                src_ptr, dst_ptr,
                src_rows=sr, src_cols=sc, dst_rows=dr, dst_cols=dc,
                dtype=dtype, pattern=pattern, axis=axis,
            )
        return _kernel

    _mask_kernels[_name] = _make_mask()


# ---------------------------------------------------------------------------
# Input generators and golden functions
# ---------------------------------------------------------------------------

def _scatter_index_golden(src, indices):
    dst = np.zeros_like(src).flatten()
    flat_src = src.flatten()
    flat_idx = indices.flatten()
    for i in range(len(flat_idx)):
        dst[int(flat_idx[i])] = flat_src[i]
    return dst.reshape(src.shape)


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


def _scatter_mask_row_golden(src, dst_rows, dst_cols, pattern):
    offset, stride = _ROW_PATTERN_PARAMS[pattern]
    dst = np.zeros((dst_rows, dst_cols), dtype=src.dtype)
    src_rows, src_cols = src.shape
    for i in range(src_rows):
        for j in range(src_cols):
            dst[i, j * stride + offset] = src[i, j]
    return dst


def _scatter_mask_col_golden(src, dst_rows, dst_cols, pattern):
    start, stride = _COL_PATTERN_PARAMS[pattern]
    dst = np.zeros((dst_rows, dst_cols), dtype=src.dtype)
    src_rows, src_cols = src.shape
    for i in range(src_rows):
        dst_row = i * stride + start
        for j in range(src_cols):
            dst[dst_row, j] = src[i, j]
    return dst


def _make_index_inputs(name, src_dtype, idx_dtype, src_shape, idx_shape):
    np_dt = npy_dtype(src_dtype)
    np_idx_dt = npy_dtype(idx_dtype)
    rng = np.random.RandomState(zlib.crc32(name.encode()))

    src = rng.uniform(0, 100, src_shape).astype(np_dt)
    raw_idx = rng.randint(0, 2, idx_shape).astype(np_idx_dt)
    cols = src_shape[1]
    indices = np.empty_like(raw_idx)
    for row in range(raw_idx.shape[0]):
        for col in range(raw_idx.shape[1]):
            indices[row, col] = raw_idx[row, col] * cols + col

    return [src, indices]


def _make_mask_inputs(name, dtype, src_shape):
    np_dt = npy_dtype(dtype)
    rng = np.random.RandomState(zlib.crc32(name.encode()))
    src = rng.uniform(0, 100, src_shape).astype(np_dt)
    return [src]


# ---------------------------------------------------------------------------
# Build CASES
# ---------------------------------------------------------------------------

CASES = []

for _name, _src_dtype, _idx_dtype, _src_shape, _idx_shape in INDEX_CASES:
    CASES.append(
        golden_output_case(
            "tscatter_" + _name,
            _index_kernels[_name],
            inputs=lambda _n=_name, _sd=_src_dtype, _id=_idx_dtype, _ss=_src_shape, _is=_idx_shape:
                _make_index_inputs(_n, _sd, _id, _ss, _is),
            expected=lambda src, idx: _scatter_index_golden(src, idx),
            rtol=1e-6,
            atol=1e-6,
        )
    )

for _name, _dtype, _src_shape, _dst_shape, _pattern, _axis in MASK_CASES:
    _golden = (_scatter_mask_row_golden if _axis == "row"
               else _scatter_mask_col_golden)
    CASES.append(
        golden_output_case(
            "tscatter_" + _name,
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
