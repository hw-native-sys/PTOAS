#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# PTODSL rewrite of pto-isa/tests/npu/a5/src/st/testcase/mscatter.
#
# MSCATTER scatters elements from source tiles into a destination table in
# global memory. Two coalesce modes are supported:
#   - Row: each row of src is written to table[row_idx[i]]
#   - Elem: each element of src is written to table[idx[i]]
# Atomic operations (add, max, min) and OOB handling (skip, clamp, wrap) are
# also tested.

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


# ---------------------------------------------------------------------------
# Kernel bodies
# ---------------------------------------------------------------------------

def _mscatter_row_body(src_ptr, idx_ptr, table_ptr, *, src_rows, src_cols,
                       table_rows, dtype, idx_dtype, atomic, oob):
    src_view = pto.make_tensor_view(src_ptr, shape=[src_rows, src_cols],
                                    strides=[src_cols, 1])
    idx_view = pto.make_tensor_view(idx_ptr, shape=[1, src_rows],
                                    strides=[src_rows, 1])
    table_view = pto.make_tensor_view(table_ptr, shape=[table_rows, src_cols],
                                      strides=[src_cols, 1])

    src_tile = pto.alloc_tile(shape=[src_rows, src_cols], dtype=dtype)
    idx_tile = pto.alloc_tile(shape=[1, src_rows], dtype=idx_dtype)

    pto.tile.load(src_view, src_tile)
    pto.tile.load(idx_view, idx_tile)

    pto.tile.mscatter(src_tile, idx_tile, table_view,
                      coalesce="row", atomic=atomic, oob=oob)


def _mscatter_elem_body(src_ptr, idx_ptr, table_ptr, *, src_rows, src_cols,
                        table_size, dtype, idx_dtype, atomic, oob):
    src_view = pto.make_tensor_view(src_ptr, shape=[src_rows, src_cols],
                                    strides=[src_cols, 1])
    idx_view = pto.make_tensor_view(idx_ptr, shape=[src_rows, src_cols],
                                    strides=[src_cols, 1])
    table_view = pto.make_tensor_view(table_ptr, shape=[1, table_size],
                                      strides=[table_size, 1])

    src_tile = pto.alloc_tile(shape=[src_rows, src_cols], dtype=dtype)
    idx_tile = pto.alloc_tile(shape=[src_rows, src_cols], dtype=idx_dtype)

    pto.tile.load(src_view, src_tile)
    pto.tile.load(idx_view, idx_tile)

    pto.tile.mscatter(src_tile, idx_tile, table_view,
                      coalesce="elem", atomic=atomic, oob=oob)


# ---------------------------------------------------------------------------
# Case definitions
# ---------------------------------------------------------------------------

# (name, dtype, idx_dtype, src_shape, table_rows_or_size, coalesce, atomic, oob)
CASES_DEF = [
    ("row_float_random_8x32_64rows", pto.f32, pto.i32, (8, 32), 64, "row", "none", "undefined"),
    ("row_float_same_8x32_16rows", pto.f32, pto.i32, (8, 32), 16, "row", "none", "undefined"),
    ("row_half_random_16x64_64rows", pto.f16, pto.i32, (16, 64), 64, "row", "none", "undefined"),
    ("row_int32_random_8x16_32rows", pto.i32, pto.i32, (8, 16), 32, "row", "none", "undefined"),
    ("row_uint8_random_8x32_32rows", pto.ui8, pto.i32, (8, 32), 32, "row", "none", "undefined"),
    ("row_int16_random_8x16_32rows", pto.i16, pto.i32, (8, 16), 32, "row", "none", "undefined"),
    ("row_float_atomicadd_8x32_8rows", pto.f32, pto.i32, (8, 32), 8, "row", "add", "undefined"),
    ("row_float_skip_8x32_8rows", pto.f32, pto.i32, (8, 32), 8, "row", "none", "skip"),
    ("row_int32_clamp_8x16_8rows", pto.i32, pto.i32, (8, 16), 8, "row", "none", "clamp"),
    ("row_half_wrap_8x32_8rows", pto.f16, pto.i32, (8, 32), 8, "row", "none", "wrap"),
    ("row_int8_random_8x32_32rows", pto.i8, pto.i32, (8, 32), 32, "row", "none", "undefined"),
    ("row_uint16_random_8x16_32rows", pto.ui16, pto.i32, (8, 16), 32, "row", "none", "undefined"),
    ("row_uint32_random_8x16_32rows", pto.ui32, pto.i32, (8, 16), 32, "row", "none", "undefined"),
    ("row_int32_atomicmax_8x16_8rows", pto.i32, pto.i32, (8, 16), 8, "row", "max", "undefined"),
    ("row_float_atomicmin_8x32_8rows", pto.f32, pto.i32, (8, 32), 8, "row", "min", "undefined"),
    ("row_int32_wrap_8x16_8rows", pto.i32, pto.i32, (8, 16), 8, "row", "none", "wrap"),
    ("row_half_atomicadd_8x32_8rows", pto.f16, pto.i32, (8, 32), 8, "row", "add", "undefined"),
    ("row_float_uint32idx_8x32_64rows", pto.f32, pto.ui32, (8, 32), 64, "row", "none", "undefined"),

    ("elem_float_random_64_128size", pto.f32, pto.i32, (1, 64), 128, "elem", "none", "undefined"),
    ("elem_float_same_64_8size", pto.f32, pto.i32, (1, 64), 8, "elem", "none", "undefined"),
    ("elem_float_seq_32_32size", pto.f32, pto.i32, (1, 32), 32, "elem", "none", "undefined"),
    ("elem_half_random_64_128size", pto.f16, pto.i32, (1, 64), 128, "elem", "none", "undefined"),
    ("elem_int32_random_32_64size", pto.i32, pto.i32, (1, 32), 64, "elem", "none", "undefined"),
    ("elem_uint8_random_64_128size", pto.ui8, pto.i32, (1, 64), 128, "elem", "none", "undefined"),
    ("elem_int16_random_32_64size", pto.i16, pto.i32, (1, 32), 64, "elem", "none", "undefined"),
    ("elem_float_atomicadd_32_32size", pto.f32, pto.i32, (1, 32), 32, "elem", "add", "undefined"),
    ("elem_int32_atomicadd_skip_32_16size", pto.i32, pto.i32, (1, 32), 16, "elem", "add", "skip"),
    ("elem_float_skip_32_16size", pto.f32, pto.i32, (1, 32), 16, "elem", "none", "skip"),
    ("elem_int32_clamp_32_16size", pto.i32, pto.i32, (1, 32), 16, "elem", "none", "clamp"),
    ("elem_half_wrap_32_16size", pto.f16, pto.i32, (1, 32), 16, "elem", "none", "wrap"),
    ("elem_float_small_16_32size", pto.f32, pto.i32, (1, 16), 32, "elem", "none", "undefined"),
    ("elem_int32_atomicmax_random_32_32size", pto.i32, pto.i32, (1, 32), 32, "elem", "max", "undefined"),
    ("elem_float_atomicmin_random_32_32size", pto.f32, pto.i32, (1, 32), 32, "elem", "min", "undefined"),
    ("elem_uint8_wrap_64_16size", pto.ui8, pto.i32, (1, 64), 16, "elem", "none", "wrap"),
    ("elem_int16_clamp_32_16size", pto.i16, pto.i32, (1, 32), 16, "elem", "none", "clamp"),
    ("elem_int8_random_32_64size", pto.i8, pto.i32, (1, 32), 64, "elem", "none", "undefined"),
    ("elem_uint16_random_32_64size", pto.ui16, pto.i32, (1, 32), 64, "elem", "none", "undefined"),
    ("elem_uint32_random_32_64size", pto.ui32, pto.i32, (1, 32), 64, "elem", "none", "undefined"),
    ("elem_int32_wrap_32_16size", pto.i32, pto.i32, (1, 32), 16, "elem", "none", "wrap"),
    ("elem_half_skip_32_16size", pto.f16, pto.i32, (1, 32), 16, "elem", "none", "skip"),
    ("elem_float_wrap_32_16size", pto.f32, pto.i32, (1, 32), 16, "elem", "none", "wrap"),
    ("elem_half_atomicadd_32_32size", pto.f16, pto.i32, (1, 32), 32, "elem", "add", "undefined"),
    ("elem_float_uint32idx_32_32size", pto.f32, pto.ui32, (1, 32), 32, "elem", "none", "undefined"),

    ("elem2d_float_8x32_random_256size", pto.f32, pto.i32, (8, 32), 256, "elem", "none", "undefined"),
    ("elem2d_int32_8x16_random_256size", pto.i32, pto.i32, (8, 16), 256, "elem", "none", "undefined"),
    ("elem2d_half_4x32_random_256size", pto.f16, pto.i32, (4, 32), 256, "elem", "none", "undefined"),
    ("elem2d_int32_unaligned_3x8_64size", pto.i32, pto.i32, (3, 8), 64, "elem", "none", "undefined"),
    ("elem2d_uint8_unaligned_3x32_256size", pto.ui8, pto.i32, (3, 32), 256, "elem", "none", "undefined"),
    ("elem2d_int8_4x32_random_256size", pto.i8, pto.i32, (4, 32), 256, "elem", "none", "undefined"),
    ("elem2d_int16_4x16_random_128size", pto.i16, pto.i32, (4, 16), 128, "elem", "none", "undefined"),
    ("elem2d_uint16_4x16_random_128size", pto.ui16, pto.i32, (4, 16), 128, "elem", "none", "undefined"),
    ("elem2d_uint32_4x16_random_128size", pto.ui32, pto.i32, (4, 16), 128, "elem", "none", "undefined"),
]


# ---------------------------------------------------------------------------
# One @pto.jit kernel per case variant
# ---------------------------------------------------------------------------

_kernels = {}
for _name, _dtype, _idx_dtype, _src_shape, _table_size, _coalesce, _atomic, _oob in CASES_DEF:
    _sr, _sc = _src_shape

    def _make(sr=_sr, sc=_sc, ts=_table_size, dtype=_dtype, idx_dtype=_idx_dtype,
              coalesce=_coalesce, atomic=_atomic, oob=_oob,
              kernel_name=f"mscatter_{_name}"):
        if coalesce == "row":
            @pto.jit(name=kernel_name, target="a5")
            def _kernel(
                src_ptr: pto.ptr(dtype, "gm"),
                idx_ptr: pto.ptr(idx_dtype, "gm"),
                table_ptr: pto.ptr(dtype, "gm"),
            ):
                _mscatter_row_body(
                    src_ptr, idx_ptr, table_ptr,
                    src_rows=sr, src_cols=sc, table_rows=ts,
                    dtype=dtype, idx_dtype=idx_dtype,
                    atomic=atomic, oob=oob,
                )
            return _kernel
        else:
            @pto.jit(name=kernel_name, target="a5")
            def _kernel(
                src_ptr: pto.ptr(dtype, "gm"),
                idx_ptr: pto.ptr(idx_dtype, "gm"),
                table_ptr: pto.ptr(dtype, "gm"),
            ):
                _mscatter_elem_body(
                    src_ptr, idx_ptr, table_ptr,
                    src_rows=sr, src_cols=sc, table_size=ts,
                    dtype=dtype, idx_dtype=idx_dtype,
                    atomic=atomic, oob=oob,
                )
            return _kernel

    _kernels[_name] = _make()


# ---------------------------------------------------------------------------
# Input generators and golden functions
# ---------------------------------------------------------------------------

def _make_src(dtype, count, start=1):
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        mod = min(info.max - info.min + 1, 251)
    else:
        mod = 251
    arr = (np.arange(start, start + count) % mod) + 1
    return arr.astype(dtype)


def _resolve(raw, table_size, oob):
    if oob == "skip":
        return (raw, raw >= table_size)
    if oob == "clamp":
        return (min(raw, table_size - 1) if raw >= 0 else 0, False)
    if oob == "wrap":
        return (raw % table_size, False)
    return (raw, False)


def _golden_row(src, idx, table_rows, table_cols, atomic, oob):
    n_rows = src.shape[0]
    table = np.zeros((table_rows, table_cols), dtype=src.dtype)
    for i in range(n_rows):
        raw = int(idx.reshape(-1)[i])
        safe, skip = _resolve(raw, table_rows, oob)
        if skip:
            continue
        if atomic == "none":
            overridden = False
            for j in range(i + 1, n_rows):
                raw2 = int(idx.reshape(-1)[j])
                safe2, skip2 = _resolve(raw2, table_rows, oob)
                if not skip2 and safe2 == safe:
                    overridden = True
                    break
            if overridden:
                continue
        if atomic == "add":
            table[safe, :] = src.dtype.type(table[safe, :] + src[i, :])
        elif atomic == "max":
            table[safe, :] = np.maximum(table[safe, :], src[i, :])
        elif atomic == "min":
            table[safe, :] = np.minimum(table[safe, :], src[i, :])
        else:
            table[safe, :] = src[i, :]
    return table


def _golden_elem(src, idx, table_size, atomic, oob):
    n = src.shape[0] * src.shape[1]
    src_flat = src.reshape(n)
    idx_flat = idx.reshape(n)
    table = np.zeros(table_size, dtype=src.dtype)
    for i in range(n):
        raw = int(idx_flat[i])
        safe, skip = _resolve(raw, table_size, oob)
        if skip:
            continue
        if atomic == "none":
            overridden = False
            for j in range(i + 1, n):
                raw2 = int(idx_flat[j])
                safe2, skip2 = _resolve(raw2, table_size, oob)
                if not skip2 and safe2 == safe:
                    overridden = True
                    break
            if overridden:
                continue
        if atomic == "add":
            table[safe] = src.dtype.type(table[safe] + src_flat[i])
        elif atomic == "max":
            table[safe] = max(table[safe], src_flat[i])
        elif atomic == "min":
            table[safe] = min(table[safe], src_flat[i])
        else:
            table[safe] = src_flat[i]
    return table


def _make_row_idx(rng, r, table_size, oob, name, np_idx_dt):
    if oob in ("skip", "clamp", "wrap"):
        arr = rng.integers(low=0, high=table_size, size=r, dtype=np.int32)
        pos = rng.choice(r, size=max(1, r // 2), replace=False)
        arr[pos] = rng.integers(low=table_size, high=table_size * 2,
                                size=max(1, r // 2), dtype=np.int32)
        return arr.reshape(1, r).astype(np_idx_dt)
    if "same" in name:
        return np.full((1, r), table_size // 2, dtype=np_idx_dt)
    return rng.integers(low=0, high=table_size, size=(1, r),
                        dtype=np.int32).astype(np_idx_dt)


def _make_elem_idx(rng, r, c, table_size, oob, name, np_idx_dt):
    n = r * c
    if oob in ("skip", "clamp", "wrap"):
        arr = rng.integers(low=0, high=table_size, size=n, dtype=np.int32)
        pos = rng.choice(n, size=max(1, n // 2), replace=False)
        arr[pos] = rng.integers(low=table_size, high=table_size * 2,
                                size=max(1, n // 2), dtype=np.int32)
        return arr.reshape(r, c).astype(np_idx_dt)
    if "seq" in name:
        return np.arange(n, dtype=np.int32).reshape(r, c).astype(np_idx_dt)
    if "same" in name:
        return np.full((r, c), table_size // 2, dtype=np_idx_dt)
    return rng.integers(low=0, high=table_size, size=(r, c),
                        dtype=np.int32).astype(np_idx_dt)


def _make_inputs(name, dtype, idx_dtype, src_shape, table_size, coalesce, oob):
    np_dt = npy_dtype(dtype)
    np_idx_dt = npy_dtype(idx_dtype)
    rng = np.random.default_rng(zlib.crc32(name.encode()) & 0xFFFFFFFF)
    r, c = src_shape
    src = _make_src(np_dt, r * c).reshape(r, c)
    if coalesce == "row":
        idx = _make_row_idx(rng, r, table_size, oob, name, np_idx_dt)
    else:
        idx = _make_elem_idx(rng, r, c, table_size, oob, name, np_idx_dt)
    return [src, idx]


# ---------------------------------------------------------------------------
# Build CASES
# ---------------------------------------------------------------------------

CASES = []

for _name, _dtype, _idx_dtype, _src_shape, _table_size, _coalesce, _atomic, _oob in CASES_DEF:
    _golden_fn = _golden_row if _coalesce == "row" else _golden_elem
    CASES.append(
        golden_output_case(
            "mscatter_" + _name,
            _kernels[_name],
            inputs=lambda _n=_name, _d=_dtype, _id=_idx_dtype, _ss=_src_shape,
                          _ts=_table_size, _c=_coalesce, _o=_oob:
                _make_inputs(_n, _d, _id, _ss, _ts, _c, _o),
            expected=lambda src, idx, _ts=_table_size, _sc=_src_shape[1],
                            _a=_atomic, _o=_oob, _g=_golden_fn:
                _g(src, idx, _ts, _sc, _a, _o) if _g is _golden_row
                else _g(src, idx, _ts, _a, _o),
            rtol=1e-5,
            atol=1e-5,
        )
    )


auto_main(globals())
