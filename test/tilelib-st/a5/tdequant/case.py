#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# PTODSL ST for pto.tdequant: dst[r,c] = (float(src[r,c]) - offset[r,0]) * scale[r,0].

from pathlib import Path
import sys

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import auto_main, golden_output_case
from ptodsl import pto


_SRC_NP_TO_PTO = {
    np.dtype(np.int16): pto.i16,
    np.dtype(np.int8): pto.i8,
}

# (case_name, src_np_dtype, flag_offset_zero, valid_row, valid_col).
CASE_SPECS = [
    ("i16_64x64_offzero", np.int16, True, 64, 64),
    ("i16_64x64", np.int16, False, 64, 64),
    ("i16_63x63", np.int16, False, 63, 63),
    ("i8_64x64_offzero", np.int8, True, 64, 64),
    ("i8_64x64", np.int8, False, 64, 64),
    ("i8_63x63", np.int8, False, 63, 63),
    ("i16_51x112_offzero", np.int16, True, 51, 112),
]


def _aligned_cols(valid_cols, elemsize):
    row_bytes = valid_cols * elemsize
    aligned = ((row_bytes + 31) // 32) * 32 // elemsize
    return max(aligned, valid_cols)


def _alloc_row_tile(rows, valid_cols, elemsize, dtype):
    aligned_cols = _aligned_cols(valid_cols, elemsize)
    kwargs = {"shape": [rows, aligned_cols], "dtype": dtype}
    if aligned_cols != valid_cols:
        kwargs["valid_shape"] = [rows, valid_cols]
    return pto.alloc_tile(**kwargs)


def _alloc_col_tile(valid_rows, dtype, elemsize=4):
    # scale/offset: f32 ColMajor column-vector tile (32-byte-aligned physical rows).
    para_rows = ((valid_rows * elemsize + 31) // 32) * 32 // elemsize
    para_rows = max(para_rows, valid_rows)
    kwargs = {"shape": [para_rows, 1], "dtype": dtype, "blayout": "ColMajor"}
    if para_rows != valid_rows:
        kwargs["valid_shape"] = [valid_rows, 1]
    return pto.alloc_tile(**kwargs)


def _dequant_body(src_ptr, scale_ptr, off_ptr, dst_ptr, *, src_dtype, rows, cols):
    src_view = pto.make_tensor_view(src_ptr, shape=[rows, cols], strides=[cols, 1])
    scale_view = pto.make_tensor_view(scale_ptr, shape=[rows, 1], strides=[1, 1])
    off_view = pto.make_tensor_view(off_ptr, shape=[rows, 1], strides=[1, 1])
    dst_view = pto.make_tensor_view(dst_ptr, shape=[rows, cols], strides=[cols, 1])

    src_esize = np.dtype(np.int16).itemsize if src_dtype is pto.i16 else np.dtype(np.int8).itemsize
    src_tile = _alloc_row_tile(rows, cols, src_esize, src_dtype)
    scale_tile = _alloc_col_tile(rows, pto.f32)
    off_tile = _alloc_col_tile(rows, pto.f32)
    dst_tile = _alloc_row_tile(rows, cols, 4, pto.f32)

    pto.tile.load(src_view, src_tile)
    pto.tile.load(scale_view, scale_tile)
    pto.tile.load(off_view, off_tile)
    pto.tile.dequant(src_tile, scale_tile, off_tile, dst_tile)
    pto.tile.store(dst_tile, dst_view)


_dequant_kernels = {}
for _name, _npdt, _flag, _r, _c in CASE_SPECS:
    _sdt = _SRC_NP_TO_PTO[np.dtype(_npdt)]
    def _make(src_dtype=_sdt, r=_r, c=_c, kernel_name=f"tdequant_{_name}"):
        @pto.jit(name=kernel_name, target="a5")
        def _kernel(src_ptr: pto.ptr(src_dtype, "gm"), scale_ptr: pto.ptr(pto.f32, "gm"),
                    off_ptr: pto.ptr(pto.f32, "gm"), dst_ptr: pto.ptr(pto.f32, "gm")):
            _dequant_body(src_ptr, scale_ptr, off_ptr, dst_ptr, src_dtype=src_dtype, rows=r, cols=c)
        return _kernel

    _dequant_kernels[_name] = _make()


def _make_inputs(name, npdt, flag, rows, cols):
    # Deterministic per-case seed; value range (-100, 100).
    import zlib
    np.random.seed(zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF)
    src = np.random.uniform(-100, 100, size=(rows, cols)).astype(npdt)
    scale = np.random.uniform(-100, 100, size=(rows, 1)).astype(np.float32)
    offset = np.random.uniform(-100, 100, size=(rows, 1)).astype(np.float32)
    if flag:
        offset[:, :] = 0.0
    return [src, scale, offset]


def _make_expected(src, scale, offset):
    temp = src.astype(np.float32)
    return ((temp - offset[:, 0:1]) * scale[:, 0:1]).astype(np.float32)


CASES = []
for _name, _npdt, _flag, _r, _c in CASE_SPECS:
    CASES.append(
        golden_output_case(
            "tdequant_" + _name,
            _dequant_kernels[_name],
            inputs=lambda _name=_name, _npdt=_npdt, _flag=_flag, _r=_r, _c=_c: _make_inputs(
                "tdequant_" + _name, _npdt, _flag, _r, _c
            ),
            expected=_make_expected,
            output_shape=(_r, _c),
            output_dtype=np.float32,
            rtol=1e-3,
            atol=1e-3,
        )
    )


auto_main(globals())
