#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# PTODSL ST for pto.tconcat: dst[:, 0:c0] = src0; dst[:, c0:] = src1.

from pathlib import Path
import sys

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import auto_main, golden_output_case
from ptodsl import pto


_NP_TO_PTO = {
    np.dtype(np.float32): pto.f32,
    np.dtype(np.float16): pto.f16,
    np.dtype(np.int32): pto.i32,
    np.dtype(np.int16): pto.i16,
}

# (case_name, np.dtype, valid_row, valid_col0, valid_col1).
CASE_SPECS = [
    ("f32_64x64_cat_64x64", np.float32, 64, 64, 64),
    ("int32_64x64_cat_64x64", np.int32, 64, 64, 64),
    ("f16_16x128_cat_16x128", np.float16, 16, 128, 128),
    ("f32_16x32_cat_16x32", np.float32, 16, 32, 32),
    ("int16_32x128_cat_32x128", np.int16, 32, 128, 128),
    ("f16_16x63_cat_16x64", np.float16, 16, 63, 64),
    ("f32_16x31_cat_16x32", np.float32, 16, 31, 32),
    ("int16_32x127_cat_32x128", np.int16, 32, 127, 128),
]


def _aligned_cols(valid_cols, elemsize):
    # Row-major physical rows must be 32-byte aligned; the tail uses valid_shape.
    row_bytes = valid_cols * elemsize
    aligned = ((row_bytes + 31) // 32) * 32 // elemsize
    return max(aligned, valid_cols)


def _alloc_vec(rows, valid_cols, elemsize, dtype):
    aligned_cols = _aligned_cols(valid_cols, elemsize)
    kwargs = {"shape": [rows, aligned_cols], "dtype": dtype}
    if aligned_cols != valid_cols:
        kwargs["valid_shape"] = [rows, valid_cols]
    return pto.alloc_tile(**kwargs)


def _concat_body(s0_ptr, s1_ptr, d_ptr, *, dtype, elemsize, rows, c0, c1):
    s0_view = pto.make_tensor_view(s0_ptr, shape=[rows, c0], strides=[c0, 1])
    s1_view = pto.make_tensor_view(s1_ptr, shape=[rows, c1], strides=[c1, 1])
    d_view = pto.make_tensor_view(d_ptr, shape=[rows, c0 + c1], strides=[c0 + c1, 1])

    t0 = _alloc_vec(rows, c0, elemsize, dtype)
    t1 = _alloc_vec(rows, c1, elemsize, dtype)
    td = _alloc_vec(rows, c0 + c1, elemsize, dtype)

    pto.tile.load(s0_view, t0)
    pto.tile.load(s1_view, t1)
    pto.tile.concat(t0, t1, td)
    pto.tile.store(td, d_view)


_tconcat_kernels = {}
for _name, _npdt, _r, _c0, _c1 in CASE_SPECS:
    _dt = _NP_TO_PTO[np.dtype(_npdt)]
    _esz = np.dtype(_npdt).itemsize
    def _make(dt=_dt, esz=_esz, r=_r, c0=_c0, c1=_c1, kernel_name=f"tconcat_{_name}"):
        @pto.jit(name=kernel_name, target="a5")
        def _kernel(s0_ptr: pto.ptr(dt, "gm"), s1_ptr: pto.ptr(dt, "gm"), d_ptr: pto.ptr(dt, "gm")):
            _concat_body(s0_ptr, s1_ptr, d_ptr, dtype=dt, elemsize=esz, rows=r, c0=c0, c1=c1)
        return _kernel

    _tconcat_kernels[_name] = _make()


def _make_inputs(name, npdt, rows, c0, c1):
    # Deterministic per-case seed (crc32(name)); value range (-1000, 1000).
    import zlib
    np.random.seed(zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF)
    a = np.random.uniform(-1000, 1000, size=(rows, c0)).astype(npdt)
    b = np.random.uniform(-1000, 1000, size=(rows, c1)).astype(npdt)
    return [a, b]


def _make_expected(a, b):
    return np.concatenate([a, b], axis=1)


CASES = []
for _name, _npdt, _r, _c0, _c1 in CASE_SPECS:
    CASES.append(
        golden_output_case(
            "tconcat_" + _name,
            _tconcat_kernels[_name],
            inputs=lambda _name=_name, _npdt=_npdt, _r=_r, _c0=_c0, _c1=_c1: _make_inputs(
                "tconcat_" + _name, _npdt, _r, _c0, _c1
            ),
            expected=_make_expected,
            output_shape=(_r, _c0 + _c1),
            output_dtype=_npdt,
            rtol=1e-3,
            atol=1e-3,
        )
    )


auto_main(globals())
