#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""PTODSL TileLib ST case for ``pto.ttrans`` (A5 2D tile transpose).

Port of test/tilelang_st/npu/a5/src/st/testcase/ttrans from the pto-isa repo.
Each case loads a source tile, transposes it via ``pto.tile.transpose``, and
stores the result; the golden is a numpy ``.T``.
"""

from pathlib import Path
import sys

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import auto_main, golden_output_case
from ptodsl import pto


# (name, dtype, rows, cols). rows/cols chosen so the major dim is 32-byte
# aligned: cols * sizeof(dtype) % 32 == 0.
CASE_SHAPES = [
    ("f32_8x64", pto.f32, 8, 64),    # wide b32: RowWise path
    ("f32_64x8", pto.f32, 64, 8),    # tall b32: ColWise path
    ("f32_32x32", pto.f32, 32, 32),  # square b32
    ("i32_8x64", pto.i32, 8, 64),
    ("f16_64x16", pto.f16, 64, 16),  # tall b16 (16*2=32 aligned)
]


def _ttrans_body(src_ptr, dst_ptr, *, rows, cols, dtype):
    """Shared kernel body: load src, transpose into dst (with tmp), store."""
    src_view = pto.make_tensor_view(src_ptr, shape=[rows, cols], strides=[cols, 1])
    dst_view = pto.make_tensor_view(dst_ptr, shape=[cols, rows], strides=[rows, 1])

    src_tile = pto.alloc_tile(shape=[rows, cols], dtype=dtype)
    tmp_tile = pto.alloc_tile(shape=[cols, rows], dtype=dtype)
    dst_tile = pto.alloc_tile(shape=[cols, rows], dtype=dtype)

    pto.tile.load(src_view, src_tile)
    pto.tile.transpose(src_tile, tmp_tile, dst_tile)
    pto.tile.store(dst_tile, dst_view)


_ttrans_kernels = {}
for _name, _dtype, _rows, _cols in CASE_SHAPES:
    _r, _c = _rows, _cols

    def _make(r=_r, c=_c, dtype=_dtype, kernel_name=f"ttrans_{_name}"):
        @pto.jit(name=kernel_name, target="a5")
        def _kernel(
            src_ptr: pto.ptr(dtype, "gm"),
            dst_ptr: pto.ptr(dtype, "gm"),
        ):
            _ttrans_body(src_ptr, dst_ptr, rows=r, cols=c, dtype=dtype)

        return _kernel

    _ttrans_kernels[_name] = _make()


# numpy dtype mapping for input generation + golden
_PTO_TO_NP = {
    pto.f32: np.float32, pto.f16: np.float16,
    pto.i32: np.int32,
}


def _make_inputs(name, dtype, rows, cols):
    import zlib
    np_dtype = _PTO_TO_NP[dtype]
    np.random.seed(zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF)
    # use small ints in float range to avoid f16 overflow / bf16 precision loss
    a = np.random.randint(1, 10, size=(rows, cols)).astype(np_dtype)
    return [a]


def _make_expected(a):
    return np.ascontiguousarray(a.T)


CASES = []
for _name, _dtype, _rows, _cols in CASE_SHAPES:
    CASES.append(
        golden_output_case(
            "ttrans_" + _name,
            _ttrans_kernels[_name],
            inputs=lambda _n=_name, _d=_dtype, _r=_rows, _c=_cols: _make_inputs(_n, _d, _r, _c),
            expected=_make_expected,
            rtol=1e-3,  # f16/bf16 need looser tol
            atol=1e-3,
        )
    )


auto_main(globals())
