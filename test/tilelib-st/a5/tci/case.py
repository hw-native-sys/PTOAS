#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# PTODSL rewrite of test/tilelang_st/npu/a5/src/st/testcase/tci.

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
    """Return the numpy dtype corresponding to a pto scalar-dtype name."""
    return PTO_TO_NP_DTYPE[pto_type]

# Cover short tails, an aligned vector span, a cross-256 span, both directions,
# and every supported integer width/sign.  Larger spans exercise the same
# scalar-store loop and only add simulator pressure.
CASE_SHAPES = [
    ("i32_1x7", pto.i32, (1, 7), False),
    ("i32_1x128_desc", pto.i32, (1, 128), True),
    ("i32_1x257", pto.i32, (1, 257), False),
    ("i16_1x7", pto.i16, (1, 7), False),
    ("i16_1x192_desc", pto.i16, (1, 192), True),
    ("i16_1x257", pto.i16, (1, 257), False),
    ("ui32_1x7", pto.ui32, (1, 7), False),
    ("ui32_1x128_desc", pto.ui32, (1, 128), True),
    ("ui16_1x7", pto.ui16, (1, 7), False),
    ("ui16_1x128_desc", pto.ui16, (1, 128), True),
]


def _tci_body(start, dst_ptr, *, dst_rows, dst_cols, dtype, descending):
    itemsize = np.dtype(npy_dtype(dtype)).itemsize
    block_size = 32 // itemsize
    aligned_cols = ((dst_cols + block_size - 1) // block_size) * block_size

    dst_view = pto.make_tensor_view(dst_ptr, shape=[dst_rows, dst_cols], strides=[dst_cols, 1])
    dst_tile = pto.alloc_tile(shape=[dst_rows, aligned_cols], dtype=dtype, valid_shape=[dst_rows, dst_cols])
    pto.tile.ci(start, dst_tile, descending=descending)
    pto.tile.store(dst_tile, dst_view)


# One decorated kernel per case, each binding static shapes at definition time.
_tci_kernels = {}
for _name, _dtype, _dst_shape, _desc in CASE_SHAPES:
    _dr, _dc = _dst_shape

    def _make(dr=_dr, dc=_dc, dtype=_dtype, desc=_desc, kernel_name=f"tci_{_name}"):
        @pto.jit(
            name=kernel_name,
            target="a5"
        )
        def _kernel(
            start: dtype,
            dst_ptr: pto.ptr(dtype, "gm"),
        ):
            _tci_body(
                start, dst_ptr,
                dst_rows=dr, dst_cols=dc, dtype=dtype,
                descending=desc
            )

        return _kernel

    _tci_kernels[_name] = _make()


def _make_inputs(name, dtype):
    rng_seed = zlib.crc32(name.encode())
    rng = np.random.RandomState(rng_seed)
    start = rng.randint(0, 100, dtype=npy_dtype(dtype))
    return [start]


def _make_expected(src, dtype, dst_shape, descending=False):
    dtype = npy_dtype(dtype)
    start = int(src)
    if descending:
        dst = np.arange(start, start - dst_shape[1], -1).astype(dtype)
    else:
        dst = np.arange(start, start + dst_shape[1]).astype(dtype)
    return dst.reshape(dst_shape)


CASES = []
for _name, _dtype, _dst_shape, _desc in CASE_SHAPES:
    CASES.append(
        golden_output_case(
            "tci_" + _name,
            _tci_kernels[_name],
            inputs=lambda _n=_name, _d=_dtype: _make_inputs(_n, _d),
            expected=lambda src, _d=_dtype, _ds=_dst_shape, _dc=_desc: _make_expected(src, _d, _ds, _dc),
            rtol=1e-6,
            atol=1e-6,
        )
    )


auto_main(globals())
