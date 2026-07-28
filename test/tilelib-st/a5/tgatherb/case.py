#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# PTODSL rewrite of test/tilelang_st/npu/a5/src/st/testcase/tgatherb.
#
# tgatherb gathers elements from a source tile into a destination tile using
# byte offsets.  Each element of *offset* is a byte address into the flat byte
# representation of *src*; the element at that byte position is written into
# the corresponding position of *dst*.

from pathlib import Path
import sys

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


# Each case is (name, dtype, src_shape, dst_shape).
# The offset tile has shape [dst_rows, dst_cols // elems_per_block] and contains
# byte offsets into the flat src tile.
# elems_per_block = 32 // dtype.itemsize
# Keep at least two template-loop iterations for the reduced wide cases while
# covering every supported dtype and both single-row and multi-row tiles.
CASE_SHAPES = [
    ("f32_2x128", pto.f32, (2, 128), (2, 128)),
    ("i32_2x128", pto.i32, (2, 128), (2, 128)),
    ("ui32_2x128", pto.ui32, (2, 128), (2, 128)),
    ("i16_1x256", pto.i16, (1, 256), (1, 256)),
    ("ui16_3x128", pto.ui16, (3, 128), (3, 128)),
    ("f16_1x256", pto.f16, (1, 256), (1, 256)),
    ("i8_2x256", pto.i8, (2, 256), (2, 256)),
    ("i8_2x512", pto.i8, (2, 512), (2, 512)),
    ("ui8_2x512", pto.ui8, (2, 512), (2, 512)),
]


def _tgatherb_body(src_ptr, offset_ptr, dst_ptr, *, src_rows, src_cols, dst_rows, dst_cols, dtype):
    """Shared kernel body for the tgatherb cases.

    Loads *src* and *offset* tiles from GM, performs block-gather using
    ``pto.vgatherb``, and stores *dst* back to GM.
    """
    block_size_elem = 32 // np.dtype(npy_dtype(dtype)).itemsize
    offset_cols = dst_cols // block_size_elem

    src_view = pto.make_tensor_view(src_ptr, shape=[src_rows, src_cols], strides=[src_cols, 1])
    offset_view = pto.make_tensor_view(offset_ptr, shape=[dst_rows, offset_cols], strides=[offset_cols, 1])
    dst_view = pto.make_tensor_view(dst_ptr, shape=[dst_rows, dst_cols], strides=[dst_cols, 1])

    src_tile = pto.alloc_tile(shape=[src_rows, src_cols], dtype=dtype)
    offset_tile = pto.alloc_tile(shape=[dst_rows, offset_cols], dtype=pto.ui32)
    dst_tile = pto.alloc_tile(shape=[dst_rows, dst_cols], dtype=dtype)

    pto.tile.load(src_view, src_tile)
    pto.tile.load(offset_view, offset_tile)
    pto.tile.gatherb(src_tile, offset_tile, dst_tile)
    pto.tile.store(dst_tile, dst_view)


# One decorated kernel per case, each binding static shapes at definition time.
_tgatherb_kernels = {}
for _name, _dtype, _src_shape, _dst_shape in CASE_SHAPES:
    _sr, _sc = _src_shape
    _dr, _dc = _dst_shape

    def _make(sr=_sr, sc=_sc, dr=_dr, dc=_dc, dtype=_dtype, kernel_name=f"tgatherb_{_name}"):
        @pto.jit(
            name=kernel_name,
            target="a5"
        )
        def _kernel(
            src_ptr: pto.ptr(dtype, "gm"),
            offset_ptr: pto.ptr(pto.ui32, "gm"),
            dst_ptr: pto.ptr(dtype, "gm"),
        ):
            _tgatherb_body(
                src_ptr, offset_ptr, dst_ptr,
                src_rows=sr, src_cols=sc, dst_rows=dr, dst_cols=dc, dtype=dtype,
            )

        return _kernel

    _tgatherb_kernels[_name] = _make()


def _get_data_size(dtype):
    """Return the element size in bytes for a given numpy dtype."""
    return np.dtype(dtype).itemsize


def _make_inputs(name, dtype, src_shape, dst_shape):
    dtype = npy_dtype(dtype)
    data_size = _get_data_size(dtype)
    block_size_elem = 32 // data_size

    rng_seed = hash(name) & 0xFFFFFFFF
    rng = np.random.RandomState(rng_seed)

    src = np.arange(np.prod(src_shape)).astype(dtype)

    offset_col = dst_shape[1] // block_size_elem
    offset_shape = (dst_shape[0], offset_col)
    offset = np.zeros(np.prod(offset_shape)).astype(np.int32)
    for i in range(len(offset)):
        offset[i] = i * 32
    offset = offset.reshape(offset_shape)

    return [src, offset]


def _make_expected(src, offsets, dtype):
    dtype = npy_dtype(dtype)
    data_size = _get_data_size(dtype)
    block_size_elem = 32 // data_size

    flat_src = src.reshape(-1)
    dst = np.zeros(offsets.size * block_size_elem).astype(dtype)
    flat_offsets = offsets.ravel()

    count = 0
    for i in range(len(flat_offsets)):
        byte_off = int(flat_offsets[i])
        block_start = byte_off // data_size
        for j in range(block_size_elem):
            dst[count] = flat_src[block_start + j]
            count += 1

    return dst.reshape(src.shape)


CASES = []
for _name, _dtype, _src_shape, _dst_shape in CASE_SHAPES:
    CASES.append(
        golden_output_case(
            "tgatherb_" + _name,
            _tgatherb_kernels[_name],
            inputs=lambda _n=_name, _d=_dtype, _ss=_src_shape, _ds=_dst_shape: _make_inputs(_n, _d, _ss, _ds),
            expected=lambda src, offsets, _d=_dtype: _make_expected(src, offsets, _d),
            rtol=1e-6,
            atol=1e-6,
        )
    )


auto_main(globals())
