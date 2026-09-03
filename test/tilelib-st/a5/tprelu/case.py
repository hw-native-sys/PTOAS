#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""PTODSL migration of the legacy A5 ``tprelu`` TileLang ST cases."""

from pathlib import Path
import sys
import zlib

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import auto_main, golden_output_case
from ptodsl import pto


PTO_TO_NP_DTYPE = {
    pto.f16: np.float16,
    pto.f32: np.float32,
}

# (legacy case name, pto dtype, physical shape, valid shape, tolerance).
# The 63x63 cases intentionally use a 64x64 physical tile and valid_shape,
# matching the legacy tile_buf and partition_view contract.
CASE_SPECS = [
    ("f16_64x64",   pto.f16, (64, 64),     (64, 64),     1e-3),
    ("f16_63x63",   pto.f16, (64, 64),     (63, 63),     1e-3),
    ("f16_1x16384", pto.f16, (1, 16384),   (1, 16384),   1e-3),
    ("f16_2048x16", pto.f16, (2048, 16),   (2048, 16),   1e-3),
    ("f32_64x64",   pto.f32, (64, 64),     (64, 64),     1e-6),
    ("f32_63x63",   pto.f32, (64, 64),     (63, 63),     1e-6),
    ("f32_1x16384", pto.f32, (1, 16384),   (1, 16384),   1e-6),
    ("f32_2048x8",  pto.f32, (2048, 8),   (2048, 8),    1e-6),
]


def _make_kernel(name, dtype, shape, valid_shape):
    rows, cols = shape
    valid_rows, valid_cols = valid_shape

    @pto.jit(name="tprelu_" + name, target="a5")
    def _kernel(
        src0_ptr: pto.ptr(dtype, "gm"),
        src1_ptr: pto.ptr(dtype, "gm"),
        dst_ptr: pto.ptr(dtype, "gm"),
    ):
        src0_view = pto.make_tensor_view(src0_ptr, shape=[rows, cols], strides=[cols, 1])
        src1_view = pto.make_tensor_view(src1_ptr, shape=[rows, cols], strides=[cols, 1])
        dst_view = pto.make_tensor_view(dst_ptr, shape=[rows, cols], strides=[cols, 1])

        src0_tile = pto.alloc_tile(
            shape=[rows, cols], dtype=dtype, valid_shape=[valid_rows, valid_cols]
        )
        src1_tile = pto.alloc_tile(
            shape=[rows, cols], dtype=dtype, valid_shape=[valid_rows, valid_cols]
        )
        dst_tile = pto.alloc_tile(
            shape=[rows, cols], dtype=dtype, valid_shape=[valid_rows, valid_cols]
        )

        pto.tile.load(src0_view, src0_tile)
        pto.tile.load(src1_view, src1_tile)
        pto.tile.prelu(src0_tile, src1_tile, dst_tile)
        pto.tile.store(dst_tile, dst_view)

    return _kernel


_kernels = {
    name: _make_kernel(name, dtype, shape, valid_shape)
    for name, dtype, shape, valid_shape, _ in CASE_SPECS
}


def _make_inputs(name, dtype, shape):
    np.random.seed(zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF)
    np_dtype = PTO_TO_NP_DTYPE[dtype]
    src0 = np.random.uniform(-8, 8, size=shape).astype(np_dtype)
    src1 = np.random.uniform(-8, 8, size=shape).astype(np_dtype)
    return [src0, src1]


def _make_expected(src0, src1, valid_shape):
    valid_rows, valid_cols = valid_shape
    golden = np.zeros(src0.shape, dtype=src0.dtype)
    golden[:valid_rows, :valid_cols] = np.where(
        src0[:valid_rows, :valid_cols] > 0,
        src0[:valid_rows, :valid_cols],
        src0[:valid_rows, :valid_cols] * src1[:valid_rows, :valid_cols],
    ).astype(src0.dtype)
    return golden


CASES = [
    golden_output_case(
        "tprelu_" + name,
        _kernels[name],
        inputs=lambda _n=name, _d=dtype, _s=shape: _make_inputs(_n, _d, _s),
        expected=lambda src0, src1, _v=valid_shape: _make_expected(src0, src1, _v),
        rtol=eps,
        atol=eps,
    )
    for name, dtype, shape, valid_shape, eps in CASE_SPECS
]


auto_main(globals())
