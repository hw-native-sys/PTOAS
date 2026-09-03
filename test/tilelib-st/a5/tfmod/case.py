#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# PTODSL rewrite of test/tilelang_st/npu/a5/src/st/testcase/tfmod.

from pathlib import Path
import sys
import zlib

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import auto_main, golden_output_case
from ptodsl import pto


CASE_SPECS = [
    ("f32_16x64", pto.f32, np.float32, (16, 64), 1e-6, "default"),
    ("f32_32x32", pto.f32, np.float32, (32, 32), 1e-6, "default"),
    ("f32_16x64_high_precision", pto.f32, np.float32, (16, 64), 1e-6, "high_precision"),
]


def _make_kernel(name, dtype, shape, precision):
    rows, cols = shape

    @pto.jit(name="tfmod_" + name, target="a5")
    def _kernel(
        src0_ptr: pto.ptr(dtype, "gm"),
        src1_ptr: pto.ptr(dtype, "gm"),
        dst_ptr: pto.ptr(dtype, "gm"),
    ):
        src0_view = pto.make_tensor_view(src0_ptr, shape=[rows, cols], strides=[cols, 1])
        src1_view = pto.make_tensor_view(src1_ptr, shape=[rows, cols], strides=[cols, 1])
        dst_view = pto.make_tensor_view(dst_ptr, shape=[rows, cols], strides=[cols, 1])

        src0_tile = pto.alloc_tile(shape=[rows, cols], dtype=dtype)
        src1_tile = pto.alloc_tile(shape=[rows, cols], dtype=dtype)
        dst_tile = pto.alloc_tile(shape=[rows, cols], dtype=dtype)

        pto.tile.load(src0_view, src0_tile)
        pto.tile.load(src1_view, src1_tile)
        pto.tile.fmod(src0_tile, src1_tile, dst_tile, precision=precision)
        pto.tile.store(dst_tile, dst_view)

    return _kernel


_KERNELS = {
    name: _make_kernel(name, dtype, shape, precision)
    for name, dtype, _, shape, _, precision in CASE_SPECS
}


def _make_inputs(name, np_dtype, shape):
    np.random.seed(zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF)
    src0 = np.random.randint(-9, 10, size=shape).astype(np_dtype)
    src1 = np.random.randint(-9, 10, size=shape).astype(np_dtype)
    src1[src1 == 0] = 1
    return [src0, src1]


def _make_expected(src0, src1):
    return np.fmod(src0, src1).astype(src0.dtype, copy=False)


CASES = [
    golden_output_case(
        "tfmod_" + name,
        _KERNELS[name],
        inputs=lambda name=name, np_dtype=np_dtype, shape=shape: _make_inputs(name, np_dtype, shape),
        expected=_make_expected,
        rtol=eps,
        atol=eps,
    )
    for name, _, np_dtype, shape, eps, _ in CASE_SPECS
]


auto_main(globals())
