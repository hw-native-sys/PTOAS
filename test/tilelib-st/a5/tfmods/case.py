#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# PTODSL rewrite of test/tilelang_st/npu/a5/src/st/testcase/tfmods.
# The legacy A5 suite runs only its f32/f16 cases; integer cases are omitted
# because the A5 vector divide path does not support integer element types.

from pathlib import Path
import sys
import zlib

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import auto_main, golden_output_case
from ptodsl import pto


CASE_SPECS = [
    ("f32_32x64", pto.f32, np.float32, (32, 64), 1e-6),
    ("f16_63x64", pto.f16, np.float16, (63, 64), 1e-3),
    ("f32_7x448", pto.f32, np.float32, (7, 448), 1e-6),
    ("f32_256x16", pto.f32, np.float32, (256, 16), 1e-6),
]


def _make_kernel(name, dtype, shape):
    rows, cols = shape

    @pto.jit(name="tfmods_" + name, target="a5")
    def _kernel(
        src_ptr: pto.ptr(dtype, "gm"),
        dst_ptr: pto.ptr(dtype, "gm"),
    ):
        src_view = pto.make_tensor_view(src_ptr, shape=[rows, cols], strides=[cols, 1])
        dst_view = pto.make_tensor_view(dst_ptr, shape=[rows, cols], strides=[cols, 1])

        src_tile = pto.alloc_tile(shape=[rows, cols], dtype=dtype)
        dst_tile = pto.alloc_tile(shape=[rows, cols], dtype=dtype)

        pto.tile.load(src_view, src_tile)
        pto.tile.fmods(src_tile, 3.0, dst_tile)
        pto.tile.store(dst_tile, dst_view)

    return _kernel


_KERNELS = {
    name: _make_kernel(name, dtype, shape)
    for name, dtype, _, shape, _ in CASE_SPECS
}


def _make_inputs(name, np_dtype, shape):
    np.random.seed(zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF)
    return [np.random.randint(1, 10, size=shape).astype(np_dtype)]


def _make_expected(src):
    scalar = src.dtype.type(3.0)
    return np.fmod(src, scalar).astype(src.dtype, copy=False)


CASES = [
    golden_output_case(
        "tfmods_" + name,
        _KERNELS[name],
        inputs=lambda name=name, np_dtype=np_dtype, shape=shape: _make_inputs(name, np_dtype, shape),
        expected=_make_expected,
        rtol=eps,
        atol=eps,
    )
    for name, _, np_dtype, shape, eps in CASE_SPECS
]


auto_main(globals())
