#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Minimal PTODSL reduction pilot for A5:
#   tload(src) + tcolsum(src)->dst + tstore(dst)

from pathlib import Path
import sys

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import auto_main, golden_output_case
from ptodsl import pto


ROWS = 16
COLS = 128


@pto.jit(
    name="tcolsum_f32_16x128",
    target="a5",
)
def _tcolsum_kernel(
    src_ptr: pto.ptr(pto.f32, "gm"),
    dst_ptr: pto.ptr(pto.f32, "gm"),
):
    src_view = pto.make_tensor_view(
        src_ptr,
        shape=[ROWS, COLS],
        strides=[COLS, 1],
    )
    dst_view = pto.make_tensor_view(
        dst_ptr,
        shape=[1, COLS],
        strides=[COLS, 1],
    )

    src_tile = pto.alloc_tile(shape=[ROWS, COLS], dtype=pto.f32)
    dst_tile = pto.alloc_tile(shape=[1, COLS], dtype=pto.f32)

    pto.tile.load(src_view, src_tile)
    pto.tile.colsum(src_tile, dst_tile)
    pto.tile.store(dst_tile, dst_view)


def _make_input():
    rng = np.random.default_rng(0xC01A5EED)
    return rng.uniform(-3.0, 3.0, size=(ROWS, COLS)).astype(np.float32)


def _make_expected(src):
    return np.sum(src, axis=0, keepdims=True, dtype=np.float32)


CASES = [
    golden_output_case(
        "tcolsum_f32_16x128",
        _tcolsum_kernel,
        inputs=lambda: [_make_input()],
        expected=_make_expected,
        rtol=1e-5,
        atol=1e-5,
    ),
]


auto_main(globals())
