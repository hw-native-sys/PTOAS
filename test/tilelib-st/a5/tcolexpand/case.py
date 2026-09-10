#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Minimal PTODSL broadcast pilot for A5:
#   tload(src) + tcolexpand(src)->dst + tstore(dst)

from pathlib import Path
import sys

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import auto_main, golden_output_case
from ptodsl import pto


SRC_ROWS = 1
DST_ROWS = 8
COLS = 128


@pto.jit(
    name="tcolexpand_f32_1x8x128",
    target="a5",
)
def _tcolexpand_kernel(
    src_ptr: pto.ptr(pto.f32, "gm"),
    dst_ptr: pto.ptr(pto.f32, "gm"),
):
    src_view = pto.make_tensor_view(
        src_ptr,
        shape=[SRC_ROWS, COLS],
        strides=[COLS, 1],
    )
    dst_view = pto.make_tensor_view(
        dst_ptr,
        shape=[DST_ROWS, COLS],
        strides=[COLS, 1],
    )

    src_tile = pto.alloc_tile(shape=[SRC_ROWS, COLS], dtype=pto.f32)
    dst_tile = pto.alloc_tile(shape=[DST_ROWS, COLS], dtype=pto.f32)

    pto.tile.load(src_view, src_tile)
    pto.tile.colexpand(src_tile, dst_tile)
    pto.tile.store(dst_tile, dst_view)


def _make_input():
    rng = np.random.default_rng(0xC01E0A5)
    return rng.uniform(-2.0, 2.0, size=(SRC_ROWS, COLS)).astype(np.float32)


def _make_expected(src):
    return np.repeat(src, DST_ROWS, axis=0).astype(np.float32)


CASES = [
    golden_output_case(
        "tcolexpand_f32_1x8x128",
        _tcolexpand_kernel,
        inputs=lambda: [_make_input()],
        expected=_make_expected,
        rtol=1e-6,
        atol=1e-6,
    ),
]


auto_main(globals())
