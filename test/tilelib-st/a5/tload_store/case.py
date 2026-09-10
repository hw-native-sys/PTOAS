#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# PTODSL rewrite of the minimal GM -> tile -> GM coverage from
# test/tilelang_st/npu/a5/src/st/testcase/tload/tload.pto.
#
# Start with two static f32 round-trips:
#   1. ND / row-major
#   2. DN / col-major
# These are the smallest data-movement cases needed to validate that PTODSL can
# drive tload/tstore on A5 without the tilelang_st harness.

from pathlib import Path
import sys

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import auto_main, golden_output_case
from ptodsl import pto


CASE_SPECS = [
    {
        "case_name": "nd_f32_16x64",
        "kernel_name": "tload_store_nd_f32_16x64",
        "shape": (16, 64),
        "view_strides": None,
        "tile_kwargs": {},
    },
    {
        "case_name": "dn_f32_16x64",
        "kernel_name": "tload_store_dn_f32_16x64",
        "shape": (16, 64),
        "view_strides": None,
        "tile_kwargs": {"blayout": "ColMajor"},
    },
]

def _roundtrip_body(src_ptr, dst_ptr, *, rows, cols, view_strides=None, tile_kwargs=None):
    total = rows * cols
    if view_strides is None:
        view_strides = [total, total, total, cols, 1]

    if view_strides is not None and len(view_strides) == 5:
        view_strides = view_strides[-2:]

    src_view = pto.make_tensor_view(src_ptr, shape=[rows, cols], strides=view_strides)
    dst_view = pto.make_tensor_view(dst_ptr, shape=[rows, cols], strides=view_strides)

    tile = pto.alloc_tile(
        shape=[rows, cols],
        dtype=pto.f32,
        **(tile_kwargs or {}),
    )

    pto.tile.load(src_view, tile)
    pto.tile.store(tile, dst_view)


_tload_store_kernels = {}
for _spec in CASE_SPECS:
    _rows, _cols = _spec["shape"]
    _view_strides = _spec["view_strides"]
    if _view_strides is None and _spec["tile_kwargs"].get("blayout") == "ColMajor":
        _view_strides = [_rows * _cols, _rows * _cols, _rows * _cols, 1, _rows]
    _tile_kwargs = dict(_spec["tile_kwargs"])
    _kernel_name = _spec["kernel_name"]
    _case_name = _spec["case_name"]

    def _make(rows=_rows, cols=_cols, view_strides=_view_strides, tile_kwargs=_tile_kwargs, kernel_name=_kernel_name):
        @pto.jit(
            name=kernel_name,
            target="a5",
        )
        def _kernel(
            src_ptr: pto.ptr(pto.f32, "gm"),
            dst_ptr: pto.ptr(pto.f32, "gm"),
        ):
            _roundtrip_body(
                src_ptr,
                dst_ptr,
                rows=rows,
                cols=cols,
                view_strides=view_strides,
                tile_kwargs=tile_kwargs,
            )

        return _kernel

    _tload_store_kernels[_case_name] = _make()


def _make_input(name, shape):
    import zlib

    np.random.seed(zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF)
    return np.random.randint(1, 32, size=shape).astype(np.float32)


def _make_expected(src):
    return np.asarray(src, dtype=np.float32).copy()


CASES = []
for _spec in CASE_SPECS:
    _case_name = _spec["case_name"]
    _shape = _spec["shape"]
    CASES.append(
        golden_output_case(
            "tload_store_" + _case_name,
            _tload_store_kernels[_case_name],
            inputs=lambda _case_name=_case_name, _shape=_shape: [_make_input(_case_name, _shape)],
            expected=_make_expected,
            rtol=1e-6,
            atol=1e-6,
        )
    )


auto_main(globals())
