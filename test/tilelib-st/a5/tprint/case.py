#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# PTODSL ST for pto.tprint. TPRINT is a device-side debug side effect; the
# simulator does not consistently forward device print payloads to msprof
# stdout, so these ST cases validate the observable data path like the other
# TileLib ST cases: load, print, store, then compare the stored tile.

from pathlib import Path
import sys

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import auto_main
from common import assert_close
from ptodsl import pto


_NP_TO_PTO = {
    np.dtype(np.float32): pto.f32,
    np.dtype(np.int32): pto.i32,
    np.dtype(np.uint32): pto.ui32,
    np.dtype(np.int8): pto.i8,
    np.dtype(np.uint8): pto.ui8,
}


CASE_SPECS = [
    {
        "name": "tprint_float_formats_default",
        "np_dtype": np.float32,
        "shape": (1, 8),
        "values": (1.234567, -3.456789),
    },
    {
        "name": "tprint_float_formats_precision2",
        "np_dtype": np.float32,
        "shape": (1, 8),
        "values": (1.234567, -3.456789),
        "use_tmp": True,
        "print_format": "width8_precision2",
    },
    {
        "name": "tprint_float_formats_precision6",
        "np_dtype": np.float32,
        "shape": (1, 8),
        "values": (1.234567, -3.456789),
        "use_tmp": True,
        "print_format": "width10_precision6",
    },
    {
        "name": "tprint_signed_int_formats",
        "np_dtype": np.int32,
        "shape": (1, 8),
        "values": (42, -17, 1024, -9999),
    },
    {
        "name": "tprint_unsigned_int_formats",
        "np_dtype": np.uint32,
        "shape": (1, 8),
        "values": (0, 17, 65535, 123456),
    },
    {
        "name": "tprint_int8_printed_as_numbers",
        "np_dtype": np.int8,
        "shape": (1, 32),
        "values": (-12, 65),
    },
    {
        "name": "tprint_uint8_printed_as_numbers",
        "np_dtype": np.uint8,
        "shape": (1, 32),
        "values": (255, 127),
    },
    {
        "name": "tprint_tile_shape_header",
        "np_dtype": np.float32,
        "shape": (2, 8),
        "values": (1.0,),
    },
    {
        "name": "tprint_overload_with_tmp",
        "np_dtype": np.float32,
        "shape": (1, 8),
        "values": (3.141592,),
        "use_tmp": True,
        "print_format": "width10_precision6",
    },
]


def _make_inputs(spec):
    data = np.zeros(spec["shape"], dtype=spec["np_dtype"])
    flat = data.reshape(-1)
    for idx, value in enumerate(spec["values"]):
        flat[idx] = value
    out = np.zeros(spec["shape"], dtype=spec["np_dtype"])
    if spec.get("use_tmp", False):
        tmp = np.zeros(spec["shape"], dtype=spec["np_dtype"])
        return [data, tmp, out]
    return [data, out]


def _make_kernel(spec):
    dtype = _NP_TO_PTO[np.dtype(spec["np_dtype"])]
    rows, cols = spec["shape"]
    use_tmp = spec.get("use_tmp", False)
    print_format = spec.get("print_format")
    kernel_name = spec["name"]

    if use_tmp:
        @pto.jit(name=kernel_name, target="a5", backend="emitc")
        def _kernel(
            src_ptr: pto.ptr(dtype, "gm"),
            tmp_ptr: pto.ptr(dtype, "gm"),
            out_ptr: pto.ptr(dtype, "gm"),
        ):
            src_view = pto.make_tensor_view(src_ptr, shape=[rows, cols], strides=[cols, 1])
            tmp_view = pto.make_tensor_view(tmp_ptr, shape=[rows, cols], strides=[cols, 1])
            out_view = pto.make_tensor_view(out_ptr, shape=[rows, cols], strides=[cols, 1])
            tmp = pto.partition_view(tmp_view, offsets=[0, 0], sizes=[rows, cols])
            tile = pto.alloc_tile(shape=[rows, cols], dtype=dtype)

            pto.tile.load(src_view, tile)
            pto.tile.print(tile, tmp=tmp, print_format=print_format)
            pto.tile.store(tile, out_view)
    else:
        @pto.jit(name=kernel_name, target="a5", backend="emitc")
        def _kernel(src_ptr: pto.ptr(dtype, "gm"), out_ptr: pto.ptr(dtype, "gm")):
            src_view = pto.make_tensor_view(src_ptr, shape=[rows, cols], strides=[cols, 1])
            out_view = pto.make_tensor_view(out_ptr, shape=[rows, cols], strides=[cols, 1])
            tile = pto.alloc_tile(shape=[rows, cols], dtype=dtype)

            pto.tile.load(src_view, tile)
            pto.tile.print(tile, print_format=print_format)
            pto.tile.store(tile, out_view)

    return _kernel


def _make_case(spec):
    inputs = _make_inputs(spec)
    return inputs, inputs[0]


def _check_case(device_inputs, golden):
    actual = device_inputs[-1].cpu().numpy()
    assert_close(actual, golden, rtol=1e-6, atol=1e-6)


CASES = []
for _spec in CASE_SPECS:
    CASES.append(
        {
            "name": _spec["name"],
            "kernel": _make_kernel(_spec),
            "make_case": lambda _spec=_spec: _make_case(_spec),
            "check": _check_case,
        }
    )


auto_main(globals())
