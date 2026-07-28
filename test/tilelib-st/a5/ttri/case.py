#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from pathlib import Path
import sys, numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import auto_main, golden_output_case
from ptodsl import pto

_DTYPE_MAP = {
    np.float32: {"ptodsl": pto.f32, "name": "f32", "bytes": 4},
    np.float16: {"ptodsl": pto.f16, "name": "f16", "bytes": 2},
    np.int32:   {"ptodsl": pto.i32, "name": "i32", "bytes": 4},
    np.int16:   {"ptodsl": pto.i16, "name": "i16", "bytes": 2},
    np.uint32:  {"ptodsl": pto.ui32, "name": "ui32", "bytes": 4},
    np.int8:    {"ptodsl": pto.i8, "name": "i8", "bytes": 1},
    np.uint8:   {"ptodsl": pto.ui8, "name": "ui8", "bytes": 1},
    np.uint16:  {"ptodsl": pto.ui16, "name": "ui16", "bytes": 2},
}

_BLOCK_BYTES = 32


def _ceil(val, align):
    return ((val + align - 1) // align) * align


def _aligned_cols(cols, elem_bytes):
    return _ceil(cols * elem_bytes, _BLOCK_BYTES) // elem_bytes

# (tile_shape, valid_shape, np.dtype, upper_or_lower, diagonal)
# upper_or_lower: "lower" (mask[i,j]=1 if j<=i+diagonal) or
#                  "upper" (mask[i,j]=1 if j>=i+diagonal)
# Static cases: tile_shape == valid_shape (matches TTRIParams in pto-isa reference)
# Dynamic cases: tile_shape != valid_shape (matches TTRIDynParams in pto-isa reference)
_TRIL_CASES = [
    # The 32x91 cases retain unaligned valid widths; the 32x64 cases cover the
    # aligned path without repeating the former 128x128 simulator workload.
    # ── static cases (16) ──
    ((20, 32),   (20, 32),   np.float16, "lower", 0),
    ((20, 32),   (20, 32),   np.uint8,   "lower", 0),
    ((32, 91),   (32, 91),   np.float32, "lower", 0),
    ((32, 64),   (32, 64),   np.float32, "lower", 0),
    ((32, 91),   (32, 91),   np.float32, "lower", 3),
    ((32, 64),   (32, 64),   np.float32, "lower", 3),
    ((32, 91),   (32, 91),   np.float32, "lower", -3),
    ((32, 64),   (32, 64),   np.float32, "lower", -3),
    ((32, 91),   (32, 91),   np.float32, "upper", 0),
    ((32, 64),   (32, 64),   np.float32, "upper", 0),
    ((32, 91),   (32, 91),   np.float32, "upper", 3),
    ((32, 64),   (32, 64),   np.float32, "upper", 3),
    ((32, 91),   (32, 91),   np.float32, "upper", -3),
    ((32, 64),   (32, 64),   np.float32, "upper", -3),
    ((48, 32),   (48, 32),   np.float32, "lower", -41),
    ((48, 32),   (48, 32),   np.float32, "upper", -41),

    # ── dynamic cases (9) ──
    ((4, 80),    (4, 80),    np.float16, "upper", 0),
    ((4, 80),    (3, 67),    np.float16, "upper", 0),
    ((48, 16),   (43, 16),   np.float16, "lower", -41),
    ((48, 16),   (48, 16),   np.float16, "lower", -41),
    ((48, 16),   (47, 16),   np.float16, "lower", -41),
    ((8, 64),    (8, 64),    np.int8,    "lower", 0),
    ((8, 64),    (5, 48),    np.int8,    "lower", 0),
    ((8, 16),    (1, 16),    np.float16, "lower", 0),
    ((8, 16),    (2, 16),    np.float16, "lower", 0),
]

_TRIL_KERNELS = {}
CASES = []

for _tile, _valid, _np_dtype, _upper, _diag in _TRIL_CASES:
    _tr, _tc = _tile
    _vr, _vc = _valid
    _elem_bytes = _DTYPE_MAP[_np_dtype]["bytes"]
    _ac = _aligned_cols(_tc, _elem_bytes)
    _pdsl = _DTYPE_MAP[_np_dtype]["ptodsl"]
    _dname = _DTYPE_MAP[_np_dtype]["name"]
    _ulab = _upper
    _dstr = str(_diag) if _diag >= 0 else f"m{abs(_diag)}"
    _kname = f"ttri_{_ulab}_{_dname}_{_vr}x{_vc}_d{_dstr}"

    def _make(tr=_tr, vr=_vr, vc=_vc, ac=_ac, pdsl=_pdsl, upper=_upper,
              diag=_diag, kname=_kname):
        @pto.jit(name=kname, target="a5")
        def _kernel(out_ptr: pto.ptr(pdsl, "gm")):
            out_view = pto.make_tensor_view(out_ptr, shape=[vr, vc], strides=[vc, 1])
            out_tile = pto.alloc_tile(shape=[tr, ac], dtype=pdsl,
                                      valid_shape=[vr, vc])
            pto.tile.tri(diag, out_tile, upper_or_lower=upper)
            pto.tile.store(out_tile, out_view)
        return _kernel
    _TRIL_KERNELS[_kname] = _make()

    _out_dtype = _np_dtype

    def _expected(np_dtype=_np_dtype, vs=_valid, up=_upper, dg=_diag,
                  out_dtype=_out_dtype):
        vr, vc = vs
        data = np.zeros((vr, vc), dtype=out_dtype)
        mask = (np.triu if up == "upper" else np.tril)(np.ones((vr, vc), dtype=out_dtype), k=dg)
        data[:vr, :vc] = mask
        return data

    CASES.append(golden_output_case(
        _kname, _TRIL_KERNELS[_kname],
        inputs=lambda: [],
        expected=_expected,
        output_shape=list(_valid),
        output_dtype=_out_dtype,
        rtol=0.0, atol=1e-6,
    ))

auto_main(globals())
