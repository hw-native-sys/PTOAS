#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Migrated from test/vpto/cases/vmi_new/kernels/dynamic-quant-perchannel-f16-8x128."""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

from common import auto_main
from ptodsl import pto

KERNEL_SOURCE = (
    '// Copyright (c) 2026 Huawei Technologies Co., Ltd.\n// This program is free software, you can redistribute '
    'it and/or modify it under the terms and conditions of\n// CANN Open Software License Agreement Version 2.0 ('
    'the "License").\n// Please refer to the License for details. You may not use this file except in compliance '
    'with the License.\n// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER '
    'EXPRESS OR IMPLIED,\n// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PAR'
    'TICULAR PURPOSE.\n// See LICENSE in the root of the software repository for the full text of the License.\n'
    '\nmodule attributes {pto.target_arch = "a5", pto.kernel_kind = #pto.kernel_kind<vector>} {\n  func.func @vmi'
    '_dynamic_quant_perchannel_f16_8x128_kernel(\n      %src_gm: !pto.ptr<f16, gm>, %scale_gm: !pto.ptr<f32, gm>,'
    '\n      %out_gm: !pto.ptr<ui8, gm>) attributes {pto.kernel} {\n    %c0 = arith.constant 0 : index\n    %c1 ='
    ' arith.constant 1 : index\n    %c2 = arith.constant 2 : index\n    %c8 = arith.constant 8 : index\n    %c4 ='
    ' arith.constant 4 : index\n    %c32 = arith.constant 32 : index\n    %c64 = arith.constant 64 : index\n    %'
    'c128 = arith.constant 128 : index\n    %c256 = arith.constant 256 : index\n    %c0_i32 = arith.constant 0 : '
    'i32\n    %c7_i32 = arith.constant 7 : i32\n    %c127_i32 = arith.constant 127 : i32\n    %c8_i32 = arith.con'
    'stant 8 : i32\n    %c128_i32 = arith.constant 128 : i32\n    %c256_i32 = arith.constant 256 : i32\n    %c0_f'
    '32 = arith.constant 0.000000e+00 : f32\n    %c-128_f32 = arith.constant -1.280000e+02 : f32\n    %c127_f32 ='
    ' arith.constant 1.270000e+02 : f32\n    %c0_i64 = arith.constant 0 : i64\n    %c1_i64 = arith.constant 1 : i'
    '64\n    %c128_i64 = arith.constant 128 : i64\n    %c256_i64 = arith.constant 256 : i64\n    %c512_i64 = arit'
    'h.constant 512 : i64\n    %c1024_i64 = arith.constant 1024 : i64\n    %c2048_i64 = arith.constant 2048 : i64'
    '\n    %c4096_i64 = arith.constant 4096 : i64\n    %c8192_i64 = arith.constant 8192 : i64\n    %c16384_i64 = '
    'arith.constant 16384 : i64\n    %c32768_i64 = arith.constant 32768 : i64\n    %c98304_i64 = arith.constant 9'
    '8304 : i64\n    %c102400_i64 = arith.constant 102400 : i64\n    %c106496_i64 = arith.constant 106496 : i64\n'
    '\n    %ub_src = pto.castptr %c0_i64 : i64 -> !pto.ptr<f16, ub>\n    %ub_scratch = pto.castptr %c32768_i64 : '
    'i64 -> !pto.ptr<f32, ub>\n    %ub_scale_padded = pto.castptr %c98304_i64 : i64 -> !pto.ptr<f32, ub>\n    %ub'
    '_scale = pto.castptr %c102400_i64 : i64 -> !pto.ptr<f32, ub>\n    %ub_out = pto.castptr %c106496_i64 : i64 -'
    '> !pto.ptr<ui8, ub>\n\n    pto.mte_gm_ub %src_gm, %ub_src, %c0_i64, %c2048_i64\n      nburst(%c1_i64, %c2048'
    '_i64, %c2048_i64)\n      : !pto.ptr<f16, gm>, !pto.ptr<f16, ub>, i64, i64, i64, i64, i64\n\n    pto.set_flag'
    '["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n    pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n\n    pto.vecsc'
    'ope {\n      scf.for %row = %c0 to %c8 step %c1 {\n        %elem_offset = arith.muli %row, %c128 : index\n  '
    '      %x_f16 = pto.vmi.vload %ub_src[%elem_offset]\n            : !pto.ptr<f16, ub>, index -> !pto.vmi.vreg<128xf1'
    '6>\n        %x_f32 = pto.vmi.vcvt %x_f16\n            : !pto.vmi.vreg<128xf16> -> !pto.vmi.vreg<128xf32>\n   '
    '     pto.vmi.vstore %x_f32, %ub_scratch[%elem_offset]\n            : !pto.vmi.vreg<128xf32>, !pto.ptr<f32, u'
    'b>, index\n      }\n\n      scf.for %col = %c0 to %c128 step %c1 {\n        %col_i32 = arith.index_cast %col : ind'
    'ex to i32\n        %mask = pto.vmi.create_mask %c8 : index -> !pto.vmi.mask<128xpred>\n        %init = pto.vm'
    'i.vbrc %c0_f32\n            : f32 -> !pto.vmi.vreg<1xf32>\n        %lane = pto.vmi.vci %c0_i32\n            '
    ': i32 -> !pto.vmi.vreg<128xi32>\n        %stride = pto.vmi.vbrc %c128_i32\n            : i32 -> !pto.vmi.vre'
    'g<128xi32>\n        %row_offsets = pto.vmi.vmul %lane, %stride\n            : !pto.vmi.vreg<128xi32>, !pto.v'
    'mi.vreg<128xi32>\n            -> !pto.vmi.vreg<128xi32>\n        %col_vec = pto.vmi.vbrc %col_i32\n         '
    '   : i32 -> !pto.vmi.vreg<128xi32>\n        %indices = pto.vmi.vadd %row_offsets, %col_vec\n            : !p'
    'to.vmi.vreg<128xi32>, !pto.vmi.vreg<128xi32>\n            -> !pto.vmi.vreg<128xi32>\n        %col_values = p'
    'to.vmi.vgather %ub_scratch, %indices, %mask\n            : !pto.ptr<f32, ub>, !pto.vmi.vreg<128xi32>,\n     '
    '         !pto.vmi.mask<128xpred>\n            -> !pto.vmi.vreg<128xf32>\n        %abs = pto.vmi.vabs %col_va'
    'lues\n            : !pto.vmi.vreg<128xf32> -> !pto.vmi.vreg<128xf32>\n        %amax = pto.vmi.vcmax %abs, %m'
    'ask\n            : !pto.vmi.vreg<128xf32>,\n              !pto.vmi.mask<128xpred> -> !pto.vmi.vreg<1xf32>\n '
    '       %max_int8 = pto.vmi.vbrc %c127_f32\n            : f32 -> !pto.vmi.vreg<1xf32>\n        %scale = pto.v'
    'mi.vdiv %amax, %max_int8\n            : !pto.vmi.vreg<1xf32>, !pto.vmi.vreg<1xf32>\n            -> !pto.vmi.'
    'vreg<1xf32>\n        %padded_col = arith.muli %col, %c8 : index\n        pto.vmi.vstore %scale, %ub_scale_pa'
    'dded[%padded_col]\n            : !pto.vmi.vreg<1xf32>, !pto.ptr<f32, ub>, index\n      }\n\n      %scale_mask = pt'
    'o.vmi.create_mask %c128 : index -> !pto.vmi.mask<128xpred>\n      %scale_lane = pto.vmi.vci %c0_i32\n        '
    '  : i32 -> !pto.vmi.vreg<128xi32>\n      %scale_stride = pto.vmi.vbrc %c8_i32\n          : i32 -> !pto.vmi.v'
    'reg<128xi32>\n      %scale_indices = pto.vmi.vmul %scale_lane, %scale_stride\n          : !pto.vmi.vreg<128x'
    'i32>, !pto.vmi.vreg<128xi32>\n          -> !pto.vmi.vreg<128xi32>\n      %scale_dense = pto.vmi.vgather %ub_'
    'scale_padded, %scale_indices, %scale_mask\n          : !pto.ptr<f32, ub>, !pto.vmi.vreg<128xi32>,\n         '
    '   !pto.vmi.mask<128xpred>\n          -> !pto.vmi.vreg<128xf32>\n      pto.vmi.vstore %scale_dense, %ub_scal'
    'e[%c0]\n          : !pto.vmi.vreg<128xf32>, !pto.ptr<f32, ub>, index\n\n      %scale_pair_mask = pto.vmi.create_ma'
    'sk %c256 : index -> !pto.vmi.mask<256xpred>\n      %scale_pair_lane = pto.vmi.vci %c0_i32\n          : i32 ->'
    ' !pto.vmi.vreg<256xi32>\n      %scale_pair_group_mask = pto.vmi.vbrc %c127_i32\n          : i32 -> !pto.vmi.'
    'vreg<256xi32>\n      %scale_pair_col = pto.vmi.vand %scale_pair_lane, %scale_pair_group_mask\n          : !p'
    'to.vmi.vreg<256xi32>, !pto.vmi.vreg<256xi32>\n          -> !pto.vmi.vreg<256xi32>\n      %scale_pair_stride '
    '= pto.vmi.vbrc %c8_i32\n          : i32 -> !pto.vmi.vreg<256xi32>\n      %scale_pair_indices = pto.vmi.vmul '
    '%scale_pair_col, %scale_pair_stride\n          : !pto.vmi.vreg<256xi32>, !pto.vmi.vreg<256xi32>\n          -'
    '> !pto.vmi.vreg<256xi32>\n      %scale_vec = pto.vmi.vgather %ub_scale_padded, %scale_pair_indices, %scale_p'
    'air_mask\n          : !pto.ptr<f32, ub>, !pto.vmi.vreg<256xi32>,\n            !pto.vmi.mask<256xpred>\n     '
    '     -> !pto.vmi.vreg<256xf32>\n      scf.for %pair = %c0 to %c4 step %c1 {\n        %row = arith.muli %pair'
    ', %c2 : index\n        %elem_offset = arith.muli %row, %c128 : index\n        %x = pto.vmi.vload %ub_scratch'
    '[%elem_offset]\n            : !pto.ptr<f32, ub>, index -> !pto.vmi.vreg<256xf32>\n        %scaled = pto.vmi.vdiv %'
    'x, %scale_vec\n            : !pto.vmi.vreg<256xf32>, !pto.vmi.vreg<256xf32>\n            -> !pto.vmi.vreg<256'
    'xf32>\n        %lo = pto.vmi.vbrc %c-128_f32\n            : f32 -> !pto.vmi.vreg<256xf32>\n        %hi = pto'
    '.vmi.vbrc %c127_f32\n            : f32 -> !pto.vmi.vreg<256xf32>\n        %clamped_lo = pto.vmi.vmax %scaled'
    ', %lo\n            : !pto.vmi.vreg<256xf32>, !pto.vmi.vreg<256xf32>\n            -> !pto.vmi.vreg<256xf32>\n'
    '        %clamped = pto.vmi.vmin %clamped_lo, %hi\n            : !pto.vmi.vreg<256xf32>, !pto.vmi.vreg<256xf3'
    '2>\n            -> !pto.vmi.vreg<256xf32>\n        %i32 = pto.vmi.vcvt %clamped {saturate = "SAT"}\n        '
    '    : !pto.vmi.vreg<256xf32> -> !pto.vmi.vreg<256xsi32>\n        %zero_i32_bits = pto.vmi.vbrc %c0_i32\n    '
    '        : i32 -> !pto.vmi.vreg<256xi32>\n        %zero_i32 = pto.vmi.vinterpret_cast %zero_i32_bits\n       '
    '     : !pto.vmi.vreg<256xi32> -> !pto.vmi.vreg<256xsi32>\n        %byte_bias_bits = pto.vmi.vbrc %c256_i32\n'
    '            : i32 -> !pto.vmi.vreg<256xi32>\n        %byte_bias = pto.vmi.vinterpret_cast %byte_bias_bits\n '
    '           : !pto.vmi.vreg<256xi32> -> !pto.vmi.vreg<256xsi32>\n        %cmp_seed = pto.vmi.create_mask %c25'
    '6 : index -> !pto.vmi.mask<256xpred>\n        %neg = pto.vmi.vcmp %i32, %zero_i32, %cmp_seed {cmp = "lt"}\n '
    '           : !pto.vmi.vreg<256xsi32>, !pto.vmi.vreg<256xsi32>,\n              !pto.vmi.mask<256xpred> -> !pt'
    'o.vmi.mask<256xpred>\n        %wrapped = pto.vmi.vadd %i32, %byte_bias\n            : !pto.vmi.vreg<256xsi32'
    '>, !pto.vmi.vreg<256xsi32>\n            -> !pto.vmi.vreg<256xsi32>\n        %byte_i32 = pto.vmi.vsel %neg, %'
    'wrapped, %i32\n            : !pto.vmi.mask<256xpred>, !pto.vmi.vreg<256xsi32>,\n              !pto.vmi.vreg<'
    '256xsi32>\n            -> !pto.vmi.vreg<256xsi32>\n        %u8 = pto.vmi.vcvt %byte_i32 {saturate = "SAT"}\n'
    '            : !pto.vmi.vreg<256xsi32> -> !pto.vmi.vreg<256xui8>\n        pto.vmi.vstore %u8, %ub_out[%elem_o'
    'ffset]\n            : !pto.vmi.vreg<256xui8>, !pto.ptr<ui8, ub>, index\n      }\n    }\n\n    pto.set_flag["PIPE_V'
    '", "PIPE_MTE3", "EVENT_ID0"]\n    pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n    pto.mte_ub_gm %ub_sc'
    'ale, %scale_gm, %c512_i64\n      nburst(%c1_i64, %c512_i64, %c512_i64)\n      : !pto.ptr<f32, ub>, !pto.ptr<'
    'f32, gm>, i64, i64, i64, i64\n    pto.mte_ub_gm %ub_out, %out_gm, %c1024_i64\n      nburst(%c1_i64, %c1024_i'
    '64, %c1024_i64)\n      : !pto.ptr<ui8, ub>, !pto.ptr<ui8, gm>, i64, i64, i64, i64\n    pto.barrier #pto.pipe'
    '<PIPE_ALL>\n    return\n  }\n}\n'
)

def _golden_scope():
    _files = {}

    output_dir = None

    # Copyright (c) 2026 Huawei Technologies Co., Ltd.
    # This program is free software, you can redistribute it and/or modify it under the terms and conditions of
    # CANN Open Software License Agreement Version 2.0 (the "License").
    # Please refer to the License for details. You may not use this file except in compliance with the License.
    # THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
    # INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
    # See LICENSE in the root of the software repository for the full text of the License.

    import argparse
    from pathlib import Path

    import numpy as np

    ROWS = 8
    COLS = 128
    INT8_MAX = np.float32(127.0)
    SENTINEL_F32 = np.float32(-777.0)
    SENTINEL_U8 = np.uint8(0xA5)

    ROW_Q = np.array(
        [
            -127,
            -96,
            -64,
            -32,
            -7,
            -1,
            0,
            1,
            7,
            16,
            31,
            63,
            95,
            120,
            127,
            64,
        ],
        dtype=np.float32,
    )
    COL_SCALES = np.array([0.125, 0.25, 0.5, 1.0, 2.0], dtype=np.float32)


    def _generate(seed: int) -> dict:
        q = np.tile(ROW_Q, (ROWS + len(ROW_Q) - 1) // len(ROW_Q))[:ROWS]
        col_scales = np.tile(COL_SCALES, (COLS + len(COL_SCALES) - 1) // len(COL_SCALES))[:COLS]

        src = (q[:, None] * col_scales[None, :]).astype(np.float16)
        x_f32 = src.astype(np.float32)
        golden_scale = (np.max(np.abs(x_f32), axis=0) / INT8_MAX).astype(np.float32)
        scale_safe = np.where(golden_scale > 0, golden_scale, np.ones_like(golden_scale))
        golden_out = np.round(x_f32 / scale_safe[None, :]).astype(np.float32)
        golden_out = np.clip(golden_out, -128, 127).astype(np.int8)

        scale = np.full(COLS, SENTINEL_F32, dtype=np.float32)
        out = np.full((ROWS, COLS), SENTINEL_U8, dtype=np.uint8)

        _files["v1.bin"] = src.view(np.uint16)
        _files["v2.bin"] = scale
        _files["v3.bin"] = out
        _files["golden_v2.bin"] = golden_scale
        _files["golden_v3.bin"] = golden_out.view(np.uint8)


    def main() -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--output-dir", type=Path, default=Path("."))
        args = parser.parse_args()
        generate(args.output_dir)


    return locals()

def _compare_scope():
    # Copyright (c) 2026 Huawei Technologies Co., Ltd.
    # This program is free software, you can redistribute it and/or modify it under the terms and conditions of
    # CANN Open Software License Agreement Version 2.0 (the "License").
    # Please refer to the License for details. You may not use this file except in compliance with the License.
    # THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
    # INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
    # See LICENSE in the root of the software repository for the full text of the License.

    import sys

    import numpy as np


    def main() -> None:
        golden_scale = np.fromfile("golden_v2.bin", dtype=np.float32)
        scale = np.fromfile("v2.bin", dtype=np.float32)
        golden_out = np.fromfile("golden_v3.bin", dtype=np.uint8)
        out = np.fromfile("v3.bin", dtype=np.uint8)

        if golden_scale.shape != scale.shape or not np.allclose(
            golden_scale, scale, rtol=1.0e-6, atol=1.0e-6
        ):
            if golden_scale.shape != scale.shape:
                idx = -1
            else:
                diff = np.nonzero(~np.isclose(golden_scale, scale, rtol=1.0e-6, atol=1.0e-6))[0]
                idx = int(diff[0]) if diff.size else -1
            print(
                f"[ERROR] scale compare failed idx={idx} "
                f"golden={golden_scale[idx] if idx >= 0 else 'n/a'} "
                f"output={scale[idx] if idx >= 0 else 'n/a'}"
            )
            sys.exit(2)

        if golden_out.shape != out.shape or not np.array_equal(golden_out, out):
            diff = np.nonzero(golden_out != out)[0] if golden_out.shape == out.shape else []
            idx = int(diff[0]) if len(diff) else -1
            print(
                f"[ERROR] int8 compare failed idx={idx} "
                f"golden={int(golden_out[idx]) if idx >= 0 else 'n/a'} "
                f"output={int(out[idx]) if idx >= 0 else 'n/a'}"
            )
            sys.exit(2)

        print("[INFO] compare passed")


    return locals()

HELPERS = {}
SEED_EXPR = None
COMPARE_PAIRS = []

NUMPY_DTYPE = {
    'i8': 'int8',
    'i16': 'int16',
    'i32': 'int32',
    'i64': 'int64',
    'si8': 'int8',
    'si16': 'int16',
    'si32': 'int32',
    'si64': 'int64',
    'ui8': 'uint8',
    'ui16': 'uint16',
    'ui32': 'uint32',
    'ui64': 'uint64',
    'f16': 'float16',
    'f32': 'float32',
    'f64': 'float64',
    'bf16': 'uint16',
    'hif8': 'uint8',
    'f8E4M3FN': 'uint8',
    'f8E5M2': 'uint8',
    'f8e4m3': 'uint8',
    'f8e5m2': 'uint8',
    'f8e8m0': 'uint8',
    'f8E8M0FNU': 'uint8',
    'f4E2M1x2': 'uint8',
    'f4E1M2x2': 'uint8',
    'hif8x2': 'uint16',
    'f8e4m3x2': 'uint16',
    'f8e5m2x2': 'uint16',
    'f8e4m3x4': 'uint32',
    'f8e4m3x8': 'uint64',
    'f8e5m2x4': 'uint32',
    'f8e5m2x8': 'uint64',
    'f16x2': 'uint32',
    'bf16x2': 'uint32',
    'i8x2': 'uint16',
    'i16x2': 'uint32',
    'i32x2': 'uint64',
    'f32x2': 'uint64'
}


def _as_buffer_array(value, numpy_dtype):
    """Match a golden array to the raw bytes a kernel element type expects.

    A golden may build its array with ml_dtypes types (bfloat16) while the
    kernel argument is a plain numpy type (uint16); torch.from_numpy rejects
    the former, so reinterpret the bytes when the element widths agree.
    """
    array = np.asarray(value)
    want = np.dtype(numpy_dtype)
    if array.dtype == want:
        return array
    # The golden bytes are the device buffer; reinterpret them whenever the
    # byte count divides evenly so a narrow golden (for example uint8 init or
    # an f16 source) still fills the element type the kernel argument expects
    # instead of shrinking or doubling the buffer through astype.
    flat = np.ascontiguousarray(array).reshape(-1)
    if flat.nbytes % want.itemsize == 0:
        return flat.view(want)
    return array.astype(want)

BUFFERS = [
    (
        'v1',
        'f16',
        'src_gm'
    ),
    (
        'v2',
        'f32',
        'scale_gm'
    ),
    (
        'v3',
        'ui8',
        'out_gm'
    )
]
SCALARS = []
SCALAR_SOURCES = []
SCALAR_DTYPES = []


@pto.jit(
    name='vmi_dynamic_quant_perchannel_f16_8x128_kernel',
    target="a5",
    backend="vpto",
    mode="explicit",
    source=KERNEL_SOURCE,
)
def kernel(
    src_gm: pto.ptr(pto.f16, "gm"),
    scale_gm: pto.ptr(pto.f32, "gm"),
    out_gm: pto.ptr(pto.ui8, "gm"),
):
    pass


class _TrackedArray(np.ndarray):
    """An array that records the positions written through __setitem__."""

    def __new__(cls, array):
        obj = np.asarray(array).view(cls)
        obj._write_mask = np.zeros(obj.shape, dtype=bool)
        return obj

    def __array_finalize__(self, source):
        if source is None:
            return
        self._write_mask = getattr(source, "_write_mask", None)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        mask = getattr(self, "_write_mask", None)
        if mask is None or mask.shape != self.shape:
            mask = np.zeros(self.shape, dtype=bool)
            self._write_mask = mask
        mask[key] = True


def _materialize():
    """Run the case's own golden generator in memory; no .bin files.

    The golden's output buffers are created through the numpy constructors and
    then filled in place, so wrapping those constructors records exactly which
    positions the golden defines; the rest are don't care for the compare.
    """
    trackers = {name: getattr(np, name) for name in ("zeros", "full", "empty", "ones")}

    building = [False]

    def _track(name):
        original = trackers[name]

        def wrapped(*args, **kwargs):
            if building[0]:
                return original(*args, **kwargs)
            building[0] = True
            try:
                tracked = _TrackedArray(original(*args, **kwargs))
                if name != "empty":
                    # zeros/full/ones initialize every element, so the whole
                    # buffer is defined and compared; only np.empty leaves
                    # elements undefined, where the __setitem__ trace defines
                    # the valid mask.
                    tracked._write_mask = np.ones(tracked.shape, dtype=bool)
                return tracked
            finally:
                building[0] = False

        return wrapped

    for name in trackers:
        setattr(np, name, _track(name))
    try:
        namespace = _golden_scope()
        if "_generate" in namespace:
            seed = namespace[SEED_EXPR] if SEED_EXPR else 0
            namespace["_generate"](seed)
        else:
            sys.argv = ["golden.py"]
            namespace["main"]()
    finally:
        for name, original in trackers.items():
            setattr(np, name, original)
    produced = namespace["_files"]
    masks = {}
    for key, value in produced.items():
        if not key.startswith("golden_"):
            continue
        mask = getattr(value, "_write_mask", None)
        if mask is not None and np.asarray(mask).size and bool(np.asarray(mask).any()):
            masks[key] = np.asarray(mask).reshape(-1).copy()
        else:
            # a computed output defines every element; there is no partial
            # in-place write to record, so the whole file is compared.
            masks[key] = np.ones(np.asarray(value).size, dtype=bool)
    if masks:
        produced["@write_mask"] = masks
    return produced


def _apply_write_masks(files, produced):
    """Restrict each golden/output comparison to the golden's written set."""
    masks = produced.get("@write_mask") if hasattr(produced, "get") else None
    if not masks:
        return
    for golden_key, mask in masks.items():
        if not golden_key.startswith("golden_"):
            continue
        output_key = golden_key[len("golden_"):]
        if golden_key not in files or output_key not in files:
            continue
        golden = np.asarray(files[golden_key])
        output = np.asarray(files[output_key]).reshape(-1)
        flat = golden.reshape(-1)
        written = np.asarray(mask, dtype=bool).reshape(-1)
        size = min(flat.size, output.size, written.size)
        if size == 0 or not written[:size].any():
            continue
        adjusted = flat.copy()
        donotcare = ~written[:size]
        if output.dtype != flat.dtype and output.dtype.itemsize == flat.dtype.itemsize:
            # Same byte width but a different element type (for example the
            # kernel typed the buffer f16 while the golden built uint16): copy
            # the raw bytes so the donot-care fill stays bit-exact instead of
            # rounding through a numeric cast such as f16(-1.0) -> 0xffff.
            output = np.ascontiguousarray(output).view(flat.dtype)
        adjusted[:size][donotcare] = output[:size][donotcare]
        files[golden_key] = adjusted.reshape(golden.shape)


def _run_compare(files):
    """Run the case's own compare.py against in-memory arrays."""
    with tempfile.TemporaryDirectory(
            prefix="legacy-helpers-", dir=os.environ.get("TMPDIR")) as _helpers:
        _run_compare_in(files, Path(_helpers))


def _run_compare_in(files, helper_dir):
    for name, text in HELPERS.items():
        (helper_dir / name).write_text(text)

    real_fromfile, real_exists, real_argv = np.fromfile, os.path.exists, sys.argv
    real_exists_path = Path.exists
    real_environ, real_path = dict(os.environ), list(sys.path)

    def _fromfile(path, dtype=None, count=-1, offset=0, *args, **kwargs):
        name = os.path.basename(str(path))
        if name not in files:
            raise FileNotFoundError(name)
        # The legacy compare read the raw .bin bytes; reinterpret the stored
        # array's bytes instead of casting, so a deep-merged buffer written
        # through one element type still reads back as the type compare asks
        # for. count/offset mirror np.fromfile's element prefix semantics.
        array = np.asarray(files[name])
        data = array.tobytes()
        if offset:
            data = data[int(offset):]
        if dtype is None:
            dtype = array.dtype
        itemsize = np.dtype(dtype).itemsize
        available = len(data) // itemsize
        if count is None or int(count) < 0 or int(count) > available:
            count = available
        return np.frombuffer(data, dtype=dtype, count=count)

    np.fromfile = _fromfile
    os.path.exists = lambda p: os.path.basename(str(p)) in files or real_exists(p)
    Path.exists = lambda self: os.path.basename(str(self)) in files or real_exists_path(self)
    sys.argv = ["compare.py"]
    sys.path.insert(0, str(helper_dir))
    os.environ["COMPARE_STRICT"] = "1"
    try:
        namespace = _compare_scope()
        if "main" in namespace:
            namespace["main"]()
    except SystemExit as exc:
        if exc.code not in (0, None):
            raise AssertionError("compare.py exited with %r" % (exc.code,))
    finally:
        np.fromfile, os.path.exists, Path.exists, sys.argv = (
            real_fromfile, real_exists, real_exists_path, real_argv)
        os.environ.clear()
        os.environ.update(real_environ)
        sys.path[:] = real_path


def make_case():
    import torch
    produced = _materialize()
    output_to_golden = {out: gold for gold, out in COMPARE_PAIRS}
    arrays = []
    for name, ty, _slot in BUFFERS:
        if name.startswith("@same:"):
            arrays.append(np.asarray(produced["%s.bin" % name[len("@same:"):]]))
            continue
        if name.startswith("@scratch:"):
            arrays.append(np.empty_like(np.asarray(produced["%s.bin" % name[len("@scratch:"):]]),
                                        dtype=np.dtype(NUMPY_DTYPE[ty])))
            continue
        key = "%s.bin" % name
        if key in produced:
            arrays.append(_as_buffer_array(produced[key], NUMPY_DTYPE[ty]))
            continue
        golden_key = output_to_golden.get(key)
        if golden_key is None or golden_key not in produced:
            raise RuntimeError("golden did not produce %s" % key)
        arrays.append(np.zeros_like(np.asarray(produced[golden_key]),
                                    dtype=np.dtype(NUMPY_DTYPE[ty])))
    values = list(SCALARS)
    for _source, _ty in zip(SCALAR_SOURCES, SCALAR_DTYPES):
        _array = np.asarray(produced["%s.bin" % _source]).reshape(-1)
        values.append(float(_array[0]) if _ty == "f32" else int(_array[0]))
    if GRID_FROM_VALUES is not None:
        CASES[0]["grid"] = GRID_FROM_VALUES(values)
    return arrays, produced, values


def check_case(device_inputs, produced):
    if COMPARE_PAIRS:
        files = {gold: np.asarray(produced[gold]) for gold, _out in COMPARE_PAIRS if gold in produced}
    else:
        # no explicit pairs: the case compares whole files, so expose every
        # array the golden built (its .tofile names became dictionary keys).
        files = {name: np.asarray(value) for name, value in produced.items()
                 if not name.startswith("@")}
    if not files:
        raise AssertionError("golden.py produced no golden output")
    for _source in SCALAR_SOURCES:
        _key = "%s.bin" % _source
        if _key in produced:
            files[_key] = np.asarray(produced[_key])
    for (name, _ty, _slot), tensor in zip(BUFFERS, device_inputs):
        if not name.startswith("@"):
            files["%s.bin" % name] = tensor.cpu().numpy()
    _apply_write_masks(files, produced)
    _run_compare(files)


GRID_FROM_VALUES = None


CASES = [dict(
    name='legacy_vmi_new_kernels_dynamic_quant_perchannel_f16_8x128',
    grid=1,
    kernel=kernel,
    make_case=make_case,
    check=check_case,
)]

auto_main(globals())
