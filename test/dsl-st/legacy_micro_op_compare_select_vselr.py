#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Migrated from test/vpto/cases/micro-op/compare-select/vselr."""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

from common import auto_main
from ptodsl import pto

KERNEL_SOURCE = (
    'module attributes {pto.target_arch = "a5", pto.kernel_kind = #pto.kernel_kind<vector>} {\n    func.func @vse'
    'lr_deep_merged_kernel(%arg0: !pto.ptr<f16, gm>, %arg1: !pto.ptr<ui16, gm>, %arg2: !pto.ptr<f16, gm>, %arg3: '
    '!pto.ptr<ui8, gm>, %arg4: !pto.ptr<ui8, gm>, %arg5: !pto.ptr<ui8, gm>, %arg6: !pto.ptr<f32, gm>, %arg7: !pto'
    '.ptr<i32, gm>, %arg8: !pto.ptr<f32, gm>) attributes {pto.kernel} {\n    %__deep_merge_guard = arith.constant'
    ' false\n    // inactive merged from vselr_f16_kernel_2d\n    scf.if %__deep_merge_guard {\n\n    %c0_m0 = ar'
    'ith.constant 0 : index\n    %c1_m0 = arith.constant 1 : index\n    %c128_m0 = arith.constant 128 : index\n  '
    '  %c1024_m0 = arith.constant 1024 : index\n    %c0_i64_m0 = arith.constant 0 : i64\n    %c1_i64_m0 = arith.c'
    'onstant 1 : i64\n    %c8_i64_m0 = arith.constant 8 : i64\n    %c256_i64_m0 = arith.constant 256 : i64\n    %'
    'c2048_i64_m0 = arith.constant 2048 : i64\n    %c4096_i64_m0 = arith.constant 4096 : i64\n    %c1024_i32_m0 ='
    ' arith.constant 1024 : i32\n    %false_m0 = arith.constant false\n\n    %ub_src_m0 = pto.castptr %c0_i64_m0 '
    ': i64 -> !pto.ptr<f16, ub>\n    %ub_idx_m0 = pto.castptr %c2048_i64_m0 : i64 -> !pto.ptr<ui16, ub>\n    %ub_'
    'out_m0 = pto.castptr %c4096_i64_m0 : i64 -> !pto.ptr<f16, ub>\n    pto.mte_gm_ub %arg0, %ub_src_m0, %c0_i64_'
    'm0, %c256_i64_m0\n      nburst(%c8_i64_m0, %c256_i64_m0, %c256_i64_m0)\n      : !pto.ptr<f16, gm>, !pto.ptr<'
    'f16, ub>, i64, i64, i64, i64, i64\n    pto.mte_gm_ub %arg1, %ub_idx_m0, %c0_i64_m0, %c256_i64_m0\n      nbur'
    'st(%c8_i64_m0, %c256_i64_m0, %c256_i64_m0)\n      : !pto.ptr<ui16, gm>, !pto.ptr<ui16, ub>, i64, i64, i64, i'
    '64, i64\n\n    pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n    pto.wait_flag["PIPE_MTE2", "PIPE_V", "E'
    'VENT_ID0"]\n\n    pto.vecscope {\n      %__m0:1 = scf.for %offset_m0 = %c0_m0 to %c1024_m0 step %c128_m0 ite'
    'r_args(%remaining_m0 = %c1024_i32_m0) -> (i32) {\n        %mask_m0, %next_remaining_m0 = pto.plt_b16 %remain'
    'ing_m0 : i32 -> !pto.mask<b16>, i32\n        %src_m0 = pto.vlds %ub_src_m0[%offset_m0] : !pto.ptr<f16, ub>, index -'
    '> !pto.vreg<128xf16>\n        %idx_m0 = pto.vlds %ub_idx_m0[%offset_m0] : !pto.ptr<ui16, ub>, index -> !pto.vreg<12'
    '8xui16>\n        %out_m0 = pto.vselr %src_m0, %idx_m0 : !pto.vreg<128xf16>, !pto.vreg<128xui16> -> !pto.vreg'
    '<128xf16>\n        pto.vsts %out_m0, %ub_out_m0[%offset_m0], %mask_m0 : !pto.vreg<128xf16>, !pto.ptr<f16, ub'
    '>, index, !pto.mask<b16>\n        scf.yield %next_remaining_m0 : i32\n      }\n    }\n\n    pto.set_flag["PIPE_V", '
    '"PIPE_MTE3", "EVENT_ID0"]\n    pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n    pto.mte_ub_gm %ub_out_'
    'm0, %arg2, %c256_i64_m0\n      nburst(%c8_i64_m0, %c256_i64_m0, %c256_i64_m0)\n      : !pto.ptr<f16, ub>, !p'
    'to.ptr<f16, gm>, i64, i64, i64, i64\n    pto.barrier #pto.pipe<PIPE_ALL>\n\n    }\n    // inactive merged fr'
    'om vselr_u8_kernel_2d\n    scf.if %__deep_merge_guard {\n\n    %c0_m1 = arith.constant 0 : index\n    %c1_m1'
    ' = arith.constant 1 : index\n    %c256_m1 = arith.constant 256 : index\n    %c1024_m1 = arith.constant 1024 '
    ': index\n    %c0_i64_m1 = arith.constant 0 : i64\n    %c1_i64_m1 = arith.constant 1 : i64\n    %c4_i64_m1 = '
    'arith.constant 4 : i64\n    %c256_i64_m1 = arith.constant 256 : i64\n    %c1024_i64_m1 = arith.constant 1024'
    ' : i64\n    %c2048_i64_m1 = arith.constant 2048 : i64\n    %c1024_i32_m1 = arith.constant 1024 : i32\n    %f'
    'alse_m1 = arith.constant false\n\n    %ub_src_m1 = pto.castptr %c0_i64_m1 : i64 -> !pto.ptr<ui8, ub>\n    %u'
    'b_idx_m1 = pto.castptr %c1024_i64_m1 : i64 -> !pto.ptr<ui8, ub>\n    %ub_out_m1 = pto.castptr %c2048_i64_m1 '
    ': i64 -> !pto.ptr<ui8, ub>\n    pto.mte_gm_ub %arg3, %ub_src_m1, %c0_i64_m1, %c256_i64_m1\n      nburst(%c4_'
    'i64_m1, %c256_i64_m1, %c256_i64_m1)\n      : !pto.ptr<ui8, gm>, !pto.ptr<ui8, ub>, i64, i64, i64, i64, i64\n'
    '    pto.mte_gm_ub %arg4, %ub_idx_m1, %c0_i64_m1, %c256_i64_m1\n      nburst(%c4_i64_m1, %c256_i64_m1, %c256_'
    'i64_m1)\n      : !pto.ptr<ui8, gm>, !pto.ptr<ui8, ub>, i64, i64, i64, i64, i64\n\n    pto.set_flag["PIPE_MTE'
    '2", "PIPE_V", "EVENT_ID0"]\n    pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n\n    pto.vecscope {\n   '
    '   %__m1:1 = scf.for %offset_m1 = %c0_m1 to %c1024_m1 step %c256_m1 iter_args(%remaining_m1 = %c1024_i32_m1)'
    ' -> (i32) {\n        %mask_m1, %next_remaining_m1 = pto.plt_b8 %remaining_m1 : i32 -> !pto.mask<b8>, i32\n  '
    '      %src_m1 = pto.vlds %ub_src_m1[%offset_m1] : !pto.ptr<ui8, ub>, index -> !pto.vreg<256xui8>\n        %idx_m1 ='
    ' pto.vlds %ub_idx_m1[%offset_m1] : !pto.ptr<ui8, ub>, index -> !pto.vreg<256xui8>\n        %out_m1 = pto.vselr %src'
    '_m1, %idx_m1 : !pto.vreg<256xui8>, !pto.vreg<256xui8> -> !pto.vreg<256xui8>\n        pto.vsts %out_m1, %ub_o'
    'ut_m1[%offset_m1], %mask_m1 : !pto.vreg<256xui8>, !pto.ptr<ui8, ub>, index, !pto.mask<b8>\n        scf.yield %next_'
    'remaining_m1 : i32\n      }\n    }\n\n    pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n    pto.wait_fla'
    'g["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n    pto.mte_ub_gm %ub_out_m1, %arg5, %c256_i64_m1\n      nburst(%c4_i'
    '64_m1, %c256_i64_m1, %c256_i64_m1)\n      : !pto.ptr<ui8, ub>, !pto.ptr<ui8, gm>, i64, i64, i64, i64\n    pt'
    'o.barrier #pto.pipe<PIPE_ALL>\n\n    }\n    // active merged from vselr_kernel_2d\n\n    %c8192_i64_m2 = ari'
    'th.constant 8192 : i64\n    %c4096_i64_m2 = arith.constant 4096 : i64\n    %c0_i64_m2 = arith.constant 0 : i'
    '64\n    %c32_m2 = arith.constant 32 : index\n    %0 = pto.castptr %c0_i64_m2 : i64 -> !pto.ptr<f32, ub>\n   '
    ' %1 = arith.index_castui %c32_m2 : index to i64\n    %c1_i64_m2 = arith.constant 1 : i64\n    %c32_i64_m2 = '
    'arith.constant 32 : i64\n    %c4_i64_m2 = arith.constant 4 : i64\n    %2 = arith.muli %1, %c4_i64_m2 : i64\n'
    '    %c128_i64_m2 = arith.constant 128 : i64\n    %3 = pto.castptr %arg6 : !pto.ptr<f32, gm> -> !pto.ptr<i8, '
    'gm>\n    %4 = arith.index_castui %c0_i64_m2 : i64 to index\n    %5 = pto.addptr %3, %4 : <i8, gm>, index -> <i8, gm'
    '>\n    pto.set_loop2_stride_outtoub %c4096_i64_m2, %c4096_i64_m2 : i64, i64\n    pto.set_loop1_stride_outtou'
    'b %c4096_i64_m2, %c4096_i64_m2 : i64, i64\n    %6 = pto.castptr %5 : !pto.ptr<i8, gm> -> !pto.ptr<f32, gm>\n'
    '    %false_m2 = arith.constant false\n    pto.mte_gm_ub %6, %0, %c0_i64_m2, %2\n      nburst(%c32_i64_m2, %c'
    '128_i64_m2, %c128_i64_m2)\n      : !pto.ptr<f32, gm>, !pto.ptr<f32, ub>, i64, i64, i64, i64, i64\n    %7 = p'
    'to.castptr %c4096_i64_m2 : i64 -> !pto.ptr<i32, ub>\n    %8 = pto.castptr %arg7 : !pto.ptr<i32, gm> -> !pto.'
    'ptr<i8, gm>\n    %9 = pto.addptr %8, %4 : <i8, gm>, index -> <i8, gm>\n    pto.set_loop2_stride_outtoub %c4096_i64_'
    'm2, %c4096_i64_m2 : i64, i64\n    pto.set_loop1_stride_outtoub %c4096_i64_m2, %c4096_i64_m2 : i64, i64\n    '
    '%10 = pto.castptr %9 : !pto.ptr<i8, gm> -> !pto.ptr<i32, gm>\n    pto.mte_gm_ub %10, %7, %c0_i64_m2, %2\n   '
    '   nburst(%c32_i64_m2, %c128_i64_m2, %c128_i64_m2)\n      : !pto.ptr<i32, gm>, !pto.ptr<i32, ub>, i64, i64, '
    'i64, i64, i64\n    pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n    pto.wait_flag["PIPE_MTE2", "PIPE_V"'
    ', "EVENT_ID0"]\n    %11 = pto.castptr %c8192_i64_m2 : i64 -> !pto.ptr<f32, ub>\n    %c0_m2 = arith.constant '
    '0 : index\n    %c1_m2 = arith.constant 1 : index\n    %c16_m2 = arith.constant 16 : index\n    %c64_m2 = ari'
    'th.constant 64 : index\n    %c128_m2 = arith.constant 128 : index\n    %c1024_i32_m2 = arith.constant 1024 :'
    ' i32\n    pto.vecscope {\n      %16 = scf.for %arg4_m2 = %c0_m2 to %c16_m2 step %c1_m2 iter_args(%arg5_m2 = '
    '%c1024_i32_m2) -> (i32) {\n        %17 = arith.muli %arg4_m2, %c64_m2 : index\n        %mask_m2, %scalar_out'
    '_m2 = pto.plt_b32 %arg5_m2 : i32 -> !pto.mask<b32>, i32\n        %25 = pto.vlds %0[%17] : !pto.ptr<f32, ub>, index '
    '-> !pto.vreg<64xf32>\n        %26 = pto.vlds %7[%17] : !pto.ptr<i32, ub>, index -> !pto.vreg<64xi32>\n        %27 ='
    ' pto.vselr %25, %26 : !pto.vreg<64xf32>, !pto.vreg<64xi32> -> !pto.vreg<64xf32>\n        pto.vsts %27, %11[%'
    '17], %mask_m2 : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, index, !pto.mask<b32>\n        scf.yield %scalar_out_m2 : i32'
    '\n      }\n    }\n    pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n    pto.wait_flag["PIPE_V", "PIPE_MT'
    'E3", "EVENT_ID0"]\n    %c1024_i64_m2 = arith.constant 1024 : i64\n    %12 = arith.muli %1, %c4_i64_m2 : i64'
    '\n    %13 = pto.castptr %arg8 : !pto.ptr<f32, gm> -> !pto.ptr<i8, gm>\n    %14 = pto.addptr %13, %4 : <i8, g'
    'm>, index -> <i8, gm>\n    pto.set_loop1_stride_ubtoout %c4096_i64_m2, %c4096_i64_m2 : i64, i64\n    pto.set_loop2_'
    'stride_ubtoout %c4096_i64_m2, %c4096_i64_m2 : i64, i64\n    %15 = pto.castptr %14 : !pto.ptr<i8, gm> -> !pto'
    '.ptr<f32, gm>\n    pto.mte_ub_gm %11, %15, %12\n      nburst(%c32_i64_m2, %c128_i64_m2, %c128_i64_m2)\n     '
    ' : !pto.ptr<f32, ub>, !pto.ptr<f32, gm>, i64, i64, i64, i64\n    pto.barrier #pto.pipe<PIPE_ALL>\n\n    retu'
    'rn\n  }\n}\n'
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

    # case: micro-op/compare-select/vselr
    # family: compare-select
    # target_ops: pto.vselr
    # scenarios: core-f32, full-mask, explicit-lane-index
    # coding=utf-8

    import argparse
    from pathlib import Path

    import numpy as np


    ROWS = 32
    COLS = 32
    SEED = 19


    def _generate(seed: int) -> dict:
        rng = np.random.default_rng(seed)
        v1 = rng.uniform(-3.0, 3.0, size=(ROWS, COLS)).astype(np.float32, copy=False)
        src = v1.reshape(16, 64)
        lane_ids = np.arange(64, dtype=np.int32)
        idx = np.empty((16, 64), dtype=np.int32)
        for row in range(16):
            idx[row] = (lane_ids[::-1] + row * 3 + (lane_ids // 8) * 3) % 64
        golden_v3 = np.take_along_axis(src, idx, axis=1).astype(np.float32, copy=False).reshape(ROWS, COLS)
        v3 = np.zeros((ROWS, COLS), dtype=np.float32)

        _files["v1.bin"] = v1.reshape(-1)
        _files["v2.bin"] = idx.reshape(-1)
        _files["v3.bin"] = v3.reshape(-1)
        _files["golden_v3.bin"] = golden_v3.reshape(-1)


    def main() -> None:
        parser = argparse.ArgumentParser(
            description="Generate numpy-based inputs/golden for VPTO micro-op vselr validation."
        )
        parser.add_argument(
            "--output-dir",
            type=Path,
            default=Path("."),
            help="Directory where v1.bin/v2.bin/v3.bin/golden_v3.bin are written.",
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=SEED,
            help="Numpy random seed.",
        )
        args = parser.parse_args()

        generate(args.output_dir, args.seed)


    return locals()

def _compare_scope():
    # Copyright (c) 2026 Huawei Technologies Co., Ltd.
    # This program is free software, you can redistribute it and/or modify it under the terms and conditions of
    # CANN Open Software License Agreement Version 2.0 (the "License").
    # Please refer to the License for details. You may not use this file except in compliance with the License.
    # THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
    # INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
    # See LICENSE in the root of the software repository for the full text of the License.

    # case: micro-op/compare-select/vselr
    # family: compare-select
    # target_ops: pto.vselr
    # scenarios: core-f32, full-mask, explicit-lane-index

    import os
    import sys
    import numpy as np

    def compare_tensor(golden_path, output_path):
        if not os.path.exists(golden_path) or not os.path.exists(output_path):
            return False
        golden = np.fromfile(golden_path, dtype=np.float32)
        output = np.fromfile(output_path, dtype=np.float32)
        if golden.shape != output.shape:
            return False
        if not np.allclose(golden, output, rtol=0.0, atol=0.0):
            diff = np.nonzero(golden != output)[0]
            idx = int(diff[0]) if diff.size else 0
            print(f"[ERROR] Mismatch: idx={idx} golden={float(golden[idx])} out={float(output[idx])}")
            return False
        return True


    def main():
        strict = os.getenv("COMPARE_STRICT", "1") != "0"
        ok = compare_tensor("golden_v3.bin", "v3.bin")
        if not ok:
            if strict:
                print("[ERROR] compare failed")
                sys.exit(2)
            print("[WARN] compare failed (non-gating)")
            return
        print("[INFO] compare passed")


    return locals()

HELPERS = {}
SEED_EXPR = 'SEED'
COMPARE_PAIRS = [
    (
        'golden_v3.bin',
        'v3.bin'
    )
]

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

# launch.cpp binds some host buffers to several kernel arguments; ARG_TO_BUFFER
# says which distinct buffer each kernel argument receives, so the module can
# pass aliased device tensors. BUFFER_FILES lists those distinct buffers.
BUFFER_FILES = [
    (
        'v1',
        'f16'
    ),
    (
        'v2',
        'i32'
    ),
    (
        'v3',
        'f32'
    )
]
ARG_TO_BUFFER = [
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    2
]
ARG_DTYPES = [
    'f16',
    'ui16',
    'f16',
    'ui8',
    'ui8',
    'ui8',
    'f32',
    'i32',
    'f32'
]
SCALARS = []

def _dtype_map():
    import torch
    return {
        "int8": torch.int8, "int16": torch.int16, "int32": torch.int32, "int64": torch.int64,
        "uint8": torch.uint8, "uint16": torch.int16, "uint32": torch.int32, "uint64": torch.int64,
        "float16": torch.float16, "float32": torch.float32, "float64": torch.float64,
    }


@pto.jit(
    name='vselr_deep_merged_kernel',
    target="a5",
    backend="vpto",
    mode="explicit",
    source=KERNEL_SOURCE,
)
def kernel(
    arg0: pto.ptr(pto.f16, "gm"),
    arg1: pto.ptr(pto.ui16, "gm"),
    arg2: pto.ptr(pto.f16, "gm"),
    arg3: pto.ptr(pto.ui8, "gm"),
    arg4: pto.ptr(pto.ui8, "gm"),
    arg5: pto.ptr(pto.ui8, "gm"),
    arg6: pto.ptr(pto.f32, "gm"),
    arg7: pto.ptr(pto.i32, "gm"),
    arg8: pto.ptr(pto.f32, "gm"),
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
    base, per_arg = {}, []
    for index, ty in zip(ARG_TO_BUFFER, ARG_DTYPES):
        name, _buf_ty = BUFFER_FILES[index]
        key = "%s.bin" % name
        if index not in base:
            if key in produced:
                # One host buffer can back several kernel arguments of
                # different widths, so keep the golden bytes and let each
                # argument reinterpret them below instead of narrowing the
                # shared buffer to the first argument type.
                array = np.ascontiguousarray(np.asarray(produced[key]))
            else:
                golden_key = output_to_golden.get(key)
                if golden_key is None or golden_key not in produced:
                    raise RuntimeError("golden did not produce %s" % key)
                array = np.ascontiguousarray(
                    np.zeros_like(np.asarray(produced[golden_key]), dtype=np.dtype(NUMPY_DTYPE[ty])))
            base[index] = torch.from_numpy(array).to("npu:0")
        tensor = base[index]
        want = _dtype_map()[NUMPY_DTYPE[ty]]
        if tensor.dtype != want and tensor.numel() * tensor.element_size() % want.itemsize == 0:
            tensor = tensor.view(want)
        per_arg.append(tensor)
    return per_arg, produced, list(SCALARS)


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
    seen = {}
    for index, tensor in zip(ARG_TO_BUFFER, device_inputs):
        seen.setdefault(index, tensor)
    for index, tensor in seen.items():
        name, _ty = BUFFER_FILES[index]
        files["%s.bin" % name] = tensor.cpu().numpy()
    _apply_write_masks(files, produced)
    _run_compare(files)


GRID_FROM_VALUES = None


CASES = [dict(
    name='legacy_micro_op_compare_select_vselr',
    grid=1,
    kernel=kernel,
    make_case=make_case,
    check=check_case,
)]

auto_main(globals())
