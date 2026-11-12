#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Migrated from test/vpto/cases/micro-op/binary-vector/vdiv."""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

from common import auto_main
from ptodsl import pto

KERNEL_SOURCE = (
    'module attributes {pto.target_arch = "a5", pto.kernel_kind = #pto.kernel_kind<vector>} {\n    func.func @vdi'
    'v_deep_merged_kernel(%arg0: !pto.ptr<f32, gm>, %arg1: !pto.ptr<f32, gm>, %arg2: !pto.ptr<f32, gm>, %arg3: !p'
    'to.ptr<f16, gm>, %arg4: !pto.ptr<f16, gm>, %arg5: !pto.ptr<f16, gm>, %arg6: !pto.ptr<f32, gm>, %arg7: !pto.p'
    'tr<f32, gm>, %arg8: !pto.ptr<f32, gm>, %arg9: !pto.ptr<f32, gm>, %arg10: !pto.ptr<f32, gm>, %arg11: !pto.ptr'
    '<f32, gm>) attributes {pto.kernel} {\n    // merged from div_kernel_2d via LaunchDiv_kernel_2d\n\n    %c0_m0'
    ' = arith.constant 0 : index\n    %c1_m0 = arith.constant 1 : index\n    %c64_m0 = arith.constant 64 : index'
    '\n    %c1024_m0 = arith.constant 1024 : index\n    %c0_i64_m0 = arith.constant 0 : i64\n    %c1_i64_m0 = ari'
    'th.constant 1 : i64\n    %c32_i64_m0 = arith.constant 32 : i64\n    %c128_i64_m0 = arith.constant 128 : i64'
    '\n    %c4096_i64_m0 = arith.constant 4096 : i64\n    %c8192_i64_m0 = arith.constant 8192 : i64\n    %c1024_i'
    '32_m0 = arith.constant 1024 : i32\n\n    %ub_lhs_m0 = pto.castptr %c0_i64_m0 : i64 -> !pto.ptr<f32, ub>\n   '
    ' %ub_rhs_m0 = pto.castptr %c4096_i64_m0 : i64 -> !pto.ptr<f32, ub>\n    %ub_out_m0 = pto.castptr %c8192_i64_'
    'm0 : i64 -> !pto.ptr<f32, ub>\n\n    %false_m0 = arith.constant false\n    pto.mte_gm_ub %arg0, %ub_lhs_m0, '
    '%c0_i64_m0, %c128_i64_m0\n      nburst(%c32_i64_m0, %c128_i64_m0, %c128_i64_m0)\n      : !pto.ptr<f32, gm>, '
    '!pto.ptr<f32, ub>, i64, i64, i64, i64, i64\n    pto.mte_gm_ub %arg1, %ub_rhs_m0, %c0_i64_m0, %c128_i64_m0\n '
    '     nburst(%c32_i64_m0, %c128_i64_m0, %c128_i64_m0)\n      : !pto.ptr<f32, gm>, !pto.ptr<f32, ub>, i64, i64'
    ', i64, i64, i64\n\n    pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n    pto.wait_flag["PIPE_MTE2", "PIP'
    'E_V", "EVENT_ID0"]\n\n    pto.vecscope {\n      %__m0:1 = scf.for %offset_m0 = %c0_m0 to %c1024_m0 step %c64'
    '_m0 iter_args(%remaining_m0 = %c1024_i32_m0) -> (i32) {\n        %mask_m0, %next_remaining_m0 = pto.plt_b32 '
    '%remaining_m0 : i32 -> !pto.mask<b32>, i32\n        %lhs_m0 = pto.vlds %ub_lhs_m0[%offset_m0] : !pto.ptr<f32'
    ', ub>, index -> !pto.vreg<64xf32>\n        %rhs_m0 = pto.vlds %ub_rhs_m0[%offset_m0] : !pto.ptr<f32, ub>, index -> !pto.vr'
    'eg<64xf32>\n        %quot_m0 = pto.vdiv %lhs_m0, %rhs_m0, %mask_m0 : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !'
    'pto.mask<b32> -> !pto.vreg<64xf32>\n        pto.vsts %quot_m0, %ub_out_m0[%offset_m0], %mask_m0 : !pto.vreg<'
    '64xf32>, !pto.ptr<f32, ub>, index, !pto.mask<b32>\n        scf.yield %next_remaining_m0 : i32\n      }\n    }\n\n  '
    '  pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n    pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n '
    '   pto.mte_ub_gm %ub_out_m0, %arg2, %c128_i64_m0\n      nburst(%c32_i64_m0, %c128_i64_m0, %c128_i64_m0)\n   '
    '   : !pto.ptr<f32, ub>, !pto.ptr<f32, gm>, i64, i64, i64, i64\n    pto.barrier #pto.pipe<PIPE_ALL>\n\n    //'
    ' merged from vdiv_f16_kernel via LaunchVdiv_f16_kernel\n\n    %c0_m1 = arith.constant 0 : index\n    %c128_m'
    '1 = arith.constant 128 : index\n    %c1024_m1 = arith.constant 1024 : index\n    %c0_i64_m1 = arith.constant'
    ' 0 : i64\n    %c1_i64_m1 = arith.constant 1 : i64\n    %c32_i64_m1 = arith.constant 32 : i64\n    %c64_i64_m'
    '1 = arith.constant 64 : i64\n    %c2048_i64_m1 = arith.constant 2048 : i64\n    %c4096_i64_m1 = arith.consta'
    'nt 4096 : i64\n    %c1024_i32_m1 = arith.constant 1024 : i32\n\n    %ub_lhs_m1 = pto.castptr %c0_i64_m1 : i6'
    '4 -> !pto.ptr<f16, ub>\n    %ub_rhs_m1 = pto.castptr %c2048_i64_m1 : i64 -> !pto.ptr<f16, ub>\n    %ub_out_m'
    '1 = pto.castptr %c4096_i64_m1 : i64 -> !pto.ptr<f16, ub>\n\n    %false_m1 = arith.constant false\n    pto.mt'
    'e_gm_ub %arg3, %ub_lhs_m1, %c0_i64_m1, %c64_i64_m1\n      nburst(%c32_i64_m1, %c64_i64_m1, %c64_i64_m1)\n   '
    '   : !pto.ptr<f16, gm>, !pto.ptr<f16, ub>, i64, i64, i64, i64, i64\n    pto.mte_gm_ub %arg4, %ub_rhs_m1, %c0'
    '_i64_m1, %c64_i64_m1\n      nburst(%c32_i64_m1, %c64_i64_m1, %c64_i64_m1)\n      : !pto.ptr<f16, gm>, !pto.p'
    'tr<f16, ub>, i64, i64, i64, i64, i64\n\n    pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n    pto.wait_f'
    'lag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n\n    pto.vecscope {\n      %__m1:1 = scf.for %offset_m1 = %c0_m1 t'
    'o %c1024_m1 step %c128_m1 iter_args(%remaining_m1 = %c1024_i32_m1) -> (i32) {\n        %mask_m1, %next_remai'
    'ning_m1 = pto.plt_b16 %remaining_m1 : i32 -> !pto.mask<b16>, i32\n        %lhs_m1 = pto.vlds %ub_lhs_m1[%off'
    'set_m1] : !pto.ptr<f16, ub>, index -> !pto.vreg<128xf16>\n        %rhs_m1 = pto.vlds %ub_rhs_m1[%offset_m1] : !pto.'
    'ptr<f16, ub>, index -> !pto.vreg<128xf16>\n        %quot_m1 = pto.vdiv %lhs_m1, %rhs_m1, %mask_m1 : !pto.vreg<128xf'
    '16>, !pto.vreg<128xf16>, !pto.mask<b16> -> !pto.vreg<128xf16>\n        pto.vsts %quot_m1, %ub_out_m1[%offset'
    '_m1], %mask_m1 : !pto.vreg<128xf16>, !pto.ptr<f16, ub>, index, !pto.mask<b16>\n        scf.yield %next_remaining_m1'
    ' : i32\n      }\n    }\n\n    pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n    pto.wait_flag["PIPE_V", '
    '"PIPE_MTE3", "EVENT_ID0"]\n    pto.mte_ub_gm %ub_out_m1, %arg5, %c64_i64_m1\n      nburst(%c32_i64_m1, %c64_'
    'i64_m1, %c64_i64_m1)\n      : !pto.ptr<f16, ub>, !pto.ptr<f16, gm>, i64, i64, i64, i64\n    pto.barrier #pto'
    '.pipe<PIPE_ALL>\n\n    // merged from vdiv_f32_exceptional_kernel_2d via LaunchVdiv_f32_exceptional_kernel_2'
    'd\n\n    %c0_m2 = arith.constant 0 : index\n    %c64_m2 = arith.constant 64 : index\n    %c1024_m2 = arith.c'
    'onstant 1024 : index\n    %c0_i64_m2 = arith.constant 0 : i64\n    %c1_i64_m2 = arith.constant 1 : i64\n    '
    '%c32_i64_m2 = arith.constant 32 : i64\n    %c128_i64_m2 = arith.constant 128 : i64\n    %c4096_i64_m2 = arit'
    'h.constant 4096 : i64\n    %c8192_i64_m2 = arith.constant 8192 : i64\n    %c1024_i32_m2 = arith.constant 102'
    '4 : i32\n\n    %ub_lhs_m2 = pto.castptr %c0_i64_m2 : i64 -> !pto.ptr<f32, ub>\n    %ub_rhs_m2 = pto.castptr '
    '%c4096_i64_m2 : i64 -> !pto.ptr<f32, ub>\n    %ub_out_m2 = pto.castptr %c8192_i64_m2 : i64 -> !pto.ptr<f32, '
    'ub>\n\n    %false_m2 = arith.constant false\n    pto.mte_gm_ub %arg6, %ub_lhs_m2, %c0_i64_m2, %c128_i64_m2\n'
    '      nburst(%c32_i64_m2, %c128_i64_m2, %c128_i64_m2)\n      : !pto.ptr<f32, gm>, !pto.ptr<f32, ub>, i64, i6'
    '4, i64, i64, i64\n    pto.mte_gm_ub %arg7, %ub_rhs_m2, %c0_i64_m2, %c128_i64_m2\n      nburst(%c32_i64_m2, %'
    'c128_i64_m2, %c128_i64_m2)\n      : !pto.ptr<f32, gm>, !pto.ptr<f32, ub>, i64, i64, i64, i64, i64\n\n    pto'
    '.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n    pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n\n    '
    'pto.vecscope {\n      %__m2:1 = scf.for %offset_m2 = %c0_m2 to %c1024_m2 step %c64_m2 iter_args(%remaining_m'
    '2 = %c1024_i32_m2) -> (i32) {\n        %mask_m2, %next_remaining_m2 = pto.plt_b32 %remaining_m2 : i32 -> !pt'
    'o.mask<b32>, i32\n        %lhs_m2 = pto.vlds %ub_lhs_m2[%offset_m2] : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>'
    '\n        %rhs_m2 = pto.vlds %ub_rhs_m2[%offset_m2] : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>\n        %quot_'
    'm2 = pto.vdiv %lhs_m2, %rhs_m2, %mask_m2 : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg'
    '<64xf32>\n        pto.vsts %quot_m2, %ub_out_m2[%offset_m2], %mask_m2 : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, index'
    ', !pto.mask<b32>\n        scf.yield %next_remaining_m2 : i32\n      }\n    }\n\n    pto.set_flag["PIPE_V", "'
    'PIPE_MTE3", "EVENT_ID0"]\n    pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n    pto.mte_ub_gm %ub_out_m'
    '2, %arg8, %c128_i64_m2\n      nburst(%c32_i64_m2, %c128_i64_m2, %c128_i64_m2)\n      : !pto.ptr<f32, ub>, !p'
    'to.ptr<f32, gm>, i64, i64, i64, i64\n    pto.barrier #pto.pipe<PIPE_ALL>\n\n    // merged from vdiv_tail_ker'
    'nel_2d via LaunchVdiv_tail_kernel_2d\n\n    %c0_m3 = arith.constant 0 : index\n    %c64_m3 = arith.constant '
    '64 : index\n    %c1024_m3 = arith.constant 1024 : index\n    %c0_i64_m3 = arith.constant 0 : i64\n    %c1_i6'
    '4_m3 = arith.constant 1 : i64\n    %c32_i64_m3 = arith.constant 32 : i64\n    %c128_i64_m3 = arith.constant '
    '128 : i64\n    %c4096_i64_m3 = arith.constant 4096 : i64\n    %c8192_i64_m3 = arith.constant 8192 : i64\n   '
    ' %c1000_i32_m3 = arith.constant 1000 : i32\n\n    %ub_lhs_m3 = pto.castptr %c0_i64_m3 : i64 -> !pto.ptr<f32,'
    ' ub>\n    %ub_rhs_m3 = pto.castptr %c4096_i64_m3 : i64 -> !pto.ptr<f32, ub>\n    %ub_out_m3 = pto.castptr %c'
    '8192_i64_m3 : i64 -> !pto.ptr<f32, ub>\n\n    %false_m3 = arith.constant false\n    pto.mte_gm_ub %arg9, %ub'
    '_lhs_m3, %c0_i64_m3, %c128_i64_m3\n      nburst(%c32_i64_m3, %c128_i64_m3, %c128_i64_m3)\n      : !pto.ptr<f'
    '32, gm>, !pto.ptr<f32, ub>, i64, i64, i64, i64, i64\n    pto.mte_gm_ub %arg10, %ub_rhs_m3, %c0_i64_m3, %c128'
    '_i64_m3\n      nburst(%c32_i64_m3, %c128_i64_m3, %c128_i64_m3)\n      : !pto.ptr<f32, gm>, !pto.ptr<f32, ub>'
    ', i64, i64, i64, i64, i64\n\n    pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n    pto.wait_flag["PIPE_M'
    'TE2", "PIPE_V", "EVENT_ID0"]\n\n    pto.vecscope {\n      %__m3:1 = scf.for %offset_m3 = %c0_m3 to %c1024_m3'
    ' step %c64_m3 iter_args(%remaining_m3 = %c1000_i32_m3) -> (i32) {\n        %mask_m3, %next_remaining_m3 = pt'
    'o.plt_b32 %remaining_m3 : i32 -> !pto.mask<b32>, i32\n        %lhs_m3 = pto.vlds %ub_lhs_m3[%offset_m3] : !p'
    'to.ptr<f32, ub>, index -> !pto.vreg<64xf32>\n        %rhs_m3 = pto.vlds %ub_rhs_m3[%offset_m3] : !pto.ptr<f32, ub>, index '
    '-> !pto.vreg<64xf32>\n        %quot_m3 = pto.vdiv %lhs_m3, %rhs_m3, %mask_m3 : !pto.vreg<64xf32>, !pto.vreg<'
    '64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>\n        pto.vsts %quot_m3, %ub_out_m3[%offset_m3], %mask_m3 : '
    '!pto.vreg<64xf32>, !pto.ptr<f32, ub>, index, !pto.mask<b32>\n        scf.yield %next_remaining_m3 : i32\n      }\n '
    '   }\n\n    pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n    pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVEN'
    'T_ID0"]\n    pto.mte_ub_gm %ub_out_m3, %arg11, %c128_i64_m3\n      nburst(%c32_i64_m3, %c128_i64_m3, %c128_i'
    '64_m3)\n      : !pto.ptr<f32, ub>, !pto.ptr<f32, gm>, i64, i64, i64, i64\n    pto.barrier #pto.pipe<PIPE_ALL'
    '>\n\n    return\n  }\n}\n'
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

    # Merged vdiv golden data generator.
    import argparse
    from pathlib import Path
    import numpy as np

    ROWS = 32
    COLS = 32
    SEED = 19
    LOGICAL_ELEMS = 1000
    OUT_SENTINEL = np.float32(-123.25)

    def f32_to_bf16_bits(values):
        wide = values.astype(np.float32, copy=False).view(np.uint32)
        rounding = np.uint32(0x7FFF) + ((wide >> 16) & np.uint32(1))
        return ((wide + rounding) >> 16).astype(np.uint16)

    def bf16_bits_to_f32(bits):
        return (bits.astype(np.uint32) << 16).view(np.float32)

    def gen_f32(out, rng):
        v1 = rng.uniform(-8.0, 8.0, size=(ROWS, COLS)).astype(np.float32)
        v2 = rng.uniform(-8.0, 8.0, size=(ROWS, COLS)).astype(np.float32)
        g = (np.divide(v1, v2, where=v2!=0, out=np.full_like(v1, np.nan))).astype(np.float32, copy=False)
        v3 = np.zeros((ROWS, COLS), dtype=np.float32)
        _files["v1.bin"] = v1.reshape(-1)
        _files["v2.bin"] = v2.reshape(-1)
        _files["v3.bin"] = v3.reshape(-1)
        _files["golden_v3.bin"] = g.reshape(-1)

    def gen_f16(out, rng):
        v1 = rng.uniform(-8.0, 8.0, size=(ROWS, COLS)).astype(np.float16)
        v2 = rng.uniform(-8.0, 8.0, size=(ROWS, COLS)).astype(np.float16)
        g = ((v1.astype(np.float32) / np.maximum(np.abs(v2.astype(np.float32)), 1e-8) * np.sign(v2.astype(
            np.float32))).astype(np.float16)).astype(np.float16)
        v3 = np.zeros((ROWS, COLS), dtype=np.float16)
        _files["v1_f16.bin"] = v1.reshape(-1)
        _files["v2_f16.bin"] = v2.reshape(-1)
        _files["v3_f16.bin"] = v3.reshape(-1)
        _files["golden_v3_f16.bin"] = g.reshape(-1)

    def gen_f32_exceptional(out, rng):
        specials_a = np.array([-np.inf, -7.5, -0.0, 0.0, 1.0, np.inf, np.nan, 3.5], dtype=np.float32)
        specials_b = np.array([np.inf, 2.5, 0.0, -0.0, -1.0, -np.inf, 1.0, np.nan], dtype=np.float32)
        v1 = np.resize(specials_a, ROWS * COLS).reshape(ROWS, COLS).astype(np.float32)
        v2 = np.resize(specials_b, ROWS * COLS).reshape(ROWS, COLS).astype(np.float32)
        g = (np.divide(v1, v2, where=v2!=0, out=np.full_like(v1, np.nan))).astype(np.float32, copy=False)
        v3 = np.zeros((ROWS, COLS), dtype=np.float32)
        _files["v1_f32_exceptional.bin"] = v1.reshape(-1)
        _files["v2_f32_exceptional.bin"] = v2.reshape(-1)
        _files["v3_f32_exceptional.bin"] = v3.reshape(-1)
        _files["golden_v3_f32_exceptional.bin"] = g.reshape(-1)

    def gen_tail(out, rng):
        v1 = rng.random((ROWS, COLS), dtype=np.float32)
        v2 = rng.random((ROWS, COLS), dtype=np.float32)
        v3 = np.full((ROWS, COLS), OUT_SENTINEL, dtype=np.float32)
        g = np.full((ROWS, COLS), OUT_SENTINEL, dtype=np.float32)
        g.reshape(-1)[:LOGICAL_ELEMS] = (np.divide(v1.reshape(-1)[:LOGICAL_ELEMS], v2.reshape(-1)[:LOGICAL_ELEMS],
            where=v2.reshape(-1)[:LOGICAL_ELEMS]!=0, out=np.full(LOGICAL_ELEMS, np.nan))).astype(np.float32, copy=False)
        _files["v1_tail.bin"] = v1.reshape(-1)
        _files["v2_tail.bin"] = v2.reshape(-1)
        _files["v3_tail.bin"] = v3.reshape(-1)
        _files["golden_v3_tail.bin"] = g.reshape(-1)

    GENERATORS = [
        gen_f32,
        gen_f16,
        gen_f32_exceptional,
        gen_tail,
    ]

    def main():
        p = argparse.ArgumentParser()
        p.add_argument("--output-dir", type=Path, default=Path("."))
        p.add_argument("--seed", type=int, default=SEED)
        a = p.parse_args()
        rng = np.random.default_rng(a.seed)
        out = a.output_dir
        out.mkdir(parents=True, exist_ok=True)
        for gen in GENERATORS:
            gen(out, rng)

    return locals()

def _compare_scope():
    # Copyright (c) 2026 Huawei Technologies Co., Ltd.
    # This program is free software, you can redistribute it and/or modify it under the terms and conditions of
    # CANN Open Software License Agreement Version 2.0 (the "License").
    # Please refer to the License for details. You may not use this file except in compliance with the License.
    # THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
    # INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
    # See LICENSE in the root of the software repository for the full text of the License.

    # Merged vdiv compare.
    import os, sys
    import numpy as np

    def _cmp(golden, output, dtype, eps, count=-1):
        if not os.path.exists(golden) or not os.path.exists(output):
            return False
        kw = {} if count < 0 else {"count": count}
        g = np.fromfile(golden, dtype=dtype, **kw)
        o = np.fromfile(output, dtype=dtype, **kw)
        return g.shape == o.shape and np.allclose(g, o, atol=eps, rtol=eps, equal_nan=True)

    def _cmpeq(golden, output, dtype):
        if not os.path.exists(golden) or not os.path.exists(output):
            return False
        g = np.fromfile(golden, dtype=dtype)
        o = np.fromfile(output, dtype=dtype)
        return g.shape == o.shape and np.array_equal(g, o)

    def main():
        strict = os.getenv("COMPARE_STRICT", "1") != "0"
        failed = []
        if not (_cmp("golden_v3.bin","v3.bin",np.float32,1e-4,-1)):
            failed.append('f32')
            print('[ERROR] compare failed: f32')
        else:
            print('[INFO] f32: passed')
        if not (_cmp("golden_v3_f16.bin","v3_f16.bin",np.float16,5e-3,1024)):
            failed.append('f16')
            print('[ERROR] compare failed: f16')
        else:
            print('[INFO] f16: passed')
        if not (_cmp("golden_v3_f32_exceptional.bin","v3_f32_exceptional.bin",np.float32,1e-4,-1)):
            failed.append('f32_exceptional')
            print('[ERROR] compare failed: f32_exceptional')
        else:
            print('[INFO] f32_exceptional: passed')
        if not (_cmp("golden_v3_tail.bin","v3_tail.bin",np.float32,1e-4,1000)):
            failed.append('tail')
            print('[ERROR] compare failed: tail')
        else:
            print('[INFO] tail: passed')
        if failed:
            if strict:
                print(f"[ERROR] {len(failed)} variant(s) failed: {','.join(failed)}")
                sys.exit(2)
            print(f"[WARN] {len(failed)} variant(s) failed (non-gating): {','.join(failed)}")
            return
        print("[INFO] compare passed (all 4 variants)")

    return locals()

HELPERS = {}
SEED_EXPR = 'SEED'
COMPARE_PAIRS = [
    (
        'golden_v3.bin',
        'v3.bin'
    ),
    (
        'golden_v3_f16.bin',
        'v3_f16.bin'
    ),
    (
        'golden_v3_f32_exceptional.bin',
        'v3_f32_exceptional.bin'
    ),
    (
        'golden_v3_tail.bin',
        'v3_tail.bin'
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

BUFFERS = [
    (
        'v1',
        'f32',
        'arg0'
    ),
    (
        'v2',
        'f32',
        'arg1'
    ),
    (
        'v3',
        'f32',
        'arg2'
    ),
    (
        'v1_f16',
        'f16',
        'arg3'
    ),
    (
        'v2_f16',
        'f16',
        'arg4'
    ),
    (
        'v3_f16',
        'f16',
        'arg5'
    ),
    (
        'v1_f32_exceptional',
        'f32',
        'arg6'
    ),
    (
        'v2_f32_exceptional',
        'f32',
        'arg7'
    ),
    (
        'v3_f32_exceptional',
        'f32',
        'arg8'
    ),
    (
        'v1_tail',
        'f32',
        'arg9'
    ),
    (
        'v2_tail',
        'f32',
        'arg10'
    ),
    (
        'v3_tail',
        'f32',
        'arg11'
    )
]
SCALARS = []
SCALAR_SOURCES = []
SCALAR_DTYPES = []


@pto.jit(
    name='vdiv_deep_merged_kernel',
    target="a5",
    backend="vpto",
    mode="explicit",
    source=KERNEL_SOURCE,
)
def kernel(
    arg0: pto.ptr(pto.f32, "gm"),
    arg1: pto.ptr(pto.f32, "gm"),
    arg2: pto.ptr(pto.f32, "gm"),
    arg3: pto.ptr(pto.f16, "gm"),
    arg4: pto.ptr(pto.f16, "gm"),
    arg5: pto.ptr(pto.f16, "gm"),
    arg6: pto.ptr(pto.f32, "gm"),
    arg7: pto.ptr(pto.f32, "gm"),
    arg8: pto.ptr(pto.f32, "gm"),
    arg9: pto.ptr(pto.f32, "gm"),
    arg10: pto.ptr(pto.f32, "gm"),
    arg11: pto.ptr(pto.f32, "gm"),
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
    name='legacy_micro_op_binary_vector_vdiv',
    grid=1,
    kernel=kernel,
    make_case=make_case,
    check=check_case,
)]

auto_main(globals())
