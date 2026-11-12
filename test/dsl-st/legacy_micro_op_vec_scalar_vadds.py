#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Migrated from test/vpto/cases/micro-op/vec-scalar/vadds."""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

from common import auto_main
from ptodsl import pto

KERNEL_SOURCE = (
    'module attributes {pto.target_arch = "a5", pto.kernel_kind = #pto.kernel_kind<vector>} {\n    func.func @vad'
    'ds_deep_merged_kernel(%arg0: !pto.ptr<bf16, gm>, %arg1: !pto.ptr<bf16, gm>, %arg2: !pto.ptr<f16, gm>, %arg3:'
    ' !pto.ptr<f16, gm>, %arg4: !pto.ptr<f32, gm>, %arg5: !pto.ptr<f32, gm>, %arg6: !pto.ptr<i16, gm>, %arg7: !pt'
    'o.ptr<i16, gm>, %arg8: !pto.ptr<i16, gm>, %arg9: !pto.ptr<i16, gm>, %arg10: !pto.ptr<ui16, gm>, %arg11: !pto'
    '.ptr<ui16, gm>, %arg12: !pto.ptr<ui16, gm>, %arg13: !pto.ptr<ui16, gm>, %arg14: !pto.ptr<f32, gm>, %arg15: !'
    'pto.ptr<f32, gm>, %arg16: !pto.ptr<f32, gm>, %arg17: !pto.ptr<f32, gm>) attributes {pto.kernel} {\n    %__de'
    'ep_merge_guard = arith.constant false\n    // inactive merged from vadds_bf16_kernel\n    scf.if %__deep_mer'
    'ge_guard {\n\n    %c0_m0 = arith.constant 0 : index\n    %c128_m0 = arith.constant 128 : index\n    %c1024_m'
    '0 = arith.constant 1024 : index\n    %c0_i64_m0 = arith.constant 0 : i64\n    %c1_i64_m0 = arith.constant 1 '
    ': i64\n    %c32_i64_m0 = arith.constant 32 : i64\n    %c64_i64_m0 = arith.constant 64 : i64\n    %c2048_i64_'
    'm0 = arith.constant 2048 : i64\n    %cst_m0 = arith.constant 1.500000e+00 : bf16\n\n    %ub_in_m0 = pto.cast'
    'ptr %c0_i64_m0 : i64 -> !pto.ptr<bf16, ub>\n    %ub_out_m0 = pto.castptr %c2048_i64_m0 : i64 -> !pto.ptr<bf1'
    '6, ub>\n\n    %false_m0 = arith.constant false\n    pto.mte_gm_ub %arg0, %ub_in_m0, %c0_i64_m0, %c64_i64_m0'
    '\n      nburst(%c32_i64_m0, %c64_i64_m0, %c64_i64_m0)\n      : !pto.ptr<bf16, gm>, !pto.ptr<bf16, ub>, i64, '
    'i64, i64, i64, i64\n\n    pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n    pto.wait_flag["PIPE_MTE2", "'
    'PIPE_V", "EVENT_ID0"]\n\n    pto.vecscope {\n      %mask_m0 = pto.pset_b16 "PAT_ALL" : !pto.mask<b16>\n     '
    ' scf.for %offset_m0 = %c0_m0 to %c1024_m0 step %c128_m0 {\n        %vec_m0 = pto.vlds %ub_in_m0[%offset_m0] '
    ': !pto.ptr<bf16, ub>, index -> !pto.vreg<128xbf16>\n        %sum_m0 = pto.vadds %vec_m0, %cst_m0, %mask_m0 : !pto.v'
    'reg<128xbf16>, bf16, !pto.mask<b16> -> !pto.vreg<128xbf16>\n        pto.vsts %sum_m0, %ub_out_m0[%offset_m0]'
    ', %mask_m0 : !pto.vreg<128xbf16>, !pto.ptr<bf16, ub>, index, !pto.mask<b16>\n      }\n    }\n\n    pto.set_flag["PI'
    'PE_V", "PIPE_MTE3", "EVENT_ID0"]\n    pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n    pto.mte_ub_gm %'
    'ub_out_m0, %arg1, %c64_i64_m0\n      nburst(%c32_i64_m0, %c64_i64_m0, %c64_i64_m0)\n      : !pto.ptr<bf16, u'
    'b>, !pto.ptr<bf16, gm>, i64, i64, i64, i64\n    pto.barrier #pto.pipe<PIPE_ALL>\n\n    }\n    // inactive me'
    'rged from vadds_f16_kernel\n    scf.if %__deep_merge_guard {\n\n    %c0_m1 = arith.constant 0 : index\n    %'
    'c128_m1 = arith.constant 128 : index\n    %c1024_m1 = arith.constant 1024 : index\n    %c0_i64_m1 = arith.co'
    'nstant 0 : i64\n    %c1_i64_m1 = arith.constant 1 : i64\n    %c32_i64_m1 = arith.constant 32 : i64\n    %c64'
    '_i64_m1 = arith.constant 64 : i64\n    %c2048_i64_m1 = arith.constant 2048 : i64\n    %cst_m1 = arith.consta'
    'nt 1.500000e+00 : f16\n\n    %ub_in_m1 = pto.castptr %c0_i64_m1 : i64 -> !pto.ptr<f16, ub>\n    %ub_out_m1 ='
    ' pto.castptr %c2048_i64_m1 : i64 -> !pto.ptr<f16, ub>\n\n    %false_m1 = arith.constant false\n    pto.mte_g'
    'm_ub %arg2, %ub_in_m1, %c0_i64_m1, %c64_i64_m1\n      nburst(%c32_i64_m1, %c64_i64_m1, %c64_i64_m1)\n      :'
    ' !pto.ptr<f16, gm>, !pto.ptr<f16, ub>, i64, i64, i64, i64, i64\n\n    pto.set_flag["PIPE_MTE2", "PIPE_V", "E'
    'VENT_ID0"]\n    pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n\n    pto.vecscope {\n      %mask_m1 = pt'
    'o.pset_b16 "PAT_ALL" : !pto.mask<b16>\n      scf.for %offset_m1 = %c0_m1 to %c1024_m1 step %c128_m1 {\n     '
    '   %vec_m1 = pto.vlds %ub_in_m1[%offset_m1] : !pto.ptr<f16, ub>, index -> !pto.vreg<128xf16>\n        %sum_m1 = pto'
    '.vadds %vec_m1, %cst_m1, %mask_m1 : !pto.vreg<128xf16>, f16, !pto.mask<b16> -> !pto.vreg<128xf16>\n        p'
    'to.vsts %sum_m1, %ub_out_m1[%offset_m1], %mask_m1 : !pto.vreg<128xf16>, !pto.ptr<f16, ub>, index, !pto.mask<b16>\n '
    '     }\n    }\n\n    pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n    pto.wait_flag["PIPE_V", "PIPE_MTE'
    '3", "EVENT_ID0"]\n    pto.mte_ub_gm %ub_out_m1, %arg3, %c64_i64_m1\n      nburst(%c32_i64_m1, %c64_i64_m1, %'
    'c64_i64_m1)\n      : !pto.ptr<f16, ub>, !pto.ptr<f16, gm>, i64, i64, i64, i64\n    pto.barrier #pto.pipe<PIP'
    'E_ALL>\n\n    }\n    // inactive merged from vadds_f32_exceptional_kernel_2d\n    scf.if %__deep_merge_guard'
    ' {\n\n    %c0_m2 = arith.constant 0 : index\n    %c64_m2 = arith.constant 64 : index\n    %c1024_m2 = arith.'
    'constant 1024 : index\n    %c0_i64_m2 = arith.constant 0 : i64\n    %c1_i64_m2 = arith.constant 1 : i64\n   '
    ' %c32_i64_m2 = arith.constant 32 : i64\n    %c128_i64_m2 = arith.constant 128 : i64\n    %c4096_i64_m2 = ari'
    'th.constant 4096 : i64\n    %c1024_i32_m2 = arith.constant 1024 : i32\n    %cst_m2 = arith.constant 5.000000'
    'e-01 : f32\n\n    %ub_in_m2 = pto.castptr %c0_i64_m2 : i64 -> !pto.ptr<f32, ub>\n    %ub_out_m2 = pto.castpt'
    'r %c4096_i64_m2 : i64 -> !pto.ptr<f32, ub>\n\n    %false_m2 = arith.constant false\n    pto.mte_gm_ub %arg4,'
    ' %ub_in_m2, %c0_i64_m2, %c128_i64_m2\n      nburst(%c32_i64_m2, %c128_i64_m2, %c128_i64_m2)\n      : !pto.pt'
    'r<f32, gm>, !pto.ptr<f32, ub>, i64, i64, i64, i64, i64\n\n    pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0'
    '"]\n    pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n\n    pto.vecscope {\n      %__m2:1 = scf.for %of'
    'fset_m2 = %c0_m2 to %c1024_m2 step %c64_m2 iter_args(%remaining_m2 = %c1024_i32_m2) -> (i32) {\n        %mas'
    'k_m2, %next_remaining_m2 = pto.plt_b32 %remaining_m2 : i32 -> !pto.mask<b32>, i32\n        %vec_m2 = pto.vld'
    's %ub_in_m2[%offset_m2] : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>\n        %sum_m2 = pto.vadds %vec_m2, %cst_'
    'm2, %mask_m2 : !pto.vreg<64xf32>, f32, !pto.mask<b32> -> !pto.vreg<64xf32>\n        pto.vsts %sum_m2, %ub_ou'
    't_m2[%offset_m2], %mask_m2 : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, index, !pto.mask<b32>\n        scf.yield %next_r'
    'emaining_m2 : i32\n      }\n    }\n\n    pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n    pto.wait_flag'
    '["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n    pto.mte_ub_gm %ub_out_m2, %arg5, %c128_i64_m2\n      nburst(%c32_i'
    '64_m2, %c128_i64_m2, %c128_i64_m2)\n      : !pto.ptr<f32, ub>, !pto.ptr<f32, gm>, i64, i64, i64, i64\n    pt'
    'o.barrier #pto.pipe<PIPE_ALL>\n\n    }\n    // inactive merged from vadds_i16_signed_kernel\n    scf.if %__d'
    'eep_merge_guard {\n\n    %c0_m3 = arith.constant 0 : index\n    %c128_m3 = arith.constant 128 : index\n    %'
    'c1024_m3 = arith.constant 1024 : index\n    %c0_i64_m3 = arith.constant 0 : i64\n    %c1_i64_m3 = arith.cons'
    'tant 1 : i64\n    %c32_i64_m3 = arith.constant 32 : i64\n    %c64_i64_m3 = arith.constant 64 : i64\n    %c20'
    '48_i64_m3 = arith.constant 2048 : i64\n    %c1024_i32_m3 = arith.constant 1024 : i32\n    %scalar_m3 = arith'
    '.constant 37 : i16\n\n    %ub_in_m3 = pto.castptr %c0_i64_m3 : i64 -> !pto.ptr<i16, ub>\n    %ub_out_m3 = pt'
    'o.castptr %c2048_i64_m3 : i64 -> !pto.ptr<i16, ub>\n\n    %false_m3 = arith.constant false\n    pto.mte_gm_u'
    'b %arg6, %ub_in_m3, %c0_i64_m3, %c64_i64_m3\n      nburst(%c32_i64_m3, %c64_i64_m3, %c64_i64_m3)\n      : !p'
    'to.ptr<i16, gm>, !pto.ptr<i16, ub>, i64, i64, i64, i64, i64\n\n    pto.set_flag["PIPE_MTE2", "PIPE_V", "EVEN'
    'T_ID0"]\n    pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n\n    pto.vecscope {\n      %mask_m3 = pto.p'
    'set_b16 "PAT_ALL" : !pto.mask<b16>\n      scf.for %offset_m3 = %c0_m3 to %c1024_m3 step %c128_m3 {\n        '
    '%vec_m3 = pto.vlds %ub_in_m3[%offset_m3] : !pto.ptr<i16, ub>, index -> !pto.vreg<128xi16>\n        %sum_m3 = pto.va'
    'dds %vec_m3, %scalar_m3, %mask_m3 : !pto.vreg<128xi16>, i16, !pto.mask<b16> -> !pto.vreg<128xi16>\n        p'
    'to.vsts %sum_m3, %ub_out_m3[%offset_m3], %mask_m3 : !pto.vreg<128xi16>, !pto.ptr<i16, ub>, index, !pto.mask<b16>\n '
    '     }\n    }\n\n    pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n    pto.wait_flag["PIPE_V", "PIPE_MTE'
    '3", "EVENT_ID0"]\n    pto.mte_ub_gm %ub_out_m3, %arg7, %c64_i64_m3\n      nburst(%c32_i64_m3, %c64_i64_m3, %'
    'c64_i64_m3)\n      : !pto.ptr<i16, ub>, !pto.ptr<i16, gm>, i64, i64, i64, i64\n    pto.barrier #pto.pipe<PIP'
    'E_ALL>\n\n    }\n    // inactive merged from vadds_i16_signed_overflow_kernel\n    scf.if %__deep_merge_guar'
    'd {\n\n    %c0_m4 = arith.constant 0 : index\n    %c128_m4 = arith.constant 128 : index\n    %c1024_m4 = ari'
    'th.constant 1024 : index\n    %c0_i64_m4 = arith.constant 0 : i64\n    %c1_i64_m4 = arith.constant 1 : i64\n'
    '    %c32_i64_m4 = arith.constant 32 : i64\n    %c64_i64_m4 = arith.constant 64 : i64\n    %c2048_i64_m4 = ar'
    'ith.constant 2048 : i64\n    %scalar_m4 = arith.constant 1024 : i16\n\n    %ub_in_m4 = pto.castptr %c0_i64_m'
    '4 : i64 -> !pto.ptr<i16, ub>\n    %ub_out_m4 = pto.castptr %c2048_i64_m4 : i64 -> !pto.ptr<i16, ub>\n\n    %'
    'false_m4 = arith.constant false\n    pto.mte_gm_ub %arg8, %ub_in_m4, %c0_i64_m4, %c64_i64_m4\n      nburst(%'
    'c32_i64_m4, %c64_i64_m4, %c64_i64_m4)\n      : !pto.ptr<i16, gm>, !pto.ptr<i16, ub>, i64, i64, i64, i64, i64'
    '\n\n    pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n    pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID'
    '0"]\n\n    pto.vecscope {\n      %mask_m4 = pto.pset_b16 "PAT_ALL" : !pto.mask<b16>\n      scf.for %offset_m'
    '4 = %c0_m4 to %c1024_m4 step %c128_m4 {\n        %vec_m4 = pto.vlds %ub_in_m4[%offset_m4] : !pto.ptr<i16, ub'
    '>, index -> !pto.vreg<128xi16>\n        %sum_m4 = pto.vadds %vec_m4, %scalar_m4, %mask_m4 : !pto.vreg<128xi16>, i16'
    ', !pto.mask<b16> -> !pto.vreg<128xi16>\n        pto.vsts %sum_m4, %ub_out_m4[%offset_m4], %mask_m4 : !pto.vr'
    'eg<128xi16>, !pto.ptr<i16, ub>, index, !pto.mask<b16>\n      }\n    }\n\n    pto.set_flag["PIPE_V", "PIPE_MTE3", "E'
    'VENT_ID0"]\n    pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n    pto.mte_ub_gm %ub_out_m4, %arg9, %c64'
    '_i64_m4\n      nburst(%c32_i64_m4, %c64_i64_m4, %c64_i64_m4)\n      : !pto.ptr<i16, ub>, !pto.ptr<i16, gm>, '
    'i64, i64, i64, i64\n    pto.barrier #pto.pipe<PIPE_ALL>\n\n    }\n    // inactive merged from vadds_i16_unsi'
    'gned_kernel\n    scf.if %__deep_merge_guard {\n\n    %c0_m5 = arith.constant 0 : index\n    %c128_m5 = arith'
    '.constant 128 : index\n    %c1024_m5 = arith.constant 1024 : index\n    %c0_i64_m5 = arith.constant 0 : i64'
    '\n    %c1_i64_m5 = arith.constant 1 : i64\n    %c32_i64_m5 = arith.constant 32 : i64\n    %c64_i64_m5 = arit'
    'h.constant 64 : i64\n    %c2048_i64_m5 = arith.constant 2048 : i64\n    %scalar_m5 = arith.constant 37 : i16'
    '\n    %ub_in_m5 = pto.castptr %c0_i64_m5 : i64 -> !pto.ptr<ui16, ub>\n    %ub_out_m5 = pto.castptr %c2048_i6'
    '4_m5 : i64 -> !pto.ptr<ui16, ub>\n\n    %false_m5 = arith.constant false\n    pto.mte_gm_ub %arg10, %ub_in_m'
    '5, %c0_i64_m5, %c64_i64_m5\n      nburst(%c32_i64_m5, %c64_i64_m5, %c64_i64_m5)\n      : !pto.ptr<ui16, gm>,'
    ' !pto.ptr<ui16, ub>, i64, i64, i64, i64, i64\n\n    pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n    pt'
    'o.wait_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n\n    pto.vecscope {\n      %mask_m5 = pto.pset_b16 "PAT_AL'
    'L" : !pto.mask<b16>\n      scf.for %offset_m5 = %c0_m5 to %c1024_m5 step %c128_m5 {\n        %vec_m5 = pto.v'
    'lds %ub_in_m5[%offset_m5] : !pto.ptr<ui16, ub>, index -> !pto.vreg<128xui16>\n        %sum_m5 = pto.vadds %vec_m5, '
    '%scalar_m5, %mask_m5 : !pto.vreg<128xui16>, i16, !pto.mask<b16> -> !pto.vreg<128xui16>\n        pto.vsts %su'
    'm_m5, %ub_out_m5[%offset_m5], %mask_m5 : !pto.vreg<128xui16>, !pto.ptr<ui16, ub>, index, !pto.mask<b16>\n      }\n '
    '   }\n\n    pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n    pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVEN'
    'T_ID0"]\n    pto.mte_ub_gm %ub_out_m5, %arg11, %c64_i64_m5\n      nburst(%c32_i64_m5, %c64_i64_m5, %c64_i64_'
    'm5)\n      : !pto.ptr<ui16, ub>, !pto.ptr<ui16, gm>, i64, i64, i64, i64\n    pto.barrier #pto.pipe<PIPE_ALL>'
    '\n\n    }\n    // inactive merged from vadds_i16_unsigned_overflow_kernel\n    scf.if %__deep_merge_guard {'
    '\n\n    %c0_m6 = arith.constant 0 : index\n    %c128_m6 = arith.constant 128 : index\n    %c1024_m6 = arith.'
    'constant 1024 : index\n    %c0_i64_m6 = arith.constant 0 : i64\n    %c1_i64_m6 = arith.constant 1 : i64\n   '
    ' %c32_i64_m6 = arith.constant 32 : i64\n    %c64_i64_m6 = arith.constant 64 : i64\n    %c2048_i64_m6 = arith'
    '.constant 2048 : i64\n    %scalar_m6 = arith.constant 4096 : i16\n    %ub_in_m6 = pto.castptr %c0_i64_m6 : i'
    '64 -> !pto.ptr<ui16, ub>\n    %ub_out_m6 = pto.castptr %c2048_i64_m6 : i64 -> !pto.ptr<ui16, ub>\n\n    %fal'
    'se_m6 = arith.constant false\n    pto.mte_gm_ub %arg12, %ub_in_m6, %c0_i64_m6, %c64_i64_m6\n      nburst(%c3'
    '2_i64_m6, %c64_i64_m6, %c64_i64_m6)\n      : !pto.ptr<ui16, gm>, !pto.ptr<ui16, ub>, i64, i64, i64, i64, i64'
    '\n\n    pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n    pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID'
    '0"]\n\n    pto.vecscope {\n      %mask_m6 = pto.pset_b16 "PAT_ALL" : !pto.mask<b16>\n      scf.for %offset_m'
    '6 = %c0_m6 to %c1024_m6 step %c128_m6 {\n        %vec_m6 = pto.vlds %ub_in_m6[%offset_m6] : !pto.ptr<ui16, u'
    'b>, index -> !pto.vreg<128xui16>\n        %sum_m6 = pto.vadds %vec_m6, %scalar_m6, %mask_m6 : !pto.vreg<128xui16>, '
    'i16, !pto.mask<b16> -> !pto.vreg<128xui16>\n        pto.vsts %sum_m6, %ub_out_m6[%offset_m6], %mask_m6 : !pt'
    'o.vreg<128xui16>, !pto.ptr<ui16, ub>, index, !pto.mask<b16>\n      }\n    }\n\n    pto.set_flag["PIPE_V", "PIPE_MTE'
    '3", "EVENT_ID0"]\n    pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n    pto.mte_ub_gm %ub_out_m6, %arg1'
    '3, %c64_i64_m6\n      nburst(%c32_i64_m6, %c64_i64_m6, %c64_i64_m6)\n      : !pto.ptr<ui16, ub>, !pto.ptr<ui'
    '16, gm>, i64, i64, i64, i64\n    pto.barrier #pto.pipe<PIPE_ALL>\n\n    }\n    // inactive merged from vadds'
    '_tail_kernel_2d\n    scf.if %__deep_merge_guard {\n\n    %c0_m7 = arith.constant 0 : index\n    %c64_m7 = ar'
    'ith.constant 64 : index\n    %c1024_m7 = arith.constant 1024 : index\n    %c0_i64_m7 = arith.constant 0 : i6'
    '4\n    %c1_i64_m7 = arith.constant 1 : i64\n    %c32_i64_m7 = arith.constant 32 : i64\n    %c128_i64_m7 = ar'
    'ith.constant 128 : i64\n    %c4096_i64_m7 = arith.constant 4096 : i64\n    %c1000_i32_m7 = arith.constant 10'
    '00 : i32\n    %cst_m7 = arith.constant 3.140000e+00 : f32\n\n    %ub_in_m7 = pto.castptr %c0_i64_m7 : i64 ->'
    ' !pto.ptr<f32, ub>\n    %ub_out_m7 = pto.castptr %c4096_i64_m7 : i64 -> !pto.ptr<f32, ub>\n\n    %false_m7 ='
    ' arith.constant false\n    pto.mte_gm_ub %arg14, %ub_in_m7, %c0_i64_m7, %c128_i64_m7\n      nburst(%c32_i64_'
    'm7, %c128_i64_m7, %c128_i64_m7)\n      : !pto.ptr<f32, gm>, !pto.ptr<f32, ub>, i64, i64, i64, i64, i64\n\n  '
    '  pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n    pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n'
    '\n    pto.vecscope {\n      %__m7:1 = scf.for %offset_m7 = %c0_m7 to %c1024_m7 step %c64_m7 iter_args(%remai'
    'ning_m7 = %c1000_i32_m7) -> (i32) {\n        %mask_m7, %next_remaining_m7 = pto.plt_b32 %remaining_m7 : i32 '
    '-> !pto.mask<b32>, i32\n        %vec_m7 = pto.vlds %ub_in_m7[%offset_m7] : !pto.ptr<f32, ub>, index -> !pto.vreg<64'
    'xf32>\n        %sum_m7 = pto.vadds %vec_m7, %cst_m7, %mask_m7 : !pto.vreg<64xf32>, f32, !pto.mask<b32> -> !p'
    'to.vreg<64xf32>\n        pto.vsts %sum_m7, %ub_out_m7[%offset_m7], %mask_m7 : !pto.vreg<64xf32>, !pto.ptr<f3'
    '2, ub>, index, !pto.mask<b32>\n        scf.yield %next_remaining_m7 : i32\n      }\n    }\n\n    pto.set_flag["PIPE'
    '_V", "PIPE_MTE3", "EVENT_ID0"]\n    pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n    pto.mte_ub_gm %ub'
    '_out_m7, %arg15, %c128_i64_m7\n      nburst(%c32_i64_m7, %c128_i64_m7, %c128_i64_m7)\n      : !pto.ptr<f32, '
    'ub>, !pto.ptr<f32, gm>, i64, i64, i64, i64\n    pto.barrier #pto.pipe<PIPE_ALL>\n\n    }\n    // active merg'
    'ed from vec_add_scalar_kernel_2d\n\n    %c0_m8 = arith.constant 0 : index\n    %c1_m8 = arith.constant 1 : i'
    'ndex\n    %c64_m8 = arith.constant 64 : index\n    %c1024_m8 = arith.constant 1024 : index\n    %c0_i64_m8 ='
    ' arith.constant 0 : i64\n    %c1_i64_m8 = arith.constant 1 : i64\n    %c32_i64_m8 = arith.constant 32 : i64'
    '\n    %c128_i64_m8 = arith.constant 128 : i64\n    %c4096_i64_m8 = arith.constant 4096 : i64\n    %c1024_i32'
    '_m8 = arith.constant 1024 : i32\n    %cst_m8 = arith.constant 3.140000e+00 : f32\n\n    %ub_in_m8 = pto.cast'
    'ptr %c0_i64_m8 : i64 -> !pto.ptr<f32, ub>\n    %ub_out_m8 = pto.castptr %c4096_i64_m8 : i64 -> !pto.ptr<f32,'
    ' ub>\n\n    %false_m8 = arith.constant false\n    pto.mte_gm_ub %arg16, %ub_in_m8, %c0_i64_m8, %c128_i64_m8'
    '\n      nburst(%c32_i64_m8, %c128_i64_m8, %c128_i64_m8)\n      : !pto.ptr<f32, gm>, !pto.ptr<f32, ub>, i64, '
    'i64, i64, i64, i64\n\n    pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n    pto.wait_flag["PIPE_MTE2", "'
    'PIPE_V", "EVENT_ID0"]\n\n    pto.vecscope {\n      %__m8:1 = scf.for %offset_m8 = %c0_m8 to %c1024_m8 step %'
    'c64_m8 iter_args(%remaining_m8 = %c1024_i32_m8) -> (i32) {\n        %mask_m8, %next_remaining_m8 = pto.plt_b'
    '32 %remaining_m8 : i32 -> !pto.mask<b32>, i32\n        %vec_m8 = pto.vlds %ub_in_m8[%offset_m8] : !pto.ptr<f'
    '32, ub>, index -> !pto.vreg<64xf32>\n        %sum_m8 = pto.vadds %vec_m8, %cst_m8, %mask_m8 : !pto.vreg<64xf32>, f3'
    '2, !pto.mask<b32> -> !pto.vreg<64xf32>\n        pto.vsts %sum_m8, %ub_out_m8[%offset_m8], %mask_m8 : !pto.vr'
    'eg<64xf32>, !pto.ptr<f32, ub>, index, !pto.mask<b32>\n        scf.yield %next_remaining_m8 : i32\n      }\n    }\n'
    '\n    pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n    pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"'
    ']\n    pto.mte_ub_gm %ub_out_m8, %arg17, %c128_i64_m8\n      nburst(%c32_i64_m8, %c128_i64_m8, %c128_i64_m8)'
    '\n      : !pto.ptr<f32, ub>, !pto.ptr<f32, gm>, i64, i64, i64, i64\n    pto.barrier #pto.pipe<PIPE_ALL>\n\n '
    '   return\n  }\n}\n'
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

    # coding=utf-8

    import argparse
    from pathlib import Path

    import numpy as np


    ROWS = 32
    COLS = 32
    SEED = 19
    SCALE = np.float32(3.14)


    def _generate(seed: int) -> dict:
        rng = np.random.default_rng(seed)
        v1 = rng.uniform(-8.0, 8.0, size=(ROWS, COLS)).astype(np.float32)
        v2 = np.zeros((ROWS, COLS), dtype=np.float32)
        golden_v2 = (v1 + SCALE).astype(np.float32, copy=False)

        _files["v1.bin"] = v1.reshape(-1)
        _files["v2.bin"] = v2.reshape(-1)
        _files["golden_v2.bin"] = golden_v2.reshape(-1)


    def main() -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--output-dir", type=Path, default=Path("."))
        parser.add_argument("--seed", type=int, default=SEED)
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


    import os
    import sys
    import numpy as np


    def compare_bin(golden_path, output_path, dtype, eps):
        if not os.path.exists(golden_path) or not os.path.exists(output_path):
            return False
        golden = np.fromfile(golden_path, dtype=dtype)
        output = np.fromfile(output_path, dtype=dtype)
        return golden.shape == output.shape and np.allclose(golden, output, atol=eps, rtol=eps, equal_nan=True)


    def main():
        strict = os.getenv("COMPARE_STRICT", "1") != "0"
        ok = compare_bin("golden_v2.bin", "v2.bin", np.float32, 1e-4)
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
        'golden_v2.bin',
        'v2.bin'
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
        'bf16'
    ),
    (
        'v2',
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
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    1
]
ARG_DTYPES = [
    'bf16',
    'bf16',
    'f16',
    'f16',
    'f32',
    'f32',
    'i16',
    'i16',
    'i16',
    'i16',
    'ui16',
    'ui16',
    'ui16',
    'ui16',
    'f32',
    'f32',
    'f32',
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
    name='vadds_deep_merged_kernel',
    target="a5",
    backend="vpto",
    mode="explicit",
    source=KERNEL_SOURCE,
)
def kernel(
    arg0: pto.ptr(pto.bf16, "gm"),
    arg1: pto.ptr(pto.bf16, "gm"),
    arg2: pto.ptr(pto.f16, "gm"),
    arg3: pto.ptr(pto.f16, "gm"),
    arg4: pto.ptr(pto.f32, "gm"),
    arg5: pto.ptr(pto.f32, "gm"),
    arg6: pto.ptr(pto.i16, "gm"),
    arg7: pto.ptr(pto.i16, "gm"),
    arg8: pto.ptr(pto.i16, "gm"),
    arg9: pto.ptr(pto.i16, "gm"),
    arg10: pto.ptr(pto.ui16, "gm"),
    arg11: pto.ptr(pto.ui16, "gm"),
    arg12: pto.ptr(pto.ui16, "gm"),
    arg13: pto.ptr(pto.ui16, "gm"),
    arg14: pto.ptr(pto.f32, "gm"),
    arg15: pto.ptr(pto.f32, "gm"),
    arg16: pto.ptr(pto.f32, "gm"),
    arg17: pto.ptr(pto.f32, "gm"),
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
    name='legacy_micro_op_vec_scalar_vadds',
    grid=1,
    kernel=kernel,
    make_case=make_case,
    check=check_case,
)]

auto_main(globals())
