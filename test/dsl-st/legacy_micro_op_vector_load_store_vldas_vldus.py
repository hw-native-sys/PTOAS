#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Migrated from test/vpto/cases/micro-op/vector-load-store/vldas-vldus."""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

from common import auto_main
from ptodsl import pto

KERNEL_SOURCE = (
    'module attributes {pto.target_arch = "a5", pto.kernel_kind = #pto.kernel_kind<vector>} {\n    func.func @vld'
    'as_vldus_deep_merged_kernel(%arg0: !pto.ptr<f32, gm>, %arg1: !pto.ptr<f32, gm>, %arg2: !pto.ptr<f32, gm>, %a'
    'rg3: !pto.ptr<f32, gm>) attributes {pto.kernel} {\n    %__deep_merge_guard = arith.constant false\n    // in'
    'active merged from vldas_vldus_state_chain_kernel_2d\n    scf.if %__deep_merge_guard {\n\n    %c0_m0 = arith'
    '.constant 0 : index\n    %c1_m0 = arith.constant 1 : index\n    %c64_m0 = arith.constant 64 : index\n    %c1'
    '024_m0 = arith.constant 1024 : index\n    %c0_i64_m0 = arith.constant 0 : i64\n    %c1_i64_m0 = arith.consta'
    'nt 1 : i64\n    %c32_i64_m0 = arith.constant 32 : i64\n    %c128_i64_m0 = arith.constant 128 : i64\n    %c40'
    '96_i64_m0 = arith.constant 4096 : i64\n    %c1024_i32_m0 = arith.constant 1024 : i32\n\n    %ub_in_m0 = pto.'
    'castptr %c0_i64_m0 : i64 -> !pto.ptr<f32, ub>\n    %ub_out_m0 = pto.castptr %c4096_i64_m0 : i64 -> !pto.ptr<'
    'f32, ub>\n\n    %false_m0 = arith.constant false\n    pto.mte_gm_ub %arg0, %ub_in_m0, %c0_i64_m0, %c128_i64_'
    'm0\n      nburst(%c32_i64_m0, %c128_i64_m0, %c128_i64_m0)\n      : !pto.ptr<f32, gm>, !pto.ptr<f32, ub>, i64'
    ', i64, i64, i64, i64\n\n    pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n    pto.wait_flag["PIPE_MTE2",'
    ' "PIPE_V", "EVENT_ID0"]\n\n    pto.vecscope {\n      scf.for %offset_m0 = %c0_m0 to %c64_m0 step %c64_m0 {\n'
    '        %src0_m0 = pto.addptr %ub_in_m0, %c1_m0 : !pto.ptr<f32, ub>, index -> !pto.ptr<f32, ub>\n        %src1_m0 ='
    ' pto.addptr %src0_m0, %c64_m0 : !pto.ptr<f32, ub>, index -> !pto.ptr<f32, ub>\n        %align0_m0 = pto.vldas %src0'
    '_m0 : !pto.ptr<f32, ub> -> !pto.align\n        %mask_m0, %next_remaining_m0 = pto.plt_b32 %c1024_i32_m0 : i3'
    '2 -> !pto.mask<b32>, i32\n        %out0_m0, %align1_m0 = pto.vldus %src0_m0, %align0_m0\n            : !pto.'
    'ptr<f32, ub>, !pto.align -> !pto.vreg<64xf32>, !pto.align\n        %align2_m0 = pto.vldas %src1_m0 : !pto.pt'
    'r<f32, ub> -> !pto.align\n        %out1_m0, %align3_m0 = pto.vldus %src1_m0, %align2_m0\n            : !pto.'
    'ptr<f32, ub>, !pto.align -> !pto.vreg<64xf32>, !pto.align\n        pto.vsts %out0_m0, %ub_out_m0[%offset_m0]'
    ', %mask_m0 : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, index, !pto.mask<b32>\n        pto.vsts %out1_m0, %ub_out_m0[%c6'
    '4_m0], %mask_m0 : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, index, !pto.mask<b32>\n      }\n    }\n\n    pto.set_flag["'
    'PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n    pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n    pto.mte_ub_gm'
    ' %ub_out_m0, %arg1, %c128_i64_m0\n      nburst(%c32_i64_m0, %c128_i64_m0, %c128_i64_m0)\n      : !pto.ptr<f3'
    '2, ub>, !pto.ptr<f32, gm>, i64, i64, i64, i64\n    pto.barrier #pto.pipe<PIPE_ALL>\n\n    }\n    // active m'
    'erged from vabs_kernel_2d\n\n    %c0_m1 = arith.constant 0 : index\n    %c1_m1 = arith.constant 1 : index\n '
    '   %c64_m1 = arith.constant 64 : index\n    %c1024_m1 = arith.constant 1024 : index\n    %c0_i64_m1 = arith.'
    'constant 0 : i64\n    %c1_i64_m1 = arith.constant 1 : i64\n    %c32_i64_m1 = arith.constant 32 : i64\n    %c'
    '128_i64_m1 = arith.constant 128 : i64\n    %c4096_i64_m1 = arith.constant 4096 : i64\n    %c1024_i32_m1 = ar'
    'ith.constant 1024 : i32\n\n    %ub_in_m1 = pto.castptr %c0_i64_m1 : i64 -> !pto.ptr<f32, ub>\n    %ub_out_m1'
    ' = pto.castptr %c4096_i64_m1 : i64 -> !pto.ptr<f32, ub>\n\n    %false_m1 = arith.constant false\n    pto.mte'
    '_gm_ub %arg2, %ub_in_m1, %c0_i64_m1, %c128_i64_m1\n      nburst(%c32_i64_m1, %c128_i64_m1, %c128_i64_m1)\n  '
    '    : !pto.ptr<f32, gm>, !pto.ptr<f32, ub>, i64, i64, i64, i64, i64\n\n    pto.set_flag["PIPE_MTE2", "PIPE_V'
    '", "EVENT_ID0"]\n    pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n\n    pto.vecscope {\n      scf.for '
    '%offset_m1 = %c0_m1 to %c64_m1 step %c64_m1 {\n        %src0_m1 = pto.addptr %ub_in_m1, %c1_m1 : !pto.ptr<f3'
    '2, ub>, index -> !pto.ptr<f32, ub>\n        %align0_m1 = pto.vldas %src0_m1 : !pto.ptr<f32, ub> -> !pto.align\n    '
    '    %mask_m1, %next_remaining_m1 = pto.plt_b32 %c1024_i32_m1 : i32 -> !pto.mask<b32>, i32\n        %out_m1, '
    '%next_align_m1 = pto.vldus %src0_m1, %align0_m1\n            : !pto.ptr<f32, ub>, !pto.align -> !pto.vreg<64'
    'xf32>, !pto.align\n        pto.vsts %out_m1, %ub_out_m1[%offset_m1], %mask_m1 : !pto.vreg<64xf32>, !pto.ptr<'
    'f32, ub>, index, !pto.mask<b32>\n      }\n    }\n\n    pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n    pto.wa'
    'it_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n    pto.mte_ub_gm %ub_out_m1, %arg3, %c128_i64_m1\n      nburst'
    '(%c32_i64_m1, %c128_i64_m1, %c128_i64_m1)\n      : !pto.ptr<f32, ub>, !pto.ptr<f32, gm>, i64, i64, i64, i64'
    '\n    pto.barrier #pto.pipe<PIPE_ALL>\n    %__case_merge_guard = arith.constant false\n\n    // inactive cro'
    'ss-case merge from micro-op/vector-load-store/vlds\n    scf.if %__case_merge_guard {\n      \n          %__d'
    'eep_merge_guard_cmg22_1 = arith.constant false\n          // inactive merged from vabs_kernel_2d_vlds_tail\n'
    '          scf.if %__deep_merge_guard_cmg22_1 {\n      \n          %c0_m0_cmg22_1 = arith.constant 0 : index'
    '\n          %c1_m0_cmg22_1 = arith.constant 1 : index\n          %c64_m0_cmg22_1 = arith.constant 64 : index'
    '\n          %c1024_m0_cmg22_1 = arith.constant 1024 : index\n          %c0_i64_m0_cmg22_1 = arith.constant 0'
    ' : i64\n          %c1_i64_m0_cmg22_1 = arith.constant 1 : i64\n          %c32_i64_m0_cmg22_1 = arith.constan'
    't 32 : i64\n          %c128_i64_m0_cmg22_1 = arith.constant 128 : i64\n          %c4096_i64_m0_cmg22_1 = ari'
    'th.constant 4096 : i64\n          %c1000_i32_m0_cmg22_1 = arith.constant 1000 : i32\n          %c1024_i32_m0'
    '_cmg22_1 = arith.constant 1024 : i32\n      \n          %ub_in_m0_cmg22_1 = pto.castptr %c0_i64_m0_cmg22_1 :'
    ' i64 -> !pto.ptr<f32, ub>\n          %ub_out_m0_cmg22_1 = pto.castptr %c4096_i64_m0_cmg22_1 : i64 -> !pto.pt'
    'r<f32, ub>\n      \n          %false_m0_cmg22_1 = arith.constant false\n          pto.mte_gm_ub %arg0, %ub_i'
    'n_m0_cmg22_1, %c0_i64_m0_cmg22_1, %c128_i64_m0_cmg22_1\n            nburst(%c32_i64_m0_cmg22_1, %c128_i64_m0'
    '_cmg22_1, %c128_i64_m0_cmg22_1)\n            : !pto.ptr<f32, gm>, !pto.ptr<f32, ub>, i64, i64, i64, i64, i64'
    '\n      \n          pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n          pto.wait_flag["PIPE_MTE2", "'
    'PIPE_V", "EVENT_ID0"]\n      \n          pto.vecscope {\n            %__m0_cmg22_1:1 = scf.for %offset_m0_cm'
    'g22_1 = %c0_m0_cmg22_1 to %c1024_m0_cmg22_1 step %c64_m0_cmg22_1 iter_args(%remaining_m0_cmg22_1 = %c1000_i3'
    '2_m0_cmg22_1) -> (i32) {\n              %mask_m0_cmg22_1, %next_remaining_m0_cmg22_1 = pto.plt_b32 %remainin'
    'g_m0_cmg22_1 : i32 -> !pto.mask<b32>, i32\n              %out_m0_cmg22_1 = pto.vlds %ub_in_m0_cmg22_1[%offse'
    't_m0_cmg22_1] {dist = "NORM"} : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>\n              pto.vsts %out_m0_cmg22'
    '_1, %ub_out_m0_cmg22_1[%offset_m0_cmg22_1], %mask_m0_cmg22_1 : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, index, !pto.ma'
    'sk<b32>\n              scf.yield %next_remaining_m0_cmg22_1 : i32\n            }\n          }\n      \n     '
    '     pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n          pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT'
    '_ID0"]\n          pto.mte_ub_gm %ub_out_m0_cmg22_1, %arg1, %c128_i64_m0_cmg22_1\n            nburst(%c32_i64'
    '_m0_cmg22_1, %c128_i64_m0_cmg22_1, %c128_i64_m0_cmg22_1)\n            : !pto.ptr<f32, ub>, !pto.ptr<f32, gm>'
    ', i64, i64, i64, i64\n          pto.barrier #pto.pipe<PIPE_ALL>\n      \n          }\n          // active me'
    'rged from vabs_kernel_2d\n      \n          %c0_m1_cmg22_1 = arith.constant 0 : index\n          %c1_m1_cmg2'
    '2_1 = arith.constant 1 : index\n          %c64_m1_cmg22_1 = arith.constant 64 : index\n          %c1024_m1_c'
    'mg22_1 = arith.constant 1024 : index\n          %c0_i64_m1_cmg22_1 = arith.constant 0 : i64\n          %c1_i'
    '64_m1_cmg22_1 = arith.constant 1 : i64\n          %c32_i64_m1_cmg22_1 = arith.constant 32 : i64\n          %'
    'c128_i64_m1_cmg22_1 = arith.constant 128 : i64\n          %c4096_i64_m1_cmg22_1 = arith.constant 4096 : i64'
    '\n          %c1024_i32_m1_cmg22_1 = arith.constant 1024 : i32\n      \n          %ub_in_m1_cmg22_1 = pto.cas'
    'tptr %c0_i64_m1_cmg22_1 : i64 -> !pto.ptr<f32, ub>\n          %ub_out_m1_cmg22_1 = pto.castptr %c4096_i64_m1'
    '_cmg22_1 : i64 -> !pto.ptr<f32, ub>\n      \n          %false_m1_cmg22_1 = arith.constant false\n          p'
    'to.mte_gm_ub %arg2, %ub_in_m1_cmg22_1, %c0_i64_m1_cmg22_1, %c128_i64_m1_cmg22_1\n            nburst(%c32_i64'
    '_m1_cmg22_1, %c128_i64_m1_cmg22_1, %c128_i64_m1_cmg22_1)\n            : !pto.ptr<f32, gm>, !pto.ptr<f32, ub>'
    ', i64, i64, i64, i64, i64\n      \n          pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n          pto'
    '.wait_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n      \n          pto.vecscope {\n            %__m1_cmg22_1:'
    '1 = scf.for %offset_m1_cmg22_1 = %c0_m1_cmg22_1 to %c1024_m1_cmg22_1 step %c64_m1_cmg22_1 iter_args(%remaini'
    'ng_m1_cmg22_1 = %c1024_i32_m1_cmg22_1) -> (i32) {\n              %mask_m1_cmg22_1, %next_remaining_m1_cmg22_'
    '1 = pto.plt_b32 %remaining_m1_cmg22_1 : i32 -> !pto.mask<b32>, i32\n              %out_m1_cmg22_1 = pto.vlds'
    ' %ub_in_m1_cmg22_1[%offset_m1_cmg22_1] {dist = "NORM"} : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>\n           '
    '   pto.vsts %out_m1_cmg22_1, %ub_out_m1_cmg22_1[%offset_m1_cmg22_1], %mask_m1_cmg22_1 : !pto.vreg<64xf32>, !'
    'pto.ptr<f32, ub>, index, !pto.mask<b32>\n              scf.yield %next_remaining_m1_cmg22_1 : i32\n            }\n '
    '         }\n      \n          pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n          pto.wait_flag["PIP'
    'E_V", "PIPE_MTE3", "EVENT_ID0"]\n          pto.mte_ub_gm %ub_out_m1_cmg22_1, %arg3, %c128_i64_m1_cmg22_1\n  '
    '          nburst(%c32_i64_m1_cmg22_1, %c128_i64_m1_cmg22_1, %c128_i64_m1_cmg22_1)\n            : !pto.ptr<f3'
    '2, ub>, !pto.ptr<f32, gm>, i64, i64, i64, i64\n          pto.barrier #pto.pipe<PIPE_ALL>\n    }\n\n    // in'
    'active cross-case merge from micro-op/vector-load-store/vldsx2-vstsx2\n    scf.if %__case_merge_guard {\n   '
    '   \n          %__deep_merge_guard_cmg22_2 = arith.constant false\n          // inactive merged from vldx2_v'
    'stsx2_b8_f32_kernel\n          scf.if %__deep_merge_guard_cmg22_2 {\n      \n          %c0_m0_cmg22_2 = arit'
    'h.constant 0 : index\n          %c1_m0_cmg22_2 = arith.constant 1 : index\n          %c8_m0_cmg22_2 = arith.'
    'constant 8 : index\n          %c128_m0_cmg22_2 = arith.constant 128 : index\n          %c1_i64_m0_cmg22_2 = '
    'arith.constant 1 : i64\n          %c0_i64_m0_cmg22_2 = arith.constant 0 : i64\n          %c32_i64_m0_cmg22_2'
    ' = arith.constant 32 : i64\n          %c128_i64_m0_cmg22_2 = arith.constant 128 : i64\n          %c4096_i64_'
    'm0_cmg22_2 = arith.constant 4096 : i64\n          %c64_i32_m0_cmg22_2 = arith.constant 64 : i32\n          %'
    'false_m0_cmg22_2 = arith.constant false\n      \n          %ub_in_m0_cmg22_2 = pto.castptr %c0_i64_m0_cmg22_'
    '2 : i64 -> !pto.ptr<f32, ub>\n          %ub_out_m0_cmg22_2 = pto.castptr %c4096_i64_m0_cmg22_2 : i64 -> !pto'
    '.ptr<f32, ub>\n          pto.mte_gm_ub %arg0, %ub_in_m0_cmg22_2, %c0_i64_m0_cmg22_2, %c128_i64_m0_cmg22_2\n '
    '           nburst(%c32_i64_m0_cmg22_2, %c128_i64_m0_cmg22_2, %c128_i64_m0_cmg22_2)\n            : !pto.ptr<f'
    '32, gm>, !pto.ptr<f32, ub>, i64, i64, i64, i64, i64\n      \n          pto.set_flag["PIPE_MTE2", "PIPE_V", "'
    'EVENT_ID0"]\n          pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n      \n          pto.vecscope {\n'
    '            scf.for %group_m0_cmg22_2 = %c0_m0_cmg22_2 to %c8_m0_cmg22_2 step %c1_m0_cmg22_2 {\n            '
    '  %group_base_m0_cmg22_2 = arith.muli %group_m0_cmg22_2, %c128_m0_cmg22_2 : index\n              scf.for %ch'
    'unk_m0_cmg22_2 = %c0_m0_cmg22_2 to %c128_m0_cmg22_2 step %c128_m0_cmg22_2 {\n                %offset_m0_cmg2'
    '2_2 = arith.addi %group_base_m0_cmg22_2, %chunk_m0_cmg22_2 : index\n                %mask_m0_cmg22_2, %remai'
    'ning_m0_cmg22_2 = pto.plt_b32 %c64_i32_m0_cmg22_2 : i32 -> !pto.mask<b32>, i32\n                %low_m0_cmg2'
    '2_2, %high_m0_cmg22_2 = pto.vldsx2 %ub_in_m0_cmg22_2[%offset_m0_cmg22_2], "DINTLV_B8"\n                    :'
    ' !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>, !pto.vreg<64xf32>\n                pto.vstsx2 %low_m0_cmg22_'
    '2, %high_m0_cmg22_2, %ub_out_m0_cmg22_2[%offset_m0_cmg22_2], "INTLV_B8", %mask_m0_cmg22_2\n                 '
    '   : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.ptr<f32, ub>, index,\n                      !pto.mask<b32>\n'
    '              }\n            }\n          }\n      \n          pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID'
    '0"]\n          pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n          pto.mte_ub_gm %ub_out_m0_cmg22_2'
    ', %arg1, %c128_i64_m0_cmg22_2\n            nburst(%c32_i64_m0_cmg22_2, %c128_i64_m0_cmg22_2, %c128_i64_m0_cm'
    'g22_2)\n            : !pto.ptr<f32, ub>, !pto.ptr<f32, gm>, i64, i64, i64, i64\n          pto.barrier #pto.p'
    'ipe<PIPE_ALL>\n      \n          }\n          // active merged from vldx2_vstsx2_kernel\n      \n          %'
    'c0_m1_cmg22_2 = arith.constant 0 : index\n          %c1_m1_cmg22_2 = arith.constant 1 : index\n          %c8'
    '_m1_cmg22_2 = arith.constant 8 : index\n          %c128_m1_cmg22_2 = arith.constant 128 : index\n          %'
    'c1_i64_m1_cmg22_2 = arith.constant 1 : i64\n          %c0_i64_m1_cmg22_2 = arith.constant 0 : i64\n         '
    ' %c32_i64_m1_cmg22_2 = arith.constant 32 : i64\n          %c128_i64_m1_cmg22_2 = arith.constant 128 : i64\n '
    '         %c4096_i64_m1_cmg22_2 = arith.constant 4096 : i64\n          %c64_i32_m1_cmg22_2 = arith.constant 6'
    '4 : i32\n          %false_m1_cmg22_2 = arith.constant false\n      \n          %ub_in_m1_cmg22_2 = pto.castp'
    'tr %c0_i64_m1_cmg22_2 : i64 -> !pto.ptr<f32, ub>\n          %ub_out_m1_cmg22_2 = pto.castptr %c4096_i64_m1_c'
    'mg22_2 : i64 -> !pto.ptr<f32, ub>\n          pto.mte_gm_ub %arg2, %ub_in_m1_cmg22_2, %c0_i64_m1_cmg22_2, %c1'
    '28_i64_m1_cmg22_2\n            nburst(%c32_i64_m1_cmg22_2, %c128_i64_m1_cmg22_2, %c128_i64_m1_cmg22_2)\n    '
    '        : !pto.ptr<f32, gm>, !pto.ptr<f32, ub>, i64, i64, i64, i64, i64\n      \n          pto.set_flag["PIP'
    'E_MTE2", "PIPE_V", "EVENT_ID0"]\n          pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n      \n      '
    '    pto.vecscope {\n            scf.for %group_m1_cmg22_2 = %c0_m1_cmg22_2 to %c8_m1_cmg22_2 step %c1_m1_cmg'
    '22_2 {\n              %group_base_m1_cmg22_2 = arith.muli %group_m1_cmg22_2, %c128_m1_cmg22_2 : index\n     '
    '         scf.for %chunk_m1_cmg22_2 = %c0_m1_cmg22_2 to %c128_m1_cmg22_2 step %c128_m1_cmg22_2 {\n           '
    '     %offset_m1_cmg22_2 = arith.addi %group_base_m1_cmg22_2, %chunk_m1_cmg22_2 : index\n                %mas'
    'k_m1_cmg22_2, %remaining_m1_cmg22_2 = pto.plt_b32 %c64_i32_m1_cmg22_2 : i32 -> !pto.mask<b32>, i32\n        '
    '        %low_m1_cmg22_2, %high_m1_cmg22_2 = pto.vldsx2 %ub_in_m1_cmg22_2[%offset_m1_cmg22_2], "DINTLV_B32"\n'
    '                    : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>, !pto.vreg<64xf32>\n                pto.'
    'vstsx2 %low_m1_cmg22_2, %high_m1_cmg22_2, %ub_out_m1_cmg22_2[%offset_m1_cmg22_2], "INTLV_B32", %mask_m1_cmg2'
    '2_2\n                    : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.ptr<f32, ub>, index,\n                '
    '      !pto.mask<b32>\n              }\n            }\n          }\n      \n          pto.set_flag["PIPE_V", '
    '"PIPE_MTE3", "EVENT_ID0"]\n          pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n          pto.mte_ub'
    '_gm %ub_out_m1_cmg22_2, %arg3, %c128_i64_m1_cmg22_2\n            nburst(%c32_i64_m1_cmg22_2, %c128_i64_m1_cm'
    'g22_2, %c128_i64_m1_cmg22_2)\n            : !pto.ptr<f32, ub>, !pto.ptr<f32, gm>, i64, i64, i64, i64\n      '
    '    pto.barrier #pto.pipe<PIPE_ALL>\n    }\n\n    // inactive cross-case merge from micro-op/vector-load-sto'
    're/vsts\n    scf.if %__case_merge_guard {\n      \n          %__deep_merge_guard_cmg22_3 = arith.constant fa'
    'lse\n          // inactive merged from vsts_tail_kernel\n          scf.if %__deep_merge_guard_cmg22_3 {\n   '
    '   \n          %c0_m0_cmg22_3 = arith.constant 0 : index\n          %c1_i64_m0_cmg22_3 = arith.constant 1 : '
    'i64\n          %c0_i64_m0_cmg22_3 = arith.constant 0 : i64\n          %c32_i64_m0_cmg22_3 = arith.constant 3'
    '2 : i64\n          %c128_i64_m0_cmg22_3 = arith.constant 128 : i64\n          %c4096_i64_m0_cmg22_3 = arith.'
    'constant 4096 : i64\n          %c13_i32_m0_cmg22_3 = arith.constant 13 : i32\n          %false_m0_cmg22_3 = '
    'arith.constant false\n      \n          %ub_in_m0_cmg22_3 = pto.castptr %c0_i64_m0_cmg22_3 : i64 -> !pto.ptr'
    '<f32, ub>\n          %ub_out_m0_cmg22_3 = pto.castptr %c4096_i64_m0_cmg22_3 : i64 -> !pto.ptr<f32, ub>\n    '
    '      pto.mte_gm_ub %arg0, %ub_in_m0_cmg22_3, %c0_i64_m0_cmg22_3, %c128_i64_m0_cmg22_3\n            nburst(%'
    'c32_i64_m0_cmg22_3, %c128_i64_m0_cmg22_3, %c128_i64_m0_cmg22_3)\n            : !pto.ptr<f32, gm>, !pto.ptr<f'
    '32, ub>, i64, i64, i64, i64, i64\n      \n          pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n      '
    '    pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n      \n          %c1_m0_cmg22_3 = arith.constant 1 :'
    ' index\n          pto.vecscope {\n            scf.for %iv_m0_cmg22_3 = %c0_m0_cmg22_3 to %c1_m0_cmg22_3 step'
    ' %c1_m0_cmg22_3 {\n              %mask_m0_cmg22_3, %remaining_m0_cmg22_3 = pto.plt_b32 %c13_i32_m0_cmg22_3 :'
    ' i32 -> !pto.mask<b32>, i32\n              %vec_m0_cmg22_3 = pto.vlds %ub_in_m0_cmg22_3[%c0_m0_cmg22_3] : !p'
    'to.ptr<f32, ub>, index -> !pto.vreg<64xf32>\n              pto.vsts %vec_m0_cmg22_3, %ub_out_m0_cmg22_3[%c0_m0_cmg2'
    '2_3], %mask_m0_cmg22_3\n                  : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, index, !pto.mask<b32>\n          '
    '  }\n          }\n      \n          pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n          pto.wait_fla'
    'g["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n          pto.mte_ub_gm %ub_out_m0_cmg22_3, %arg1, %c128_i64_m0_cmg22'
    '_3\n            nburst(%c32_i64_m0_cmg22_3, %c128_i64_m0_cmg22_3, %c128_i64_m0_cmg22_3)\n            : !pto.'
    'ptr<f32, ub>, !pto.ptr<f32, gm>, i64, i64, i64, i64\n          pto.barrier #pto.pipe<PIPE_ALL>\n      \n    '
    '      }\n          // active merged from vsts_kernel_2d\n      \n          %c0_m1_cmg22_3 = arith.constant 0'
    ' : index\n          %c1_m1_cmg22_3 = arith.constant 1 : index\n          %c64_m1_cmg22_3 = arith.constant 64'
    ' : index\n          %c1024_m1_cmg22_3 = arith.constant 1024 : index\n          %c0_i64_m1_cmg22_3 = arith.co'
    'nstant 0 : i64\n          %c1_i64_m1_cmg22_3 = arith.constant 1 : i64\n          %c32_i64_m1_cmg22_3 = arith'
    '.constant 32 : i64\n          %c128_i64_m1_cmg22_3 = arith.constant 128 : i64\n          %c4096_i64_m1_cmg22'
    '_3 = arith.constant 4096 : i64\n          %c1024_i32_m1_cmg22_3 = arith.constant 1024 : i32\n      \n       '
    '   %ub_in_m1_cmg22_3 = pto.castptr %c0_i64_m1_cmg22_3 : i64 -> !pto.ptr<f32, ub>\n          %ub_out_m1_cmg22'
    '_3 = pto.castptr %c4096_i64_m1_cmg22_3 : i64 -> !pto.ptr<f32, ub>\n      \n          %false_m1_cmg22_3 = ari'
    'th.constant false\n          pto.mte_gm_ub %arg2, %ub_in_m1_cmg22_3, %c0_i64_m1_cmg22_3, %c128_i64_m1_cmg22_'
    '3\n            nburst(%c32_i64_m1_cmg22_3, %c128_i64_m1_cmg22_3, %c128_i64_m1_cmg22_3)\n            : !pto.p'
    'tr<f32, gm>, !pto.ptr<f32, ub>, i64, i64, i64, i64, i64\n      \n          pto.set_flag["PIPE_MTE2", "PIPE_V'
    '", "EVENT_ID0"]\n          pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n      \n          pto.vecscope'
    ' {\n            %__m1_cmg22_3:1 = scf.for %offset_m1_cmg22_3 = %c0_m1_cmg22_3 to %c1024_m1_cmg22_3 step %c64'
    '_m1_cmg22_3 iter_args(%remaining_m1_cmg22_3 = %c1024_i32_m1_cmg22_3) -> (i32) {\n              %mask_m1_cmg2'
    '2_3, %next_remaining_m1_cmg22_3 = pto.plt_b32 %remaining_m1_cmg22_3 : i32 -> !pto.mask<b32>, i32\n          '
    '    %vec_m1_cmg22_3 = pto.vlds %ub_in_m1_cmg22_3[%offset_m1_cmg22_3] : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32'
    '>\n              pto.vsts %vec_m1_cmg22_3, %ub_out_m1_cmg22_3[%offset_m1_cmg22_3], %mask_m1_cmg22_3 {dist = '
    '"NORM_B32"} : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, index, !pto.mask<b32>\n              scf.yield %next_remaining_'
    'm1_cmg22_3 : i32\n            }\n          }\n      \n          pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_I'
    'D0"]\n          pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n          pto.mte_ub_gm %ub_out_m1_cmg22_'
    '3, %arg3, %c128_i64_m1_cmg22_3\n            nburst(%c32_i64_m1_cmg22_3, %c128_i64_m1_cmg22_3, %c128_i64_m1_c'
    'mg22_3)\n            : !pto.ptr<f32, ub>, !pto.ptr<f32, gm>, i64, i64, i64, i64\n          pto.barrier #pto.'
    'pipe<PIPE_ALL>\n    }\n\n    // inactive cross-case merge from micro-op/vector-load-store/vstur\n    scf.if '
    '%__case_merge_guard {\n      \n          %__deep_merge_guard_cmg22_4 = arith.constant false\n          // in'
    'active merged from vstur_init_align_outside_loop_kernel_2d\n          scf.if %__deep_merge_guard_cmg22_4 {\n'
    '      \n          %c1_m0_cmg22_4 = arith.constant 1 : index\n          %c0_i64_m0_cmg22_4 = arith.constant 0'
    ' : i64\n          %c1_i64_m0_cmg22_4 = arith.constant 1 : i64\n          %c8_i32_m0_cmg22_4 = arith.constant'
    ' 8 : i32\n          %c32_i64_m0_cmg22_4 = arith.constant 32 : i64\n          %c128_i64_m0_cmg22_4 = arith.co'
    'nstant 128 : i64\n          %c4096_i64_m0_cmg22_4 = arith.constant 4096 : i64\n          %c0_m0_cmg22_4 = ar'
    'ith.constant 0 : index\n      \n          %ub_in_m0_cmg22_4 = pto.castptr %c0_i64_m0_cmg22_4 : i64 -> !pto.p'
    'tr<f32, ub>\n          %ub_out_m0_cmg22_4 = pto.castptr %c4096_i64_m0_cmg22_4 : i64 -> !pto.ptr<f32, ub>\n  '
    '        %ub_out1_m0_cmg22_4 = pto.addptr %ub_out_m0_cmg22_4, %c1_m0_cmg22_4 : !pto.ptr<f32, ub>, index -> !pto.ptr<'
    'f32, ub>\n      \n          %false_m0_cmg22_4 = arith.constant false\n          pto.mte_gm_ub %arg0, %ub_in_'
    'm0_cmg22_4, %c0_i64_m0_cmg22_4, %c128_i64_m0_cmg22_4\n            nburst(%c32_i64_m0_cmg22_4, %c128_i64_m0_c'
    'mg22_4, %c128_i64_m0_cmg22_4)\n            : !pto.ptr<f32, gm>, !pto.ptr<f32, ub>, i64, i64, i64, i64, i64\n'
    '      \n          pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n          pto.wait_flag["PIPE_MTE2", "PI'
    'PE_V", "EVENT_ID0"]\n      \n          pto.vecscope {\n            pto.sprclr "AR"\n            %align0_m0_c'
    'mg22_4 = pto.init_align : !pto.align\n            %align_final_m0_cmg22_4 = scf.for %offset_m0_cmg22_4 = %c0'
    '_m0_cmg22_4 to %c1_m0_cmg22_4 step %c1_m0_cmg22_4\n                iter_args(%align_iter_m0_cmg22_4 = %align'
    '0_m0_cmg22_4) -> (!pto.align) {\n              %vec_m0_cmg22_4 = pto.vlds %ub_in_m0_cmg22_4[%c0_m0_cmg22_4] '
    ': !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>\n              %mask_m0_cmg22_4, %unused_m0_cmg22_4 = pto.plt_b32 %'
    'c8_i32_m0_cmg22_4 : i32 -> !pto.mask<b32>, i32\n              %sqz_m0_cmg22_4 = pto.vsqz %vec_m0_cmg22_4, %m'
    'ask_m0_cmg22_4 : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>\n              %align1_m0_cmg22_4 = '
    'pto.vstur %align_iter_m0_cmg22_4, %sqz_m0_cmg22_4, %ub_out1_m0_cmg22_4, "POST_UPDATE"\n                  : !'
    'pto.align, !pto.vreg<64xf32>, !pto.ptr<f32, ub> -> !pto.align\n              scf.yield %align1_m0_cmg22_4 : '
    '!pto.align\n            }\n            pto.vstar %align_final_m0_cmg22_4, %ub_out1_m0_cmg22_4 : !pto.align, '
    '!pto.ptr<f32, ub>\n          }\n      \n          pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n        '
    '  pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n          pto.mte_ub_gm %ub_out_m0_cmg22_4, %arg1, %c12'
    '8_i64_m0_cmg22_4\n            nburst(%c32_i64_m0_cmg22_4, %c128_i64_m0_cmg22_4, %c128_i64_m0_cmg22_4)\n     '
    '       : !pto.ptr<f32, ub>, !pto.ptr<f32, gm>, i64, i64, i64, i64\n          pto.barrier #pto.pipe<PIPE_ALL>'
    '\n      \n          }\n          // active merged from vstur_kernel_2d\n      \n          %c1_m1_cmg22_4 = a'
    'rith.constant 1 : index\n          %c0_i64_m1_cmg22_4 = arith.constant 0 : i64\n          %c1_i64_m1_cmg22_4'
    ' = arith.constant 1 : i64\n          %c8_i32_m1_cmg22_4 = arith.constant 8 : i32\n          %c32_i64_m1_cmg2'
    '2_4 = arith.constant 32 : i64\n          %c128_i64_m1_cmg22_4 = arith.constant 128 : i64\n          %c4096_i'
    '64_m1_cmg22_4 = arith.constant 4096 : i64\n          %c0_m1_cmg22_4 = arith.constant 0 : index\n      \n    '
    '      %ub_in_m1_cmg22_4 = pto.castptr %c0_i64_m1_cmg22_4 : i64 -> !pto.ptr<f32, ub>\n          %ub_out_m1_cm'
    'g22_4 = pto.castptr %c4096_i64_m1_cmg22_4 : i64 -> !pto.ptr<f32, ub>\n          %ub_out1_m1_cmg22_4 = pto.ad'
    'dptr %ub_out_m1_cmg22_4, %c1_m1_cmg22_4 : !pto.ptr<f32, ub>, index -> !pto.ptr<f32, ub>\n      \n          %false_m'
    '1_cmg22_4 = arith.constant false\n          pto.mte_gm_ub %arg2, %ub_in_m1_cmg22_4, %c0_i64_m1_cmg22_4, %c12'
    '8_i64_m1_cmg22_4\n            nburst(%c32_i64_m1_cmg22_4, %c128_i64_m1_cmg22_4, %c128_i64_m1_cmg22_4)\n     '
    '       : !pto.ptr<f32, gm>, !pto.ptr<f32, ub>, i64, i64, i64, i64, i64\n      \n          pto.set_flag["PIPE'
    '_MTE2", "PIPE_V", "EVENT_ID0"]\n          pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n      \n       '
    '   pto.vecscope {\n            pto.sprclr "AR"\n            scf.for %offset_m1_cmg22_4 = %c0_m1_cmg22_4 to %'
    'c1_m1_cmg22_4 step %c1_m1_cmg22_4 {\n              %vec_m1_cmg22_4 = pto.vlds %ub_in_m1_cmg22_4[%c0_m1_cmg22'
    '_4] : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>\n              %mask_m1_cmg22_4, %unused_m1_cmg22_4 = pto.plt_b'
    '32 %c8_i32_m1_cmg22_4 : i32 -> !pto.mask<b32>, i32\n              %sqz_m1_cmg22_4 = pto.vsqz %vec_m1_cmg22_4'
    ', %mask_m1_cmg22_4 : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>\n              %align0_m1_cmg22_'
    '4 = pto.init_align : !pto.align\n              %align1_m1_cmg22_4 = pto.vstur %align0_m1_cmg22_4, %sqz_m1_cm'
    'g22_4, %ub_out1_m1_cmg22_4, "POST_UPDATE"\n                  : !pto.align, !pto.vreg<64xf32>, !pto.ptr<f32, '
    'ub> -> !pto.align\n              pto.vstar %align1_m1_cmg22_4, %ub_out1_m1_cmg22_4 : !pto.align, !pto.ptr<f3'
    '2, ub>\n            }\n          }\n      \n          pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n    '
    '      pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n          pto.mte_ub_gm %ub_out_m1_cmg22_4, %arg3, '
    '%c128_i64_m1_cmg22_4\n            nburst(%c32_i64_m1_cmg22_4, %c128_i64_m1_cmg22_4, %c128_i64_m1_cmg22_4)\n '
    '           : !pto.ptr<f32, ub>, !pto.ptr<f32, gm>, i64, i64, i64, i64\n          pto.barrier #pto.pipe<PIPE_'
    'ALL>\n    }\n\n    return\n  }\n}\n'
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

    # case: micro-op/vector-load-store/vldas-vldus
    # family: vector-load-store
    # target_ops: pto.vldas, pto.vldus
    # scenarios: core-f32, full-mask, unaligned, stream-state
    # NOTE: bulk-generated coverage skeleton.
    # coding=utf-8

    import argparse
    from pathlib import Path

    import numpy as np


    ROWS = 32
    COLS = 32
    LANES = 64
    SEED = 19


    def _generate(seed: int) -> dict:
        rng = np.random.default_rng(seed)
        v1 = rng.uniform(-8.0, 8.0, size=(ROWS, COLS)).astype(np.float32)
        v2 = np.zeros((ROWS, COLS), dtype=np.float32)
        golden_v2 = np.zeros((ROWS, COLS), dtype=np.float32)
        flat_in = v1.reshape(-1)
        flat_out = golden_v2.reshape(-1)
        flat_out[:LANES] = flat_in[1 : 1 + LANES]

        _files["v1.bin"] = v1.reshape(-1)
        _files["v2.bin"] = v2.reshape(-1)
        _files["golden_v2.bin"] = golden_v2.reshape(-1)


    def main() -> None:
        parser = argparse.ArgumentParser(
            description="Generate numpy-based inputs/golden for VPTO micro-op vldas-vldus validation."
        )
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

    # case: micro-op/vector-load-store/vldas-vldus
    # family: vector-load-store
    # target_ops: pto.vldas, pto.vldus
    # scenarios: core-f32, full-mask, unaligned, stream-state
    # NOTE: bulk-generated coverage skeleton.
    # coding=utf-8

    import os
    import sys
    import numpy as np


    REPEAT_BYTES = 256
    PREFIX_ELEMS = 64


    def _ceil_div(x, y):
        return (x + y - 1) // y


    def _packed_pred_storage_bytes(logical_elems, src_elem_bytes):
        logical_elems = int(logical_elems)
        src_elem_bytes = int(src_elem_bytes)
        if logical_elems <= 0:
            raise ValueError(f"logical_elems must be > 0, got {logical_elems}")
        if src_elem_bytes not in (1, 2, 4):
            raise ValueError(f"unsupported packed predicate source size: {src_elem_bytes}")

        repeat_elems = REPEAT_BYTES // src_elem_bytes
        if src_elem_bytes == 4:
            repeat_times = _ceil_div(logical_elems, repeat_elems) + 1
            loop_count = repeat_times // 2
            return loop_count * 16

        repeat_times = _ceil_div(logical_elems, repeat_elems)
        return repeat_times * (repeat_elems // 8)


    def compare_bin(golden_path, output_path, dtype, eps):
        if not os.path.exists(output_path):
            print(f"[ERROR] Output missing: {output_path}")
            return False
        if not os.path.exists(golden_path):
            print(f"[ERROR] Golden missing: {golden_path}")
            return False
        dtype_np = np.dtype(dtype)
        golden = np.fromfile(golden_path, dtype=dtype_np)
        output = np.fromfile(output_path, dtype=dtype_np)
        if golden.shape != output.shape:
            print(f"[ERROR] Shape mismatch: {golden_path} {golden.shape} vs {output_path} {output.shape}")
            return False
        if not np.allclose(golden, output, atol=eps, rtol=eps, equal_nan=True):
            if golden.size:
                if np.issubdtype(dtype_np, np.floating):
                    g = golden.astype(np.float64, copy=False)
                    o = output.astype(np.float64, copy=False)
                elif np.issubdtype(dtype_np, np.integer) or np.issubdtype(dtype_np, np.unsignedinteger):
                    g = golden.astype(np.int64, copy=False)
                    o = output.astype(np.int64, copy=False)
                else:
                    g = golden.astype(np.float64, copy=False)
                    o = output.astype(np.float64, copy=False)
                abs_diff = np.abs(g - o)
                idx = int(np.argmax(abs_diff))
                diff = float(abs_diff[idx])
                print(
                    f"[ERROR] Mismatch: {golden_path} vs {output_path}, max diff={diff} at idx={idx} "
                    f"(golden={g[idx]}, out={o[idx]}, dtype={dtype_np})"
                )
            else:
                print(f"[ERROR] Mismatch: {golden_path} vs {output_path}, empty buffers, dtype={dtype_np}")
            return False
        return True


    def compare_bin_prefix(golden_path, output_path, dtype, eps, count):
        if not os.path.exists(output_path):
            print(f"[ERROR] Output missing: {output_path}")
            return False
        if not os.path.exists(golden_path):
            print(f"[ERROR] Golden missing: {golden_path}")
            return False
        try:
            count = int(count)
        except Exception:
            print(f"[ERROR] Invalid prefix count: {count}")
            return False
        if count <= 0:
            print(f"[ERROR] Invalid prefix count: {count}")
            return False

        dtype_np = np.dtype(dtype)
        golden = np.fromfile(golden_path, dtype=dtype_np, count=count)
        output = np.fromfile(output_path, dtype=dtype_np, count=count)

        if golden.size != count or output.size != count:
            print(
                f"[ERROR] Prefix read too small: need={count} elems, "
                f"golden={golden.size}, out={output.size}"
            )
            return False

        if not np.allclose(golden, output, atol=eps, rtol=eps, equal_nan=True):
            if golden.size:
                if np.issubdtype(dtype_np, np.floating):
                    g = golden.astype(np.float64, copy=False)
                    o = output.astype(np.float64, copy=False)
                elif np.issubdtype(dtype_np, np.integer) or np.issubdtype(dtype_np, np.unsignedinteger):
                    g = golden.astype(np.int64, copy=False)
                    o = output.astype(np.int64, copy=False)
                else:
                    g = golden.astype(np.float64, copy=False)
                    o = output.astype(np.float64, copy=False)
                abs_diff = np.abs(g - o)
                idx = int(np.argmax(abs_diff))
                diff = float(abs_diff[idx])
                print(
                    f"[ERROR] Mismatch (prefix): {golden_path} vs {output_path}, max diff={diff} at idx={idx} "
                    f"(golden={g[idx]}, out={o[idx]}, dtype={dtype_np}, count={count})"
                )
            else:
                print(f"[ERROR] Mismatch (prefix): {golden_path} vs {output_path}, empty buffers, dtype={dtype_np}")
            return False
        return True


    def compare_packed_pred_mask(golden_path, output_path, logical_elems, src_elem_bytes):
        """
        Compare outputs of pto.tcmp / pto.tcmps.

        PTO-ISA stores packed predicate results as a linear PK byte stream via
        `psts`, with the exact written prefix length determined by the typed
        TCMP/TCMPS repeat schedule. Compare only that semantic prefix.
        """
        if not os.path.exists(output_path):
            print(f"[ERROR] Output missing: {output_path}")
            return False
        if not os.path.exists(golden_path):
            print(f"[ERROR] Golden missing: {golden_path}")
            return False
        try:
            logical_elems = int(logical_elems)
            src_elem_bytes = int(src_elem_bytes)
        except Exception:
            print(
                "[ERROR] Invalid packed mask compare arguments: "
                f"logical_elems={logical_elems} src_elem_bytes={src_elem_bytes}"
            )
            return False
        if logical_elems <= 0 or src_elem_bytes <= 0:
            print(
                "[ERROR] Invalid packed mask compare arguments: "
                f"logical_elems={logical_elems} src_elem_bytes={src_elem_bytes}"
            )
            return False

        golden = np.fromfile(golden_path, dtype=np.uint8)
        output = np.fromfile(output_path, dtype=np.uint8)
        try:
            prefix_bytes = _packed_pred_storage_bytes(logical_elems, src_elem_bytes)
        except ValueError as exc:
            print(f"[ERROR] {exc}")
            return False

        if golden.size < prefix_bytes or output.size < prefix_bytes:
            print(
                f"[ERROR] Packed mask buffer too small: need={prefix_bytes} bytes, "
                f"golden={golden.size}, out={output.size}"
            )
            return False

        golden_sel = golden[:prefix_bytes]
        output_sel = output[:prefix_bytes]

        if not np.array_equal(golden_sel, output_sel):
            diff = np.nonzero(golden_sel != output_sel)[0]
            idx = int(diff[0]) if diff.size else 0
            print(
                f"[ERROR] Mismatch (packed mask): {golden_path} vs {output_path}, first diff at idx={idx} "
                f"(golden={int(golden_sel[idx])}, out={int(output_sel[idx])}, "
                f"logical_elems={logical_elems}, src_elem_bytes={src_elem_bytes}, prefix_bytes={prefix_bytes})"
            )
            return False
        return True


    def main():
        strict = os.getenv("COMPARE_STRICT", "1") != "0"
        ok = True
        ok = compare_bin_prefix("golden_v2.bin", "v2.bin", np.float32, 0.0001, PREFIX_ELEMS) and ok
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
        'f32'
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
    1
]
ARG_DTYPES = [
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
    name='vldas_vldus_deep_merged_kernel',
    target="a5",
    backend="vpto",
    mode="explicit",
    source=KERNEL_SOURCE,
)
def kernel(
    arg0: pto.ptr(pto.f32, "gm"),
    arg1: pto.ptr(pto.f32, "gm"),
    arg2: pto.ptr(pto.f32, "gm"),
    arg3: pto.ptr(pto.f32, "gm"),
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
    name='legacy_micro_op_vector_load_store_vldas_vldus',
    grid=1,
    kernel=kernel,
    make_case=make_case,
    check=check_case,
)]

auto_main(globals())
