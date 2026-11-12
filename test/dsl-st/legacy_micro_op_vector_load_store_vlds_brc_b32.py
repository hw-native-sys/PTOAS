#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Migrated from test/vpto/cases/micro-op/vector-load-store/vlds-brc-b32."""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

from common import auto_main
from ptodsl import pto

KERNEL_SOURCE = (
    '// -----------------------------------------------------------------------------\n// case: micro-op/vector-l'
    'oad-store/vlds-brc-b32\n// family: vector-load-store\n// target_ops: pto.vlds\n// scenarios: core-f32, full-'
    'mask, aligned, dist-brc-b32\n// NOTE: bulk-generated coverage skeleton. Parser/verifier/lowering failure is'
    '\n// still a valid test conclusion in the current coverage-first phase.\n// --------------------------------'
    '---------------------------------------------\n// =========================================================='
    '===================\n// abs_kernel_2d: Element-wise absolute value on a 32x32 f32 tile\n// ================='
    '============================================================\n// This kernel computes abs(input) for a 32x32'
    ' float32 matrix (1024 elements).\n//\n// Memory Layout:\n//   - Input:  arg0 -> GM (Global Memory)\n//   - O'
    'utput: arg1 -> GM (Global Memory)\n//   - UB (Unified Buffer) at offset 0:    input tile  (4096 bytes = 32*3'
    '2*4)\n//   - UB at offset 4096:                  output tile (4096 bytes = 32*32*4)\n//\n// Pipeline:\n//   '
    '1. DMA: GM -> UB (MTE2 pipe) - copy input tile to UB\n//   2. Sync: wait for MTE2 -> V pipe handoff\n//   3.'
    ' Compute: vabs on 64-element vectors (V pipe) - 16 iterations for 1024 elements\n//   4. Sync: wait for V ->'
    ' MTE3 pipe handoff\n//   5. DMA: UB -> GM (MTE3 pipe) - copy result tile back to GM\n// ===================='
    '=========================================================\n\nmodule attributes {pto.target_arch = "a5", pto.'
    'kernel_kind = #pto.kernel_kind<vector>} {\n  func.func @vabs_kernel_2d(%arg0: !pto.ptr<f32, gm>, %arg1: !pto'
    '.ptr<f32, gm>) attributes {pto.kernel} {\n    %c0 = arith.constant 0 : index\n    %c1 = arith.constant 1 : i'
    'ndex\n    %c64 = arith.constant 64 : index\n    %c1024 = arith.constant 1024 : index\n    %c0_i64 = arith.co'
    'nstant 0 : i64\n    %c1_i64 = arith.constant 1 : i64\n    %c32_i64 = arith.constant 32 : i64\n    %c128_i64 '
    '= arith.constant 128 : i64\n    %c4096_i64 = arith.constant 4096 : i64\n    %c1024_i32 = arith.constant 1024'
    ' : i32\n\n    %ub_in = pto.castptr %c0_i64 : i64 -> !pto.ptr<f32, ub>\n    %ub_out = pto.castptr %c4096_i64 '
    ': i64 -> !pto.ptr<f32, ub>\n\n    %false = arith.constant false\n    pto.mte_gm_ub %arg0, %ub_in, %c0_i64, %'
    'c128_i64\n      nburst(%c32_i64, %c128_i64, %c128_i64)\n      : !pto.ptr<f32, gm>, !pto.ptr<f32, ub>, i64, i'
    '64, i64, i64, i64\n\n    pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n    pto.wait_flag["PIPE_MTE2", "P'
    'IPE_V", "EVENT_ID0"]\n\n    pto.vecscope {\n      %_:1 = scf.for %offset = %c0 to %c1024 step %c64 iter_args'
    '(%remaining = %c1024_i32) -> (i32) {\n        %mask, %next_remaining = pto.plt_b32 %remaining : i32 -> !pto.'
    'mask<b32>, i32\n        %out = pto.vlds %ub_in[%offset] {dist = "BRC_B32"} : !pto.ptr<f32, ub>, index -> !pto.vreg<'
    '64xf32>\n        pto.vsts %out, %ub_out[%offset], %mask : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, index, !pto.mask<b3'
    '2>\n        scf.yield %next_remaining : i32\n      }\n    }\n\n    pto.set_flag["PIPE_V", "PIPE_MTE3", "EVEN'
    'T_ID0"]\n    pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n    pto.mte_ub_gm %ub_out, %arg1, %c128_i64'
    '\n      nburst(%c32_i64, %c128_i64, %c128_i64)\n      : !pto.ptr<f32, ub>, !pto.ptr<f32, gm>, i64, i64, i64,'
    ' i64\n    pto.barrier #pto.pipe<PIPE_ALL>\n    %__case_merge_guard = arith.constant false\n\n    // inactive'
    ' cross-case merge from micro-op/vector-load-store/vlds-brc-b8-f32\n    scf.if %__case_merge_guard {\n      '
    '\n          %c0_cmg23_1 = arith.constant 0 : index\n          %c64_cmg23_1 = arith.constant 64 : index\n    '
    '      %c1024_cmg23_1 = arith.constant 1024 : index\n          %c0_i64_cmg23_1 = arith.constant 0 : i64\n    '
    '      %c1_i64_cmg23_1 = arith.constant 1 : i64\n          %c32_i64_cmg23_1 = arith.constant 32 : i64\n      '
    '    %c128_i64_cmg23_1 = arith.constant 128 : i64\n          %c4096_i64_cmg23_1 = arith.constant 4096 : i64\n'
    '          %c1024_i32_cmg23_1 = arith.constant 1024 : i32\n      \n          %ub_in_cmg23_1 = pto.castptr %c0'
    '_i64_cmg23_1 : i64 -> !pto.ptr<f32, ub>\n          %ub_out_cmg23_1 = pto.castptr %c4096_i64_cmg23_1 : i64 ->'
    ' !pto.ptr<f32, ub>\n      \n          %false_cmg23_1 = arith.constant false\n          pto.mte_gm_ub %arg0, '
    '%ub_in_cmg23_1, %c0_i64_cmg23_1, %c128_i64_cmg23_1\n            nburst(%c32_i64_cmg23_1, %c128_i64_cmg23_1, '
    '%c128_i64_cmg23_1)\n            : !pto.ptr<f32, gm>, !pto.ptr<f32, ub>, i64, i64, i64, i64, i64\n      \n   '
    '       pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n          pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVE'
    'NT_ID0"]\n      \n          pto.vecscope {\n            %__cmg23_1:1 = scf.for %offset_cmg23_1 = %c0_cmg23_1'
    ' to %c1024_cmg23_1 step %c64_cmg23_1 iter_args(%remaining_cmg23_1 = %c1024_i32_cmg23_1) -> (i32) {\n        '
    '      %mask_cmg23_1, %next_remaining_cmg23_1 = pto.plt_b32 %remaining_cmg23_1 : i32 -> !pto.mask<b32>, i32\n'
    '              %out_cmg23_1 = pto.vlds %ub_in_cmg23_1[%offset_cmg23_1] {dist = "BRC_B8"} : !pto.ptr<f32, ub>, index '
    '-> !pto.vreg<64xf32>\n              pto.vsts %out_cmg23_1, %ub_out_cmg23_1[%offset_cmg23_1], %mask_cmg23_1 :'
    ' !pto.vreg<64xf32>, !pto.ptr<f32, ub>, index, !pto.mask<b32>\n              scf.yield %next_remaining_cmg23_1 : i32'
    '\n            }\n          }\n      \n          pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n          '
    'pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n          pto.mte_ub_gm %ub_out_cmg23_1, %arg1, %c128_i64'
    '_cmg23_1\n            nburst(%c32_i64_cmg23_1, %c128_i64_cmg23_1, %c128_i64_cmg23_1)\n            : !pto.ptr'
    '<f32, ub>, !pto.ptr<f32, gm>, i64, i64, i64, i64\n          pto.barrier #pto.pipe<PIPE_ALL>\n    }\n\n    //'
    ' inactive cross-case merge from micro-op/vector-load-store/vlds-dma-loop\n    scf.if %__case_merge_guard {\n'
    '      \n          %c0_cmg23_2 = arith.constant 0 : index\n          %c1_cmg23_2 = arith.constant 1 : index\n'
    '          %c64_cmg23_2 = arith.constant 64 : index\n          %c512_cmg23_2 = arith.constant 512 : index\n  '
    '        %c0_i64_cmg23_2 = arith.constant 0 : i64\n          %c1_i64_cmg23_2 = arith.constant 1 : i64\n      '
    '    %c2_i64_cmg23_2 = arith.constant 2 : i64\n          %c8_i64_cmg23_2 = arith.constant 8 : i64\n          '
    '%c224_i64_cmg23_2 = arith.constant 224 : i64\n          %c256_i64_cmg23_2 = arith.constant 256 : i64\n      '
    '    %c448_i64_cmg23_2 = arith.constant 448 : i64\n          %c512_i64_cmg23_2 = arith.constant 512 : i64\n  '
    '        %c896_i64_cmg23_2 = arith.constant 896 : i64\n          %c1024_i64_cmg23_2 = arith.constant 1024 : i'
    '64\n          %c2048_i64_cmg23_2 = arith.constant 2048 : i64\n          %c512_i32_cmg23_2 = arith.constant 5'
    '12 : i32\n          %pad_cmg23_2 = arith.constant 1.000000e+00 : f32\n      \n          %ub_in_cmg23_2 = pto'
    '.castptr %c0_i64_cmg23_2 : i64 -> !pto.ptr<f32, ub>\n          %ub_out_cmg23_2 = pto.castptr %c2048_i64_cmg2'
    '3_2 : i64 -> !pto.ptr<f32, ub>\n      \n          pto.mte_gm_ub %arg0, %ub_in_cmg23_2, %c0_i64_cmg23_2, %c22'
    '4_i64_cmg23_2\n            nburst(%c1_i64_cmg23_2, %c224_i64_cmg23_2, %c256_i64_cmg23_2)\n            loop(%'
    'c2_i64_cmg23_2, %c224_i64_cmg23_2, %c256_i64_cmg23_2)\n            loop(%c2_i64_cmg23_2, %c448_i64_cmg23_2, '
    '%c512_i64_cmg23_2)\n            loop(%c2_i64_cmg23_2, %c896_i64_cmg23_2, %c1024_i64_cmg23_2)\n            pa'
    'd(%pad_cmg23_2, %c0_i64_cmg23_2, %c8_i64_cmg23_2)\n            : !pto.ptr<f32, gm>, !pto.ptr<f32, ub>, i64, '
    'i64, i64, i64, i64,\n              loop i64, i64, i64, loop i64, i64, i64, loop i64, i64, i64, pad f32, i64,'
    ' i64\n      \n          pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n          pto.wait_flag["PIPE_MTE2'
    '", "PIPE_V", "EVENT_ID0"]\n      \n          pto.vecscope {\n            %__cmg23_2:1 = scf.for %offset_cmg2'
    '3_2 = %c0_cmg23_2 to %c512_cmg23_2 step %c64_cmg23_2 iter_args(%remaining_cmg23_2 = %c512_i32_cmg23_2) -> (i'
    '32) {\n              %mask_cmg23_2, %next_remaining_cmg23_2 = pto.plt_b32 %remaining_cmg23_2 : i32 -> !pto.m'
    'ask<b32>, i32\n              %value_cmg23_2 = pto.vlds %ub_in_cmg23_2[%offset_cmg23_2] {dist = "NORM"} : !pt'
    'o.ptr<f32, ub>, index -> !pto.vreg<64xf32>\n              pto.vsts %value_cmg23_2, %ub_out_cmg23_2[%offset_cmg23_2]'
    ', %mask_cmg23_2 : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, index, !pto.mask<b32>\n              scf.yield %next_remain'
    'ing_cmg23_2 : i32\n            }\n          }\n      \n          pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_'
    'ID0"]\n          pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n      \n          pto.mte_ub_gm %ub_out_'
    'cmg23_2, %arg1, %c256_i64_cmg23_2\n            nburst(%c1_i64_cmg23_2, %c256_i64_cmg23_2, %c256_i64_cmg23_2)'
    '\n            loop(%c2_i64_cmg23_2, %c256_i64_cmg23_2, %c256_i64_cmg23_2)\n            loop(%c2_i64_cmg23_2,'
    ' %c512_i64_cmg23_2, %c512_i64_cmg23_2)\n            loop(%c2_i64_cmg23_2, %c1024_i64_cmg23_2, %c1024_i64_cmg'
    '23_2)\n            : !pto.ptr<f32, ub>, !pto.ptr<f32, gm>, i64, i64, i64, i64,\n              loop i64, i64,'
    ' i64, loop i64, i64, i64, loop i64, i64, i64\n      \n          pto.barrier #pto.pipe<PIPE_ALL>\n    }\n\n  '
    '  // inactive cross-case merge from micro-op/vector-load-store/vldsx2-layout-check\n    scf.if %__case_merge'
    '_guard {\n      \n          %c0_cmg23_3 = arith.constant 0 : index\n          %c1_cmg23_3 = arith.constant 1'
    ' : index\n          %c64_cmg23_3 = arith.constant 64 : index\n          %c8_cmg23_3 = arith.constant 8 : ind'
    'ex\n          %c128_cmg23_3 = arith.constant 128 : index\n          %c1_i64_cmg23_3 = arith.constant 1 : i64'
    '\n          %c0_i64_cmg23_3 = arith.constant 0 : i64\n          %c32_i64_cmg23_3 = arith.constant 32 : i64\n'
    '          %c128_i64_cmg23_3 = arith.constant 128 : i64\n          %c4096_i64_cmg23_3 = arith.constant 4096 :'
    ' i64\n          %c64_i32_cmg23_3 = arith.constant 64 : i32\n          %false_cmg23_3 = arith.constant false'
    '\n      \n          %ub_in_cmg23_3 = pto.castptr %c0_i64_cmg23_3 : i64 -> !pto.ptr<f32, ub>\n          %ub_o'
    'ut_cmg23_3 = pto.castptr %c4096_i64_cmg23_3 : i64 -> !pto.ptr<f32, ub>\n          pto.mte_gm_ub %arg0, %ub_i'
    'n_cmg23_3, %c0_i64_cmg23_3, %c128_i64_cmg23_3\n            nburst(%c32_i64_cmg23_3, %c128_i64_cmg23_3, %c128'
    '_i64_cmg23_3)\n            : !pto.ptr<f32, gm>, !pto.ptr<f32, ub>, i64, i64, i64, i64, i64\n      \n        '
    '  pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n          pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID'
    '0"]\n      \n          pto.vecscope {\n            scf.for %group_cmg23_3 = %c0_cmg23_3 to %c8_cmg23_3 step '
    '%c1_cmg23_3 {\n              %group_base_cmg23_3 = arith.muli %group_cmg23_3, %c128_cmg23_3 : index\n       '
    '       scf.for %chunk_cmg23_3 = %c0_cmg23_3 to %c128_cmg23_3 step %c128_cmg23_3 {\n                %offset_c'
    'mg23_3 = arith.addi %group_base_cmg23_3, %chunk_cmg23_3 : index\n                %high_offset_cmg23_3 = arit'
    'h.addi %offset_cmg23_3, %c64_cmg23_3 : index\n                %mask_cmg23_3, %remaining_cmg23_3 = pto.plt_b3'
    '2 %c64_i32_cmg23_3 : i32 -> !pto.mask<b32>, i32\n                %x_cmg23_3, %y_cmg23_3 = pto.vldsx2 %ub_in_'
    'cmg23_3[%offset_cmg23_3], "DINTLV_B32"\n                    : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>,'
    ' !pto.vreg<64xf32>\n                pto.vsts %x_cmg23_3, %ub_out_cmg23_3[%offset_cmg23_3], %mask_cmg23_3 {di'
    'st = "NORM_B32"}\n                    : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, index, !pto.mask<b32>\n              '
    '  pto.vsts %y_cmg23_3, %ub_out_cmg23_3[%high_offset_cmg23_3], %mask_cmg23_3 {dist = "NORM_B32"}\n           '
    '         : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, index, !pto.mask<b32>\n              }\n            }\n          }'
    '\n      \n          pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n          pto.wait_flag["PIPE_V", "PIP'
    'E_MTE3", "EVENT_ID0"]\n          pto.mte_ub_gm %ub_out_cmg23_3, %arg1, %c128_i64_cmg23_3\n            nburst'
    '(%c32_i64_cmg23_3, %c128_i64_cmg23_3, %c128_i64_cmg23_3)\n            : !pto.ptr<f32, ub>, !pto.ptr<f32, gm>'
    ', i64, i64, i64, i64\n          pto.barrier #pto.pipe<PIPE_ALL>\n    }\n\n    // inactive cross-case merge f'
    'rom micro-op/vector-load-store/vsldb\n    scf.if %__case_merge_guard {\n      \n          %c0_cmg23_4 = arit'
    'h.constant 0 : index\n          %c1_i64_cmg23_4 = arith.constant 1 : i64\n          %c0_i64_cmg23_4 = arith.'
    'constant 0 : i64\n          %c2_i16_cmg23_4 = arith.constant 2 : i16\n          %c4_i16_cmg23_4 = arith.cons'
    'tant 4 : i16\n          %c32_i64_cmg23_4 = arith.constant 32 : i64\n          %c64_i32_cmg23_4 = arith.const'
    'ant 64 : i32\n          %c128_i64_cmg23_4 = arith.constant 128 : i64\n          %c4096_i64_cmg23_4 = arith.c'
    'onstant 4096 : i64\n      \n          %ub_in_cmg23_4 = pto.castptr %c0_i64_cmg23_4 : i64 -> !pto.ptr<f32, ub'
    '>\n          %ub_out_cmg23_4 = pto.castptr %c4096_i64_cmg23_4 : i64 -> !pto.ptr<f32, ub>\n      \n          '
    '%false_cmg23_4 = arith.constant false\n          pto.mte_gm_ub %arg0, %ub_in_cmg23_4, %c0_i64_cmg23_4, %c128'
    '_i64_cmg23_4\n            nburst(%c32_i64_cmg23_4, %c128_i64_cmg23_4, %c128_i64_cmg23_4)\n            : !pto'
    '.ptr<f32, gm>, !pto.ptr<f32, ub>, i64, i64, i64, i64, i64\n      \n          pto.set_flag["PIPE_MTE2", "PIPE'
    '_V", "EVENT_ID0"]\n          pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n      \n          %c1_cmg23_'
    '4 = arith.constant 1 : index\n          pto.vecscope {\n            scf.for %iv_cmg23_4 = %c0_cmg23_4 to %c1'
    '_cmg23_4 step %c1_cmg23_4 {\n              %mask_cmg23_4, %next_remaining_cmg23_4 = pto.plt_b32 %c64_i32_cmg'
    '23_4 : i32 -> !pto.mask<b32>, i32\n              %loaded_cmg23_4 = pto.vsldb %ub_in_cmg23_4, %c2_i16_cmg23_4'
    ', %c4_i16_cmg23_4, %mask_cmg23_4 : !pto.ptr<f32, ub>, i16, i16, !pto.mask<b32> -> !pto.vreg<64xf32>\n       '
    '       pto.vsts %loaded_cmg23_4, %ub_out_cmg23_4[%c0_cmg23_4], %mask_cmg23_4 : !pto.vreg<64xf32>, !pto.ptr<f'
    '32, ub>, index, !pto.mask<b32>\n            }\n          }\n      \n          pto.set_flag["PIPE_V", "PIPE_MTE3", "'
    'EVENT_ID0"]\n          pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n          pto.mte_ub_gm %ub_out_cm'
    'g23_4, %arg1, %c128_i64_cmg23_4\n            nburst(%c32_i64_cmg23_4, %c128_i64_cmg23_4, %c128_i64_cmg23_4)'
    '\n            : !pto.ptr<f32, ub>, !pto.ptr<f32, gm>, i64, i64, i64, i64\n          pto.barrier #pto.pipe<PI'
    'PE_ALL>\n    }\n\n    // inactive cross-case merge from micro-op/vector-load-store/vsstb\n    scf.if %__case'
    '_merge_guard {\n      \n          %c0_cmg23_5 = arith.constant 0 : index\n          %c1_i64_cmg23_5 = arith.'
    'constant 1 : i64\n          %c0_i64_cmg23_5 = arith.constant 0 : i64\n          %c2_i16_cmg23_5 = arith.cons'
    'tant 2 : i16\n          %c4_i16_cmg23_5 = arith.constant 4 : i16\n          %c32_i64_cmg23_5 = arith.constan'
    't 32 : i64\n          %c64_i32_cmg23_5 = arith.constant 64 : i32\n          %c128_i64_cmg23_5 = arith.consta'
    'nt 128 : i64\n          %c4096_i64_cmg23_5 = arith.constant 4096 : i64\n      \n          %ub_in_cmg23_5 = p'
    'to.castptr %c0_i64_cmg23_5 : i64 -> !pto.ptr<f32, ub>\n          %ub_out_cmg23_5 = pto.castptr %c4096_i64_cm'
    'g23_5 : i64 -> !pto.ptr<f32, ub>\n      \n          %false_cmg23_5 = arith.constant false\n          pto.mte'
    '_gm_ub %arg0, %ub_in_cmg23_5, %c0_i64_cmg23_5, %c128_i64_cmg23_5\n            nburst(%c32_i64_cmg23_5, %c128'
    '_i64_cmg23_5, %c128_i64_cmg23_5)\n            : !pto.ptr<f32, gm>, !pto.ptr<f32, ub>, i64, i64, i64, i64, i6'
    '4\n      \n          pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n          pto.wait_flag["PIPE_MTE2", '
    '"PIPE_V", "EVENT_ID0"]\n      \n          %c1_cmg23_5 = arith.constant 1 : index\n          pto.vecscope {\n'
    '            scf.for %iv_cmg23_5 = %c0_cmg23_5 to %c1_cmg23_5 step %c1_cmg23_5 {\n              %mask_cmg23_5'
    ', %next_remaining_cmg23_5 = pto.plt_b32 %c64_i32_cmg23_5 : i32 -> !pto.mask<b32>, i32\n              %value_'
    'cmg23_5 = pto.vlds %ub_in_cmg23_5[%c0_cmg23_5] : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>\n              pto.v'
    'sstb %value_cmg23_5, %ub_out_cmg23_5, %c2_i16_cmg23_5, %c4_i16_cmg23_5, %mask_cmg23_5 : !pto.vreg<64xf32>, !'
    'pto.ptr<f32, ub>, i16, i16, !pto.mask<b32>\n              pto.mem_bar "VST_VLD"\n              %roundtrip_cm'
    'g23_5 = pto.vsldb %ub_out_cmg23_5, %c2_i16_cmg23_5, %c4_i16_cmg23_5, %mask_cmg23_5 : !pto.ptr<f32, ub>, i16,'
    ' i16, !pto.mask<b32> -> !pto.vreg<64xf32>\n              pto.vsts %roundtrip_cmg23_5, %ub_in_cmg23_5[%c0_cmg'
    '23_5], %mask_cmg23_5 : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, index, !pto.mask<b32>\n            }\n          }\n   '
    '   \n          pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n          pto.wait_flag["PIPE_V", "PIPE_MTE'
    '3", "EVENT_ID0"]\n          pto.mte_ub_gm %ub_in_cmg23_5, %arg1, %c128_i64_cmg23_5\n            nburst(%c32_'
    'i64_cmg23_5, %c128_i64_cmg23_5, %c128_i64_cmg23_5)\n            : !pto.ptr<f32, ub>, !pto.ptr<f32, gm>, i64,'
    ' i64, i64, i64\n          pto.barrier #pto.pipe<PIPE_ALL>\n    }\n\n    // inactive cross-case merge from mi'
    'cro-op/vector-load-store/vstar\n    scf.if %__case_merge_guard {\n      \n          %c0_cmg23_6 = arith.cons'
    'tant 0 : index\n          %c1_cmg23_6 = arith.constant 1 : index\n          %c0_i64_cmg23_6 = arith.constant'
    ' 0 : i64\n          %c1_i64_cmg23_6 = arith.constant 1 : i64\n          %c8_i32_cmg23_6 = arith.constant 8 :'
    ' i32\n          %c32_i64_cmg23_6 = arith.constant 32 : i64\n          %c128_i64_cmg23_6 = arith.constant 128'
    ' : i64\n          %c4096_i64_cmg23_6 = arith.constant 4096 : i64\n          %c1_elem_cmg23_6 = arith.constan'
    't 1 : index\n      \n          %ub_in_cmg23_6 = pto.castptr %c0_i64_cmg23_6 : i64 -> !pto.ptr<f32, ub>\n    '
    '      %ub_out_cmg23_6 = pto.castptr %c4096_i64_cmg23_6 : i64 -> !pto.ptr<f32, ub>\n          %ub_out1_cmg23_'
    '6 = pto.addptr %ub_out_cmg23_6, %c1_elem_cmg23_6 : !pto.ptr<f32, ub>, index -> !pto.ptr<f32, ub>\n      \n         '
    ' %false_cmg23_6 = arith.constant false\n          pto.mte_gm_ub %arg0, %ub_in_cmg23_6, %c0_i64_cmg23_6, %c12'
    '8_i64_cmg23_6\n            nburst(%c32_i64_cmg23_6, %c128_i64_cmg23_6, %c128_i64_cmg23_6)\n            : !pt'
    'o.ptr<f32, gm>, !pto.ptr<f32, ub>, i64, i64, i64, i64, i64\n      \n          pto.set_flag["PIPE_MTE2", "PIP'
    'E_V", "EVENT_ID0"]\n          pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n      \n          pto.vecsc'
    'ope {\n            pto.sprclr "AR"\n            scf.for %iter_cmg23_6 = %c0_cmg23_6 to %c1_cmg23_6 step %c1_'
    'cmg23_6 {\n              %vec_cmg23_6 = pto.vlds %ub_in_cmg23_6[%c0_cmg23_6] : !pto.ptr<f32, ub>, index -> !pto.vre'
    'g<64xf32>\n              %mask_cmg23_6, %unused_cmg23_6 = pto.plt_b32 %c8_i32_cmg23_6 : i32 -> !pto.mask<b32'
    '>, i32\n              %sqz_cmg23_6 = pto.vsqz %vec_cmg23_6, %mask_cmg23_6 : !pto.vreg<64xf32>, !pto.mask<b32'
    '> -> !pto.vreg<64xf32>\n              %align0_cmg23_6 = pto.init_align : !pto.align\n              %align1_c'
    'mg23_6 = pto.vstur %align0_cmg23_6, %sqz_cmg23_6, %ub_out1_cmg23_6, "POST_UPDATE"\n                  : !pto.'
    'align, !pto.vreg<64xf32>, !pto.ptr<f32, ub> -> !pto.align\n              pto.vstar %align1_cmg23_6, %ub_out1'
    '_cmg23_6 : !pto.align, !pto.ptr<f32, ub>\n            }\n          }\n      \n          pto.set_flag["PIPE_V'
    '", "PIPE_MTE3", "EVENT_ID0"]\n          pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n          pto.mte'
    '_ub_gm %ub_out_cmg23_6, %arg1, %c128_i64_cmg23_6\n            nburst(%c32_i64_cmg23_6, %c128_i64_cmg23_6, %c'
    '128_i64_cmg23_6)\n            : !pto.ptr<f32, ub>, !pto.ptr<f32, gm>, i64, i64, i64, i64\n          pto.barr'
    'ier #pto.pipe<PIPE_ALL>\n    }\n\n    // inactive cross-case merge from micro-op/vector-load-store/vstas-vst'
    'us-offset-update\n    scf.if %__case_merge_guard {\n      \n          %c0_cmg23_7 = arith.constant 0 : index'
    '\n          %c1_cmg23_7 = arith.constant 1 : index\n          %c64_cmg23_7 = arith.constant 64 : index\n    '
    '      %c0_i64_cmg23_7 = arith.constant 0 : i64\n          %c1_i64_cmg23_7 = arith.constant 1 : i64\n        '
    '  %c32_i64_cmg23_7 = arith.constant 32 : i64\n          %c128_i64_cmg23_7 = arith.constant 128 : i64\n      '
    '    %c4096_i64_cmg23_7 = arith.constant 4096 : i64\n          %c3_i32_cmg23_7 = arith.constant 3 : i32\n    '
    '      %c0_i32_cmg23_7 = arith.constant 0 : i32\n      \n          %ub_in_cmg23_7 = pto.castptr %c0_i64_cmg23'
    '_7 : i64 -> !pto.ptr<f32, ub>\n          %ub_out_cmg23_7 = pto.castptr %c4096_i64_cmg23_7 : i64 -> !pto.ptr<'
    'f32, ub>\n      \n          %false_cmg23_7 = arith.constant false\n          pto.mte_gm_ub %arg0, %ub_in_cmg'
    '23_7, %c0_i64_cmg23_7, %c128_i64_cmg23_7\n            nburst(%c32_i64_cmg23_7, %c128_i64_cmg23_7, %c128_i64_'
    'cmg23_7)\n            : !pto.ptr<f32, gm>, !pto.ptr<f32, ub>, i64, i64, i64, i64, i64\n          pto.mte_gm_'
    'ub %arg1, %ub_out_cmg23_7, %c0_i64_cmg23_7, %c128_i64_cmg23_7\n            nburst(%c32_i64_cmg23_7, %c128_i6'
    '4_cmg23_7, %c128_i64_cmg23_7)\n            : !pto.ptr<f32, gm>, !pto.ptr<f32, ub>, i64, i64, i64, i64, i64\n'
    '      \n          pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n          pto.wait_flag["PIPE_MTE2", "PI'
    'PE_V", "EVENT_ID0"]\n      \n          pto.vecscope {\n            scf.for %offset_cmg23_7 = %c0_cmg23_7 to '
    '%c64_cmg23_7 step %c64_cmg23_7 {\n              %align0_cmg23_7 = pto.init_align : !pto.align\n             '
    ' %vec_cmg23_7 = pto.vlds %ub_in_cmg23_7[%c0_cmg23_7] : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>\n             '
    ' %align1_cmg23_7 = pto.vstus %align0_cmg23_7, %c3_i32_cmg23_7, %vec_cmg23_7, %ub_out_cmg23_7\n              '
    '    : !pto.align, i32, !pto.vreg<64xf32>, !pto.ptr<f32, ub> -> !pto.align\n              pto.vstas %align1_c'
    'mg23_7, %ub_out_cmg23_7, %c3_i32_cmg23_7 : !pto.align, !pto.ptr<f32, ub>, i32\n            }\n          }\n '
    '     \n          pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n          pto.wait_flag["PIPE_V", "PIPE_M'
    'TE3", "EVENT_ID0"]\n          pto.mte_ub_gm %ub_out_cmg23_7, %arg1, %c128_i64_cmg23_7\n            nburst(%c'
    '32_i64_cmg23_7, %c128_i64_cmg23_7, %c128_i64_cmg23_7)\n            : !pto.ptr<f32, ub>, !pto.ptr<f32, gm>, i'
    '64, i64, i64, i64\n          pto.barrier #pto.pipe<PIPE_ALL>\n    }\n\n    // inactive cross-case merge from'
    ' micro-op/vector-load-store/vsts-pk-b64-f32\n    scf.if %__case_merge_guard {\n      \n          %c0_cmg23_8'
    ' = arith.constant 0 : index\n          %c64_cmg23_8 = arith.constant 64 : index\n          %c1024_cmg23_8 = '
    'arith.constant 1024 : index\n          %c0_i64_cmg23_8 = arith.constant 0 : i64\n          %c1_i64_cmg23_8 ='
    ' arith.constant 1 : i64\n          %c32_i64_cmg23_8 = arith.constant 32 : i64\n          %c128_i64_cmg23_8 ='
    ' arith.constant 128 : i64\n          %c4096_i64_cmg23_8 = arith.constant 4096 : i64\n          %c1024_i32_cm'
    'g23_8 = arith.constant 1024 : i32\n      \n          %ub_in_cmg23_8 = pto.castptr %c0_i64_cmg23_8 : i64 -> !'
    'pto.ptr<f32, ub>\n          %ub_out_cmg23_8 = pto.castptr %c4096_i64_cmg23_8 : i64 -> !pto.ptr<f32, ub>\n   '
    '   \n          %false_cmg23_8 = arith.constant false\n          pto.mte_gm_ub %arg0, %ub_in_cmg23_8, %c0_i64'
    '_cmg23_8, %c128_i64_cmg23_8\n            nburst(%c32_i64_cmg23_8, %c128_i64_cmg23_8, %c128_i64_cmg23_8)\n   '
    '         : !pto.ptr<f32, gm>, !pto.ptr<f32, ub>, i64, i64, i64, i64, i64\n          pto.mte_gm_ub %arg1, %ub'
    '_out_cmg23_8, %c0_i64_cmg23_8, %c128_i64_cmg23_8\n            nburst(%c32_i64_cmg23_8, %c128_i64_cmg23_8, %c'
    '128_i64_cmg23_8)\n            : !pto.ptr<f32, gm>, !pto.ptr<f32, ub>, i64, i64, i64, i64, i64\n      \n     '
    '     pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n          pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVENT'
    '_ID0"]\n      \n          pto.vecscope {\n            %__cmg23_8:1 = scf.for %offset_cmg23_8 = %c0_cmg23_8 t'
    'o %c1024_cmg23_8 step %c64_cmg23_8 iter_args(%remaining_cmg23_8 = %c1024_i32_cmg23_8) -> (i32) {\n          '
    '    %mask_cmg23_8, %next_remaining_cmg23_8 = pto.plt_b32 %remaining_cmg23_8 : i32 -> !pto.mask<b32>, i32\n  '
    '            %vec_cmg23_8 = pto.vlds %ub_in_cmg23_8[%offset_cmg23_8] : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>'
    '\n              pto.vsts %vec_cmg23_8, %ub_out_cmg23_8[%offset_cmg23_8], %mask_cmg23_8 {dist = "PK_B64"} : !'
    'pto.vreg<64xf32>, !pto.ptr<f32, ub>, index, !pto.mask<b32>\n              scf.yield %next_remaining_cmg23_8 : i32\n'
    '            }\n          }\n      \n          pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n          pt'
    'o.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n          pto.mte_ub_gm %ub_out_cmg23_8, %arg1, %c128_i64_c'
    'mg23_8\n            nburst(%c32_i64_cmg23_8, %c128_i64_cmg23_8, %c128_i64_cmg23_8)\n            : !pto.ptr<f'
    '32, ub>, !pto.ptr<f32, gm>, i64, i64, i64, i64\n          pto.barrier #pto.pipe<PIPE_ALL>\n    }\n\n    // i'
    'nactive cross-case merge from micro-op/vector-load-store/vstsx2-layout-check\n    scf.if %__case_merge_guard'
    ' {\n      \n          %c0_cmg23_9 = arith.constant 0 : index\n          %c64_cmg23_9 = arith.constant 64 : i'
    'ndex\n          %c1_i64_cmg23_9 = arith.constant 1 : i64\n          %c0_i64_cmg23_9 = arith.constant 0 : i64'
    '\n          %c32_i64_cmg23_9 = arith.constant 32 : i64\n          %c128_i64_cmg23_9 = arith.constant 128 : i'
    '64\n          %c4096_i64_cmg23_9 = arith.constant 4096 : i64\n          %c64_i32_cmg23_9 = arith.constant 64'
    ' : i32\n          %false_cmg23_9 = arith.constant false\n      \n          %ub_in_cmg23_9 = pto.castptr %c0_'
    'i64_cmg23_9 : i64 -> !pto.ptr<f32, ub>\n          %ub_out_cmg23_9 = pto.castptr %c4096_i64_cmg23_9 : i64 -> '
    '!pto.ptr<f32, ub>\n          pto.mte_gm_ub %arg0, %ub_in_cmg23_9, %c0_i64_cmg23_9, %c128_i64_cmg23_9\n      '
    '      nburst(%c32_i64_cmg23_9, %c128_i64_cmg23_9, %c128_i64_cmg23_9)\n            : !pto.ptr<f32, gm>, !pto.'
    'ptr<f32, ub>, i64, i64, i64, i64, i64\n      \n          pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n '
    '         pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n      \n          %c1_cmg23_9 = arith.constant 1'
    ' : index\n          pto.vecscope {\n            scf.for %iv_cmg23_9 = %c0_cmg23_9 to %c1_cmg23_9 step %c1_cm'
    'g23_9 {\n              %mask_cmg23_9, %remaining_cmg23_9 = pto.plt_b32 %c64_i32_cmg23_9 : i32 -> !pto.mask<b'
    '32>, i32\n              %x_cmg23_9 = pto.vlds %ub_in_cmg23_9[%c0_cmg23_9] : !pto.ptr<f32, ub>, index -> !pto.vreg<6'
    '4xf32>\n              %y_cmg23_9 = pto.vlds %ub_in_cmg23_9[%c64_cmg23_9] : !pto.ptr<f32, ub>, index -> !pto.vreg<64'
    'xf32>\n              pto.vstsx2 %x_cmg23_9, %y_cmg23_9, %ub_out_cmg23_9[%c0_cmg23_9], "INTLV_B32", %mask_cmg'
    '23_9\n                  : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.ptr<f32, ub>, index,\n                 '
    '   !pto.mask<b32>\n            }\n          }\n      \n          pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_'
    'ID0"]\n          pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n          pto.mte_ub_gm %ub_out_cmg23_9,'
    ' %arg1, %c128_i64_cmg23_9\n            nburst(%c32_i64_cmg23_9, %c128_i64_cmg23_9, %c128_i64_cmg23_9)\n     '
    '       : !pto.ptr<f32, ub>, !pto.ptr<f32, gm>, i64, i64, i64, i64\n          pto.barrier #pto.pipe<PIPE_ALL>'
    '\n    }\n    return\n  }\n}\n'
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

    # case: micro-op/vector-load-store/vlds-brc-b32
    # family: vector-load-store
    # target_ops: pto.vlds
    # scenarios: core-f32, full-mask, aligned, dist-brc-b32
    # NOTE: BRC on b32 broadcasts the first f32 element of each 64-lane chunk.
    # coding=utf-8

    import argparse
    from pathlib import Path

    import numpy as np


    ELEMENTS = 1024
    LANES = 64
    SEED = 19


    def _generate(seed: int) -> dict:
        rng = np.random.default_rng(seed)
        v1 = rng.uniform(-8.0, 8.0, size=(ELEMENTS,)).astype(np.float32)
        v2 = np.zeros((ELEMENTS,), dtype=np.float32)
        golden_v2 = np.empty((ELEMENTS,), dtype=np.float32)
        for offset in range(0, ELEMENTS, LANES):
            golden_v2[offset : offset + LANES] = v1[offset]

        _files["v1.bin"] = v1
        _files["v2.bin"] = v2
        _files["golden_v2.bin"] = golden_v2


    def main() -> None:
        parser = argparse.ArgumentParser(
            description="Generate numpy-based inputs/golden for VPTO micro-op vlds b32 broadcast validation."
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

    # case: micro-op/vector-load-store/vlds-brc-b32
    # family: vector-load-store
    # target_ops: pto.vlds
    # scenarios: core-f32, full-mask, aligned, dist-brc-b32
    # NOTE: bulk-generated coverage skeleton.
    # coding=utf-8

    import os
    import sys
    import numpy as np


    REPEAT_BYTES = 256


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
        ok = compare_bin("golden_v2.bin", "v2.bin", np.float32, 0.0001) and ok
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
    )
]
SCALARS = []
SCALAR_SOURCES = []
SCALAR_DTYPES = []


@pto.jit(
    name='vabs_kernel_2d',
    target="a5",
    backend="vpto",
    mode="explicit",
    source=KERNEL_SOURCE,
)
def kernel(
    arg0: pto.ptr(pto.f32, "gm"),
    arg1: pto.ptr(pto.f32, "gm"),
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
    name='legacy_micro_op_vector_load_store_vlds_brc_b32',
    grid=1,
    kernel=kernel,
    make_case=make_case,
    check=check_case,
)]

auto_main(globals())
