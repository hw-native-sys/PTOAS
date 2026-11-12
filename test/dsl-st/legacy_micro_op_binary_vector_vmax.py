#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Migrated from test/vpto/cases/micro-op/binary-vector/vmax."""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

from common import auto_main
from ptodsl import pto

KERNEL_SOURCE = (
    'module attributes {pto.target_arch = "a5", pto.kernel_kind = #pto.kernel_kind<vector>} {\n    func.func @vma'
    'x_deep_merged_kernel(%arg0: !pto.ptr<f32, gm>, %arg1: !pto.ptr<f32, gm>, %arg2: !pto.ptr<f32, gm>, %arg3: !p'
    'to.ptr<f32, gm>, %arg4: !pto.ptr<f32, gm>, %arg5: !pto.ptr<f32, gm>) attributes {pto.kernel} {\n    // merge'
    'd from max_kernel_2d via LaunchMax_kernel_2d\n\n    %c0_m0 = arith.constant 0 : index\n    %c1_m0 = arith.co'
    'nstant 1 : index\n    %c64_m0 = arith.constant 64 : index\n    %c1024_m0 = arith.constant 1024 : index\n    '
    '%c0_i64_m0 = arith.constant 0 : i64\n    %c1_i64_m0 = arith.constant 1 : i64\n    %c32_i64_m0 = arith.consta'
    'nt 32 : i64\n    %c128_i64_m0 = arith.constant 128 : i64\n    %c4096_i64_m0 = arith.constant 4096 : i64\n   '
    ' %c8192_i64_m0 = arith.constant 8192 : i64\n    %c1024_i32_m0 = arith.constant 1024 : i32\n\n    %ub_lhs_m0 '
    '= pto.castptr %c0_i64_m0 : i64 -> !pto.ptr<f32, ub>\n    %ub_rhs_m0 = pto.castptr %c4096_i64_m0 : i64 -> !pt'
    'o.ptr<f32, ub>\n    %ub_out_m0 = pto.castptr %c8192_i64_m0 : i64 -> !pto.ptr<f32, ub>\n\n    %false_m0 = ari'
    'th.constant false\n    pto.mte_gm_ub %arg0, %ub_lhs_m0, %c0_i64_m0, %c128_i64_m0\n      nburst(%c32_i64_m0, '
    '%c128_i64_m0, %c128_i64_m0)\n      : !pto.ptr<f32, gm>, !pto.ptr<f32, ub>, i64, i64, i64, i64, i64\n    pto.'
    'mte_gm_ub %arg1, %ub_rhs_m0, %c0_i64_m0, %c128_i64_m0\n      nburst(%c32_i64_m0, %c128_i64_m0, %c128_i64_m0)'
    '\n      : !pto.ptr<f32, gm>, !pto.ptr<f32, ub>, i64, i64, i64, i64, i64\n\n    pto.set_flag["PIPE_MTE2", "PI'
    'PE_V", "EVENT_ID0"]\n    pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n\n    pto.vecscope {\n      %__m'
    '0:1 = scf.for %offset_m0 = %c0_m0 to %c1024_m0 step %c64_m0 iter_args(%remaining_m0 = %c1024_i32_m0) -> (i32'
    ') {\n        %mask_m0, %next_remaining_m0 = pto.plt_b32 %remaining_m0 : i32 -> !pto.mask<b32>, i32\n        '
    '%lhs_m0 = pto.vlds %ub_lhs_m0[%offset_m0] : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>\n        %rhs_m0 = pto.vl'
    'ds %ub_rhs_m0[%offset_m0] : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>\n        %maxv_m0 = pto.vmax %lhs_m0, %rh'
    's_m0, %mask_m0 : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>\n        pto.vsts'
    ' %maxv_m0, %ub_out_m0[%offset_m0], %mask_m0 : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, index, !pto.mask<b32>\n        '
    'scf.yield %next_remaining_m0 : i32\n      }\n    }\n\n    pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n'
    '    pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n    pto.mte_ub_gm %ub_out_m0, %arg2, %c128_i64_m0\n  '
    '    nburst(%c32_i64_m0, %c128_i64_m0, %c128_i64_m0)\n      : !pto.ptr<f32, ub>, !pto.ptr<f32, gm>, i64, i64,'
    ' i64, i64\n    pto.barrier #pto.pipe<PIPE_ALL>\n\n    // merged from vmax_tail_kernel_2d via LaunchVmax_tail'
    '_kernel_2d\n\n    %c0_m1 = arith.constant 0 : index\n    %c64_m1 = arith.constant 64 : index\n    %c1024_m1 '
    '= arith.constant 1024 : index\n    %c0_i64_m1 = arith.constant 0 : i64\n    %c1_i64_m1 = arith.constant 1 : '
    'i64\n    %c32_i64_m1 = arith.constant 32 : i64\n    %c128_i64_m1 = arith.constant 128 : i64\n    %c4096_i64_'
    'm1 = arith.constant 4096 : i64\n    %c8192_i64_m1 = arith.constant 8192 : i64\n    %c1000_i32_m1 = arith.con'
    'stant 1000 : i32\n\n    %ub_lhs_m1 = pto.castptr %c0_i64_m1 : i64 -> !pto.ptr<f32, ub>\n    %ub_rhs_m1 = pto'
    '.castptr %c4096_i64_m1 : i64 -> !pto.ptr<f32, ub>\n    %ub_out_m1 = pto.castptr %c8192_i64_m1 : i64 -> !pto.'
    'ptr<f32, ub>\n\n    %false_m1 = arith.constant false\n    pto.mte_gm_ub %arg3, %ub_lhs_m1, %c0_i64_m1, %c128'
    '_i64_m1\n      nburst(%c32_i64_m1, %c128_i64_m1, %c128_i64_m1)\n      : !pto.ptr<f32, gm>, !pto.ptr<f32, ub>'
    ', i64, i64, i64, i64, i64\n    pto.mte_gm_ub %arg4, %ub_rhs_m1, %c0_i64_m1, %c128_i64_m1\n      nburst(%c32_'
    'i64_m1, %c128_i64_m1, %c128_i64_m1)\n      : !pto.ptr<f32, gm>, !pto.ptr<f32, ub>, i64, i64, i64, i64, i64\n'
    '\n    pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n    pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"'
    ']\n\n    pto.vecscope {\n      %__m1:1 = scf.for %offset_m1 = %c0_m1 to %c1024_m1 step %c64_m1 iter_args(%re'
    'maining_m1 = %c1000_i32_m1) -> (i32) {\n        %mask_m1, %next_remaining_m1 = pto.plt_b32 %remaining_m1 : i'
    '32 -> !pto.mask<b32>, i32\n        %lhs_m1 = pto.vlds %ub_lhs_m1[%offset_m1] : !pto.ptr<f32, ub>, index -> !pto.vre'
    'g<64xf32>\n        %rhs_m1 = pto.vlds %ub_rhs_m1[%offset_m1] : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>\n     '
    '   %maxv_m1 = pto.vmax %lhs_m1, %rhs_m1, %mask_m1 : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> '
    '!pto.vreg<64xf32>\n        pto.vsts %maxv_m1, %ub_out_m1[%offset_m1], %mask_m1 : !pto.vreg<64xf32>, !pto.ptr'
    '<f32, ub>, index, !pto.mask<b32>\n        scf.yield %next_remaining_m1 : i32\n      }\n    }\n\n    pto.set_flag["P'
    'IPE_V", "PIPE_MTE3", "EVENT_ID0"]\n    pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n    pto.mte_ub_gm '
    '%ub_out_m1, %arg5, %c128_i64_m1\n      nburst(%c32_i64_m1, %c128_i64_m1, %c128_i64_m1)\n      : !pto.ptr<f32'
    ', ub>, !pto.ptr<f32, gm>, i64, i64, i64, i64\n    pto.barrier #pto.pipe<PIPE_ALL>\n    %__case_merge_guard ='
    ' arith.constant false\n\n    // inactive cross-case merge from micro-op/binary-vector/vmul\n    scf.if %__ca'
    'se_merge_guard {\n      \n          // merged from mul_kernel_2d via LaunchMul_kernel_2d\n      \n          '
    '%c0_m0_cmg2_1 = arith.constant 0 : index\n          %c1_m0_cmg2_1 = arith.constant 1 : index\n          %c64'
    '_m0_cmg2_1 = arith.constant 64 : index\n          %c1024_m0_cmg2_1 = arith.constant 1024 : index\n          '
    '%c0_i64_m0_cmg2_1 = arith.constant 0 : i64\n          %c1_i64_m0_cmg2_1 = arith.constant 1 : i64\n          '
    '%c32_i64_m0_cmg2_1 = arith.constant 32 : i64\n          %c128_i64_m0_cmg2_1 = arith.constant 128 : i64\n    '
    '      %c4096_i64_m0_cmg2_1 = arith.constant 4096 : i64\n          %c8192_i64_m0_cmg2_1 = arith.constant 8192'
    ' : i64\n          %c1024_i32_m0_cmg2_1 = arith.constant 1024 : i32\n      \n          %ub_lhs_m0_cmg2_1 = pt'
    'o.castptr %c0_i64_m0_cmg2_1 : i64 -> !pto.ptr<f32, ub>\n          %ub_rhs_m0_cmg2_1 = pto.castptr %c4096_i64'
    '_m0_cmg2_1 : i64 -> !pto.ptr<f32, ub>\n          %ub_out_m0_cmg2_1 = pto.castptr %c8192_i64_m0_cmg2_1 : i64 '
    '-> !pto.ptr<f32, ub>\n      \n          %false_m0_cmg2_1 = arith.constant false\n          pto.mte_gm_ub %ar'
    'g0, %ub_lhs_m0_cmg2_1, %c0_i64_m0_cmg2_1, %c128_i64_m0_cmg2_1\n            nburst(%c32_i64_m0_cmg2_1, %c128_'
    'i64_m0_cmg2_1, %c128_i64_m0_cmg2_1)\n            : !pto.ptr<f32, gm>, !pto.ptr<f32, ub>, i64, i64, i64, i64,'
    ' i64\n          pto.mte_gm_ub %arg1, %ub_rhs_m0_cmg2_1, %c0_i64_m0_cmg2_1, %c128_i64_m0_cmg2_1\n            '
    'nburst(%c32_i64_m0_cmg2_1, %c128_i64_m0_cmg2_1, %c128_i64_m0_cmg2_1)\n            : !pto.ptr<f32, gm>, !pto.'
    'ptr<f32, ub>, i64, i64, i64, i64, i64\n      \n          pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n '
    '         pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n      \n          pto.vecscope {\n            %_'
    '_m0_cmg2_1:1 = scf.for %offset_m0_cmg2_1 = %c0_m0_cmg2_1 to %c1024_m0_cmg2_1 step %c64_m0_cmg2_1 iter_args(%'
    'remaining_m0_cmg2_1 = %c1024_i32_m0_cmg2_1) -> (i32) {\n              %mask_m0_cmg2_1, %next_remaining_m0_cm'
    'g2_1 = pto.plt_b32 %remaining_m0_cmg2_1 : i32 -> !pto.mask<b32>, i32\n              %lhs_m0_cmg2_1 = pto.vld'
    's %ub_lhs_m0_cmg2_1[%offset_m0_cmg2_1] : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>\n              %rhs_m0_cmg2_'
    '1 = pto.vlds %ub_rhs_m0_cmg2_1[%offset_m0_cmg2_1] : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>\n              %p'
    'rod_m0_cmg2_1 = pto.vmul %lhs_m0_cmg2_1, %rhs_m0_cmg2_1, %mask_m0_cmg2_1 : !pto.vreg<64xf32>, !pto.vreg<64xf'
    '32>, !pto.mask<b32> -> !pto.vreg<64xf32>\n              pto.vsts %prod_m0_cmg2_1, %ub_out_m0_cmg2_1[%offset_'
    'm0_cmg2_1], %mask_m0_cmg2_1 : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, index, !pto.mask<b32>\n              scf.yield '
    '%next_remaining_m0_cmg2_1 : i32\n            }\n          }\n      \n          pto.set_flag["PIPE_V", "PIPE_'
    'MTE3", "EVENT_ID0"]\n          pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n          pto.mte_ub_gm %u'
    'b_out_m0_cmg2_1, %arg2, %c128_i64_m0_cmg2_1\n            nburst(%c32_i64_m0_cmg2_1, %c128_i64_m0_cmg2_1, %c1'
    '28_i64_m0_cmg2_1)\n            : !pto.ptr<f32, ub>, !pto.ptr<f32, gm>, i64, i64, i64, i64\n          pto.bar'
    'rier #pto.pipe<PIPE_ALL>\n      \n          // merged from vmul_tail_kernel_2d via LaunchVmul_tail_kernel_2d'
    '\n      \n          %c0_m1_cmg2_1 = arith.constant 0 : index\n          %c64_m1_cmg2_1 = arith.constant 64 :'
    ' index\n          %c1024_m1_cmg2_1 = arith.constant 1024 : index\n          %c0_i64_m1_cmg2_1 = arith.consta'
    'nt 0 : i64\n          %c1_i64_m1_cmg2_1 = arith.constant 1 : i64\n          %c32_i64_m1_cmg2_1 = arith.const'
    'ant 32 : i64\n          %c128_i64_m1_cmg2_1 = arith.constant 128 : i64\n          %c4096_i64_m1_cmg2_1 = ari'
    'th.constant 4096 : i64\n          %c8192_i64_m1_cmg2_1 = arith.constant 8192 : i64\n          %c1000_i32_m1_'
    'cmg2_1 = arith.constant 1000 : i32\n      \n          %ub_lhs_m1_cmg2_1 = pto.castptr %c0_i64_m1_cmg2_1 : i6'
    '4 -> !pto.ptr<f32, ub>\n          %ub_rhs_m1_cmg2_1 = pto.castptr %c4096_i64_m1_cmg2_1 : i64 -> !pto.ptr<f32'
    ', ub>\n          %ub_out_m1_cmg2_1 = pto.castptr %c8192_i64_m1_cmg2_1 : i64 -> !pto.ptr<f32, ub>\n      \n  '
    '        %false_m1_cmg2_1 = arith.constant false\n          pto.mte_gm_ub %arg3, %ub_lhs_m1_cmg2_1, %c0_i64_m'
    '1_cmg2_1, %c128_i64_m1_cmg2_1\n            nburst(%c32_i64_m1_cmg2_1, %c128_i64_m1_cmg2_1, %c128_i64_m1_cmg2'
    '_1)\n            : !pto.ptr<f32, gm>, !pto.ptr<f32, ub>, i64, i64, i64, i64, i64\n          pto.mte_gm_ub %a'
    'rg4, %ub_rhs_m1_cmg2_1, %c0_i64_m1_cmg2_1, %c128_i64_m1_cmg2_1\n            nburst(%c32_i64_m1_cmg2_1, %c128'
    '_i64_m1_cmg2_1, %c128_i64_m1_cmg2_1)\n            : !pto.ptr<f32, gm>, !pto.ptr<f32, ub>, i64, i64, i64, i64'
    ', i64\n      \n          pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n          pto.wait_flag["PIPE_MTE'
    '2", "PIPE_V", "EVENT_ID0"]\n      \n          pto.vecscope {\n            %__m1_cmg2_1:1 = scf.for %offset_m'
    '1_cmg2_1 = %c0_m1_cmg2_1 to %c1024_m1_cmg2_1 step %c64_m1_cmg2_1 iter_args(%remaining_m1_cmg2_1 = %c1000_i32'
    '_m1_cmg2_1) -> (i32) {\n              %mask_m1_cmg2_1, %next_remaining_m1_cmg2_1 = pto.plt_b32 %remaining_m1'
    '_cmg2_1 : i32 -> !pto.mask<b32>, i32\n              %lhs_m1_cmg2_1 = pto.vlds %ub_lhs_m1_cmg2_1[%offset_m1_c'
    'mg2_1] : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>\n              %rhs_m1_cmg2_1 = pto.vlds %ub_rhs_m1_cmg2_1[%'
    'offset_m1_cmg2_1] : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>\n              %prod_m1_cmg2_1 = pto.vmul %lhs_m1'
    '_cmg2_1, %rhs_m1_cmg2_1, %mask_m1_cmg2_1 : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg'
    '<64xf32>\n              pto.vsts %prod_m1_cmg2_1, %ub_out_m1_cmg2_1[%offset_m1_cmg2_1], %mask_m1_cmg2_1 : !p'
    'to.vreg<64xf32>, !pto.ptr<f32, ub>, index, !pto.mask<b32>\n              scf.yield %next_remaining_m1_cmg2_1 : i32'
    '\n            }\n          }\n      \n          pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n          '
    'pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n          pto.mte_ub_gm %ub_out_m1_cmg2_1, %arg5, %c128_i'
    '64_m1_cmg2_1\n            nburst(%c32_i64_m1_cmg2_1, %c128_i64_m1_cmg2_1, %c128_i64_m1_cmg2_1)\n            '
    ': !pto.ptr<f32, ub>, !pto.ptr<f32, gm>, i64, i64, i64, i64\n          pto.barrier #pto.pipe<PIPE_ALL>\n    }'
    '\n\n    // inactive cross-case merge from micro-op/binary-vector/vsub\n    scf.if %__case_merge_guard {\n   '
    '   \n          // merged from sub_kernel_2d via LaunchSub_kernel_2d\n      \n          %c0_m0_cmg2_2 = arith'
    '.constant 0 : index\n          %c1_m0_cmg2_2 = arith.constant 1 : index\n          %c64_m0_cmg2_2 = arith.co'
    'nstant 64 : index\n          %c1024_m0_cmg2_2 = arith.constant 1024 : index\n          %c0_i64_m0_cmg2_2 = a'
    'rith.constant 0 : i64\n          %c1_i64_m0_cmg2_2 = arith.constant 1 : i64\n          %c32_i64_m0_cmg2_2 = '
    'arith.constant 32 : i64\n          %c128_i64_m0_cmg2_2 = arith.constant 128 : i64\n          %c4096_i64_m0_c'
    'mg2_2 = arith.constant 4096 : i64\n          %c8192_i64_m0_cmg2_2 = arith.constant 8192 : i64\n          %c1'
    '024_i32_m0_cmg2_2 = arith.constant 1024 : i32\n      \n          %ub_lhs_m0_cmg2_2 = pto.castptr %c0_i64_m0_'
    'cmg2_2 : i64 -> !pto.ptr<f32, ub>\n          %ub_rhs_m0_cmg2_2 = pto.castptr %c4096_i64_m0_cmg2_2 : i64 -> !'
    'pto.ptr<f32, ub>\n          %ub_out_m0_cmg2_2 = pto.castptr %c8192_i64_m0_cmg2_2 : i64 -> !pto.ptr<f32, ub>'
    '\n      \n          %false_m0_cmg2_2 = arith.constant false\n          pto.mte_gm_ub %arg0, %ub_lhs_m0_cmg2_'
    '2, %c0_i64_m0_cmg2_2, %c128_i64_m0_cmg2_2\n            nburst(%c32_i64_m0_cmg2_2, %c128_i64_m0_cmg2_2, %c128'
    '_i64_m0_cmg2_2)\n            : !pto.ptr<f32, gm>, !pto.ptr<f32, ub>, i64, i64, i64, i64, i64\n          pto.'
    'mte_gm_ub %arg1, %ub_rhs_m0_cmg2_2, %c0_i64_m0_cmg2_2, %c128_i64_m0_cmg2_2\n            nburst(%c32_i64_m0_c'
    'mg2_2, %c128_i64_m0_cmg2_2, %c128_i64_m0_cmg2_2)\n            : !pto.ptr<f32, gm>, !pto.ptr<f32, ub>, i64, i'
    '64, i64, i64, i64\n      \n          pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n          pto.wait_fl'
    'ag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n      \n          pto.vecscope {\n            %__m0_cmg2_2:1 = scf.f'
    'or %offset_m0_cmg2_2 = %c0_m0_cmg2_2 to %c1024_m0_cmg2_2 step %c64_m0_cmg2_2 iter_args(%remaining_m0_cmg2_2 '
    '= %c1024_i32_m0_cmg2_2) -> (i32) {\n              %mask_m0_cmg2_2, %next_remaining_m0_cmg2_2 = pto.plt_b32 %'
    'remaining_m0_cmg2_2 : i32 -> !pto.mask<b32>, i32\n              %lhs_m0_cmg2_2 = pto.vlds %ub_lhs_m0_cmg2_2['
    '%offset_m0_cmg2_2] : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>\n              %rhs_m0_cmg2_2 = pto.vlds %ub_rhs'
    '_m0_cmg2_2[%offset_m0_cmg2_2] : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>\n              %diff_m0_cmg2_2 = pto.'
    'vsub %lhs_m0_cmg2_2, %rhs_m0_cmg2_2, %mask_m0_cmg2_2 : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> '
    '-> !pto.vreg<64xf32>\n              pto.vsts %diff_m0_cmg2_2, %ub_out_m0_cmg2_2[%offset_m0_cmg2_2], %mask_m0'
    '_cmg2_2 : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, index, !pto.mask<b32>\n              scf.yield %next_remaining_m0_c'
    'mg2_2 : i32\n            }\n          }\n      \n          pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]'
    '\n          pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n          pto.mte_ub_gm %ub_out_m0_cmg2_2, %a'
    'rg2, %c128_i64_m0_cmg2_2\n            nburst(%c32_i64_m0_cmg2_2, %c128_i64_m0_cmg2_2, %c128_i64_m0_cmg2_2)\n'
    '            : !pto.ptr<f32, ub>, !pto.ptr<f32, gm>, i64, i64, i64, i64\n          pto.barrier #pto.pipe<PIPE'
    '_ALL>\n      \n          // merged from vsub_tail_kernel_2d via LaunchVsub_tail_kernel_2d\n      \n         '
    ' %c0_m1_cmg2_2 = arith.constant 0 : index\n          %c64_m1_cmg2_2 = arith.constant 64 : index\n          %'
    'c1024_m1_cmg2_2 = arith.constant 1024 : index\n          %c0_i64_m1_cmg2_2 = arith.constant 0 : i64\n       '
    '   %c1_i64_m1_cmg2_2 = arith.constant 1 : i64\n          %c32_i64_m1_cmg2_2 = arith.constant 32 : i64\n     '
    '     %c128_i64_m1_cmg2_2 = arith.constant 128 : i64\n          %c4096_i64_m1_cmg2_2 = arith.constant 4096 : '
    'i64\n          %c8192_i64_m1_cmg2_2 = arith.constant 8192 : i64\n          %c1000_i32_m1_cmg2_2 = arith.cons'
    'tant 1000 : i32\n      \n          %ub_lhs_m1_cmg2_2 = pto.castptr %c0_i64_m1_cmg2_2 : i64 -> !pto.ptr<f32, '
    'ub>\n          %ub_rhs_m1_cmg2_2 = pto.castptr %c4096_i64_m1_cmg2_2 : i64 -> !pto.ptr<f32, ub>\n          %u'
    'b_out_m1_cmg2_2 = pto.castptr %c8192_i64_m1_cmg2_2 : i64 -> !pto.ptr<f32, ub>\n      \n          %false_m1_c'
    'mg2_2 = arith.constant false\n          pto.mte_gm_ub %arg3, %ub_lhs_m1_cmg2_2, %c0_i64_m1_cmg2_2, %c128_i64'
    '_m1_cmg2_2\n            nburst(%c32_i64_m1_cmg2_2, %c128_i64_m1_cmg2_2, %c128_i64_m1_cmg2_2)\n            : '
    '!pto.ptr<f32, gm>, !pto.ptr<f32, ub>, i64, i64, i64, i64, i64\n          pto.mte_gm_ub %arg4, %ub_rhs_m1_cmg'
    '2_2, %c0_i64_m1_cmg2_2, %c128_i64_m1_cmg2_2\n            nburst(%c32_i64_m1_cmg2_2, %c128_i64_m1_cmg2_2, %c1'
    '28_i64_m1_cmg2_2)\n            : !pto.ptr<f32, gm>, !pto.ptr<f32, ub>, i64, i64, i64, i64, i64\n      \n    '
    '      pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n          pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVEN'
    'T_ID0"]\n      \n          pto.vecscope {\n            %__m1_cmg2_2:1 = scf.for %offset_m1_cmg2_2 = %c0_m1_c'
    'mg2_2 to %c1024_m1_cmg2_2 step %c64_m1_cmg2_2 iter_args(%remaining_m1_cmg2_2 = %c1000_i32_m1_cmg2_2) -> (i32'
    ') {\n              %mask_m1_cmg2_2, %next_remaining_m1_cmg2_2 = pto.plt_b32 %remaining_m1_cmg2_2 : i32 -> !p'
    'to.mask<b32>, i32\n              %lhs_m1_cmg2_2 = pto.vlds %ub_lhs_m1_cmg2_2[%offset_m1_cmg2_2] : !pto.ptr<f'
    '32, ub>, index -> !pto.vreg<64xf32>\n              %rhs_m1_cmg2_2 = pto.vlds %ub_rhs_m1_cmg2_2[%offset_m1_cmg2_2] :'
    ' !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>\n              %diff_m1_cmg2_2 = pto.vsub %lhs_m1_cmg2_2, %rhs_m1_cm'
    'g2_2, %mask_m1_cmg2_2 : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>\n         '
    '     pto.vsts %diff_m1_cmg2_2, %ub_out_m1_cmg2_2[%offset_m1_cmg2_2], %mask_m1_cmg2_2 : !pto.vreg<64xf32>, !p'
    'to.ptr<f32, ub>, index, !pto.mask<b32>\n              scf.yield %next_remaining_m1_cmg2_2 : i32\n            }\n   '
    '       }\n      \n          pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n          pto.wait_flag["PIPE_'
    'V", "PIPE_MTE3", "EVENT_ID0"]\n          pto.mte_ub_gm %ub_out_m1_cmg2_2, %arg5, %c128_i64_m1_cmg2_2\n      '
    '      nburst(%c32_i64_m1_cmg2_2, %c128_i64_m1_cmg2_2, %c128_i64_m1_cmg2_2)\n            : !pto.ptr<f32, ub>,'
    ' !pto.ptr<f32, gm>, i64, i64, i64, i64\n          pto.barrier #pto.pipe<PIPE_ALL>\n    }\n\n    return\n  }'
    '\n}\n'
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

    # Merged vmax test case.

    import argparse
    from pathlib import Path
    import numpy as np

    ROWS = 32
    COLS = 32
    SEED = 19
    LOGICAL_ELEMS = 1000
    OUT_SENTINEL = np.float32(-123.25)

    def f32_to_bf16_bits(v):
        w=v.astype(np.float32,copy=False).view(np.uint32)
        r=np.uint32(0x7FFF)+((w>>16)&np.uint32(1))
        return ((w+r)>>16).astype(np.uint16)
    def bf16_bits_to_f32(b):
        return (b.astype(np.uint32)<<16).view(np.float32)


    # ---- f32 ----
    def gen_f32(out, rng):
        v1=rng.uniform(-8.0,8.0,size=(ROWS,COLS)).astype(np.float32)
        v2=rng.uniform(-8.0,8.0,size=(ROWS,COLS)).astype(np.float32)
        g=np.maximum(v1,v2).astype(np.float32,copy=False)
        v3=np.zeros((ROWS,COLS),dtype=np.float32)
        _files["v1.bin"] = v1.reshape(-1)
        _files["v2.bin"] = v2.reshape(-1)
        _files["v3.bin"] = v3.reshape(-1)
        _files["golden_v3.bin"] = g.reshape(-1)

    # ---- tail ----
    def gen_tail(out, rng):
        v1=rng.random((ROWS,COLS),dtype=np.float32)
        v2=rng.random((ROWS,COLS),dtype=np.float32)
        v3=np.full((ROWS,COLS),OUT_SENTINEL,dtype=np.float32)
        g=np.full((ROWS,COLS),OUT_SENTINEL,dtype=np.float32)
        g.reshape(-1)[:LOGICAL_ELEMS]=np.maximum(v1.reshape(-1)[:LOGICAL_ELEMS],v2.reshape(-1)[:LOGICAL_ELEMS]).astype(
            np.float32,copy=False)
        _files["v1_tail.bin"] = v1.reshape(-1)
        _files["v2_tail.bin"] = v2.reshape(-1)
        _files["v3_tail.bin"] = v3.reshape(-1)
        _files["golden_v3_tail.bin"] = g.reshape(-1)

    GENERATORS = [
        gen_f32,
        gen_tail,
    ]

    def main():
        p=argparse.ArgumentParser()
        p.add_argument("--output-dir",type=Path,default=Path("."))
        p.add_argument("--seed",type=int,default=SEED)
        a=p.parse_args()
        rng=np.random.default_rng(a.seed)
        out=a.output_dir; out.mkdir(parents=True,exist_ok=True)
        for gen in GENERATORS:
            gen(out,rng)

    return locals()

def _compare_scope():
    # Copyright (c) 2026 Huawei Technologies Co., Ltd.
    # This program is free software, you can redistribute it and/or modify it under the terms and conditions of
    # CANN Open Software License Agreement Version 2.0 (the "License").
    # Please refer to the License for details. You may not use this file except in compliance with the License.
    # THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
    # INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
    # See LICENSE in the root of the software repository for the full text of the License.

    # Merged vmax test case.

    import os,sys
    import numpy as np

    def _cmp(golden,output,dtype,eps,count=-1):
        if not os.path.exists(golden) or not os.path.exists(output): return False
        kw={} if count<0 else {"count":count}
        g=np.fromfile(golden,dtype=dtype,**kw)
        o=np.fromfile(output,dtype=dtype,**kw)
        return g.shape==o.shape and np.allclose(g,o,atol=eps,rtol=eps,equal_nan=True)

    def _cmpeq(golden,output,dtype):
        if not os.path.exists(golden) or not os.path.exists(output): return False
        g=np.fromfile(golden,dtype=dtype)
        o=np.fromfile(output,dtype=dtype)
        return g.shape==o.shape and np.array_equal(g,o)

    def main():
        strict=os.getenv('COMPARE_STRICT','1')!='0'
        failed=[]
        if not (_cmp("golden_v3.bin","v3.bin",np.float32,1e-4,-1)):
            failed.append('f32')
            print('[ERROR] compare failed: f32')
        else:
            print('[INFO] f32: passed')
        if not (_cmp("golden_v3_tail.bin","v3_tail.bin",np.float32,1e-4,1000)):
            failed.append('tail')
            print('[ERROR] compare failed: tail')
        else:
            print('[INFO] tail: passed')
        if failed:
            if strict: print(f"[ERROR] {len(failed)} variant(s) failed"); sys.exit(2)
            print(f"[WARN] {len(failed)} variant(s) failed (non-gating)")
            return
        print("[INFO] compare passed (all 2 variants)")

    return locals()

HELPERS = {}
SEED_EXPR = 'SEED'
COMPARE_PAIRS = [
    (
        'golden_v3.bin',
        'v3.bin'
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
        'v1_tail',
        'f32',
        'arg3'
    ),
    (
        'v2_tail',
        'f32',
        'arg4'
    ),
    (
        'v3_tail',
        'f32',
        'arg5'
    )
]
SCALARS = []
SCALAR_SOURCES = []
SCALAR_DTYPES = []


@pto.jit(
    name='vmax_deep_merged_kernel',
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
    arg4: pto.ptr(pto.f32, "gm"),
    arg5: pto.ptr(pto.f32, "gm"),
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
    name='legacy_micro_op_binary_vector_vmax',
    grid=1,
    kernel=kernel,
    make_case=make_case,
    check=check_case,
)]

auto_main(globals())
