#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Migrated from test/vpto/cases/micro-op/binary-vector/vand."""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

from common import auto_main
from ptodsl import pto

KERNEL_SOURCE = (
    'module attributes {pto.target_arch = "a5", pto.kernel_kind = #pto.kernel_kind<vector>} {\n    func.func @van'
    'd_deep_merged_kernel(%arg0: !pto.ptr<ui16, gm>, %arg1: !pto.ptr<ui16, gm>, %arg2: !pto.ptr<ui16, gm>, %arg3:'
    ' !pto.ptr<ui16, gm>, %arg4: !pto.ptr<ui16, gm>, %arg5: !pto.ptr<ui16, gm>, %arg6: !pto.ptr<f8E4M3FN, gm>, %a'
    'rg7: !pto.ptr<f8E4M3FN, gm>, %arg8: !pto.ptr<f8E4M3FN, gm>, %arg9: !pto.ptr<f8E4M3FN, gm>, %arg10: !pto.ptr<'
    'f8E4M3FN, gm>, %arg11: !pto.ptr<!pto.hif8, gm>, %arg12: !pto.ptr<!pto.hif8, gm>, %arg13: !pto.ptr<!pto.hif8,'
    ' gm>, %arg14: !pto.ptr<!pto.hif8, gm>, %arg15: !pto.ptr<!pto.hif8, gm>) attributes {pto.kernel} {\n    // me'
    'rged from vand_i16_unsigned_kernel via LaunchVand_i16_unsigned_kernel\n\n    %c0_m0 = arith.constant 0 : ind'
    'ex\n    %c1024_m0 = arith.constant 1024 : index\n    %c128_m0 = arith.constant 128 : index\n    %c0_i64_m0 ='
    ' arith.constant 0 : i64\n    %c1_i64_m0 = arith.constant 1 : i64\n    %c32_i64_m0 = arith.constant 32 : i64'
    '\n    %c64_i64_m0 = arith.constant 64 : i64\n    %c2048_i64_m0 = arith.constant 2048 : i64\n    %c4096_i64_m'
    '0 = arith.constant 4096 : i64\n\n    %ub_lhs_m0 = pto.castptr %c0_i64_m0 : i64 -> !pto.ptr<ui16, ub>\n    %u'
    'b_rhs_m0 = pto.castptr %c2048_i64_m0 : i64 -> !pto.ptr<ui16, ub>\n    %ub_out_m0 = pto.castptr %c4096_i64_m0'
    ' : i64 -> !pto.ptr<ui16, ub>\n    %c6144_i64_lowp = arith.constant 6144 : i64\n    %c6400_i64_lowp = arith.c'
    'onstant 6400 : i64\n    %c6656_i64_lowp = arith.constant 6656 : i64\n    %c6912_i64_lowp = arith.constant 69'
    '12 : i64\n    %c7168_i64_lowp = arith.constant 7168 : i64\n    %c7424_i64_lowp = arith.constant 7424 : i64\n'
    '    %c7680_i64_lowp = arith.constant 7680 : i64\n    %c7936_i64_lowp = arith.constant 7936 : i64\n    %c8192'
    '_i64_lowp = arith.constant 8192 : i64\n    %c8448_i64_lowp = arith.constant 8448 : i64\n    %c4_i64_lowp = a'
    'rith.constant 4 : i64\n    %f8_lhs_lowp = pto.castptr %c6144_i64_lowp : i64 -> !pto.ptr<f8E4M3FN, ub>\n    %'
    'f8_rhs_lowp = pto.castptr %c6400_i64_lowp : i64 -> !pto.ptr<f8E4M3FN, ub>\n    %f8_and_lowp = pto.castptr %c'
    '6656_i64_lowp : i64 -> !pto.ptr<f8E4M3FN, ub>\n    %f8_or_lowp = pto.castptr %c6912_i64_lowp : i64 -> !pto.p'
    'tr<f8E4M3FN, ub>\n    %f8_xor_lowp = pto.castptr %c7168_i64_lowp : i64 -> !pto.ptr<f8E4M3FN, ub>\n    %hif8_'
    'lhs_lowp = pto.castptr %c7424_i64_lowp : i64 -> !pto.ptr<!pto.hif8, ub>\n    %hif8_rhs_lowp = pto.castptr %c'
    '7680_i64_lowp : i64 -> !pto.ptr<!pto.hif8, ub>\n    %hif8_and_lowp = pto.castptr %c7936_i64_lowp : i64 -> !p'
    'to.ptr<!pto.hif8, ub>\n    %hif8_or_lowp = pto.castptr %c8192_i64_lowp : i64 -> !pto.ptr<!pto.hif8, ub>\n   '
    ' %hif8_xor_lowp = pto.castptr %c8448_i64_lowp : i64 -> !pto.ptr<!pto.hif8, ub>\n\n    %false_m0 = arith.cons'
    'tant false\n    pto.mte_gm_ub %arg0, %ub_lhs_m0, %c0_i64_m0, %c64_i64_m0\n      nburst(%c32_i64_m0, %c64_i64'
    '_m0, %c64_i64_m0)\n      : !pto.ptr<ui16, gm>, !pto.ptr<ui16, ub>, i64, i64, i64, i64, i64\n    pto.mte_gm_u'
    'b %arg1, %ub_rhs_m0, %c0_i64_m0, %c64_i64_m0\n      nburst(%c32_i64_m0, %c64_i64_m0, %c64_i64_m0)\n      : !'
    'pto.ptr<ui16, gm>, !pto.ptr<ui16, ub>, i64, i64, i64, i64, i64\n    pto.mte_gm_ub %arg6, %f8_lhs_lowp, %c0_i'
    '64_m0, %c64_i64_m0\n      nburst(%c4_i64_lowp, %c64_i64_m0, %c64_i64_m0)\n      : !pto.ptr<f8E4M3FN, gm>, !p'
    'to.ptr<f8E4M3FN, ub>, i64, i64, i64, i64, i64\n    pto.mte_gm_ub %arg7, %f8_rhs_lowp, %c0_i64_m0, %c64_i64_m'
    '0\n      nburst(%c4_i64_lowp, %c64_i64_m0, %c64_i64_m0)\n      : !pto.ptr<f8E4M3FN, gm>, !pto.ptr<f8E4M3FN, '
    'ub>, i64, i64, i64, i64, i64\n    pto.mte_gm_ub %arg11, %hif8_lhs_lowp, %c0_i64_m0, %c64_i64_m0\n      nburs'
    't(%c4_i64_lowp, %c64_i64_m0, %c64_i64_m0)\n      : !pto.ptr<!pto.hif8, gm>, !pto.ptr<!pto.hif8, ub>, i64, i6'
    '4, i64, i64, i64\n    pto.mte_gm_ub %arg12, %hif8_rhs_lowp, %c0_i64_m0, %c64_i64_m0\n      nburst(%c4_i64_lo'
    'wp, %c64_i64_m0, %c64_i64_m0)\n      : !pto.ptr<!pto.hif8, gm>, !pto.ptr<!pto.hif8, ub>, i64, i64, i64, i64,'
    ' i64\n\n    pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n    pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVEN'
    'T_ID0"]\n\n    pto.vecscope {\n      %mask_m0 = pto.pset_b16 "PAT_ALL" : !pto.mask<b16>\n      scf.for %offs'
    'et_m0 = %c0_m0 to %c1024_m0 step %c128_m0 {\n        %lhs_m0 = pto.vlds %ub_lhs_m0[%offset_m0] : !pto.ptr<ui'
    '16, ub>, index -> !pto.vreg<128xui16>\n        %rhs_m0 = pto.vlds %ub_rhs_m0[%offset_m0] : !pto.ptr<ui16, ub>, index -> !p'
    'to.vreg<128xui16>\n        %out_m0 = pto.vand %lhs_m0, %rhs_m0, %mask_m0 : !pto.vreg<128xui16>, !pto.vreg<12'
    '8xui16>, !pto.mask<b16> -> !pto.vreg<128xui16>\n        pto.vsts %out_m0, %ub_out_m0[%offset_m0], %mask_m0 :'
    ' !pto.vreg<128xui16>, !pto.ptr<ui16, ub>, index, !pto.mask<b16>\n      }\n      %mask_lowp = pto.pset_b8 "PAT_ALL" '
    ': !pto.mask<b8>\n      %f8_lhs_val = pto.vlds %f8_lhs_lowp[%c0_m0] : !pto.ptr<f8E4M3FN, ub>, index -> !pto.vreg<256'
    'xf8E4M3FN>\n      %f8_rhs_val = pto.vlds %f8_rhs_lowp[%c0_m0] : !pto.ptr<f8E4M3FN, ub>, index -> !pto.vreg<256xf8E4'
    'M3FN>\n      %f8_and_val = pto.vand %f8_lhs_val, %f8_rhs_val, %mask_lowp : !pto.vreg<256xf8E4M3FN>, !pto.vre'
    'g<256xf8E4M3FN>, !pto.mask<b8> -> !pto.vreg<256xf8E4M3FN>\n      %f8_or_val = pto.vor %f8_lhs_val, %f8_rhs_v'
    'al, %mask_lowp : !pto.vreg<256xf8E4M3FN>, !pto.vreg<256xf8E4M3FN>, !pto.mask<b8> -> !pto.vreg<256xf8E4M3FN>'
    '\n      %f8_xor_val = pto.vxor %f8_lhs_val, %f8_rhs_val, %mask_lowp : !pto.vreg<256xf8E4M3FN>, !pto.vreg<256'
    'xf8E4M3FN>, !pto.mask<b8> -> !pto.vreg<256xf8E4M3FN>\n      pto.vsts %f8_and_val, %f8_and_lowp[%c0_m0], %mas'
    'k_lowp : !pto.vreg<256xf8E4M3FN>, !pto.ptr<f8E4M3FN, ub>, index, !pto.mask<b8>\n      pto.vsts %f8_or_val, %f8_or_l'
    'owp[%c0_m0], %mask_lowp : !pto.vreg<256xf8E4M3FN>, !pto.ptr<f8E4M3FN, ub>, index, !pto.mask<b8>\n      pto.vsts %f8'
    '_xor_val, %f8_xor_lowp[%c0_m0], %mask_lowp : !pto.vreg<256xf8E4M3FN>, !pto.ptr<f8E4M3FN, ub>, index, !pto.mask<b8>'
    '\n\n      %hif8_lhs_val = pto.vlds %hif8_lhs_lowp[%c0_m0] : !pto.ptr<!pto.hif8, ub>, index -> !pto.vreg<256x!pto.hi'
    'f8>\n      %hif8_rhs_val = pto.vlds %hif8_rhs_lowp[%c0_m0] : !pto.ptr<!pto.hif8, ub>, index -> !pto.vreg<256x!pto.h'
    'if8>\n      %hif8_and_val = pto.vand %hif8_lhs_val, %hif8_rhs_val, %mask_lowp : !pto.vreg<256x!pto.hif8>, !p'
    'to.vreg<256x!pto.hif8>, !pto.mask<b8> -> !pto.vreg<256x!pto.hif8>\n      %hif8_or_val = pto.vor %hif8_lhs_va'
    'l, %hif8_rhs_val, %mask_lowp : !pto.vreg<256x!pto.hif8>, !pto.vreg<256x!pto.hif8>, !pto.mask<b8> -> !pto.vre'
    'g<256x!pto.hif8>\n      %hif8_xor_val = pto.vxor %hif8_lhs_val, %hif8_rhs_val, %mask_lowp : !pto.vreg<256x!p'
    'to.hif8>, !pto.vreg<256x!pto.hif8>, !pto.mask<b8> -> !pto.vreg<256x!pto.hif8>\n      pto.vsts %hif8_and_val,'
    ' %hif8_and_lowp[%c0_m0], %mask_lowp : !pto.vreg<256x!pto.hif8>, !pto.ptr<!pto.hif8, ub>, index, !pto.mask<b8>\n    '
    '  pto.vsts %hif8_or_val, %hif8_or_lowp[%c0_m0], %mask_lowp : !pto.vreg<256x!pto.hif8>, !pto.ptr<!pto.hif8, u'
    'b>, index, !pto.mask<b8>\n      pto.vsts %hif8_xor_val, %hif8_xor_lowp[%c0_m0], %mask_lowp : !pto.vreg<256x!pto.hif'
    '8>, !pto.ptr<!pto.hif8, ub>, index, !pto.mask<b8>\n    }\n\n    pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n '
    '   pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n    pto.mte_ub_gm %ub_out_m0, %arg2, %c64_i64_m0\n    '
    '  nburst(%c32_i64_m0, %c64_i64_m0, %c64_i64_m0)\n      : !pto.ptr<ui16, ub>, !pto.ptr<ui16, gm>, i64, i64, i'
    '64, i64\n    pto.mte_ub_gm %f8_and_lowp, %arg8, %c64_i64_m0\n      nburst(%c4_i64_lowp, %c64_i64_m0, %c64_i6'
    '4_m0)\n      : !pto.ptr<f8E4M3FN, ub>, !pto.ptr<f8E4M3FN, gm>, i64, i64, i64, i64\n    pto.mte_ub_gm %f8_xor'
    '_lowp, %arg9, %c64_i64_m0\n      nburst(%c4_i64_lowp, %c64_i64_m0, %c64_i64_m0)\n      : !pto.ptr<f8E4M3FN, '
    'ub>, !pto.ptr<f8E4M3FN, gm>, i64, i64, i64, i64\n    pto.mte_ub_gm %f8_or_lowp, %arg10, %c64_i64_m0\n      n'
    'burst(%c4_i64_lowp, %c64_i64_m0, %c64_i64_m0)\n      : !pto.ptr<f8E4M3FN, ub>, !pto.ptr<f8E4M3FN, gm>, i64, '
    'i64, i64, i64\n    pto.mte_ub_gm %hif8_and_lowp, %arg13, %c64_i64_m0\n      nburst(%c4_i64_lowp, %c64_i64_m0'
    ', %c64_i64_m0)\n      : !pto.ptr<!pto.hif8, ub>, !pto.ptr<!pto.hif8, gm>, i64, i64, i64, i64\n    pto.mte_ub'
    '_gm %hif8_xor_lowp, %arg14, %c64_i64_m0\n      nburst(%c4_i64_lowp, %c64_i64_m0, %c64_i64_m0)\n      : !pto.'
    'ptr<!pto.hif8, ub>, !pto.ptr<!pto.hif8, gm>, i64, i64, i64, i64\n    pto.mte_ub_gm %hif8_or_lowp, %arg15, %c'
    '64_i64_m0\n      nburst(%c4_i64_lowp, %c64_i64_m0, %c64_i64_m0)\n      : !pto.ptr<!pto.hif8, ub>, !pto.ptr<!'
    'pto.hif8, gm>, i64, i64, i64, i64\n    pto.barrier #pto.pipe<PIPE_ALL>\n\n    // merged from vand_mask_edge_'
    'kernel via LaunchVand_mask_edge_kernel\n\n    %c0_m1 = arith.constant 0 : index\n    %c0_i64_m1 = arith.cons'
    'tant 0 : i64\n    %c1_i64_m1 = arith.constant 1 : i64\n    %c32_i64_m1 = arith.constant 32 : i64\n    %c64_i'
    '64_m1 = arith.constant 64 : i64\n    %c1024_m1 = arith.constant 1024 : index\n    %c128_m1 = arith.constant '
    '128 : index\n    %c2048_i64_m1 = arith.constant 2048 : i64\n    %c4096_i64_m1 = arith.constant 4096 : i64\n'
    '\n    %ub_lhs_m1 = pto.castptr %c0_i64_m1 : i64 -> !pto.ptr<ui16, ub>\n    %ub_rhs_m1 = pto.castptr %c2048_i'
    '64_m1 : i64 -> !pto.ptr<ui16, ub>\n    %ub_out_m1 = pto.castptr %c4096_i64_m1 : i64 -> !pto.ptr<ui16, ub>\n'
    '\n    %false_m1 = arith.constant false\n    pto.mte_gm_ub %arg3, %ub_lhs_m1, %c0_i64_m1, %c64_i64_m1\n      '
    'nburst(%c32_i64_m1, %c64_i64_m1, %c64_i64_m1)\n      : !pto.ptr<ui16, gm>, !pto.ptr<ui16, ub>, i64, i64, i64'
    ', i64, i64\n    pto.mte_gm_ub %arg4, %ub_rhs_m1, %c0_i64_m1, %c64_i64_m1\n      nburst(%c32_i64_m1, %c64_i64'
    '_m1, %c64_i64_m1)\n      : !pto.ptr<ui16, gm>, !pto.ptr<ui16, ub>, i64, i64, i64, i64, i64\n\n    pto.set_fl'
    'ag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n    pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n\n    pto.vec'
    'scope {\n      %mask_m1 = pto.pset_b16 "PAT_ALL" : !pto.mask<b16>\n      scf.for %offset_m1 = %c0_m1 to %c10'
    '24_m1 step %c128_m1 {\n        %lhs_m1 = pto.vlds %ub_lhs_m1[%offset_m1] : !pto.ptr<ui16, ub>, index -> !pto.vreg<1'
    '28xui16>\n        %rhs_m1 = pto.vlds %ub_rhs_m1[%offset_m1] : !pto.ptr<ui16, ub>, index -> !pto.vreg<128xui16>\n   '
    '     %out_m1 = pto.vand %lhs_m1, %rhs_m1, %mask_m1 : !pto.vreg<128xui16>, !pto.vreg<128xui16>, !pto.mask<b16'
    '> -> !pto.vreg<128xui16>\n        pto.vsts %out_m1, %ub_out_m1[%offset_m1], %mask_m1 : !pto.vreg<128xui16>, '
    '!pto.ptr<ui16, ub>, index, !pto.mask<b16>\n      }\n    }\n\n    pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n'
    '    pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n    pto.mte_ub_gm %ub_out_m1, %arg5, %c64_i64_m1\n   '
    '   nburst(%c32_i64_m1, %c64_i64_m1, %c64_i64_m1)\n      : !pto.ptr<ui16, ub>, !pto.ptr<ui16, gm>, i64, i64, '
    'i64, i64\n    pto.barrier #pto.pipe<PIPE_ALL>\n    %__case_merge_guard = arith.constant false\n\n    // inac'
    'tive cross-case merge from micro-op/binary-vector/vxor\n    scf.if %__case_merge_guard {\n      \n          '
    '// merged from vxor_i16_unsigned_kernel via LaunchVxor_i16_unsigned_kernel\n      \n          %c0_m0_cmg1_1 '
    '= arith.constant 0 : index\n          %c1024_m0_cmg1_1 = arith.constant 1024 : index\n          %c128_m0_cmg'
    '1_1 = arith.constant 128 : index\n          %c0_i64_m0_cmg1_1 = arith.constant 0 : i64\n          %c1_i64_m0'
    '_cmg1_1 = arith.constant 1 : i64\n          %c32_i64_m0_cmg1_1 = arith.constant 32 : i64\n          %c64_i64'
    '_m0_cmg1_1 = arith.constant 64 : i64\n          %c2048_i64_m0_cmg1_1 = arith.constant 2048 : i64\n          '
    '%c4096_i64_m0_cmg1_1 = arith.constant 4096 : i64\n      \n          %ub_lhs_m0_cmg1_1 = pto.castptr %c0_i64_'
    'm0_cmg1_1 : i64 -> !pto.ptr<ui16, ub>\n          %ub_rhs_m0_cmg1_1 = pto.castptr %c2048_i64_m0_cmg1_1 : i64 '
    '-> !pto.ptr<ui16, ub>\n          %ub_out_m0_cmg1_1 = pto.castptr %c4096_i64_m0_cmg1_1 : i64 -> !pto.ptr<ui16'
    ', ub>\n      \n          %false_m0_cmg1_1 = arith.constant false\n          pto.mte_gm_ub %arg0, %ub_lhs_m0_'
    'cmg1_1, %c0_i64_m0_cmg1_1, %c64_i64_m0_cmg1_1\n            nburst(%c32_i64_m0_cmg1_1, %c64_i64_m0_cmg1_1, %c'
    '64_i64_m0_cmg1_1)\n            : !pto.ptr<ui16, gm>, !pto.ptr<ui16, ub>, i64, i64, i64, i64, i64\n          '
    'pto.mte_gm_ub %arg1, %ub_rhs_m0_cmg1_1, %c0_i64_m0_cmg1_1, %c64_i64_m0_cmg1_1\n            nburst(%c32_i64_m'
    '0_cmg1_1, %c64_i64_m0_cmg1_1, %c64_i64_m0_cmg1_1)\n            : !pto.ptr<ui16, gm>, !pto.ptr<ui16, ub>, i64'
    ', i64, i64, i64, i64\n      \n          pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n          pto.wait'
    '_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n      \n          pto.vecscope {\n            %mask_m0_cmg1_1 = p'
    'to.pset_b16 "PAT_ALL" : !pto.mask<b16>\n            scf.for %offset_m0_cmg1_1 = %c0_m0_cmg1_1 to %c1024_m0_c'
    'mg1_1 step %c128_m0_cmg1_1 {\n              %lhs_m0_cmg1_1 = pto.vlds %ub_lhs_m0_cmg1_1[%offset_m0_cmg1_1] :'
    ' !pto.ptr<ui16, ub>, index -> !pto.vreg<128xui16>\n              %rhs_m0_cmg1_1 = pto.vlds %ub_rhs_m0_cmg1_1[%offse'
    't_m0_cmg1_1] : !pto.ptr<ui16, ub>, index -> !pto.vreg<128xui16>\n              %out_m0_cmg1_1 = pto.vxor %lhs_m0_cm'
    'g1_1, %rhs_m0_cmg1_1, %mask_m0_cmg1_1 : !pto.vreg<128xui16>, !pto.vreg<128xui16>, !pto.mask<b16> -> !pto.vre'
    'g<128xui16>\n              pto.vsts %out_m0_cmg1_1, %ub_out_m0_cmg1_1[%offset_m0_cmg1_1], %mask_m0_cmg1_1 : '
    '!pto.vreg<128xui16>, !pto.ptr<ui16, ub>, index, !pto.mask<b16>\n            }\n          }\n      \n          pto.s'
    'et_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n          pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n  '
    '        pto.mte_ub_gm %ub_out_m0_cmg1_1, %arg2, %c64_i64_m0_cmg1_1\n            nburst(%c32_i64_m0_cmg1_1, %'
    'c64_i64_m0_cmg1_1, %c64_i64_m0_cmg1_1)\n            : !pto.ptr<ui16, ub>, !pto.ptr<ui16, gm>, i64, i64, i64,'
    ' i64\n          pto.barrier #pto.pipe<PIPE_ALL>\n      \n          // merged from vxor_mask_edge_kernel via '
    'LaunchVxor_mask_edge_kernel\n      \n          %c0_m1_cmg1_1 = arith.constant 0 : index\n          %c0_i64_m'
    '1_cmg1_1 = arith.constant 0 : i64\n          %c1_i64_m1_cmg1_1 = arith.constant 1 : i64\n          %c32_i64_'
    'm1_cmg1_1 = arith.constant 32 : i64\n          %c64_i64_m1_cmg1_1 = arith.constant 64 : i64\n          %c102'
    '4_m1_cmg1_1 = arith.constant 1024 : index\n          %c128_m1_cmg1_1 = arith.constant 128 : index\n         '
    ' %c2048_i64_m1_cmg1_1 = arith.constant 2048 : i64\n          %c4096_i64_m1_cmg1_1 = arith.constant 4096 : i6'
    '4\n      \n          %ub_lhs_m1_cmg1_1 = pto.castptr %c0_i64_m1_cmg1_1 : i64 -> !pto.ptr<ui16, ub>\n        '
    '  %ub_rhs_m1_cmg1_1 = pto.castptr %c2048_i64_m1_cmg1_1 : i64 -> !pto.ptr<ui16, ub>\n          %ub_out_m1_cmg'
    '1_1 = pto.castptr %c4096_i64_m1_cmg1_1 : i64 -> !pto.ptr<ui16, ub>\n      \n          %false_m1_cmg1_1 = ari'
    'th.constant false\n          pto.mte_gm_ub %arg3, %ub_lhs_m1_cmg1_1, %c0_i64_m1_cmg1_1, %c64_i64_m1_cmg1_1\n'
    '            nburst(%c32_i64_m1_cmg1_1, %c64_i64_m1_cmg1_1, %c64_i64_m1_cmg1_1)\n            : !pto.ptr<ui16,'
    ' gm>, !pto.ptr<ui16, ub>, i64, i64, i64, i64, i64\n          pto.mte_gm_ub %arg4, %ub_rhs_m1_cmg1_1, %c0_i64'
    '_m1_cmg1_1, %c64_i64_m1_cmg1_1\n            nburst(%c32_i64_m1_cmg1_1, %c64_i64_m1_cmg1_1, %c64_i64_m1_cmg1_'
    '1)\n            : !pto.ptr<ui16, gm>, !pto.ptr<ui16, ub>, i64, i64, i64, i64, i64\n      \n          pto.set'
    '_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n          pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n    '
    '  \n          pto.vecscope {\n            %mask_m1_cmg1_1 = pto.pset_b16 "PAT_ALL" : !pto.mask<b16>\n       '
    '     scf.for %offset_m1_cmg1_1 = %c0_m1_cmg1_1 to %c1024_m1_cmg1_1 step %c128_m1_cmg1_1 {\n              %lh'
    's_m1_cmg1_1 = pto.vlds %ub_lhs_m1_cmg1_1[%offset_m1_cmg1_1] : !pto.ptr<ui16, ub>, index -> !pto.vreg<128xui16>\n   '
    '           %rhs_m1_cmg1_1 = pto.vlds %ub_rhs_m1_cmg1_1[%offset_m1_cmg1_1] : !pto.ptr<ui16, ub>, index -> !pto.vreg<'
    '128xui16>\n              %out_m1_cmg1_1 = pto.vxor %lhs_m1_cmg1_1, %rhs_m1_cmg1_1, %mask_m1_cmg1_1 : !pto.vr'
    'eg<128xui16>, !pto.vreg<128xui16>, !pto.mask<b16> -> !pto.vreg<128xui16>\n              pto.vsts %out_m1_cmg'
    '1_1, %ub_out_m1_cmg1_1[%offset_m1_cmg1_1], %mask_m1_cmg1_1 : !pto.vreg<128xui16>, !pto.ptr<ui16, ub>, index, !pto.m'
    'ask<b16>\n            }\n          }\n      \n          pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n  '
    '        pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n          pto.mte_ub_gm %ub_out_m1_cmg1_1, %arg5,'
    ' %c64_i64_m1_cmg1_1\n            nburst(%c32_i64_m1_cmg1_1, %c64_i64_m1_cmg1_1, %c64_i64_m1_cmg1_1)\n       '
    '     : !pto.ptr<ui16, ub>, !pto.ptr<ui16, gm>, i64, i64, i64, i64\n          pto.barrier #pto.pipe<PIPE_ALL>'
    '\n    }\n\n    return\n  }\n}\n'
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

    # Merged vand test case.

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
        v1=rng.integers(0,0x10000,size=(ROWS,COLS),dtype=np.uint16)
        v2=rng.integers(0,0x10000,size=(ROWS,COLS),dtype=np.uint16)
        g=np.bitwise_and(v1,v2)
        v3=np.zeros((ROWS,COLS),dtype=np.uint16)
        _files["v1.bin"] = v1.reshape(-1)
        _files["v2.bin"] = v2.reshape(-1)
        _files["v3.bin"] = v3.reshape(-1)
        _files["golden_v3.bin"] = g.reshape(-1)

    # ---- mask_edge ----
    def gen_mask_edge(out, rng):
        v1=rng.integers(0,0x10000,size=(ROWS,COLS),dtype=np.uint16)
        v2=rng.integers(0,0x10000,size=(ROWS,COLS),dtype=np.uint16)
        g=np.bitwise_and(v1,v2)
        v3=np.zeros((ROWS,COLS),dtype=np.uint16)
        _files["v1_mask_edge.bin"] = v1.reshape(-1)
        _files["v2_mask_edge.bin"] = v2.reshape(-1)
        _files["v3_mask_edge.bin"] = v3.reshape(-1)
        _files["golden_v3_mask_edge.bin"] = g.reshape(-1)

    # ---- lowp ----
    def gen_lowp(out, rng):
        f8_v1=rng.integers(0,256,size=256,dtype=np.uint8)
        f8_v2=rng.integers(0,256,size=256,dtype=np.uint8)
        hif8_v1=rng.integers(0,256,size=256,dtype=np.uint8)
        hif8_v2=rng.integers(0,256,size=256,dtype=np.uint8)
        zeros=np.zeros(256,dtype=np.uint8)
        _files["v1_f8.bin"] = f8_v1
        _files["v2_f8.bin"] = f8_v2
        _files["v3_f8_and.bin"] = zeros
        _files["v4_f8_xor.bin"] = zeros
        _files["v5_f8_or.bin"] = zeros
        _files["v1_hif8.bin"] = hif8_v1
        _files["v2_hif8.bin"] = hif8_v2
        _files["v3_hif8_and.bin"] = zeros
        _files["v4_hif8_xor.bin"] = zeros
        _files["v5_hif8_or.bin"] = zeros
        _files["golden_v3_f8_and.bin"] = np.bitwise_and(f8_v1,f8_v2)
        _files["golden_v4_f8_xor.bin"] = np.bitwise_xor(f8_v1,f8_v2)
        _files["golden_v5_f8_or.bin"] = np.bitwise_or(f8_v1,f8_v2)
        _files["golden_v3_hif8_and.bin"] = np.bitwise_and(hif8_v1,hif8_v2)
        _files["golden_v4_hif8_xor.bin"] = np.bitwise_xor(hif8_v1,hif8_v2)
        _files["golden_v5_hif8_or.bin"] = np.bitwise_or(hif8_v1,hif8_v2)

    GENERATORS = [
        gen_f32,
        gen_mask_edge,
        gen_lowp,
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

    # Merged vand test case.

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
        if not (_cmp("golden_v3.bin","v3.bin",np.uint16,0,1024)):
            failed.append('f32')
            print('[ERROR] compare failed: f32')
        else:
            print('[INFO] f32: passed')
        if not (_cmpeq("golden_v3_mask_edge.bin","v3_mask_edge.bin",np.uint16)):
            failed.append('mask_edge')
            print('[ERROR] compare failed: mask_edge')
        else:
            print('[INFO] mask_edge: passed')
        if not (_cmpeq("golden_v3_f8_and.bin","v3_f8_and.bin",np.uint8)):
            failed.append('f8_and')
            print('[ERROR] compare failed: f8_and')
        else:
            print('[INFO] f8_and: passed')
        if not (_cmpeq("golden_v4_f8_xor.bin","v4_f8_xor.bin",np.uint8)):
            failed.append('f8_xor')
            print('[ERROR] compare failed: f8_xor')
        else:
            print('[INFO] f8_xor: passed')
        if not (_cmpeq("golden_v5_f8_or.bin","v5_f8_or.bin",np.uint8)):
            failed.append('f8_or')
            print('[ERROR] compare failed: f8_or')
        else:
            print('[INFO] f8_or: passed')
        if not (_cmpeq("golden_v3_hif8_and.bin","v3_hif8_and.bin",np.uint8)):
            failed.append('hif8_and')
            print('[ERROR] compare failed: hif8_and')
        else:
            print('[INFO] hif8_and: passed')
        if not (_cmpeq("golden_v4_hif8_xor.bin","v4_hif8_xor.bin",np.uint8)):
            failed.append('hif8_xor')
            print('[ERROR] compare failed: hif8_xor')
        else:
            print('[INFO] hif8_xor: passed')
        if not (_cmpeq("golden_v5_hif8_or.bin","v5_hif8_or.bin",np.uint8)):
            failed.append('hif8_or')
            print('[ERROR] compare failed: hif8_or')
        else:
            print('[INFO] hif8_or: passed')
        if failed:
            if strict: print(f"[ERROR] {len(failed)} variant(s) failed"); sys.exit(2)
            print(f"[WARN] {len(failed)} variant(s) failed (non-gating)")
            return
        print("[INFO] compare passed (all 8 variants)")

    return locals()

HELPERS = {}
SEED_EXPR = 'SEED'
COMPARE_PAIRS = [
    (
        'golden_v3.bin',
        'v3.bin'
    ),
    (
        'golden_v3_mask_edge.bin',
        'v3_mask_edge.bin'
    ),
    (
        'golden_v3_f8_and.bin',
        'v3_f8_and.bin'
    ),
    (
        'golden_v4_f8_xor.bin',
        'v4_f8_xor.bin'
    ),
    (
        'golden_v5_f8_or.bin',
        'v5_f8_or.bin'
    ),
    (
        'golden_v3_hif8_and.bin',
        'v3_hif8_and.bin'
    ),
    (
        'golden_v4_hif8_xor.bin',
        'v4_hif8_xor.bin'
    ),
    (
        'golden_v5_hif8_or.bin',
        'v5_hif8_or.bin'
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
        'ui16',
        'arg0'
    ),
    (
        'v2',
        'ui16',
        'arg1'
    ),
    (
        'v3',
        'ui16',
        'arg2'
    ),
    (
        'v1_mask_edge',
        'ui16',
        'arg3'
    ),
    (
        'v2_mask_edge',
        'ui16',
        'arg4'
    ),
    (
        'v3_mask_edge',
        'ui16',
        'arg5'
    ),
    (
        'v1_f8',
        'f8e4m3',
        'arg6'
    ),
    (
        'v2_f8',
        'f8e4m3',
        'arg7'
    ),
    (
        'v3_f8_and',
        'f8e4m3',
        'arg8'
    ),
    (
        'v4_f8_xor',
        'f8e4m3',
        'arg9'
    ),
    (
        'v5_f8_or',
        'f8e4m3',
        'arg10'
    ),
    (
        'v1_hif8',
        'hif8',
        'arg11'
    ),
    (
        'v2_hif8',
        'hif8',
        'arg12'
    ),
    (
        'v3_hif8_and',
        'hif8',
        'arg13'
    ),
    (
        'v4_hif8_xor',
        'hif8',
        'arg14'
    ),
    (
        'v5_hif8_or',
        'hif8',
        'arg15'
    )
]
SCALARS = []
SCALAR_SOURCES = []
SCALAR_DTYPES = []


@pto.jit(
    name='vand_deep_merged_kernel',
    target="a5",
    backend="vpto",
    mode="explicit",
    source=KERNEL_SOURCE,
)
def kernel(
    arg0: pto.ptr(pto.ui16, "gm"),
    arg1: pto.ptr(pto.ui16, "gm"),
    arg2: pto.ptr(pto.ui16, "gm"),
    arg3: pto.ptr(pto.ui16, "gm"),
    arg4: pto.ptr(pto.ui16, "gm"),
    arg5: pto.ptr(pto.ui16, "gm"),
    arg6: pto.ptr(pto.f8e4m3, "gm"),
    arg7: pto.ptr(pto.f8e4m3, "gm"),
    arg8: pto.ptr(pto.f8e4m3, "gm"),
    arg9: pto.ptr(pto.f8e4m3, "gm"),
    arg10: pto.ptr(pto.f8e4m3, "gm"),
    arg11: pto.ptr(pto.hif8, "gm"),
    arg12: pto.ptr(pto.hif8, "gm"),
    arg13: pto.ptr(pto.hif8, "gm"),
    arg14: pto.ptr(pto.hif8, "gm"),
    arg15: pto.ptr(pto.hif8, "gm"),
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
    name='legacy_micro_op_binary_vector_vand',
    grid=1,
    kernel=kernel,
    make_case=make_case,
    check=check_case,
)]

auto_main(globals())
