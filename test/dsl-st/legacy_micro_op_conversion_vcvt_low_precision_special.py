#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Migrated from test/vpto/cases/micro-op/conversion/vcvt-low-precision-special.

The C++ host of this case launched more than one entry point, so the module
compiles each one and a run() hook issues the launches in order.
"""

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
    '\n// -----------------------------------------------------------------------------\n// case: micro-op/conver'
    'sion/vcvt-low-precision-special\n// family: conversion\n// target_ops: pto.vcvt\n// scenarios: lowp-to-f32/b'
    'f16, special-values, all-parts\n// -------------------------------------------------------------------------'
    '----\nmodule attributes {pto.target_arch = "a5", pto.kernel_kind = #pto.kernel_kind<vector>} {\n  func.func '
    '@vcvt_low_precision_quantize_special_kernel(%f8e4_in: !pto.ptr<ui8, gm>,\n                                  '
    '                      %f8e5_in: !pto.ptr<ui8, gm>,\n                                                        '
    '%hif8_in: !pto.ptr<ui8, gm>,\n                                                        %f4e1_in: !pto.ptr<ui8'
    ', gm>,\n                                                        %f4e2_in: !pto.ptr<ui8, gm>,\n              '
    '                                          %f8e4_out: !pto.ptr<ui8, gm>,\n                                   '
    '                     %f8e5_out: !pto.ptr<ui8, gm>,\n                                                        '
    '%hif8_out: !pto.ptr<ui8, gm>,\n                                                        %f4e1_out: !pto.ptr<u'
    'i8, gm>,\n                                                        %f4e2_out: !pto.ptr<ui8, gm>) attributes {'
    'pto.kernel} {\n    %c0 = arith.constant 0 : index\n    %c64 = arith.constant 64 : index\n    %c128 = arith.c'
    'onstant 128 : index\n    %c192 = arith.constant 192 : index\n    %c256 = arith.constant 256 : index\n    %c3'
    '84 = arith.constant 384 : index\n    %c512 = arith.constant 512 : index\n    %c768 = arith.constant 768 : in'
    'dex\n    %c0_i64 = arith.constant 0 : i64\n    %c1_i64 = arith.constant 1 : i64\n    %c1024_i64 = arith.cons'
    'tant 1024 : i64\n    %c2048_i64 = arith.constant 2048 : i64\n    %c3072_i64 = arith.constant 3072 : i64\n   '
    ' %c4096_i64 = arith.constant 4096 : i64\n    %c8192_i64 = arith.constant 8192 : i64\n    %c9216_i64 = arith.'
    'constant 9216 : i64\n    %c10240_i64 = arith.constant 10240 : i64\n    %c11264_i64 = arith.constant 11264 : '
    'i64\n    %c12288_i64 = arith.constant 12288 : i64\n\n    %ub_f8e4_in_u8 = pto.castptr %c0_i64 : i64 -> !pto.'
    'ptr<ui8, ub>\n    %ub_f8e5_in_u8 = pto.castptr %c1024_i64 : i64 -> !pto.ptr<ui8, ub>\n    %ub_hif8_in_u8 = p'
    'to.castptr %c2048_i64 : i64 -> !pto.ptr<ui8, ub>\n    %ub_f4e1_in_u8 = pto.castptr %c3072_i64 : i64 -> !pto.'
    'ptr<ui8, ub>\n    %ub_f4e2_in_u8 = pto.castptr %c4096_i64 : i64 -> !pto.ptr<ui8, ub>\n    %ub_f8e4_out_u8 = '
    'pto.castptr %c8192_i64 : i64 -> !pto.ptr<ui8, ub>\n    %ub_f8e5_out_u8 = pto.castptr %c9216_i64 : i64 -> !pt'
    'o.ptr<ui8, ub>\n    %ub_hif8_out_u8 = pto.castptr %c10240_i64 : i64 -> !pto.ptr<ui8, ub>\n    %ub_f4e1_out_u'
    '8 = pto.castptr %c11264_i64 : i64 -> !pto.ptr<ui8, ub>\n    %ub_f4e2_out_u8 = pto.castptr %c12288_i64 : i64 '
    '-> !pto.ptr<ui8, ub>\n\n    %ub_f8e4_in = pto.castptr %c0_i64 : i64 -> !pto.ptr<f32, ub>\n    %ub_f8e5_in = '
    'pto.castptr %c1024_i64 : i64 -> !pto.ptr<f32, ub>\n    %ub_hif8_in = pto.castptr %c2048_i64 : i64 -> !pto.pt'
    'r<f32, ub>\n    %ub_f4e1_in = pto.castptr %c3072_i64 : i64 -> !pto.ptr<bf16, ub>\n    %ub_f4e2_in = pto.cast'
    'ptr %c4096_i64 : i64 -> !pto.ptr<bf16, ub>\n    %ub_f8e4_out = pto.castptr %c8192_i64 : i64 -> !pto.ptr<f8E4'
    'M3FN, ub>\n    %ub_f8e5_out = pto.castptr %c9216_i64 : i64 -> !pto.ptr<f8E5M2, ub>\n    %ub_hif8_out = pto.c'
    'astptr %c10240_i64 : i64 -> !pto.ptr<!pto.hif8, ub>\n    %ub_f4e1_out = pto.castptr %c11264_i64 : i64 -> !pt'
    'o.ptr<!pto.f4E1M2x2, ub>\n    %ub_f4e2_out = pto.castptr %c12288_i64 : i64 -> !pto.ptr<!pto.f4E2M1x2, ub>\n'
    '\n    pto.mte_gm_ub %f8e4_in, %ub_f8e4_in_u8, %c0_i64, %c1024_i64\n      nburst(%c1_i64, %c1024_i64, %c1024_'
    'i64)\n      : !pto.ptr<ui8, gm>, !pto.ptr<ui8, ub>, i64, i64, i64, i64, i64\n    pto.mte_gm_ub %f8e5_in, %ub'
    '_f8e5_in_u8, %c0_i64, %c1024_i64\n      nburst(%c1_i64, %c1024_i64, %c1024_i64)\n      : !pto.ptr<ui8, gm>, '
    '!pto.ptr<ui8, ub>, i64, i64, i64, i64, i64\n    pto.mte_gm_ub %hif8_in, %ub_hif8_in_u8, %c0_i64, %c1024_i64'
    '\n      nburst(%c1_i64, %c1024_i64, %c1024_i64)\n      : !pto.ptr<ui8, gm>, !pto.ptr<ui8, ub>, i64, i64, i64'
    ', i64, i64\n    pto.mte_gm_ub %f4e1_in, %ub_f4e1_in_u8, %c0_i64, %c1024_i64\n      nburst(%c1_i64, %c1024_i6'
    '4, %c1024_i64)\n      : !pto.ptr<ui8, gm>, !pto.ptr<ui8, ub>, i64, i64, i64, i64, i64\n    pto.mte_gm_ub %f4'
    'e2_in, %ub_f4e2_in_u8, %c0_i64, %c1024_i64\n      nburst(%c1_i64, %c1024_i64, %c1024_i64)\n      : !pto.ptr<'
    'ui8, gm>, !pto.ptr<ui8, ub>, i64, i64, i64, i64, i64\n\n    pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]'
    '\n    pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n\n    pto.vecscope {\n      %mask_b8 = pto.pset_b8 '
    '"PAT_ALL" : !pto.mask<b8>\n      %mask_b16 = pto.pset_b16 "PAT_ALL" : !pto.mask<b16>\n      %mask_b32 = pto.'
    'pset_b32 "PAT_ALL" : !pto.mask<b32>\n\n      %f8e4_src_p0 = pto.vlds %ub_f8e4_in[%c0] : !pto.ptr<f32, ub>, index ->'
    ' !pto.vreg<64xf32>\n      %f8e4_p0 = pto.vcvt %f8e4_src_p0, %mask_b32 {rnd = "R", sat = "SAT", part = "P0"} '
    ': !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<256xf8E4M3FN>\n      pto.vsts %f8e4_p0, %ub_f8e4_out[%c0], '
    '%mask_b8 : !pto.vreg<256xf8E4M3FN>, !pto.ptr<f8E4M3FN, ub>, index, !pto.mask<b8>\n      %f8e4_src_p1 = pto.vlds %ub'
    '_f8e4_in[%c64] : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>\n      %f8e4_p1 = pto.vcvt %f8e4_src_p1, %mask_b32 {'
    'rnd = "R", sat = "SAT", part = "P1"} : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<256xf8E4M3FN>\n      p'
    'to.vsts %f8e4_p1, %ub_f8e4_out[%c256], %mask_b8 : !pto.vreg<256xf8E4M3FN>, !pto.ptr<f8E4M3FN, ub>, index, !pto.mask'
    '<b8>\n      %f8e4_src_p2 = pto.vlds %ub_f8e4_in[%c128] : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>\n      %f8e4'
    '_p2 = pto.vcvt %f8e4_src_p2, %mask_b32 {rnd = "R", sat = "SAT", part = "P2"} : !pto.vreg<64xf32>, !pto.mask<'
    'b32> -> !pto.vreg<256xf8E4M3FN>\n      pto.vsts %f8e4_p2, %ub_f8e4_out[%c512], %mask_b8 : !pto.vreg<256xf8E4'
    'M3FN>, !pto.ptr<f8E4M3FN, ub>, index, !pto.mask<b8>\n      %f8e4_src_p3 = pto.vlds %ub_f8e4_in[%c192] : !pto.ptr<f3'
    '2, ub>, index -> !pto.vreg<64xf32>\n      %f8e4_p3 = pto.vcvt %f8e4_src_p3, %mask_b32 {rnd = "R", sat = "SAT", part'
    ' = "P3"} : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<256xf8E4M3FN>\n      pto.vsts %f8e4_p3, %ub_f8e4_o'
    'ut[%c768], %mask_b8 : !pto.vreg<256xf8E4M3FN>, !pto.ptr<f8E4M3FN, ub>, index, !pto.mask<b8>\n\n      %f8e5_src_p0 ='
    ' pto.vlds %ub_f8e5_in[%c0] : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>\n      %f8e5_p0 = pto.vcvt %f8e5_src_p0,'
    ' %mask_b32 {rnd = "R", sat = "SAT", part = "P0"} : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<256xf8E5M2'
    '>\n      pto.vsts %f8e5_p0, %ub_f8e5_out[%c0], %mask_b8 : !pto.vreg<256xf8E5M2>, !pto.ptr<f8E5M2, ub>, index, !pto.'
    'mask<b8>\n      %f8e5_src_p1 = pto.vlds %ub_f8e5_in[%c64] : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>\n      %f'
    '8e5_p1 = pto.vcvt %f8e5_src_p1, %mask_b32 {rnd = "R", sat = "SAT", part = "P1"} : !pto.vreg<64xf32>, !pto.ma'
    'sk<b32> -> !pto.vreg<256xf8E5M2>\n      pto.vsts %f8e5_p1, %ub_f8e5_out[%c256], %mask_b8 : !pto.vreg<256xf8E'
    '5M2>, !pto.ptr<f8E5M2, ub>, index, !pto.mask<b8>\n      %f8e5_src_p2 = pto.vlds %ub_f8e5_in[%c128] : !pto.ptr<f32, '
    'ub>, index -> !pto.vreg<64xf32>\n      %f8e5_p2 = pto.vcvt %f8e5_src_p2, %mask_b32 {rnd = "R", sat = "SAT", part = '
    '"P2"} : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<256xf8E5M2>\n      pto.vsts %f8e5_p2, %ub_f8e5_out[%c'
    '512], %mask_b8 : !pto.vreg<256xf8E5M2>, !pto.ptr<f8E5M2, ub>, index, !pto.mask<b8>\n      %f8e5_src_p3 = pto.vlds %'
    'ub_f8e5_in[%c192] : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>\n      %f8e5_p3 = pto.vcvt %f8e5_src_p3, %mask_b3'
    '2 {rnd = "R", sat = "SAT", part = "P3"} : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<256xf8E5M2>\n      '
    'pto.vsts %f8e5_p3, %ub_f8e5_out[%c768], %mask_b8 : !pto.vreg<256xf8E5M2>, !pto.ptr<f8E5M2, ub>, index, !pto.mask<b8'
    '>\n\n      %hif8_src_p0 = pto.vlds %ub_hif8_in[%c0] : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>\n      %hif8_p0'
    ' = pto.vcvt %hif8_src_p0, %mask_b32 {rnd = "A", sat = "NOSAT", part = "P0"} : !pto.vreg<64xf32>, !pto.mask<b'
    '32> -> !pto.vreg<256x!pto.hif8>\n      pto.vsts %hif8_p0, %ub_hif8_out[%c0], %mask_b8 : !pto.vreg<256x!pto.h'
    'if8>, !pto.ptr<!pto.hif8, ub>, index, !pto.mask<b8>\n      %hif8_src_p1 = pto.vlds %ub_hif8_in[%c64] : !pto.ptr<f32'
    ', ub>, index -> !pto.vreg<64xf32>\n      %hif8_p1 = pto.vcvt %hif8_src_p1, %mask_b32 {rnd = "A", sat = "NOSAT", par'
    't = "P1"} : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<256x!pto.hif8>\n      pto.vsts %hif8_p1, %ub_hif8'
    '_out[%c256], %mask_b8 : !pto.vreg<256x!pto.hif8>, !pto.ptr<!pto.hif8, ub>, index, !pto.mask<b8>\n      %hif8_src_p2'
    ' = pto.vlds %ub_hif8_in[%c128] : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>\n      %hif8_p2 = pto.vcvt %hif8_src'
    '_p2, %mask_b32 {rnd = "A", sat = "NOSAT", part = "P2"} : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<256x'
    '!pto.hif8>\n      pto.vsts %hif8_p2, %ub_hif8_out[%c512], %mask_b8 : !pto.vreg<256x!pto.hif8>, !pto.ptr<!pto'
    '.hif8, ub>, index, !pto.mask<b8>\n      %hif8_src_p3 = pto.vlds %ub_hif8_in[%c192] : !pto.ptr<f32, ub>, index -> !pto.vreg'
    '<64xf32>\n      %hif8_p3 = pto.vcvt %hif8_src_p3, %mask_b32 {rnd = "A", sat = "NOSAT", part = "P3"} : !pto.v'
    'reg<64xf32>, !pto.mask<b32> -> !pto.vreg<256x!pto.hif8>\n      pto.vsts %hif8_p3, %ub_hif8_out[%c768], %mask'
    '_b8 : !pto.vreg<256x!pto.hif8>, !pto.ptr<!pto.hif8, ub>, index, !pto.mask<b8>\n\n      %f4e1_src_p0 = pto.vlds %ub_'
    'f4e1_in[%c0] : !pto.ptr<bf16, ub>, index -> !pto.vreg<128xbf16>\n      %f4e1_p0 = pto.vcvt %f4e1_src_p0, %mask_b16 '
    '{rnd = "R", part = "P0"} : !pto.vreg<128xbf16>, !pto.mask<b16> -> !pto.vreg<256x!pto.f4E1M2x2>\n      pto.vs'
    'ts %f4e1_p0, %ub_f4e1_out[%c0], %mask_b8 : !pto.vreg<256x!pto.f4E1M2x2>, !pto.ptr<!pto.f4E1M2x2, ub>, index, !pto.m'
    'ask<b8>\n      %f4e1_src_p1 = pto.vlds %ub_f4e1_in[%c128] : !pto.ptr<bf16, ub>, index -> !pto.vreg<128xbf16>\n     '
    ' %f4e1_p1 = pto.vcvt %f4e1_src_p1, %mask_b16 {rnd = "R", part = "P1"} : !pto.vreg<128xbf16>, !pto.mask<b16> '
    '-> !pto.vreg<256x!pto.f4E1M2x2>\n      pto.vsts %f4e1_p1, %ub_f4e1_out[%c256], %mask_b8 : !pto.vreg<256x!pto'
    '.f4E1M2x2>, !pto.ptr<!pto.f4E1M2x2, ub>, index, !pto.mask<b8>\n      %f4e1_src_p2 = pto.vlds %ub_f4e1_in[%c256] : !'
    'pto.ptr<bf16, ub>, index -> !pto.vreg<128xbf16>\n      %f4e1_p2 = pto.vcvt %f4e1_src_p2, %mask_b16 {rnd = "R", part'
    ' = "P2"} : !pto.vreg<128xbf16>, !pto.mask<b16> -> !pto.vreg<256x!pto.f4E1M2x2>\n      pto.vsts %f4e1_p2, %ub'
    '_f4e1_out[%c512], %mask_b8 : !pto.vreg<256x!pto.f4E1M2x2>, !pto.ptr<!pto.f4E1M2x2, ub>, index, !pto.mask<b8>\n     '
    ' %f4e1_src_p3 = pto.vlds %ub_f4e1_in[%c384] : !pto.ptr<bf16, ub>, index -> !pto.vreg<128xbf16>\n      %f4e1_p3 = pt'
    'o.vcvt %f4e1_src_p3, %mask_b16 {rnd = "R", part = "P3"} : !pto.vreg<128xbf16>, !pto.mask<b16> -> !pto.vreg<2'
    '56x!pto.f4E1M2x2>\n      pto.vsts %f4e1_p3, %ub_f4e1_out[%c768], %mask_b8 : !pto.vreg<256x!pto.f4E1M2x2>, !p'
    'to.ptr<!pto.f4E1M2x2, ub>, index, !pto.mask<b8>\n\n      %f4e2_src_p0 = pto.vlds %ub_f4e2_in[%c0] : !pto.ptr<bf16, '
    'ub>, index -> !pto.vreg<128xbf16>\n      %f4e2_p0 = pto.vcvt %f4e2_src_p0, %mask_b16 {rnd = "R", part = "P0"} : !pt'
    'o.vreg<128xbf16>, !pto.mask<b16> -> !pto.vreg<256x!pto.f4E2M1x2>\n      pto.vsts %f4e2_p0, %ub_f4e2_out[%c0]'
    ', %mask_b8 : !pto.vreg<256x!pto.f4E2M1x2>, !pto.ptr<!pto.f4E2M1x2, ub>, index, !pto.mask<b8>\n      %f4e2_src_p1 = '
    'pto.vlds %ub_f4e2_in[%c128] : !pto.ptr<bf16, ub>, index -> !pto.vreg<128xbf16>\n      %f4e2_p1 = pto.vcvt %f4e2_src'
    '_p1, %mask_b16 {rnd = "R", part = "P1"} : !pto.vreg<128xbf16>, !pto.mask<b16> -> !pto.vreg<256x!pto.f4E2M1x2'
    '>\n      pto.vsts %f4e2_p1, %ub_f4e2_out[%c256], %mask_b8 : !pto.vreg<256x!pto.f4E2M1x2>, !pto.ptr<!pto.f4E2'
    'M1x2, ub>, index, !pto.mask<b8>\n      %f4e2_src_p2 = pto.vlds %ub_f4e2_in[%c256] : !pto.ptr<bf16, ub>, index -> !pto.vreg'
    '<128xbf16>\n      %f4e2_p2 = pto.vcvt %f4e2_src_p2, %mask_b16 {rnd = "R", part = "P2"} : !pto.vreg<128xbf16>'
    ', !pto.mask<b16> -> !pto.vreg<256x!pto.f4E2M1x2>\n      pto.vsts %f4e2_p2, %ub_f4e2_out[%c512], %mask_b8 : !'
    'pto.vreg<256x!pto.f4E2M1x2>, !pto.ptr<!pto.f4E2M1x2, ub>, index, !pto.mask<b8>\n      %f4e2_src_p3 = pto.vlds %ub_f'
    '4e2_in[%c384] : !pto.ptr<bf16, ub>, index -> !pto.vreg<128xbf16>\n      %f4e2_p3 = pto.vcvt %f4e2_src_p3, %mask_b16'
    ' {rnd = "R", part = "P3"} : !pto.vreg<128xbf16>, !pto.mask<b16> -> !pto.vreg<256x!pto.f4E2M1x2>\n      pto.v'
    'sts %f4e2_p3, %ub_f4e2_out[%c768], %mask_b8 : !pto.vreg<256x!pto.f4E2M1x2>, !pto.ptr<!pto.f4E2M1x2, ub>, index, !pt'
    'o.mask<b8>\n    }\n\n    pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n    pto.wait_flag["PIPE_V", "PIPE'
    '_MTE3", "EVENT_ID0"]\n    pto.mte_ub_gm %ub_f8e4_out_u8, %f8e4_out, %c1024_i64\n      nburst(%c1_i64, %c1024'
    '_i64, %c1024_i64)\n      : !pto.ptr<ui8, ub>, !pto.ptr<ui8, gm>, i64, i64, i64, i64\n    pto.mte_ub_gm %ub_f'
    '8e5_out_u8, %f8e5_out, %c1024_i64\n      nburst(%c1_i64, %c1024_i64, %c1024_i64)\n      : !pto.ptr<ui8, ub>,'
    ' !pto.ptr<ui8, gm>, i64, i64, i64, i64\n    pto.mte_ub_gm %ub_hif8_out_u8, %hif8_out, %c1024_i64\n      nbur'
    'st(%c1_i64, %c1024_i64, %c1024_i64)\n      : !pto.ptr<ui8, ub>, !pto.ptr<ui8, gm>, i64, i64, i64, i64\n    p'
    'to.mte_ub_gm %ub_f4e1_out_u8, %f4e1_out, %c1024_i64\n      nburst(%c1_i64, %c1024_i64, %c1024_i64)\n      : '
    '!pto.ptr<ui8, ub>, !pto.ptr<ui8, gm>, i64, i64, i64, i64\n    pto.mte_ub_gm %ub_f4e2_out_u8, %f4e2_out, %c10'
    '24_i64\n      nburst(%c1_i64, %c1024_i64, %c1024_i64)\n      : !pto.ptr<ui8, ub>, !pto.ptr<ui8, gm>, i64, i6'
    '4, i64, i64\n    pto.barrier #pto.pipe<PIPE_ALL>\n    return\n  }\n\n  func.func @vcvt_low_precision_special'
    '_kernel(%f8e4_in: !pto.ptr<ui8, gm>,\n                                               %f8e5_in: !pto.ptr<ui8,'
    ' gm>,\n                                               %hif8_in: !pto.ptr<ui8, gm>,\n                        '
    '                       %f4e1_in: !pto.ptr<ui8, gm>,\n                                               %f4e2_in'
    ': !pto.ptr<ui8, gm>,\n                                               %f8e4_out: !pto.ptr<ui8, gm>,\n        '
    '                                       %f8e5_out: !pto.ptr<ui8, gm>,\n                                      '
    '         %hif8_out: !pto.ptr<ui8, gm>,\n                                               %f4e1_out: !pto.ptr<u'
    'i8, gm>,\n                                               %f4e2_out: !pto.ptr<ui8, gm>) attributes {pto.kerne'
    'l} {\n    %c0 = arith.constant 0 : index\n    %c64 = arith.constant 64 : index\n    %c128 = arith.constant 1'
    '28 : index\n    %c192 = arith.constant 192 : index\n    %c256 = arith.constant 256 : index\n    %c384 = arit'
    'h.constant 384 : index\n    %c512 = arith.constant 512 : index\n    %c768 = arith.constant 768 : index\n    '
    '%c0_i64 = arith.constant 0 : i64\n    %c1_i64 = arith.constant 1 : i64\n    %c1024_i64 = arith.constant 1024'
    ' : i64\n    %c2048_i64 = arith.constant 2048 : i64\n    %c3072_i64 = arith.constant 3072 : i64\n    %c4096_i'
    '64 = arith.constant 4096 : i64\n    %c8192_i64 = arith.constant 8192 : i64\n    %c9216_i64 = arith.constant '
    '9216 : i64\n    %c10240_i64 = arith.constant 10240 : i64\n    %c11264_i64 = arith.constant 11264 : i64\n    '
    '%c12288_i64 = arith.constant 12288 : i64\n\n    %ub_f8e4_in_u8 = pto.castptr %c0_i64 : i64 -> !pto.ptr<ui8, '
    'ub>\n    %ub_f8e5_in_u8 = pto.castptr %c1024_i64 : i64 -> !pto.ptr<ui8, ub>\n    %ub_hif8_in_u8 = pto.castpt'
    'r %c2048_i64 : i64 -> !pto.ptr<ui8, ub>\n    %ub_f4e1_in_u8 = pto.castptr %c3072_i64 : i64 -> !pto.ptr<ui8, '
    'ub>\n    %ub_f4e2_in_u8 = pto.castptr %c4096_i64 : i64 -> !pto.ptr<ui8, ub>\n    %ub_f8e4_out_u8 = pto.castp'
    'tr %c8192_i64 : i64 -> !pto.ptr<ui8, ub>\n    %ub_f8e5_out_u8 = pto.castptr %c9216_i64 : i64 -> !pto.ptr<ui8'
    ', ub>\n    %ub_hif8_out_u8 = pto.castptr %c10240_i64 : i64 -> !pto.ptr<ui8, ub>\n    %ub_f4e1_out_u8 = pto.c'
    'astptr %c11264_i64 : i64 -> !pto.ptr<ui8, ub>\n    %ub_f4e2_out_u8 = pto.castptr %c12288_i64 : i64 -> !pto.p'
    'tr<ui8, ub>\n\n    %ub_f8e4_in = pto.castptr %c0_i64 : i64 -> !pto.ptr<f8E4M3FN, ub>\n    %ub_f8e5_in = pto.'
    'castptr %c1024_i64 : i64 -> !pto.ptr<f8E5M2, ub>\n    %ub_hif8_in = pto.castptr %c2048_i64 : i64 -> !pto.ptr'
    '<!pto.hif8, ub>\n    %ub_f4e1_in = pto.castptr %c3072_i64 : i64 -> !pto.ptr<!pto.f4E1M2x2, ub>\n    %ub_f4e2'
    '_in = pto.castptr %c4096_i64 : i64 -> !pto.ptr<!pto.f4E2M1x2, ub>\n    %ub_f8e4_out = pto.castptr %c8192_i64'
    ' : i64 -> !pto.ptr<f32, ub>\n    %ub_f8e5_out = pto.castptr %c9216_i64 : i64 -> !pto.ptr<f32, ub>\n    %ub_h'
    'if8_out = pto.castptr %c10240_i64 : i64 -> !pto.ptr<f32, ub>\n    %ub_f4e1_out = pto.castptr %c11264_i64 : i'
    '64 -> !pto.ptr<bf16, ub>\n    %ub_f4e2_out = pto.castptr %c12288_i64 : i64 -> !pto.ptr<bf16, ub>\n\n    pto.'
    'mte_gm_ub %f8e4_in, %ub_f8e4_in_u8, %c0_i64, %c1024_i64\n      nburst(%c1_i64, %c1024_i64, %c1024_i64)\n    '
    '  : !pto.ptr<ui8, gm>, !pto.ptr<ui8, ub>, i64, i64, i64, i64, i64\n    pto.mte_gm_ub %f8e5_in, %ub_f8e5_in_u'
    '8, %c0_i64, %c1024_i64\n      nburst(%c1_i64, %c1024_i64, %c1024_i64)\n      : !pto.ptr<ui8, gm>, !pto.ptr<u'
    'i8, ub>, i64, i64, i64, i64, i64\n    pto.mte_gm_ub %hif8_in, %ub_hif8_in_u8, %c0_i64, %c1024_i64\n      nbu'
    'rst(%c1_i64, %c1024_i64, %c1024_i64)\n      : !pto.ptr<ui8, gm>, !pto.ptr<ui8, ub>, i64, i64, i64, i64, i64'
    '\n    pto.mte_gm_ub %f4e1_in, %ub_f4e1_in_u8, %c0_i64, %c1024_i64\n      nburst(%c1_i64, %c1024_i64, %c1024_'
    'i64)\n      : !pto.ptr<ui8, gm>, !pto.ptr<ui8, ub>, i64, i64, i64, i64, i64\n    pto.mte_gm_ub %f4e2_in, %ub'
    '_f4e2_in_u8, %c0_i64, %c1024_i64\n      nburst(%c1_i64, %c1024_i64, %c1024_i64)\n      : !pto.ptr<ui8, gm>, '
    '!pto.ptr<ui8, ub>, i64, i64, i64, i64, i64\n\n    pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n    pto.'
    'wait_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n\n    pto.vecscope {\n      %mask_b8 = pto.pset_b8 "PAT_ALL" '
    ': !pto.mask<b8>\n      %mask_b16 = pto.pset_b16 "PAT_ALL" : !pto.mask<b16>\n      %mask_b32 = pto.pset_b32 "'
    'PAT_ALL" : !pto.mask<b32>\n\n      %f8e4_p0 = pto.vlds %ub_f8e4_in[%c0] : !pto.ptr<f8E4M3FN, ub>, index -> !pto.vre'
    'g<256xf8E4M3FN>\n      %f8e4_out_p0 = pto.vcvt %f8e4_p0, %mask_b8 {part = "P0"} : !pto.vreg<256xf8E4M3FN>, !'
    'pto.mask<b8> -> !pto.vreg<64xf32>\n      pto.vsts %f8e4_out_p0, %ub_f8e4_out[%c0], %mask_b32 : !pto.vreg<64x'
    'f32>, !pto.ptr<f32, ub>, index, !pto.mask<b32>\n      %f8e4_p1 = pto.vlds %ub_f8e4_in[%c256] : !pto.ptr<f8E4M3FN, u'
    'b>, index -> !pto.vreg<256xf8E4M3FN>\n      %f8e4_out_p1 = pto.vcvt %f8e4_p1, %mask_b8 {part = "P1"} : !pto.vreg<25'
    '6xf8E4M3FN>, !pto.mask<b8> -> !pto.vreg<64xf32>\n      pto.vsts %f8e4_out_p1, %ub_f8e4_out[%c64], %mask_b32 '
    ': !pto.vreg<64xf32>, !pto.ptr<f32, ub>, index, !pto.mask<b32>\n      %f8e4_p2 = pto.vlds %ub_f8e4_in[%c512] : !pto.'
    'ptr<f8E4M3FN, ub>, index -> !pto.vreg<256xf8E4M3FN>\n      %f8e4_out_p2 = pto.vcvt %f8e4_p2, %mask_b8 {part = "P2"}'
    ' : !pto.vreg<256xf8E4M3FN>, !pto.mask<b8> -> !pto.vreg<64xf32>\n      pto.vsts %f8e4_out_p2, %ub_f8e4_out[%c'
    '128], %mask_b32 : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, index, !pto.mask<b32>\n      %f8e4_p3 = pto.vlds %ub_f8e4_i'
    'n[%c768] : !pto.ptr<f8E4M3FN, ub>, index -> !pto.vreg<256xf8E4M3FN>\n      %f8e4_out_p3 = pto.vcvt %f8e4_p3, %mask_'
    'b8 {part = "P3"} : !pto.vreg<256xf8E4M3FN>, !pto.mask<b8> -> !pto.vreg<64xf32>\n      pto.vsts %f8e4_out_p3,'
    ' %ub_f8e4_out[%c192], %mask_b32 : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, index, !pto.mask<b32>\n\n      %f8e5_p0 = p'
    'to.vlds %ub_f8e5_in[%c0] : !pto.ptr<f8E5M2, ub>, index -> !pto.vreg<256xf8E5M2>\n      %f8e5_out_p0 = pto.vcvt %f8e'
    '5_p0, %mask_b8 {part = "P0"} : !pto.vreg<256xf8E5M2>, !pto.mask<b8> -> !pto.vreg<64xf32>\n      pto.vsts %f8'
    'e5_out_p0, %ub_f8e5_out[%c0], %mask_b32 : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, index, !pto.mask<b32>\n      %f8e5_'
    'p1 = pto.vlds %ub_f8e5_in[%c256] : !pto.ptr<f8E5M2, ub>, index -> !pto.vreg<256xf8E5M2>\n      %f8e5_out_p1 = pto.v'
    'cvt %f8e5_p1, %mask_b8 {part = "P1"} : !pto.vreg<256xf8E5M2>, !pto.mask<b8> -> !pto.vreg<64xf32>\n      pto.'
    'vsts %f8e5_out_p1, %ub_f8e5_out[%c64], %mask_b32 : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, index, !pto.mask<b32>\n   '
    '   %f8e5_p2 = pto.vlds %ub_f8e5_in[%c512] : !pto.ptr<f8E5M2, ub>, index -> !pto.vreg<256xf8E5M2>\n      %f8e5_out_p'
    '2 = pto.vcvt %f8e5_p2, %mask_b8 {part = "P2"} : !pto.vreg<256xf8E5M2>, !pto.mask<b8> -> !pto.vreg<64xf32>\n '
    '     pto.vsts %f8e5_out_p2, %ub_f8e5_out[%c128], %mask_b32 : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, index, !pto.mask'
    '<b32>\n      %f8e5_p3 = pto.vlds %ub_f8e5_in[%c768] : !pto.ptr<f8E5M2, ub>, index -> !pto.vreg<256xf8E5M2>\n      %'
    'f8e5_out_p3 = pto.vcvt %f8e5_p3, %mask_b8 {part = "P3"} : !pto.vreg<256xf8E5M2>, !pto.mask<b8> -> !pto.vreg<'
    '64xf32>\n      pto.vsts %f8e5_out_p3, %ub_f8e5_out[%c192], %mask_b32 : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, index,'
    ' !pto.mask<b32>\n\n      %hif8_p0 = pto.vlds %ub_hif8_in[%c0] : !pto.ptr<!pto.hif8, ub>, index -> !pto.vreg<256x!pt'
    'o.hif8>\n      %hif8_out_p0 = pto.vcvt %hif8_p0, %mask_b8 {part = "P0"} : !pto.vreg<256x!pto.hif8>, !pto.mas'
    'k<b8> -> !pto.vreg<64xf32>\n      pto.vsts %hif8_out_p0, %ub_hif8_out[%c0], %mask_b32 : !pto.vreg<64xf32>, !'
    'pto.ptr<f32, ub>, index, !pto.mask<b32>\n      %hif8_p1 = pto.vlds %ub_hif8_in[%c256] : !pto.ptr<!pto.hif8, ub>, index -> '
    '!pto.vreg<256x!pto.hif8>\n      %hif8_out_p1 = pto.vcvt %hif8_p1, %mask_b8 {part = "P1"} : !pto.vreg<256x!pt'
    'o.hif8>, !pto.mask<b8> -> !pto.vreg<64xf32>\n      pto.vsts %hif8_out_p1, %ub_hif8_out[%c64], %mask_b32 : !p'
    'to.vreg<64xf32>, !pto.ptr<f32, ub>, index, !pto.mask<b32>\n      %hif8_p2 = pto.vlds %ub_hif8_in[%c512] : !pto.ptr<'
    '!pto.hif8, ub>, index -> !pto.vreg<256x!pto.hif8>\n      %hif8_out_p2 = pto.vcvt %hif8_p2, %mask_b8 {part = "P2"} :'
    ' !pto.vreg<256x!pto.hif8>, !pto.mask<b8> -> !pto.vreg<64xf32>\n      pto.vsts %hif8_out_p2, %ub_hif8_out[%c1'
    '28], %mask_b32 : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, index, !pto.mask<b32>\n      %hif8_p3 = pto.vlds %ub_hif8_in'
    '[%c768] : !pto.ptr<!pto.hif8, ub>, index -> !pto.vreg<256x!pto.hif8>\n      %hif8_out_p3 = pto.vcvt %hif8_p3, %mask'
    '_b8 {part = "P3"} : !pto.vreg<256x!pto.hif8>, !pto.mask<b8> -> !pto.vreg<64xf32>\n      pto.vsts %hif8_out_p'
    '3, %ub_hif8_out[%c192], %mask_b32 : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, index, !pto.mask<b32>\n\n      %f4e1_p0 ='
    ' pto.vlds %ub_f4e1_in[%c0] : !pto.ptr<!pto.f4E1M2x2, ub>, index -> !pto.vreg<256x!pto.f4E1M2x2>\n      %f4e1_out_p0'
    ' = pto.vcvt %f4e1_p0, %mask_b8 {part = "P0"} : !pto.vreg<256x!pto.f4E1M2x2>, !pto.mask<b8> -> !pto.vreg<128x'
    'bf16>\n      pto.vsts %f4e1_out_p0, %ub_f4e1_out[%c0], %mask_b16 : !pto.vreg<128xbf16>, !pto.ptr<bf16, ub>, index, '
    '!pto.mask<b16>\n      %f4e1_p1 = pto.vlds %ub_f4e1_in[%c256] : !pto.ptr<!pto.f4E1M2x2, ub>, index -> !pto.vreg<256x'
    '!pto.f4E1M2x2>\n      %f4e1_out_p1 = pto.vcvt %f4e1_p1, %mask_b8 {part = "P1"} : !pto.vreg<256x!pto.f4E1M2x2'
    '>, !pto.mask<b8> -> !pto.vreg<128xbf16>\n      pto.vsts %f4e1_out_p1, %ub_f4e1_out[%c128], %mask_b16 : !pto.'
    'vreg<128xbf16>, !pto.ptr<bf16, ub>, index, !pto.mask<b16>\n      %f4e1_p2 = pto.vlds %ub_f4e1_in[%c512] : !pto.ptr<'
    '!pto.f4E1M2x2, ub>, index -> !pto.vreg<256x!pto.f4E1M2x2>\n      %f4e1_out_p2 = pto.vcvt %f4e1_p2, %mask_b8 {part ='
    ' "P2"} : !pto.vreg<256x!pto.f4E1M2x2>, !pto.mask<b8> -> !pto.vreg<128xbf16>\n      pto.vsts %f4e1_out_p2, %u'
    'b_f4e1_out[%c256], %mask_b16 : !pto.vreg<128xbf16>, !pto.ptr<bf16, ub>, index, !pto.mask<b16>\n      %f4e1_p3 = pto'
    '.vlds %ub_f4e1_in[%c768] : !pto.ptr<!pto.f4E1M2x2, ub>, index -> !pto.vreg<256x!pto.f4E1M2x2>\n      %f4e1_out_p3 ='
    ' pto.vcvt %f4e1_p3, %mask_b8 {part = "P3"} : !pto.vreg<256x!pto.f4E1M2x2>, !pto.mask<b8> -> !pto.vreg<128xbf'
    '16>\n      pto.vsts %f4e1_out_p3, %ub_f4e1_out[%c384], %mask_b16 : !pto.vreg<128xbf16>, !pto.ptr<bf16, ub>, index, '
    '!pto.mask<b16>\n\n      %f4e2_p0 = pto.vlds %ub_f4e2_in[%c0] : !pto.ptr<!pto.f4E2M1x2, ub>, index -> !pto.vreg<256x'
    '!pto.f4E2M1x2>\n      %f4e2_out_p0 = pto.vcvt %f4e2_p0, %mask_b8 {part = "P0"} : !pto.vreg<256x!pto.f4E2M1x2'
    '>, !pto.mask<b8> -> !pto.vreg<128xbf16>\n      pto.vsts %f4e2_out_p0, %ub_f4e2_out[%c0], %mask_b16 : !pto.vr'
    'eg<128xbf16>, !pto.ptr<bf16, ub>, index, !pto.mask<b16>\n      %f4e2_p1 = pto.vlds %ub_f4e2_in[%c256] : !pto.ptr<!p'
    'to.f4E2M1x2, ub>, index -> !pto.vreg<256x!pto.f4E2M1x2>\n      %f4e2_out_p1 = pto.vcvt %f4e2_p1, %mask_b8 {part = "'
    'P1"} : !pto.vreg<256x!pto.f4E2M1x2>, !pto.mask<b8> -> !pto.vreg<128xbf16>\n      pto.vsts %f4e2_out_p1, %ub_'
    'f4e2_out[%c128], %mask_b16 : !pto.vreg<128xbf16>, !pto.ptr<bf16, ub>, index, !pto.mask<b16>\n      %f4e2_p2 = pto.v'
    'lds %ub_f4e2_in[%c512] : !pto.ptr<!pto.f4E2M1x2, ub>, index -> !pto.vreg<256x!pto.f4E2M1x2>\n      %f4e2_out_p2 = p'
    'to.vcvt %f4e2_p2, %mask_b8 {part = "P2"} : !pto.vreg<256x!pto.f4E2M1x2>, !pto.mask<b8> -> !pto.vreg<128xbf16'
    '>\n      pto.vsts %f4e2_out_p2, %ub_f4e2_out[%c256], %mask_b16 : !pto.vreg<128xbf16>, !pto.ptr<bf16, ub>, index, !p'
    'to.mask<b16>\n      %f4e2_p3 = pto.vlds %ub_f4e2_in[%c768] : !pto.ptr<!pto.f4E2M1x2, ub>, index -> !pto.vreg<256x!p'
    'to.f4E2M1x2>\n      %f4e2_out_p3 = pto.vcvt %f4e2_p3, %mask_b8 {part = "P3"} : !pto.vreg<256x!pto.f4E2M1x2>,'
    ' !pto.mask<b8> -> !pto.vreg<128xbf16>\n      pto.vsts %f4e2_out_p3, %ub_f4e2_out[%c384], %mask_b16 : !pto.vr'
    'eg<128xbf16>, !pto.ptr<bf16, ub>, index, !pto.mask<b16>\n    }\n\n    pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID'
    '0"]\n    pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n    pto.mte_ub_gm %ub_f8e4_out_u8, %f8e4_out, %c'
    '1024_i64\n      nburst(%c1_i64, %c1024_i64, %c1024_i64)\n      : !pto.ptr<ui8, ub>, !pto.ptr<ui8, gm>, i64, '
    'i64, i64, i64\n    pto.mte_ub_gm %ub_f8e5_out_u8, %f8e5_out, %c1024_i64\n      nburst(%c1_i64, %c1024_i64, %'
    'c1024_i64)\n      : !pto.ptr<ui8, ub>, !pto.ptr<ui8, gm>, i64, i64, i64, i64\n    pto.mte_ub_gm %ub_hif8_out'
    '_u8, %hif8_out, %c1024_i64\n      nburst(%c1_i64, %c1024_i64, %c1024_i64)\n      : !pto.ptr<ui8, ub>, !pto.p'
    'tr<ui8, gm>, i64, i64, i64, i64\n    pto.mte_ub_gm %ub_f4e1_out_u8, %f4e1_out, %c1024_i64\n      nburst(%c1_'
    'i64, %c1024_i64, %c1024_i64)\n      : !pto.ptr<ui8, ub>, !pto.ptr<ui8, gm>, i64, i64, i64, i64\n    pto.mte_'
    'ub_gm %ub_f4e2_out_u8, %f4e2_out, %c1024_i64\n      nburst(%c1_i64, %c1024_i64, %c1024_i64)\n      : !pto.pt'
    'r<ui8, ub>, !pto.ptr<ui8, gm>, i64, i64, i64, i64\n    pto.barrier #pto.pipe<PIPE_ALL>\n    return\n  }\n\n}'
    '\n'
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

    BYTES = 1024


    def tiled(values):
        return np.resize(np.array(values, dtype=np.uint8), BYTES)


    def bits_to_f32(bits):
        return np.array([bits], dtype=np.uint32).view(np.float32)[0]


    def f32_to_bits(value):
        return np.array([value], dtype=np.float32).view(np.uint32)[0]


    def f32_to_bf16_bits(value):
        bits = int(f32_to_bits(value))
        lsb = (bits >> 16) & 1
        rounded = bits + 0x7FFF + lsb
        return np.uint16((rounded >> 16) & 0xFFFF)


    def decode_f8e4m3fn(byte):
        sign = -1.0 if byte & 0x80 else 1.0
        exp = (byte >> 3) & 0x0F
        mant = byte & 0x07
        if byte in (0x7F, 0xFF):
            return np.float32(np.nan)
        if exp == 0:
            return np.float32(sign * (mant / 8.0) * (2.0 ** -6))
        return np.float32(sign * (1.0 + mant / 8.0) * (2.0 ** (exp - 7)))


    def decode_f8e5m2(byte):
        sign = -1.0 if byte & 0x80 else 1.0
        exp = (byte >> 2) & 0x1F
        mant = byte & 0x03
        if exp == 0x1F:
            if mant == 0:
                return np.float32(sign * np.inf)
            return np.float32(np.nan)
        if exp == 0:
            return np.float32(sign * (mant / 4.0) * (2.0 ** -14))
        return np.float32(sign * (1.0 + mant / 4.0) * (2.0 ** (exp - 15)))


    def fp32_constructor(sign, exp, mant):
        return ((sign & 1) << 31) | ((exp & 0xFF) << 23) | (mant & 0x7FFFFF)


    def decode_hif8(byte):
        if byte == 0x00:
            return bits_to_f32(0x00000000)
        if byte == 0x80:
            return np.float32(np.nan)
        if byte == 0x6F:
            return bits_to_f32(0x7F800000)
        if byte == 0xEF:
            return bits_to_f32(0xFF800000)

        input_sign = (byte >> 7) & 0x01
        bit6 = (byte >> 6) & 0x01
        bit5 = (byte >> 5) & 0x01
        bit4 = (byte >> 4) & 0x01
        bit3 = (byte >> 3) & 0x01
        if bit6 == 0 and bit5 == 0 and bit4 == 0 and bit3 == 0:
            return bits_to_f32(fp32_constructor(input_sign, (byte & 0x7) - 23 + 127, 0))

        if bit6 == 0:
            if bit5 == 0:
                if bit4 == 0:
                    exp_width = 0
                    d_width = 4
                else:
                    exp_width = 1
                    d_width = 3
            else:
                exp_width = 2
                d_width = 2
        else:
            exp_width = 3 if bit5 == 0 else 4
            d_width = 2
        man_width = 8 - d_width - exp_width - 1

        exp_mask = (1 << exp_width) - 1
        exp = 0
        if exp_width != 0:
            exp = ((byte >> man_width) & exp_mask) | (1 << (exp_width - 1))
            exp_msb = (byte >> (man_width + exp_width - 1)) & 0x1
            if exp_msb != 0:
                exp = -exp
        exp += 127

        man_mask = (1 << man_width) - 1
        mant = (byte & man_mask) << (23 - man_width)
        return bits_to_f32(fp32_constructor(input_sign, exp, mant))


    FP4E1M2_TO_BF16 = np.array(
        [
            0x0000, 0x3E80, 0x3F00, 0x3F40, 0x3F80, 0x3FA0, 0x3FC0, 0x3FE0,
            0x8000, 0xBE80, 0xBF00, 0xBF40, 0xBF80, 0xBFA0, 0xBFC0, 0xBFE0,
        ],
        dtype=np.uint16,
    )

    FP4E2M1_TO_BF16 = np.array(
        [
            0x0000, 0x3F00, 0x3F80, 0x3FC0, 0x4000, 0x4040, 0x4080, 0x40C0,
            0x8000, 0xBF00, 0xBF80, 0xBFC0, 0xC000, 0xC040, 0xC080, 0xC0C0,
        ],
        dtype=np.uint16,
    )


    def expected_f8(data, decoder):
        expected = np.zeros(256, dtype=np.float32)
        for part in range(4):
            start = part * 256
            selected = data[start + part : start + 256 : 4]
            values = [decoder(int(byte)) for byte in selected]
            expected[part * 64 : (part + 1) * 64] = np.array(values, dtype=np.float32)
        return expected


    def expected_f4(data, table):
        expected = np.zeros(512, dtype=np.uint16)
        for part in range(4):
            start = part * 256
            selected = data[start + part : start + 256 : 4]
            values = []
            for byte in selected:
                values.append(table[int(byte) & 0x0F])
                values.append(table[(int(byte) >> 4) & 0x0F])
            expected[part * 128 : (part + 1) * 128] = np.array(values, dtype=np.uint16)
        return expected


    def write_f32_quantize_case(output_dir, input_name, golden_name, pairs):
        values = np.resize(np.array([value for value, _ in pairs], dtype=np.float32), 64)
        expected_bytes = np.resize(np.array([byte for _, byte in pairs], dtype=np.uint8), 64)
        input_values = np.tile(values, 4)
        expected = np.zeros(BYTES, dtype=np.uint8)
        for part in range(4):
            start = part * 256
            expected[start + part : start + 256 : 4] = expected_bytes
        _files[input_name] = input_values
        _files[golden_name] = expected


    def write_bf16_quantize_case(output_dir, input_name, golden_name, nibble_pairs):
        values = np.resize(np.array([bits for bits, _ in nibble_pairs], dtype=np.uint16), 128)
        nibbles = np.resize(np.array([nibble for _, nibble in nibble_pairs], dtype=np.uint8), 128)
        packed = np.array([nibbles[i] | (nibbles[i + 1] << 4) for i in range(0, 128, 2)], dtype=np.uint8)
        input_values = np.tile(values, 4)
        expected = np.zeros(BYTES, dtype=np.uint8)
        for part in range(4):
            start = part * 256
            expected[start + part : start + 256 : 4] = packed
        _files[input_name] = input_values
        _files[golden_name] = expected


    def f8e4_quantize_pairs():
        exact = [0x00, 0x80, 0x01, 0x81, 0x07, 0x87, 0x08, 0x88, 0x38, 0xB8, 0x3C, 0xBC, 0x7E, 0xFE]
        pairs = [(decode_f8e4m3fn(byte), byte) for byte in exact]
        # This kernel uses #sat=1. In V300 instruction-controlled saturation,
        # infinities/overflow clamp to max finite and NaN is saturated to zero.
        pairs.extend(
            [
                (np.float32(np.inf), 0x7E),
                (np.float32(-np.inf), 0xFE),
                (np.float32(np.nan), 0x00),
                (np.float32(1000.0), 0x7E),
                (np.float32(-1000.0), 0xFE),
                (np.float32(1.0625), 0x38),
                (np.float32(1.1875), 0x3A),
                (np.float32(-1.0625), 0xB8),
                (np.float32(-1.1875), 0xBA),
            ]
        )
        return pairs


    def f8e5_quantize_pairs():
        exact = [0x00, 0x80, 0x01, 0x81, 0x03, 0x83, 0x04, 0x84, 0x3C, 0xBC, 0x40, 0xC0, 0x7B, 0xFB]
        pairs = [(decode_f8e5m2(byte), byte) for byte in exact]
        # This kernel uses #sat=1. In V300 instruction-controlled saturation,
        # infinities/overflow clamp to max finite and NaN is saturated to zero.
        pairs.extend(
            [
                (np.float32(np.inf), 0x7B),
                (np.float32(-np.inf), 0xFB),
                (np.float32(np.nan), 0x00),
                (np.float32(1.0e10), 0x7B),
                (np.float32(-1.0e10), 0xFB),
                (np.float32(1.125), 0x3C),
                (np.float32(1.375), 0x3E),
                (np.float32(-1.125), 0xBC),
                (np.float32(-1.375), 0xBE),
            ]
        )
        return pairs


    def hif8_quantize_pairs():
        exact = [
            0x00, 0x80, 0x6F, 0xEF, 0x01, 0x81, 0x07, 0x87,
            0x08, 0x88, 0x10, 0x90, 0x18, 0x98, 0x20, 0xA0,
            0x40, 0xC0, 0x50, 0xD0, 0x60, 0xE0, 0x70, 0xF0,
        ]
        return [(decode_hif8(byte), byte) for byte in exact]


    def f4e1_quantize_pairs():
        pairs = [(bits, nibble) for nibble, bits in enumerate(FP4E1M2_TO_BF16)]
        pairs.extend(
            [
                (f32_to_bf16_bits(0.625), 0x2),
                (f32_to_bf16_bits(0.75), 0x3),
                (f32_to_bf16_bits(0.875), 0x4),
                (f32_to_bf16_bits(-0.625), 0xA),
                (f32_to_bf16_bits(-0.75), 0xB),
                (f32_to_bf16_bits(-0.875), 0xC),
            ]
        )
        return pairs


    def f4e2_quantize_pairs():
        pairs = [(bits, nibble) for nibble, bits in enumerate(FP4E2M1_TO_BF16)]
        pairs.extend(
            [
                (f32_to_bf16_bits(0.75), 0x2),
                (f32_to_bf16_bits(1.25), 0x2),
                (f32_to_bf16_bits(1.75), 0x4),
                (f32_to_bf16_bits(-0.75), 0xA),
                (f32_to_bf16_bits(-1.25), 0xA),
                (f32_to_bf16_bits(-1.75), 0xC),
            ]
        )
        return pairs


    def _generate(seed: int) -> dict:
        inputs = {
            "f8e4": tiled(
                [
                    0x00, 0x80, 0x01, 0x81, 0x07, 0x87, 0x08, 0x88,
                    0x38, 0xB8, 0x3C, 0xBC, 0x78, 0xF8, 0x7E, 0xFE,
                    0x7F, 0xFF, 0x10, 0x90, 0x20, 0xA0, 0x70, 0xF0,
                ]
            ),
            "f8e5": tiled(
                [
                    0x00, 0x80, 0x01, 0x81, 0x03, 0x83, 0x04, 0x84,
                    0x3C, 0xBC, 0x40, 0xC0, 0x7B, 0xFB, 0x7C, 0xFC,
                    0x7D, 0xFD, 0x7F, 0xFF, 0x10, 0x90, 0x70, 0xF0,
                ]
            ),
            "hif8": tiled(
                [
                    0x00, 0x80, 0x6F, 0xEF, 0x01, 0x81, 0x07, 0x87,
                    0x08, 0x88, 0x10, 0x90, 0x18, 0x98, 0x20, 0xA0,
                    0x40, 0xC0, 0x50, 0xD0, 0x60, 0xE0, 0x70, 0xF0,
                ]
            ),
            "f4e1": tiled(
                [
                    0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE,
                    0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF,
                ]
            ),
            "f4e2": tiled(
                [
                    0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE,
                    0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF,
                ]
            ),
        }

        _files["v1_f8e4_in.bin"] = inputs["f8e4"]
        _files["v2_f8e5_in.bin"] = inputs["f8e5"]
        _files["v3_hif8_in.bin"] = inputs["hif8"]
        _files["v4_f4e1_in.bin"] = inputs["f4e1"]
        _files["v5_f4e2_in.bin"] = inputs["f4e2"]

        _files["v6_f8e4_f32_out.bin"] = np.full(BYTES, 0xA5, dtype=np.uint8)
        _files["v7_f8e5_f32_out.bin"] = np.full(BYTES, 0xA5, dtype=np.uint8)
        _files["v8_hif8_f32_out.bin"] = np.full(BYTES, 0xA5, dtype=np.uint8)
        _files["v9_f4e1_bf16_out.bin"] = np.full(BYTES, 0xA5, dtype=np.uint8)
        _files["v10_f4e2_bf16_out.bin"] = np.full(BYTES, 0xA5, dtype=np.uint8)

        _files["golden_v6_f8e4_f32_out.bin"] = expected_f8(inputs["f8e4"], decode_f8e4m3fn)
        _files["golden_v7_f8e5_f32_out.bin"] = expected_f8(inputs["f8e5"], decode_f8e5m2)
        _files["golden_v8_hif8_f32_out.bin"] = expected_f8(inputs["hif8"], decode_hif8)
        _files["golden_v9_f4e1_bf16_out.bin"] = expected_f4(inputs["f4e1"], FP4E1M2_TO_BF16)
        _files["golden_v10_f4e2_bf16_out.bin"] = expected_f4(inputs["f4e2"], FP4E2M1_TO_BF16)

        write_f32_quantize_case(
            output_dir, "v11_f8e4_f32_in.bin", "golden_v16_f8e4_out.bin", f8e4_quantize_pairs()
        )
        write_f32_quantize_case(
            output_dir, "v12_f8e5_f32_in.bin", "golden_v17_f8e5_out.bin", f8e5_quantize_pairs()
        )
        write_f32_quantize_case(
            output_dir, "v13_hif8_f32_in.bin", "golden_v18_hif8_out.bin", hif8_quantize_pairs()
        )
        write_bf16_quantize_case(
            output_dir, "v14_f4e1_bf16_in.bin", "golden_v19_f4e1_out.bin", f4e1_quantize_pairs()
        )
        write_bf16_quantize_case(
            output_dir, "v15_f4e2_bf16_in.bin", "golden_v20_f4e2_out.bin", f4e2_quantize_pairs()
        )
        for name in [
            "v16_f8e4_out.bin",
            "v17_f8e5_out.bin",
            "v18_hif8_out.bin",
            "v19_f4e1_out.bin",
            "v20_f4e2_out.bin",
        ]:
            _files[name] = np.full(BYTES, 0xA5, dtype=np.uint8)


    def main():
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

    F32_CHECKS = (
        ("f8e4m3fn-to-f32", "golden_v6_f8e4_f32_out.bin", "v6_f8e4_f32_out.bin"),
        ("f8e5m2-to-f32", "golden_v7_f8e5_f32_out.bin", "v7_f8e5_f32_out.bin"),
        ("hif8-to-f32", "golden_v8_hif8_f32_out.bin", "v8_hif8_f32_out.bin"),
    )

    BF16_CHECKS = (
        ("f4e1m2x2-to-bf16", "golden_v9_f4e1_bf16_out.bin", "v9_f4e1_bf16_out.bin"),
        ("f4e2m1x2-to-bf16", "golden_v10_f4e2_bf16_out.bin", "v10_f4e2_bf16_out.bin"),
    )

    BYTE_CHECKS = (
        ("f32-to-f8e4m3fn", "golden_v16_f8e4_out.bin", "v16_f8e4_out.bin"),
        ("f32-to-f8e5m2", "golden_v17_f8e5_out.bin", "v17_f8e5_out.bin"),
        ("f32-to-hif8", "golden_v18_hif8_out.bin", "v18_hif8_out.bin"),
        ("bf16-to-f4e1m2x2", "golden_v19_f4e1_out.bin", "v19_f4e1_out.bin"),
        ("bf16-to-f4e2m1x2", "golden_v20_f4e2_out.bin", "v20_f4e2_out.bin"),
    )


    def compare_f32(label, golden_path, output_path):
        golden = np.fromfile(golden_path, dtype=np.float32)
        output = np.fromfile(output_path, dtype=np.float32)
        if golden.shape != output.shape:
            print(f"[ERROR] {label}: shape mismatch golden={golden.shape} output={output.shape}")
            return False

        golden_bits = golden.view(np.uint32)
        output_bits = output.view(np.uint32)
        both_nan = np.isnan(golden) & np.isnan(output)
        same = (golden_bits == output_bits) | both_nan
        if np.all(same):
            return True

        idx = int(np.nonzero(~same)[0][0])
        print(
            f"[ERROR] {label}: mismatch idx={idx} "
            f"golden={golden[idx]} bits=0x{int(golden_bits[idx]):08x} "
            f"output={output[idx]} bits=0x{int(output_bits[idx]):08x}"
        )
        return False


    def compare_u16(label, golden_path, output_path):
        golden = np.fromfile(golden_path, dtype=np.uint16)
        output = np.fromfile(output_path, dtype=np.uint16)
        if golden.shape != output.shape:
            print(f"[ERROR] {label}: shape mismatch golden={golden.shape} output={output.shape}")
            return False
        if np.array_equal(golden, output):
            return True

        idx = int(np.nonzero(golden != output)[0][0])
        print(
            f"[ERROR] {label}: mismatch idx={idx} "
            f"golden=0x{int(golden[idx]):04x} output=0x{int(output[idx]):04x}"
        )
        return False


    def compare_u8(label, golden_path, output_path):
        golden = np.fromfile(golden_path, dtype=np.uint8)
        output = np.fromfile(output_path, dtype=np.uint8)
        if golden.shape != output.shape:
            print(f"[ERROR] {label}: shape mismatch golden={golden.shape} output={output.shape}")
            return False
        if np.array_equal(golden, output):
            return True

        idx = int(np.nonzero(golden != output)[0][0])
        print(
            f"[ERROR] {label}: byte mismatch idx={idx} "
            f"golden=0x{int(golden[idx]):02x} output=0x{int(output[idx]):02x}"
        )
        return False


    def main():
        ok = True
        for check in F32_CHECKS:
            ok = compare_f32(*check) and ok
        for check in BF16_CHECKS:
            ok = compare_u16(*check) and ok
        for check in BYTE_CHECKS:
            ok = compare_u8(*check) and ok
        if not ok:
            sys.exit(2)
        print("[INFO] compare passed")


    return locals()

HELPERS = {}
SEED_EXPR = None
COMPARE_PAIRS = [
    (
        'golden_v6_f8e4_f32_out.bin',
        'v6_f8e4_f32_out.bin'
    ),
    (
        'golden_v7_f8e5_f32_out.bin',
        'v7_f8e5_f32_out.bin'
    ),
    (
        'golden_v8_hif8_f32_out.bin',
        'v8_hif8_f32_out.bin'
    ),
    (
        'golden_v9_f4e1_bf16_out.bin',
        'v9_f4e1_bf16_out.bin'
    ),
    (
        'golden_v10_f4e2_bf16_out.bin',
        'v10_f4e2_bf16_out.bin'
    ),
    (
        'golden_v16_f8e4_out.bin',
        'v16_f8e4_out.bin'
    ),
    (
        'golden_v17_f8e5_out.bin',
        'v17_f8e5_out.bin'
    ),
    (
        'golden_v18_hif8_out.bin',
        'v18_hif8_out.bin'
    ),
    (
        'golden_v19_f4e1_out.bin',
        'v19_f4e1_out.bin'
    ),
    (
        'golden_v20_f4e2_out.bin',
        'v20_f4e2_out.bin'
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
BUFFER_FILES = [
    (
        'v1_f8e4_in',
        'ui8'
    ),
    (
        'v2_f8e5_in',
        'ui8'
    ),
    (
        'v3_hif8_in',
        'ui8'
    ),
    (
        'v4_f4e1_in',
        'ui8'
    ),
    (
        'v5_f4e2_in',
        'ui8'
    ),
    (
        'v6_f8e4_f32_out',
        'ui8'
    ),
    (
        'v7_f8e5_f32_out',
        'ui8'
    ),
    (
        'v8_hif8_f32_out',
        'ui8'
    ),
    (
        'v9_f4e1_bf16_out',
        'ui8'
    ),
    (
        'v10_f4e2_bf16_out',
        'ui8'
    ),
    (
        'v11_f8e4_f32_in',
        'ui8'
    ),
    (
        'v12_f8e5_f32_in',
        'ui8'
    ),
    (
        'v13_hif8_f32_in',
        'ui8'
    ),
    (
        'v14_f4e1_bf16_in',
        'ui8'
    ),
    (
        'v15_f4e2_bf16_in',
        'ui8'
    ),
    (
        'v16_f8e4_out',
        'ui8'
    ),
    (
        'v17_f8e5_out',
        'ui8'
    ),
    (
        'v18_hif8_out',
        'ui8'
    ),
    (
        'v19_f4e1_out',
        'ui8'
    ),
    (
        'v20_f4e2_out',
        'ui8'
    )
]
LAUNCHES = [
    (
        0,
        [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9
        ],
        1,
        [],
        [
            'ui8',
            'ui8',
            'ui8',
            'ui8',
            'ui8',
            'ui8',
            'ui8',
            'ui8',
            'ui8',
            'ui8'
        ],
        [
            '0',
            '0',
            '0',
            '0',
            '0',
            '0',
            '0',
            '0',
            '0',
            '0'
        ]
    ),
    (
        1,
        [
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19
        ],
        1,
        [],
        [
            'ui8',
            'ui8',
            'ui8',
            'ui8',
            'ui8',
            'ui8',
            'ui8',
            'ui8',
            'ui8',
            'ui8'
        ],
        [
            '0',
            '0',
            '0',
            '0',
            '0',
            '0',
            '0',
            '0',
            '0',
            '0'
        ]
    )
]

def _dtype_map():
    import torch
    return {
        "int8": torch.int8, "int16": torch.int16, "int32": torch.int32, "int64": torch.int64,
        "uint8": torch.uint8, "uint16": torch.int16, "uint32": torch.int32, "uint64": torch.int64,
        "float16": torch.float16, "float32": torch.float32, "float64": torch.float64,
    }


@pto.jit(
    name='vcvt_low_precision_special_kernel',
    target="a5",
    backend="vpto",
    mode="explicit",
    source=KERNEL_SOURCE,
)
def kernel_0(
    f8e4_in: pto.ptr(pto.ui8, "gm"),
    f8e5_in: pto.ptr(pto.ui8, "gm"),
    hif8_in: pto.ptr(pto.ui8, "gm"),
    f4e1_in: pto.ptr(pto.ui8, "gm"),
    f4e2_in: pto.ptr(pto.ui8, "gm"),
    f8e4_out: pto.ptr(pto.ui8, "gm"),
    f8e5_out: pto.ptr(pto.ui8, "gm"),
    hif8_out: pto.ptr(pto.ui8, "gm"),
    f4e1_out: pto.ptr(pto.ui8, "gm"),
    f4e2_out: pto.ptr(pto.ui8, "gm"),
):
    pass

@pto.jit(
    name='vcvt_low_precision_quantize_special_kernel',
    target="a5",
    backend="vpto",
    mode="explicit",
    source=KERNEL_SOURCE,
)
def kernel_1(
    f8e4_in: pto.ptr(pto.ui8, "gm"),
    f8e5_in: pto.ptr(pto.ui8, "gm"),
    hif8_in: pto.ptr(pto.ui8, "gm"),
    f4e1_in: pto.ptr(pto.ui8, "gm"),
    f4e2_in: pto.ptr(pto.ui8, "gm"),
    f8e4_out: pto.ptr(pto.ui8, "gm"),
    f8e5_out: pto.ptr(pto.ui8, "gm"),
    hif8_out: pto.ptr(pto.ui8, "gm"),
    f4e1_out: pto.ptr(pto.ui8, "gm"),
    f4e2_out: pto.ptr(pto.ui8, "gm"),
):
    pass


KERNELS = [kernel_0, kernel_1]


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
    tensors = []
    for name, ty in BUFFER_FILES:
        key = "%s.bin" % name
        if key in produced:
            array = _as_buffer_array(produced[key], NUMPY_DTYPE[ty])
        else:
            golden_key = output_to_golden.get(key)
            if golden_key is None or golden_key not in produced:
                raise RuntimeError("golden did not produce %s" % key)
            array = np.ascontiguousarray(
                np.zeros_like(np.asarray(produced[golden_key]), dtype=np.dtype(NUMPY_DTYPE[ty])))
        tensors.append(torch.from_numpy(array).to("npu:0"))
    return tensors, produced


def run_case(device_inputs, launch_args, stream):
    import torch
    compiled = [kernel.compile() for kernel in KERNELS]
    for kernel_index, indices, grid, scalars, dtypes, offsets in LAUNCHES:
        args = []
        for index, ty, offset in zip(indices, dtypes, offsets):
            tensor = device_inputs[index]
            if offset not in ('', '0'):
                tensor = tensor[int(offset):]
            want = _dtype_map()[NUMPY_DTYPE[ty]]
            if tensor.dtype != want and tensor.numel() * tensor.element_size() % want.itemsize == 0:
                tensor = tensor.view(want)
            args.append(tensor)
        compiled[kernel_index][grid, stream](*args, *scalars)


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
    for index, (name, _ty) in enumerate(BUFFER_FILES):
        files["%s.bin" % name] = device_inputs[index].cpu().numpy()
    _apply_write_masks(files, produced)
    _run_compare(files)


CASES = [dict(
    name='legacy_micro_op_conversion_vcvt_low_precision_special',
    grid=1,
    kernel=KERNELS[0],
    make_case=make_case,
    check=check_case,
    run=run_case,
)]

auto_main(globals())
