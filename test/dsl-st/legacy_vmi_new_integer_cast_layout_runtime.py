#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Migrated from test/vpto/cases/vmi_new/integer-cast-layout-runtime.

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
    '\nmodule attributes {pto.target_arch = "a5", pto.kernel_kind = #pto.kernel_kind<vector>} {\n  func.func @int'
    'eger_cast_u8_kernel(\n      %src_gm: !pto.ptr<ui8, gm>, %dst16_gm: !pto.ptr<ui16, gm>,\n      %dst32_gm: !pt'
    'o.ptr<ui32, gm>) attributes {pto.kernel} {\n    %c0 = arith.constant 0 : index\n    %c0_i64 = arith.constant'
    ' 0 : i64\n    %c1_i64 = arith.constant 1 : i64\n    %c256_i64 = arith.constant 256 : i64\n    %c512_i64 = ar'
    'ith.constant 512 : i64\n    %c1024_i64 = arith.constant 1024 : i64\n    %c4096_i64 = arith.constant 4096 : i'
    '64\n    %c8192_i64 = arith.constant 8192 : i64\n    %ub_src = pto.castptr %c0_i64 : i64 -> !pto.ptr<ui8, ub>'
    '\n    %ub_dst16 = pto.castptr %c4096_i64 : i64 -> !pto.ptr<ui16, ub>\n    %ub_dst32 = pto.castptr %c8192_i64'
    ' : i64 -> !pto.ptr<ui32, ub>\n\n    pto.mte_gm_ub %src_gm, %ub_src, %c0_i64, %c256_i64\n      nburst(%c1_i64'
    ', %c256_i64, %c256_i64)\n      : !pto.ptr<ui8, gm>, !pto.ptr<ui8, ub>, i64, i64, i64, i64, i64\n    pto.set_'
    'flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n    pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n    pto.vec'
    'scope {\n      %src16 = pto.vmi.load %ub_src[%c0]\n          : !pto.ptr<ui8, ub>, index -> !pto.vmi.vreg<256xui8>'
    '\n      %dst16 = pto.vmi.extui %src16\n          : !pto.vmi.vreg<256xui8> -> !pto.vmi.vreg<256xui16>\n      %s'
    'rc32 = pto.vmi.load %ub_src[%c0]\n          : !pto.ptr<ui8, ub>, index -> !pto.vmi.vreg<256xui8>\n      %dst32 = p'
    'to.vmi.extui %src32\n          : !pto.vmi.vreg<256xui8> -> !pto.vmi.vreg<256xui32>\n      pto.vmi.store %dst1'
    '6, %ub_dst16[%c0]\n          : !pto.vmi.vreg<256xui16>, !pto.ptr<ui16, ub>, index\n      pto.vmi.store %dst32, %ub'
    '_dst32[%c0]\n          : !pto.vmi.vreg<256xui32>, !pto.ptr<ui32, ub>, index\n    }\n    pto.set_flag["PIPE_V", "PI'
    'PE_MTE3", "EVENT_ID0"]\n    pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n    pto.mte_ub_gm %ub_dst16, %d'
    'st16_gm, %c512_i64\n      nburst(%c1_i64, %c512_i64, %c512_i64)\n      : !pto.ptr<ui16, ub>, !pto.ptr<ui16, '
    'gm>, i64, i64, i64, i64\n    pto.mte_ub_gm %ub_dst32, %dst32_gm, %c1024_i64\n      nburst(%c1_i64, %c1024_i6'
    '4, %c1024_i64)\n      : !pto.ptr<ui32, ub>, !pto.ptr<ui32, gm>, i64, i64, i64, i64\n    pto.barrier #pto.pip'
    'e<PIPE_ALL>\n    return\n  }\n\n  func.func @integer_cast_i8_kernel(\n      %src_gm: !pto.ptr<si8, gm>, %dst'
    '16_gm: !pto.ptr<si16, gm>,\n      %dst32_gm: !pto.ptr<si32, gm>) attributes {pto.kernel} {\n    %c0 = arith.'
    'constant 0 : index\n    %c0_i64 = arith.constant 0 : i64\n    %c1_i64 = arith.constant 1 : i64\n    %c256_i6'
    '4 = arith.constant 256 : i64\n    %c512_i64 = arith.constant 512 : i64\n    %c1024_i64 = arith.constant 1024'
    ' : i64\n    %c4096_i64 = arith.constant 4096 : i64\n    %c8192_i64 = arith.constant 8192 : i64\n    %ub_src '
    '= pto.castptr %c0_i64 : i64 -> !pto.ptr<si8, ub>\n    %ub_dst16 = pto.castptr %c4096_i64 : i64 -> !pto.ptr<s'
    'i16, ub>\n    %ub_dst32 = pto.castptr %c8192_i64 : i64 -> !pto.ptr<si32, ub>\n\n    pto.mte_gm_ub %src_gm, %'
    'ub_src, %c0_i64, %c256_i64\n      nburst(%c1_i64, %c256_i64, %c256_i64)\n      : !pto.ptr<si8, gm>, !pto.ptr'
    '<si8, ub>, i64, i64, i64, i64, i64\n    pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n    pto.wait_flag['
    '"PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n    pto.vecscope {\n      %src16 = pto.vmi.load %ub_src[%c0]\n         '
    ' : !pto.ptr<si8, ub>, index -> !pto.vmi.vreg<256xsi8>\n      %dst16 = pto.vmi.extsi %src16\n          : !pto.vmi.v'
    'reg<256xsi8> -> !pto.vmi.vreg<256xsi16>\n      %src32 = pto.vmi.load %ub_src[%c0]\n          : !pto.ptr<si8, '
    'ub>, index -> !pto.vmi.vreg<256xsi8>\n      %dst32 = pto.vmi.extsi %src32\n          : !pto.vmi.vreg<256xsi8> -> !'
    'pto.vmi.vreg<256xsi32>\n      pto.vmi.store %dst16, %ub_dst16[%c0]\n          : !pto.vmi.vreg<256xsi16>, !pto'
    '.ptr<si16, ub>, index\n      pto.vmi.store %dst32, %ub_dst32[%c0]\n          : !pto.vmi.vreg<256xsi32>, !pto.ptr<s'
    'i32, ub>, index\n    }\n    pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n    pto.wait_flag["PIPE_V", "PIPE_MT'
    'E3", "EVENT_ID0"]\n    pto.mte_ub_gm %ub_dst16, %dst16_gm, %c512_i64\n      nburst(%c1_i64, %c512_i64, %c512_i'
    '64)\n      : !pto.ptr<si16, ub>, !pto.ptr<si16, gm>, i64, i64, i64, i64\n    pto.mte_ub_gm %ub_dst32, %dst32'
    '_gm, %c1024_i64\n      nburst(%c1_i64, %c1024_i64, %c1024_i64)\n      : !pto.ptr<si32, ub>, !pto.ptr<si32, g'
    'm>, i64, i64, i64, i64\n    pto.barrier #pto.pipe<PIPE_ALL>\n    return\n  }\n\n  func.func @integer_cast_u1'
    '6_kernel(\n      %src_gm: !pto.ptr<ui16, gm>, %dst32_gm: !pto.ptr<ui32, gm>,\n      %dst8_gm: !pto.ptr<ui8, '
    'gm>) attributes {pto.kernel} {\n    %c0 = arith.constant 0 : index\n    %c128 = arith.constant 128 : index\n'
    '    %c0_i64 = arith.constant 0 : i64\n    %c1_i64 = arith.constant 1 : i64\n    %c128_i64 = arith.constant 1'
    '28 : i64\n    %c256_i64 = arith.constant 256 : i64\n    %c512_i64 = arith.constant 512 : i64\n    %c4096_i64'
    ' = arith.constant 4096 : i64\n    %c8192_i64 = arith.constant 8192 : i64\n    %c12288_i64 = arith.constant 1'
    '2288 : i64\n    %ub_src = pto.castptr %c0_i64 : i64 -> !pto.ptr<ui16, ub>\n    %ub_dst32 = pto.castptr %c409'
    '6_i64 : i64 -> !pto.ptr<ui32, ub>\n    %ub_dst8 = pto.castptr %c8192_i64 : i64 -> !pto.ptr<ui8, ub>\n    %ub'
    '_scratch32 = pto.castptr %c12288_i64 : i64 -> !pto.ptr<ui32, ub>\n\n    pto.mte_gm_ub %src_gm, %ub_src, %c0_'
    'i64, %c256_i64\n      nburst(%c1_i64, %c256_i64, %c256_i64)\n      : !pto.ptr<ui16, gm>, !pto.ptr<ui16, ub>,'
    ' i64, i64, i64, i64, i64\n    pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n    pto.wait_flag["PIPE_MTE2'
    '", "PIPE_V", "EVENT_ID0"]\n    pto.vecscope {\n      %src = pto.vmi.load %ub_src[%c0]\n          : !pto.ptr<'
    'ui16, ub>, index -> !pto.vmi.vreg<128xui16>\n      %dst32 = pto.vmi.extui %src\n          : !pto.vmi.vreg<128xui16'
    '>\n          -> !pto.vmi.vreg<128xui32>\n      %dst8 = pto.vmi.trunci %src {saturate = "NOSAT"}\n          : '
    '!pto.vmi.vreg<128xui16>\n          -> !pto.vmi.vreg<128xui8>\n      pto.vmi.store %dst32, %ub_dst32[%c0]\n  '
    '        : !pto.vmi.vreg<128xui32>, !pto.ptr<ui32, ub>, index\n      pto.vmi.store %dst8, %ub_dst8[%c0]\n          '
    ': !pto.vmi.vreg<128xui8>, !pto.ptr<ui8, ub>, index\n      // One-block groups make the widening result contiguous '
    'through layout assignment.\n      %mask = pto.vmi.create_mask %c128 : index -> !pto.vmi.mask<128xpred>\n      '
    '%sum32 = pto.vmi.group_reduce_addi %dst32, %mask {num_groups = 16 : i64}\n          : !pto.vmi.vreg<128xui32'
    '>, !pto.vmi.mask<128xpred>\n          -> !pto.vmi.vreg<16xui32>\n      %broadcast32 = pto.vmi.group_broadcas'
    't %sum32 {num_groups = 16 : i64}\n          : !pto.vmi.vreg<16xui32> -> !pto.vmi.vreg<128xui32>\n      pto.v'
    'mi.store %broadcast32, %ub_scratch32[%c0]\n          : !pto.vmi.vreg<128xui32>, !pto.ptr<ui32, ub>, index\n    }\n'
    '    pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n    pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n'
    '    pto.mte_ub_gm %ub_dst32, %dst32_gm, %c512_i64\n      nburst(%c1_i64, %c512_i64, %c512_i64)\n      : !pto'
    '.ptr<ui32, ub>, !pto.ptr<ui32, gm>, i64, i64, i64, i64\n    pto.mte_ub_gm %ub_dst8, %dst8_gm, %c128_i64\n   '
    '   nburst(%c1_i64, %c128_i64, %c128_i64)\n      : !pto.ptr<ui8, ub>, !pto.ptr<ui8, gm>, i64, i64, i64, i64\n'
    '    pto.barrier #pto.pipe<PIPE_ALL>\n    return\n  }\n\n  func.func @integer_cast_i16_kernel(\n      %src_gm'
    ': !pto.ptr<si16, gm>, %dst32_gm: !pto.ptr<si32, gm>,\n      %dst8_gm: !pto.ptr<si8, gm>) attributes {pto.ker'
    'nel} {\n    %c0 = arith.constant 0 : index\n    %c128 = arith.constant 128 : index\n    %c0_i64 = arith.cons'
    'tant 0 : i64\n    %c1_i64 = arith.constant 1 : i64\n    %c128_i64 = arith.constant 128 : i64\n    %c256_i64 '
    '= arith.constant 256 : i64\n    %c512_i64 = arith.constant 512 : i64\n    %c4096_i64 = arith.constant 4096 :'
    ' i64\n    %c8192_i64 = arith.constant 8192 : i64\n    %c12288_i64 = arith.constant 12288 : i64\n    %ub_src '
    '= pto.castptr %c0_i64 : i64 -> !pto.ptr<si16, ub>\n    %ub_dst32 = pto.castptr %c4096_i64 : i64 -> !pto.ptr<'
    'si32, ub>\n    %ub_dst8 = pto.castptr %c8192_i64 : i64 -> !pto.ptr<si8, ub>\n    %ub_scratch32 = pto.castptr'
    ' %c12288_i64 : i64 -> !pto.ptr<si32, ub>\n\n    pto.mte_gm_ub %src_gm, %ub_src, %c0_i64, %c256_i64\n      nb'
    'urst(%c1_i64, %c256_i64, %c256_i64)\n      : !pto.ptr<si16, gm>, !pto.ptr<si16, ub>, i64, i64, i64, i64, i64'
    '\n    pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n    pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"'
    ']\n    pto.vecscope {\n      %src = pto.vmi.load %ub_src[%c0]\n          : !pto.ptr<si16, ub>, index -> !pto.vmi.v'
    'reg<128xsi16>\n      %dst32 = pto.vmi.extsi %src\n          : !pto.vmi.vreg<128xsi16>\n          -> !pto.vmi.'
    'vreg<128xsi32>\n      %dst8 = pto.vmi.trunci %src {saturate = "NOSAT"}\n          : !pto.vmi.vreg<128xsi16>'
    '\n          -> !pto.vmi.vreg<128xsi8>\n      pto.vmi.store %dst32, %ub_dst32[%c0]\n          : !pto.vmi.vreg'
    '<128xsi32>, !pto.ptr<si32, ub>, index\n      pto.vmi.store %dst8, %ub_dst8[%c0]\n          : !pto.vmi.vreg<128xsi8'
    '>, !pto.ptr<si8, ub>, index\n      // One-block groups make the widening result contiguous through layout assignme'
    'nt.\n      %mask = pto.vmi.create_mask %c128 : index -> !pto.vmi.mask<128xpred>\n      %sum32 = pto.vmi.group_'
    'reduce_addi %dst32, %mask {num_groups = 16 : i64}\n          : !pto.vmi.vreg<128xsi32>, !pto.vmi.mask<128xpr'
    'ed>\n          -> !pto.vmi.vreg<16xsi32>\n      %broadcast32 = pto.vmi.group_broadcast %sum32 {num_groups = '
    '16 : i64}\n          : !pto.vmi.vreg<16xsi32> -> !pto.vmi.vreg<128xsi32>\n      pto.vmi.store %broadcast32, '
    '%ub_scratch32[%c0]\n          : !pto.vmi.vreg<128xsi32>, !pto.ptr<si32, ub>, index\n    }\n    pto.set_flag["PIPE_'
    'V", "PIPE_MTE3", "EVENT_ID0"]\n    pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n    pto.mte_ub_gm %ub_d'
    'st32, %dst32_gm, %c512_i64\n      nburst(%c1_i64, %c512_i64, %c512_i64)\n      : !pto.ptr<si32, ub>, !pto.pt'
    'r<si32, gm>, i64, i64, i64, i64\n    pto.mte_ub_gm %ub_dst8, %dst8_gm, %c128_i64\n      nburst(%c1_i64, %c12'
    '8_i64, %c128_i64)\n      : !pto.ptr<si8, ub>, !pto.ptr<si8, gm>, i64, i64, i64, i64\n    pto.barrier #pto.pi'
    'pe<PIPE_ALL>\n    return\n  }\n\n  func.func @integer_cast_u32_kernel(\n      %src_gm: !pto.ptr<ui32, gm>, %'
    'dst16_gm: !pto.ptr<ui16, gm>,\n      %dst8_gm: !pto.ptr<ui8, gm>) attributes {pto.kernel} {\n    %c0 = arith'
    '.constant 0 : index\n    %c0_i64 = arith.constant 0 : i64\n    %c1_i64 = arith.constant 1 : i64\n    %c64_i6'
    '4 = arith.constant 64 : i64\n    %c128_i64 = arith.constant 128 : i64\n    %c256_i64 = arith.constant 256 : '
    'i64\n    %c4096_i64 = arith.constant 4096 : i64\n    %c8192_i64 = arith.constant 8192 : i64\n    %ub_src = p'
    'to.castptr %c0_i64 : i64 -> !pto.ptr<ui32, ub>\n    %ub_dst16 = pto.castptr %c4096_i64 : i64 -> !pto.ptr<ui1'
    '6, ub>\n    %ub_dst8 = pto.castptr %c8192_i64 : i64 -> !pto.ptr<ui8, ub>\n\n    pto.mte_gm_ub %src_gm, %ub_s'
    'rc, %c0_i64, %c256_i64\n      nburst(%c1_i64, %c256_i64, %c256_i64)\n      : !pto.ptr<ui32, gm>, !pto.ptr<ui'
    '32, ub>, i64, i64, i64, i64, i64\n    pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n    pto.wait_flag["P'
    'IPE_MTE2", "PIPE_V", "EVENT_ID0"]\n    pto.vecscope {\n      %src = pto.vmi.load %ub_src[%c0]\n          : !'
    'pto.ptr<ui32, ub>, index -> !pto.vmi.vreg<64xui32>\n      %dst16 = pto.vmi.trunci %src {saturate = "NOSAT"}\n     '
    '     : !pto.vmi.vreg<64xui32>\n          -> !pto.vmi.vreg<64xui16>\n      %dst8 = pto.vmi.trunci %src {satura'
    'te = "NOSAT"}\n          : !pto.vmi.vreg<64xui32>\n          -> !pto.vmi.vreg<64xui8>\n      pto.vmi.store %'
    'dst16, %ub_dst16[%c0]\n          : !pto.vmi.vreg<64xui16>, !pto.ptr<ui16, ub>, index\n      pto.vmi.store %dst8, %'
    'ub_dst8[%c0]\n          : !pto.vmi.vreg<64xui8>, !pto.ptr<ui8, ub>, index\n    }\n    pto.set_flag["PIPE_V", "PIPE'
    '_MTE3", "EVENT_ID0"]\n    pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n    pto.mte_ub_gm %ub_dst16, %dst'
    '16_gm, %c128_i64\n      nburst(%c1_i64, %c128_i64, %c128_i64)\n      : !pto.ptr<ui16, ub>, !pto.ptr<ui16, gm'
    '>, i64, i64, i64, i64\n    pto.mte_ub_gm %ub_dst8, %dst8_gm, %c64_i64\n      nburst(%c1_i64, %c64_i64, %c64_'
    'i64)\n      : !pto.ptr<ui8, ub>, !pto.ptr<ui8, gm>, i64, i64, i64, i64\n    pto.barrier #pto.pipe<PIPE_ALL>'
    '\n    return\n  }\n\n  func.func @integer_cast_i32_kernel(\n      %src_gm: !pto.ptr<si32, gm>, %dst16_gm: !p'
    'to.ptr<si16, gm>,\n      %dst8_gm: !pto.ptr<si8, gm>) attributes {pto.kernel} {\n    %c0 = arith.constant 0 '
    ': index\n    %c0_i64 = arith.constant 0 : i64\n    %c1_i64 = arith.constant 1 : i64\n    %c64_i64 = arith.co'
    'nstant 64 : i64\n    %c128_i64 = arith.constant 128 : i64\n    %c256_i64 = arith.constant 256 : i64\n    %c4'
    '096_i64 = arith.constant 4096 : i64\n    %c8192_i64 = arith.constant 8192 : i64\n    %ub_src = pto.castptr %'
    'c0_i64 : i64 -> !pto.ptr<si32, ub>\n    %ub_dst16 = pto.castptr %c4096_i64 : i64 -> !pto.ptr<si16, ub>\n    '
    '%ub_dst8 = pto.castptr %c8192_i64 : i64 -> !pto.ptr<si8, ub>\n\n    pto.mte_gm_ub %src_gm, %ub_src, %c0_i64,'
    ' %c256_i64\n      nburst(%c1_i64, %c256_i64, %c256_i64)\n      : !pto.ptr<si32, gm>, !pto.ptr<si32, ub>, i64'
    ', i64, i64, i64, i64\n    pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n    pto.wait_flag["PIPE_MTE2", "'
    'PIPE_V", "EVENT_ID0"]\n    pto.vecscope {\n      %src = pto.vmi.load %ub_src[%c0]\n          : !pto.ptr<si32'
    ', ub>, index -> !pto.vmi.vreg<64xsi32>\n      %dst16 = pto.vmi.trunci %src {saturate = "NOSAT"}\n          : !pto.'
    'vmi.vreg<64xsi32>\n          -> !pto.vmi.vreg<64xsi16>\n      %dst8 = pto.vmi.trunci %src {saturate = "NOSAT"'
    '}\n          : !pto.vmi.vreg<64xsi32>\n          -> !pto.vmi.vreg<64xsi8>\n      pto.vmi.store %dst16, %ub_d'
    'st16[%c0]\n          : !pto.vmi.vreg<64xsi16>, !pto.ptr<si16, ub>, index\n      pto.vmi.store %dst8, %ub_dst8[%c0]'
    '\n          : !pto.vmi.vreg<64xsi8>, !pto.ptr<si8, ub>, index\n    }\n    pto.set_flag["PIPE_V", "PIPE_MTE3", "EVE'
    'NT_ID0"]\n    pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n    pto.mte_ub_gm %ub_dst16, %dst16_gm, %c12'
    '8_i64\n      nburst(%c1_i64, %c128_i64, %c128_i64)\n      : !pto.ptr<si16, ub>, !pto.ptr<si16, gm>, i64, i64'
    ', i64, i64\n    pto.mte_ub_gm %ub_dst8, %dst8_gm, %c64_i64\n      nburst(%c1_i64, %c64_i64, %c64_i64)\n     '
    ' : !pto.ptr<si8, ub>, !pto.ptr<si8, gm>, i64, i64, i64, i64\n    pto.barrier #pto.pipe<PIPE_ALL>\n    return'
    '\n  }\n\n  func.func @integer_cast_zero_gap_kernel(\n      %src8_gm: !pto.ptr<ui8, gm>, %src16_gm: !pto.ptr<'
    'ui16, gm>,\n      %dst16_gm: !pto.ptr<ui16, gm>, %dst32_from8_gm: !pto.ptr<ui32, gm>,\n      %dst32_from16_g'
    'm: !pto.ptr<ui32, gm>) attributes {pto.kernel} {\n    %c0 = arith.constant 0 : index\n    %c0_i64 = arith.co'
    'nstant 0 : i64\n    %c1_i64 = arith.constant 1 : i64\n    %c128_i64 = arith.constant 128 : i64\n    %c256_i6'
    '4 = arith.constant 256 : i64\n    %c4096_i64 = arith.constant 4096 : i64\n    %c8192_i64 = arith.constant 81'
    '92 : i64\n    %c12288_i64 = arith.constant 12288 : i64\n    %c16384_i64 = arith.constant 16384 : i64\n    %u'
    'b_src8 = pto.castptr %c0_i64 : i64 -> !pto.ptr<ui8, ub>\n    %ub_src16 = pto.castptr %c4096_i64 : i64 -> !pt'
    'o.ptr<ui16, ub>\n    %ub_dst16 = pto.castptr %c8192_i64 : i64 -> !pto.ptr<ui16, ub>\n    %ub_dst32_from8 = p'
    'to.castptr %c12288_i64 : i64 -> !pto.ptr<ui32, ub>\n    %ub_dst32_from16 = pto.castptr %c16384_i64 : i64 -> '
    '!pto.ptr<ui32, ub>\n\n    pto.mte_gm_ub %src8_gm, %ub_src8, %c0_i64, %c128_i64\n      nburst(%c1_i64, %c128_'
    'i64, %c128_i64)\n      : !pto.ptr<ui8, gm>, !pto.ptr<ui8, ub>, i64, i64, i64, i64, i64\n    pto.mte_gm_ub %s'
    'rc16_gm, %ub_src16, %c0_i64, %c128_i64\n      nburst(%c1_i64, %c128_i64, %c128_i64)\n      : !pto.ptr<ui16, '
    'gm>, !pto.ptr<ui16, ub>, i64, i64, i64, i64, i64\n    pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n    '
    'pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n    pto.vecscope {\n      %src8_ls2 = pto.vmi.load %ub_sr'
    'c8[%c0]\n          : !pto.ptr<ui8, ub>, index\n          -> !pto.vmi.vreg<128xui8>\n      %dst16 = pto.vmi.extui %'
    'src8_ls2\n          : !pto.vmi.vreg<128xui8> -> !pto.vmi.vreg<128xui16>\n      pto.vmi.store %dst16, %ub_dst1'
    '6[%c0]\n          : !pto.vmi.vreg<128xui16>, !pto.ptr<ui16, ub>, index\n\n      %src8_ls4 = pto.vmi.load %ub_src8['
    '%c0]\n          : !pto.ptr<ui8, ub>, index\n          -> !pto.vmi.vreg<64xui8>\n      %dst32_from8 = pto.vmi.extui'
    ' %src8_ls4\n          : !pto.vmi.vreg<64xui8> -> !pto.vmi.vreg<64xui32>\n      pto.vmi.store %dst32_from8, %ub'
    '_dst32_from8[%c0]\n          : !pto.vmi.vreg<64xui32>, !pto.ptr<ui32, ub>, index\n\n      %src16_ls2 = pto.vmi.loa'
    'd %ub_src16[%c0]\n          : !pto.ptr<ui16, ub>, index\n          -> !pto.vmi.vreg<64xui16>\n      %dst32_from16 '
    '= pto.vmi.extui %src16_ls2\n          : !pto.vmi.vreg<64xui16> -> !pto.vmi.vreg<64xui32>\n      pto.vmi.store '
    '%dst32_from16, %ub_dst32_from16[%c0]\n          : !pto.vmi.vreg<64xui32>, !pto.ptr<ui32, ub>, index\n    }\n    pt'
    'o.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n    pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n    pt'
    'o.mte_ub_gm %ub_dst16, %dst16_gm, %c256_i64\n      nburst(%c1_i64, %c256_i64, %c256_i64)\n      : !pto.ptr<u'
    'i16, ub>, !pto.ptr<ui16, gm>, i64, i64, i64, i64\n    pto.mte_ub_gm %ub_dst32_from8, %dst32_from8_gm, %c256_'
    'i64\n      nburst(%c1_i64, %c256_i64, %c256_i64)\n      : !pto.ptr<ui32, ub>, !pto.ptr<ui32, gm>, i64, i64, '
    'i64, i64\n    pto.mte_ub_gm %ub_dst32_from16, %dst32_from16_gm, %c256_i64\n      nburst(%c1_i64, %c256_i64, '
    '%c256_i64)\n      : !pto.ptr<ui32, ub>, !pto.ptr<ui32, gm>, i64, i64, i64, i64\n    pto.barrier #pto.pipe<PI'
    'PE_ALL>\n    return\n  }\n}\n'
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


    SENTINEL = 0xA5


    def repeat(values, size, dtype):
        values = np.asarray(values, dtype=dtype)
        return np.resize(values, size).astype(dtype)


    def write_case(output_dir, first_file, source, outputs):
        _files[f"v{first_file}.bin"] = source
        for offset, expected in enumerate(outputs, start=1):
            path = f"v{first_file + offset}.bin"
            _files[path] = np.full(expected.nbytes, SENTINEL, dtype=np.uint8)
            _files[f"golden_v{first_file + offset}.bin"] = expected


    def _generate(seed: int) -> dict:
        u8 = np.arange(256, dtype=np.uint8)
        i8 = np.arange(-128, 128, dtype=np.int16).astype(np.int8)
        u16 = repeat(
            [0, 1, 127, 128, 255, 256, 257, 32767, 32768, 65534, 65535,
             0x1234, 0xABCD, 0xFF80, 0x00FF, 0x0100], 128, np.uint16)
        i16 = repeat(
            [-32768, -32767, -1025, -257, -256, -255, -129, -128, -127,
             -1, 0, 1, 127, 128, 255, 256, 257, 32767], 128, np.int16)
        u32 = repeat(
            [0, 1, 127, 128, 255, 256, 65535, 65536, 0x12345678,
             0x80000000, 0xFFFFFF80, 0xFFFFFFFF, 0xA5A55A5A], 64, np.uint32)
        i32 = repeat(
            [-2147483648, -2147483647, -65537, -65536, -32769, -32768,
             -257, -256, -129, -128, -1, 0, 1, 127, 128, 255, 256, 32767,
             32768, 65535, 65536, 2147483647], 64, np.int32)

        write_case(output_dir, 1, u8, [u8.astype(np.uint16), u8.astype(np.uint32)])
        write_case(output_dir, 4, i8, [i8.astype(np.int16), i8.astype(np.int32)])
        write_case(output_dir, 7, u16, [u16.astype(np.uint32), u16.astype(np.uint8)])
        write_case(output_dir, 10, i16, [i16.astype(np.int32), i16.astype(np.int8)])
        write_case(output_dir, 13, u32, [u32.astype(np.uint16), u32.astype(np.uint8)])
        write_case(output_dir, 16, i32, [i32.astype(np.int16), i32.astype(np.int8)])

        zero_gap_u8 = u8[:128]
        zero_gap_u16 = u16[:64]
        _files["v19.bin"] = zero_gap_u8
        _files["v20.bin"] = zero_gap_u16
        for index, expected in {
            21: zero_gap_u8.astype(np.uint16),
            22: zero_gap_u8[:64].astype(np.uint32),
            23: zero_gap_u16.astype(np.uint32),
        }.items():
            _files[f"v{index}.bin"] = np.full(expected.nbytes, SENTINEL, dtype=np.uint8)
            _files[f"golden_v{index}.bin"] = expected

        assert np.any(u16.astype(np.uint8) != np.minimum(u16, 255).astype(np.uint8))
        assert np.any(i16.astype(np.int8) != np.clip(i16, -128, 127).astype(np.int8))
        assert np.any(u32.astype(np.uint16) != np.minimum(u32, 65535).astype(np.uint16))
        assert np.any(i32.astype(np.int16) != np.clip(i32, -32768, 32767).astype(np.int16))


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


    OUTPUT_DTYPES = {
        2: np.uint16, 3: np.uint32, 5: np.int16, 6: np.int32,
        8: np.uint32, 9: np.uint8, 11: np.int32, 12: np.int8,
        14: np.uint16, 15: np.uint8, 17: np.int16, 18: np.int8,
        21: np.uint16, 22: np.uint32, 23: np.uint32,
    }


    def main() -> None:
        failed = False
        for index, dtype in OUTPUT_DTYPES.items():
            golden = np.fromfile(f"golden_v{index}.bin", dtype=dtype)
            output = np.fromfile(f"v{index}.bin", dtype=dtype)
            if golden.shape == output.shape and np.array_equal(golden, output):
                continue
            failed = True
            if golden.shape != output.shape:
                print(f"[ERROR] v{index}: shape {output.shape}, expected {golden.shape}")
                continue
            mismatches = np.flatnonzero(golden != output)
            shown = mismatches[:8]
            details = ", ".join(
                f"{int(i)}:{golden[i]}!={output[i]}" for i in shown)
            print(f"[ERROR] v{index}: {mismatches.size} mismatches ({details})")
        if failed:
            sys.exit(2)
        print("[INFO] all integer cast outputs match exactly")


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
BUFFER_FILES = [
    (
        'v1',
        'ui8'
    ),
    (
        'v2',
        'ui16'
    ),
    (
        'v3',
        'ui32'
    ),
    (
        'v4',
        'si8'
    ),
    (
        'v5',
        'si16'
    ),
    (
        'v6',
        'si32'
    ),
    (
        'v7',
        'ui16'
    ),
    (
        'v8',
        'ui32'
    ),
    (
        'v9',
        'ui8'
    ),
    (
        'v10',
        'si16'
    ),
    (
        'v11',
        'si32'
    ),
    (
        'v12',
        'si8'
    ),
    (
        'v13',
        'ui32'
    ),
    (
        'v14',
        'ui16'
    ),
    (
        'v15',
        'ui8'
    ),
    (
        'v16',
        'si32'
    ),
    (
        'v17',
        'si16'
    ),
    (
        'v18',
        'si8'
    ),
    (
        'v19',
        'ui8'
    ),
    (
        'v20',
        'ui16'
    ),
    (
        'v21',
        'ui16'
    ),
    (
        'v22',
        'ui32'
    ),
    (
        'v23',
        'ui32'
    )
]
LAUNCHES = [
    (
        0,
        [
            0,
            1,
            2
        ],
        1,
        [],
        [
            'ui8',
            'ui16',
            'ui32'
        ],
        [
            '0',
            '0',
            '0'
        ]
    ),
    (
        1,
        [
            3,
            4,
            5
        ],
        1,
        [],
        [
            'si8',
            'si16',
            'si32'
        ],
        [
            '0',
            '0',
            '0'
        ]
    ),
    (
        2,
        [
            6,
            7,
            8
        ],
        1,
        [],
        [
            'ui16',
            'ui32',
            'ui8'
        ],
        [
            '0',
            '0',
            '0'
        ]
    ),
    (
        3,
        [
            9,
            10,
            11
        ],
        1,
        [],
        [
            'si16',
            'si32',
            'si8'
        ],
        [
            '0',
            '0',
            '0'
        ]
    ),
    (
        4,
        [
            12,
            13,
            14
        ],
        1,
        [],
        [
            'ui32',
            'ui16',
            'ui8'
        ],
        [
            '0',
            '0',
            '0'
        ]
    ),
    (
        5,
        [
            15,
            16,
            17
        ],
        1,
        [],
        [
            'si32',
            'si16',
            'si8'
        ],
        [
            '0',
            '0',
            '0'
        ]
    ),
    (
        6,
        [
            18,
            19,
            20,
            21,
            22
        ],
        1,
        [],
        [
            'ui8',
            'ui16',
            'ui16',
            'ui32',
            'ui32'
        ],
        [
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
    name='integer_cast_u8_kernel',
    target="a5",
    backend="vpto",
    mode="explicit",
    source=KERNEL_SOURCE,
)
def kernel_0(
    src_gm: pto.ptr(pto.ui8, "gm"),
    dst16_gm: pto.ptr(pto.ui16, "gm"),
    dst32_gm: pto.ptr(pto.ui32, "gm"),
):
    pass

@pto.jit(
    name='integer_cast_i8_kernel',
    target="a5",
    backend="vpto",
    mode="explicit",
    source=KERNEL_SOURCE,
)
def kernel_1(
    src_gm: pto.ptr(pto.si8, "gm"),
    dst16_gm: pto.ptr(pto.si16, "gm"),
    dst32_gm: pto.ptr(pto.si32, "gm"),
):
    pass

@pto.jit(
    name='integer_cast_u16_kernel',
    target="a5",
    backend="vpto",
    mode="explicit",
    source=KERNEL_SOURCE,
)
def kernel_2(
    src_gm: pto.ptr(pto.ui16, "gm"),
    dst32_gm: pto.ptr(pto.ui32, "gm"),
    dst8_gm: pto.ptr(pto.ui8, "gm"),
):
    pass

@pto.jit(
    name='integer_cast_i16_kernel',
    target="a5",
    backend="vpto",
    mode="explicit",
    source=KERNEL_SOURCE,
)
def kernel_3(
    src_gm: pto.ptr(pto.si16, "gm"),
    dst32_gm: pto.ptr(pto.si32, "gm"),
    dst8_gm: pto.ptr(pto.si8, "gm"),
):
    pass

@pto.jit(
    name='integer_cast_u32_kernel',
    target="a5",
    backend="vpto",
    mode="explicit",
    source=KERNEL_SOURCE,
)
def kernel_4(
    src_gm: pto.ptr(pto.ui32, "gm"),
    dst16_gm: pto.ptr(pto.ui16, "gm"),
    dst8_gm: pto.ptr(pto.ui8, "gm"),
):
    pass

@pto.jit(
    name='integer_cast_i32_kernel',
    target="a5",
    backend="vpto",
    mode="explicit",
    source=KERNEL_SOURCE,
)
def kernel_5(
    src_gm: pto.ptr(pto.si32, "gm"),
    dst16_gm: pto.ptr(pto.si16, "gm"),
    dst8_gm: pto.ptr(pto.si8, "gm"),
):
    pass

@pto.jit(
    name='integer_cast_zero_gap_kernel',
    target="a5",
    backend="vpto",
    mode="explicit",
    source=KERNEL_SOURCE,
)
def kernel_6(
    src8_gm: pto.ptr(pto.ui8, "gm"),
    src16_gm: pto.ptr(pto.ui16, "gm"),
    dst16_gm: pto.ptr(pto.ui16, "gm"),
    dst32_from8_gm: pto.ptr(pto.ui32, "gm"),
    dst32_from16_gm: pto.ptr(pto.ui32, "gm"),
):
    pass


KERNELS = [kernel_0, kernel_1, kernel_2, kernel_3, kernel_4, kernel_5, kernel_6]


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
    name='legacy_vmi_new_integer_cast_layout_runtime',
    grid=1,
    kernel=KERNELS[0],
    make_case=make_case,
    check=check_case,
    run=run_case,
)]

auto_main(globals())
