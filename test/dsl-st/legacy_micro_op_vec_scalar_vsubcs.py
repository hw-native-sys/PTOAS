#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Migrated from test/vpto/cases/micro-op/vec-scalar/vsubcs."""

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
    'EXPRESS OR IMPLIED,\n// INCLUDING, BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTI'
    'CULAR PURPOSE.\n// See LICENSE in the root of the software repository for the full text of the License.\n\n/'
    '/ -----------------------------------------------------------------------------\n// case: micro-op/vec-scala'
    'r/vsubcs\n// family: vec-scalar\n// target_ops: pto.vsubcs\n// scenarios: core-u32-unsigned, full-mask, borr'
    'ow-chain\n// -----------------------------------------------------------------------------\nmodule attribute'
    's {pto.target_arch = "a5", pto.kernel_kind = #pto.kernel_kind<vector>} {\n  func.func @vsubcs_kernel(%arg0: '
    '!pto.ptr<ui32, gm>, %arg1: !pto.ptr<ui32, gm>,\n                           %arg2: !pto.ptr<ui32, gm>, %arg3:'
    ' !pto.ptr<ui8, gm>) attributes {pto.kernel} {\n    %c0 = arith.constant 0 : index\n    %c0_i64 = arith.const'
    'ant 0 : i64\n    %c1_i64 = arith.constant 1 : i64\n    %c128_i64 = arith.constant 128 : i64\n    %c256_i64 ='
    ' arith.constant 256 : i64\n    %c4096_i64 = arith.constant 4096 : i64\n    %c8192_i64 = arith.constant 8192 '
    ': i64\n    %c12288_i64 = arith.constant 12288 : i64\n    %false = arith.constant false\n\n    %ub_lhs = pto.'
    'castptr %c0_i64 : i64 -> !pto.ptr<ui32, ub>\n    %ub_rhs = pto.castptr %c4096_i64 : i64 -> !pto.ptr<ui32, ub'
    '>\n    %ub_out = pto.castptr %c8192_i64 : i64 -> !pto.ptr<ui32, ub>\n    %ub_borrow = pto.castptr %c12288_i6'
    '4 : i64 -> !pto.ptr<ui8, ub>\n    pto.mte_gm_ub %arg0, %ub_lhs, %c0_i64, %c256_i64\n      nburst(%c1_i64, %c'
    '256_i64, %c256_i64)\n      : !pto.ptr<ui32, gm>, !pto.ptr<ui32, ub>, i64, i64, i64, i64, i64\n    pto.mte_gm'
    '_ub %arg1, %ub_rhs, %c0_i64, %c256_i64\n      nburst(%c1_i64, %c256_i64, %c256_i64)\n      : !pto.ptr<ui32, '
    'gm>, !pto.ptr<ui32, ub>, i64, i64, i64, i64, i64\n\n    pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n  '
    '  pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n\n    pto.vecscope {\n      %mask = pto.pset_b32 "PAT_A'
    'LL" : !pto.mask<b32>\n      %borrow_in = pto.pset_b32 "PAT_ALLF" : !pto.mask<b32>\n      %lhs = pto.vlds %ub'
    '_lhs[%c0] : !pto.ptr<ui32, ub>, index -> !pto.vreg<64xui32>\n      %rhs = pto.vlds %ub_rhs[%c0] : !pto.ptr<ui32, ub'
    '>, index -> !pto.vreg<64xui32>\n      %diff, %borrow = pto.vsubcs %lhs, %rhs, %borrow_in, %mask : !pto.vreg<64xui32'
    '>, !pto.vreg<64xui32>, !pto.mask<b32>, !pto.mask<b32> -> !pto.vreg<64xui32>, !pto.mask<b32>\n      pto.vsts '
    '%diff, %ub_out[%c0], %mask : !pto.vreg<64xui32>, !pto.ptr<ui32, ub>, index, !pto.mask<b32>\n      pto.psti %borrow,'
    ' %ub_borrow[%c0], "NORM" : !pto.mask<b32>, !pto.ptr<ui8, ub>, index\n    }\n\n    pto.set_flag["PIPE_V", "PI'
    'PE_MTE3", "EVENT_ID0"]\n    pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n    pto.mte_ub_gm %ub_out, %a'
    'rg2, %c256_i64\n      nburst(%c1_i64, %c256_i64, %c256_i64)\n      : !pto.ptr<ui32, ub>, !pto.ptr<ui32, gm>,'
    ' i64, i64, i64, i64\n    pto.mte_ub_gm %ub_borrow, %arg3, %c128_i64\n      nburst(%c1_i64, %c256_i64, %c256_'
    'i64)\n      : !pto.ptr<ui8, ub>, !pto.ptr<ui8, gm>, i64, i64, i64, i64\n    pto.barrier #pto.pipe<PIPE_ALL>'
    '\n    %__case_merge_guard = arith.constant false\n\n    // inactive cross-case merge from micro-op/vec-scala'
    'r/vsubcs-borrow-boundary\n    scf.if %__case_merge_guard {\n      \n          %c0_cmg17_1 = arith.constant 0'
    ' : index\n          %c0_i64_cmg17_1 = arith.constant 0 : i64\n          %c1_i64_cmg17_1 = arith.constant 1 :'
    ' i64\n          %c128_i64_cmg17_1 = arith.constant 128 : i64\n          %c256_i64_cmg17_1 = arith.constant 2'
    '56 : i64\n          %c4096_i64_cmg17_1 = arith.constant 4096 : i64\n          %c8192_i64_cmg17_1 = arith.con'
    'stant 8192 : i64\n          %c12288_i64_cmg17_1 = arith.constant 12288 : i64\n          %false_cmg17_1 = ari'
    'th.constant false\n      \n          %ub_lhs_cmg17_1 = pto.castptr %c0_i64_cmg17_1 : i64 -> !pto.ptr<ui32, u'
    'b>\n          %ub_rhs_cmg17_1 = pto.castptr %c4096_i64_cmg17_1 : i64 -> !pto.ptr<ui32, ub>\n          %ub_ou'
    't_cmg17_1 = pto.castptr %c8192_i64_cmg17_1 : i64 -> !pto.ptr<ui32, ub>\n          %ub_borrow_cmg17_1 = pto.c'
    'astptr %c12288_i64_cmg17_1 : i64 -> !pto.ptr<ui8, ub>\n          pto.mte_gm_ub %arg0, %ub_lhs_cmg17_1, %c0_i'
    '64_cmg17_1, %c256_i64_cmg17_1\n            nburst(%c1_i64_cmg17_1, %c256_i64_cmg17_1, %c256_i64_cmg17_1)\n  '
    '          : !pto.ptr<ui32, gm>, !pto.ptr<ui32, ub>, i64, i64, i64, i64, i64\n          pto.mte_gm_ub %arg1, '
    '%ub_rhs_cmg17_1, %c0_i64_cmg17_1, %c256_i64_cmg17_1\n            nburst(%c1_i64_cmg17_1, %c256_i64_cmg17_1, '
    '%c256_i64_cmg17_1)\n            : !pto.ptr<ui32, gm>, !pto.ptr<ui32, ub>, i64, i64, i64, i64, i64\n      \n '
    '         pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n          pto.wait_flag["PIPE_MTE2", "PIPE_V", "E'
    'VENT_ID0"]\n      \n          pto.vecscope {\n            %mask_cmg17_1 = pto.pset_b32 "PAT_ALL" : !pto.mask'
    '<b32>\n            %borrow_in_cmg17_1 = pto.pset_b32 "PAT_ALLF" : !pto.mask<b32>\n            %lhs_cmg17_1 ='
    ' pto.vlds %ub_lhs_cmg17_1[%c0_cmg17_1] : !pto.ptr<ui32, ub>, index -> !pto.vreg<64xui32>\n            %rhs_cmg17_1 '
    '= pto.vlds %ub_rhs_cmg17_1[%c0_cmg17_1] : !pto.ptr<ui32, ub>, index -> !pto.vreg<64xui32>\n            %diff_cmg17_'
    '1, %borrow_cmg17_1 = pto.vsubcs %lhs_cmg17_1, %rhs_cmg17_1, %borrow_in_cmg17_1, %mask_cmg17_1 : !pto.vreg<64'
    'xui32>, !pto.vreg<64xui32>, !pto.mask<b32>, !pto.mask<b32> -> !pto.vreg<64xui32>, !pto.mask<b32>\n          '
    '  pto.vsts %diff_cmg17_1, %ub_out_cmg17_1[%c0_cmg17_1], %mask_cmg17_1 : !pto.vreg<64xui32>, !pto.ptr<ui32, u'
    'b>, index, !pto.mask<b32>\n            pto.psti %borrow_cmg17_1, %ub_borrow_cmg17_1[%c0_cmg17_1], "NORM" : !pto.mas'
    'k<b32>, !pto.ptr<ui8, ub>, index\n          }\n      \n          pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_'
    'ID0"]\n          pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n          pto.mte_ub_gm %ub_out_cmg17_1,'
    ' %arg2, %c256_i64_cmg17_1\n            nburst(%c1_i64_cmg17_1, %c256_i64_cmg17_1, %c256_i64_cmg17_1)\n      '
    '      : !pto.ptr<ui32, ub>, !pto.ptr<ui32, gm>, i64, i64, i64, i64\n          pto.mte_ub_gm %ub_borrow_cmg17'
    '_1, %arg3, %c128_i64_cmg17_1\n            nburst(%c1_i64_cmg17_1, %c256_i64_cmg17_1, %c256_i64_cmg17_1)\n   '
    '         : !pto.ptr<ui8, ub>, !pto.ptr<ui8, gm>, i64, i64, i64, i64\n          pto.barrier #pto.pipe<PIPE_AL'
    'L>\n    }\n\n    // inactive cross-case merge from micro-op/vec-scalar/vsubcs\n    scf.if %__case_merge_guar'
    'd {\n      \n          %c0_cmg17_2 = arith.constant 0 : index\n          %c0_i64_cmg17_2 = arith.constant 0 '
    ': i64\n          %c1_i64_cmg17_2 = arith.constant 1 : i64\n          %c128_i64_cmg17_2 = arith.constant 128 '
    ': i64\n          %c256_i64_cmg17_2 = arith.constant 256 : i64\n          %c4096_i64_cmg17_2 = arith.constant'
    ' 4096 : i64\n          %c8192_i64_cmg17_2 = arith.constant 8192 : i64\n          %c12288_i64_cmg17_2 = arith'
    '.constant 12288 : i64\n          %false_cmg17_2 = arith.constant false\n      \n          %ub_lhs_cmg17_2 = '
    'pto.castptr %c0_i64_cmg17_2 : i64 -> !pto.ptr<ui32, ub>\n          %ub_rhs_cmg17_2 = pto.castptr %c4096_i64_'
    'cmg17_2 : i64 -> !pto.ptr<ui32, ub>\n          %ub_out_cmg17_2 = pto.castptr %c8192_i64_cmg17_2 : i64 -> !pt'
    'o.ptr<ui32, ub>\n          %ub_borrow_cmg17_2 = pto.castptr %c12288_i64_cmg17_2 : i64 -> !pto.ptr<ui8, ub>\n'
    '          pto.mte_gm_ub %arg0, %ub_lhs_cmg17_2, %c0_i64_cmg17_2, %c256_i64_cmg17_2\n            nburst(%c1_i'
    '64_cmg17_2, %c256_i64_cmg17_2, %c256_i64_cmg17_2)\n            : !pto.ptr<ui32, gm>, !pto.ptr<ui32, ub>, i64'
    ', i64, i64, i64, i64\n          pto.mte_gm_ub %arg1, %ub_rhs_cmg17_2, %c0_i64_cmg17_2, %c256_i64_cmg17_2\n  '
    '          nburst(%c1_i64_cmg17_2, %c256_i64_cmg17_2, %c256_i64_cmg17_2)\n            : !pto.ptr<ui32, gm>, !'
    'pto.ptr<ui32, ub>, i64, i64, i64, i64, i64\n      \n          pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0'
    '"]\n          pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n      \n          pto.vecscope {\n         '
    '   %mask_cmg17_2 = pto.pset_b32 "PAT_ALL" : !pto.mask<b32>\n            %borrow_in_cmg17_2 = pto.pset_b32 "P'
    'AT_ALLF" : !pto.mask<b32>\n            %lhs_cmg17_2 = pto.vlds %ub_lhs_cmg17_2[%c0_cmg17_2] : !pto.ptr<ui32,'
    ' ub>, index -> !pto.vreg<64xui32>\n            %rhs_cmg17_2 = pto.vlds %ub_rhs_cmg17_2[%c0_cmg17_2] : !pto.ptr<ui32'
    ', ub>, index -> !pto.vreg<64xui32>\n            %diff_cmg17_2, %borrow_cmg17_2 = pto.vsubcs %lhs_cmg17_2, %rhs_cmg1'
    '7_2, %borrow_in_cmg17_2, %mask_cmg17_2 : !pto.vreg<64xui32>, !pto.vreg<64xui32>, !pto.mask<b32>, !pto.mask<b'
    '32> -> !pto.vreg<64xui32>, !pto.mask<b32>\n            pto.vsts %diff_cmg17_2, %ub_out_cmg17_2[%c0_cmg17_2],'
    ' %mask_cmg17_2 : !pto.vreg<64xui32>, !pto.ptr<ui32, ub>, index, !pto.mask<b32>\n            pto.psti %borrow_cmg17_'
    '2, %ub_borrow_cmg17_2[%c0_cmg17_2], "NORM" : !pto.mask<b32>, !pto.ptr<ui8, ub>, index\n          }\n      \n'
    '          pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n          pto.wait_flag["PIPE_V", "PIPE_MTE3", "'
    'EVENT_ID0"]\n          pto.mte_ub_gm %ub_out_cmg17_2, %arg2, %c256_i64_cmg17_2\n            nburst(%c1_i64_c'
    'mg17_2, %c256_i64_cmg17_2, %c256_i64_cmg17_2)\n            : !pto.ptr<ui32, ub>, !pto.ptr<ui32, gm>, i64, i6'
    '4, i64, i64\n          pto.mte_ub_gm %ub_borrow_cmg17_2, %arg3, %c128_i64_cmg17_2\n            nburst(%c1_i6'
    '4_cmg17_2, %c256_i64_cmg17_2, %c256_i64_cmg17_2)\n            : !pto.ptr<ui8, ub>, !pto.ptr<ui8, gm>, i64, i'
    '64, i64, i64\n          pto.barrier #pto.pipe<PIPE_ALL>\n    }\n\n    // inactive cross-case merge from micr'
    'o-op/vec-scalar/vsubcs-borrow-boundary\n    scf.if %__case_merge_guard {\n      \n          %c0_cmg17_3 = ar'
    'ith.constant 0 : index\n          %c0_i64_cmg17_3 = arith.constant 0 : i64\n          %c1_i64_cmg17_3 = arit'
    'h.constant 1 : i64\n          %c128_i64_cmg17_3 = arith.constant 128 : i64\n          %c256_i64_cmg17_3 = ar'
    'ith.constant 256 : i64\n          %c4096_i64_cmg17_3 = arith.constant 4096 : i64\n          %c8192_i64_cmg17'
    '_3 = arith.constant 8192 : i64\n          %c12288_i64_cmg17_3 = arith.constant 12288 : i64\n          %false'
    '_cmg17_3 = arith.constant false\n      \n          %ub_lhs_cmg17_3 = pto.castptr %c0_i64_cmg17_3 : i64 -> !p'
    'to.ptr<ui32, ub>\n          %ub_rhs_cmg17_3 = pto.castptr %c4096_i64_cmg17_3 : i64 -> !pto.ptr<ui32, ub>\n  '
    '        %ub_out_cmg17_3 = pto.castptr %c8192_i64_cmg17_3 : i64 -> !pto.ptr<ui32, ub>\n          %ub_borrow_c'
    'mg17_3 = pto.castptr %c12288_i64_cmg17_3 : i64 -> !pto.ptr<ui8, ub>\n          pto.mte_gm_ub %arg0, %ub_lhs_'
    'cmg17_3, %c0_i64_cmg17_3, %c256_i64_cmg17_3\n            nburst(%c1_i64_cmg17_3, %c256_i64_cmg17_3, %c256_i6'
    '4_cmg17_3)\n            : !pto.ptr<ui32, gm>, !pto.ptr<ui32, ub>, i64, i64, i64, i64, i64\n          pto.mte'
    '_gm_ub %arg1, %ub_rhs_cmg17_3, %c0_i64_cmg17_3, %c256_i64_cmg17_3\n            nburst(%c1_i64_cmg17_3, %c256'
    '_i64_cmg17_3, %c256_i64_cmg17_3)\n            : !pto.ptr<ui32, gm>, !pto.ptr<ui32, ub>, i64, i64, i64, i64, '
    'i64\n      \n          pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n          pto.wait_flag["PIPE_MTE2"'
    ', "PIPE_V", "EVENT_ID0"]\n      \n          pto.vecscope {\n            %mask_cmg17_3 = pto.pset_b32 "PAT_AL'
    'L" : !pto.mask<b32>\n            %borrow_in_cmg17_3 = pto.pset_b32 "PAT_ALLF" : !pto.mask<b32>\n            '
    '%lhs_cmg17_3 = pto.vlds %ub_lhs_cmg17_3[%c0_cmg17_3] : !pto.ptr<ui32, ub>, index -> !pto.vreg<64xui32>\n           '
    ' %rhs_cmg17_3 = pto.vlds %ub_rhs_cmg17_3[%c0_cmg17_3] : !pto.ptr<ui32, ub>, index -> !pto.vreg<64xui32>\n          '
    '  %diff_cmg17_3, %borrow_cmg17_3 = pto.vsubcs %lhs_cmg17_3, %rhs_cmg17_3, %borrow_in_cmg17_3, %mask_cmg17_3 '
    ': !pto.vreg<64xui32>, !pto.vreg<64xui32>, !pto.mask<b32>, !pto.mask<b32> -> !pto.vreg<64xui32>, !pto.mask<b3'
    '2>\n            pto.vsts %diff_cmg17_3, %ub_out_cmg17_3[%c0_cmg17_3], %mask_cmg17_3 : !pto.vreg<64xui32>, !p'
    'to.ptr<ui32, ub>, index, !pto.mask<b32>\n            pto.psti %borrow_cmg17_3, %ub_borrow_cmg17_3[%c0_cmg17_3], "NO'
    'RM" : !pto.mask<b32>, !pto.ptr<ui8, ub>, index\n          }\n      \n          pto.set_flag["PIPE_V", "PIPE_'
    'MTE3", "EVENT_ID0"]\n          pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n          pto.mte_ub_gm %u'
    'b_out_cmg17_3, %arg2, %c256_i64_cmg17_3\n            nburst(%c1_i64_cmg17_3, %c256_i64_cmg17_3, %c256_i64_cm'
    'g17_3)\n            : !pto.ptr<ui32, ub>, !pto.ptr<ui32, gm>, i64, i64, i64, i64\n          pto.mte_ub_gm %u'
    'b_borrow_cmg17_3, %arg3, %c128_i64_cmg17_3\n            nburst(%c1_i64_cmg17_3, %c256_i64_cmg17_3, %c256_i64'
    '_cmg17_3)\n            : !pto.ptr<ui8, ub>, !pto.ptr<ui8, gm>, i64, i64, i64, i64\n          pto.barrier #pt'
    'o.pipe<PIPE_ALL>\n    }\n    return\n  }\n}\n'
)

def _golden_scope():
    _files = {}

    output_dir = None

    # Copyright (c) 2026 Huawei Technologies Co., Ltd.
    # This program is free software, you can redistribute it and/or modify it under the terms and conditions of
    # CANN Open Software License Agreement Version 2.0 (the "License").
    # Please refer to the License for details. You may not use this file except in compliance with the License.
    # THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
    # INCLUDING, BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE.
    # See LICENSE in the root of the software repository for the full text of the License.

    # case: micro-op/vec-scalar/vsubcs
    # family: vec-scalar
    # target_ops: pto.vsubcs
    # scenarios: core-u32-unsigned, full-mask, borrow-chain

    import argparse
    from pathlib import Path

    import numpy as np


    LANES = 64
    SEED = 19


    def pack_mask_nibbles(bits):
        out = np.zeros(256, dtype=np.uint8)
        for idx, bit in enumerate(bits):
            if bit:
                out[idx // 2] |= np.uint8(0x1 if idx % 2 == 0 else 0x10)
        return out


    def _generate(seed: int) -> dict:
        rng = np.random.default_rng(seed)
        lhs = rng.integers(0, 0x100000000, size=LANES, dtype=np.uint32)
        rhs = rng.integers(0, 0x100000000, size=LANES, dtype=np.uint32)
        lhs[:3] = [3, 0, 0x80000000]
        rhs[:3] = [7, 0, 0x7FFFFFFF]
        lhs64 = lhs.astype(np.uint64)
        rhs64 = rhs.astype(np.uint64)
        carry_in = np.zeros(LANES, dtype=np.uint64)
        incoming_borrow = np.uint64(1) - carry_in
        result = ((lhs64 - rhs64 - incoming_borrow) & np.uint64(0xFFFFFFFF)).astype(np.uint32)
        not_borrow = lhs64 >= rhs64 + incoming_borrow

        _files["v1.bin"] = lhs
        _files["v2.bin"] = rhs
        _files["v3.bin"] = np.zeros(LANES, dtype=np.uint32)
        _files["v4.bin"] = np.zeros(256, dtype=np.uint8)
        _files["golden_v3.bin"] = result
        _files["golden_v4.bin"] = pack_mask_nibbles(not_borrow)


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

    # case: micro-op/vec-scalar/vsubcs
    # family: vec-scalar
    # target_ops: pto.vsubcs
    # scenarios: core-u32-unsigned, full-mask, borrow-boundary

    import os
    import sys
    import numpy as np


    REPEAT_BYTES = 256
    LOGICAL_ELEMS = 64
    SRC_ELEM_BYTES = 4


    def _ceil_div(x, y):
        return (x + y - 1) // y


    def _packed_pred_storage_bytes(logical_elems, src_elem_bytes):
        repeat_elems = REPEAT_BYTES // src_elem_bytes
        repeat_times = _ceil_div(logical_elems, repeat_elems) + 1
        loop_count = repeat_times // 2
        return loop_count * 16


    def compare_result():
        golden = np.fromfile("golden_v3.bin", dtype=np.uint32, count=64)
        output = np.fromfile("v3.bin", dtype=np.uint32, count=64)
        return golden.shape == output.shape and np.array_equal(golden, output)


    def compare_borrow():
        prefix_bytes = _packed_pred_storage_bytes(LOGICAL_ELEMS, SRC_ELEM_BYTES)
        golden = np.fromfile("golden_v4.bin", dtype=np.uint8)
        output = np.fromfile("v4.bin", dtype=np.uint8)
        if golden.size < prefix_bytes or output.size < prefix_bytes:
            return False
        return np.array_equal(golden[:prefix_bytes], output[:prefix_bytes])


    def main():
        strict = os.getenv("COMPARE_STRICT", "1") != "0"
        ok = compare_result() and compare_borrow()
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
        'ui32',
        'arg0'
    ),
    (
        'v2',
        'ui32',
        'arg1'
    ),
    (
        'v3',
        'ui32',
        'arg2'
    ),
    (
        'v4',
        'ui8',
        'arg3'
    )
]
SCALARS = []
SCALAR_SOURCES = []
SCALAR_DTYPES = []


@pto.jit(
    name='vsubcs_kernel',
    target="a5",
    backend="vpto",
    mode="explicit",
    source=KERNEL_SOURCE,
)
def kernel(
    arg0: pto.ptr(pto.ui32, "gm"),
    arg1: pto.ptr(pto.ui32, "gm"),
    arg2: pto.ptr(pto.ui32, "gm"),
    arg3: pto.ptr(pto.ui8, "gm"),
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
    name='legacy_micro_op_vec_scalar_vsubcs',
    grid=1,
    kernel=kernel,
    make_case=make_case,
    check=check_case,
)]

auto_main(globals())
