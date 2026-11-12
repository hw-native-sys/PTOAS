#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Migrated from test/vpto/cases/micro-op/vector-load-store/dma-copy-rearrange."""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

from common import auto_main
from ptodsl import pto

KERNEL_SOURCE = (
    '// -----------------------------------------------------------------------------\n// case: micro-op/vector-l'
    'oad-store/dma-copy-rearrange\n// family: micro-op/vector-load-store\n// target_ops: pto.copy_gm_to_ubuf, pto'
    '.mte_ub_ub, pto.copy_ubuf_to_gm\n// scenarios: i16, ub-rearrange, permute-4x16-rows\n// --------------------'
    '---------------------------------------------------------\nmodule attributes {pto.target_arch = "a5", pto.ke'
    'rnel_kind = #pto.kernel_kind<vector>} {\n  func.func @dma_copy_rearrange_kernel(%arg0: !pto.ptr<i16, gm>,\n '
    '                                      %arg1: !pto.ptr<i16, gm>,\n                                       %arg'
    '2: i64,\n                                       %arg3: i64,\n                                       %arg4: i'
    '64,\n                                       %arg5: i64) attributes {pto.kernel} {\n    %false = arith.consta'
    'nt false\n    %c0_i64 = arith.constant 0 : i64\n    %c1_i64 = arith.constant 1 : i64\n    %c32_i64 = arith.c'
    'onstant 32 : i64\n    %c64_i64 = arith.constant 64 : i64\n    %c96_i64 = arith.constant 96 : i64\n    %c128_'
    'i64 = arith.constant 128 : i64\n    %c160_i64 = arith.constant 160 : i64\n    %c192_i64 = arith.constant 192'
    ' : i64\n    %c224_i64 = arith.constant 224 : i64\n    %c256_i64 = arith.constant 256 : i64\n\n    %ub_in = p'
    'to.castptr %c0_i64 : i64 -> !pto.ptr<i16, ub>\n    %ub_out = pto.castptr %c128_i64 : i64 -> !pto.ptr<i16, ub'
    '>\n\n    pto.set_loop_size_outtoub %c1_i64, %c1_i64 : i64, i64\n    pto.copy_gm_to_ubuf %arg0, %ub_in, %c0_i'
    '64, %c1_i64, %c128_i64,\n                        %c0_i64, %c0_i64, %false, %c0_i64, %c128_i64, %c128_i64\n  '
    '    : !pto.ptr<i16, gm>, !pto.ptr<i16, ub>, i64, i64, i64, i64, i64, i1, i64, i64, i64\n\n    pto.barrier #p'
    'to.pipe<PIPE_ALL>\n\n    %src_row0 = pto.castptr %c0_i64 : i64 -> !pto.ptr<i16, ub>\n    %src_row1 = pto.cas'
    'tptr %c32_i64 : i64 -> !pto.ptr<i16, ub>\n    %src_row2 = pto.castptr %c64_i64 : i64 -> !pto.ptr<i16, ub>\n '
    '   %src_row3 = pto.castptr %c96_i64 : i64 -> !pto.ptr<i16, ub>\n\n    %dst_row0 = pto.castptr %c128_i64 : i6'
    '4 -> !pto.ptr<i16, ub>\n    %dst_row1 = pto.castptr %c160_i64 : i64 -> !pto.ptr<i16, ub>\n    %dst_row2 = pt'
    'o.castptr %c192_i64 : i64 -> !pto.ptr<i16, ub>\n    %dst_row3 = pto.castptr %c224_i64 : i64 -> !pto.ptr<i16,'
    ' ub>\n\n    pto.mte_ub_ub %src_row2, %dst_row0, %arg3\n      nburst(%arg2, %arg4, %arg5)\n      : !pto.ptr<i'
    '16, ub>, !pto.ptr<i16, ub>, i64, i64, i64, i64\n    pto.mte_ub_ub %src_row0, %dst_row1, %arg3\n      nburst('
    '%arg2, %arg4, %arg5)\n      : !pto.ptr<i16, ub>, !pto.ptr<i16, ub>, i64, i64, i64, i64\n    pto.mte_ub_ub %s'
    'rc_row3, %dst_row2, %arg3\n      nburst(%arg2, %arg4, %arg5)\n      : !pto.ptr<i16, ub>, !pto.ptr<i16, ub>, '
    'i64, i64, i64, i64\n    pto.mte_ub_ub %src_row1, %dst_row3, %arg3\n      nburst(%arg2, %arg4, %arg5)\n      '
    ': !pto.ptr<i16, ub>, !pto.ptr<i16, ub>, i64, i64, i64, i64\n\n    pto.barrier #pto.pipe<PIPE_ALL>\n\n    pto'
    '.set_loop_size_ubtoout %c1_i64, %c1_i64 : i64, i64\n    pto.copy_ubuf_to_gm %ub_out, %arg1, %c0_i64, %c1_i64'
    ', %c128_i64, %c0_i64, %c128_i64, %c128_i64\n      : !pto.ptr<i16, ub>, !pto.ptr<i16, gm>, i64, i64, i64, i64'
    ', i64, i64\n    pto.barrier #pto.pipe<PIPE_ALL>\n    %__case_merge_guard = arith.constant false\n\n    // in'
    'active cross-case merge from micro-op/vector-load-store/vlds-ds-b16\n    scf.if %__case_merge_guard {\n     '
    ' \n          %c0_cmg21_1 = arith.constant 0 : index\n          %c1_cmg21_1 = arith.constant 1 : index\n     '
    '     %c128_cmg21_1 = arith.constant 128 : index\n          %c1024_cmg21_1 = arith.constant 1024 : index\n   '
    '       %c0_i64_cmg21_1 = arith.constant 0 : i64\n          %c1_i64_cmg21_1 = arith.constant 1 : i64\n       '
    '   %c32_i64_cmg21_1 = arith.constant 32 : i64\n          %c128_i64_cmg21_1 = arith.constant 128 : i64\n     '
    '     %c4096_i64_cmg21_1 = arith.constant 4096 : i64\n          %c1024_i32_cmg21_1 = arith.constant 1024 : i3'
    '2\n      \n          %ub_in_cmg21_1 = pto.castptr %c0_i64_cmg21_1 : i64 -> !pto.ptr<i16, ub>\n          %ub_'
    'out_cmg21_1 = pto.castptr %c4096_i64_cmg21_1 : i64 -> !pto.ptr<i16, ub>\n      \n          %false_cmg21_1 = '
    'arith.constant false\n          pto.mte_gm_ub %arg0, %ub_in_cmg21_1, %c0_i64_cmg21_1, %c128_i64_cmg21_1\n   '
    '         nburst(%c32_i64_cmg21_1, %c128_i64_cmg21_1, %c128_i64_cmg21_1)\n            : !pto.ptr<i16, gm>, !p'
    'to.ptr<i16, ub>, i64, i64, i64, i64, i64\n      \n          pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]'
    '\n          pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n      \n          pto.vecscope {\n           '
    ' %__cmg21_1:1 = scf.for %offset_cmg21_1 = %c0_cmg21_1 to %c1024_cmg21_1 step %c128_cmg21_1 iter_args(%remain'
    'ing_cmg21_1 = %c1024_i32_cmg21_1) -> (i32) {\n              %mask_cmg21_1, %next_remaining_cmg21_1 = pto.plt'
    '_b16 %remaining_cmg21_1 : i32 -> !pto.mask<b16>, i32\n              %out_cmg21_1 = pto.vlds %ub_in_cmg21_1[%'
    'offset_cmg21_1] {dist = "DS_B16"} : !pto.ptr<i16, ub>, index -> !pto.vreg<128xi16>\n              pto.vsts %out_cmg'
    '21_1, %ub_out_cmg21_1[%offset_cmg21_1], %mask_cmg21_1 : !pto.vreg<128xi16>, !pto.ptr<i16, ub>, index, !pto.mask<b16'
    '>\n              scf.yield %next_remaining_cmg21_1 : i32\n            }\n          }\n      \n          pto.'
    'set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n          pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n '
    '         pto.mte_ub_gm %ub_out_cmg21_1, %arg1, %c128_i64_cmg21_1\n            nburst(%c32_i64_cmg21_1, %c128'
    '_i64_cmg21_1, %c128_i64_cmg21_1)\n            : !pto.ptr<i16, ub>, !pto.ptr<i16, gm>, i64, i64, i64, i64\n  '
    '        pto.barrier #pto.pipe<PIPE_ALL>\n    }\n\n    // inactive cross-case merge from micro-op/vector-load'
    '-store/vlds-us-b16\n    scf.if %__case_merge_guard {\n      \n          %c0_cmg21_2 = arith.constant 0 : ind'
    'ex\n          %c1_cmg21_2 = arith.constant 1 : index\n          %c128_cmg21_2 = arith.constant 128 : index\n'
    '          %c1024_cmg21_2 = arith.constant 1024 : index\n          %c0_i64_cmg21_2 = arith.constant 0 : i64\n'
    '          %c1_i64_cmg21_2 = arith.constant 1 : i64\n          %c32_i64_cmg21_2 = arith.constant 32 : i64\n  '
    '        %c128_i64_cmg21_2 = arith.constant 128 : i64\n          %c4096_i64_cmg21_2 = arith.constant 4096 : i'
    '64\n          %c1024_i32_cmg21_2 = arith.constant 1024 : i32\n      \n          %ub_in_cmg21_2 = pto.castptr'
    ' %c0_i64_cmg21_2 : i64 -> !pto.ptr<i16, ub>\n          %ub_out_cmg21_2 = pto.castptr %c4096_i64_cmg21_2 : i6'
    '4 -> !pto.ptr<i16, ub>\n      \n          %false_cmg21_2 = arith.constant false\n          pto.mte_gm_ub %ar'
    'g0, %ub_in_cmg21_2, %c0_i64_cmg21_2, %c128_i64_cmg21_2\n            nburst(%c32_i64_cmg21_2, %c128_i64_cmg21'
    '_2, %c128_i64_cmg21_2)\n            : !pto.ptr<i16, gm>, !pto.ptr<i16, ub>, i64, i64, i64, i64, i64\n      '
    '\n          pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n          pto.wait_flag["PIPE_MTE2", "PIPE_V",'
    ' "EVENT_ID0"]\n      \n          pto.vecscope {\n            %__cmg21_2:1 = scf.for %offset_cmg21_2 = %c0_cm'
    'g21_2 to %c1024_cmg21_2 step %c128_cmg21_2 iter_args(%remaining_cmg21_2 = %c1024_i32_cmg21_2) -> (i32) {\n  '
    '            %mask_cmg21_2, %next_remaining_cmg21_2 = pto.plt_b16 %remaining_cmg21_2 : i32 -> !pto.mask<b16>,'
    ' i32\n              %out_cmg21_2 = pto.vlds %ub_in_cmg21_2[%offset_cmg21_2] {dist = "US_B16"} : !pto.ptr<i16'
    ', ub>, index -> !pto.vreg<128xi16>\n              pto.vsts %out_cmg21_2, %ub_out_cmg21_2[%offset_cmg21_2], %mask_cm'
    'g21_2 : !pto.vreg<128xi16>, !pto.ptr<i16, ub>, index, !pto.mask<b16>\n              scf.yield %next_remaining_cmg21'
    '_2 : i32\n            }\n          }\n      \n          pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n  '
    '        pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n          pto.mte_ub_gm %ub_out_cmg21_2, %arg1, %'
    'c128_i64_cmg21_2\n            nburst(%c32_i64_cmg21_2, %c128_i64_cmg21_2, %c128_i64_cmg21_2)\n            : '
    '!pto.ptr<i16, ub>, !pto.ptr<i16, gm>, i64, i64, i64, i64\n          pto.barrier #pto.pipe<PIPE_ALL>\n    }\n'
    '\n    // inactive cross-case merge from micro-op/vector-load-store/vsts-1pt-b16\n    scf.if %__case_merge_gu'
    'ard {\n      \n          %c0_cmg21_3 = arith.constant 0 : index\n          %c1_cmg21_3 = arith.constant 1 : '
    'index\n          %c128_cmg21_3 = arith.constant 128 : index\n          %c1024_cmg21_3 = arith.constant 1024 '
    ': index\n          %c0_i64_cmg21_3 = arith.constant 0 : i64\n          %c1_i64_cmg21_3 = arith.constant 1 : '
    'i64\n          %c32_i64_cmg21_3 = arith.constant 32 : i64\n          %c128_i64_cmg21_3 = arith.constant 128 '
    ': i64\n          %c4096_i64_cmg21_3 = arith.constant 4096 : i64\n          %c1024_i32_cmg21_3 = arith.consta'
    'nt 1024 : i32\n      \n          %ub_in_cmg21_3 = pto.castptr %c0_i64_cmg21_3 : i64 -> !pto.ptr<i16, ub>\n  '
    '        %ub_out_cmg21_3 = pto.castptr %c4096_i64_cmg21_3 : i64 -> !pto.ptr<i16, ub>\n      \n          %fals'
    'e_cmg21_3 = arith.constant false\n          pto.mte_gm_ub %arg0, %ub_in_cmg21_3, %c0_i64_cmg21_3, %c128_i64_'
    'cmg21_3\n            nburst(%c32_i64_cmg21_3, %c128_i64_cmg21_3, %c128_i64_cmg21_3)\n            : !pto.ptr<'
    'i16, gm>, !pto.ptr<i16, ub>, i64, i64, i64, i64, i64\n      \n          pto.set_flag["PIPE_MTE2", "PIPE_V", '
    '"EVENT_ID0"]\n          pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n      \n          pto.vecscope {'
    '\n            %__cmg21_3:1 = scf.for %offset_cmg21_3 = %c0_cmg21_3 to %c1024_cmg21_3 step %c128_cmg21_3 iter'
    '_args(%remaining_cmg21_3 = %c1024_i32_cmg21_3) -> (i32) {\n              %mask_cmg21_3, %next_remaining_cmg2'
    '1_3 = pto.plt_b16 %remaining_cmg21_3 : i32 -> !pto.mask<b16>, i32\n              %vec_cmg21_3 = pto.vlds %ub'
    '_in_cmg21_3[%offset_cmg21_3] : !pto.ptr<i16, ub>, index -> !pto.vreg<128xi16>\n              pto.vsts %vec_cmg21_3,'
    ' %ub_out_cmg21_3[%offset_cmg21_3], %mask_cmg21_3 {dist = "1PT_B16"} : !pto.vreg<128xi16>, !pto.ptr<i16, ub>, index,'
    ' !pto.mask<b16>\n              scf.yield %next_remaining_cmg21_3 : i32\n            }\n          }\n      \n'
    '          pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n          pto.wait_flag["PIPE_V", "PIPE_MTE3", "'
    'EVENT_ID0"]\n          pto.mte_ub_gm %ub_out_cmg21_3, %arg1, %c128_i64_cmg21_3\n            nburst(%c32_i64_'
    'cmg21_3, %c128_i64_cmg21_3, %c128_i64_cmg21_3)\n            : !pto.ptr<i16, ub>, !pto.ptr<i16, gm>, i64, i64'
    ', i64, i64\n          pto.barrier #pto.pipe<PIPE_ALL>\n    }\n\n    // inactive cross-case merge from micro-'
    'op/vector-load-store/vsts-pk-b16\n    scf.if %__case_merge_guard {\n      \n          %c0_cmg21_4 = arith.co'
    'nstant 0 : index\n          %c1_cmg21_4 = arith.constant 1 : index\n          %c128_cmg21_4 = arith.constant'
    ' 128 : index\n          %c1024_cmg21_4 = arith.constant 1024 : index\n          %c0_i64_cmg21_4 = arith.cons'
    'tant 0 : i64\n          %c1_i64_cmg21_4 = arith.constant 1 : i64\n          %c32_i64_cmg21_4 = arith.constan'
    't 32 : i64\n          %c128_i64_cmg21_4 = arith.constant 128 : i64\n          %c4096_i64_cmg21_4 = arith.con'
    'stant 4096 : i64\n          %c1024_i32_cmg21_4 = arith.constant 1024 : i32\n      \n          %ub_in_cmg21_4'
    ' = pto.castptr %c0_i64_cmg21_4 : i64 -> !pto.ptr<i16, ub>\n          %ub_out_cmg21_4 = pto.castptr %c4096_i6'
    '4_cmg21_4 : i64 -> !pto.ptr<i16, ub>\n      \n          %false_cmg21_4 = arith.constant false\n          pto'
    '.mte_gm_ub %arg0, %ub_in_cmg21_4, %c0_i64_cmg21_4, %c128_i64_cmg21_4\n            nburst(%c32_i64_cmg21_4, %'
    'c128_i64_cmg21_4, %c128_i64_cmg21_4)\n            : !pto.ptr<i16, gm>, !pto.ptr<i16, ub>, i64, i64, i64, i64'
    ', i64\n      \n          pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n          pto.wait_flag["PIPE_MTE'
    '2", "PIPE_V", "EVENT_ID0"]\n      \n          pto.vecscope {\n            %__cmg21_4:1 = scf.for %offset_cmg'
    '21_4 = %c0_cmg21_4 to %c1024_cmg21_4 step %c128_cmg21_4 iter_args(%remaining_cmg21_4 = %c1024_i32_cmg21_4) -'
    '> (i32) {\n              %mask_cmg21_4, %next_remaining_cmg21_4 = pto.plt_b16 %remaining_cmg21_4 : i32 -> !p'
    'to.mask<b16>, i32\n              %vec_cmg21_4 = pto.vlds %ub_in_cmg21_4[%offset_cmg21_4] : !pto.ptr<i16, ub>, index'
    ' -> !pto.vreg<128xi16>\n              pto.vsts %vec_cmg21_4, %ub_out_cmg21_4[%offset_cmg21_4], %mask_cmg21_4'
    ' {dist = "PK_B16"} : !pto.vreg<128xi16>, !pto.ptr<i16, ub>, index, !pto.mask<b16>\n              scf.yield %next_re'
    'maining_cmg21_4 : i32\n            }\n          }\n      \n          pto.set_flag["PIPE_V", "PIPE_MTE3", "EV'
    'ENT_ID0"]\n          pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n          pto.mte_ub_gm %ub_out_cmg2'
    '1_4, %arg1, %c128_i64_cmg21_4\n            nburst(%c32_i64_cmg21_4, %c128_i64_cmg21_4, %c128_i64_cmg21_4)\n '
    '           : !pto.ptr<i16, ub>, !pto.ptr<i16, gm>, i64, i64, i64, i64\n          pto.barrier #pto.pipe<PIPE_'
    'ALL>\n    }\n    return\n  }\n}\n'
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

    # case: micro-op/vector-load-store/dma-copy-rearrange
    # family: micro-op/vector-load-store
    # target_ops: pto.copy_gm_to_ubuf, pto.mte_ub_ub, pto.copy_ubuf_to_gm
    # scenarios: i16, ub-rearrange, permute-4x16-rows

    import argparse
    from pathlib import Path

    import numpy as np


    ROWS = 4
    COLS = 16


    def _generate(seed: int) -> dict:
        v1 = np.arange(ROWS * COLS, dtype=np.int16).reshape(ROWS, COLS)
        v2 = np.zeros((ROWS, COLS), dtype=np.int16)
        golden_v2 = v1[[2, 0, 3, 1], :].copy()

        _files["v1.bin"] = v1.reshape(-1)
        _files["v2.bin"] = v2.reshape(-1)
        _files["golden_v2.bin"] = golden_v2.reshape(-1)


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

    # case: micro-op/vector-load-store/dma-copy-rearrange
    # family: micro-op/vector-load-store
    # target_ops: pto.copy_gm_to_ubuf, pto.mte_ub_ub, pto.copy_ubuf_to_gm
    # scenarios: i16, ub-rearrange, permute-4x16-rows

    import os
    import sys

    import numpy as np


    def compare_bin(golden_path: str, output_path: str) -> bool:
        if not os.path.exists(golden_path) or not os.path.exists(output_path):
            return False
        golden = np.fromfile(golden_path, dtype=np.int16)
        output = np.fromfile(output_path, dtype=np.int16)
        if golden.shape != output.shape:
            print(f"[ERROR] shape mismatch: {golden.shape} vs {output.shape}")
            return False
        if np.array_equal(golden, output):
            return True
        diff = np.nonzero(golden != output)[0]
        idx = int(diff[0]) if diff.size else 0
        print(f"[ERROR] first mismatch at idx={idx}: golden={int(golden[idx])}, out={int(output[idx])}")
        return False


    def main() -> None:
        strict = os.getenv("COMPARE_STRICT", "1") != "0"
        ok = compare_bin("golden_v2.bin", "v2.bin")
        if not ok:
            if strict:
                print("[ERROR] compare failed")
                sys.exit(2)
            print("[WARN] compare failed (non-gating)")
            return
        print("[INFO] compare passed")


    return locals()

HELPERS = {}
SEED_EXPR = None
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
        'i16',
        'arg0'
    ),
    (
        'v2',
        'i16',
        'arg1'
    )
]
SCALARS = [1, 1, 0, 0]
SCALAR_SOURCES = []
SCALAR_DTYPES = [
    'i64',
    'i64',
    'i64',
    'i64'
]


@pto.jit(
    name='dma_copy_rearrange_kernel',
    target="a5",
    backend="vpto",
    mode="explicit",
    source=KERNEL_SOURCE,
)
def kernel(
    arg0: pto.ptr(pto.i16, "gm"),
    arg1: pto.ptr(pto.i16, "gm"),
    arg2: pto.i64,
    arg3: pto.i64,
    arg4: pto.i64,
    arg5: pto.i64,
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
    name='legacy_micro_op_vector_load_store_dma_copy_rearrange',
    grid=1,
    kernel=kernel,
    make_case=make_case,
    check=check_case,
)]

auto_main(globals())
