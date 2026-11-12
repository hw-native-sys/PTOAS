#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Migrated from test/vpto/cases/micro-op/simt/simt-float-convert-core."""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

from common import auto_main
from ptodsl import pto

KERNEL_SOURCE = (
    'module attributes {pto.target_arch = "a5", pto.kernel_kind = #pto.kernel_kind<vector>} {\n  func.func @simt_'
    'float_convert_core_kernel(%arg0: !pto.ptr<i32, gm>) attributes {pto.aicore} {\n    %c0_i64 = arith.constant '
    '0 : i64\n    %c128_i64 = arith.constant 128 : i64\n    %c32_i64 = arith.constant 32 : i64\n    %dim_z = arit'
    'h.constant 1 : i32\n    %dim_y = arith.constant 1 : i32\n    %dim_x = arith.constant 32 : i32\n\n    %ub_out'
    ' = pto.castptr %c0_i64 : i64 -> !pto.ptr<i32, ub>\n    pto.simt_launch @simt_float_convert_core_body<<<%dim_'
    'x, %dim_y, %dim_z>>>(%ub_out) : (!pto.ptr<i32, ub>) -> ()\n\n    pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_'
    'ID0"]\n    pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n    pto.mte_ub_gm %ub_out, %arg0, %c128_i64\n '
    '     nburst(%c32_i64, %c128_i64, %c128_i64)\n      : !pto.ptr<i32, ub>, !pto.ptr<i32, gm>, i64, i64, i64, i6'
    '4\n    pto.barrier #pto.pipe<PIPE_ALL>\n    return\n  }\n\n  func.func @simt_float_convert_core_body(%dst: !'
    'pto.ptr<i32, ub>) attributes {pto.simt_entry} {\n    %c0 = arith.constant 0 : index\n    %c1 = arith.constan'
    't 1 : index\n    %c2 = arith.constant 2 : index\n    %c3 = arith.constant 3 : index\n    %c4 = arith.constan'
    't 4 : index\n    %c5 = arith.constant 5 : index\n    %c6 = arith.constant 6 : index\n    %c7 = arith.constan'
    't 7 : index\n    %c8 = arith.constant 8 : index\n    %c9 = arith.constant 9 : index\n    %c10 = arith.consta'
    'nt 10 : index\n    %c11 = arith.constant 11 : index\n    %c12 = arith.constant 12 : index\n    %c13 = arith.'
    'constant 13 : index\n    %c14 = arith.constant 14 : index\n    %c15 = arith.constant 15 : index\n    %c16_i3'
    '2 = arith.constant 16 : i32\n    %c32_i32 = arith.constant 32 : i32\n    %c64_i32 = arith.constant 64 : i32'
    '\n    %c96_i32 = arith.constant 96 : i32\n    %c128_i32 = arith.constant 128 : i32\n    %c224_i32 = arith.co'
    'nstant 224 : i32\n    %c544_i32 = arith.constant 544 : i32\n\n    %f32_neg = arith.constant -3.000000e+00 : '
    'f32\n    %f32_one = arith.constant 1.000000e+00 : f32\n    %f32_two = arith.constant 2.000000e+00 : f32\n   '
    ' %f32_four = arith.constant 4.000000e+00 : f32\n    %f16_one = arith.constant 1.000000e+00 : f16\n    %f16_t'
    'wo = arith.constant 2.000000e+00 : f16\n    %f16_four = arith.constant 4.000000e+00 : f16\n    %bf16_one = a'
    'rith.constant 1.000000e+00 : bf16\n    %bf16_two = arith.constant 2.000000e+00 : bf16\n    %bf16_four = arit'
    'h.constant 4.000000e+00 : bf16\n    %i32_neg = arith.constant -4 : i32\n    %u32_four = arith.constant 4 : i'
    '32\n    %f16_three = arith.constant 3.000000e+00 : f16\n    %f32_pos_half = arith.constant 2.500000e+00 : f3'
    '2\n    %f32_neg_half = arith.constant -2.500000e+00 : f32\n    %f32_large = arith.constant 7.000000e+04 : f3'
    '2\n    %f32_i32_overflow = arith.constant 1.000000e+20 : f32\n    %f32_i32_underflow = arith.constant -1.000'
    '000e+20 : f32\n    %tid = pto.get_tid_x : i32\n    %signed_tid = arith.subi %tid, %c16_i32 : i32\n    %tid_f'
    '16 = pto.itof %signed_tid signed round(r) : i32 -> f16\n\n    %abs = pto.absf %f32_neg : f32\n    %abs_f16 ='
    ' pto.absf %tid_f16 : f16\n    %sqrt_f32 = pto.sqrt %f32_four : f32 -> f32\n    %sqrt_f16 = pto.sqrt %f16_fou'
    'r : f16 -> f16\n    %min = pto.minf %f32_one, %f32_two : f32\n    %max = pto.maxf %bf16_one, %bf16_two : bf1'
    '6\n    %exp = pto.exp %f32_one : f32 -> f32\n    %log = pto.log %f32_one : f32 -> f32\n    %pow = pto.pow %f'
    '32_two, %f32_four : f32, f32 -> f32\n    %pow16 = pto.pow %f16_two, %f16_four : f16, f16 -> f16\n    %ceil ='
    ' pto.ceil %bf16_one : bf16 -> bf16\n    %floor = pto.floor %f32_two : f32 -> f32\n    %rint = pto.rint %f16_'
    'two : f16 -> f16\n    %round = pto.round %bf16_two : bf16 -> bf16\n    %fma = pto.fma %f32_one, %f32_two, %f'
    '32_four : f32, f32, f32 -> f32\n    %fma16 = pto.fma %f16_one, %f16_two, %f16_four : f16, f16, f16 -> f16\n '
    '   %fmabf16 = pto.fma %bf16_one, %bf16_two, %bf16_four : bf16, bf16, bf16 -> bf16\n    %s32_f32 = pto.itof %'
    'i32_neg signed round(r) : i32 -> f32\n    %u32_f16 = pto.itof %u32_four unsigned round(r) : i32 -> f16\n\n  '
    '  %abs_i = pto.ftoi %abs signed round(z) sat : f32 -> i32\n    %sqrt_f32_i = pto.ftoi %sqrt_f32 signed round'
    '(z) sat : f32 -> i32\n    %sqrt_f16_i = pto.ftoi %sqrt_f16 signed round(z) sat : f16 -> i32\n    %min_i = pt'
    'o.ftoi %min signed round(z) sat : f32 -> i32\n    %max_i = pto.ftoi %max signed round(z) sat : bf16 -> i32\n'
    '    %exp_i = pto.ftoi %exp signed round(z) sat : f32 -> i32\n    %log_i = pto.ftoi %log signed round(z) sat '
    ': f32 -> i32\n    %pow_i = pto.ftoi %pow signed round(z) sat : f32 -> i32\n    %pow16_i = pto.ftoi %pow16 si'
    'gned round(z) sat : f16 -> i32\n    %ceil_i = pto.ftoi %ceil signed round(z) sat : bf16 -> i32\n    %floor_i'
    ' = pto.ftoi %floor signed round(z) sat : f32 -> i32\n    %rint_i = pto.ftoi %rint signed round(z) sat : f16 '
    '-> i32\n    %round_i = pto.ftoi %round signed round(z) sat : bf16 -> i32\n    %fma_i = pto.ftoi %fma signed '
    'round(z) sat : f32 -> i32\n    %fma16_i = pto.ftoi %fma16 signed round(z) sat : f16 -> i32\n    %fmabf16_i ='
    ' pto.ftoi %fmabf16 signed round(z) sat : bf16 -> i32\n    %s32_f32_i = pto.ftoi %s32_f32 signed round(z) sat'
    ' : f32 -> i32\n    %u32_f16_i = pto.ftoi %u32_f16 unsigned round(z) sat : f16 -> i32\n    %generic_itof = pt'
    'o.itof %signed_tid signed : i32 -> f32\n    %generic_ftoi = pto.ftoi %generic_itof signed : f32 -> i32\n    '
    '%generic_ftof = pto.ftof %generic_itof : f32 -> f16\n    %generic_itof_bits = llvm.bitcast %generic_itof : f'
    '32 to i32\n    %generic_ftof_bits = llvm.bitcast %generic_ftof : f16 to i16\n    %generic_ftof_out = arith.e'
    'xtui %generic_ftof_bits : i16 to i32\n    %round_pos_r = pto.ftoi %f32_pos_half signed round(r) sat : f32 ->'
    ' i32\n    %round_neg_r = pto.ftoi %f32_neg_half signed round(r) sat : f32 -> i32\n    %round_pos_a = pto.fto'
    'i %f32_pos_half signed round(a) sat : f32 -> i32\n    %round_neg_a = pto.ftoi %f32_neg_half signed round(a) '
    'sat : f32 -> i32\n    %round_pos_f = pto.ftoi %f32_pos_half signed round(f) sat : f32 -> i32\n    %round_neg'
    '_f = pto.ftoi %f32_neg_half signed round(f) sat : f32 -> i32\n    %round_pos_c = pto.ftoi %f32_pos_half sign'
    'ed round(c) sat : f32 -> i32\n    %round_neg_c = pto.ftoi %f32_neg_half signed round(c) sat : f32 -> i32\n  '
    '  %round_pos_z = pto.ftoi %f32_pos_half signed round(z) sat : f32 -> i32\n    %round_neg_z = pto.ftoi %f32_n'
    'eg_half signed round(z) sat : f32 -> i32\n    %ftoi_sat_hi = pto.ftoi %f32_i32_overflow signed round(z) sat '
    ': f32 -> i32\n    %ftoi_sat_lo = pto.ftoi %f32_i32_underflow signed round(z) sat : f32 -> i32\n    %ftof_sat'
    ' = pto.ftof %f32_large sat : f32 -> f16\n    %ftof_nosat = pto.ftof %f32_large : f32 -> f16\n    %ftof_sat_b'
    'its = llvm.bitcast %ftof_sat : f16 to i16\n    %ftof_nosat_bits = llvm.bitcast %ftof_nosat : f16 to i16\n   '
    ' %ftof_sat_out = arith.extui %ftof_sat_bits : i16 to i32\n    %ftof_nosat_out = arith.extui %ftof_nosat_bits'
    ' : i16 to i32\n    %fmin16 = pto.minf %tid_f16, %f16_three : f16\n    %fmax16 = pto.maxf %tid_f16, %f16_thre'
    'e : f16\n    %fmin16_bits = llvm.bitcast %fmin16 : f16 to i16\n    %fmax16_bits = llvm.bitcast %fmax16 : f16'
    ' to i16\n    %abs_f16_bits = llvm.bitcast %abs_f16 : f16 to i16\n    %fmin16_out = arith.extui %fmin16_bits '
    ': i16 to i32\n    %fmax16_out = arith.extui %fmax16_bits : i16 to i32\n    %abs_f16_out = arith.extui %abs_f'
    '16_bits : i16 to i32\n\n    pto.store %abs_i, %dst[%c0] : !pto.ptr<i32, ub>, index, i32\n    pto.store %sqrt_f32_i,'
    ' %dst[%c1] : !pto.ptr<i32, ub>, index, i32\n    pto.store %sqrt_f16_i, %dst[%c2] : !pto.ptr<i32, ub>, index, i32\n    pto.'
    'store %min_i, %dst[%c3] : !pto.ptr<i32, ub>, index, i32\n    pto.store %max_i, %dst[%c4] : !pto.ptr<i32, ub>, index, i32\n'
    '    pto.store %exp_i, %dst[%c5] : !pto.ptr<i32, ub>, index, i32\n    pto.store %log_i, %dst[%c6] : !pto.ptr<i32, ub'
    '>, index, i32\n    pto.store %pow_i, %dst[%c7] : !pto.ptr<i32, ub>, index, i32\n    pto.store %pow16_i, %dst[%c8] : !pto.p'
    'tr<i32, ub>, index, i32\n    pto.store %ceil_i, %dst[%c9] : !pto.ptr<i32, ub>, index, i32\n    pto.store %floor_i, %dst[%c'
    '10] : !pto.ptr<i32, ub>, index, i32\n    pto.store %rint_i, %dst[%c11] : !pto.ptr<i32, ub>, index, i32\n    pto.store %rou'
    'nd_i, %dst[%c12] : !pto.ptr<i32, ub>, index, i32\n    pto.store %fma_i, %dst[%c13] : !pto.ptr<i32, ub>, index, i32\n    pt'
    'o.store %fma16_i, %dst[%c14] : !pto.ptr<i32, ub>, index, i32\n    pto.store %fmabf16_i, %dst[%c15] : !pto.ptr<i32, '
    'ub>, index, i32\n    %c16 = arith.constant 16 : index\n    %c17 = arith.constant 17 : index\n    pto.store %s32_f32'
    '_i, %dst[%c16] : !pto.ptr<i32, ub>, index, i32\n    pto.store %u32_f16_i, %dst[%c17] : !pto.ptr<i32, ub>, index, i32\n    '
    '%fmin16_offset = arith.addi %tid, %c32_i32 : i32\n    %fmax16_offset = arith.addi %tid, %c64_i32 : i32\n    '
    '%fmin16_idx = arith.index_castui %fmin16_offset : i32 to index\n    %fmax16_idx = arith.index_castui %fmax16'
    '_offset : i32 to index\n    pto.store %fmin16_out, %dst[%fmin16_idx] : !pto.ptr<i32, ub>, index, i32\n    pto.store'
    ' %fmax16_out, %dst[%fmax16_idx] : !pto.ptr<i32, ub>, index, i32\n    %abs_f16_offset = arith.addi %tid, %c96_i32 : '
    'i32\n    %abs_f16_idx = arith.index_castui %abs_f16_offset : i32 to index\n    pto.store %abs_f16_out, %dst['
    '%abs_f16_idx] : !pto.ptr<i32, ub>, index, i32\n    %generic_itof_offset = arith.addi %tid, %c128_i32 : i32\n    %ge'
    'neric_itof_idx = arith.index_castui %generic_itof_offset : i32 to index\n    pto.store %generic_itof_bits, %'
    'dst[%generic_itof_idx] : !pto.ptr<i32, ub>, index, i32\n    %generic_ftoi_offset = arith.addi %tid, %c128_i32 : i32'
    '\n    %generic_ftoi_base = arith.addi %generic_ftoi_offset, %c32_i32 : i32\n    %generic_ftoi_idx = arith.in'
    'dex_castui %generic_ftoi_base : i32 to index\n    pto.store %generic_ftoi, %dst[%generic_ftoi_idx] : !pto.pt'
    'r<i32, ub>, index, i32\n    %generic_ftof_offset = arith.addi %generic_ftoi_base, %c32_i32 : i32\n    %generic_ftof'
    '_idx = arith.index_castui %generic_ftof_offset : i32 to index\n    pto.store %generic_ftof_out, %dst[%generi'
    'c_ftof_idx] : !pto.ptr<i32, ub>, index, i32\n    %round_base = arith.addi %tid, %c224_i32 : i32\n    %round_idx = a'
    'rith.index_castui %round_base : i32 to index\n    pto.store %round_pos_r, %dst[%round_idx] : !pto.ptr<i32, u'
    'b>, index, i32\n    %round_base_1 = arith.addi %round_base, %c32_i32 : i32\n    %round_idx_1 = arith.index_castui %'
    'round_base_1 : i32 to index\n    pto.store %round_neg_r, %dst[%round_idx_1] : !pto.ptr<i32, ub>, index, i32\n    %r'
    'ound_base_2 = arith.addi %round_base_1, %c32_i32 : i32\n    %round_idx_2 = arith.index_castui %round_base_2 '
    ': i32 to index\n    pto.store %round_pos_a, %dst[%round_idx_2] : !pto.ptr<i32, ub>, index, i32\n    %round_base_3 ='
    ' arith.addi %round_base_2, %c32_i32 : i32\n    %round_idx_3 = arith.index_castui %round_base_3 : i32 to inde'
    'x\n    pto.store %round_neg_a, %dst[%round_idx_3] : !pto.ptr<i32, ub>, index, i32\n    %round_base_4 = arith.addi %'
    'round_base_3, %c32_i32 : i32\n    %round_idx_4 = arith.index_castui %round_base_4 : i32 to index\n    pto.st'
    'ore %round_pos_f, %dst[%round_idx_4] : !pto.ptr<i32, ub>, index, i32\n    %round_base_5 = arith.addi %round_base_4,'
    ' %c32_i32 : i32\n    %round_idx_5 = arith.index_castui %round_base_5 : i32 to index\n    pto.store %round_ne'
    'g_f, %dst[%round_idx_5] : !pto.ptr<i32, ub>, index, i32\n    %round_base_6 = arith.addi %round_base_5, %c32_i32 : i'
    '32\n    %round_idx_6 = arith.index_castui %round_base_6 : i32 to index\n    pto.store %round_pos_c, %dst[%ro'
    'und_idx_6] : !pto.ptr<i32, ub>, index, i32\n    %round_base_7 = arith.addi %round_base_6, %c32_i32 : i32\n    %roun'
    'd_idx_7 = arith.index_castui %round_base_7 : i32 to index\n    pto.store %round_neg_c, %dst[%round_idx_7] : '
    '!pto.ptr<i32, ub>, index, i32\n    %round_base_8 = arith.addi %round_base_7, %c32_i32 : i32\n    %round_idx_8 = ari'
    'th.index_castui %round_base_8 : i32 to index\n    pto.store %round_pos_z, %dst[%round_idx_8] : !pto.ptr<i32,'
    ' ub>, index, i32\n    %round_base_9 = arith.addi %round_base_8, %c32_i32 : i32\n    %round_idx_9 = arith.index_cast'
    'ui %round_base_9 : i32 to index\n    pto.store %round_neg_z, %dst[%round_idx_9] : !pto.ptr<i32, ub>, index, i32\n  '
    '  %ftoi_sat_hi_offset = arith.addi %tid, %c544_i32 : i32\n    %ftoi_sat_hi_idx = arith.index_castui %ftoi_sa'
    't_hi_offset : i32 to index\n    pto.store %ftoi_sat_hi, %dst[%ftoi_sat_hi_idx] : !pto.ptr<i32, ub>, index, i32\n   '
    ' %ftoi_sat_lo_offset = arith.addi %ftoi_sat_hi_offset, %c32_i32 : i32\n    %ftoi_sat_lo_idx = arith.index_ca'
    'stui %ftoi_sat_lo_offset : i32 to index\n    pto.store %ftoi_sat_lo, %dst[%ftoi_sat_lo_idx] : !pto.ptr<i32, '
    'ub>, index, i32\n    %ftof_sat_base = arith.addi %ftoi_sat_lo_offset, %c32_i32 : i32\n    %ftof_sat_idx = arith.ind'
    'ex_castui %ftof_sat_base : i32 to index\n    pto.store %ftof_sat_out, %dst[%ftof_sat_idx] : !pto.ptr<i32, ub'
    '>, index, i32\n    %ftof_nosat_offset = arith.addi %ftof_sat_base, %c32_i32 : i32\n    %ftof_nosat_idx = arith.inde'
    'x_castui %ftof_nosat_offset : i32 to index\n    pto.store %ftof_nosat_out, %dst[%ftof_nosat_idx] : !pto.ptr<'
    'i32, ub>, index, i32\n    return\n  }\n}\n'
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

    ELEMS = 1024


    def _generate(seed: int) -> dict:
        v1 = np.full(ELEMS, -1, dtype=np.int32)
        golden_v1 = np.zeros(ELEMS, dtype=np.int32)
        golden_v1[:18] = np.array(
            [3, 2, 2, 1, 2, 2, 0, 16, 16, 1, 2, 2, 2, 6, 6, 6, -4, 4],
            dtype=np.int32,
        )
        values = np.arange(-16, 16, dtype=np.float16)
        three = np.float16(3)
        golden_v1[32:64] = np.minimum(values, three).view(np.uint16).astype(np.int32)
        golden_v1[64:96] = np.maximum(values, three).view(np.uint16).astype(np.int32)
        golden_v1[96:128] = np.abs(np.arange(-16, 16, dtype=np.float16)).view(np.uint16)
        signed_tid = np.arange(-16, 16, dtype=np.int32)
        as_f32 = signed_tid.astype(np.float32)
        golden_v1[128:160] = as_f32.view(np.int32)
        golden_v1[160:192] = as_f32.astype(np.int32)
        golden_v1[192:224] = as_f32.astype(np.float16).view(np.uint16).astype(np.int32)
        rounding_results = [2, -2, 3, -3, 2, -3, 3, -2, 2, -2]
        for index, value in enumerate(rounding_results):
            start = 224 + index * 32
            golden_v1[start : start + 32] = value
        golden_v1[544:576] = np.iinfo(np.int32).max
        golden_v1[576:608] = np.iinfo(np.int32).min
        golden_v1[608:640] = 0x7BFF
        golden_v1[640:672] = 0x7C00
        _files["v1.bin"] = v1
        _files["golden_v1.bin"] = golden_v1


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

    import os
    import sys

    import numpy as np


    def main():
        strict = os.getenv("COMPARE_STRICT", "1") != "0"
        golden = np.fromfile("golden_v1.bin", dtype=np.int32)
        out = np.fromfile("v1.bin", dtype=np.int32)
        written = np.concatenate(
            (
                np.arange(0, 18, dtype=np.intp),
                np.arange(32, 96, dtype=np.intp),
                np.arange(96, 128, dtype=np.intp),
                np.arange(128, 224, dtype=np.intp),
                np.arange(224, 672, dtype=np.intp),
            )
        )
        ok = golden.shape == out.shape and np.array_equal(golden[written], out[written])
        if not ok:
            idxs = written[golden[written] != out[written]]
            idx = int(idxs[0]) if idxs.size else 0
            print(
                f"[ERROR] mismatch at idx={idx}, golden={int(golden[idx])}, out={int(out[idx])}"
            )
            if strict:
                sys.exit(2)
        print("[INFO] compare passed" if ok else "[WARN] compare failed (non-gating)")


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

BUFFERS = [
    (
        'v1',
        'i32',
        'arg0'
    )
]
SCALARS = []
SCALAR_SOURCES = []
SCALAR_DTYPES = []


@pto.jit(
    name='simt_float_convert_core_kernel',
    target="a5",
    backend="vpto",
    mode="explicit",
    source=KERNEL_SOURCE,
)
def kernel(
    arg0: pto.ptr(pto.i32, "gm"),
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
    name='legacy_micro_op_simt_simt_float_convert_core',
    grid=1,
    kernel=kernel,
    make_case=make_case,
    check=check_case,
)]

auto_main(globals())
