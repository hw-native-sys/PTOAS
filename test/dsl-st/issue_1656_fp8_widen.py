#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Issue #1656: fp8 -> f16/f32/bf16 widening must be dense at every lane.

The original failure was bf16-only: the compiler expands fp8 -> bf16 into
extf(fp8 -> f32) + truncf(f32 -> bf16) and hard-codes rounding="R"/saturate="SAT"
on the second step.  That state cannot use the NaN-relaxed (dense, vdintlv)
lowering, so the lowering writes the converted lanes at lane_stride 2; layout
assignment nevertheless picked a dense group-slot result, no vpack was emitted
and only half of the values reached memory.  On the device the eight lanes read
back as [0,0,1,0,2,0,3,0] instead of [0,1,2,3,1.5,-1,256,448].
"""

import numpy as np

from common import auto_main, golden_output_case
from ptodsl import pto

VL = 8

# e4m3 bytes for [0, 1, 2, 3, 1.5, -1, 256, 448].
E4M3_BYTES = np.array([0x00, 0x38, 0x40, 0x44, 0x3C, 0xB8, 0x78, 0x7E], dtype=np.uint8)
FP8_VALUES = np.array([0.0, 1.0, 2.0, 3.0, 1.5, -1.0, 256.0, 448.0], dtype=np.float32)


def _make_kernel(name, dst_dtype, out_bytes_per_elem):
    @pto.jit(name=name, kernel_kind="vector", target="a5", backend="vpto", mode="explicit", entry=True)
    def kernel(src_ptr: pto.ptr(pto.ui8, "gm"), out_ptr: pto.ptr(dst_dtype, "gm")):
        pto.init_core()
        src_ub = pto.castptr(pto.i64(0), pto.ptr(pto.ui8, "ub"))
        out_ub = pto.castptr(pto.i64(VL), pto.ptr(dst_dtype, "ub"))
        fp8_ub = pto.castptr(pto.i64(0), pto.ptr(pto.f8e4m3, "ub"))
        pto.mte_gm_ub(src_ptr, src_ub, 0, VL, nburst=(1, 0, 0))
        pto.set_flag("MTE2", "V", event_id=0)
        pto.wait_flag("MTE2", "V", event_id=0)
        with pto.vecscope():
            source = pto.vmi.vload(fp8_ub, pto.const(0, dtype=pto.index), size=VL)
            result = pto.vmi.vcvt(source, dst_dtype)
            pto.vmi.vstore(result, out_ub, pto.const(0, dtype=pto.index))
        pto.set_flag("V", "MTE3", event_id=0)
        pto.wait_flag("V", "MTE3", event_id=0)
        pto.mte_ub_gm(out_ub, out_ptr, VL * out_bytes_per_elem, nburst=(1, 0, 0))

    return kernel


def _f32_to_bf16_bits(values):
    # All golden values are exactly representable in bf16, so truncating the
    # f32 bit pattern is the same as the round-to-nearest-even conversion.
    return (values.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)


CASES = [
    golden_output_case(
        "issue_1656_fp8_to_f32",
        _make_kernel("issue_1656_fp8_to_f32", pto.f32, 4),
        inputs=[E4M3_BYTES.copy()],
        expected=FP8_VALUES.copy(),
        rtol=0.0,
        atol=0.0,
    ),
    golden_output_case(
        "issue_1656_fp8_to_f16",
        _make_kernel("issue_1656_fp8_to_f16", pto.f16, 2),
        inputs=[E4M3_BYTES.copy()],
        expected=FP8_VALUES.astype(np.float16),
        rtol=0.0,
        atol=0.0,
    ),
    golden_output_case(
        "issue_1656_fp8_to_bf16",
        _make_kernel("issue_1656_fp8_to_bf16", pto.bf16, 2),
        inputs=[E4M3_BYTES.copy()],
        expected=_f32_to_bf16_bits(FP8_VALUES),
        output_dtype=np.uint16,
        rtol=0.0,
        atol=0.0,
    ),
]


auto_main(globals())
