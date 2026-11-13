#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See the License in the root of the software repository for the full text of the License.

"""Issue #26: bounded runtime VMI copy without unnecessary latch barriers.

Run with: python test/dsl-st/vecscope_membar_dynamic_disjoint.py
Untouched output lanes retain a sentinel, including the zero-trip case.
"""

import numpy as np

from common import auto_main
from ptodsl import pto

SOURCE = r"""
module attributes {pto.target_arch = "a5", pto.kernel_kind = #pto.kernel_kind<vector>} {
  func.func @bounded_copy(%input: !pto.ptr<f32, gm>, %output: !pto.ptr<f32, gm>, %n32: i32) attributes {pto.aicore} {
    %zero = arith.constant 0 : i64
    %ub = pto.castptr %zero : i64 -> !pto.ptr<f32, ub>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c4096 = arith.constant 4096 : index
    %src = pto.addptr %ub, %c0 : !pto.ptr<f32, ub>, index -> !pto.ptr<f32, ub>
    %dst = pto.addptr %ub, %c4096 : !pto.ptr<f32, ub>, index -> !pto.ptr<f32, ub>
    %n = arith.index_castui %n32 : i32 to index
    %bound = arith.minui %n, %c64 : index
    %bytes = arith.constant 16384 : i64
    %one = arith.constant 1 : i64
    %zero64 = arith.constant 0 : i64
    pto.mte_gm_ub %input, %src, %zero64, %bytes nburst(%one, %zero64, %zero64)
      : !pto.ptr<f32, gm>, !pto.ptr<f32, ub>, i64, i64, i64, i64, i64
    pto.mte_gm_ub %output, %dst, %zero64, %bytes nburst(%one, %zero64, %zero64)
      : !pto.ptr<f32, gm>, !pto.ptr<f32, ub>, i64, i64, i64, i64, i64
    pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]
    pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]
    pto.vecscope {
      %mask = pto.vmi.create_mask %c64 : index -> !pto.vmi.mask<64xpred>
      scf.for %i = %c0 to %bound step %c1 {
        %offset = arith.muli %i, %c64 : index
        %v = pto.vmi.vload %src[%offset] : !pto.ptr<f32, ub> -> !pto.vmi.vreg<64xf32>
        pto.vmi.vstore %v, %dst[%offset], %mask : !pto.vmi.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.vmi.mask<64xpred>
      }
    }
    pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]
    pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]
    pto.mte_ub_gm %dst, %output, %bytes nburst(%one, %zero64, %zero64)
      : !pto.ptr<f32, ub>, !pto.ptr<f32, gm>, i64, i64, i64, i64
    pto.barrier #pto.pipe<PIPE_ALL>
    return
  }
}
"""


@pto.jit(name="bounded_copy", target="a5", backend="vpto", mode="explicit", source=SOURCE)
def kernel(src: pto.ptr(pto.f32, "gm"), dst: pto.ptr(pto.f32, "gm"), n: pto.i32):
    pass


def make_case(n):
    src = np.random.default_rng(26).uniform(-4, 4, 4096).astype(np.float32)
    dst = np.full(4096, 241.0, dtype=np.float32)
    expected = dst.copy()
    copied = min(n, 64) * 64
    expected[:copied] = src[:copied]
    return [src, dst], expected, [n]


def check(inputs, expected):
    np.testing.assert_array_equal(inputs[1].cpu().numpy(), expected)


CASES = [
    {"name": f"bounded_copy_n{n}", "kernel": kernel, "grid": 1,
     "make_case": lambda n=n: make_case(n), "check": check}
    for n in (0, 1, 17, 63, 64, 65)
]

auto_main(globals())
