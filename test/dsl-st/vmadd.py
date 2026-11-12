#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import numpy as np

from common import auto_main, golden_output_case
from ptodsl import pto

ELEMS = 1024
SEED = 29

VMADD_SOURCE = """module attributes {pto.target_arch = "a5", pto.kernel_kind = #pto.kernel_kind<vector>} {
  func.func @a5_extra_vmadd_kernel(
      %f_acc: !pto.ptr<f32, gm>, %f_lhs: !pto.ptr<f32, gm>,
      %f_rhs: !pto.ptr<f32, gm>, %out_vmadd: !pto.ptr<f32, gm>)
      attributes {pto.kernel} {
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c1024 = arith.constant 1024 : index
    %c0_i64 = arith.constant 0 : i64
    %c32_i64 = arith.constant 32 : i64
    %c128_i64 = arith.constant 128 : i64
    %c4096_i64 = arith.constant 4096 : i64
    %c8192_i64 = arith.constant 8192 : i64
    %c12288_i64 = arith.constant 12288 : i64

    %ub_f_acc = pto.castptr %c0_i64 : i64 -> !pto.ptr<f32, ub>
    %ub_f_lhs = pto.castptr %c4096_i64 : i64 -> !pto.ptr<f32, ub>
    %ub_f_rhs = pto.castptr %c8192_i64 : i64 -> !pto.ptr<f32, ub>
    %ub_f_vmadd = pto.castptr %c12288_i64 : i64 -> !pto.ptr<f32, ub>

    pto.mte_gm_ub %f_acc, %ub_f_acc, %c0_i64, %c128_i64
      nburst(%c32_i64, %c128_i64, %c128_i64)
      : !pto.ptr<f32, gm>, !pto.ptr<f32, ub>, i64, i64, i64, i64, i64
    pto.mte_gm_ub %f_lhs, %ub_f_lhs, %c0_i64, %c128_i64
      nburst(%c32_i64, %c128_i64, %c128_i64)
      : !pto.ptr<f32, gm>, !pto.ptr<f32, ub>, i64, i64, i64, i64, i64
    pto.mte_gm_ub %f_rhs, %ub_f_rhs, %c0_i64, %c128_i64
      nburst(%c32_i64, %c128_i64, %c128_i64)
      : !pto.ptr<f32, gm>, !pto.ptr<f32, ub>, i64, i64, i64, i64, i64
    pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]
    pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]

    pto.vecscope {
      %mask = pto.pset_b32 "PAT_ALL" : !pto.mask<b32>
      scf.for %offset = %c0 to %c1024 step %c64 {
        %acc = pto.vlds %ub_f_acc[%offset] : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>
        %lhs = pto.vlds %ub_f_lhs[%offset] : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>
        %rhs = pto.vlds %ub_f_rhs[%offset] : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>
        %vmadd = pto.vmadd %acc, %lhs, %rhs, %mask : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
        pto.vsts %vmadd, %ub_f_vmadd[%offset], %mask : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, index, !pto.mask<b32>
      }
    }
    pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]
    pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]
    pto.mte_ub_gm %ub_f_vmadd, %out_vmadd, %c128_i64
      nburst(%c32_i64, %c128_i64, %c128_i64)
      : !pto.ptr<f32, ub>, !pto.ptr<f32, gm>, i64, i64, i64, i64
    pto.barrier #pto.pipe<PIPE_ALL>
    return
  }
}
"""

@pto.jit(
    name="a5_extra_vmadd_kernel",
    target="a5",
    backend="vpto",
    mode="explicit",
    source=VMADD_SOURCE,
)
def a5_extra_vmadd_kernel(
    f_acc: pto.ptr(pto.f32, "gm"),
    f_lhs: pto.ptr(pto.f32, "gm"),
    f_rhs: pto.ptr(pto.f32, "gm"),
    out_vmadd: pto.ptr(pto.f32, "gm"),
):
    pass

def make_inputs():
    rng = np.random.default_rng(SEED)
    f_acc = rng.uniform(-2.0, 2.0, size=ELEMS).astype(np.float32)
    f_lhs = rng.uniform(-3.0, 3.0, size=ELEMS).astype(np.float32)
    f_rhs = rng.uniform(-1.0, 1.0, size=ELEMS).astype(np.float32)
    return [f_acc, f_lhs, f_rhs]

def make_expected(f_acc, f_lhs, f_rhs):
    return (f_lhs * f_acc + f_rhs).astype(np.float32)

CASES = [
    golden_output_case(
        "a5_extra_vmadd",
        a5_extra_vmadd_kernel,
        inputs=make_inputs,
        expected=make_expected,
        rtol=2e-4,
        atol=2e-4,
    ),
]

auto_main(globals())
