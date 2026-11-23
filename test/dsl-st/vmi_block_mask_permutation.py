#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Apply a nonuniform contiguous predicate to strided 32B block group loads."""

import numpy as np
from common import auto_main
from ptodsl import pto


PREFIX = """
module attributes {pto.target_arch = "a5", pto.kernel_kind = #pto.kernel_kind<vector>} {
  func.func @ENTRY(%src: !pto.ptr<f32, gm>, %dst: !pto.ptr<f32, gm>) attributes {pto.kernel} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %stride = arith.constant STRIDE : index
    %active = arith.constant ACTIVE : index
    %zero = arith.constant 0.0 : f32
    %i0 = arith.constant 0 : i64
    %i1 = arith.constant 1 : i64
    %bytes = arith.constant BYTES : i64
    %outbytes = arith.constant 1024 : i64
    %addr = arith.constant 4096 : i64
    %ub = pto.castptr %i0 : i64 -> !pto.ptr<f32, ub>
    %out = pto.castptr %addr : i64 -> !pto.ptr<f32, ub>
    pto.mte_gm_ub %src, %ub, %i0, %bytes nburst(%i1, %bytes, %bytes)
      : !pto.ptr<f32, gm>, !pto.ptr<f32, ub>, i64, i64, i64, i64, i64
    pto.mte_gm_ub %dst, %out, %i0, %outbytes nburst(%i1, %outbytes, %outbytes)
      : !pto.ptr<f32, gm>, !pto.ptr<f32, ub>, i64, i64, i64, i64, i64
    pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]
    pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]
    pto.vecscope {
      BODY
    }
    pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]
    pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]
    pto.mte_ub_gm %out, %dst, %outbytes nburst(%i1, %outbytes, %outbytes)
      : !pto.ptr<f32, ub>, !pto.ptr<f32, gm>, i64, i64, i64, i64
    pto.barrier #pto.pipe<PIPE_ALL>
    return
  }
}
"""


def build_kernel(factor):
    lanes, stride = 64 * factor, 8 * factor + 8
    mask_type = f"!pto.vmi.mask<{lanes}xpred>"
    vector_type = f"!pto.vmi.vreg<{lanes}xf32>"
    body = f"""
      %all = pto.vmi.create_mask %active : index -> {mask_type}
      %gates = pto.vmi.vload %ub[%c0] : !pto.ptr<f32, ub>, index -> {vector_type}
      %mask = pto.vmi.vcmps %gates, %zero, %all {{cmp = "gt"}} : {vector_type}, f32, {mask_type} -> {mask_type}
    """
    result = "!pto.vmi.vreg<8xf32>"
    body += f"""
      %data = pto.vmi.group_load %ub[%c0], %stride {{num_groups = 8}} : !pto.ptr<f32, ub>, index -> {vector_type}
      %sum = pto.vmi.group_reduce_addf %data, %mask {{num_groups = 8, reassoc}}
        : {vector_type}, {mask_type} -> {result}
      pto.vmi.group_store %sum, %out[%c0], %c1 {{num_groups = 8}} : {result}, !pto.ptr<f32, ub>, index
    """
    name = f"block_mask_f{factor}_from_c"
    source = PREFIX.replace("ENTRY", name).replace("STRIDE", str(stride)).replace("ACTIVE", str(lanes - 13))
    source = source.replace("BYTES", str(8 * stride * 4)).replace("BODY", body)

    @pto.jit(name=name, backend="vpto", mode="explicit", source=source)
    def kernel(src: pto.ptr(pto.f32, "gm"), dst: pto.ptr(pto.f32, "gm")):
        pass

    data = ((np.arange(8 * stride) * 17 % 31) - 15).astype(np.float32)
    output = np.full(256, -777, dtype=np.float32)
    rows = data.reshape(8, stride)[:, :8 * factor]
    active = np.arange(lanes) < lanes - 13
    expected = output.copy()
    selected = ((data[:lanes] > 0) & active).reshape(8, 8 * factor)
    expected[:8] = np.where(selected, rows, 0).sum(axis=1, dtype=np.float32)

    def make_case():
        return [data.copy(), output.copy()], expected

    def check_case(inputs, golden):
        np.testing.assert_array_equal(inputs[1].cpu().numpy(), golden)

    return dict(name=name, kernel=kernel, make_case=make_case, check=check_case)


CASES = [build_kernel(factor) for factor in (2, 4)]
auto_main(globals())
