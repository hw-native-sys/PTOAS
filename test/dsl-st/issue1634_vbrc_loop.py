#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Issue 1634: CSE leaves a vector-carrying SCF yield after reduce/broadcast."""

import numpy as np

from common import auto_main, golden_output_case
from ptodsl import pto

LANES = 64
MODULUS = 1 << 32

SOURCE = """module attributes {pto.target_arch = "a5", pto.kernel_kind = #pto.kernel_kind<vector>} {
  func.func @KERNEL(%src_gm: !pto.ptr<DTYPE, gm>,
                    %dst_gm: !pto.ptr<DTYPE, gm>) attributes {pto.kernel} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c64 = arith.constant 64 : index
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c256_i64 = arith.constant 256 : i64
    %c4096_i64 = arith.constant 4096 : i64
    %src = pto.castptr %c0_i64 : i64 -> !pto.ptr<DTYPE, ub>
    %dst = pto.castptr %c4096_i64 : i64 -> !pto.ptr<DTYPE, ub>
    pto.mte_gm_ub %src_gm, %src, %c0_i64, %c256_i64
      nburst(%c1_i64, %c256_i64, %c256_i64)
      : !pto.ptr<DTYPE, gm>, !pto.ptr<DTYPE, ub>, i64, i64, i64, i64, i64
    pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]
    pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]

    %x = pto.vmi.vload %src[%c0] : !pto.ptr<DTYPE, ub>, index -> !pto.vmi.vreg<64xDTYPE>
    %mask = pto.vmi.create_mask %c64 : index -> !pto.vmi.mask<64xpred>
    %reduced = pto.vmi.REDUCE %x, %mask {group = 1 : i64, pmode = "zero"}
      : !pto.vmi.vreg<64xDTYPE>, !pto.vmi.mask<64xpred> -> !pto.vmi.vreg<1xDTYPE>
    %broadcast = pto.vmi.vbrc %reduced : !pto.vmi.vreg<1xDTYPE> -> !pto.vmi.vreg<64xDTYPE>
    %result = scf.for %i = %c0 to %c2 step %c1
        iter_args(%acc = %broadcast) -> (!pto.vmi.vreg<64xDTYPE>) {
      %repeated = pto.vmi.REDUCE %x, %mask {group = 1 : i64, pmode = "zero"}
        : !pto.vmi.vreg<64xDTYPE>, !pto.vmi.mask<64xpred> -> !pto.vmi.vreg<1xDTYPE>
      %again = pto.vmi.vbrc %repeated : !pto.vmi.vreg<1xDTYPE> -> !pto.vmi.vreg<64xDTYPE>
      scf.yield %again : !pto.vmi.vreg<64xDTYPE>
    }
    pto.vmi.vstore %result, %dst[%c0] : !pto.vmi.vreg<64xDTYPE>, !pto.ptr<DTYPE, ub>, index

    pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]
    pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]
    pto.mte_ub_gm %dst, %dst_gm, %c256_i64
      nburst(%c1_i64, %c256_i64, %c256_i64)
      : !pto.ptr<DTYPE, ub>, !pto.ptr<DTYPE, gm>, i64, i64, i64, i64
    pto.barrier #pto.pipe<PIPE_ALL>
    return
  }
}
"""


def _kernel(label, dtype, reduction):
    name = f"issue1634_{reduction}_{label}_kernel"
    source = SOURCE.replace("KERNEL", name).replace("DTYPE", label).replace("REDUCE", reduction)

    @pto.jit(name=name, target="a5", backend="vpto", mode="explicit",
             source=source, insert_sync=False)
    def kernel(src_gm: pto.ptr(dtype, "gm"), dst_gm: pto.ptr(dtype, "gm")):
        pass

    return kernel


def _inputs(np_dtype):
    if np_dtype == np.int32:
        values = [2147483647, 2147483644, -2147483648, -2147483641,
                  -1000000000, -17, 0, 123456789]
    else:
        values = [0xFFFFFFFF, 0x80000000, 0x80000001, 17,
                  0x7FFFFFFF, 3, 0xFFFFFFFE, 0x12345678]
    return [np.array(values * (LANES // len(values)), dtype=np_dtype)]


def _golden(source, reduction):
    # Python integers avoid NumPy's fixed-width accumulator and overflow rules.
    values = [int(value) for value in source]
    if reduction == "vcadd":
        reduced = sum(values)
    elif reduction == "vcmax":
        reduced = max(values)
    else:
        reduced = min(values)
    bits = reduced % MODULUS
    if source.dtype == np.dtype(np.int32) and bits >= MODULUS // 2:
        bits -= MODULUS
    return np.full(LANES, bits, dtype=source.dtype)


def _case(label, dtype, np_dtype, reduction):
    return golden_output_case(
        f"issue1634_{reduction}_{label}",
        _kernel(label, dtype, reduction),
        inputs=lambda: _inputs(np_dtype),
        expected=lambda source: _golden(source, reduction),
        rtol=0.0,
        atol=0.0,
    )


CASES = [
    _case(label, dtype, np_dtype, reduction)
    for label, dtype, np_dtype in (("si32", pto.si32, np.int32),
                                   ("ui32", pto.ui32, np.uint32))
    for reduction in ("vcadd", "vcmax", "vcmin")
]

auto_main(globals())
