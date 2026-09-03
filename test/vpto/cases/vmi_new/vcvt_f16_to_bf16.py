#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from pathlib import Path
import sys

import numpy as np


def _bootstrap_dsl_st_common() -> None:
    here = Path(__file__).resolve()
    for candidate in here.parents:
        common_dir = candidate / "test" / "dsl-st"
        if (common_dir / "common.py").exists():
            sys.path.insert(0, str(common_dir))
            return
    raise RuntimeError("Unable to locate test/dsl-st/common.py from vcvt_f16_to_bf16.py")


_bootstrap_dsl_st_common()

from common import auto_main, golden_output_case
from ptodsl import pto


ELEMS = 256

VCVT_SOURCE = """module attributes {pto.target_arch = "a5", pto.kernel_kind = #pto.kernel_kind<vector>} {
  func.func @vcvt_f16_to_bf16_kernel(
      %src_gm: !pto.ptr<f16, gm>,
      %dst_gm: !pto.ptr<bf16, gm>)
      attributes {pto.kernel} {
    %c0 = arith.constant 0 : index
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c512_i64 = arith.constant 512 : i64
    %c4096_i64 = arith.constant 4096 : i64

    %ub_src = pto.castptr %c0_i64 : i64 -> !pto.ptr<f16, ub>
    %ub_dst = pto.castptr %c4096_i64 : i64 -> !pto.ptr<bf16, ub>

    pto.mte_gm_ub %src_gm, %ub_src, %c0_i64, %c512_i64
      nburst(%c1_i64, %c512_i64, %c512_i64)
      : !pto.ptr<f16, gm>, !pto.ptr<f16, ub>, i64, i64, i64, i64, i64

    pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]
    pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]

    pto.vecscope {
      %wide = pto.vmi.vload %ub_src[%c0] : !pto.ptr<f16, ub> -> !pto.vmi.vreg<256xf16>
      %cvt = pto.vmi.vcvt %wide {rounding = "R"}
          : !pto.vmi.vreg<256xf16> -> !pto.vmi.vreg<256xbf16>
      pto.vmi.vstore %cvt, %ub_dst[%c0]
          : !pto.vmi.vreg<256xbf16>, !pto.ptr<bf16, ub>
    }

    pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]
    pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]
    pto.mte_ub_gm %ub_dst, %dst_gm, %c512_i64
      nburst(%c1_i64, %c512_i64, %c512_i64)
      : !pto.ptr<bf16, ub>, !pto.ptr<bf16, gm>, i64, i64, i64, i64
    pto.barrier #pto.pipe<PIPE_ALL>
    return
  }
}
"""


@pto.jit(
    name="vcvt_f16_to_bf16_kernel",
    target="a5",
    backend="vpto",
    mode="explicit",
    source=VCVT_SOURCE,
)
def vcvt_f16_to_bf16_kernel(
    src_gm: pto.ptr(pto.f16, "gm"),
    dst_gm: pto.ptr(pto.bf16, "gm"),
):
    pass


def to_bf16_bits(x):
    """f32 -> bfloat16 bit pattern (uint16), round-to-nearest-even."""
    x = np.asarray(x, dtype=np.float32)
    u = x.view(np.uint32)
    lsb = (u >> np.uint32(16)) & np.uint32(1)
    rounding_bias = np.uint32(0x7FFF) + lsb
    r = ((u + rounding_bias) >> np.uint32(16)).astype(np.uint16)
    r = np.where(np.isnan(x), np.uint16(0x7FC0), r)
    return r


def make_inputs():
    probe = np.array(
        [
            0.0, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 1.5,
            -1.5, 3.0, -3.0, 127.0, -128.0, 255.0, -255.0,
            1023.0, -1024.0, 0.333251953125, -0.333251953125,
            0.1, -0.1, 100.5, -100.5, 10000.0, -10000.0,
            65504.0, -65504.0, 0.0001, -0.0001,
            2.0 ** -14, -(2.0 ** -14), 5.5,
        ],
        dtype=np.float16,
    )
    src = np.tile(probe, ELEMS // len(probe))[:ELEMS].astype(np.float16)
    return [src]


def make_expected(src):
    return to_bf16_bits(src.astype(np.float32))


CASES = [
    golden_output_case(
        "vcvt_f16_to_bf16",
        vcvt_f16_to_bf16_kernel,
        inputs=make_inputs,
        expected=make_expected,
        rtol=0.0,
        atol=0.0,
    ),
]


auto_main(globals())
