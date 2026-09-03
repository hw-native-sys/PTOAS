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
    raise RuntimeError("Unable to locate test/dsl-st/common.py from vcvt_s8_to_f16.py")


_bootstrap_dsl_st_common()

from common import auto_main, golden_output_case
from ptodsl import pto


ELEMS = 256

VCVT_SOURCE = """module attributes {pto.target_arch = "a5", pto.kernel_kind = #pto.kernel_kind<vector>} {
  func.func @vcvt_s8_to_f16_kernel(
      %src_gm: !pto.ptr<si8, gm>,
      %dst_gm: !pto.ptr<f16, gm>)
      attributes {pto.kernel} {
    %c0 = arith.constant 0 : index
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c256_i64 = arith.constant 256 : i64
    %c512_i64 = arith.constant 512 : i64
    %c4096_i64 = arith.constant 4096 : i64

    %ub_src = pto.castptr %c0_i64 : i64 -> !pto.ptr<si8, ub>
    %ub_dst = pto.castptr %c4096_i64 : i64 -> !pto.ptr<f16, ub>

    pto.mte_gm_ub %src_gm, %ub_src, %c0_i64, %c256_i64
      nburst(%c1_i64, %c256_i64, %c256_i64)
      : !pto.ptr<si8, gm>, !pto.ptr<si8, ub>, i64, i64, i64, i64, i64

    pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]
    pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]

    pto.vecscope {
      %wide = pto.vmi.vload %ub_src[%c0] : !pto.ptr<si8, ub> -> !pto.vmi.vreg<256xsi8>
      %cvt = pto.vmi.vcvt %wide
          : !pto.vmi.vreg<256xsi8> -> !pto.vmi.vreg<256xf16>
      pto.vmi.vstore %cvt, %ub_dst[%c0]
          : !pto.vmi.vreg<256xf16>, !pto.ptr<f16, ub>
    }

    pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]
    pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]
    pto.mte_ub_gm %ub_dst, %dst_gm, %c512_i64
      nburst(%c1_i64, %c512_i64, %c512_i64)
      : !pto.ptr<f16, ub>, !pto.ptr<f16, gm>, i64, i64, i64, i64
    pto.barrier #pto.pipe<PIPE_ALL>
    return
  }
}
"""


@pto.jit(
    name="vcvt_s8_to_f16_kernel",
    target="a5",
    backend="vpto",
    mode="explicit",
    source=VCVT_SOURCE,
)
def vcvt_s8_to_f16_kernel(
    src_gm: pto.ptr(pto.si8, "gm"),
    dst_gm: pto.ptr(pto.f16, "gm"),
):
    pass


def make_inputs():
    src = np.arange(-128, 128, dtype=np.int8)
    if src.size < ELEMS:
        src = np.tile(src, ELEMS // src.size + 1)[:ELEMS]
    src = src.astype(np.int8)
    return [src]


def make_expected(src):
    golden_f16 = src.astype(np.float16)
    return golden_f16.view(np.uint16).astype(np.uint16)


CASES = [
    golden_output_case(
        "vcvt_s8_to_f16",
        vcvt_s8_to_f16_kernel,
        inputs=make_inputs,
        expected=make_expected,
        rtol=0.0,
        atol=0.0,
    ),
]


auto_main(globals())
