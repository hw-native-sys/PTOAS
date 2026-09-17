#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""block 形态（块步长）`pto.vmi.vload`：payload ≤ 一个物理 carrier。

覆盖 pto-as 单提交 `c26431493` 的 §1.2（surface 阶段 mask 粒度改用 `pred`）：
该形态在改动前报
`VMI-PASS-INVARIANT: VMI producer boundary requires surface !pto.vmi.vreg or !pto.vmi.mask<Nxpred>`。
形态取自 pto_test `vload_i8_vl64_g1_a64_n_nomask_tS8o0n`（i8 / vl=64 / block_stride=8）。
"""

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
    raise RuntimeError("Unable to locate test/dsl-st/common.py from vmi-block-stride-load-basic.py")


_bootstrap_dsl_st_common()

from common import auto_main, golden_output_case
from ptodsl import pto


VL = 64
BLOCK_STRIDE = 8
ELEM_BYTES = 1
EV = 32 // ELEM_BYTES  # 一个 32B block 的元素数（按声明 dtype 算）
NBLK = max(1, (VL + EV - 1) // EV)
SRC_ELEMS = (NBLK - 1) * BLOCK_STRIDE * EV + EV  # 288

SRC = """module attributes {pto.target_arch = "a5", pto.kernel_kind = #pto.kernel_kind<vector>} {
  func.func @vmi_block_stride_load_basic_kernel(%src_gm: !pto.ptr<i8, gm>,
                                                %dst_gm: !pto.ptr<i8, gm>) attributes {pto.kernel} {
    %c0 = arith.constant 0 : index
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c288_i64 = arith.constant 288 : i64
    %c64_i64 = arith.constant 64 : i64
    %c512_i64 = arith.constant 512 : i64

    %ub_src = pto.castptr %c0_i64 : i64 -> !pto.ptr<i8, ub>
    %ub_dst = pto.castptr %c512_i64 : i64 -> !pto.ptr<i8, ub>

    pto.mte_gm_ub %src_gm, %ub_src, %c0_i64, %c288_i64
      nburst(%c1_i64, %c288_i64, %c288_i64)
      : !pto.ptr<i8, gm>, !pto.ptr<i8, ub>, i64, i64, i64, i64, i64

    pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]
    pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]

    pto.vecscope {
      %bs = arith.constant 8 : i16
      %v = pto.vmi.vload %ub_src[%c0], %bs
          : !pto.ptr<i8, ub> -> !pto.vmi.vreg<64xi8>
      pto.vmi.vstore %v, %ub_dst[%c0]
          : !pto.vmi.vreg<64xi8>, !pto.ptr<i8, ub>
    }

    pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]
    pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]
    pto.mte_ub_gm %ub_dst, %dst_gm, %c64_i64
      nburst(%c1_i64, %c64_i64, %c64_i64)
      : !pto.ptr<i8, ub>, !pto.ptr<i8, gm>, i64, i64, i64, i64
    pto.barrier #pto.pipe<PIPE_ALL>
    return
  }
}
"""


@pto.jit(
    name="vmi_block_stride_load_basic_kernel",
    target="a5",
    backend="vpto",
    mode="explicit",
    source=SRC,
)
def vmi_block_stride_load_basic_kernel(
    src_gm: pto.ptr(pto.i8, "gm"),
    dst_gm: pto.ptr(pto.i8, "gm"),
):
    pass


def make_inputs():
    return [(np.arange(SRC_ELEMS) % 100).astype(np.int8)]


def make_expected(src):
    out = np.empty(VL, dtype=np.int8)
    for lane in range(VL):
        blk = lane // EV
        out[lane] = src[blk * BLOCK_STRIDE * EV + (lane % EV)]
    return out


CASES = [
    golden_output_case(
        "vmi_block_stride_load_basic",
        vmi_block_stride_load_basic_kernel,
        inputs=make_inputs,
        expected=make_expected,
        rtol=0.0,
        atol=0.0,
    ),
]


auto_main(globals())
