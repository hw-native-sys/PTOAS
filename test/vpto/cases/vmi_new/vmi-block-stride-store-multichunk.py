#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""block 形态（块步长）`pto.vmi.vstore`：多 carrier `vsstb` + 绝对落位校验。

覆盖 pto-as 单提交 `c26431493` 的 §1.2（mask 粒度）与 §1.3（多 carrier stride 访存）的写侧：
i32 / vl=256 / block_stride=8 的 256 lane 会按“每 8 个元素一组、组间跨 64 个元素”
落到 dst 的 0..1991 span 上。

校验方式（**MTE 读回**，不走向量读端口）：
- 整个 dst span（1992×i32 = 7968B）用 `mte_ub_gm` 原样搬回 GM 比对；
- 落位处（`pos = (lane // 8) * 8 * 8 + lane % 8`）必须等于对应输入值；
- 其余位置必须保持预置 sentinel（证明没有错位/越界写）。

这里刻意不用向量 `vload` 读回 dst：pto-as 的“向量 store → 向量 load”可见性在
“直连 runtime_camodel”这条验证路径上不成立（同一 launch 内 MTE 读得到刚写的字节、
向量读端口读不到，见 `vload-vstore-c3c4-failure-analysis.md`）；MTE 字节搬既绕开该路径
问题，又比 round-trip 更强（绝对落位 + 越界检查）。

形态取自 pto_test `vstore_i32_vl256_g1_a256_n_nomask_tS8o0n`。
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
    raise RuntimeError("Unable to locate test/dsl-st/common.py from vmi-block-stride-store-multichunk.py")


_bootstrap_dsl_st_common()

from common import auto_main, golden_output_case
from ptodsl import pto


VL = 256
BLOCK_STRIDE = 8
ELEM_BYTES = 4
EV = 32 // ELEM_BYTES  # i32 -> 一个 32B block 装 8 个元素
DST_ELEMS = (VL // EV - 1) * BLOCK_STRIDE * EV + EV  # 1992
IN_BYTES = VL * ELEM_BYTES  # 1024
DST_BYTES = DST_ELEMS * ELEM_BYTES  # 7968
SENTINEL = -777

SRC = """module attributes {pto.target_arch = "a5", pto.kernel_kind = #pto.kernel_kind<vector>} {
  func.func @vmi_block_stride_store_multichunk_kernel(%in_gm: !pto.ptr<i32, gm>,
                                                      %sentinel_gm: !pto.ptr<i32, gm>,
                                                      %out_gm: !pto.ptr<i32, gm>) attributes {pto.kernel} {
    %c0 = arith.constant 0 : index
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c1024_i64 = arith.constant 1024 : i64
    %c7968_i64 = arith.constant 7968 : i64

    %ub_in = pto.castptr %c0_i64 : i64 -> !pto.ptr<i32, ub>
    %ub_dst = pto.castptr %c1024_i64 : i64 -> !pto.ptr<i32, ub>

    pto.mte_gm_ub %in_gm, %ub_in, %c0_i64, %c1024_i64
      nburst(%c1_i64, %c1024_i64, %c1024_i64)
      : !pto.ptr<i32, gm>, !pto.ptr<i32, ub>, i64, i64, i64, i64, i64
    pto.mte_gm_ub %sentinel_gm, %ub_dst, %c0_i64, %c7968_i64
      nburst(%c1_i64, %c7968_i64, %c7968_i64)
      : !pto.ptr<i32, gm>, !pto.ptr<i32, ub>, i64, i64, i64, i64, i64

    pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]
    pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]

    pto.vecscope {
      %v = pto.vmi.vload %ub_in[%c0]
          : !pto.ptr<i32, ub> -> !pto.vmi.vreg<256xi32>
      %bs = arith.constant 8 : i16
      pto.vmi.vstore %v, %ub_dst[%c0], %bs
          : !pto.vmi.vreg<256xi32>, !pto.ptr<i32, ub>
    }

    pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]
    pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]
    pto.mte_ub_gm %ub_dst, %out_gm, %c7968_i64
      nburst(%c1_i64, %c7968_i64, %c7968_i64)
      : !pto.ptr<i32, ub>, !pto.ptr<i32, gm>, i64, i64, i64, i64
    pto.barrier #pto.pipe<PIPE_ALL>
    return
  }
}
"""


@pto.jit(
    name="vmi_block_stride_store_multichunk_kernel",
    target="a5",
    backend="vpto",
    mode="explicit",
    source=SRC,
)
def vmi_block_stride_store_multichunk_kernel(
    in_gm: pto.ptr(pto.i32, "gm"),
    sentinel_gm: pto.ptr(pto.i32, "gm"),
    out_gm: pto.ptr(pto.i32, "gm"),
):
    pass


def make_inputs():
    values = (np.arange(VL) % 100).astype(np.int32)
    sentinel = np.full(DST_ELEMS, SENTINEL, dtype=np.int32)
    return [values, sentinel]


def make_expected(values, sentinel):
    out = np.array(sentinel, dtype=np.int32).copy()
    for lane in range(VL):
        pos = (lane // EV) * BLOCK_STRIDE * EV + (lane % EV)
        out[pos] = values[lane]
    return out


CASES = [
    golden_output_case(
        "vmi_block_stride_store_multichunk",
        vmi_block_stride_store_multichunk_kernel,
        inputs=make_inputs,
        expected=make_expected,
        rtol=0.0,
        atol=0.0,
    ),
]


auto_main(globals())
