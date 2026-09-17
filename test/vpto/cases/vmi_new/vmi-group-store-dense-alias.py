#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""`group_store` 稠密别名：`stride == groupSize` 的退化解直接走稠密 `vstore`。

覆盖 pto-as 单提交 `c26431493` 的 §1.4（`isDenseAliasedGroupStore`）：
group=8 / stride=8 的 `pto.vmi.vstore` 语义上就是一次稠密连续写，
改动前报 `VMI-UNSUPPORTED: pto.vmi.group_store requires a supported UB destination and a
table-supported value layout ...`。形态取自 pto_test
`vstore_ui8_vl64_g8_a64_n_nomask_tG8o0n`。

校验方式（**MTE 读回**，不走向量读端口）：把 dst 的整个 128B span 用 `mte_ub_gm` 原样搬回 GM，
- 前 64 个位置应被稠密写覆盖（等于输入）；
- 第 64..127 个位置必须保持 sentinel（证明没有越界写）。

这里刻意不用向量 `vload` 读回 dst：pto-as 的“向量 store → 向量 load”可见性在
“直连 runtime_camodel”这条验证路径上不成立（同一 launch 内 MTE 读得到刚写的字节、
向量读端口读不到，见 `vload-vstore-c3c4-failure-analysis.md`）。
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
    raise RuntimeError("Unable to locate test/dsl-st/common.py from vmi-group-store-dense-alias.py")


_bootstrap_dsl_st_common()

from common import auto_main, golden_output_case
from ptodsl import pto


VL = 64
GROUP = 8
STRIDE = GROUP  # 稠密别名：stride == groupSize
IN_BYTES = VL
DST_ELEMS = 128
DST_BYTES = DST_ELEMS
SENTINEL = 0xAB

SRC = """module attributes {pto.target_arch = "a5", pto.kernel_kind = #pto.kernel_kind<vector>} {
  func.func @vmi_group_store_dense_alias_kernel(%in_gm: !pto.ptr<ui8, gm>,
                                                %sentinel_gm: !pto.ptr<ui8, gm>,
                                                %out_gm: !pto.ptr<ui8, gm>) attributes {pto.kernel} {
    %c0 = arith.constant 0 : index
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %c128_i64 = arith.constant 128 : i64
    %c1024_i64 = arith.constant 1024 : i64

    %ub_in = pto.castptr %c0_i64 : i64 -> !pto.ptr<ui8, ub>
    %ub_dst = pto.castptr %c1024_i64 : i64 -> !pto.ptr<ui8, ub>

    pto.mte_gm_ub %in_gm, %ub_in, %c0_i64, %c64_i64
      nburst(%c1_i64, %c64_i64, %c64_i64)
      : !pto.ptr<ui8, gm>, !pto.ptr<ui8, ub>, i64, i64, i64, i64, i64
    pto.mte_gm_ub %sentinel_gm, %ub_dst, %c0_i64, %c128_i64
      nburst(%c1_i64, %c128_i64, %c128_i64)
      : !pto.ptr<ui8, gm>, !pto.ptr<ui8, ub>, i64, i64, i64, i64, i64

    pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]
    pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]

    pto.vecscope {
      %v = pto.vmi.vload %ub_in[%c0]
          : !pto.ptr<ui8, ub> -> !pto.vmi.vreg<64xui8>
      %stride = arith.constant 8 : index
      pto.vmi.vstore %v, %ub_dst[%c0], %stride {group = 8 : i64}
          : !pto.vmi.vreg<64xui8>, !pto.ptr<ui8, ub>
    }

    pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]
    pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]
    pto.mte_ub_gm %ub_dst, %out_gm, %c128_i64
      nburst(%c1_i64, %c128_i64, %c128_i64)
      : !pto.ptr<ui8, ub>, !pto.ptr<ui8, gm>, i64, i64, i64, i64
    pto.barrier #pto.pipe<PIPE_ALL>
    return
  }
}
"""


@pto.jit(
    name="vmi_group_store_dense_alias_kernel",
    target="a5",
    backend="vpto",
    mode="explicit",
    source=SRC,
)
def vmi_group_store_dense_alias_kernel(
    in_gm: pto.ptr(pto.ui8, "gm"),
    sentinel_gm: pto.ptr(pto.ui8, "gm"),
    out_gm: pto.ptr(pto.ui8, "gm"),
):
    pass


def make_inputs():
    values = (np.arange(VL) % 200).astype(np.uint8)
    sentinel = np.full(DST_ELEMS, SENTINEL, dtype=np.uint8)
    return [values, sentinel]


def make_expected(values, sentinel):
    out = np.array(sentinel, dtype=np.uint8).copy()
    out[:VL] = values[:VL]
    return out


CASES = [
    golden_output_case(
        "vmi_group_store_dense_alias",
        vmi_group_store_dense_alias_kernel,
        inputs=make_inputs,
        expected=make_expected,
        rtol=0.0,
        atol=0.0,
    ),
]


auto_main(globals())
