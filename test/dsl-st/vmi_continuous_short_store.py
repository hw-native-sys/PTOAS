#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Four adjacent BF16 short writes, one tail flush, and an automatic RAW fence."""

import numpy as np

from common import auto_main, golden_output_case
from ptodsl import pto


@pto.jit(
    name="vmi_continuous_short_store_kernel",
    # Select the available CA simulator; the compiler proof is target independent.
    target="a5",
    backend="vpto",
    mode="explicit",
    kernel_kind="vector",
    insert_sync=False,
    ptoas_flags=("--enable-vecscope-mem-bar=true", "--vmi-fastmath=false"),
)
def _kernel(src: pto.ptr(pto.bf16, "gm"), dst: pto.ptr(pto.bf16, "gm")):
    source = pto.castptr(pto.i64(0), pto.ptr(pto.bf16, "ub"))
    maxima = pto.castptr(pto.i64(4096), pto.ptr(pto.bf16, "ub"))
    output = pto.castptr(pto.i64(6144), pto.ptr(pto.bf16, "ub"))
    pto.mte_gm_ub(src, source, 0, 1024, nburst=(1, 1024, 1024))
    pto.set_flag("MTE2", "V", event_id=0)
    pto.wait_flag("MTE2", "V", event_id=0)
    with pto.vecscope():
        mask = pto.vmi.create_mask(128, size=128)
        for chunk in range(4):
            values = pto.vmi.vload(source, chunk * 128, size=128)
            bits = pto.vmi.vinterpret_cast(values, pto.ui16)
            groups = pto.vmi.vcmax(bits, mask, group=8)
            bf16_groups = pto.vmi.vinterpret_cast(groups, pto.bf16)
            pto.vmi.vstore(bf16_groups, maxima, chunk * 8, stride=1, group=8)
        result = pto.vmi.vload(maxima, 0, size=128)
        output_mask = pto.vmi.create_mask(32, size=128)
        pto.vmi.vstore(result, output, 0, output_mask)
    pto.set_flag("V", "MTE3", event_id=0)
    pto.wait_flag("V", "MTE3", event_id=0)
    pto.mte_ub_gm(output, dst, 64, nburst=(1, 64, 64))
    pto.pipe_barrier(pto.Pipe.ALL)


def _source():
    rng = np.random.default_rng(1764)
    values = rng.uniform(0.0, 16.0, 512).astype(np.float32)
    return [(values.view(np.uint32) >> np.uint32(16)).astype(np.uint16)]


def _expected(source):
    # Positive finite BF16 values have the same ordering as their uint16 bits.
    return source.reshape(32, 16).max(axis=1)


CASES = [
    golden_output_case(
        "continuous_short_store_four_chunks",
        _kernel,
        inputs=_source,
        expected=_expected,
        rtol=0.0,
        atol=0.0,
    ),
]


auto_main(globals())
