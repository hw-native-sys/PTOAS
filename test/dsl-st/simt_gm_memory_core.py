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
from ptodsl import pto


LANES = 32


@pto.simt
def simt_gm_memory_core_body(
    inp: pto.ptr(pto.i32, "gm"),
    out: pto.ptr(pto.i32, "gm"),
):
    tid = pto.get_tid_x()
    idx = pto.index_cast(tid)
    loaded = pto.load(inp, idx)
    pto.store(loaded + tid + 1000, out, idx)
    pto.store(tid, out, pto.index_cast(tid + LANES))


@pto.jit(
    name="simt_gm_memory_core_kernel",
    kernel_kind="vector",
    target="a5",
    mode="explicit",
    insert_sync=False,
)
def simt_gm_memory_core_kernel(
    inp_ptr: pto.ptr(pto.i32, "gm"),
    out_ptr: pto.ptr(pto.i32, "gm"),
):
    simt_gm_memory_core_body[LANES, 1, 1](inp_ptr, out_ptr)
    pto.pipe_barrier(pto.Pipe.ALL)


def make_inputs():
    return [(np.arange(LANES, dtype=np.int32) * 3) - 17]


def make_expected(inp):
    tid = np.arange(LANES, dtype=np.int32)
    return np.concatenate([inp + tid + 1000, tid])


CASES = [
    golden_output_case(
        "simt_gm_memory_core",
        simt_gm_memory_core_kernel,
        inputs=make_inputs,
        expected=make_expected,
        rtol=0.0,
        atol=0.0,
    ),
]


auto_main(globals())
