#!/usr/bin/env python3
# coding=utf-8
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


LANES = 32
dtype = pto.i32


@pto.simt
def simt_pipe_barrier_body(out: pto.ptr(dtype, "gm")):
    tid = pto.get_tid_x()
    pto.pipe_barrier(pto.Pipe.ALL)
    pto.stg(tid, out, pto.cast(tid, pto.index))


@pto.jit(
    name="simt_pipe_barrier_kernel",
    kernel_kind="vector",
    target="a5",
    backend="vpto",
    mode="explicit",
)
def simt_pipe_barrier_kernel(out_ptr: pto.ptr(dtype, "gm")):
    pto.simt_launch(simt_pipe_barrier_body, out_ptr, dims=(LANES, 1, 1))
    pto.pipe_barrier(pto.Pipe.ALL)


@pto.jit(
    name="simt_pipe_barrier_call_kernel",
    kernel_kind="vector",
    target="a5",
    backend="vpto",
    mode="explicit",
)
def simt_pipe_barrier_call_kernel(out_ptr: pto.ptr(dtype, "gm")):
    simt_pipe_barrier_body[LANES, 1, 1](out_ptr)
    pto.pipe_barrier(pto.Pipe.ALL)


def make_expected():
    out = np.zeros(2 * LANES, dtype=np.int32)
    out[:LANES] = np.arange(LANES, dtype=np.int32)
    return out


CASES = [
    golden_output_case(
        "simt_pipe_barrier_i32_h32",
        simt_pipe_barrier_kernel,
        inputs=lambda: [],
        expected=make_expected,
        rtol=0.0,
        atol=0.0,
    ),
    golden_output_case(
        "simt_pipe_barrier_call_i32_h32",
        simt_pipe_barrier_call_kernel,
        inputs=lambda: [],
        expected=make_expected,
        rtol=0.0,
        atol=0.0,
    ),
]

auto_main(globals())
