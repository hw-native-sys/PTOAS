#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""
VecValue arithmetic end-to-end ST.

Exercises the add / sub / mul floating-point vector arithmetic operators on
``vector<4xf32>`` values produced by ``pto.load(..., contiguous=VEC)`` and
consumed by ``pto.store`` inside an ``@pto.simt`` body. ``pto.Vec`` is the
SIMT lane-local vector type, so the SIMT kernel is the target scenario.

Vector division is intentionally omitted: the backend does not support it, so
the PTODSL surface does not expose ``__truediv__`` on ``VecValue``.
"""

import numpy as np

from common import auto_main, golden_output_case
from ptodsl import pto


VEC = 4
THREADS = 32
SEED = 0x5761


@pto.simt
def vec_value_arith_body(
    a_ptr: pto.ptr(pto.f32, "gm"),
    o_ptr: pto.ptr(pto.f32, "gm"),
):
    tid = pto.get_tid_x()
    idx = pto.index_cast(tid)
    base = idx * 2 * VEC
    x = pto.load(a_ptr, base, contiguous=VEC)
    y = pto.load(a_ptr, base + VEC, contiguous=VEC)
    pto.store(x + y, o_ptr, idx * VEC)
    pto.store(x - y, o_ptr, THREADS * VEC + idx * VEC)
    pto.store(x * y, o_ptr, 2 * THREADS * VEC + idx * VEC)


@pto.jit(
    name="vec_value_arith_kernel",
    kernel_kind="vector",
    target="a5",
    mode="explicit",
    insert_sync=False,
)
def vec_value_arith_kernel(
    a_ptr: pto.ptr(pto.f32, "gm"),
    o_ptr: pto.ptr(pto.f32, "gm"),
):
    vec_value_arith_body[THREADS, 1, 1](a_ptr, o_ptr)
    pto.pipe_barrier(pto.Pipe.ALL)


def make_inputs():
    rng = np.random.RandomState(SEED)
    xy = rng.uniform(0.5, 2.0, size=(THREADS, 2 * VEC)).astype(np.float32)
    return [xy.reshape(-1)]


def make_expected(inp):
    xy = inp.reshape(THREADS, 2 * VEC)
    x = xy[:, :VEC]
    y = xy[:, VEC:]
    return np.concatenate([
        (x + y).reshape(-1),
        (x - y).reshape(-1),
        (x * y).reshape(-1),
    ]).astype(np.float32)


CASES = [
    golden_output_case(
        "vec_value_arith",
        vec_value_arith_kernel,
        inputs=make_inputs,
        expected=make_expected,
        rtol=1.0e-5,
        atol=1.0e-5,
    ),
]


auto_main(globals())
