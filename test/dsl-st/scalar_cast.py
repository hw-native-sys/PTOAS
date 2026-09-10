#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""End-to-end SIMT coverage for the public ``scalar.cast`` API.

The SIMT body exercises both per-thread runtime scalars and lane-local builtin
vectors returned by ``scalar.load(..., contiguous=VEC)``.
"""

import numpy as np

from common import auto_main, golden_output_case
from ptodsl import pto, scalar


VEC = 4
THREADS = 32
ELEMENTS = THREADS * (1 + VEC)


@pto.simt
def scalar_cast_simt_body(
    input_ptr: pto.ptr(pto.f32, "gm"),
    output_ptr: pto.ptr(pto.f32, "gm"),
) -> None:
    tid = pto.get_tid_x()
    idx = scalar.index_cast(tid)

    scalar_value = scalar.load(input_ptr, idx)
    narrowed_scalar = scalar.cast(scalar_value, pto.f16)
    widened_scalar = scalar.cast(narrowed_scalar, pto.f32)
    scalar.store(widened_scalar, output_ptr, idx)

    vector_base = THREADS + idx * VEC
    vector_value = scalar.load(input_ptr, vector_base, contiguous=VEC)
    narrowed_vector = scalar.cast(vector_value, pto.f16)
    widened_vector = scalar.cast(narrowed_vector, pto.Vec(pto.f32, VEC))
    scalar.store(widened_vector, output_ptr, vector_base)


@pto.jit(
    name="scalar_cast_simt_f32_f16_round_trip",
    kernel_kind="vector",
    target="a5",
    mode="explicit",
    insert_sync=False,
)
def scalar_cast_simt_f32_f16_round_trip(
    input_ptr: pto.ptr(pto.f32, "gm"),
    output_ptr: pto.ptr(pto.f32, "gm"),
) -> None:
    scalar_cast_simt_body[THREADS, 1, 1](input_ptr, output_ptr)
    pto.pipe_barrier(pto.Pipe.ALL)


def make_inputs():
    input_data = np.linspace(
        -3.25,
        3.25,
        num=ELEMENTS,
        dtype=np.float32,
    )
    input_data += np.float32(0.0013)
    return [input_data]


def make_expected(input_data):
    return input_data.astype(np.float16).astype(np.float32)


CASES = [
    golden_output_case(
        "scalar_cast_simt_f32_f16_round_trip",
        scalar_cast_simt_f32_f16_round_trip,
        inputs=make_inputs,
        expected=make_expected,
        rtol=0.0,
        atol=0.0,
    ),
]


auto_main(globals())
