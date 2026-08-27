#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Runtime numerical coverage for Python ``and``/``or`` expressions."""

import numpy as np

from common import auto_main, golden_output_case
from ptodsl import pto, scalar


ELEMENTS = 8


@pto.jit(
    name="boolop_and_short_circuit",
    kernel_kind="vector",
    target="a5",
    mode="explicit",
    insert_sync=False,
)
def boolop_and_short_circuit(
    values: pto.ptr(pto.i32, "gm"),
    divisors: pto.ptr(pto.i32, "gm"),
    output: pto.ptr(pto.i8, "gm"),
):
    for i in range(ELEMENTS):
        value = scalar.load(values, i)
        divisor = scalar.load(divisors, i)
        result = (divisor != 0) and ((value // divisor) > 0)
        scalar.store(result, output, i)


@pto.jit(
    name="boolop_or_short_circuit",
    kernel_kind="vector",
    target="a5",
    mode="explicit",
    insert_sync=False,
)
def boolop_or_short_circuit(
    values: pto.ptr(pto.i32, "gm"),
    divisors: pto.ptr(pto.i32, "gm"),
    output: pto.ptr(pto.i8, "gm"),
):
    for i in range(ELEMENTS):
        value = scalar.load(values, i)
        divisor = scalar.load(divisors, i)
        result = (divisor == 0) or ((value // divisor) > 0)
        scalar.store(result, output, i)


@pto.jit(
    name="boolop_int_or_keeps_operand",
    kernel_kind="vector",
    target="a5",
    mode="explicit",
    insert_sync=False,
)
def boolop_int_or_keeps_operand(
    lhs: pto.ptr(pto.i32, "gm"),
    rhs: pto.ptr(pto.i32, "gm"),
    output: pto.ptr(pto.i32, "gm"),
):
    for i in range(ELEMENTS):
        x = scalar.load(lhs, i)
        y = scalar.load(rhs, i)
        result = x or (y > 0)
        scalar.store(result, output, i)


@pto.jit(
    name="boolop_flag_and_integer_rhs",
    kernel_kind="vector",
    target="a5",
    mode="explicit",
    insert_sync=False,
)
def boolop_flag_and_integer_rhs(
    flags: pto.ptr(pto.i8, "gm"),
    values: pto.ptr(pto.i32, "gm"),
    output: pto.ptr(pto.i32, "gm"),
):
    for i in range(ELEMENTS):
        flag = scalar.load(flags, i)
        value = scalar.load(values, i)
        result = (flag != 0) and value
        scalar.store(result, output, i)


@pto.jit(
    name="boolop_index_and_i1",
    kernel_kind="vector",
    target="a5",
    mode="explicit",
    insert_sync=False,
)
def boolop_index_and_i1(
    indices: pto.ptr(pto.i32, "gm"),
    values: pto.ptr(pto.i32, "gm"),
    output: pto.ptr(pto.i32, "gm"),
):
    for i in range(ELEMENTS):
        index = scalar.index_cast(scalar.load(indices, i))
        value = scalar.load(values, i)
        result = index and (value > 0)
        scalar.store(result, output, i)


def make_division_inputs():
    values = np.array([8, 4, 0, 7, -6, 9, 3, 0], dtype=np.int32)
    divisors = np.array([2, 0, -2, 1, -3, 0, 1, 4], dtype=np.int32)
    return [values, divisors]


def make_and_expected(values, divisors):
    safe_divisors = np.where(divisors != 0, divisors, 1)
    return ((divisors != 0) & (np.floor_divide(values, safe_divisors) > 0)).astype(np.int8)


def make_or_expected(values, divisors):
    safe_divisors = np.where(divisors != 0, divisors, 1)
    return ((divisors == 0) | (np.floor_divide(values, safe_divisors) > 0)).astype(np.int8)


def make_integer_inputs():
    lhs = np.array([2, 0, 5, -3, 7, 0, 1, 0], dtype=np.int32)
    rhs = np.array([1, -2, 3, 0, 4, 5, -1, 0], dtype=np.int32)
    return [lhs, rhs]


def make_int_or_expected(lhs, rhs):
    return np.where(lhs != 0, lhs, (rhs > 0).astype(np.int32)).astype(np.int32)


def make_flag_inputs():
    flags = np.array([1, 0, 1, 0, 1, 1, 0, 1], dtype=np.int8)
    values = np.array([7, 3, -4, 6, 2, 0, 9, 8], dtype=np.int32)
    return [flags, values]


def make_flag_and_expected(flags, values):
    return np.where(flags != 0, values, 0).astype(np.int32)


def make_index_inputs():
    indices = np.array([3, 0, -2, 0, 5, 1, 0, 2], dtype=np.int32)
    values = np.array([1, 4, 2, 0, -3, 0, 6, 7], dtype=np.int32)
    return [indices, values]


def make_index_and_expected(indices, values):
    return np.where(indices != 0, (values > 0).astype(np.int32), 0).astype(np.int32)


CASES = [
    golden_output_case(
        "boolop_and_short_circuit",
        boolop_and_short_circuit,
        inputs=make_division_inputs,
        expected=make_and_expected,
        rtol=0.0,
        atol=0.0,
    ),
    golden_output_case(
        "boolop_or_short_circuit",
        boolop_or_short_circuit,
        inputs=make_division_inputs,
        expected=make_or_expected,
        rtol=0.0,
        atol=0.0,
    ),
    golden_output_case(
        "boolop_int_or_keeps_operand",
        boolop_int_or_keeps_operand,
        inputs=make_integer_inputs,
        expected=make_int_or_expected,
        rtol=0.0,
        atol=0.0,
    ),
    golden_output_case(
        "boolop_flag_and_integer_rhs",
        boolop_flag_and_integer_rhs,
        inputs=make_flag_inputs,
        expected=make_flag_and_expected,
        rtol=0.0,
        atol=0.0,
    ),
    golden_output_case(
        "boolop_index_and_i1",
        boolop_index_and_i1,
        inputs=make_index_inputs,
        expected=make_index_and_expected,
        rtol=0.0,
        atol=0.0,
    ),
]


auto_main(globals())
