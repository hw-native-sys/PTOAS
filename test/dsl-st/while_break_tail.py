#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Numeric ST for break/continue tail predication (issue #1256 follow-up).

Statements after a break/continue must be skipped for the rest of the
iteration.  Before the fix the whole loop body ran under a constant-true
guard, so the tail executed unconditionally in the stopping iteration:

- case 0 (while True + if/break + tail): limit=3 must yield 3, not 4.
- case 1 (while + continue + tail): the +10 after continue must be skipped,
  giving 20, not 30.
- case 2 (for + break + tail): limit=3 must yield 3, not 4.
"""

import numpy as np

from common import auto_main, golden_output_case
from ptodsl import pto


@pto.simt
def while_break_tail_body(output_ptr: pto.ptr(pto.i32, "gm")) -> None:
    limit = pto.const(3, dtype=pto.i32)

    value = pto.const(0, dtype=pto.i32)
    while True:
        should_break = value >= limit
        if should_break:
            break
        value = value + pto.const(1, dtype=pto.i32)
    pto.store(value, output_ptr, 0)

    i = pto.const(0, dtype=pto.i32)
    total = pto.const(0, dtype=pto.i32)
    while i < limit:
        i = i + pto.const(1, dtype=pto.i32)
        if i == pto.const(2, dtype=pto.i32):
            continue
        total = total + pto.const(10, dtype=pto.i32)
    pto.store(total, output_ptr, 1)

    fval = pto.const(0, dtype=pto.i32)
    ten = pto.const(10, dtype=pto.i32)
    for j in range(ten):
        if j == limit:
            break
        fval = fval + pto.const(1, dtype=pto.i32)
    pto.store(fval, output_ptr, 2)


@pto.jit(
    name="while_break_tail_numeric",
    kernel_kind="vector",
    target="a5",
    mode="explicit",
    insert_sync=False,
)
def while_break_tail_numeric(
    output_ptr: pto.ptr(pto.i32, "gm"),
) -> None:
    while_break_tail_body[1, 1, 1](output_ptr)
    pto.pipe_barrier(pto.Pipe.ALL)


CASES = [
    golden_output_case(
        "while_break_tail_numeric",
        while_break_tail_numeric,
        inputs=lambda: [],
        expected=np.array([3, 20, 3], dtype=np.int32),
        output_shape=(3,),
        output_dtype=np.int32,
    )
]


if __name__ == "__main__":
    auto_main(globals())
