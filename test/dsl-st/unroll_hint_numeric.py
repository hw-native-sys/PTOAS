#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Numeric ST for loop unroll hints (issue #1242 requirement 2).

Exercises the native-unroll hint paths with checkable numerics:

- out[0]: pto.range(..., unroll="full") -> native full unroll (pto-unroll-loops).
- out[1]: pto.range(..., unroll_factor=4) with a non-divisible trip count ->
  native main loop + epilogue (pto-unroll-loops).
- out[2]: pto.range(..., unroll_factor=3) with an odd factor -> another
  main loop + epilogue combination; the loop must still compute correctly.
- out[3]: pto.range(..., unroll="enable") -> the loop is kept and forwarded
  as llvm.loop.unroll.enable metadata (pto-convert-scf-to-cf-with-loop-hints); it must still
  execute correctly.
- out[4]: nested unroll="full" -> unrolling the outer loop clones the inner
  one; the clones must be consumed as well (review fix).
"""

import numpy as np

from common import auto_main, golden_output_case
from ptodsl import pto


@pto.simt
def unroll_hint_body(output_ptr: pto.ptr(pto.i32, "gm")) -> None:
    full = pto.const(0, dtype=pto.i32)
    for i in pto.range(4, unroll="full"):
        full = full + pto.cast(i, pto.i32)
    pto.store(full, output_ptr, 0)

    fact = pto.const(0, dtype=pto.i32)
    for i in pto.range(0, 10, 1, unroll_factor=4):
        fact = fact + pto.cast(i, pto.i32)
    pto.store(fact, output_ptr, 1)

    odd = pto.const(0, dtype=pto.i32)
    for i in pto.range(8, unroll_factor=3):
        odd = odd + pto.cast(i, pto.i32)
    pto.store(odd, output_ptr, 2)

    meta = pto.const(0, dtype=pto.i32)
    for i in pto.range(8, unroll="enable"):
        meta = meta + pto.cast(i, pto.i32)
    pto.store(meta, output_ptr, 3)

    nested = pto.const(0, dtype=pto.i32)
    ten = pto.const(10, dtype=pto.i32)
    for i in pto.range(2, unroll="full"):
        i32 = pto.cast(i, pto.i32)
        for j in pto.range(3, unroll="full"):
            nested = nested + i32 * ten + pto.cast(j, pto.i32)
    pto.store(nested, output_ptr, 4)


@pto.jit(
    name="unroll_hint_numeric",
    kernel_kind="vector",
    target="a5",
    mode="explicit",
    insert_sync=False,
)
def unroll_hint_numeric(
    output_ptr: pto.ptr(pto.i32, "gm"),
) -> None:
    unroll_hint_body[1, 1, 1](output_ptr)
    pto.pipe_barrier(pto.Pipe.ALL)


CASES = [
    golden_output_case(
        "unroll_hint_numeric",
        unroll_hint_numeric,
        inputs=lambda: [],
        expected=np.array([6, 45, 28, 28, 36], dtype=np.int32),
        output_shape=(5,),
        output_dtype=np.int32,
    )
]


if __name__ == "__main__":
    auto_main(globals())
