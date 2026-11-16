#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Read ACC-to-MAT insertion back through an identity matmul (issue #1696)."""

import numpy as np

from common import auto_main, golden_output_case
from ptodsl import pto


def _tile(dtype, space, *, right=False):
    return pto.alloc_tile(
        shape=[16, 16], dtype=dtype, memory_space=space, valid_shape=[16, 16],
        blayout="RowMajor" if right else "ColMajor",
        slayout="ColMajor" if right else "RowMajor", fractal_size=512,
    )


def _view(ptr):
    return pto.make_tensor_view(ptr, shape=[1, 1, 1, 16, 16], strides=[256, 256, 256, 16, 1])


@pto.jit(
    name="insert_f16f16f32_acc2mat_m16k16n16", kernel_kind="cube",
    target="a5", backend="vpto", mode="auto",
)
def insert_acc_to_mat(
    lhs_ptr: pto.ptr(pto.f16, "gm"), rhs_ptr: pto.ptr(pto.f16, "gm"),
    eye_ptr: pto.ptr(pto.f32, "gm"), out_ptr: pto.ptr(pto.f32, "gm"),
):
    lhs_mat = _tile(pto.f16, pto.MemorySpace.MAT)
    rhs_mat = _tile(pto.f16, pto.MemorySpace.MAT)
    lhs = _tile(pto.f16, pto.MemorySpace.LEFT)
    rhs = _tile(pto.f16, pto.MemorySpace.RIGHT, right=True)
    acc = _tile(pto.f32, pto.MemorySpace.ACC)
    dst = _tile(pto.f32, pto.MemorySpace.MAT)
    left = _tile(pto.f32, pto.MemorySpace.LEFT)
    eye_mat = _tile(pto.f32, pto.MemorySpace.MAT)
    right = _tile(pto.f32, pto.MemorySpace.RIGHT, right=True)
    acc2 = _tile(pto.f32, pto.MemorySpace.ACC)
    pto.tile.load(_view(lhs_ptr), lhs_mat, sizes=[1, 1, 1, 16, 16])
    pto.tile.load(_view(rhs_ptr), rhs_mat, sizes=[1, 1, 1, 16, 16])
    pto.tile.mov(lhs_mat, lhs)
    pto.tile.mov(rhs_mat, rhs)
    pto.tile.matmul(lhs, rhs, acc)
    pto.tile.insert(acc, dst, pto.const(0, dtype=pto.index), pto.const(0, dtype=pto.index))
    pto.tile.mov(dst, left)
    pto.tile.load(_view(eye_ptr), eye_mat, sizes=[1, 1, 1, 16, 16])
    pto.tile.mov(eye_mat, right)
    pto.tile.matmul(left, right, acc2)
    pto.tile.store(acc2, _view(out_ptr), sizes=[1, 1, 1, 16, 16])


def make_inputs():
    rng = np.random.default_rng(1696)
    return [rng.uniform(-2, 2, (16, 16)).astype(np.float16),
            rng.uniform(-2, 2, (16, 16)).astype(np.float16), np.eye(16, dtype=np.float32)]


def make_expected(lhs, rhs, eye):
    return lhs.astype(np.float32) @ rhs.astype(np.float32)


CASES = [golden_output_case(
    "insert_f16f16f32_acc2mat_m16k16n16", insert_acc_to_mat,
    inputs=make_inputs, expected=make_expected, rtol=1e-4, atol=1e-4,
)]


auto_main(globals())
