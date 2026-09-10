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


M = 16
K = 32
N = 64
ELEM_BYTES = 4

L1_A_ADDR = 0
L1_B_ADDR = 4096
L0A_ADDR = 0
L0B_ADDR = 0
L0C_ADDR = 0


@pto.jit(
    name="cube_matrix_pipeline_kernel",
    kernel_kind="cube",
    target="a5",
    mode="explicit",
    insert_sync=False,
)
def cube_matrix_pipeline_kernel(
    a_ptr: pto.ptr(pto.f32, "gm"),
    b_ptr: pto.ptr(pto.f32, "gm"),
    o_ptr: pto.ptr(pto.f32, "gm"),
):
    a_mat = pto.alloc_tile(
        shape=[M, K],
        dtype=pto.f32,
        memory_space=pto.MemorySpace.MAT,
        addr=L1_A_ADDR,
        valid_shape=[M, K],
        blayout="ColMajor",
        slayout="RowMajor",
    )
    b_mat = pto.alloc_tile(
        shape=[K, N],
        dtype=pto.f32,
        memory_space=pto.MemorySpace.MAT,
        addr=L1_B_ADDR,
        valid_shape=[K, N],
        blayout="ColMajor",
        slayout="RowMajor",
    )
    a_l0a = pto.alloc_tile(
        shape=[M, K],
        dtype=pto.f32,
        memory_space=pto.MemorySpace.LEFT,
        addr=L0A_ADDR,
        valid_shape=[M, K],
        blayout="ColMajor",
        slayout="RowMajor",
    )
    b_l0b = pto.alloc_tile(
        shape=[K, N],
        dtype=pto.f32,
        memory_space=pto.MemorySpace.RIGHT,
        addr=L0B_ADDR,
        valid_shape=[K, N],
        blayout="RowMajor",
        slayout="ColMajor",
    )
    o_acc = pto.alloc_tile(
        shape=[M, N],
        dtype=pto.f32,
        memory_space=pto.MemorySpace.ACC,
        addr=L0C_ADDR,
        valid_shape=[M, N],
        blayout="ColMajor",
        slayout="RowMajor",
        fractal_size=1024,
    )

    a_l1_ptr = pto.castptr(pto.ui64(L1_A_ADDR), pto.ptr(pto.f32, "mat"))
    b_l1_ptr = pto.castptr(pto.ui64(L1_B_ADDR), pto.ptr(pto.f32, "mat"))

    pto.mte_gm_l1_frac(
        a_ptr,
        a_l1_ptr,
        pto.FractalMode.ND2NZ,
        shape=(M, K),
        src_layout=(K * ELEM_BYTES,),
        dst_group=(1, 1, M, 0),
        ctrl=(0, False),
    )
    pto.set_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=0)
    pto.wait_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=0)
    pto.mte_l1_l0a(a_l1_ptr, a_l0a.as_ptr(), M, K)

    pto.mte_gm_l1_frac(
        b_ptr,
        b_l1_ptr,
        pto.FractalMode.ND2NZ,
        shape=(K, N),
        src_layout=(N * ELEM_BYTES,),
        dst_group=(1, 1, K, 0),
        ctrl=(0, False),
    )
    pto.set_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=1)
    pto.wait_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=1)
    pto.mte_l1_l0b(b_l1_ptr, b_l0b.as_ptr(), K, N, transpose=True)

    pto.set_flag(pto.Pipe.MTE1, pto.Pipe.M, event_id=0)
    pto.wait_flag(pto.Pipe.MTE1, pto.Pipe.M, event_id=0)
    pto.tile.matmul(a_l0a, b_l0b, o_acc)

    pto.set_flag(pto.Pipe.M, pto.Pipe.FIX, event_id=1)
    pto.wait_flag(pto.Pipe.M, pto.Pipe.FIX, event_id=1)
    pto.mte_l0c_gm(
        o_acc.as_ptr(),
        o_ptr,
        M,
        N,
        M,
        N,
        0,
        0,
        layout="nz2nd",
    )
    pto.pipe_barrier(pto.Pipe.ALL)


def make_inputs():
    a = (np.arange(M * K, dtype=np.float32).reshape(M, K) % 7) - 3.0
    b = (np.arange(K * N, dtype=np.float32).reshape(K, N) % 5) - 2.0
    return [a, b]


def make_expected(a, b):
    return a @ b


CASES = [
    golden_output_case(
        "cube_matrix_pipeline_gemm",
        cube_matrix_pipeline_kernel,
        inputs=make_inputs,
        expected=make_expected,
        rtol=1e-4,
        atol=1e-4,
    ),
]


auto_main(globals())
