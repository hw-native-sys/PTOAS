#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Minimal PTODSL cube/tmatmul pilot for A5.
# Goal: validate plain cube tile.matmul lowering/runtime first, without mixing
# MX-specific scale/bias handling or decorated tileop helper boundaries.

from pathlib import Path
import sys

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

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

# This case keeps explicit L1/L0 addresses because the current GM->L1 fractal
# path passes raw MAT pointers into mte_gm_l1_frac. Vector tile cases in this
# directory use automatic tile address allocation.


@pto.tileop
def cube_matmul_tile(
    a_mat: pto.Tile,
    b_mat: pto.Tile,
    o_tile: pto.Tile,
    a_l0a: pto.Tile,
    b_l0b: pto.Tile,
    c_acc: pto.Tile,
):
    m = a_mat.valid_shape[0]
    k = a_mat.valid_shape[1]
    n = b_mat.valid_shape[1]

    pto.mte_l1_l0a(a_mat.as_ptr(), a_l0a.as_ptr(), m, k)
    pto.mte_l1_l0b(b_mat.as_ptr(), b_l0b.as_ptr(), k, n, transpose=True)
    pto.set_flag(pto.Pipe.MTE1, pto.Pipe.M, event_id=1)
    pto.wait_flag(pto.Pipe.MTE1, pto.Pipe.M, event_id=1)
    pto.tile.matmul(a_l0a, b_l0b, c_acc)
    pto.set_flag(pto.Pipe.M, pto.Pipe.FIX, event_id=2)
    pto.wait_flag(pto.Pipe.M, pto.Pipe.FIX, event_id=2)
    pto.mte_l0c_ub(
        c_acc.as_ptr(),
        o_tile.as_ptr(),
        m,
        n,
        n,
        n,
    )


@pto.jit(
    name="tmatmul_f32_16x32x64",
    kernel_kind="cube",
    target="a5",
    mode="explicit",
    insert_sync=False,
)
def _tmatmul_kernel(
    a_ptr: pto.ptr(pto.f32, "gm"),
    b_ptr: pto.ptr(pto.f32, "gm"),
    c_ptr: pto.ptr(pto.f32, "gm"),
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
    c_acc = pto.alloc_tile(
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
    pto.tile.matmul(a_l0a, b_l0b, c_acc)

    pto.set_flag(pto.Pipe.M, pto.Pipe.FIX, event_id=1)
    pto.wait_flag(pto.Pipe.M, pto.Pipe.FIX, event_id=1)
    pto.mte_l0c_gm(
        c_acc.as_ptr(),
        c_ptr,
        M,
        N,
        M,
        N,
        0,
        0,
        layout="nz2nd",
    )
    pto.pipe_barrier(pto.Pipe.ALL)


def _make_inputs():
    rng = np.random.default_rng(0x7A7A7A71)
    a = rng.uniform(-2.0, 2.0, size=(M, K)).astype(np.float32)
    b = rng.uniform(-2.0, 2.0, size=(K, N)).astype(np.float32)
    return [a, b]


def _make_expected(a, b):
    return (a @ b).astype(np.float32)


CASES = [
    golden_output_case(
        "tmatmul_f32_16x32x64",
        _tmatmul_kernel,
        inputs=_make_inputs,
        expected=_make_expected,
        rtol=1e-4,
        atol=1e-4,
    ),
]


auto_main(globals())
