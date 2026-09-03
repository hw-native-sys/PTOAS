#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""PTODSL port of the legacy A5 ordinary GEMV ST suite.

The two kernels preserve the legacy explicit cube movement: the first is a
split-K ``gemv``/``gemv_acc`` pair, and the second is a
``gemv_bias``/``gemv_acc`` pair.  Inputs are supplied as separate K chunks,
which is equivalent to the legacy launcher passing offset pointers into its
padded GM buffers.

Movement surface: the ordinary GM->L1 ND2NZ data loads (rank-5 views through
``pto.tile.load``) and the ACC->GM ``pto.tile.store`` output go through the
public tile surface.  The cube-specific pieces stay explicit: the bias GM->L1
load (``mte_gm_l1``), the L1->L0A/L0B moves
(``mte_l1_l0a``/``mte_l1_l0b``), the bias L1->BT staging (``mte_l1_bt``), and
the set_flag/wait_flag syncs.
"""

from pathlib import Path
import sys
import zlib

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import auto_main, golden_output_case
from ptodsl import pto


@pto.jit(
    name="gemv_f16_1x300x60",
    kernel_kind="cube",
    target="a5",
    mode="explicit",
    insert_sync=False,
)
def _kernel_gemv_f16_1x300x60(
    a0_ptr: pto.ptr(pto.f16, "gm"),
    b0_ptr: pto.ptr(pto.f16, "gm"),
    a1_ptr: pto.ptr(pto.f16, "gm"),
    b1_ptr: pto.ptr(pto.f16, "gm"),
    c_ptr: pto.ptr(pto.f32, "gm"),
):
    l1_a0 = pto.alloc_tile(
        shape=[1, 256], dtype=pto.f16, memory_space=pto.MemorySpace.MAT, addr=0,
        valid_shape=[1, 256], blayout="ColMajor", slayout="RowMajor",
    )
    l1_b0 = pto.alloc_tile(
        shape=[256, 64], dtype=pto.f16, memory_space=pto.MemorySpace.MAT, addr=512,
        valid_shape=[256, 64], blayout="ColMajor", slayout="RowMajor",
    )
    l1_a1 = pto.alloc_tile(
        shape=[1, 64], dtype=pto.f16, memory_space=pto.MemorySpace.MAT, addr=33280,
        valid_shape=[1, 64], blayout="ColMajor", slayout="RowMajor",
    )
    l1_b1 = pto.alloc_tile(
        shape=[64, 64], dtype=pto.f16, memory_space=pto.MemorySpace.MAT, addr=33408,
        valid_shape=[64, 64], blayout="ColMajor", slayout="RowMajor",
    )
    l0a0 = pto.alloc_tile(
        shape=[1, 256], dtype=pto.f16, memory_space=pto.MemorySpace.LEFT, addr=0,
        valid_shape=[1, 256], blayout="ColMajor", slayout="RowMajor",
    )
    l0b0 = pto.alloc_tile(
        shape=[256, 64], dtype=pto.f16, memory_space=pto.MemorySpace.RIGHT, addr=0,
        valid_shape=[256, 64], blayout="RowMajor", slayout="ColMajor",
    )
    l0a1 = pto.alloc_tile(
        shape=[1, 64], dtype=pto.f16, memory_space=pto.MemorySpace.LEFT, addr=0,
        valid_shape=[1, 64], blayout="ColMajor", slayout="RowMajor",
    )
    l0b1 = pto.alloc_tile(
        shape=[64, 64], dtype=pto.f16, memory_space=pto.MemorySpace.RIGHT, addr=0,
        valid_shape=[64, 64], blayout="RowMajor", slayout="ColMajor",
    )
    acc = pto.alloc_tile(
        shape=[1, 64], dtype=pto.f32, memory_space=pto.MemorySpace.ACC, addr=0,
        valid_shape=[1, 64], blayout="ColMajor", slayout="RowMajor", fractal_size=1024,
    )

    a0_view = pto.make_tensor_view(a0_ptr, shape=[1, 1, 1, 1, 256], strides=[256, 256, 256, 256, 1])
    b0_view = pto.make_tensor_view(
        b0_ptr, shape=[1, 1, 1, 256, 64], strides=[256 * 64, 256 * 64, 256 * 64, 64, 1],
    )
    a1_view = pto.make_tensor_view(a1_ptr, shape=[1, 1, 1, 1, 64], strides=[64, 64, 64, 64, 1])
    b1_view = pto.make_tensor_view(
        b1_ptr, shape=[1, 1, 1, 64, 64], strides=[64 * 64, 64 * 64, 64 * 64, 64, 1],
    )
    c_view = pto.make_tensor_view(
        c_ptr, shape=[1, 1, 1, 1, 64], strides=[64, 64, 64, 64, 1],
    )

    pto.tile.load(a0_view, l1_a0, offsets=[0, 0, 0, 0, 0], sizes=[1, 1, 1, 1, 256])
    pto.tile.load(
        b0_view, l1_b0, offsets=[0, 0, 0, 0, 0], sizes=[1, 1, 1, 256, 64],
    )
    pto.set_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=0)
    pto.wait_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=0)
    pto.mte_l1_l0a(l1_a0.as_ptr(), l0a0.as_ptr(), 1, 256)
    pto.mte_l1_l0b(l1_b0.as_ptr(), l0b0.as_ptr(), 256, 64, transpose=True)
    pto.set_flag(pto.Pipe.MTE1, pto.Pipe.M, event_id=0)
    pto.wait_flag(pto.Pipe.MTE1, pto.Pipe.M, event_id=0)
    pto.tile.gemv(l0a0, l0b0, acc)
    pto.set_flag(pto.Pipe.M, pto.Pipe.MTE2, event_id=0)
    pto.wait_flag(pto.Pipe.M, pto.Pipe.MTE2, event_id=0)

    pto.tile.load(a1_view, l1_a1, offsets=[0, 0, 0, 0, 0], sizes=[1, 1, 1, 1, 64])
    pto.tile.load(b1_view, l1_b1, offsets=[0, 0, 0, 0, 0], sizes=[1, 1, 1, 64, 64])
    pto.set_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=1)
    pto.wait_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=1)
    pto.mte_l1_l0a(l1_a1.as_ptr(), l0a1.as_ptr(), 1, 64)
    pto.mte_l1_l0b(l1_b1.as_ptr(), l0b1.as_ptr(), 64, 64, transpose=True)
    pto.set_flag(pto.Pipe.MTE1, pto.Pipe.M, event_id=1)
    pto.wait_flag(pto.Pipe.MTE1, pto.Pipe.M, event_id=1)
    pto.tile.gemv_acc(acc, l0a1, l0b1, acc)

    pto.set_flag(pto.Pipe.M, pto.Pipe.FIX, event_id=1)
    pto.wait_flag(pto.Pipe.M, pto.Pipe.FIX, event_id=1)
    pto.tile.store(acc, c_view, offsets=[0, 0, 0, 0, 0], sizes=[1, 1, 1, 1, 64])
    pto.pipe_barrier(pto.Pipe.ALL)


@pto.jit(
    name="gemv_bias_f16_1x512x85",
    kernel_kind="cube",
    target="a5",
    mode="explicit",
    insert_sync=False,
)
def _kernel_gemv_bias_f16_1x512x85(
    a0_ptr: pto.ptr(pto.f16, "gm"),
    b0_ptr: pto.ptr(pto.f16, "gm"),
    a1_ptr: pto.ptr(pto.f16, "gm"),
    b1_ptr: pto.ptr(pto.f16, "gm"),
    bias_ptr: pto.ptr(pto.f32, "gm"),
    c_ptr: pto.ptr(pto.f32, "gm"),
):
    l0a = pto.alloc_tile(
        shape=[1, 256], dtype=pto.f16, memory_space=pto.MemorySpace.LEFT, addr=0,
        valid_shape=[1, 256], blayout="ColMajor", slayout="RowMajor",
    )
    l0b = pto.alloc_tile(
        shape=[256, 96], dtype=pto.f16, memory_space=pto.MemorySpace.RIGHT, addr=0,
        valid_shape=[256, 96], blayout="RowMajor", slayout="ColMajor",
    )
    bias = pto.alloc_tile(
        shape=[1, 96], dtype=pto.f32, memory_space=pto.MemorySpace.BIAS, addr=0,
        valid_shape=[1, 96], blayout="RowMajor", slayout="NoneBox", fractal_size=512,
    )
    acc = pto.alloc_tile(
        shape=[1, 96], dtype=pto.f32, memory_space=pto.MemorySpace.ACC, addr=0,
        valid_shape=[1, 96], blayout="ColMajor", slayout="RowMajor", fractal_size=1024,
    )
    c_view = pto.make_tensor_view(
        c_ptr, shape=[1, 1, 1, 1, 96], strides=[96, 96, 96, 96, 1],
    )

    l1_a = pto.alloc_tile(
        shape=[1, 256], dtype=pto.f16, memory_space=pto.MemorySpace.MAT, addr=0,
        valid_shape=[1, 256], blayout="ColMajor", slayout="RowMajor",
    )
    l1_b = pto.alloc_tile(
        shape=[256, 96], dtype=pto.f16, memory_space=pto.MemorySpace.MAT, addr=512,
        valid_shape=[256, 96], blayout="ColMajor", slayout="RowMajor",
    )
    l1_bias_ptr = pto.castptr(pto.ui64(49664), pto.ptr(pto.f32, "mat"))

    a0_view = pto.make_tensor_view(
        a0_ptr, shape=[1, 1, 1, 1, 256], strides=[256, 256, 256, 256, 1],
    )
    b0_view = pto.make_tensor_view(
        b0_ptr, shape=[1, 1, 1, 256, 96], strides=[256 * 96, 256 * 96, 256 * 96, 96, 1],
    )
    a1_view = pto.make_tensor_view(
        a1_ptr, shape=[1, 1, 1, 1, 256], strides=[256, 256, 256, 256, 1],
    )
    b1_view = pto.make_tensor_view(
        b1_ptr, shape=[1, 1, 1, 256, 96], strides=[256 * 96, 256 * 96, 256 * 96, 96, 1],
    )

    pto.tile.load(a0_view, l1_a, offsets=[0, 0, 0, 0, 0], sizes=[1, 1, 1, 1, 256])
    pto.tile.load(b0_view, l1_b, offsets=[0, 0, 0, 0, 0], sizes=[1, 1, 1, 256, 96])
    pto.mte_gm_l1(bias_ptr, l1_bias_ptr, 384, nburst=(1, 0, 0))
    pto.set_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=0)
    pto.wait_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=0)
    pto.mte_l1_l0a(l1_a.as_ptr(), l0a.as_ptr(), 1, 256)
    pto.mte_l1_l0b(l1_b.as_ptr(), l0b.as_ptr(), 256, 96, transpose=True)
    pto.mte_l1_bt(l1_bias_ptr, bias.as_ptr(), 96, nburst=(1, 0, 0))
    pto.set_flag(pto.Pipe.MTE1, pto.Pipe.M, event_id=0)
    pto.wait_flag(pto.Pipe.MTE1, pto.Pipe.M, event_id=0)
    pto.tile.gemv_bias(l0a, l0b, bias, acc)
    pto.set_flag(pto.Pipe.M, pto.Pipe.MTE2, event_id=0)
    pto.wait_flag(pto.Pipe.M, pto.Pipe.MTE2, event_id=0)

    pto.tile.load(a1_view, l1_a, offsets=[0, 0, 0, 0, 0], sizes=[1, 1, 1, 1, 256])
    pto.tile.load(b1_view, l1_b, offsets=[0, 0, 0, 0, 0], sizes=[1, 1, 1, 256, 96])
    pto.set_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=1)
    pto.wait_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=1)
    pto.mte_l1_l0a(l1_a.as_ptr(), l0a.as_ptr(), 1, 256)
    pto.mte_l1_l0b(l1_b.as_ptr(), l0b.as_ptr(), 256, 96, transpose=True)
    pto.set_flag(pto.Pipe.MTE1, pto.Pipe.M, event_id=1)
    pto.wait_flag(pto.Pipe.MTE1, pto.Pipe.M, event_id=1)
    pto.tile.gemv_acc(acc, l0a, l0b, acc)

    pto.set_flag(pto.Pipe.M, pto.Pipe.FIX, event_id=1)
    pto.wait_flag(pto.Pipe.M, pto.Pipe.FIX, event_id=1)
    pto.tile.store(acc, c_view, offsets=[0, 0, 0, 0, 0], sizes=[1, 1, 1, 1, 96])
    pto.pipe_barrier(pto.Pipe.ALL)


def _make_inputs_gemv():
    np.random.seed(zlib.crc32(b"gemv_f16_1x300x60") & 0xFFFFFFFF)
    a = np.random.uniform(-1.0, 1.0, size=(1, 300)).astype(np.float16)
    b = np.random.uniform(-1.0, 1.0, size=(300, 60)).astype(np.float16)
    a_pad = np.zeros((1, 320), dtype=np.float16)
    b_pad = np.zeros((320, 64), dtype=np.float16)
    a_pad[:, :300] = a
    b_pad[:300, :60] = b
    return [a_pad[:, :256], b_pad[:256, :], a_pad[:, 256:], b_pad[256:, :]]


def _expected_gemv(a0, b0, a1, b1):
    a = np.concatenate([a0, a1], axis=1).astype(np.float64)
    b = np.concatenate([b0, b1], axis=0).astype(np.float64)
    return (a @ b).astype(np.float32)


def _make_inputs_bias():
    np.random.seed(zlib.crc32(b"gemv_bias_f16_1x512x85") & 0xFFFFFFFF)
    a = np.random.uniform(-1.0, 1.0, size=(1, 512)).astype(np.float16)
    b = np.random.uniform(-1.0, 1.0, size=(512, 85)).astype(np.float16)
    bias = np.random.uniform(-1.0, 1.0, size=(85,)).astype(np.float32)
    a0, a1 = a[:, :256], a[:, 256:]
    b_pad = np.zeros((512, 96), dtype=np.float16)
    b_pad[:, :85] = b
    bias_pad = np.zeros((96,), dtype=np.float32)
    bias_pad[:85] = bias
    return [a0, b_pad[:256, :], a1, b_pad[256:, :], bias_pad]


def _expected_bias(a0, b0, a1, b1, bias):
    a = np.concatenate([a0, a1], axis=1).astype(np.float64)
    b = np.concatenate([b0, b1], axis=0).astype(np.float64)
    return (a @ b + bias.astype(np.float64)).astype(np.float32)


CASES = [
    golden_output_case(
        "gemv_f16_1x300x60",
        _kernel_gemv_f16_1x300x60,
        inputs=_make_inputs_gemv,
        expected=_expected_gemv,
        rtol=1e-2,
        atol=1e-2,
    ),
    golden_output_case(
        "gemv_bias_f16_1x512x85",
        _kernel_gemv_bias_f16_1x512x85,
        inputs=_make_inputs_bias,
        expected=_expected_bias,
        rtol=1e-2,
        atol=1e-2,
    ),
]


auto_main(globals())
