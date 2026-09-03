#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""PTODSL migration of the legacy A5 ``trandom`` case."""

from pathlib import Path
import sys
import zlib

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import auto_main, golden_output_case
from ptodsl import pto


ROWS, COLS = 4, 256
_M0 = np.uint64(0xD2511F53)
_M1 = np.uint64(0xCD9E8D57)
_K0 = np.uint32(0x9E3779B9)
_K1 = np.uint32(0xBB67AE85)


def _make_kernel(rounds):
    @pto.jit(name=f"trandom_int32_4x256_round{rounds}", target="a5")
    def _kernel(
        key_ptr: pto.ptr(pto.ui32, "gm"),
        counter_ptr: pto.ptr(pto.ui32, "gm"),
        output_ptr: pto.ptr(pto.ui32, "gm"),
    ):
        # The public trandom contract uses signless i32 scalar words. The host
        # ABI stores the same bits as ui32, so reinterpret the GM pointer once.
        key_ptr_i32 = pto.castptr(key_ptr, pto.ptr(pto.i32, "gm"))
        counter_ptr_i32 = pto.castptr(counter_ptr, pto.ptr(pto.i32, "gm"))
        key0 = pto.load_scalar(key_ptr_i32, 0)
        key1 = pto.load_scalar(key_ptr_i32, 1)
        counter0 = pto.load_scalar(counter_ptr_i32, 0)
        counter1 = pto.load_scalar(counter_ptr_i32, 1)
        counter2 = pto.load_scalar(counter_ptr_i32, 2)
        counter3 = pto.load_scalar(counter_ptr_i32, 3)
        output_view = pto.make_tensor_view(
            output_ptr,
            shape=[1, 1, 1, ROWS, COLS],
            strides=[ROWS * COLS, ROWS * COLS, ROWS * COLS, COLS, 1],
        )
        output_tile = pto.alloc_tile(
            shape=[ROWS, COLS], dtype=pto.ui32, valid_shape=[ROWS, COLS]
        )
        pto.tile.random(
            key0, key1, counter0, counter1, counter2, counter3, output_tile, rounds=rounds
        )
        pto.tile.store(
            output_tile,
            output_view,
            offsets=[0, 0, 0, 0, 0],
            sizes=[1, 1, 1, ROWS, COLS],
        )

    return _kernel


def _philox_round(ctr0, ctr1, ctr2, ctr3, key0, key1):
    product0 = ctr0.astype(np.uint64) * _M0
    product1 = ctr2.astype(np.uint64) * _M1
    low0 = product0.astype(np.uint32)
    high0 = (product0 >> 32).astype(np.uint32)
    low1 = product1.astype(np.uint32)
    high1 = (product1 >> 32).astype(np.uint32)
    return ((high1 ^ ctr1) ^ key0, low1, (high0 ^ ctr3) ^ key1, low0,
            (key0 + _K0).astype(np.uint32), (key1 + _K1).astype(np.uint32))


def _interleave(ctr0, ctr1, ctr2, ctr3):
    half = len(ctr0) // 2
    low0 = np.empty_like(ctr0)
    high0 = np.empty_like(ctr0)
    low1 = np.empty_like(ctr0)
    high1 = np.empty_like(ctr0)
    for i in range(half):
        low0[2 * i:2 * i + 2] = (ctr0[i], ctr2[i])
        high0[2 * i:2 * i + 2] = (ctr0[i + half], ctr2[i + half])
        low1[2 * i:2 * i + 2] = (ctr1[i], ctr3[i])
        high1[2 * i:2 * i + 2] = (ctr1[i + half], ctr3[i + half])
    result0 = np.empty_like(ctr0)
    result1 = np.empty_like(ctr0)
    result2 = np.empty_like(ctr0)
    result3 = np.empty_like(ctr0)
    for i in range(half):
        result0[2 * i:2 * i + 2] = (low0[i], low1[i])
        result1[2 * i:2 * i + 2] = (low0[i + half], low1[i + half])
        result2[2 * i:2 * i + 2] = (high0[i], high1[i])
        result3[2 * i:2 * i + 2] = (high0[i + half], high1[i + half])
    return result0, result1, result2, result3


def _add_with_128bits(ctr0, ctr1, ctr2, ctr3, value):
    """Mirror the legacy generator's lane-wise 128-bit counter addition."""
    ctr0_old = ctr0.copy()
    ctr0 = (ctr0.astype(np.uint64) + value.astype(np.uint64)).astype(np.uint32)
    carry = (ctr0.astype(np.uint64) < ctr0_old.astype(np.uint64)).astype(np.uint32)

    ctr1_old = ctr1.copy()
    ctr1 = (ctr1.astype(np.uint64) + carry.astype(np.uint64)).astype(np.uint32)
    carry = (ctr1.astype(np.uint64) < ctr1_old.astype(np.uint64)).astype(np.uint32)

    ctr2_old = ctr2.copy()
    ctr2 = (ctr2.astype(np.uint64) + carry.astype(np.uint64)).astype(np.uint32)
    carry = (ctr2.astype(np.uint64) < ctr2_old.astype(np.uint64)).astype(np.uint32)

    ctr3 = (ctr3.astype(np.uint64) + carry.astype(np.uint64)).astype(np.uint32)
    return ctr0, ctr1, ctr2, ctr3


def _golden(key, counter, rounds):
    key0, key1 = key
    ctr0 = np.full(64, counter[0], dtype=np.uint32)
    ctr1 = np.full(64, counter[1], dtype=np.uint32)
    ctr2 = np.full(64, counter[2], dtype=np.uint32)
    ctr3 = np.full(64, counter[3], dtype=np.uint32)
    ctr0, ctr1, ctr2, ctr3 = _add_with_128bits(
        ctr0, ctr1, ctr2, ctr3, np.arange(64, dtype=np.uint32)
    )
    out = np.empty((ROWS, COLS), dtype=np.uint32)
    for row in range(ROWS):
        tmp0, tmp1, tmp2, tmp3 = ctr0.copy(), ctr1.copy(), ctr2.copy(), ctr3.copy()
        tmp_key0 = np.full(64, key0, dtype=np.uint32)
        tmp_key1 = np.full(64, key1, dtype=np.uint32)
        for _ in range(rounds):
            tmp0, tmp1, tmp2, tmp3, tmp_key0, tmp_key1 = _philox_round(
                tmp0, tmp1, tmp2, tmp3, tmp_key0, tmp_key1
            )
        out[row] = np.concatenate(_interleave(tmp0, tmp1, tmp2, tmp3))
        ctr0, ctr1, ctr2, ctr3 = _add_with_128bits(
            ctr0, ctr1, ctr2, ctr3, np.full(64, 64, dtype=np.uint32)
        )
    return out


def _inputs():
    np.random.seed(zlib.crc32(b"int32_4x256") & 0xFFFFFFFF)
    key = np.random.randint(-(2**31), 2**31 - 1, size=2, dtype=np.int32).view(np.uint32)
    counter = np.random.randint(-(2**31), 2**31 - 1, size=4, dtype=np.int32).view(np.uint32)
    return [key, counter]


def _make_expected(rounds):
    def _expected(key, counter):
        return _golden(key, counter, rounds)

    return _expected


CASES = [
    golden_output_case(
        f"trandom_int32_4x256_round{rounds}",
        _make_kernel(rounds),
        inputs=_inputs,
        expected=_make_expected(rounds),
        rtol=0.0,
        atol=0.0,
    )
    for rounds in (7, 10)
]


auto_main(globals())
