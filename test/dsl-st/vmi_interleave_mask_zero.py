#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Masked vintlv/vdintlv zero both results in every inactive lane.

The lowering owns the inactive-lane zeroing now, so the frontend no longer has
to add its own vsel workaround.  BF16 cases compare raw uint16 bits so active
lane payloads, including non-finite patterns, are preserved exactly.
"""

import numpy as np

from common import auto_main
from ptodsl import pto

VL = 128
NBYTES_F32 = VL * 4
NBYTES_BF16 = VL * 2


def _build_kernel(name, op_name, dtype, nbytes, active):
    @pto.jit(
        name=name,
        kernel_kind="vector",
        target="a5",
        backend="vpto",
        mode="explicit",
    )
    def kernel(
        lhs_ptr: pto.ptr(dtype, "gm"),
        rhs_ptr: pto.ptr(dtype, "gm"),
        low_ptr: pto.ptr(dtype, "gm"),
        high_ptr: pto.ptr(dtype, "gm"),
    ):
        lhs_ub = pto.castptr(pto.i64(0), pto.ptr(dtype, "ub"))
        rhs_ub = pto.castptr(pto.i64(nbytes), pto.ptr(dtype, "ub"))
        low_ub = pto.castptr(pto.i64(2 * nbytes), pto.ptr(dtype, "ub"))
        high_ub = pto.castptr(pto.i64(3 * nbytes), pto.ptr(dtype, "ub"))
        pto.mte_gm_ub(lhs_ptr, lhs_ub, 0, nbytes, nburst=(1, 0, 0))
        pto.mte_gm_ub(rhs_ptr, rhs_ub, 0, nbytes, nburst=(1, 0, 0))
        pto.set_flag("MTE2", "V", event_id=0)
        pto.wait_flag("MTE2", "V", event_id=0)
        lhs = pto.vmi.vload(lhs_ub, pto.const(0, dtype=pto.index), size=VL)
        rhs = pto.vmi.vload(rhs_ub, pto.const(0, dtype=pto.index), size=VL)
        mask = pto.vmi.create_mask(active, size=VL)
        if op_name == "vintlv":
            low, high = pto.vmi.vintlv(lhs, rhs, mask, pmode="zero")
        else:
            low, high = pto.vmi.vdintlv(lhs, rhs, mask, pmode="zero")
        pto.vmi.vstore(low, low_ub, pto.const(0, dtype=pto.index))
        pto.vmi.vstore(high, high_ub, pto.const(0, dtype=pto.index))
        pto.set_flag("V", "MTE3", event_id=0)
        pto.wait_flag("V", "MTE3", event_id=0)
        pto.mte_ub_gm(low_ub, low_ptr, nbytes, nburst=(1, 0, 0))
        pto.mte_ub_gm(high_ub, high_ptr, nbytes, nburst=(1, 0, 0))

    return kernel


def _interleave_golden(op_name, lhs, rhs):
    low = np.zeros(VL, dtype=lhs.dtype)
    high = np.zeros(VL, dtype=lhs.dtype)
    half = VL // 2
    if op_name == "vintlv":
        for i in range(half):
            low[2 * i] = lhs[i]
            low[2 * i + 1] = rhs[i]
            high[2 * i] = lhs[half + i]
            high[2 * i + 1] = rhs[half + i]
    else:
        for i in range(half):
            low[i] = lhs[2 * i]
            low[half + i] = rhs[2 * i]
            high[i] = lhs[2 * i + 1]
            high[half + i] = rhs[2 * i + 1]
    return low, high


def _make_case(op_name, dtype, active, make_inputs):
    def make_case():
        lhs, rhs = make_inputs()
        low = np.zeros_like(lhs)
        high = np.zeros_like(rhs)
        expected_low = np.zeros(VL, dtype=lhs.dtype)
        expected_high = np.zeros(VL, dtype=rhs.dtype)
        golden_low, golden_high = _interleave_golden(
            op_name, lhs.reshape(-1), rhs.reshape(-1)
        )
        expected_low[:active] = golden_low[:active]
        expected_high[:active] = golden_high[:active]
        return [lhs, rhs, low, high], (expected_low, expected_high)

    def check(device_inputs, expected):
        actual_low = device_inputs[2].cpu().numpy().reshape(-1)
        actual_high = device_inputs[3].cpu().numpy().reshape(-1)
        assert np.array_equal(actual_low, expected[0])
        assert np.array_equal(actual_high, expected[1])

    return make_case, check


def _make_f32_inputs():
    lhs = (np.arange(VL, dtype=np.float32) + 0.25).reshape(1, VL)
    rhs = (np.arange(VL, dtype=np.float32) + 100.25).reshape(1, VL)
    return lhs, rhs


def _make_bf16_bits():
    rng = np.random.default_rng(1717)
    lhs = rng.integers(0, 1 << 16, size=(1, VL), dtype=np.uint16)
    rhs = rng.integers(0, 1 << 16, size=(1, VL), dtype=np.uint16)
    return lhs, rhs


CASES = []
for op_name in ("vintlv", "vdintlv"):
    for active in (0, VL // 2, VL):
        make_case, check = _make_case(op_name, pto.f32, active, _make_f32_inputs)
        name = f"{op_name}_f32_vl{VL}_a{active}"
        CASES.append(
            {
                "name": name,
                "kernel": _build_kernel(name, op_name, pto.f32, NBYTES_F32, active),
                "grid": 1,
                "make_case": make_case,
                "check": check,
            }
        )

for op_name in ("vintlv", "vdintlv"):
    make_case, check = _make_case(
        op_name, pto.bf16, VL // 2, _make_bf16_bits
    )
    name = f"{op_name}_bf16_bits_vl{VL}_a{VL // 2}"
    CASES.append(
        {
            "name": name,
            "kernel": _build_kernel(
                name, op_name, pto.bf16, NBYTES_BF16, VL // 2
            ),
            "grid": 1,
            "make_case": make_case,
            "check": check,
        }
    )

auto_main(globals())
