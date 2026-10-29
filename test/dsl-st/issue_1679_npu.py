#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Issue #1679: three-IV, two-IV control, and scope-external offset ST.

Preserve the issue's loop nests, explicit casts and UB offsets. Make hazards
observable by adding one at each update and two at drain (A/A-prime).
Only initialized regions are copied to GM after V-to-MTE3 synchronization.
Numeric success alone does not prove barrier coverage; inspect VPTO IR as well.
"""

import numpy as np

from common import auto_main, golden_output_case
from ptodsl import pto


@pto.jit(name="case_a_three_iv", kernel_kind="vector", target="a5", backend="vpto", mode="explicit")
def case_a(out: pto.ptr(pto.f32, "gm")):
    pto.init_core()
    with pto.vecscope():
        mask = pto.vmi.create_mask(128, size=128)
        acc = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
        for _i in range(0, 8):
            i = pto.cast(_i, pto.si32)
            for _q in range(0, 2):
                q = pto.cast(_q, pto.si32)
                pto.vmi.vstore(pto.vmi.vbrc(pto.f32(0.0), size=128), acc, (i * 256) + (q * 128), mask)
        for _k in range(0, 4):
            k = pto.cast(_k, pto.si32)
            for _i2 in range(0, 8):
                i2 = pto.cast(_i2, pto.si32)
                for _q2 in range(0, 2):
                    q2 = pto.cast(_q2, pto.si32)
                    v = pto.vmi.vload(acc, (i2 * 256) + (q2 * 128), size=128)
                    v = pto.vmi.vadd(v, pto.vmi.vbrc(pto.f32(1.0), size=128), mask)
                    pto.vmi.vstore(v, acc, (i2 * 256) + (q2 * 128), mask)
        for _i3 in range(0, 8):
            i3 = pto.cast(_i3, pto.si32)
            for _q3 in range(0, 2):
                q3 = pto.cast(_q3, pto.si32)
                v3 = pto.vmi.vload(acc, (i3 * 256) + (q3 * 128), size=128)
                v3 = pto.vmi.vadd(v3, pto.vmi.vbrc(pto.f32(2.0), size=128), mask)
                pto.vmi.vstore(v3, acc, (i3 * 256) + (q3 * 128), mask)


    pto.set_flag("V", "MTE3", event_id=0)
    pto.wait_flag("V", "MTE3", event_id=0)
    result = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    pto.mte_ub_gm(result, out, 8192, nburst=(1, 8192, 8192), l2_cache="naci")


@pto.jit(name="case_a_prime_two_iv", kernel_kind="vector", target="a5", backend="vpto", mode="explicit")
def case_a_prime(out: pto.ptr(pto.f32, "gm")):
    pto.init_core()
    with pto.vecscope():
        mask = pto.vmi.create_mask(128, size=128)
        acc = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
        for _i in range(0, 8):
            i = pto.cast(_i, pto.si32)
            pto.vmi.vstore(pto.vmi.vbrc(pto.f32(0.0), size=128), acc, i * 256, mask)
        for _k in range(0, 4):
            k = pto.cast(_k, pto.si32)
            for _i2 in range(0, 8):
                i2 = pto.cast(_i2, pto.si32)
                v = pto.vmi.vload(acc, i2 * 256, size=128)
                v = pto.vmi.vadd(v, pto.vmi.vbrc(pto.f32(1.0), size=128), mask)
                pto.vmi.vstore(v, acc, i2 * 256, mask)
        for _i3 in range(0, 8):
            i3 = pto.cast(_i3, pto.si32)
            v3 = pto.vmi.vload(acc, i3 * 256, size=128)
            v3 = pto.vmi.vadd(v3, pto.vmi.vbrc(pto.f32(2.0), size=128), mask)
            pto.vmi.vstore(v3, acc, i3 * 256, mask)


    pto.set_flag("V", "MTE3", event_id=0)
    pto.wait_flag("V", "MTE3", event_id=0)
    result = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    # Copy only the 128 initialized lanes of each 256-element row.
    for row in range(0, 8):
        pto.mte_ub_gm(pto.addptr(result, row * 256), pto.addptr(out, row * 128),
                      512, nburst=(1, 512, 512), l2_cache="naci")


@pto.jit(name="case_b_outer_var", kernel_kind="vector", target="a5", backend="vpto", mode="explicit")
def case_b(out: pto.ptr(pto.f32, "gm")):
    pto.init_core()
    for _b in range(0, 2):
        b = pto.cast(_b, pto.si32)
        with pto.vecscope():
            mask = pto.vmi.create_mask(128, size=128)
            acc = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
            for _i in range(0, 8):
                i = pto.cast(_i, pto.si32)
                pto.vmi.vstore(pto.vmi.vbrc(pto.f32(0.0), size=128), acc, (b * 2048) + (i * 128), mask)
            for _i2 in range(0, 8):
                i2 = pto.cast(_i2, pto.si32)
                v = pto.vmi.vload(acc, (b * 2048) + (i2 * 128), size=128)
                v = pto.vmi.vadd(v, pto.vmi.vbrc(pto.f32(1.0), size=128), mask)
                pto.vmi.vstore(v, acc, (b * 2048) + (i2 * 128), mask)

    pto.set_flag("V", "MTE3", event_id=0)
    pto.wait_flag("V", "MTE3", event_id=0)
    result = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    # The two batches occupy disjoint UB ranges; omit their untouched gaps.
    for batch in range(0, 2):
        pto.mte_ub_gm(pto.addptr(result, batch * 2048), pto.addptr(out, batch * 1024),
                      4096, nburst=(1, 4096, 4096), l2_cache="naci")


CASES = [
    golden_output_case("issue_1679_a_three_iv", case_a,
                       inputs=[], expected=np.full(2048, 6, dtype=np.float32), rtol=0, atol=0),
    golden_output_case("issue_1679_a_prime_two_iv", case_a_prime,
                       inputs=[], expected=np.full(1024, 6, dtype=np.float32), rtol=0, atol=0),
    golden_output_case("issue_1679_b_outer_var", case_b,
                       inputs=[], expected=np.full(2048, 1, dtype=np.float32), rtol=0, atol=0),
]


auto_main(globals())
