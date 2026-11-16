#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Issue #1733: merge mutually exclusive masked assignments into an uninitialized VMI vreg."""

import numpy as np

from common import auto_main, golden_output_case
from ptodsl import pto

LANES = 64
TRANSFER_BYTES = LANES * np.dtype(np.float32).itemsize
B_UB_OFFSET = TRANSFER_BYTES
COND_UB_OFFSET = 2 * TRANSFER_BYTES
OUT_UB_OFFSET = 3 * TRANSFER_BYTES


@pto.jit(
    name="issue_1733_vmi_uninitialized_vreg",
    kernel_kind="vector",
    target="a5",
    backend="vpto",
    mode="explicit",
    insert_sync=False,
)
def issue_1733_vmi_uninitialized_vreg(
    a_gm: pto.ptr(pto.f32, "gm"),
    b_gm: pto.ptr(pto.f32, "gm"),
    cond_gm: pto.ptr(pto.f32, "gm"),
    out_gm: pto.ptr(pto.f32, "gm"),
):
    a_ub = pto.castptr(pto.i64(0), pto.ptr(pto.f32, "ub"))
    b_ub = pto.castptr(pto.i64(B_UB_OFFSET), pto.ptr(pto.f32, "ub"))
    cond_ub = pto.castptr(pto.i64(COND_UB_OFFSET), pto.ptr(pto.f32, "ub"))
    out_ub = pto.castptr(pto.i64(OUT_UB_OFFSET), pto.ptr(pto.f32, "ub"))

    pto.mte_gm_ub(
        a_gm,
        a_ub,
        0,
        TRANSFER_BYTES,
        nburst=(1, TRANSFER_BYTES, TRANSFER_BYTES),
    )
    pto.mte_gm_ub(
        b_gm,
        b_ub,
        0,
        TRANSFER_BYTES,
        nburst=(1, TRANSFER_BYTES, TRANSFER_BYTES),
    )
    pto.mte_gm_ub(
        cond_gm,
        cond_ub,
        0,
        TRANSFER_BYTES,
        nburst=(1, TRANSFER_BYTES, TRANSFER_BYTES),
    )
    pto.set_flag("MTE2", "V", event_id=0)
    pto.wait_flag("MTE2", "V", event_id=0)

    with pto.vecscope():
        valid_mask = pto.vmi.create_mask(LANES, size=LANES)
        a = pto.vmi.vload(a_ub, 0, size=LANES)
        b = pto.vmi.vload(b_ub, 0, size=LANES)
        cond = pto.vmi.vload(cond_ub, 0, size=LANES)

        then_mask = pto.vmi.vcmps(cond, 0.0, valid_mask, pto.CmpMode.GT)
        else_mask = pto.vmi.vcmps(cond, 0.0, valid_mask, pto.CmpMode.LE)

        # The two masks partition all valid lanes. The unspecified false lanes
        # from the first select are overwritten by the second select before the
        # result is stored, matching the branch-local assignments in issue #1733.
        value = pto.vmi.vreg(LANES, pto.f32)
        value = pto.vmi.vsel(then_mask, a, value)
        value = pto.vmi.vsel(else_mask, b, value)
        pto.vmi.vstore(value, out_ub, 0, valid_mask)

    pto.set_flag("V", "MTE3", event_id=0)
    pto.wait_flag("V", "MTE3", event_id=0)
    pto.mte_ub_gm(
        out_ub,
        out_gm,
        TRANSFER_BYTES,
        nburst=(1, TRANSFER_BYTES, TRANSFER_BYTES),
    )
    pto.pipe_barrier(pto.Pipe.ALL)


def make_inputs():
    lanes = np.arange(LANES, dtype=np.float32)
    a = 100.0 + lanes
    b = 1000.0 + lanes
    # Include positive, negative, and zero conditions; GT and LE must cover
    # every valid lane exactly once.
    cond = np.where(lanes % 3 == 0, 1.0, np.where(lanes % 3 == 1, -1.0, 0.0))
    return [a, b, cond.astype(np.float32)]


def make_expected(a, b, cond):
    return np.where(cond > 0.0, a, b)


CASES = [
    golden_output_case(
        "issue_1733_vmi_uninitialized_vreg_vsel_merge",
        issue_1733_vmi_uninitialized_vreg,
        inputs=make_inputs,
        expected=make_expected,
        rtol=0.0,
        atol=0.0,
    ),
]


auto_main(globals())
