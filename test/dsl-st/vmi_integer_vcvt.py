#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Runtime coverage for native integer VMI vcvt pairs on A5."""

import numpy as np

from common import auto_main, golden_output_case
from ptodsl import pto

LANES = 128
ACTIVE = 96
SRC_UB_BYTES = LANES * 4
DST_UB = 1024


@pto.jit(
    name="vmi_vcvt_si16_to_ui32_kernel",
    kernel_kind="vector",
    target="a5",
    backend="vpto",
    mode="explicit",
)
def vmi_vcvt_si16_to_ui32_kernel(
    src_gm: pto.ptr(pto.si16, "gm"),
    out_gm: pto.ptr(pto.ui32, "gm"),
):
    src_ub = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.si16, "ub"))
    dst_ub = pto.castptr(pto.const(DST_UB, dtype=pto.i64), pto.ptr(pto.ui32, "ub"))
    pto.mte_gm_ub(
        src_gm,
        src_ub,
        0,
        LANES * 2,
        nburst=(1, LANES * 2, LANES * 2),
    )
    pto.set_flag("MTE2", "V", event_id=0)
    pto.wait_flag("MTE2", "V", event_id=0)
    with pto.vecscope():
        mask = pto.vmi.create_mask(ACTIVE, size=LANES)
        src = pto.vmi.vload(src_ub, 0, size=LANES)
        converted = pto.vmi.vcvt(src, pto.ui32, mask)
        pto.vmi.vstore(converted, dst_ub, 0)
    pto.set_flag("V", "MTE3", event_id=0)
    pto.wait_flag("V", "MTE3", event_id=0)
    pto.mte_ub_gm(
        dst_ub,
        out_gm,
        LANES * 4,
        nburst=(1, LANES * 4, LANES * 4),
        l2_cache="naci",
    )


@pto.jit(
    name="vmi_vcvt_ui32_to_si16_sat_kernel",
    kernel_kind="vector",
    target="a5",
    backend="vpto",
    mode="explicit",
)
def vmi_vcvt_ui32_to_si16_sat_kernel(
    src_gm: pto.ptr(pto.ui32, "gm"),
    out_gm: pto.ptr(pto.si16, "gm"),
):
    src_ub = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.ui32, "ub"))
    dst_ub = pto.castptr(pto.const(DST_UB, dtype=pto.i64), pto.ptr(pto.si16, "ub"))
    pto.mte_gm_ub(
        src_gm,
        src_ub,
        0,
        SRC_UB_BYTES,
        nburst=(1, SRC_UB_BYTES, SRC_UB_BYTES),
    )
    pto.set_flag("MTE2", "V", event_id=0)
    pto.wait_flag("MTE2", "V", event_id=0)
    with pto.vecscope():
        mask = pto.vmi.create_mask(ACTIVE, size=LANES)
        src = pto.vmi.vload(src_ub, 0, size=LANES)
        converted = pto.vmi.vcvt(src, pto.si16, mask, saturate="SAT")
        pto.vmi.vstore(converted, dst_ub, 0)
    pto.set_flag("V", "MTE3", event_id=0)
    pto.wait_flag("V", "MTE3", event_id=0)
    pto.mte_ub_gm(
        dst_ub,
        out_gm,
        LANES * 2,
        nburst=(1, LANES * 2, LANES * 2),
        l2_cache="naci",
    )


def make_si16_inputs():
    probe = np.array(
        [-32768, -16385, -1, 0, 1, 7, 1024, 32766, 32767],
        dtype=np.int16,
    )
    values = np.resize(probe, LANES)
    values[ACTIVE:] = 12345
    return [values]


def make_ui32_inputs():
    probe = np.array(
        [0, 1, 32766, 32767, 32768, 65535, 2**31, 2**32 - 1],
        dtype=np.uint32,
    )
    values = np.resize(probe, LANES)
    values[ACTIVE:] = np.uint32(42)
    return [values]


def expect_si16_to_ui32(src):
    expected = np.zeros(LANES, dtype=np.uint32)
    expected[:ACTIVE] = np.maximum(src[:ACTIVE].astype(np.int32), 0).astype(np.uint32)
    return expected


def expect_ui32_to_si16_sat(src):
    expected = np.zeros(LANES, dtype=np.int16)
    expected[:ACTIVE] = np.minimum(src[:ACTIVE].astype(np.uint64), 32767).astype(np.int16)
    return expected


CASES = [
    golden_output_case(
        "vmi_vcvt_si16_to_ui32_negative_clamp_and_mask",
        vmi_vcvt_si16_to_ui32_kernel,
        inputs=make_si16_inputs,
        expected=expect_si16_to_ui32,
        rtol=0.0,
        atol=0.0,
    ),
    golden_output_case(
        "vmi_vcvt_ui32_to_si16_saturation_and_mask",
        vmi_vcvt_ui32_to_si16_sat_kernel,
        inputs=make_ui32_inputs,
        expected=expect_ui32_to_si16_sat,
        rtol=0.0,
        atol=0.0,
    ),
]


auto_main(globals())
