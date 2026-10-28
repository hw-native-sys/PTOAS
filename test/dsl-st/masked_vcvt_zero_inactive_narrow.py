#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Masked pto.vcvt in the NARROWING direction: inactive lanes come back as zero.

masked_vcvt_zero_inactive.py and its l64 sibling cover the widening half of
issue #1671 (f16 -> f32, where the predicate rides a single conversion and a
lane-strided carrier has to spread it).  This file adds the conversion half the
issue also asks for, pto.vmi.truncf / f32 -> f16 narrowing, because narrowing
carries the predicate down different lowering paths than widening does:

  * lib/PTO/Transforms/VMI/VMILowerUnifiedToLegacy.cpp: a direct f32 -> f16
    narrow forwards the predicate to the pto.vmi.truncf it emits (the first
    case below), while a STAGED narrow can only put the predicate on the
    widening leg and leaves the second leg deliberately unpredicated, because
    the widening already zeroed the inactive f32 lanes (the second case below,
    f16 -> f8E4M3, the only narrowing A5 can reach through f32).  A test that
    only ever sets the predicate on the leg the hardware converts would not
    notice either placement going wrong; here the inactive lanes turn nonzero
    under both failure modes.
  * lib/PTO/Transforms/VMIToVPTO/VMIToVPTOPatternInternals10.cpp: when a narrow
    does fan out into several packed conversion parts (P0..P3) they are merged
    with pto.vor, a pure lane combine, so an inactive lane can only come back
    zero if every part's conversion already zeroed it - the same invariant the
    integer narrowing merge in VMIToVPTOPatternInternals6.cpp rests on.  Those
    merges are only reached from IR that pins a lane-strided / deinterleaved
    carrier (see test/lit/vmi_new/vmi_to_vpto_truncf_f16_to_f8e4m3.pto); the
    dense load -> narrow -> store graphs the DSL emits here lower one
    conversion per 64-lane chunk (one P part per chunk, no pto.vor in the
    emitted VPTO), so this file pins the zero-inactive contract one level below
    that merge instead of the merge itself.

Both cases are zero-inactive checks with an unmasked store: the kernel loads
128 lanes, builds an 80-of-128 predicate with pto.vmi.create_mask, converts the
whole vector under that predicate, then stores and copies out every lane, so an
inactive lane only matches the golden if the conversion really cleared it.  The
inactive source lanes carry values that survive into a NONZERO narrow result
(f16 30000..60000 for the f16 destination, f16 1000.0 -> 0x7e for the saturating
fp8 destination), so a merge/pass-through lowering or a dropped predicate on
either staged leg shows up as a numeric mismatch instead of hiding behind a
tolerance: both cases compare with rtol=0, atol=0, and the active source lanes
are exactly representable in the narrow type, so the only error the golden can
see is a predicate error.
"""

import ml_dtypes
import numpy as np

from common import auto_main, golden_output_case
from ptodsl import pto

LANES = 128
# 80 of 128 is deliberate rather than LANES // 2: the first 64-lane carrier half
# is then fully active and the second half is only 16-of-64 active, so a
# lowering that collapses the predicate to per-chunk all-or-nothing (all-true
# when any lane in the chunk is set) converts 48 lanes it should have zeroed.
ACTIVE = 80
SRC_UB = 0
DST_UB = 1024

# Active f8E4M3 payloads: the positive and negative grid points 1.0 .. 1.625.
# Every one of them is exact in f16 as well, so the staged f16 -> f32 -> f8E4M3
# chain and the direct f32 -> f16 chain are both bit-exact on the active lanes.
F8_ACTIVE_BYTES = np.resize(
    np.array([0x38, 0x39, 0x3A, 0x3B, 0x3C, 0x3D,
              0xB8, 0xB9, 0xBA, 0xBB, 0xBC, 0xBD], dtype=np.uint8),
    ACTIVE,
)


@pto.jit(
    name="masked_vcvt_narrow_f32_to_f16_kernel",
    kernel_kind="vector",
    target="a5",
    backend="vpto",
    mode="explicit",
)
def masked_vcvt_narrow_f32_to_f16_kernel(
    src_gm: pto.ptr(pto.f32, 'gm'),
    out_gm: pto.ptr(pto.f16, 'gm'),
):
    """Direct narrowing: the predicate lands on the pto.vcvt itself."""
    src_ub = pto.castptr(pto.const(SRC_UB, dtype=pto.i64), pto.ptr(pto.f32, 'ub'))
    dst_ub = pto.castptr(pto.const(DST_UB, dtype=pto.i64), pto.ptr(pto.f16, 'ub'))
    src_bytes = LANES * 4
    out_bytes = LANES * 2
    pto.mte_gm_ub(
        src_gm, src_ub, 0, src_bytes, nburst=(1, src_bytes, src_bytes),
    )
    pto.set_flag('MTE2', 'V', event_id=0)
    pto.wait_flag('MTE2', 'V', event_id=0)
    with pto.vecscope():
        mask = pto.vmi.create_mask(ACTIVE, size=LANES)
        src = pto.vmi.vload(src_ub, 0, size=LANES)
        conv = pto.vmi.vcvt(src, to_dtype=pto.f16, mask=mask)
        # Deliberately unmasked: every lane of the narrowed vector must leave
        # with its post-conversion value, so an inactive lane exposes a missing
        # zero instead of hiding behind the store predicate.
        pto.vmi.vstore(conv, dst_ub, 0)
    pto.set_flag('V', 'MTE3', event_id=0)
    pto.wait_flag('V', 'MTE3', event_id=0)
    pto.mte_ub_gm(
        dst_ub, out_gm, out_bytes, nburst=(1, out_bytes, out_bytes), l2_cache='naci',
    )


@pto.jit(
    name="masked_vcvt_narrow_f16_to_f8_kernel",
    kernel_kind="vector",
    target="a5",
    backend="vpto",
    mode="explicit",
)
def masked_vcvt_narrow_f16_to_f8_kernel(
    src_gm: pto.ptr(pto.f16, 'gm'),
    out_gm: pto.ptr(pto.ui8, 'gm'),
):
    """Staged narrowing: the predicate lands on the widening leg instead."""
    src_ub = pto.castptr(pto.const(SRC_UB, dtype=pto.i64), pto.ptr(pto.f16, 'ub'))
    dst_ub_u8 = pto.castptr(pto.const(DST_UB, dtype=pto.i64), pto.ptr(pto.ui8, 'ub'))
    dst_ub_f8 = pto.castptr(pto.const(DST_UB, dtype=pto.i64), pto.ptr(pto.f8e4m3, 'ub'))
    src_bytes = LANES * 2
    out_bytes = LANES
    pto.mte_gm_ub(
        src_gm, src_ub, 0, src_bytes, nburst=(1, src_bytes, src_bytes),
    )
    pto.set_flag('MTE2', 'V', event_id=0)
    pto.wait_flag('MTE2', 'V', event_id=0)
    with pto.vecscope():
        mask = pto.vmi.create_mask(ACTIVE, size=LANES)
        src = pto.vmi.vload(src_ub, 0, size=LANES)
        conv = pto.vmi.vcvt(
            src,
            to_dtype=pto.f8e4m3,
            mask=mask,
            rounding=pto.VcvtRoundMode.R,
            saturate=pto.VcvtSatMode.SAT,
        )
        # Deliberately unmasked, for the same reason as the f16 case: every
        # conversion part has to come back zeroed on its own, without relying on
        # a store predicate or on a later merge to hide a leftover source value.
        pto.vmi.vstore(conv, dst_ub_f8, 0)
    pto.set_flag('V', 'MTE3', event_id=0)
    pto.wait_flag('V', 'MTE3', event_id=0)
    pto.mte_ub_gm(
        dst_ub_u8, out_gm, out_bytes, nburst=(1, out_bytes, out_bytes), l2_cache='naci',
    )


def make_f32_inputs():
    # Active lanes are negative f16-exact values; inactive lanes are large
    # f16-exact magnitudes, so the narrowing keeps them nonzero and a merge or
    # pass-through result cannot land accidentally on the golden zero.
    active = np.linspace(-17.0, -1.5, ACTIVE, dtype=np.float32).astype(np.float16)
    inactive = np.linspace(30000.0, 60000.0, LANES - ACTIVE, dtype=np.float32).astype(np.float16)
    return [np.concatenate([active, inactive]).astype(np.float32)]


def make_f16_inputs():
    # The f16 source carries the f8E4M3 grid the golden expects on the active
    # lanes; the inactive lanes are 1000.0, which saturates to the nonzero fp8
    # code 0x7e (448.0) instead of zero.
    active = np.frombuffer(F8_ACTIVE_BYTES.tobytes(), dtype=ml_dtypes.float8_e4m3fn).astype(np.float16)
    inactive = np.full(LANES - ACTIVE, 1000.0, dtype=np.float16)
    return [np.concatenate([active, inactive])]


def make_f16_expected(src):
    # The source values are f16-exact and the predicate clears every lane past
    # ACTIVE, so the golden is exactly zero there.
    expected = np.zeros(LANES, dtype=np.float16)
    expected[:ACTIVE] = src[:ACTIVE].astype(np.float16)
    return expected


def make_f8_expected(src):
    # The active source values are exactly representable in f8E4M3, so the
    # active bytes are the ones the source was built from.
    expected = np.zeros(LANES, dtype=np.uint8)
    expected[:ACTIVE] = F8_ACTIVE_BYTES
    return expected


CASES = [
    golden_output_case(
        "masked_vcvt_narrow_f32_to_f16_l128_a80",
        masked_vcvt_narrow_f32_to_f16_kernel,
        inputs=make_f32_inputs,
        expected=make_f16_expected,
        rtol=0.0,
        atol=0.0,
    ),
    golden_output_case(
        "masked_vcvt_narrow_f16_to_f8_l128_a80",
        masked_vcvt_narrow_f16_to_f8_kernel,
        inputs=make_f16_inputs,
        expected=make_f8_expected,
        rtol=0.0,
        atol=0.0,
    ),
]


auto_main(globals())
