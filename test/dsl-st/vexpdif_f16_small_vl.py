#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See the License in the root of the software repository for the full text of the License.

"""Issue #1626: f16 vexpdif below 128 lanes through the PTODSL VMI API.

Below one full f16 physical part the EVEN/ODD interleave relation of the
dedicated f16 lowering cannot materialize through the layout system, so the
lowering keeps the source packed and contiguous, computes both halves on it,
and merges them back with vintlv. Coverage:

- full, half-active, and zero-active prefix masks with pmode="zero":
  inactive lanes must read back as exactly zero;
- an output canary (pattern of vmi_short_load_alignment.py): the output is
  preloaded into UB and a fixed 128-byte region is flushed back, so a write
  beyond the vl active lanes overwrites the sentinel and fails the compare;
- VL=128 on both lowering paths (native interleave, and vcvt-then-f32) plus
  an A/B case comparing the two hardware entries on one special-value input
  vector (normal values, extremes, +/-0, +/-Inf, NaN).
"""

import numpy as np

from common import auto_main, golden_output_case
from ptodsl import pto

CANARY_LANES = 32
CANARY_BYTES = 128
SENTINEL = 241.0


def _vexpdif_kernel(name, vl, active, widen_to_f32=False):
    @pto.jit(name=name, kernel_kind="vector", target="a5", backend="vpto", mode="explicit")
    def kernel(
        x_ptr: pto.ptr(pto.f16, "gm"),
        max_ptr: pto.ptr(pto.f16, "gm"),
        out_ptr: pto.ptr(pto.f32, "gm"),
    ):
        in_nbytes = vl * 2
        out_nbytes = vl * 4
        x_ub = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.f16, "ub"))
        max_ub = pto.castptr(pto.const(512, dtype=pto.i64), pto.ptr(pto.f16, "ub"))
        out_ub = pto.castptr(pto.const(1024, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
        pto.mte_gm_ub(x_ptr, x_ub, 0, in_nbytes, nburst=(1, 0, 0))
        pto.mte_gm_ub(max_ptr, max_ub, 0, in_nbytes, nburst=(1, 0, 0))
        pto.set_flag("MTE2", "V", event_id=0)
        pto.wait_flag("MTE2", "V", event_id=0)
        x = pto.vmi.vload(x_ub, pto.const(0, dtype=pto.index), size=vl)
        max_value = pto.vmi.vload(max_ub, pto.const(0, dtype=pto.index), size=vl)
        mask = pto.vmi.create_mask(active, size=vl)
        if widen_to_f32:
            x = pto.vmi.vcvt(x, to_dtype=pto.f32)
            max_value = pto.vmi.vcvt(max_value, to_dtype=pto.f32)
        result = pto.vmi.vexpdif(x, max_value, mask, pmode="zero")
        pto.vmi.vstore(result, out_ub, pto.const(0, dtype=pto.index))
        pto.set_flag("V", "MTE3", event_id=0)
        pto.wait_flag("V", "MTE3", event_id=0)
        pto.mte_ub_gm(out_ub, out_ptr, out_nbytes, nburst=(1, 0, 0))
        pto.pipe_barrier(pto.Pipe.ALL)

    return kernel


def _vexpdif_ab_kernel(name, vl):
    @pto.jit(name=name, kernel_kind="vector", target="a5", backend="vpto", mode="explicit")
    def kernel(
        x_ptr: pto.ptr(pto.f16, "gm"),
        max_ptr: pto.ptr(pto.f16, "gm"),
        out_a_ptr: pto.ptr(pto.f32, "gm"),
        out_b_ptr: pto.ptr(pto.f32, "gm"),
    ):
        in_nbytes = vl * 2
        out_nbytes = vl * 4
        x_ub = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.f16, "ub"))
        max_ub = pto.castptr(pto.const(512, dtype=pto.i64), pto.ptr(pto.f16, "ub"))
        out_a_ub = pto.castptr(pto.const(1024, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
        out_b_ub = pto.castptr(pto.const(2048, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
        pto.mte_gm_ub(x_ptr, x_ub, 0, in_nbytes, nburst=(1, 0, 0))
        pto.mte_gm_ub(max_ptr, max_ub, 0, in_nbytes, nburst=(1, 0, 0))
        pto.set_flag("MTE2", "V", event_id=0)
        pto.wait_flag("MTE2", "V", event_id=0)
        x = pto.vmi.vload(x_ub, pto.const(0, dtype=pto.index), size=vl)
        max_value = pto.vmi.vload(max_ub, pto.const(0, dtype=pto.index), size=vl)
        mask = pto.vmi.create_mask(vl, size=vl)
        # Path A: native f16 form (EVEN/ODD interleave entry).
        result_a = pto.vmi.vexpdif(x, max_value, mask, pmode="zero")
        pto.vmi.vstore(result_a, out_a_ub, pto.const(0, dtype=pto.index))
        # Path B: widen to f32 first, then the f32 entry.
        x32 = pto.vmi.vcvt(x, to_dtype=pto.f32)
        max32 = pto.vmi.vcvt(max_value, to_dtype=pto.f32)
        result_b = pto.vmi.vexpdif(x32, max32, mask, pmode="zero")
        pto.vmi.vstore(result_b, out_b_ub, pto.const(0, dtype=pto.index))
        pto.set_flag("V", "MTE3", event_id=0)
        pto.wait_flag("V", "MTE3", event_id=0)
        pto.mte_ub_gm(out_a_ub, out_a_ptr, out_nbytes, nburst=(1, 0, 0))
        pto.mte_ub_gm(out_b_ub, out_b_ptr, out_nbytes, nburst=(1, 0, 0))
        pto.pipe_barrier(pto.Pipe.ALL)

    return kernel


def _inputs(vl):
    rng = np.random.default_rng(1626)
    x = rng.uniform(-8.0, 8.0, size=vl).astype(np.float16)
    mx = rng.uniform(-4.0, 4.0, size=vl).astype(np.float16)
    return [x, mx]


def _exp_golden(x, mx):
    # Documented per-lane semantics: expf(x - max) computed in f32.
    return np.exp(x.astype(np.float32) - mx.astype(np.float32))


def _special_inputs(vl):
    half = vl // 2
    x = np.concatenate([
        np.linspace(-8.0, 8.0, half, dtype=np.float32),
        np.array([65504.0, -65504.0, 6e-8, -6e-8, 0.0, -0.0,
                  np.inf, -np.inf, np.nan, -65504.0], dtype=np.float32),
        np.zeros(vl - half - 10, dtype=np.float32),
    ])[:vl].astype(np.float16)
    mx = np.zeros(vl, dtype=np.float16)
    mx[0] = -0.0
    mx[1] = np.float16(2.0)
    mx[-1] = np.inf
    return [x, mx]


def _value_case(name, vl, active, widen_to_f32=False):
    kernel = _vexpdif_kernel(f"{name}_kernel", vl, active, widen_to_f32)

    def make_inputs(vl=vl):
        return _inputs(vl)

    def golden(x, mx, vl=vl, active=active):
        values = _exp_golden(x, mx)
        # pmode="zero": inactive lanes read back as exactly zero.
        values[active:] = 0.0
        return values

    return golden_output_case(
        name,
        kernel,
        inputs=make_inputs,
        expected=golden,
        rtol=1e-3,
        atol=1e-6,
    )


def _canary_case(vl):
    kernel = _vexpdif_kernel(f"vexpdif_f16_vl{vl}_canary_kernel", vl, vl)
    x, mx = _inputs(vl)
    output = np.full(CANARY_LANES, SENTINEL, dtype=np.float32)
    expected = output.copy()
    expected[:vl] = _exp_golden(x, mx)

    def make_case(x=x, mx=mx, output=output, expected=expected):
        # The sentinel buffer is the kernel's own out argument, so any store
        # beyond the vl active lanes overwrites it and fails the compare.
        return [x.copy(), mx.copy(), output.copy()], expected.copy(), []

    def check(device_inputs, golden):
        np.testing.assert_allclose(
            device_inputs[-1].cpu().numpy(), golden, rtol=1e-3, atol=1e-6
        )

    return {
        "name": f"vexpdif_f16_vl{vl}_canary",
        "kernel": kernel,
        "make_case": make_case,
        "check": check,
    }


def _ab_case(vl):
    kernel = _vexpdif_ab_kernel(f"vexpdif_f16_vl{vl}_ab_kernel", vl)
    x, mx = _special_inputs(vl)

    def make_case(x=x, mx=mx, vl=vl):
        out_a = np.zeros(vl, dtype=np.float32)
        out_b = np.zeros(vl, dtype=np.float32)
        return [x.copy(), mx.copy(), out_a, out_b], None, []

    def check(device_inputs, golden):
        out_a = device_inputs[-2].cpu().numpy()
        out_b = device_inputs[-1].cpu().numpy()
        # The two hardware entries must agree bit-for-bit, NaN lanes
        # included: f16 is exact in f32, so the widened inputs carry the
        # same values the interleave entry reads from the packed pairs.
        np.testing.assert_array_equal(out_a, out_b)

    return {
        "name": f"vexpdif_f16_vl{vl}_ab_entries",
        "kernel": kernel,
        "make_case": make_case,
        "check": check,
    }


CASES = [
    _value_case("vexpdif_f16_vl1", 1, 1),
    _value_case("vexpdif_f16_vl8", 8, 8),
    _value_case("vexpdif_f16_vl64", 64, 64),
    _value_case("vexpdif_f16_vl8_half_mask", 8, 4),
    _value_case("vexpdif_f16_vl64_half_mask", 64, 32),
    _value_case("vexpdif_f16_vl8_zero_mask", 8, 0),
    _canary_case(8),
    _value_case("vexpdif_f16_vl128", 128, 128),
    _value_case("vexpdif_f16_vl128_extf", 128, 128, widen_to_f32=True),
    _ab_case(128),
]

auto_main(globals())
