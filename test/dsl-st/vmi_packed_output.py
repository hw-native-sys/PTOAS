#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Packed BF16 output preserves conversion modes and interleaved lane order."""
import numpy as np

from common import auto_main
from ptodsl import pto

ELEMENTS = 4096


def _build_kernel(lanes, fuse):
    @pto.jit(name=f"vmi_packed_output_{lanes}_{fuse}_kernel", target="a5", backend="vpto",
             mode="explicit", kernel_kind="vector", insert_sync=False)
    def _kernel(src: pto.ptr(pto.f32, "gm"), dst: pto.ptr(pto.bf16, "gm")):
        src_ub = pto.castptr(pto.i64(0), pto.ptr(pto.f32, "ub"))
        dst_ub = pto.castptr(pto.i64(2048), pto.ptr(pto.bf16, "ub"))
        scratch = pto.castptr(pto.i64(4096), pto.ptr(pto.f32, "ub"))
        for offset in pto.range(0, ELEMENTS, lanes * 2):
            pto.mte_gm_ub(pto.addptr(src, offset), src_ub, 0, lanes * 8,
                          nburst=(1, lanes * 8, lanes * 8))
            pto.set_flag("MTE2", "V", event_id=0)
            pto.wait_flag("MTE2", "V", event_id=0)
            a = pto.vmi.vload(src_ub, 0, size=lanes)
            b = pto.vmi.vload(src_ub, lanes, size=lanes)
            mask = pto.vmi.create_mask(lanes, size=lanes)
            low, high = pto.vmi.vintlv(a, b, mask)
            if not fuse:
                # An extra use keeps the original two-cast/two-store graph.
                pto.vmi.vstore(low, scratch, 0)
            x = pto.vmi.vcvt(low, pto.bf16)
            y = pto.vmi.vcvt(high, pto.bf16)
            pto.vmi.vstore(x, dst_ub, 0)
            pto.vmi.vstore(y, dst_ub, lanes)
            pto.set_flag("V", "MTE3", event_id=0)
            pto.wait_flag("V", "MTE3", event_id=0)
            pto.mte_ub_gm(dst_ub, pto.addptr(dst, offset), lanes * 4,
                          nburst=(1, lanes * 4, lanes * 4))
            pto.pipe_barrier(pto.Pipe.ALL)
    return _kernel


def _make_case(lanes):
    import torch
    # Both sides of rounding ties, signs, subnormals, finite extrema, infinities
    # and NaNs exercise numeric conversion rather than a truncating bit cast.
    special = np.array([0, 0x80000000, 1, 0x80000001, 0x00008000,
                        0x3F808000, 0x3F818000, 0xBF808000, 0xBF818000,
                        0x7F7FFFFF, 0xFF7FFFFF, 0x7F800000, 0xFF800000,
                        0x7FC01234], dtype=np.uint32).view(np.float32)
    source = np.random.default_rng(42).standard_normal(ELEMENTS).astype(np.float32)
    source[:special.size] = special
    packets = source.reshape(-1, 2, lanes).transpose(0, 2, 1).reshape(-1).copy()
    # The default SAT conversion clamps infinities and rounding overflow.
    limit = np.array([0x7F7F0000], dtype=np.uint32).view(np.float32)[0]
    packets = np.clip(packets, -limit, limit)
    expected = torch.from_numpy(packets).to(torch.bfloat16).view(torch.uint16).numpy()
    return [source, np.zeros(ELEMENTS, dtype=np.uint16),
            np.zeros(ELEMENTS, dtype=np.uint16)], expected


def _check_case(device_inputs, reference):
    expected = reference
    actual = device_inputs[1].cpu().numpy()
    nan = ((expected & 0x7F80) == 0x7F80) & ((expected & 0x007F) != 0)
    np.testing.assert_array_equal(actual[~nan], expected[~nan])
    baseline = device_inputs[2].cpu().numpy()
    np.testing.assert_array_equal(actual, baseline)


def _case(lanes):
    packed = _build_kernel(lanes, True)
    original = _build_kernel(lanes, False)

    def make_case():
        return _make_case(lanes)

    def run(inputs, launch_args, stream):
        packed.compile()[1, stream](inputs[0], inputs[1])
        original.compile()[1, stream](inputs[0], inputs[2])

    return {"name": f"vmi_packed_output_{lanes}", "kernel": packed,
            "make_case": make_case, "run": run, "check": _check_case}


CASES = [_case(64), _case(128)]


auto_main(globals())
