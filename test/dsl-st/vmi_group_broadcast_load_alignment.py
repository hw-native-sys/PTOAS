#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Group-broadcast-load alignment coverage for group=2/4/8.

group=2/4 (and any group=8 whose 32-byte alignment is not provable) lower
through the group-slot fallback.  That fallback must only use the block-strided
`pto.vsldb` plan when the effective address is provably 32-byte aligned; the
offset=1 (4-byte aligned) cases here guard the lane-zero BRC path, the aligned
ones guard the vsldb/E2B paths.  The two ctl_plain_* cases cover the plain
contiguous load, whose unaligned form lowers to the stateful vldas+vldus
sequence.
"""

import numpy as np
from common import auto_main, assert_close
from ptodsl import pto


LANES = 64
ELEMENTS = 128
BYTES = ELEMENTS * 4
UB_SRC = 1024
UB_OUT = 2048


def build(name, group=None, offset=0, plain=False):
    @pto.jit(name=name, target="a5", backend="vpto", mode="explicit",
             kernel_kind="vector", insert_sync=False)
    def kernel(src: pto.ptr(pto.f32, "gm"), out: pto.ptr(pto.f32, "gm")):
        ub_src = pto.castptr(pto.i64(UB_SRC), pto.ptr(pto.f32, "ub"))
        ub_out = pto.castptr(pto.i64(UB_OUT), pto.ptr(pto.f32, "ub"))
        pto.mte_gm_ub(src, ub_src, 0, BYTES, nburst=(1, BYTES, BYTES))
        pto.mte_gm_ub(out, ub_out, 0, BYTES, nburst=(1, BYTES, BYTES))
        pto.set_flag("MTE2", "V", event_id=0)
        pto.wait_flag("MTE2", "V", event_id=0)
        if plain:
            value = pto.vmi.vload(ub_src, offset, size=LANES)
        else:
            value = pto.vmi.vload(ub_src, offset, size=LANES, dist_mode="brc",
                                  group=group, stride=1)
        pto.vmi.vstore(value, ub_out, 0)
        pto.set_flag("V", "MTE3", event_id=0)
        pto.wait_flag("V", "MTE3", event_id=0)
        pto.mte_ub_gm(ub_out, out, BYTES, nburst=(1, BYTES, BYTES))
        pto.pipe_barrier(pto.Pipe.ALL)
    return kernel


def make_case(source, output, expected):
    def factory():
        return [source.copy(), output.copy()], expected.copy()
    return factory


def check_case(device_inputs, golden):
    assert_close(device_inputs[-1].cpu().numpy(), golden, rtol=0.0, atol=0.0)


CASES = []

source = (np.arange(ELEMENTS, dtype=np.float32) + 1.0)
canary = np.full(ELEMENTS, 241.0, dtype=np.float32)

# Control: plain contiguous load of the same 64 lanes, same store path.
plain_expected = canary.copy()
plain_expected[:LANES] = source[:LANES]
CASES.append(dict(name="ctl_plain_load", kernel=build("ctl_plain_load", plain=True),
                  make_case=make_case(source, canary, plain_expected), check=check_case))

# Same plain load, but 4-byte (non-32B) aligned: mainline's unaligned fix should
# route this one through the stateful vldas+vldus sequence.
plain_off1_expected = canary.copy()
plain_off1_expected[:LANES] = source[1:1 + LANES]
CASES.append(dict(name="ctl_plain_off1", kernel=build("ctl_plain_off1", plain=True, offset=1),
                  make_case=make_case(source, canary, plain_off1_expected), check=check_case))

for group in (8, 4, 2):
    for offset in (0, 1):
        name = f"gbl_g{group}_off{offset}"
        expected = canary.copy()
        per_group = LANES // group
        for lane in range(LANES):
            expected[lane] = source[offset + lane // per_group]
        CASES.append(dict(name=name,
                          kernel=build(name, group=group, offset=offset),
                          make_case=make_case(source, canary, expected),
                          check=check_case))

auto_main(globals())