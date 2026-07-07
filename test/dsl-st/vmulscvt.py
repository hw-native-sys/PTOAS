#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""
VMULSCVT simulator ST.

This case intentionally follows the softmax C++ micro-op shape more closely:

- `pto.vmulscvt(..., part=EVEN)`
- `pto.vbitcast(..., pto.ui32)`
- `pto.vpack(..., LOWER)`
- UB materialization via `pto.vsts`

The observable is the packed `u16` register image after the `vmulscvt + vpack`
sequence. That keeps the test close to the C++ authoring style without relying
on `vsstb.post`, which is not available on the current PTODSL surface yet.
"""

import numpy as np

from common import auto_main, golden_output_case
from ptodsl import pto


SRC_COLS = 64
OUT_COLS = 128
SCALE = -0.5


@pto.jit(
    name="vmulscvt_pack_kernel",
    kernel_kind="vector",
    target="a5",
    mode="explicit",
    insert_sync=False,
)
def vmulscvt_pack_kernel(
    inp_ptr: pto.ptr(pto.f32, "gm"),
    out_ptr: pto.ptr(pto.ui16, "gm"),
):
    inp_total = SRC_COLS
    out_total = OUT_COLS
    offsets = [0, 0, 0, 0, 0]
    inp_view = pto.make_tensor_view(
        inp_ptr,
        shape=[1, 1, 1, 1, SRC_COLS],
        strides=[inp_total, inp_total, inp_total, inp_total, 1],
    )
    out_view = pto.make_tensor_view(
        out_ptr,
        shape=[1, 1, 1, 1, OUT_COLS],
        strides=[out_total, out_total, out_total, out_total, 1],
    )
    inp_part = pto.partition_view(inp_view, offsets=offsets, sizes=[1, 1, 1, 1, SRC_COLS])
    out_part = pto.partition_view(out_view, offsets=offsets, sizes=[1, 1, 1, 1, OUT_COLS])

    src_tile = pto.alloc_tile(
        shape=[1, SRC_COLS],
        dtype=pto.f32,
        addr=0,
        valid_shape=[1, SRC_COLS],
        blayout="RowMajor",
    )
    dst_tile = pto.alloc_tile(
        shape=[1, OUT_COLS],
        dtype=pto.ui16,
        addr=2048,
        valid_shape=[1, OUT_COLS],
        blayout="RowMajor",
    )

    pto.tile.load(inp_part, src_tile)
    pto.set_flag("MTE2", "V", event_id=0)
    pto.wait_flag("MTE2", "V", event_id=0)

    with pto.tileop():
        mask32 = pto.pset_b32(pto.MaskPattern.ALL)
        mask16 = pto.pset_b16(pto.MaskPattern.ALL)

        src = pto.vlds(src_tile[0, 0:])
        packed_f16 = pto.vmulscvt(
            src,
            SCALE,
            mask32,
            rnd=pto.VcvtRoundMode.A,
            part=pto.PartMode.EVEN,
        )
        packed_u32 = pto.vbitcast(packed_f16, pto.ui32)
        packed_u16 = pto.vpack(packed_u32, pto.VPackPart.LOWER)
        pto.vsts(packed_u16, dst_tile.as_ptr(), 0, mask16, dist="NORM_B16")

    pto.set_flag("V", "MTE3", event_id=0)
    pto.wait_flag("V", "MTE3", event_id=0)
    pto.tile.store(dst_tile, out_part)


def make_inputs():
    # Use exact binary fractions so the test focuses on packing/layout rather
    # than float16 tie-breaking behavior.
    inp = ((np.arange(SRC_COLS, dtype=np.int32) - (SRC_COLS // 2)) / 8.0).astype(np.float32)
    return [inp.reshape(1, SRC_COLS)]


def make_expected(inp):
    scaled = (inp.astype(np.float32) * np.float32(SCALE)).astype(np.float16).reshape(-1)
    packed = np.zeros((OUT_COLS,), dtype=np.uint16)
    packed[:SRC_COLS] = scaled.view(np.uint16)
    return packed.reshape(1, OUT_COLS)


CASES = [
    golden_output_case(
        "vmulscvt_f32_to_f16_vpack_lower",
        vmulscvt_pack_kernel,
        inputs=make_inputs,
        expected=make_expected,
        rtol=0.0,
        atol=0.0,
    ),
]


auto_main(globals())
