#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""A5 i16 pto.vdiv with partial (non-PAT_ALL) b16 masks.

The A5 softlib splits the 128-lane i16 problem through vintlv/vcvt and must
re-interleave the active mask with pintlv_b16 the same way; otherwise lanes
gated by the wrong predicate bits produce wrong quotients (regression for
issue #1241 review: only PAT_ALL was covered before).  Cases here exercise a
prefix (PAT_VL64 / PAT_VL2), an arbitrary data-dependent predicate, and a mask
that spans the 64-lane boundary with a leading hole.

Each kernel first writes a sentinel over the whole tile with an unmasked store
and then overwrites the active lanes masked, so GM content is deterministic
regardless of leftover UB state from the previous kernel.
"""
import numpy as np

from common import auto_main, golden_output_case
from ptodsl import pto

COLS = 128
BASE = 80  # lhs values 80..207 so no quotient is 0 (sentinels are 0)
RHS = 3


def _build_kernel(name, mask_builder):
    @pto.jit(
        name=name,
        kernel_kind="vector",
        target="a5",
        mode="explicit",
        insert_sync=False,
    )
    def kernel(
        lhs_ptr: pto.ptr(pto.i16, "gm"),
        rhs_ptr: pto.ptr(pto.i16, "gm"),
        out_ptr: pto.ptr(pto.i16, "gm"),
    ):
        total = COLS
        offsets = [0, 0, 0, 0, 0]
        lhs_view = pto.make_tensor_view(
            lhs_ptr, shape=[1, 1, 1, 1, COLS],
            strides=[total, total, total, total, 1],
        )
        rhs_view = pto.make_tensor_view(
            rhs_ptr, shape=[1, 1, 1, 1, COLS],
            strides=[total, total, total, total, 1],
        )
        out_view = pto.make_tensor_view(
            out_ptr, shape=[1, 1, 1, 1, COLS],
            strides=[total, total, total, total, 1],
        )
        lhs_part = pto.partition_view(lhs_view, offsets=offsets, sizes=[1, 1, 1, 1, COLS])
        rhs_part = pto.partition_view(rhs_view, offsets=offsets, sizes=[1, 1, 1, 1, COLS])
        out_part = pto.partition_view(out_view, offsets=offsets, sizes=[1, 1, 1, 1, COLS])

        lhs_tile = pto.alloc_tile(shape=[1, COLS], dtype=pto.i16, addr=0,
                                  valid_shape=[1, COLS], blayout="RowMajor")
        rhs_tile = pto.alloc_tile(shape=[1, COLS], dtype=pto.i16, addr=2048,
                                  valid_shape=[1, COLS], blayout="RowMajor")
        dst_tile = pto.alloc_tile(shape=[1, COLS], dtype=pto.i16, addr=4096,
                                  valid_shape=[1, COLS], blayout="RowMajor")

        pto.tile.load(lhs_part, lhs_tile)
        pto.tile.load(rhs_part, rhs_tile)
        pto.set_flag("MTE2", "V", event_id=0)
        pto.wait_flag("MTE2", "V", event_id=0)

        with pto.tileop():
            full_mask = pto.pset_b16(pto.MaskPattern.ALL)
            lhs_v = pto.vlds(lhs_tile[0, 0:])
            rhs_v = pto.vlds(rhs_tile[0, 0:])
            sentinel = pto.vbr(pto.i16(0))
            pto.vsts(sentinel, dst_tile.as_ptr(), 0, full_mask, dist="NORM_B16")
            mask = mask_builder(lhs_v)
            quotient = pto.vdiv(lhs_v, rhs_v, mask)
            pto.vsts(quotient, dst_tile.as_ptr(), 0, mask, dist="NORM_B16")

        pto.set_flag("V", "MTE3", event_id=0)
        pto.wait_flag("V", "MTE3", event_id=0)
        pto.tile.store(dst_tile, out_part)

    return kernel


def _make_inputs():
    x = (np.arange(COLS, dtype=np.int32) + BASE).astype(np.int16).reshape(1, COLS)
    y = np.full((1, COLS), RHS, dtype=np.int16)
    return [x, y]


def _cases():
    x = (np.arange(COLS, dtype=np.int32) + BASE).astype(np.int32)
    q_full = (x // RHS).astype(np.int16).reshape(1, COLS)

    def expect(active_mask_np):
        expected = np.zeros((1, COLS), dtype=np.int16).reshape(-1)
        expected[active_mask_np] = q_full.reshape(-1)[active_mask_np]
        return expected.reshape(1, COLS)

    out = []

    def add_case(name, mask_builder, active_mask_np):
        out.append(
            golden_output_case(
                name,
                _build_kernel(name, mask_builder),
                inputs=_make_inputs,
                expected=lambda a, b, am=active_mask_np: expect(am),
                rtol=0.0,
                atol=0.0,
            )
        )

    # Prefix masks.
    v64 = np.zeros(COLS, dtype=bool)
    v64[:64] = True
    add_case("vdiv_i16_mask_vl64", lambda v: pto.pset_b16(pto.MaskPattern.VL64), v64)

    v2 = np.zeros(COLS, dtype=bool)
    v2[:2] = True
    add_case("vdiv_i16_mask_vl2", lambda v: pto.pset_b16(pto.MaskPattern.VL2), v2)

    # Arbitrary data-dependent predicate: lhs < 120 -> lanes 0..39.
    sparse = x < 120
    add_case(
        "vdiv_i16_mask_sparse",
        lambda v: pto.vcmps(v, pto.i16(120), pto.pset_b16(pto.MaskPattern.ALL),
                            pto.CmpMode.LT),
        sparse,
    )

    # Crosses the 64-lane boundary with a leading hole: lhs >= 96 -> lanes 16..127.
    cross = x >= 96
    add_case(
        "vdiv_i16_mask_cross_boundary",
        lambda v: pto.vcmps(v, pto.i16(96), pto.pset_b16(pto.MaskPattern.ALL),
                            pto.CmpMode.GE),
        cross,
    )

    return out


CASES = _cases()


auto_main(globals())