#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""
Predicate pack/unpack simulator ST.

Covers:
- ppack(..., LOWER)
- punpack(..., LOWER)
- ppack(..., LOWER, to_type=pto.mask_b16)
- punpack(..., LOWER, to_type=pto.mask_b32)
- ppack(..., HIGHER)
- punpack(..., HIGHER)
- ppack(..., HIGHER, to_type=pto.mask_b16)
- punpack(..., HIGHER, to_type=pto.mask_b32)

The observable is the raw predicate register image materialized via ``pto.psts``.
This matches the existing VPTO runtime predicate tests and avoids the ambiguity
of inferring mask semantics through predicated vector stores.
"""

import numpy as np

from common import auto_main, golden_output_case
from ptodsl import pto


ROW_BYTES = 32
ROWS = 9


@pto.jit(
    name="predicate_pack_part_kernel",
    kernel_kind="vector",
    target="a5",
    mode="explicit",
    insert_sync=False,
)
def predicate_pack_part_kernel(
    inp_ptr: pto.ptr(pto.ui8, "gm"),
    out_ptr: pto.ptr(pto.ui8, "gm"),
    rows: pto.i32,
    cols: pto.i32,
):
    inp_total = rows * cols
    out_total = ROWS * ROW_BYTES
    offsets = [0, 0, 0, 0, 0]
    inp_view = pto.make_tensor_view(
        inp_ptr,
        shape=[1, 1, 1, rows, cols],
        strides=[inp_total, inp_total, inp_total, cols, 1],
    )
    out_view = pto.make_tensor_view(
        out_ptr,
        shape=[1, 1, 1, ROWS, ROW_BYTES],
        strides=[out_total, out_total, out_total, ROW_BYTES, 1],
    )
    inp_part = pto.partition_view(inp_view, offsets=offsets, sizes=[1, 1, 1, rows, cols])
    out_part = pto.partition_view(out_view, offsets=offsets, sizes=[1, 1, 1, ROWS, ROW_BYTES])

    src_tile = pto.alloc_tile(
        shape=[1, 32],
        dtype=pto.ui8,
        addr=0,
        valid_shape=[rows, cols],
    )
    dst_tile = pto.alloc_tile(
        shape=[ROWS, ROW_BYTES],
        dtype=pto.ui8,
        addr=1024,
        valid_shape=[ROWS, ROW_BYTES],
    )
    src_ptr = pto.castptr(pto.ui64(0), pto.ptr(pto.ui8, "ub"))
    dst_ptr = pto.castptr(pto.ui64(1024), pto.ptr(pto.ui8, "ub"))

    pto.tile.load(inp_part, src_tile)
    pto.tile.load(out_part, dst_tile)
    pto.set_flag("MTE2", "V", event_id=0)
    pto.wait_flag("MTE2", "V", event_id=0)

    seed = pto.pset_b8(pto.MaskPattern.ALL)
    src = pto.vlds(src_ptr, 0, pto.vreg_type(256, pto.ui8))
    active_b8 = pto.vcmp(src, src, seed, pto.CmpMode.EQ)
    active = pto.pbitcast(active_b8, pto.mask_b32)

    packed_lo_same = pto.ppack(active, "LOWER")
    unpacked_lo_same = pto.punpack(packed_lo_same, "LOWER")
    packed_lo_b16 = pto.ppack(active, "LOWER", to_type=pto.mask_b16)
    unpacked_lo_b32 = pto.punpack(packed_lo_b16, "LOWER", to_type=pto.mask_b32)

    packed_hi_same = pto.ppack(active, "HIGHER")
    unpacked_hi_same = pto.punpack(packed_hi_same, "HIGHER")
    packed_hi_b16 = pto.ppack(active, "HIGHER", to_type=pto.mask_b16)
    unpacked_hi_b32 = pto.punpack(packed_hi_b16, "HIGHER", to_type=pto.mask_b32)

    pto.psts(active, dst_ptr, 0, dist="NORM")
    pto.psts(packed_lo_same, dst_ptr, ROW_BYTES, dist="NORM")
    pto.psts(unpacked_lo_same, dst_ptr, ROW_BYTES * 2, dist="NORM")
    pto.psts(packed_lo_b16, dst_ptr, ROW_BYTES * 3, dist="NORM")
    pto.psts(unpacked_lo_b32, dst_ptr, ROW_BYTES * 4, dist="NORM")
    pto.psts(packed_hi_same, dst_ptr, ROW_BYTES * 5, dist="NORM")
    pto.psts(unpacked_hi_same, dst_ptr, ROW_BYTES * 6, dist="NORM")
    pto.psts(packed_hi_b16, dst_ptr, ROW_BYTES * 7, dist="NORM")
    pto.psts(unpacked_hi_b32, dst_ptr, ROW_BYTES * 8, dist="NORM")

    pto.set_flag("V", "MTE3", event_id=0)
    pto.wait_flag("V", "MTE3", event_id=0)
    pto.tile.store(dst_tile, out_part)


def make_inputs():
    return [np.arange(32, dtype=np.uint8).reshape(1, 32)]


def make_expected(_inp):
    # `active` is produced as mask_b8 by `vcmp` over 256 ui8 lanes, then
    # reinterpreted to mask_b32 with `pbitcast`. `pbitcast` preserves the raw
    # 256-bit predicate register image, so the stored bytes remain all ones.
    active = np.full((ROW_BYTES,), 0xFF, dtype=np.uint8)

    # `ppack/punpack` operate on that raw image. For an all-ones source:
    # - `ppack(LOWER)` keeps the lower packed half, leaving the upper half zero
    # - `ppack(HIGHER)` keeps the upper packed half, leaving the lower half zero
    # - `punpack(*)` expands the selected half back into alternating low bits
    #   in the raw predicate image, which materializes as 0x55 bytes.
    packed_lo_same = np.concatenate(
        [
            np.full((ROW_BYTES // 2,), 0xFF, dtype=np.uint8),
            np.zeros((ROW_BYTES // 2,), dtype=np.uint8),
        ]
    )
    unpacked_lo_same = np.full((ROW_BYTES,), 0x55, dtype=np.uint8)
    packed_lo_b16 = packed_lo_same.copy()
    unpacked_lo_b32 = unpacked_lo_same.copy()

    packed_hi_same = np.concatenate(
        [
            np.zeros((ROW_BYTES // 2,), dtype=np.uint8),
            np.full((ROW_BYTES // 2,), 0xFF, dtype=np.uint8),
        ]
    )
    unpacked_hi_same = np.full((ROW_BYTES,), 0x55, dtype=np.uint8)
    packed_hi_b16 = packed_hi_same.copy()
    unpacked_hi_b32 = unpacked_hi_same.copy()

    expected = np.vstack(
        [
            active,
            packed_lo_same,
            unpacked_lo_same,
            packed_lo_b16,
            unpacked_lo_b32,
            packed_hi_same,
            unpacked_hi_same,
            packed_hi_b16,
            unpacked_hi_b32,
        ]
    )
    return expected


CASES = [
    golden_output_case(
        "predicate_pack_part_roundtrip",
        predicate_pack_part_kernel,
        inputs=make_inputs,
        expected=make_expected,
        launch_args=lambda inp: [inp.shape[0], inp.shape[1]],
        output_index=1,
        rtol=0.0,
        atol=0.0,
    ),
]

auto_main(globals())
