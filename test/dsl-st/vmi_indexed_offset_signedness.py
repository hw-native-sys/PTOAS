#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Signed offset bits select unsigned element positions in gather and scatter."""

import numpy as np

from common import auto_main, golden_output_case
from ptodsl import pto

LANES = 64


def _kernel(offset_bits, payload_bits):
    offset_dtype = pto.si8 if offset_bits == 8 else pto.si16
    payload_dtype = pto.ui16 if payload_bits == 16 else pto.ui32
    base = 1 << (offset_bits - 1)
    source_bytes = (base + LANES) * payload_bits // 8
    index_bytes = LANES * offset_bits // 8
    result_bytes = LANES * 2 * payload_bits // 8

    @pto.jit(
        name=f"indexed_si{offset_bits}_b{payload_bits}_kernel",
        kernel_kind="vector", target="a5", backend="vpto", mode="explicit",
    )
    def kernel(
        source: pto.ptr(payload_dtype, "gm"),
        indices: pto.ptr(offset_dtype, "gm"),
        output: pto.ptr(payload_dtype, "gm"),
    ):
        ub_source = pto.castptr(pto.i64(0), pto.ptr(payload_dtype, "ub"))
        ub_indices = pto.castptr(pto.i64(source_bytes), pto.ptr(offset_dtype, "ub"))
        ub_output = pto.castptr(pto.i64(source_bytes + index_bytes), pto.ptr(payload_dtype, "ub"))
        pto.mte_gm_ub(source, ub_source, 0, source_bytes, nburst=(1, 0, 0))
        pto.mte_gm_ub(indices, ub_indices, 0, index_bytes, nburst=(1, 0, 0))
        pto.set_flag("MTE2", "V", event_id=0)
        pto.wait_flag("MTE2", "V", event_id=0)

        offsets = pto.vmi.vload(ub_indices, 0, size=LANES)
        mask = pto.vmi.create_mask(LANES, size=LANES)
        gathered = pto.vmi.vgather(ub_source, offsets, mask)
        pto.vmi.vstore(gathered, ub_output, 0)
        values = pto.vmi.vload(ub_source, 0, size=LANES)
        pto.vmi.vscatter(values, ub_source, offsets, mask)
        pto.pipe_barrier(pto.Pipe.V)
        scattered = pto.vmi.vload(ub_source, base, size=LANES)
        pto.vmi.vstore(scattered, ub_output, LANES)

        pto.set_flag("V", "MTE3", event_id=0)
        pto.wait_flag("V", "MTE3", event_id=0)
        pto.mte_ub_gm(ub_output, output, result_bytes, nburst=(1, 0, 0))
        pto.pipe_barrier(pto.Pipe.ALL)

    return kernel


def _case(offset_bits, payload_bits):
    base = 1 << (offset_bits - 1)
    signed_dtype = np.int8 if offset_bits == 8 else np.int16
    unsigned_dtype = np.uint8 if offset_bits == 8 else np.uint16
    payload_dtype = np.uint16 if payload_bits == 16 else np.uint32
    permutation = (np.arange(LANES) * 7 + 13) % LANES
    offsets = (base + permutation).astype(unsigned_dtype).view(signed_dtype)
    source = (np.arange(base + LANES, dtype=np.uint32) * 37 + 11).astype(payload_dtype)
    expected_scatter = np.empty(LANES, dtype=payload_dtype)
    expected_scatter[permutation] = source[:LANES]
    expected = np.concatenate((source[base + permutation], expected_scatter))
    return golden_output_case(
        f"indexed_si{offset_bits}_b{payload_bits}_unsigned_positions",
        _kernel(offset_bits, payload_bits),
        inputs=[source, offsets], expected=expected, rtol=0, atol=0,
    )


CASES = [_case(8, 16), _case(8, 32), _case(16, 32)]

auto_main(globals())
