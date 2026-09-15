#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Seven public lane sizes, sparse/empty/prefix masks and output canaries.

Run with the standard DSL ST runner, or use --emit-mlir without an NPU.
The raw UB loads deliberately use size=L, including 64/128-byte histogram
inputs. Padded input allocations honor the existing 32-byte short-read ABI.
"""

import numpy as np

from common import auto_main
from ptodsl import pto

LANES = (1, 2, 4, 8, 64, 128, 256)
MODES = ("full", "empty", "prefix", "sparse")


def _copy_in(src, dst, size):
    pto.mte_gm_ub(src, dst, 0, size, nburst=(1, size, size))


def _copy_out(src, dst, size):
    pto.mte_ub_gm(src, dst, size, nburst=(1, size, size))


def _scatter_kernel(n, dtype, index_dtype, itemsize, name):
    src_bytes = max(32, n * itemsize)
    index_bytes = max(32, n * (4 if itemsize == 4 else 2))
    output_bytes = 288 * itemsize

    @pto.jit(name=name, target="a5", backend="vpto", mode="explicit",
             kernel_kind="vector", insert_sync=False)
    def kernel(src: pto.ptr(dtype, "gm"), indices: pto.ptr(index_dtype, "gm"),
               gates: pto.ptr(dtype, "gm"), output: pto.ptr(dtype, "gm"),
               active: pto.i32):
        src_ub = pto.castptr(pto.i64(0), pto.ptr(dtype, "ub"))
        idx_ub = pto.castptr(pto.i64(2048), pto.ptr(index_dtype, "ub"))
        gate_ub = pto.castptr(pto.i64(4096), pto.ptr(dtype, "ub"))
        out_ub = pto.castptr(pto.i64(6144), pto.ptr(dtype, "ub"))
        _copy_in(src, src_ub, src_bytes)
        _copy_in(indices, idx_ub, index_bytes)
        _copy_in(gates, gate_ub, src_bytes)
        _copy_in(output, out_ub, output_bytes)
        pto.set_flag("MTE2", "V", event_id=0)
        pto.wait_flag("MTE2", "V", event_id=0)
        values = pto.vmi.vload(src_ub, 0, size=n)
        offsets = pto.vmi.vload(idx_ub, 0, size=n)
        gate = pto.vmi.vload(gate_ub, 0, size=n)
        prefix = pto.vmi.create_mask(active, size=n)
        mask = pto.vmi.vcmps(gate, dtype(0), prefix, pto.CmpMode.GT)
        pto.vmi.vscatter(values, out_ub, offsets, mask)
        pto.set_flag("V", "MTE3", event_id=0)
        pto.wait_flag("V", "MTE3", event_id=0)
        _copy_out(out_ub, output, output_bytes)
        pto.pipe_barrier(pto.Pipe.ALL)

    return kernel


def _hist_kernel(n, bins, op_name, name):
    src_bytes = max(32, n)
    output_bytes = (bins + 32) * 2
    histogram = getattr(pto.vmi, op_name)

    @pto.jit(name=name, target="a5", backend="vpto", mode="explicit",
             kernel_kind="vector", insert_sync=False)
    def kernel(src: pto.ptr(pto.ui8, "gm"), acc: pto.ptr(pto.ui16, "gm"),
               gates: pto.ptr(pto.ui8, "gm"), output: pto.ptr(pto.ui16, "gm"),
               active: pto.i32):
        src_ub = pto.castptr(pto.i64(0), pto.ptr(pto.ui8, "ub"))
        acc_ub = pto.castptr(pto.i64(2048), pto.ptr(pto.ui16, "ub"))
        gate_ub = pto.castptr(pto.i64(4096), pto.ptr(pto.ui8, "ub"))
        out_ub = pto.castptr(pto.i64(6144), pto.ptr(pto.ui16, "ub"))
        _copy_in(src, src_ub, src_bytes)
        _copy_in(acc, acc_ub, bins * 2)
        _copy_in(gates, gate_ub, src_bytes)
        _copy_in(output, out_ub, output_bytes)
        pto.set_flag("MTE2", "V", event_id=0)
        pto.wait_flag("MTE2", "V", event_id=0)
        source = pto.vmi.vload(src_ub, 0, size=n)
        accumulator = pto.vmi.vload(acc_ub, 0, size=bins)
        gate = pto.vmi.vload(gate_ub, 0, size=n)
        prefix = pto.vmi.create_mask(active, size=n)
        mask = pto.vmi.vcmps(gate, pto.ui8(0), prefix, pto.CmpMode.GT)
        result = histogram(accumulator, source, mask)
        pto.vmi.vstore(result, out_ub, 16, pto.vmi.create_mask(bins, size=bins))
        pto.set_flag("V", "MTE3", event_id=0)
        pto.wait_flag("V", "MTE3", event_id=0)
        _copy_out(out_ub, output, output_bytes)
        pto.pipe_barrier(pto.Pipe.ALL)

    return kernel


def _predicate(n, mode, storage, dtype):
    active = 0 if mode == "empty" else min(96, max(0, n - 1)) if mode == "prefix" else n
    gates = np.ones(storage, dtype=dtype)
    if mode == "sparse":
        gates[np.arange(storage) % 3 == 1] = 0
    selected = (np.arange(n) < active) & (gates[:n] != 0)
    return active, gates, selected


def _case(name, kernel, arrays, expected, active):
    def make_case():
        return [a.copy() for a in arrays], expected.copy(), [active]

    def check(device_inputs, golden):
        np.testing.assert_array_equal(device_inputs[-1].cpu().numpy(), golden)

    return dict(name=name, kernel=kernel, make_case=make_case, check=check)


def _cases():
    cases = []
    for n in LANES:
        for dtype, np_dtype, index_dtype, np_index in (
            (pto.f32, np.float32, pto.i32, np.int32),
            (pto.f16, np.float16, pto.ui16, np.uint16),
            (pto.ui8, np.uint8, pto.ui16, np.uint16),
            (pto.i8, np.int8, pto.ui16, np.uint16),
        ):
            itemsize = np.dtype(np_dtype).itemsize
            storage = max(32 // itemsize, n)
            name = f"vmi_scatter_{np.dtype(np_dtype).name}_n{n}"
            kernel = _scatter_kernel(n, dtype, index_dtype, itemsize, name)
            # Distinct odd/even byte values expose a missing B8 rearrangement.
            src = ((np.arange(storage) * 53 + 137) % 256).astype(np_dtype)
            indices = np.zeros(max(32 // np.dtype(np_index).itemsize, n), dtype=np_index)
            indices[:n] = (np.arange(n) * 17 + 3) % 256 + 16
            for mode in MODES:
                active, gates, selected = _predicate(n, mode, storage, np_dtype)
                output = np.full(288, 211 if np_dtype == np.uint8 else -91, dtype=np_dtype)
                expected = output.copy()
                expected[indices[:n][selected]] = src[:n][selected]
                cases.append(_case(name + "_" + mode, kernel,
                                   [src, indices, gates, output], expected, active))
        for bins in (128, 256):
            for op_name in ("vdhist", "vchist"):
                name = f"vmi_{op_name}_bins{bins}_n{n}"
                kernel = _hist_kernel(n, bins, op_name, name)
                src = ((np.arange(max(32, n)) * 53 + 137) % 256).astype(np.uint8)
                acc = (np.arange(bins) % 5).astype(np.uint16)
                for mode in MODES:
                    active, gates, selected = _predicate(n, mode, len(src), np.uint8)
                    counts = np.bincount(src[:n][selected].astype(np.int64), minlength=256)
                    if op_name == "vchist":
                        counts = np.cumsum(counts)
                    output = np.full(bins + 32, 0xCCCC, dtype=np.uint16)
                    expected = output.copy()
                    expected[16:16 + bins] = acc + counts[:bins]
                    cases.append(_case(name + "_" + mode, kernel,
                                       [src, acc, gates, output], expected, active))
    return cases


CASES = _cases()
auto_main(globals())
