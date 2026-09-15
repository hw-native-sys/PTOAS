#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""One-carrier grouped reductions with holes, empty groups and original identities."""
import numpy as np
from common import auto_main
from ptodsl import pto

TRANSFER_BYTES = 4096
SECTION_BYTES = 512
OUTPUT_INDEX = 2


def build_kernel(dtype, lanes, groups, itemsize, name):
    section = SECTION_BYTES // itemsize

    @pto.jit(name=name, target="a5", backend="vpto", mode="explicit",
             kernel_kind="vector", insert_sync=False)
    def kernel(src: pto.ptr(dtype, "gm"), gates: pto.ptr(dtype, "gm"),
               out: pto.ptr(dtype, "gm"), active: pto.i32):
        src_ub = pto.castptr(pto.i64(0), pto.ptr(dtype, "ub"))
        gate_ub = pto.castptr(pto.i64(4096), pto.ptr(dtype, "ub"))
        out_ub = pto.castptr(pto.i64(8192), pto.ptr(dtype, "ub"))
        pto.mte_gm_ub(src, src_ub, 0, TRANSFER_BYTES,
                      nburst=(1, TRANSFER_BYTES, TRANSFER_BYTES))
        pto.mte_gm_ub(gates, gate_ub, 0, TRANSFER_BYTES,
                      nburst=(1, TRANSFER_BYTES, TRANSFER_BYTES))
        pto.mte_gm_ub(out, out_ub, 0, TRANSFER_BYTES,
                      nburst=(1, TRANSFER_BYTES, TRANSFER_BYTES))
        pto.set_flag("MTE2", "V", event_id=0)
        pto.wait_flag("MTE2", "V", event_id=0)
        value = pto.vmi.vload(src_ub, 0, size=lanes)
        gate = pto.vmi.vload(gate_ub, 0, size=lanes)
        prefix = pto.vmi.create_mask(active, size=lanes)
        mask = pto.vmi.vcmps(gate, dtype(0), prefix, pto.CmpMode.GT)
        total = pto.vmi.vcadd(value, mask, group=groups, reassoc=True)
        maximum = pto.vmi.vcmax(value, mask, group=groups)
        minimum = pto.vmi.vcmin(value, mask, group=groups)
        pto.vmi.vstore(total, out_ub, 0, stride=1, group=groups)
        pto.vmi.vstore(maximum, out_ub, section, stride=1, group=groups)
        pto.vmi.vstore(minimum, out_ub, 2 * section, stride=1, group=groups)
        sum_brc = pto.vmi.vbrc(total, size=lanes, group=groups)
        max_brc = pto.vmi.vbrc(maximum, size=lanes, group=groups)
        min_brc = pto.vmi.vbrc(minimum, size=lanes, group=groups)
        pto.vmi.vstore(sum_brc, out_ub, 3 * section)
        pto.vmi.vstore(max_brc, out_ub, 4 * section)
        pto.vmi.vstore(min_brc, out_ub, 5 * section)
        compact = pto.vmi.vload(src_ub, 0, size=groups)
        broadcast = pto.vmi.vbrc(compact, size=lanes, group=groups)
        pto.vmi.vstore(broadcast, out_ub, 6 * section)
        pto.set_flag("V", "MTE3", event_id=0)
        pto.wait_flag("V", "MTE3", event_id=0)
        pto.mte_ub_gm(out_ub, out, TRANSFER_BYTES,
                      nburst=(1, TRANSFER_BYTES, TRANSFER_BYTES))
        pto.pipe_barrier(pto.Pipe.ALL)

    return kernel


def make_case(npdtype, lanes, groups, mode):
    dtype = np.dtype(npdtype)
    floating = np.issubdtype(dtype, np.floating)
    limits = None if floating else np.iinfo(dtype)
    sample = ([-3, -1, 0.5, 2, 4, -0.25, 8, -9] if floating else
              [limits.max, limits.max - 1, limits.min, 0, 1, 3, 7, 11])
    if mode == "inf":
        sample = [-np.inf, -0.0, 0.0, np.inf, -2, 4, -8, 16]
    if mode == "nan":
        sample = [np.nan, -1, 2, np.nan, -np.inf, np.inf, -0.0, 0.0]
    if mode == "zeros":
        sample = [-0.0, 0.0, -0.0, -0.0, 0.0, -0.0, 0.0, 0.0]
    source = np.full(TRANSFER_BYTES // dtype.itemsize, 17, dtype=dtype)
    source[:lanes] = np.resize(np.array(sample, dtype=dtype), lanes)
    gates = np.ones_like(source)
    active = 0 if mode == "empty" else lanes - 1 if mode == "tail" else lanes + 1
    if mode == "holes":
        gates[np.arange(gates.size) % 3 == 1] = 0
    if mode == "middle_empty" and groups > 1:
        begin = (groups // 2) * (lanes // groups)
        gates[begin:begin + lanes // groups] = 0
    output = np.full_like(source, 23)
    expected = output.copy()
    section = SECTION_BYTES // dtype.itemsize
    selected = (np.arange(lanes) < active) & (gates[:lanes] != 0)
    identities = (0, -np.inf if floating else limits.min,
                  np.inf if floating else limits.max)
    for group in range(groups):
        begin = group * (lanes // groups)
        end = begin + lanes // groups
        values = source[begin:end][selected[begin:end]]
        if values.size:
            # Exact integer accumulation and representable ordinary FP samples
            # avoid coupling the golden to the hardware's reduction tree.
            with np.errstate(invalid="ignore", over="ignore"):
                total = sum(float(v) if floating else int(v) for v in values)
                reductions = (total, np.max(values), np.min(values))
        else:
            reductions = identities
        for op, value in enumerate(reductions):
            if not floating:
                value = int(value) % (1 << (dtype.itemsize * 8))
                if limits.min < 0 and value > limits.max:
                    value -= 1 << (dtype.itemsize * 8)
            expected[op * section + group] = value
            expected[(op + 3) * section + begin:(op + 3) * section + end] = value
    expected[6 * section:6 * section + lanes] = np.repeat(source[:groups], lanes // groups)
    return [source, gates, output], expected, [active]


def check_case(inputs, expected):
    np.testing.assert_allclose(inputs[OUTPUT_INDEX].cpu().numpy(), expected,
                               rtol=0, atol=0, equal_nan=True)


TYPE_CASES = (
    ("i8", pto.i8, np.uint8), ("si8", pto.si8, np.int8), ("ui8", pto.ui8, np.uint8),
    ("i16", pto.i16, np.uint16), ("si16", pto.si16, np.int16), ("ui16", pto.ui16, np.uint16),
    ("i32", pto.i32, np.uint32), ("si32", pto.si32, np.int32), ("ui32", pto.ui32, np.uint32),
    ("f16", pto.f16, np.float16), ("f32", pto.f32, np.float32),
)
CASES = []
KERNELS = []
for label, dtype, npdtype in TYPE_CASES:
    itemsize = np.dtype(npdtype).itemsize
    for lanes in (8, 64, 128, 256):
        if lanes * itemsize > 256:
            continue
        for groups in (1, 2, 4, 8):
            name = f"group_boundary_{label}_vl{lanes}_g{groups}"
            kernel = build_kernel(dtype, lanes, groups, itemsize, name)
            KERNELS.append(kernel)
            modes = ("empty", "tail", "full", "holes", "middle_empty")
            if np.issubdtype(npdtype, np.floating):
                modes += ("inf", "nan", "zeros")
            for mode in modes:
                def create_case(npdtype=npdtype, lanes=lanes, groups=groups, mode=mode):
                    return make_case(npdtype, lanes, groups, mode)
                CASES.append(dict(name=f"{name}_{mode}", kernel=kernel,
                                  make_case=create_case, check=check_case))


auto_main(globals())
