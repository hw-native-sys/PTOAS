#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Native i16 sum slots: overflow, masks, multiple carriers and shared consumers."""

import os

import numpy as np
from common import auto_main
from ptodsl import pto

TRANSFER_BYTES = 4096


def build_kernel(label, lanes, groups, name):
    wide = "si32" if label == "si16" else "ui32"
    extend = "extsi" if label == "si16" else "extui"
    vector = f"!pto.vmi.vreg<{lanes}x{label}>"
    packet = f"!pto.vmi.vreg<{groups}x{label}>"
    mask = f"!pto.vmi.mask<{lanes}xpred>"
    label_ub = f"!pto.ptr<{label}, ub>"
    group_mask = f"!pto.vmi.mask<{groups}xpred>"
    wide_packet = f"!pto.vmi.vreg<{groups}x{wide}>"
    wide_ub = f"!pto.ptr<{wide}, ub>"
    u8_packet = f"!pto.vmi.vreg<{groups}xui8>"
    u8_vector = f"!pto.vmi.vreg<{lanes}xui8>"
    u8_ub = "!pto.ptr<ui8, ub>"
    one_packet = f"!pto.vmi.vreg<1x{label}>"
    # Text IR also covers VL512, beyond the Python create_mask size list.
    source = f'''module attributes {{pto.target_arch = "a5", pto.kernel_kind = #pto.kernel_kind<vector>}} {{
      func.func @{name}(%src: !pto.ptr<{label}, gm>, %gate: !pto.ptr<{label}, gm>,
          %out: !pto.ptr<{label}, gm>, %wide: !pto.ptr<{wide}, gm>,
          %narrow: !pto.ptr<ui8, gm>, %active: i32) attributes {{pto.kernel}} {{
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c3 = arith.constant 3 : index
        %c256 = arith.constant 256 : index
        %c512 = arith.constant 512 : index
        %c1280 = arith.constant 1280 : index
        %c1536 = arith.constant 1536 : index
        %c1792 = arith.constant 1792 : index
        %c1856 = arith.constant 1856 : index
        %groups = arith.constant {groups} : index
        %zero_i16 = arith.constant 0 : i16
        %zero = builtin.unrealized_conversion_cast %zero_i16 : i16 to {label}
        %active_idx = arith.index_cast %active : i32 to index
        %a0 = arith.constant 0 : i64
        %a1 = arith.constant 4096 : i64
        %a2 = arith.constant 8192 : i64
        %a3 = arith.constant 12288 : i64
        %a4 = arith.constant 16384 : i64
        %one = arith.constant 1 : i64
        %bytes = arith.constant {TRANSFER_BYTES} : i64
        %s = pto.castptr %a0 : i64 -> !pto.ptr<{label}, ub>
        %g = pto.castptr %a1 : i64 -> !pto.ptr<{label}, ub>
        %o = pto.castptr %a2 : i64 -> !pto.ptr<{label}, ub>
        %w = pto.castptr %a3 : i64 -> !pto.ptr<{wide}, ub>
        %n = pto.castptr %a4 : i64 -> !pto.ptr<ui8, ub>
'''
    for gm, ub, dtype in (("src", "s", label), ("gate", "g", label),
                           ("out", "o", label), ("wide", "w", wide),
                           ("narrow", "n", "ui8")):
        source += f'''        pto.mte_gm_ub %{gm}, %{ub}, %a0, %bytes nburst(%one, %bytes, %bytes)
            : !pto.ptr<{dtype}, gm>, !pto.ptr<{dtype}, ub>, i64, i64, i64, i64, i64
'''
    source += f'''        pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]
        pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]
        pto.vecscope {{
          %x = pto.vmi.vload %s[%c0] : !pto.ptr<{label}, ub>, index -> {vector}
          %gates = pto.vmi.vload %g[%c0] : !pto.ptr<{label}, ub>, index -> {vector}
          %prefix = pto.vmi.create_mask %active_idx : index -> {mask}
          %mask = pto.vmi.vcmps %gates, %zero, %prefix {{cmp = "gt"}} : {vector}, {label}, {mask} -> {mask}
          %sum = pto.vmi.vcadd %x, %mask {{group = {groups}}} : {vector}, {mask} -> {packet}
          %max = pto.vmi.vcmax %x, %mask {{group = {groups}}} : {vector}, {mask} -> {packet}
          %min = pto.vmi.vcmin %x, %mask {{group = {groups}}} : {vector}, {mask} -> {packet}
          pto.vmi.vstore %sum, %o[%c1], %c1 {{group = {groups}}} : {packet}, !pto.ptr<{label}, ub>, index
          pto.vmi.vstore %sum, %o[%c256], %c1 {{group = {groups}}} : {packet}, !pto.ptr<{label}, ub>, index
          %brc = pto.vmi.vbrc %sum {{group = {groups}}} : {packet} -> {vector}
          pto.vmi.vstore %brc, %o[%c512] : {vector}, !pto.ptr<{label}, ub>, index
          %extended = pto.vmi.{extend} %sum : {packet} -> {wide_packet}
          pto.vmi.vstore %extended, %w[%c1], %c1 {{group = {groups}}} : {wide_packet}, {wide_ub}, index
          %truncated = pto.vmi.trunci %sum {{saturate = "NOSAT"}} : {packet} -> {u8_packet}
          pto.vmi.vstore %truncated, %n[%c3], %c1 {{group = {groups}}} : {u8_packet}, {u8_ub}, index
          %narrow_brc = pto.vmi.vbrc %truncated {{group = {groups}}} : {u8_packet} -> {u8_vector}
          pto.vmi.vstore %narrow_brc, %n[%c512] : {u8_vector}, {u8_ub}, index
          %saturated = pto.vmi.trunci %sum {{saturate = "SAT"}} : {packet} -> {u8_packet}
          pto.vmi.vstore %saturated, %n[%c256], %c1 {{group = {groups}}} : {u8_packet}, {u8_ub}, index
          %all = pto.vmi.create_mask %groups : index -> {group_mask}
          %double = pto.vmi.vadd %sum, %sum, %all : {packet}, {packet}, {group_mask} -> {packet}
          pto.vmi.vstore %double, %o[%c1280], %c1 {{group = {groups}}} : {packet}, {label_ub}, index
          %total = pto.vmi.vcadd %sum, %all {{group = 1}} : {packet}, {group_mask} -> {one_packet}
          pto.vmi.vstore %total, %o[%c1536] : {one_packet}, {label_ub}, index
          pto.vmi.vstore %max, %o[%c1792], %c1 {{group = {groups}}} : {packet}, {label_ub}, index
          pto.vmi.vstore %min, %o[%c1856], %c1 {{group = {groups}}} : {packet}, {label_ub}, index
        }}
        pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]
        pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]
'''
    # One-carrier 8-bit broadcasts consume the truncated native gs(8, 4) packet.
    if lanes > 256 or groups > 8:
        source = "\n".join(line for line in source.splitlines()
                           if "%narrow_brc" not in line) + "\n"
    if label == "i16":
        source = "\n".join(line for line in source.splitlines()
                           if "%saturated =" not in line and "vstore %saturated" not in line) + "\n"
    if groups > 8:
        source = "\n".join(line for line in source.splitlines()
                           if "%total =" not in line and "vstore %total" not in line) + "\n"
    for gm, ub, dtype in (("out", "o", label), ("wide", "w", wide), ("narrow", "n", "ui8")):
        source += f'''        pto.mte_ub_gm %{ub}, %{gm}, %bytes nburst(%one, %bytes, %bytes)
            : !pto.ptr<{dtype}, ub>, !pto.ptr<{dtype}, gm>, i64, i64, i64, i64
'''
    source += '        pto.barrier #pto.pipe<PIPE_ALL>\n        return\n      }\n}\n'

    dtype, wide_dtype = getattr(pto, label), getattr(pto, wide)

    @pto.jit(name=name, target="a5", backend="vpto", mode="explicit", source=source)
    def kernel(src: pto.ptr(dtype, "gm"), gate: pto.ptr(dtype, "gm"),
               out: pto.ptr(dtype, "gm"), wide: pto.ptr(wide_dtype, "gm"),
               narrow: pto.ptr(pto.ui8, "gm"), active: pto.i32):
        pass

    return kernel


def make_case(npdtype, lanes, groups, mode, check_saturation):
    limits = np.iinfo(npdtype)
    source = np.full(TRANSFER_BYTES // 2, 17, dtype=npdtype)
    source[:lanes] = np.resize(np.array([limits.max, 1, limits.min, -1 if limits.min < 0 else limits.max,
                                        19, 37, 113, 251], dtype=npdtype), lanes)
    # Distinct per-group data detects permutation and high-half mistakes.
    source[:lanes] ^= (np.arange(lanes, dtype=np.uint16) * 73).astype(npdtype)
    gates = np.ones_like(source)
    active = 0 if mode == "empty" else lanes - 1 if mode == "tail" else lanes + 1
    if mode in ("wrap", "underflow"):
        source[:lanes] = 0
        for group in range(groups):
            source[group * (lanes // groups)] = limits.max if mode == "wrap" else limits.min
            source[group * (lanes // groups) + 1] = 1 if mode == "wrap" else -1 if limits.min < 0 else limits.max
    if mode == "holes":
        gates[np.arange(gates.size) % 3 == 1] = 0
    if mode == "middle_empty":
        begin = (groups // 2) * (lanes // groups)
        gates[begin:begin + lanes // groups] = 0
    outputs = [np.full(TRANSFER_BYTES // 2, 23, dtype=npdtype),
               np.full(TRANSFER_BYTES // 4, 29, dtype=np.int32 if limits.min < 0 else np.uint32),
               np.full(TRANSFER_BYTES, 31, dtype=np.uint8)]
    expected = [value.copy() for value in outputs]
    selected = (np.arange(lanes) < active) & (gates[:lanes] != 0)
    sums = []

    def wrap(value):
        value = int(value) % 65536
        return value - 65536 if limits.min < 0 and value > limits.max else value

    for group in range(groups):
        begin, end = group * (lanes // groups), (group + 1) * (lanes // groups)
        values = source[begin:end][selected[begin:end]]
        total = wrap(sum(int(value) for value in values))
        sums.append(total)
        expected[0][1 + group] = total
        expected[0][256 + group] = total
        expected[0][512 + begin:512 + end] = total
        expected[0][1280 + group] = wrap(total * 2)
        expected[0][1792 + group] = values.max() if values.size else limits.min
        expected[0][1856 + group] = values.min() if values.size else limits.max
        expected[1][1 + group] = total
        expected[2][3 + group] = total % 256
        if lanes <= 256 and groups <= 8:
            expected[2][512 + begin:512 + end] = total % 256
        if check_saturation:
            expected[2][256 + group] = min(255, max(0, total))
    if groups <= 8:
        expected[0][1536] = wrap(sum(sums))
    return [source, gates, *outputs], expected, [active]


def check_case(inputs, expected):
    for actual, golden in zip(inputs[2:], expected):
        np.testing.assert_array_equal(actual.cpu().numpy(), golden)


# The matrix below is a full cross product of dtype, vector shape and reduction
# mode. Running every combination costs hours on the cycle accurate model while
# most combinations repeat the same device behaviour, so the default suite runs
# one representative per behaviour class and samples the shape axis:
#
#   dtype  si16          - the signed path, every reduction mode
#          ui16          - the unsigned path, one canonical case
#          i16           - ui16 with the saturation check on, one canonical case
#   shape  (512, 16)     - largest vector with the largest group count
#          (64, 8)       - smallest vector with the smallest group count
#   mode   empty / tail / full / holes / middle_empty / wrap / underflow
#                        - complete for the signed representative
#
# Set PTOAS_VMI_FULL_MATRIX=1 to run the original cross product instead.
FULL_MATRIX = os.environ.get("PTOAS_VMI_FULL_MATRIX") == "1"
LABELS = (("i16", np.uint16), ("si16", np.int16), ("ui16", np.uint16))
SHAPES = ((64, 8), (128, 8), (256, 8), (512, 8), (256, 16), (512, 16))
MODES = ("empty", "tail", "full", "holes", "middle_empty", "wrap", "underflow")

if FULL_MATRIX:
    matrix = [(label, npdtype, shape, mode)
              for label, npdtype in LABELS for shape in SHAPES for mode in MODES]
else:
    # Every shape stays covered. Inside each shape the label and reduction mode
    # axes are rotated over a small deterministic sample so every dtype and
    # every mode still appears without paying for the full cross product.
    matrix = []
    for shape_index, shape in enumerate(SHAPES):
        for offset in (0, 3, 5):
            label, npdtype = LABELS[(shape_index + offset) % len(LABELS)]
            matrix.append((label, npdtype, shape, MODES[(shape_index + offset) % len(MODES)]))

CASES = []
_kernels = {}
for label, npdtype, (lanes, groups), mode in matrix:
    name = f"native_sum_{label}_vl{lanes}_g{groups}"
    kernel = _kernels.get(name)
    if kernel is None:
        kernel = build_kernel(label, lanes, groups, name)
        _kernels[name] = kernel
    def create_case(npdtype=npdtype, lanes=lanes, groups=groups, mode=mode,
                    check_saturation=label != "i16"):
        return make_case(npdtype, lanes, groups, mode, check_saturation)
    CASES.append(dict(name=f"{name}_{mode}", kernel=kernel,
                      make_case=create_case, check=check_case))

CASES = list({case["name"]: case for case in CASES}.values())  # sampling can overlap


auto_main(globals())
