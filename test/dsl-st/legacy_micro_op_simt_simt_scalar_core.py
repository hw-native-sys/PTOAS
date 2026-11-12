#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Migrated from test/vpto/cases/micro-op/simt/simt-scalar-core."""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

from common import auto_main
from ptodsl import pto

KERNEL_SOURCE = (
    'module attributes {pto.target_arch = "a5", pto.kernel_kind = #pto.kernel_kind<vector>} {\n  func.func @simt_'
    'scalar_core_kernel(%arg0: !pto.ptr<i32, gm>) attributes {pto.aicore} {\n    %c0_i64 = arith.constant 0 : i64'
    '\n    %c32_i64 = arith.constant 32 : i64\n    %c128_i64 = arith.constant 128 : i64\n    %dim_z = arith.const'
    'ant 1 : i32\n    %dim_y = arith.constant 1 : i32\n    %dim_x = arith.constant 32 : i32\n\n    %ub_out = pto.'
    'castptr %c0_i64 : i64 -> !pto.ptr<i32, ub>\n\n    pto.simt_launch @simt_scalar_core_body<<<%dim_x, %dim_y, %'
    'dim_z>>>(%ub_out) : (!pto.ptr<i32, ub>) -> ()\n\n    pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n    p'
    'to.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n    pto.mte_ub_gm %ub_out, %arg0, %c128_i64\n      nburst('
    '%c32_i64, %c128_i64, %c128_i64)\n      : !pto.ptr<i32, ub>, !pto.ptr<i32, gm>, i64, i64, i64, i64\n    pto.b'
    'arrier #pto.pipe<PIPE_ALL>\n    return\n  }\n\n  func.func @simt_scalar_core_body(%dst: !pto.ptr<i32, ub>) a'
    'ttributes {pto.simt_entry} {\n    %c0_i32 = arith.constant 0 : i32\n    %c1_i32 = arith.constant 1 : i32\n  '
    '  %c2_i32 = arith.constant 2 : i32\n    %c3_i32 = arith.constant 3 : i32\n    %c32_i32 = arith.constant 32 :'
    ' i32\n    %c100_i32 = arith.constant 100 : i32\n    %c_fields_i32 = arith.constant 32 : i32\n\n    %tid_x = '
    'pto.get_tid_x : i32\n    %laneid = pto.get_laneid : i32\n    %block_dim_x = pto.get_block_dim_x : i32\n    %'
    'block_dim_y = pto.get_block_dim_y : i32\n    %block_dim_z = pto.get_block_dim_z : i32\n    %grid_dim_x = pto'
    '.get_grid_dim_x : i32\n    %grid_dim_y = pto.get_grid_dim_y : i32\n    %grid_dim_z = pto.get_grid_dim_z : i3'
    '2\n    %block_idx_x = pto.get_block_idx_x : i32\n\n    %lanemask_eq = pto.get_lanemask_eq : i32\n    %lanema'
    'sk_le = pto.get_lanemask_le : i32\n    %lanemask_lt = pto.get_lanemask_lt : i32\n    %lanemask_ge = pto.get_'
    'lanemask_ge : i32\n    %lanemask_gt = pto.get_lanemask_gt : i32\n\n    %lane_low_bit = arith.andi %laneid, %'
    'c1_i32 : i32\n    %is_even = arith.cmpi eq, %lane_low_bit, %c0_i32 : i32\n    %vote_all = pto.vote_all %is_e'
    'ven : i1 -> i1\n    %vote_any = pto.vote_any %is_even : i1 -> i1\n    %vote_uni = pto.vote_uni %is_even : i1'
    ' -> i1\n    %vote_all_i32 = arith.extui %vote_all : i1 to i32\n    %vote_any_i32 = arith.extui %vote_any : i'
    '1 to i32\n    %vote_uni_i32 = arith.extui %vote_uni : i1 to i32\n    %ballot = pto.vote_ballot %is_even : i1'
    ' -> i32\n\n    %value = arith.addi %laneid, %c100_i32 : i32\n    %shfl_idx = pto.shuffle_idx %value, %c3_i32'
    ' : i32, i32 -> i32\n    %shfl_up = pto.shuffle_up %value, %c2_i32 : i32, i32 -> i32\n    %shfl_down = pto.sh'
    'uffle_down %value, %c2_i32 : i32, i32 -> i32\n    %shfl_bfly = pto.shuffle_bfly %value, %c1_i32 : i32, i32 -'
    '> i32\n\n    %lane_plus_one = arith.addi %laneid, %c1_i32 : i32\n    %redux_add = pto.redux_addi %lane_plus_'
    'one : i32 -> i32\n    %redux_max = pto.redux_maxi %lane_plus_one signed : i32 -> i32\n    %redux_min = pto.r'
    'edux_mini %lane_plus_one signed : i32 -> i32\n    %one_f32 = arith.constant 1.000000e+00 : f32\n    %redux_f'
    '32 = pto.redux_addf %one_f32 : f32 -> f32\n    %redux_f32_i32 = pto.ftoi %redux_f32 signed round(r) sat : f3'
    '2 -> i32\n\n    %base_i32 = arith.muli %laneid, %c_fields_i32 : i32\n    %base = arith.index_castui %base_i3'
    '2 : i32 to index\n\n    pto.store %tid_x, %dst[%base] : !pto.ptr<i32, ub>, index, i32\n    %idx1_i32 = arith.addi %'
    'base_i32, %c1_i32 : i32\n    %idx1 = arith.index_castui %idx1_i32 : i32 to index\n    pto.store %laneid, %ds'
    't[%idx1] : !pto.ptr<i32, ub>, index, i32\n    %idx2_i32 = arith.addi %base_i32, %c2_i32 : i32\n    %idx2 = arith.in'
    'dex_castui %idx2_i32 : i32 to index\n    pto.store %block_dim_x, %dst[%idx2] : !pto.ptr<i32, ub>, index, i32\n    %'
    'idx3_i32 = arith.addi %base_i32, %c3_i32 : i32\n    %idx3 = arith.index_castui %idx3_i32 : i32 to index\n   '
    ' pto.store %block_dim_y, %dst[%idx3] : !pto.ptr<i32, ub>, index, i32\n    %c4_i32 = arith.constant 4 : i32\n    %id'
    'x4_i32 = arith.addi %base_i32, %c4_i32 : i32\n    %idx4 = arith.index_castui %idx4_i32 : i32 to index\n    p'
    'to.store %block_dim_z, %dst[%idx4] : !pto.ptr<i32, ub>, index, i32\n    %c5_i32 = arith.constant 5 : i32\n    %idx5'
    '_i32 = arith.addi %base_i32, %c5_i32 : i32\n    %idx5 = arith.index_castui %idx5_i32 : i32 to index\n    pto'
    '.store %grid_dim_x, %dst[%idx5] : !pto.ptr<i32, ub>, index, i32\n    %c6_i32 = arith.constant 6 : i32\n    %idx6_i3'
    '2 = arith.addi %base_i32, %c6_i32 : i32\n    %idx6 = arith.index_castui %idx6_i32 : i32 to index\n    pto.st'
    'ore %grid_dim_y, %dst[%idx6] : !pto.ptr<i32, ub>, index, i32\n    %c7_i32 = arith.constant 7 : i32\n    %idx7_i32 ='
    ' arith.addi %base_i32, %c7_i32 : i32\n    %idx7 = arith.index_castui %idx7_i32 : i32 to index\n    pto.store'
    ' %grid_dim_z, %dst[%idx7] : !pto.ptr<i32, ub>, index, i32\n    %c8_i32 = arith.constant 8 : i32\n    %idx8_i32 = ar'
    'ith.addi %base_i32, %c8_i32 : i32\n    %idx8 = arith.index_castui %idx8_i32 : i32 to index\n    pto.store %b'
    'lock_idx_x, %dst[%idx8] : !pto.ptr<i32, ub>, index, i32\n    %c9_i32 = arith.constant 9 : i32\n    %idx9_i32 = arit'
    'h.addi %base_i32, %c9_i32 : i32\n    %idx9 = arith.index_castui %idx9_i32 : i32 to index\n    pto.store %lan'
    'emask_eq, %dst[%idx9] : !pto.ptr<i32, ub>, index, i32\n    %c10_i32 = arith.constant 10 : i32\n    %idx10_i32 = ari'
    'th.addi %base_i32, %c10_i32 : i32\n    %idx10 = arith.index_castui %idx10_i32 : i32 to index\n    pto.store '
    '%lanemask_le, %dst[%idx10] : !pto.ptr<i32, ub>, index, i32\n    %c11_i32 = arith.constant 11 : i32\n    %idx11_i32 '
    '= arith.addi %base_i32, %c11_i32 : i32\n    %idx11 = arith.index_castui %idx11_i32 : i32 to index\n    pto.s'
    'tore %lanemask_lt, %dst[%idx11] : !pto.ptr<i32, ub>, index, i32\n    %c12_i32 = arith.constant 12 : i32\n    %idx12'
    '_i32 = arith.addi %base_i32, %c12_i32 : i32\n    %idx12 = arith.index_castui %idx12_i32 : i32 to index\n    '
    'pto.store %lanemask_ge, %dst[%idx12] : !pto.ptr<i32, ub>, index, i32\n    %c13_i32 = arith.constant 13 : i32\n    %'
    'idx13_i32 = arith.addi %base_i32, %c13_i32 : i32\n    %idx13 = arith.index_castui %idx13_i32 : i32 to index'
    '\n    pto.store %lanemask_gt, %dst[%idx13] : !pto.ptr<i32, ub>, index, i32\n    %c14_i32 = arith.constant 14 : i32'
    '\n    %idx14_i32 = arith.addi %base_i32, %c14_i32 : i32\n    %idx14 = arith.index_castui %idx14_i32 : i32 to'
    ' index\n    pto.store %vote_all_i32, %dst[%idx14] : !pto.ptr<i32, ub>, index, i32\n    %c15_i32 = arith.constant 15'
    ' : i32\n    %idx15_i32 = arith.addi %base_i32, %c15_i32 : i32\n    %idx15 = arith.index_castui %idx15_i32 : '
    'i32 to index\n    pto.store %vote_any_i32, %dst[%idx15] : !pto.ptr<i32, ub>, index, i32\n    %c16_i32 = arith.const'
    'ant 16 : i32\n    %idx16_i32 = arith.addi %base_i32, %c16_i32 : i32\n    %idx16 = arith.index_castui %idx16_'
    'i32 : i32 to index\n    pto.store %vote_uni_i32, %dst[%idx16] : !pto.ptr<i32, ub>, index, i32\n    %c17_i32 = arith'
    '.constant 17 : i32\n    %idx17_i32 = arith.addi %base_i32, %c17_i32 : i32\n    %idx17 = arith.index_castui %'
    'idx17_i32 : i32 to index\n    pto.store %ballot, %dst[%idx17] : !pto.ptr<i32, ub>, index, i32\n    %c18_i32 = arith'
    '.constant 18 : i32\n    %idx18_i32 = arith.addi %base_i32, %c18_i32 : i32\n    %idx18 = arith.index_castui %'
    'idx18_i32 : i32 to index\n    pto.store %shfl_idx, %dst[%idx18] : !pto.ptr<i32, ub>, index, i32\n    %c19_i32 = ari'
    'th.constant 19 : i32\n    %idx19_i32 = arith.addi %base_i32, %c19_i32 : i32\n    %idx19 = arith.index_castui'
    ' %idx19_i32 : i32 to index\n    pto.store %shfl_up, %dst[%idx19] : !pto.ptr<i32, ub>, index, i32\n    %c20_i32 = ar'
    'ith.constant 20 : i32\n    %idx20_i32 = arith.addi %base_i32, %c20_i32 : i32\n    %idx20 = arith.index_castu'
    'i %idx20_i32 : i32 to index\n    pto.store %shfl_down, %dst[%idx20] : !pto.ptr<i32, ub>, index, i32\n    %c21_i32 ='
    ' arith.constant 21 : i32\n    %idx21_i32 = arith.addi %base_i32, %c21_i32 : i32\n    %idx21 = arith.index_ca'
    'stui %idx21_i32 : i32 to index\n    pto.store %shfl_bfly, %dst[%idx21] : !pto.ptr<i32, ub>, index, i32\n    %c22_i3'
    '2 = arith.constant 22 : i32\n    %idx22_i32 = arith.addi %base_i32, %c22_i32 : i32\n    %idx22 = arith.index'
    '_castui %idx22_i32 : i32 to index\n    pto.store %redux_add, %dst[%idx22] : !pto.ptr<i32, ub>, index, i32\n    %c23'
    '_i32 = arith.constant 23 : i32\n    %idx23_i32 = arith.addi %base_i32, %c23_i32 : i32\n    %idx23 = arith.in'
    'dex_castui %idx23_i32 : i32 to index\n    pto.store %redux_max, %dst[%idx23] : !pto.ptr<i32, ub>, index, i32\n    %'
    'c24_i32 = arith.constant 24 : i32\n    %idx24_i32 = arith.addi %base_i32, %c24_i32 : i32\n    %idx24 = arith'
    '.index_castui %idx24_i32 : i32 to index\n    pto.store %redux_min, %dst[%idx24] : !pto.ptr<i32, ub>, index, i32\n  '
    '  %c25_i32 = arith.constant 25 : i32\n    %idx25_i32 = arith.addi %base_i32, %c25_i32 : i32\n    %idx25 = ar'
    'ith.index_castui %idx25_i32 : i32 to index\n    pto.store %redux_f32_i32, %dst[%idx25] : !pto.ptr<i32, ub>, index, '
    'i32\n    return\n  }\n}\n'
)

def _golden_scope():
    _files = {}

    output_dir = None

    # Copyright (c) 2026 Huawei Technologies Co., Ltd.
    # This program is free software, you can redistribute it and/or modify it under the terms and conditions of
    # CANN Open Software License Agreement Version 2.0 (the "License").
    # Please refer to the License for details. You may not use this file except in compliance with the License.
    # THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
    # INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
    # See LICENSE in the root of the software repository for the full text of the License.

    import argparse
    from pathlib import Path

    import numpy as np

    LANES = 32
    FIELDS = 32
    ELEMS = LANES * FIELDS


    def i32_bits(value: int) -> np.int32:
        value &= 0xFFFFFFFF
        if value >= 0x80000000:
            value -= 0x100000000
        return np.int32(value)


    def _generate(seed: int) -> dict:
        v1 = np.full(ELEMS, -1, dtype=np.int32)
        golden_v1 = np.zeros(ELEMS, dtype=np.int32)
        for lane in range(LANES):
            base = lane * FIELDS
            golden_v1[base + 0] = lane
            golden_v1[base + 1] = lane
            golden_v1[base + 2] = 32
            golden_v1[base + 3] = 1
            golden_v1[base + 4] = 1
            # Grid dims: the grid is one-dimensional over blocks, so x is the
            # block count (1 here) and y/z are degenerate.
            golden_v1[base + 5] = 1
            golden_v1[base + 6] = 1
            golden_v1[base + 7] = 1
            golden_v1[base + 8] = 0
            golden_v1[base + 9] = i32_bits(1 << lane)
            golden_v1[base + 10] = i32_bits((1 << (lane + 1)) - 1) if lane < 31 else np.int32(-1)
            golden_v1[base + 11] = i32_bits((1 << lane) - 1)
            golden_v1[base + 12] = i32_bits(0xFFFFFFFF << lane)
            golden_v1[base + 13] = i32_bits(0xFFFFFFFF << (lane + 1)) if lane < 31 else np.int32(0)
            golden_v1[base + 14] = 0
            golden_v1[base + 15] = 1
            golden_v1[base + 16] = 0
            golden_v1[base + 17] = i32_bits(0x55555555)
            golden_v1[base + 18] = 103
            golden_v1[base + 19] = 100 + (lane - 2 if lane >= 2 else lane)
            golden_v1[base + 20] = 100 + (lane + 2 if lane <= 29 else lane)
            golden_v1[base + 21] = 100 + (lane ^ 1)
            golden_v1[base + 22] = 528
            golden_v1[base + 23] = 32
            golden_v1[base + 24] = 1
            golden_v1[base + 25] = 32
        _files["v1.bin"] = v1
        _files["golden_v1.bin"] = golden_v1


    def main():
        parser = argparse.ArgumentParser()
        parser.add_argument("--output-dir", type=Path, default=Path("."))
        args = parser.parse_args()
        generate(args.output_dir)


    return locals()

def _compare_scope():
    # Copyright (c) 2026 Huawei Technologies Co., Ltd.
    # This program is free software, you can redistribute it and/or modify it under the terms and conditions of
    # CANN Open Software License Agreement Version 2.0 (the "License").
    # Please refer to the License for details. You may not use this file except in compliance with the License.
    # THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
    # INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
    # See LICENSE in the root of the software repository for the full text of the License.

    import os
    import sys

    import numpy as np


    def main():
        strict = os.getenv("COMPARE_STRICT", "1") != "0"
        golden = np.fromfile("golden_v1.bin", dtype=np.int32)
        out = np.fromfile("v1.bin", dtype=np.int32)
        ok = golden.shape == out.shape and np.array_equal(golden, out)
        if not ok:
            idxs = np.nonzero(golden != out)[0]
            idx = int(idxs[0]) if idxs.size else 0
            print(
                f"[ERROR] mismatch at idx={idx}, golden={int(golden[idx])}, out={int(out[idx])}"
            )
            if strict:
                sys.exit(2)
        print("[INFO] compare passed" if ok else "[WARN] compare failed (non-gating)")


    return locals()

HELPERS = {}
SEED_EXPR = None
COMPARE_PAIRS = []

NUMPY_DTYPE = {
    'i8': 'int8',
    'i16': 'int16',
    'i32': 'int32',
    'i64': 'int64',
    'si8': 'int8',
    'si16': 'int16',
    'si32': 'int32',
    'si64': 'int64',
    'ui8': 'uint8',
    'ui16': 'uint16',
    'ui32': 'uint32',
    'ui64': 'uint64',
    'f16': 'float16',
    'f32': 'float32',
    'f64': 'float64',
    'bf16': 'uint16',
    'hif8': 'uint8',
    'f8E4M3FN': 'uint8',
    'f8E5M2': 'uint8',
    'f8e4m3': 'uint8',
    'f8e5m2': 'uint8',
    'f8e8m0': 'uint8',
    'f8E8M0FNU': 'uint8',
    'f4E2M1x2': 'uint8',
    'f4E1M2x2': 'uint8',
    'hif8x2': 'uint16',
    'f8e4m3x2': 'uint16',
    'f8e5m2x2': 'uint16',
    'f8e4m3x4': 'uint32',
    'f8e4m3x8': 'uint64',
    'f8e5m2x4': 'uint32',
    'f8e5m2x8': 'uint64',
    'f16x2': 'uint32',
    'bf16x2': 'uint32',
    'i8x2': 'uint16',
    'i16x2': 'uint32',
    'i32x2': 'uint64',
    'f32x2': 'uint64'
}


def _as_buffer_array(value, numpy_dtype):
    """Match a golden array to the raw bytes a kernel element type expects.

    A golden may build its array with ml_dtypes types (bfloat16) while the
    kernel argument is a plain numpy type (uint16); torch.from_numpy rejects
    the former, so reinterpret the bytes when the element widths agree.
    """
    array = np.asarray(value)
    want = np.dtype(numpy_dtype)
    if array.dtype == want:
        return array
    # The golden bytes are the device buffer; reinterpret them whenever the
    # byte count divides evenly so a narrow golden (for example uint8 init or
    # an f16 source) still fills the element type the kernel argument expects
    # instead of shrinking or doubling the buffer through astype.
    flat = np.ascontiguousarray(array).reshape(-1)
    if flat.nbytes % want.itemsize == 0:
        return flat.view(want)
    return array.astype(want)

BUFFERS = [
    (
        'v1',
        'i32',
        'arg0'
    )
]
SCALARS = []
SCALAR_SOURCES = []
SCALAR_DTYPES = []


@pto.jit(
    name='simt_scalar_core_kernel',
    target="a5",
    backend="vpto",
    mode="explicit",
    source=KERNEL_SOURCE,
)
def kernel(
    arg0: pto.ptr(pto.i32, "gm"),
):
    pass


class _TrackedArray(np.ndarray):
    """An array that records the positions written through __setitem__."""

    def __new__(cls, array):
        obj = np.asarray(array).view(cls)
        obj._write_mask = np.zeros(obj.shape, dtype=bool)
        return obj

    def __array_finalize__(self, source):
        if source is None:
            return
        self._write_mask = getattr(source, "_write_mask", None)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        mask = getattr(self, "_write_mask", None)
        if mask is None or mask.shape != self.shape:
            mask = np.zeros(self.shape, dtype=bool)
            self._write_mask = mask
        mask[key] = True


def _materialize():
    """Run the case's own golden generator in memory; no .bin files.

    The golden's output buffers are created through the numpy constructors and
    then filled in place, so wrapping those constructors records exactly which
    positions the golden defines; the rest are don't care for the compare.
    """
    trackers = {name: getattr(np, name) for name in ("zeros", "full", "empty", "ones")}

    building = [False]

    def _track(name):
        original = trackers[name]

        def wrapped(*args, **kwargs):
            if building[0]:
                return original(*args, **kwargs)
            building[0] = True
            try:
                tracked = _TrackedArray(original(*args, **kwargs))
                if name != "empty":
                    # zeros/full/ones initialize every element, so the whole
                    # buffer is defined and compared; only np.empty leaves
                    # elements undefined, where the __setitem__ trace defines
                    # the valid mask.
                    tracked._write_mask = np.ones(tracked.shape, dtype=bool)
                return tracked
            finally:
                building[0] = False

        return wrapped

    for name in trackers:
        setattr(np, name, _track(name))
    try:
        namespace = _golden_scope()
        if "_generate" in namespace:
            seed = namespace[SEED_EXPR] if SEED_EXPR else 0
            namespace["_generate"](seed)
        else:
            sys.argv = ["golden.py"]
            namespace["main"]()
    finally:
        for name, original in trackers.items():
            setattr(np, name, original)
    produced = namespace["_files"]
    masks = {}
    for key, value in produced.items():
        if not key.startswith("golden_"):
            continue
        mask = getattr(value, "_write_mask", None)
        if mask is not None and np.asarray(mask).size and bool(np.asarray(mask).any()):
            masks[key] = np.asarray(mask).reshape(-1).copy()
        else:
            # a computed output defines every element; there is no partial
            # in-place write to record, so the whole file is compared.
            masks[key] = np.ones(np.asarray(value).size, dtype=bool)
    if masks:
        produced["@write_mask"] = masks
    return produced


def _apply_write_masks(files, produced):
    """Restrict each golden/output comparison to the golden's written set."""
    masks = produced.get("@write_mask") if hasattr(produced, "get") else None
    if not masks:
        return
    for golden_key, mask in masks.items():
        if not golden_key.startswith("golden_"):
            continue
        output_key = golden_key[len("golden_"):]
        if golden_key not in files or output_key not in files:
            continue
        golden = np.asarray(files[golden_key])
        output = np.asarray(files[output_key]).reshape(-1)
        flat = golden.reshape(-1)
        written = np.asarray(mask, dtype=bool).reshape(-1)
        size = min(flat.size, output.size, written.size)
        if size == 0 or not written[:size].any():
            continue
        adjusted = flat.copy()
        donotcare = ~written[:size]
        if output.dtype != flat.dtype and output.dtype.itemsize == flat.dtype.itemsize:
            # Same byte width but a different element type (for example the
            # kernel typed the buffer f16 while the golden built uint16): copy
            # the raw bytes so the donot-care fill stays bit-exact instead of
            # rounding through a numeric cast such as f16(-1.0) -> 0xffff.
            output = np.ascontiguousarray(output).view(flat.dtype)
        adjusted[:size][donotcare] = output[:size][donotcare]
        files[golden_key] = adjusted.reshape(golden.shape)


def _run_compare(files):
    """Run the case's own compare.py against in-memory arrays."""
    with tempfile.TemporaryDirectory(
            prefix="legacy-helpers-", dir=os.environ.get("TMPDIR")) as _helpers:
        _run_compare_in(files, Path(_helpers))


def _run_compare_in(files, helper_dir):
    for name, text in HELPERS.items():
        (helper_dir / name).write_text(text)

    real_fromfile, real_exists, real_argv = np.fromfile, os.path.exists, sys.argv
    real_exists_path = Path.exists
    real_environ, real_path = dict(os.environ), list(sys.path)

    def _fromfile(path, dtype=None, count=-1, offset=0, *args, **kwargs):
        name = os.path.basename(str(path))
        if name not in files:
            raise FileNotFoundError(name)
        # The legacy compare read the raw .bin bytes; reinterpret the stored
        # array's bytes instead of casting, so a deep-merged buffer written
        # through one element type still reads back as the type compare asks
        # for. count/offset mirror np.fromfile's element prefix semantics.
        array = np.asarray(files[name])
        data = array.tobytes()
        if offset:
            data = data[int(offset):]
        if dtype is None:
            dtype = array.dtype
        itemsize = np.dtype(dtype).itemsize
        available = len(data) // itemsize
        if count is None or int(count) < 0 or int(count) > available:
            count = available
        return np.frombuffer(data, dtype=dtype, count=count)

    np.fromfile = _fromfile
    os.path.exists = lambda p: os.path.basename(str(p)) in files or real_exists(p)
    Path.exists = lambda self: os.path.basename(str(self)) in files or real_exists_path(self)
    sys.argv = ["compare.py"]
    sys.path.insert(0, str(helper_dir))
    os.environ["COMPARE_STRICT"] = "1"
    try:
        namespace = _compare_scope()
        if "main" in namespace:
            namespace["main"]()
    except SystemExit as exc:
        if exc.code not in (0, None):
            raise AssertionError("compare.py exited with %r" % (exc.code,))
    finally:
        np.fromfile, os.path.exists, Path.exists, sys.argv = (
            real_fromfile, real_exists, real_exists_path, real_argv)
        os.environ.clear()
        os.environ.update(real_environ)
        sys.path[:] = real_path


def make_case():
    import torch
    produced = _materialize()
    output_to_golden = {out: gold for gold, out in COMPARE_PAIRS}
    arrays = []
    for name, ty, _slot in BUFFERS:
        if name.startswith("@same:"):
            arrays.append(np.asarray(produced["%s.bin" % name[len("@same:"):]]))
            continue
        if name.startswith("@scratch:"):
            arrays.append(np.empty_like(np.asarray(produced["%s.bin" % name[len("@scratch:"):]]),
                                        dtype=np.dtype(NUMPY_DTYPE[ty])))
            continue
        key = "%s.bin" % name
        if key in produced:
            arrays.append(_as_buffer_array(produced[key], NUMPY_DTYPE[ty]))
            continue
        golden_key = output_to_golden.get(key)
        if golden_key is None or golden_key not in produced:
            raise RuntimeError("golden did not produce %s" % key)
        arrays.append(np.zeros_like(np.asarray(produced[golden_key]),
                                    dtype=np.dtype(NUMPY_DTYPE[ty])))
    values = list(SCALARS)
    for _source, _ty in zip(SCALAR_SOURCES, SCALAR_DTYPES):
        _array = np.asarray(produced["%s.bin" % _source]).reshape(-1)
        values.append(float(_array[0]) if _ty == "f32" else int(_array[0]))
    if GRID_FROM_VALUES is not None:
        CASES[0]["grid"] = GRID_FROM_VALUES(values)
    return arrays, produced, values


def check_case(device_inputs, produced):
    if COMPARE_PAIRS:
        files = {gold: np.asarray(produced[gold]) for gold, _out in COMPARE_PAIRS if gold in produced}
    else:
        # no explicit pairs: the case compares whole files, so expose every
        # array the golden built (its .tofile names became dictionary keys).
        files = {name: np.asarray(value) for name, value in produced.items()
                 if not name.startswith("@")}
    if not files:
        raise AssertionError("golden.py produced no golden output")
    for _source in SCALAR_SOURCES:
        _key = "%s.bin" % _source
        if _key in produced:
            files[_key] = np.asarray(produced[_key])
    for (name, _ty, _slot), tensor in zip(BUFFERS, device_inputs):
        if not name.startswith("@"):
            files["%s.bin" % name] = tensor.cpu().numpy()
    _apply_write_masks(files, produced)
    _run_compare(files)


GRID_FROM_VALUES = None


CASES = [dict(
    name='legacy_micro_op_simt_simt_scalar_core',
    grid=1,
    kernel=kernel,
    make_case=make_case,
    check=check_case,
)]

auto_main(globals())
