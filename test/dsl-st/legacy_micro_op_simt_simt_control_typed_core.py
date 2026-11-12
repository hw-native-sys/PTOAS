#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Migrated from test/vpto/cases/micro-op/simt/simt-control-typed-core."""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

from common import auto_main
from ptodsl import pto

KERNEL_SOURCE = (
    'module attributes {pto.target_arch = "a5", pto.kernel_kind = #pto.kernel_kind<vector>} {\n  func.func @simt_'
    'control_typed_core_kernel(%arg0: !pto.ptr<i32, gm>) attributes {pto.aicore} {\n    %c0_i64 = arith.constant '
    '0 : i64\n    %c128_i64 = arith.constant 128 : i64\n    %c32_i64 = arith.constant 32 : i64\n    %dim_z = arit'
    'h.constant 1 : i32\n    %dim_y = arith.constant 1 : i32\n    %dim_x = arith.constant 32 : i32\n\n    %ub_out'
    ' = pto.castptr %c0_i64 : i64 -> !pto.ptr<i32, ub>\n    pto.simt_launch @simt_control_typed_core_body<<<%dim_'
    'x, %dim_y, %dim_z>>>(%ub_out) : (!pto.ptr<i32, ub>) -> ()\n\n    pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_'
    'ID0"]\n    pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n    pto.mte_ub_gm %ub_out, %arg0, %c128_i64\n '
    '     nburst(%c32_i64, %c128_i64, %c128_i64)\n      : !pto.ptr<i32, ub>, !pto.ptr<i32, gm>, i64, i64, i64, i6'
    '4\n    pto.barrier #pto.pipe<PIPE_ALL>\n    return\n  }\n\n  func.func @simt_control_typed_core_body(%dst: !'
    'pto.ptr<i32, ub>) attributes {pto.simt_entry} {\n    %c0_i32 = arith.constant 0 : i32\n    %c1_i32 = arith.c'
    'onstant 1 : i32\n    %c2_i32 = arith.constant 2 : i32\n    %c3_i32 = arith.constant 3 : i32\n    %c4_i32 = a'
    'rith.constant 4 : i32\n    %c5_i32 = arith.constant 5 : i32\n    %c6_i32 = arith.constant 6 : i32\n    %c7_i'
    '32 = arith.constant 7 : i32\n    %c8_i32 = arith.constant 8 : i32\n    %c9_i32 = arith.constant 9 : i32\n   '
    ' %c10_i32 = arith.constant 10 : i32\n    %c11_i32 = arith.constant 11 : i32\n    %c12_i32 = arith.constant 1'
    '2 : i32\n    %c13_i32 = arith.constant 13 : i32\n    %c14_i32 = arith.constant 14 : i32\n    %c15_i32 = arit'
    'h.constant 15 : i32\n    %c31_i32 = arith.constant 31 : i32\n    %c32_i32 = arith.constant 32 : i32\n    %c1'
    '00_i32 = arith.constant 100 : i32\n    %c_fields_i32 = arith.constant 20 : i32\n    %c0 = arith.constant 0 :'
    ' index\n    %c1 = arith.constant 1 : index\n    %c2 = arith.constant 2 : index\n    %c3 = arith.constant 3 :'
    ' index\n    %c4 = arith.constant 4 : index\n    %c5 = arith.constant 5 : index\n    %c6 = arith.constant 6 :'
    ' index\n    %c7 = arith.constant 7 : index\n    %c8 = arith.constant 8 : index\n    %c9 = arith.constant 9 :'
    ' index\n    %c10 = arith.constant 10 : index\n    %c11 = arith.constant 11 : index\n    %c12 = arith.constan'
    't 12 : index\n    %c13 = arith.constant 13 : index\n    %c14 = arith.constant 14 : index\n    %c15 = arith.c'
    'onstant 15 : index\n    %c16 = arith.constant 16 : index\n    %c17 = arith.constant 17 : index\n\n    %lanei'
    'd = pto.get_laneid : i32\n    %lane_plus_one = arith.addi %laneid, %c1_i32 : i32\n    %is_last = arith.cmpi '
    'eq, %laneid, %c31_i32 : i32\n    %minus_one = arith.constant -1 : i32\n    %u32_edge = arith.select %is_last'
    ', %minus_one, %laneid : i1, i32\n    %value_i32 = arith.addi %laneid, %c100_i32 : i32\n    %value_i64 = arit'
    'h.extui %value_i32 : i32 to i64\n    %value_f32 = pto.itof %lane_plus_one unsigned round(r) : i32 -> f32\n  '
    '  %value_f16 = pto.itof %lane_plus_one unsigned round(r) : i32 -> f16\n\n    %redux_u32_add = pto.redux_addi'
    ' %lane_plus_one : i32 -> i32\n    %redux_u32_max = pto.redux_maxi %u32_edge unsigned : i32 -> i32\n    %redu'
    'x_u32_min = pto.redux_mini %u32_edge unsigned : i32 -> i32\n    %redux_f32_add = pto.redux_addf %value_f32 :'
    ' f32 -> f32\n    %redux_f32_max = pto.redux_maxf %value_f32 : f32 -> f32\n    %redux_f32_min = pto.redux_min'
    'f %value_f32 : f32 -> f32\n    %redux_f16_add = pto.redux_addf %value_f16 : f16 -> f16\n    %redux_f16_max ='
    ' pto.redux_maxf %value_f16 : f16 -> f16\n    %redux_f16_min = pto.redux_minf %value_f16 : f16 -> f16\n    %r'
    'edux_f32_add_i = pto.ftoi %redux_f32_add signed round(z) sat : f32 -> i32\n    %redux_f32_max_i = pto.ftoi %'
    'redux_f32_max signed round(z) sat : f32 -> i32\n    %redux_f32_min_i = pto.ftoi %redux_f32_min signed round('
    'z) sat : f32 -> i32\n    %redux_f16_add_i = pto.ftoi %redux_f16_add signed round(z) sat : f16 -> i32\n    %r'
    'edux_f16_max_i = pto.ftoi %redux_f16_max signed round(z) sat : f16 -> i32\n    %redux_f16_min_i = pto.ftoi %'
    'redux_f16_min signed round(z) sat : f16 -> i32\n\n    %shfl_idx_w16 = pto.shuffle_idx %value_i32, %c3_i32 {w'
    'idth = 16 : i32} : i32, i32 -> i32\n    %shfl_up_w16 = pto.shuffle_up %value_i32, %c2_i32 {width = 16 : i32}'
    ' : i32, i32 -> i32\n    %shfl_down_i64_w16 = pto.shuffle_down %value_i64, %c2_i32 {width = 16 : i32} : i64, '
    'i32 -> i64\n    %shfl_down_i64_i32 = arith.trunci %shfl_down_i64_w16 : i64 to i32\n    %shfl_bfly_f32_w16 = '
    'pto.shuffle_bfly %value_f32, %c1_i32 {width = 16 : i32} : f32, i32 -> f32\n    %shfl_bfly_f32_i = pto.ftoi %'
    'shfl_bfly_f32_w16 signed round(z) sat : f32 -> i32\n    %shfl_idx_f16_w16 = pto.shuffle_idx %value_f16, %c3_'
    'i32 {width = 16 : i32} : f16, i32 -> f16\n    %shfl_idx_f16_i = pto.ftoi %shfl_idx_f16_w16 signed round(z) s'
    'at : f16 -> i32\n\n    %one_point_75 = arith.constant 1.750000e+00 : f32\n    %round_z = pto.ftoi %one_point'
    '_75 signed round(z) sat : f32 -> i32\n    %round_r = pto.ftoi %one_point_75 signed round(r) sat : f32 -> i32'
    '\n\n    %base_i32 = arith.muli %laneid, %c_fields_i32 : i32\n    %base = arith.index_castui %base_i32 : i32 '
    'to index\n    pto.store %redux_u32_add, %dst[%base] : !pto.ptr<i32, ub>, index, i32\n    %idx1_i32 = arith.addi %ba'
    'se_i32, %c1_i32 : i32\n    %idx1 = arith.index_castui %idx1_i32 : i32 to index\n    pto.store %redux_u32_max'
    ', %dst[%idx1] : !pto.ptr<i32, ub>, index, i32\n    %idx2_i32 = arith.addi %base_i32, %c2_i32 : i32\n    %idx2 = ari'
    'th.index_castui %idx2_i32 : i32 to index\n    pto.store %redux_u32_min, %dst[%idx2] : !pto.ptr<i32, ub>, index, i32'
    '\n    %idx3_i32 = arith.addi %base_i32, %c3_i32 : i32\n    %idx3 = arith.index_castui %idx3_i32 : i32 to ind'
    'ex\n    pto.store %redux_f32_add_i, %dst[%idx3] : !pto.ptr<i32, ub>, index, i32\n    %idx4_i32 = arith.addi %base_i'
    '32, %c4_i32 : i32\n    %idx4 = arith.index_castui %idx4_i32 : i32 to index\n    pto.store %redux_f32_max_i, '
    '%dst[%idx4] : !pto.ptr<i32, ub>, index, i32\n    %idx5_i32 = arith.addi %base_i32, %c5_i32 : i32\n    %idx5 = arith'
    '.index_castui %idx5_i32 : i32 to index\n    pto.store %redux_f32_min_i, %dst[%idx5] : !pto.ptr<i32, ub>, index, i32'
    '\n    %idx6_i32 = arith.addi %base_i32, %c6_i32 : i32\n    %idx6 = arith.index_castui %idx6_i32 : i32 to ind'
    'ex\n    pto.store %redux_f16_add_i, %dst[%idx6] : !pto.ptr<i32, ub>, index, i32\n    %idx7_i32 = arith.addi %base_i'
    '32, %c7_i32 : i32\n    %idx7 = arith.index_castui %idx7_i32 : i32 to index\n    pto.store %redux_f16_max_i, '
    '%dst[%idx7] : !pto.ptr<i32, ub>, index, i32\n    %idx8_i32 = arith.addi %base_i32, %c8_i32 : i32\n    %idx8 = arith'
    '.index_castui %idx8_i32 : i32 to index\n    pto.store %redux_f16_min_i, %dst[%idx8] : !pto.ptr<i32, ub>, index, i32'
    '\n    %idx9_i32 = arith.addi %base_i32, %c9_i32 : i32\n    %idx9 = arith.index_castui %idx9_i32 : i32 to ind'
    'ex\n    pto.store %shfl_idx_w16, %dst[%idx9] : !pto.ptr<i32, ub>, index, i32\n    %idx10_i32 = arith.addi %base_i32'
    ', %c10_i32 : i32\n    %idx10 = arith.index_castui %idx10_i32 : i32 to index\n    pto.store %shfl_up_w16, %ds'
    't[%idx10] : !pto.ptr<i32, ub>, index, i32\n    %idx11_i32 = arith.addi %base_i32, %c11_i32 : i32\n    %idx11 = arit'
    'h.index_castui %idx11_i32 : i32 to index\n    pto.store %shfl_down_i64_i32, %dst[%idx11] : !pto.ptr<i32, ub>, index'
    ', i32\n    %idx12_i32 = arith.addi %base_i32, %c12_i32 : i32\n    %idx12 = arith.index_castui %idx12_i32 : i'
    '32 to index\n    pto.store %shfl_bfly_f32_i, %dst[%idx12] : !pto.ptr<i32, ub>, index, i32\n    %idx13_i32 = arith.a'
    'ddi %base_i32, %c13_i32 : i32\n    %idx13 = arith.index_castui %idx13_i32 : i32 to index\n    pto.store %shf'
    'l_idx_f16_i, %dst[%idx13] : !pto.ptr<i32, ub>, index, i32\n    %idx14_i32 = arith.addi %base_i32, %c14_i32 : i32\n '
    '   %idx14 = arith.index_castui %idx14_i32 : i32 to index\n    pto.store %round_z, %dst[%idx14] : !pto.ptr<i3'
    '2, ub>, index, i32\n    %idx15_i32 = arith.addi %base_i32, %c15_i32 : i32\n    %idx15 = arith.index_castui %idx15_i'
    '32 : i32 to index\n    pto.store %round_r, %dst[%idx15] : !pto.ptr<i32, ub>, index, i32\n    return\n  }\n}\n'
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
    FIELDS = 20
    ELEMS = 1024


    def i32_bits(value: int) -> np.int32:
        value &= 0xFFFFFFFF
        if value >= 0x80000000:
            value -= 0x100000000
        return np.int32(value)


    def _generate(seed: int) -> dict:
        v1 = np.zeros(ELEMS, dtype=np.int32)
        golden_v1 = np.zeros(ELEMS, dtype=np.int32)
        for lane in range(LANES):
            base = lane * FIELDS
            group = 0 if lane < 16 else 16
            local = lane - group
            golden_v1[base + 0] = 528
            golden_v1[base + 1] = np.int32(-1)
            golden_v1[base + 2] = 0
            golden_v1[base + 3] = 528
            golden_v1[base + 4] = 32
            golden_v1[base + 5] = 1
            golden_v1[base + 6] = 528
            golden_v1[base + 7] = 32
            golden_v1[base + 8] = 1
            golden_v1[base + 9] = 100 + group + 3
            golden_v1[base + 10] = 100 + (lane - 2 if local >= 2 else lane)
            golden_v1[base + 11] = 100 + (lane + 2 if local <= 13 else lane)
            golden_v1[base + 12] = (lane ^ 1) + 1
            golden_v1[base + 13] = group + 4
            golden_v1[base + 14] = 1
            golden_v1[base + 15] = 2
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
    name='simt_control_typed_core_kernel',
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
    name='legacy_micro_op_simt_simt_control_typed_core',
    grid=1,
    kernel=kernel,
    make_case=make_case,
    check=check_case,
)]

auto_main(globals())
