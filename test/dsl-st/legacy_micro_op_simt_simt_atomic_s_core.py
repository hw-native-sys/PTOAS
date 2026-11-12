#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Migrated from test/vpto/cases/micro-op/simt/simt-atomic-s-core."""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

from common import auto_main
from ptodsl import pto

KERNEL_SOURCE = (
    'module attributes {pto.target_arch = "a5", pto.kernel_kind = #pto.kernel_kind<vector>} {\n  func.func @simt_'
    'atomic_s_core_kernel(%arg0: !pto.ptr<i32, gm>) attributes {pto.aicore} {\n    %c0_i64 = arith.constant 0 : i'
    '64\n    %c128_i64 = arith.constant 128 : i64\n    %c32_i64 = arith.constant 32 : i64\n    %dim_z = arith.con'
    'stant 1 : i32\n    %dim_y = arith.constant 1 : i32\n    %dim_x = arith.constant 32 : i32\n\n    %ub = pto.ca'
    'stptr %c0_i64 : i64 -> !pto.ptr<i32, ub>\n    %s = pto.castptr %c0_i64 : i64 -> !pto.ptr<i32, ub>\n    pto.m'
    'te_gm_ub %arg0, %ub, %c0_i64, %c128_i64\n      nburst(%c32_i64, %c128_i64, %c128_i64)\n      : !pto.ptr<i32,'
    ' gm>, !pto.ptr<i32, ub>, i64, i64, i64, i64, i64\n\n    pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n  '
    '  pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n\n    pto.simt_launch @simt_atomic_s_core_body<<<%dim_x'
    ', %dim_y, %dim_z>>>(%s, %ub) : (!pto.ptr<i32, ub>, !pto.ptr<i32, ub>) -> ()\n\n    pto.set_flag["PIPE_V", "P'
    'IPE_MTE3", "EVENT_ID0"]\n    pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n    pto.mte_ub_gm %ub, %arg0'
    ', %c128_i64\n      nburst(%c32_i64, %c128_i64, %c128_i64)\n      : !pto.ptr<i32, ub>, !pto.ptr<i32, gm>, i64'
    ', i64, i64, i64\n    pto.barrier #pto.pipe<PIPE_ALL>\n    return\n  }\n\n  func.func @simt_atomic_s_core_bod'
    'y(%s: !pto.ptr<i32, ub>, %out: !pto.ptr<i32, ub>) attributes {pto.simt_entry} {\n    %laneid = pto.get_lanei'
    'd : i32\n    %c32 = arith.constant 32 : i32\n    %c64 = arith.constant 64 : i32\n    %c96 = arith.constant 9'
    '6 : i32\n    %c128 = arith.constant 128 : i32\n    %c160 = arith.constant 160 : i32\n    %c192 = arith.const'
    'ant 192 : i32\n    %c224 = arith.constant 224 : i32\n    %v5 = arith.constant 5 : i32\n    %v7 = arith.const'
    'ant 7 : i32\n    %v10 = arith.constant 10 : i32\n    %v15 = arith.constant 15 : i32\n    %v42 = arith.consta'
    'nt 42 : i32\n\n    %idx_add = arith.index_castui %laneid : i32 to index\n    %off_max = arith.addi %laneid, '
    '%c32 : i32\n    %idx_max = arith.index_castui %off_max : i32 to index\n    %off_cas = arith.addi %laneid, %c'
    '64 : i32\n    %idx_cas = arith.index_castui %off_cas : i32 to index\n    %off_and = arith.addi %laneid, %c96'
    ' : i32\n    %idx_and = arith.index_castui %off_and : i32 to index\n    %out_add = arith.addi %laneid, %c128 '
    ': i32\n    %out_add_idx = arith.index_castui %out_add : i32 to index\n    %out_max = arith.addi %laneid, %c1'
    '60 : i32\n    %out_max_idx = arith.index_castui %out_max : i32 to index\n    %out_cas = arith.addi %laneid, '
    '%c192 : i32\n    %out_cas_idx = arith.index_castui %out_cas : i32 to index\n    %out_and = arith.addi %lanei'
    'd, %c224 : i32\n    %out_and_idx = arith.index_castui %out_and : i32 to index\n\n    %ptr0 = pto.addptr %s, '
    '%idx_add : !pto.ptr<i32, ub>, index -> !pto.ptr<i32, ub>\n    %old_add = pto.atomic_add %ptr0, %v5 l2cache(nmfv) : '
    '!pto.ptr<i32, ub>, i32 -> i32\n    %ptr1 = pto.addptr %s, %idx_max : !pto.ptr<i32, ub>, index -> !pto.ptr<i32, ub>'
    '\n    %old_max = pto.atomic_max %ptr1, %v7 l2cache(nmfv) signed : !pto.ptr<i32, ub>, i32 -> i32\n    %ptr2 ='
    ' pto.addptr %s, %idx_cas : !pto.ptr<i32, ub>, index -> !pto.ptr<i32, ub>\n    %old_cas = pto.atomic_cas %ptr2, %v10'
    ', %v42 l2cache(nmfv) : !pto.ptr<i32, ub>, i32 -> i32\n    %ptr3 = pto.addptr %s, %idx_and : !pto.ptr<i32, ub'
    '>, index -> !pto.ptr<i32, ub>\n    %old_and = pto.atomic_and %ptr3, %v7 l2cache(nmfv) : !pto.ptr<i32, ub>, i32 -> i'
    '32\n\n    pto.store %old_add, %out[%out_add_idx] : !pto.ptr<i32, ub>, index, i32\n    pto.store %old_max, %out[%out'
    '_max_idx] : !pto.ptr<i32, ub>, index, i32\n    pto.store %old_cas, %out[%out_cas_idx] : !pto.ptr<i32, ub>, index, i32\n   '
    ' pto.store %old_and, %out[%out_and_idx] : !pto.ptr<i32, ub>, index, i32\n    return\n  }\n}\n'
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

    ELEMS = 1024
    LANES = 32


    def _generate(seed: int) -> dict:
        v1 = np.full(ELEMS, -1, dtype=np.int32)
        golden_v1 = np.full(ELEMS, -1, dtype=np.int32)
        v1[0:LANES] = 10
        v1[32 : 32 + LANES] = 20
        v1[64 : 64 + LANES] = 10
        v1[96 : 96 + LANES] = 10
        golden_v1[0:LANES] = 15
        golden_v1[32 : 32 + LANES] = 20
        golden_v1[64 : 64 + LANES] = 42
        golden_v1[96 : 96 + LANES] = 2
        golden_v1[128 : 128 + LANES] = 10
        golden_v1[160 : 160 + LANES] = 20
        golden_v1[192 : 192 + LANES] = 10
        golden_v1[224 : 224 + LANES] = 10
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
    name='simt_atomic_s_core_kernel',
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
    name='legacy_micro_op_simt_simt_atomic_s_core',
    grid=1,
    kernel=kernel,
    make_case=make_case,
    check=check_case,
)]

auto_main(globals())
