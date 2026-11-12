#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Migrated from test/vpto/cases/micro-op/rearrangement/vusqz."""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

from common import auto_main
from ptodsl import pto

KERNEL_SOURCE = (
    'module attributes {pto.target_arch = "a5", pto.kernel_kind = #pto.kernel_kind<vector>} {\n    func.func @vus'
    'qz_deep_merged_kernel(%arg0: !pto.ptr<i32, gm>, %arg1: !pto.ptr<f32, gm>, %arg2: !pto.ptr<i32, gm>, %arg3: !'
    'pto.ptr<i32, gm>, %arg4: !pto.ptr<f32, gm>, %arg5: !pto.ptr<i32, gm>) attributes {pto.kernel} {\n    %__deep'
    '_merge_guard = arith.constant false\n    // inactive merged from vusqz_nontrivial_mask_kernel_2d\n    scf.if'
    ' %__deep_merge_guard {\n\n    %c0_m0 = arith.constant 0 : index\n    %c1_m0 = arith.constant 1 : index\n    '
    '%c64_m0 = arith.constant 64 : index\n    %c1024_m0 = arith.constant 1024 : index\n\n    %c0_i64_m0 = arith.c'
    'onstant 0 : i64\n    %c1_i64_m0 = arith.constant 1 : i64\n    %c32_i64_m0 = arith.constant 32 : i64\n    %c6'
    '4_i32_m0 = arith.constant 64 : i32\n    %c128_i64_m0 = arith.constant 128 : i64\n    %c4096_i64_m0 = arith.c'
    'onstant 4096 : i64\n    %c8192_i64_m0 = arith.constant 8192 : i64\n    %zero_f32_m0 = arith.constant 0.0 : f'
    '32\n\n    %ub_src_m0 = pto.castptr %c0_i64_m0 : i64 -> !pto.ptr<i32, ub>\n    %ub_mask_seed_m0 = pto.castptr'
    ' %c4096_i64_m0 : i64 -> !pto.ptr<f32, ub>\n    %ub_out_m0 = pto.castptr %c8192_i64_m0 : i64 -> !pto.ptr<i32,'
    ' ub>\n\n    %false_m0 = arith.constant false\n    pto.mte_gm_ub %arg0, %ub_src_m0, %c0_i64_m0, %c128_i64_m0'
    '\n      nburst(%c32_i64_m0, %c128_i64_m0, %c128_i64_m0)\n      : !pto.ptr<i32, gm>, !pto.ptr<i32, ub>, i64, '
    'i64, i64, i64, i64\n    pto.mte_gm_ub %arg1, %ub_mask_seed_m0, %c0_i64_m0, %c128_i64_m0\n      nburst(%c32_i'
    '64_m0, %c128_i64_m0, %c128_i64_m0)\n      : !pto.ptr<f32, gm>, !pto.ptr<f32, ub>, i64, i64, i64, i64, i64\n'
    '\n    pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n    pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"'
    ']\n\n    pto.vecscope {\n      scf.for %offset_m0 = %c0_m0 to %c1024_m0 step %c64_m0 {\n        %store_mask_'
    'm0, %unused_m0 = pto.plt_b32 %c64_i32_m0 : i32 -> !pto.mask<b32>, i32\n        %src_m0 = pto.vlds %ub_src_m0'
    '[%offset_m0] : !pto.ptr<i32, ub>, index -> !pto.vreg<64xi32>\n        %mask_seed_m0 = pto.vlds %ub_mask_seed_m0[%of'
    'fset_m0] : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>\n        %place_m0 = pto.vcmps %mask_seed_m0, %zero_f32_m0'
    ', %store_mask_m0, "gt" : !pto.vreg<64xf32>, f32, !pto.mask<b32> -> !pto.mask<b32>\n        %out_m0 = pto.vus'
    'qz %src_m0, %place_m0 : !pto.vreg<64xi32>, !pto.mask<b32> -> !pto.vreg<64xi32>\n        pto.vsts %out_m0, %u'
    'b_out_m0[%offset_m0], %store_mask_m0 : !pto.vreg<64xi32>, !pto.ptr<i32, ub>, index, !pto.mask<b32>\n      }\n    }'
    '\n\n    pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n    pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID'
    '0"]\n    pto.mte_ub_gm %ub_out_m0, %arg2, %c128_i64_m0\n      nburst(%c32_i64_m0, %c128_i64_m0, %c128_i64_m0'
    ')\n      : !pto.ptr<i32, ub>, !pto.ptr<i32, gm>, i64, i64, i64, i64\n    pto.barrier #pto.pipe<PIPE_ALL>\n\n'
    '    }\n    // active merged from vusqz_kernel_2d\n\n    %c0_m1 = arith.constant 0 : index\n    %c1_m1 = arit'
    'h.constant 1 : index\n    %c64_m1 = arith.constant 64 : index\n    %c1024_m1 = arith.constant 1024 : index\n'
    '\n    %c0_i64_m1 = arith.constant 0 : i64\n    %c1_i64_m1 = arith.constant 1 : i64\n    %c32_i64_m1 = arith.'
    'constant 32 : i64\n    %c64_i32_m1 = arith.constant 64 : i32\n    %c128_i64_m1 = arith.constant 128 : i64\n '
    '   %c4096_i64_m1 = arith.constant 4096 : i64\n    %c8192_i64_m1 = arith.constant 8192 : i64\n    %zero_f32_m'
    '1 = arith.constant 0.0 : f32\n\n    %ub_src_m1 = pto.castptr %c0_i64_m1 : i64 -> !pto.ptr<i32, ub>\n    %ub_'
    'mask_seed_m1 = pto.castptr %c4096_i64_m1 : i64 -> !pto.ptr<f32, ub>\n    %ub_out_m1 = pto.castptr %c8192_i64'
    '_m1 : i64 -> !pto.ptr<i32, ub>\n\n    %false_m1 = arith.constant false\n    pto.mte_gm_ub %arg3, %ub_src_m1,'
    ' %c0_i64_m1, %c128_i64_m1\n      nburst(%c32_i64_m1, %c128_i64_m1, %c128_i64_m1)\n      : !pto.ptr<i32, gm>,'
    ' !pto.ptr<i32, ub>, i64, i64, i64, i64, i64\n    pto.mte_gm_ub %arg4, %ub_mask_seed_m1, %c0_i64_m1, %c128_i6'
    '4_m1\n      nburst(%c32_i64_m1, %c128_i64_m1, %c128_i64_m1)\n      : !pto.ptr<f32, gm>, !pto.ptr<f32, ub>, i'
    '64, i64, i64, i64, i64\n\n    pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n    pto.wait_flag["PIPE_MTE2'
    '", "PIPE_V", "EVENT_ID0"]\n\n    pto.vecscope {\n      scf.for %offset_m1 = %c0_m1 to %c1024_m1 step %c64_m1'
    ' {\n        %store_mask_m1, %unused_m1 = pto.plt_b32 %c64_i32_m1 : i32 -> !pto.mask<b32>, i32\n        %src_'
    'm1 = pto.vlds %ub_src_m1[%offset_m1] : !pto.ptr<i32, ub>, index -> !pto.vreg<64xi32>\n        %mask_seed_m1 = pto.v'
    'lds %ub_mask_seed_m1[%offset_m1] : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>\n        %place_m1 = pto.vcmps %ma'
    'sk_seed_m1, %zero_f32_m1, %store_mask_m1, "gt" : !pto.vreg<64xf32>, f32, !pto.mask<b32> -> !pto.mask<b32>\n '
    '       %out_m1 = pto.vusqz %src_m1, %place_m1 : !pto.vreg<64xi32>, !pto.mask<b32> -> !pto.vreg<64xi32>\n    '
    '    pto.vsts %out_m1, %ub_out_m1[%offset_m1], %store_mask_m1 : !pto.vreg<64xi32>, !pto.ptr<i32, ub>, index, !pto.ma'
    'sk<b32>\n      }\n    }\n\n    pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n    pto.wait_flag["PIPE_V",'
    ' "PIPE_MTE3", "EVENT_ID0"]\n    pto.mte_ub_gm %ub_out_m1, %arg5, %c128_i64_m1\n      nburst(%c32_i64_m1, %c1'
    '28_i64_m1, %c128_i64_m1)\n      : !pto.ptr<i32, ub>, !pto.ptr<i32, gm>, i64, i64, i64, i64\n    pto.barrier '
    '#pto.pipe<PIPE_ALL>\n\n    return\n  }\n}\n'
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

    # case: micro-op/rearrangement/vusqz
    # family: rearrangement
    # target_ops: pto.vusqz
    # scenarios: predicate-driven-rearrangement, prefix-count

    import argparse
    from pathlib import Path

    import numpy as np


    ROWS = 32
    COLS = 32
    LANES = 64
    BLOCKS = ROWS * COLS // LANES
    ACTIVE_PER_BLOCK = 16
    SEED = 19


    def build_case() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        src = np.zeros((BLOCKS, LANES), dtype=np.int32)
        mask_seed = np.full((BLOCKS, LANES), -1.0, dtype=np.float32)
        out = np.zeros((BLOCKS, LANES), dtype=np.int32)

        for block in range(BLOCKS):
            src[block] = np.arange(block * 100 - 31, block * 100 - 31 + LANES, dtype=np.int32)
            mask_seed[block, :ACTIVE_PER_BLOCK] = 1.0
            active_count = 0
            out[block, 0] = 0
            for lane in range(1, LANES):
                if mask_seed[block, lane - 1] > 0.0:
                    active_count += 1
                out[block, lane] = active_count

        return src.reshape(ROWS, COLS), mask_seed.reshape(ROWS, COLS), out.reshape(ROWS, COLS)


    def _generate(seed: int) -> dict:
        src, mask_seed, out = build_case()
        _files["v1.bin"] = src.reshape(-1)
        _files["v2.bin"] = mask_seed.reshape(-1)
        _files["golden_v3.bin"] = out.reshape(-1)
        _files["v3.bin"] = np.zeros_like(out.reshape(-1))


    def main() -> None:
        parser = argparse.ArgumentParser(description="Generate vusqz prefix-count inputs/golden.")
        parser.add_argument("--output-dir", type=Path, default=Path("."))
        parser.add_argument("--seed", type=int, default=SEED)
        args = parser.parse_args()
        del args.seed
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

    # case: micro-op/rearrangement/vusqz
    # family: rearrangement
    # target_ops: pto.vusqz
    # scenarios: predicate-driven-rearrangement, prefix-count

    import sys
    import numpy as np


    def main() -> None:
        golden = np.fromfile("golden_v3.bin", dtype=np.int32)
        output = np.fromfile("v3.bin", dtype=np.int32)
        if golden.shape != output.shape:
            print(f"[ERROR] Shape mismatch: {golden.shape} vs {output.shape}")
            sys.exit(2)
        if not np.array_equal(golden, output):
            diff = np.nonzero(golden != output)[0]
            idx = int(diff[0]) if diff.size else 0
            print(
                f"[ERROR] Mismatch at idx={idx}: golden={int(golden[idx])} out={int(output[idx])}"
            )
            sys.exit(2)
        print("[INFO] compare passed")


    return locals()

HELPERS = {}
SEED_EXPR = 'SEED'
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

# launch.cpp binds some host buffers to several kernel arguments; ARG_TO_BUFFER
# says which distinct buffer each kernel argument receives, so the module can
# pass aliased device tensors. BUFFER_FILES lists those distinct buffers.
BUFFER_FILES = [
    (
        'v1',
        'i32'
    ),
    (
        'v2',
        'f32'
    ),
    (
        'v3',
        'i32'
    )
]
ARG_TO_BUFFER = [
    0,
    0,
    0,
    0,
    1,
    2
]
ARG_DTYPES = [
    'i32',
    'f32',
    'i32',
    'i32',
    'f32',
    'i32'
]
SCALARS = []

def _dtype_map():
    import torch
    return {
        "int8": torch.int8, "int16": torch.int16, "int32": torch.int32, "int64": torch.int64,
        "uint8": torch.uint8, "uint16": torch.int16, "uint32": torch.int32, "uint64": torch.int64,
        "float16": torch.float16, "float32": torch.float32, "float64": torch.float64,
    }


@pto.jit(
    name='vusqz_deep_merged_kernel',
    target="a5",
    backend="vpto",
    mode="explicit",
    source=KERNEL_SOURCE,
)
def kernel(
    arg0: pto.ptr(pto.i32, "gm"),
    arg1: pto.ptr(pto.f32, "gm"),
    arg2: pto.ptr(pto.i32, "gm"),
    arg3: pto.ptr(pto.i32, "gm"),
    arg4: pto.ptr(pto.f32, "gm"),
    arg5: pto.ptr(pto.i32, "gm"),
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
    base, per_arg = {}, []
    for index, ty in zip(ARG_TO_BUFFER, ARG_DTYPES):
        name, _buf_ty = BUFFER_FILES[index]
        key = "%s.bin" % name
        if index not in base:
            if key in produced:
                # One host buffer can back several kernel arguments of
                # different widths, so keep the golden bytes and let each
                # argument reinterpret them below instead of narrowing the
                # shared buffer to the first argument type.
                array = np.ascontiguousarray(np.asarray(produced[key]))
            else:
                golden_key = output_to_golden.get(key)
                if golden_key is None or golden_key not in produced:
                    raise RuntimeError("golden did not produce %s" % key)
                array = np.ascontiguousarray(
                    np.zeros_like(np.asarray(produced[golden_key]), dtype=np.dtype(NUMPY_DTYPE[ty])))
            base[index] = torch.from_numpy(array).to("npu:0")
        tensor = base[index]
        want = _dtype_map()[NUMPY_DTYPE[ty]]
        if tensor.dtype != want and tensor.numel() * tensor.element_size() % want.itemsize == 0:
            tensor = tensor.view(want)
        per_arg.append(tensor)
    return per_arg, produced, list(SCALARS)


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
    seen = {}
    for index, tensor in zip(ARG_TO_BUFFER, device_inputs):
        seen.setdefault(index, tensor)
    for index, tensor in seen.items():
        name, _ty = BUFFER_FILES[index]
        files["%s.bin" % name] = tensor.cpu().numpy()
    _apply_write_masks(files, produced)
    _run_compare(files)


GRID_FROM_VALUES = None


CASES = [dict(
    name='legacy_micro_op_rearrangement_vusqz',
    grid=1,
    kernel=kernel,
    make_case=make_case,
    check=check_case,
)]

auto_main(globals())
