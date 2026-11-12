#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Migrated from test/vpto/cases/micro-op/vector-load-store/vlds-post-update."""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

from common import auto_main
from ptodsl import pto

KERNEL_SOURCE = (
    '// Copyright (c) 2026 Huawei Technologies Co., Ltd.\n// This program is free software, you can redistribute '
    'it and/or modify it under the terms and conditions of\n// CANN Open Software License Agreement Version 2.0 ('
    'the "License").\n// Please refer to the License for details. You may not use this file except in compliance '
    'with the License.\n// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER '
    'EXPRESS OR IMPLIED,\n// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PAR'
    'TICULAR PURPOSE.\n// See LICENSE in the root of the software repository for the full text of the License.\n'
    '\n// case: micro-op/vector-load-store/vlds-post-update\n// family: vector-load-store\n// target_ops: pto.vld'
    's\n// scenarios: core-f32, contiguous, full-mask, post-update-result, dist-norm\n\nmodule attributes {pto.ta'
    'rget_arch = "a5", pto.kernel_kind = #pto.kernel_kind<vector>} {\n  func.func @vlds_post_update_kernel(%input'
    ': !pto.ptr<f32, gm>, %output: !pto.ptr<f32, gm>) attributes {pto.kernel} {\n    %c0 = arith.constant 0 : ind'
    'ex\n    %c64 = arith.constant 64 : index\n    %c1024 = arith.constant 1024 : index\n    %c64_i32 = arith.con'
    'stant 64 : i32\n    %c0_i64 = arith.constant 0 : i64\n    %c32_i64 = arith.constant 32 : i64\n    %c128_i64 '
    '= arith.constant 128 : i64\n    %c4096_i64 = arith.constant 4096 : i64\n    %c1024_i32 = arith.constant 1024'
    ' : i32\n\n    %ub_in = pto.castptr %c0_i64 : i64 -> !pto.ptr<f32, ub>\n    %ub_out = pto.castptr %c4096_i64 '
    ': i64 -> !pto.ptr<f32, ub>\n\n    pto.mte_gm_ub %input, %ub_in, %c0_i64, %c128_i64\n      nburst(%c32_i64, %'
    'c128_i64, %c128_i64)\n      : !pto.ptr<f32, gm>, !pto.ptr<f32, ub>, i64, i64, i64, i64, i64\n\n    pto.set_f'
    'lag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n    pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n\n    pto.ve'
    'cscope {\n      %_:2 = scf.for %offset = %c0 to %c1024 step %c64\n          iter_args(%remaining = %c1024_i3'
    '2, %src = %ub_in)\n          -> (i32, !pto.ptr<f32, ub>) {\n        %mask, %next_remaining = pto.plt_b32 %re'
    'maining : i32 -> !pto.mask<b32>, i32\n        %vec, %updated = pto.vlds %src[%c64] {dist = "NORM"} : !pto.pt'
    'r<f32, ub>, index -> !pto.vreg<64xf32>, !pto.ptr<f32, ub>\n        pto.vsts %vec, %ub_out[%offset], %mask : !pto.vr'
    'eg<64xf32>, !pto.ptr<f32, ub>, index, !pto.mask<b32>\n        scf.yield %next_remaining, %updated : i32, !pto.ptr<f'
    '32, ub>\n      }\n    }\n\n    pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n    pto.wait_flag["PIPE_V",'
    ' "PIPE_MTE3", "EVENT_ID0"]\n    pto.mte_ub_gm %ub_out, %output, %c128_i64\n      nburst(%c32_i64, %c128_i64,'
    ' %c128_i64)\n      : !pto.ptr<f32, ub>, !pto.ptr<f32, gm>, i64, i64, i64, i64\n    pto.barrier #pto.pipe<PIP'
    'E_ALL>\n    %__case_merge_guard = arith.constant false\n\n    // inactive cross-case merge from micro-op/vec'
    'tor-load-store/vsstb-post-update\n    scf.if %__case_merge_guard {\n      \n          %c0_cmpost_1 = arith.c'
    'onstant 0 : index\n          %c0_i64_cmpost_1 = arith.constant 0 : i64\n          %c2_i16_cmpost_1 = arith.c'
    'onstant 2 : i16\n          %c4_i16_cmpost_1 = arith.constant 4 : i16\n          %c32_i64_cmpost_1 = arith.co'
    'nstant 32 : i64\n          %c64_i32_cmpost_1 = arith.constant 64 : i32\n          %c128_i64_cmpost_1 = arith'
    '.constant 128 : i64\n          %c4096_i64_cmpost_1 = arith.constant 4096 : i64\n      \n          %ub_in_cmp'
    'ost_1 = pto.castptr %c0_i64_cmpost_1 : i64 -> !pto.ptr<f32, ub>\n          %ub_out_cmpost_1 = pto.castptr %c'
    '4096_i64_cmpost_1 : i64 -> !pto.ptr<f32, ub>\n      \n          pto.mte_gm_ub %input, %ub_in_cmpost_1, %c0_i'
    '64_cmpost_1, %c128_i64_cmpost_1\n            nburst(%c32_i64_cmpost_1, %c128_i64_cmpost_1, %c128_i64_cmpost_'
    '1)\n            : !pto.ptr<f32, gm>, !pto.ptr<f32, ub>, i64, i64, i64, i64, i64\n      \n          pto.set_f'
    'lag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n          pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n      '
    '\n          pto.vecscope {\n            %mask_cmpost_1, %next_remaining_cmpost_1 = pto.plt_b32 %c64_i32_cmpo'
    'st_1 : i32 -> !pto.mask<b32>, i32\n            %value_cmpost_1 = pto.vlds %ub_in_cmpost_1[%c0_cmpost_1] : !p'
    'to.ptr<f32, ub>, index -> !pto.vreg<64xf32>\n            %updated_cmpost_1 = pto.vsstb %value_cmpost_1, %ub_out_cmp'
    'ost_1, %c2_i16_cmpost_1, %c4_i16_cmpost_1, %mask_cmpost_1 : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, i16, i16, '
    '!pto.mask<b32> -> !pto.ptr<f32, ub>\n            pto.mem_bar "VST_VLD"\n            %roundtrip_cmpost_1 = pt'
    'o.vsldb %updated_cmpost_1, %c2_i16_cmpost_1, %c4_i16_cmpost_1, %mask_cmpost_1 : !pto.ptr<f32, ub>, i16, i16,'
    ' !pto.mask<b32> -> !pto.vreg<64xf32>\n            pto.vsts %roundtrip_cmpost_1, %ub_in_cmpost_1[%c0_cmpost_1'
    '], %mask_cmpost_1 : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, index, !pto.mask<b32>\n          }\n      \n          pto'
    '.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n          pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n'
    '          pto.mte_ub_gm %ub_in_cmpost_1, %output, %c128_i64_cmpost_1\n            nburst(%c32_i64_cmpost_1, '
    '%c128_i64_cmpost_1, %c128_i64_cmpost_1)\n            : !pto.ptr<f32, ub>, !pto.ptr<f32, gm>, i64, i64, i64, '
    'i64\n          pto.barrier #pto.pipe<PIPE_ALL>\n    }\n\n    // inactive cross-case merge from micro-op/vect'
    'or-load-store/vsts-post-update\n    scf.if %__case_merge_guard {\n      \n          %c0_cmpost_2 = arith.con'
    'stant 0 : index\n          %c64_cmpost_2 = arith.constant 64 : index\n          %c1024_cmpost_2 = arith.cons'
    'tant 1024 : index\n          %c0_i64_cmpost_2 = arith.constant 0 : i64\n          %c32_i64_cmpost_2 = arith.'
    'constant 32 : i64\n          %c128_i64_cmpost_2 = arith.constant 128 : i64\n          %c4096_i64_cmpost_2 = '
    'arith.constant 4096 : i64\n          %c1024_i32_cmpost_2 = arith.constant 1024 : i32\n      \n          %ub_'
    'in_cmpost_2 = pto.castptr %c0_i64_cmpost_2 : i64 -> !pto.ptr<f32, ub>\n          %ub_out_cmpost_2 = pto.cast'
    'ptr %c4096_i64_cmpost_2 : i64 -> !pto.ptr<f32, ub>\n      \n          pto.mte_gm_ub %input, %ub_in_cmpost_2,'
    ' %c0_i64_cmpost_2, %c128_i64_cmpost_2\n            nburst(%c32_i64_cmpost_2, %c128_i64_cmpost_2, %c128_i64_c'
    'mpost_2)\n            : !pto.ptr<f32, gm>, !pto.ptr<f32, ub>, i64, i64, i64, i64, i64\n      \n          pto'
    '.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n          pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n'
    '      \n          pto.vecscope {\n            %__cmpost_2:2 = scf.for %offset_cmpost_2 = %c0_cmpost_2 to %c1'
    '024_cmpost_2 step %c64_cmpost_2\n                iter_args(%remaining_cmpost_2 = %c1024_i32_cmpost_2, %dst_c'
    'mpost_2 = %ub_out_cmpost_2)\n                -> (i32, !pto.ptr<f32, ub>) {\n              %mask_cmpost_2, %n'
    'ext_remaining_cmpost_2 = pto.plt_b32 %remaining_cmpost_2 : i32 -> !pto.mask<b32>, i32\n              %vec_cm'
    'post_2 = pto.vlds %ub_in_cmpost_2[%offset_cmpost_2] : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>\n              '
    '%updated_cmpost_2 = pto.vsts %vec_cmpost_2, %dst_cmpost_2[%c64_cmpost_2], %mask_cmpost_2 {dist = "NORM_B32"}'
    ' : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, index, !pto.mask<b32> -> !pto.ptr<f32, ub>\n              scf.yield %next_'
    'remaining_cmpost_2, %updated_cmpost_2 : i32, !pto.ptr<f32, ub>\n            }\n          }\n      \n        '
    '  pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n          pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID'
    '0"]\n          pto.mte_ub_gm %ub_out_cmpost_2, %output, %c128_i64_cmpost_2\n            nburst(%c32_i64_cmpo'
    'st_2, %c128_i64_cmpost_2, %c128_i64_cmpost_2)\n            : !pto.ptr<f32, ub>, !pto.ptr<f32, gm>, i64, i64,'
    ' i64, i64\n          pto.barrier #pto.pipe<PIPE_ALL>\n    }\n    return\n  }\n}\n'
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


    ELEMENTS = 1024
    SEED = 19


    def _generate(seed: int) -> dict:
        rng = np.random.default_rng(seed)
        data = rng.uniform(-8.0, 8.0, size=(ELEMENTS,)).astype(np.float32)
        output = np.zeros((ELEMENTS,), dtype=np.float32)

        _files["input.bin"] = data
        _files["output.bin"] = output
        _files["golden_output.bin"] = data


    def main() -> None:
        parser = argparse.ArgumentParser(
            description="Generate inputs/golden for VPTO vlds post-update validation."
        )
        parser.add_argument("--output-dir", type=Path, default=Path("."))
        parser.add_argument("--seed", type=int, default=SEED)
        args = parser.parse_args()
        generate(args.output_dir, args.seed)


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


    def main() -> None:
        golden_path = "golden_output.bin"
        output_path = "output.bin"
        strict = os.getenv("COMPARE_STRICT", "1") != "0"

        if not os.path.exists(golden_path) or not os.path.exists(output_path):
            print("[ERROR] missing golden_output.bin or output.bin")
            sys.exit(2 if strict else 0)

        golden = np.fromfile(golden_path, dtype=np.float32)
        output = np.fromfile(output_path, dtype=np.float32)
        ok = golden.shape == output.shape and np.allclose(
            golden, output, atol=0.0001, rtol=0.0001, equal_nan=True
        )
        if not ok:
            if golden.shape != output.shape:
                print(f"[ERROR] shape mismatch: {golden.shape} vs {output.shape}")
            elif golden.size:
                diff = np.abs(golden.astype(np.float64) - output.astype(np.float64))
                idx = int(np.argmax(diff))
                print(
                    f"[ERROR] mismatch at idx={idx}: golden={golden[idx]} "
                    f"output={output[idx]} diff={diff[idx]}"
                )
            if strict:
                sys.exit(2)
            return
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

BUFFERS = [
    (
        'input',
        'f32',
        'input'
    ),
    (
        'output',
        'f32',
        'output'
    )
]
SCALARS = []
SCALAR_SOURCES = []
SCALAR_DTYPES = []


@pto.jit(
    name='vlds_post_update_kernel',
    target="a5",
    backend="vpto",
    mode="explicit",
    source=KERNEL_SOURCE,
)
def kernel(
    input: pto.ptr(pto.f32, "gm"),
    output: pto.ptr(pto.f32, "gm"),
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
    name='legacy_micro_op_vector_load_store_vlds_post_update',
    grid=1,
    kernel=kernel,
    make_case=make_case,
    check=check_case,
)]

auto_main(globals())
