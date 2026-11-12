#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Migrated from test/vpto/cases/micro-op/predicate-load-store/pldi-norm."""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

from common import auto_main
from ptodsl import pto

KERNEL_SOURCE = (
    '// -----------------------------------------------------------------------------\n// case: micro-op/predicat'
    'e-load-store/pldi-norm\n// family: predicate-load-store\n// target_ops: pto.pldi\n// scenarios: packed-load,'
    ' immediate-offset, representative-logical-elements\n// -----------------------------------------------------'
    '------------------------\nmodule attributes {pto.target_arch = "a5", pto.kernel_kind = #pto.kernel_kind<vect'
    'or>} {\n  func.func @pldi_norm_kernel_2d(%arg0: !pto.ptr<ui8, gm>, %arg1: !pto.ptr<ui8, gm>) attributes {pto'
    '.kernel} {\n    %c0 = arith.constant 0 : index\n    %c1 = arith.constant 1 : index\n    %c256 = arith.consta'
    'nt 256 : index\n    %c32 = arith.constant 32 : index\n    %c288 = arith.constant 288 : index\n    %c0_i64 = '
    'arith.constant 0 : i64\n    %c1_i64 = arith.constant 1 : i64\n    %c32_i64 = arith.constant 32 : i64\n    %c'
    '4_i64 = arith.constant 4 : i64\n    %c256_i64 = arith.constant 256 : i64\n    %c1024_i64 = arith.constant 10'
    '24 : i64\n    %c8192_i64 = arith.constant 8192 : i64\n    %c10240_i64 = arith.constant 10240 : i64\n    %c25'
    '6_i32 = arith.constant 256 : i32\n    %c256_loop_i32 = arith.constant 256 : i32\n    %false = arith.constant'
    ' false\n\n    %ub_in = pto.castptr %c8192_i64 : i64 -> !pto.ptr<ui8, ub>\n    %ub_out = pto.castptr %c10240_'
    'i64 : i64 -> !pto.ptr<ui8, ub>\n    pto.mte_gm_ub %arg0, %ub_in, %c0_i64, %c256_i64\n      nburst(%c4_i64, %'
    'c256_i64, %c256_i64)\n      : !pto.ptr<ui8, gm>, !pto.ptr<ui8, ub>, i64, i64, i64, i64, i64\n\n    pto.set_f'
    'lag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n    pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n\n    pto.ve'
    'cscope {\n      %_:1 = scf.for %offset = %c0 to %c256 step %c256 iter_args(%remaining = %c256_loop_i32) -> ('
    'i32) {\n        %loaded = pto.pldi %ub_in[%c0], "NORM" : !pto.ptr<ui8, ub>, index -> !pto.mask<b8>\n        '
    '%full_mask, %next_remaining = pto.plt_b8 %remaining : i32 -> !pto.mask<b8>, i32\n        %ones_offset = arit'
    'h.addi %offset, %c32 : index\n        %zeros_offset = arith.addi %offset, %c288 : index\n        %ones = pto'
    '.vlds %ub_in[%ones_offset] : !pto.ptr<ui8, ub>, index -> !pto.vreg<256xui8>\n        %zeros = pto.vlds %ub_in[%zero'
    's_offset] : !pto.ptr<ui8, ub>, index -> !pto.vreg<256xui8>\n        %out = pto.vsel %ones, %zeros, %loaded : !pto.v'
    'reg<256xui8>, !pto.vreg<256xui8>, !pto.mask<b8> -> !pto.vreg<256xui8>\n        pto.vsts %out, %ub_out[%offse'
    't], %full_mask : !pto.vreg<256xui8>, !pto.ptr<ui8, ub>, index, !pto.mask<b8>\n        scf.yield %next_remaining : i'
    '32\n      }\n    }\n\n    pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n    pto.wait_flag["PIPE_V", "PIP'
    'E_MTE3", "EVENT_ID0"]\n    pto.set_loop1_stride_ubtoout %c256_i64, %c256_i64 : i64, i64\n    pto.set_loop2_s'
    'tride_ubtoout %c256_i64, %c256_i64 : i64, i64\n    pto.mte_ub_gm %ub_out, %arg1, %c256_i64\n      nburst(%c4'
    '_i64, %c256_i64, %c256_i64)\n      : !pto.ptr<ui8, ub>, !pto.ptr<ui8, gm>, i64, i64, i64, i64\n    pto.barri'
    'er #pto.pipe<PIPE_ALL>\n    %__case_merge_guard = arith.constant false\n\n    // inactive cross-case merge f'
    'rom micro-op/predicate-load-store/plds-norm\n    scf.if %__case_merge_guard {\n      \n          %c0_cmg9_1 '
    '= arith.constant 0 : index\n          %c1_cmg9_1 = arith.constant 1 : index\n          %c256_cmg9_1 = arith.'
    'constant 256 : index\n          %c32_cmg9_1 = arith.constant 32 : index\n          %c288_cmg9_1 = arith.cons'
    'tant 288 : index\n          %c0_i64_cmg9_1 = arith.constant 0 : i64\n          %c1_i64_cmg9_1 = arith.consta'
    'nt 1 : i64\n          %c4_i64_cmg9_1 = arith.constant 4 : i64\n          %c256_i64_cmg9_1 = arith.constant 2'
    '56 : i64\n          %c256_i32_cmg9_1 = arith.constant 256 : i32\n          %c256_loop_i32_cmg9_1 = arith.con'
    'stant 256 : i32\n          %c8192_i64_cmg9_1 = arith.constant 8192 : i64\n          %c10240_i64_cmg9_1 = ari'
    'th.constant 10240 : i64\n          %false_cmg9_1 = arith.constant false\n      \n          %ub_in_cmg9_1 = p'
    'to.castptr %c8192_i64_cmg9_1 : i64 -> !pto.ptr<ui8, ub>\n          %ub_out_cmg9_1 = pto.castptr %c10240_i64_'
    'cmg9_1 : i64 -> !pto.ptr<ui8, ub>\n          pto.mte_gm_ub %arg0, %ub_in_cmg9_1, %c0_i64_cmg9_1, %c256_i64_c'
    'mg9_1\n            nburst(%c4_i64_cmg9_1, %c256_i64_cmg9_1, %c256_i64_cmg9_1)\n            : !pto.ptr<ui8, g'
    'm>, !pto.ptr<ui8, ub>, i64, i64, i64, i64, i64\n      \n          pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT'
    '_ID0"]\n          pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n      \n          pto.vecscope {\n     '
    '       %__cmg9_1:1 = scf.for %offset_cmg9_1 = %c0_cmg9_1 to %c256_cmg9_1 step %c256_cmg9_1 iter_args(%remain'
    'ing_cmg9_1 = %c256_loop_i32_cmg9_1) -> (i32) {\n              %byte_offset_cmg9_1 = arith.addi %offset_cmg9_'
    '1, %c0_cmg9_1 : index\n              %loaded_cmg9_1 = pto.plds %ub_in_cmg9_1[%byte_offset_cmg9_1], "NORM" : '
    '!pto.ptr<ui8, ub>, index -> !pto.mask<b8>\n              %full_mask_cmg9_1, %next_remaining_cmg9_1 = pto.plt'
    '_b8 %remaining_cmg9_1 : i32 -> !pto.mask<b8>, i32\n              %ones_offset_cmg9_1 = arith.addi %offset_cm'
    'g9_1, %c32_cmg9_1 : index\n              %zeros_offset_cmg9_1 = arith.addi %offset_cmg9_1, %c288_cmg9_1 : in'
    'dex\n              %ones_cmg9_1 = pto.vlds %ub_in_cmg9_1[%ones_offset_cmg9_1] : !pto.ptr<ui8, ub>, index -> !pto.vr'
    'eg<256xui8>\n              %zeros_cmg9_1 = pto.vlds %ub_in_cmg9_1[%zeros_offset_cmg9_1] : !pto.ptr<ui8, ub>, index '
    '-> !pto.vreg<256xui8>\n              %out_cmg9_1 = pto.vsel %ones_cmg9_1, %zeros_cmg9_1, %loaded_cmg9_1 : !p'
    'to.vreg<256xui8>, !pto.vreg<256xui8>, !pto.mask<b8> -> !pto.vreg<256xui8>\n              pto.vsts %out_cmg9_'
    '1, %ub_out_cmg9_1[%offset_cmg9_1], %full_mask_cmg9_1 : !pto.vreg<256xui8>, !pto.ptr<ui8, ub>, index, !pto.mask<b8>'
    '\n              scf.yield %next_remaining_cmg9_1 : i32\n            }\n          }\n      \n          pto.se'
    't_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n          pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n   '
    '       pto.set_loop1_stride_ubtoout %c256_i64_cmg9_1, %c256_i64_cmg9_1 : i64, i64\n          pto.set_loop2_s'
    'tride_ubtoout %c256_i64_cmg9_1, %c256_i64_cmg9_1 : i64, i64\n          pto.mte_ub_gm %ub_out_cmg9_1, %arg1, '
    '%c256_i64_cmg9_1\n            nburst(%c4_i64_cmg9_1, %c256_i64_cmg9_1, %c256_i64_cmg9_1)\n            : !pto'
    '.ptr<ui8, ub>, !pto.ptr<ui8, gm>, i64, i64, i64, i64\n          pto.barrier #pto.pipe<PIPE_ALL>\n    }\n    '
    'return\n  }\n}\n'
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

    # case: micro-op/predicate-load-store/pldi-norm
    # family: predicate-load-store
    # target_ops: pto.pldi
    # scenarios: packed-load, immediate-offset, representative-logical-elements

    import argparse
    from pathlib import Path

    import numpy as np


    SEED = 19
    ACTIVE_BITS = 145
    OUTPUT_BYTES = 1024
    VECTOR_BYTES = 256
    PACKED_BYTES = 32


    def prefix_bits(active_bits: int) -> np.ndarray:
        bits = np.zeros((256,), dtype=np.uint8)
        bits[:active_bits] = 1
        return bits


    def make_input_buffer(bits: np.ndarray) -> np.ndarray:
        packed = np.packbits(bits.astype(np.uint8, copy=False), bitorder="little")
        ones = np.ones((VECTOR_BYTES,), dtype=np.uint8)
        zeros = np.zeros((VECTOR_BYTES,), dtype=np.uint8)
        out = np.zeros((OUTPUT_BYTES,), dtype=np.uint8)
        out[:PACKED_BYTES] = packed[:PACKED_BYTES]
        out[PACKED_BYTES : PACKED_BYTES + VECTOR_BYTES] = ones
        out[PACKED_BYTES + VECTOR_BYTES : PACKED_BYTES + 2 * VECTOR_BYTES] = zeros
        return out


    def expected_selected_bytes(bits: np.ndarray) -> np.ndarray:
        out = np.zeros((OUTPUT_BYTES,), dtype=np.uint8)
        out[:VECTOR_BYTES] = bits.astype(np.uint8, copy=False)
        return out


    def _generate(seed: int) -> dict:
        del seed
        bits = prefix_bits(ACTIVE_BITS)
        input_buffer = make_input_buffer(bits)
        golden = expected_selected_bytes(bits)

        _files["v1.bin"] = input_buffer
        _files["v2.bin"] = np.zeros((OUTPUT_BYTES,), dtype=np.uint8)
        _files["golden_v2.bin"] = golden


    def main() -> None:
        parser = argparse.ArgumentParser(
            description="Generate raw packed predicate input/golden for VPTO micro-op pldi-norm validation."
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

    # case: micro-op/predicate-load-store/pldi-norm
    # family: predicate-load-store
    # target_ops: pto.pldi
    # scenarios: packed-load, immediate-offset, representative-logical-elements

    import numpy as np


    def main() -> None:
        golden = np.fromfile("golden_v2.bin", dtype=np.uint8)
        output = np.fromfile("v2.bin", dtype=np.uint8)
        if golden.size < 256 or output.size < 256:
            print(
                f"[ERROR] Packed buffer too small: golden={golden.size} out={output.size}"
            )
            raise SystemExit(2)
        if not np.array_equal(golden[:256], output[:256]):
            diff = np.nonzero(golden[:256] != output[:256])[0]
            idx = int(diff[0]) if diff.size else 0
            print(
                f"[ERROR] Mismatch (pldi NORM -> vsel): idx={idx} "
                f"golden={int(golden[idx])} out={int(output[idx])}"
            )
            raise SystemExit(2)
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
        'v1',
        'ui8',
        'arg0'
    ),
    (
        'v2',
        'ui8',
        'arg1'
    )
]
SCALARS = []
SCALAR_SOURCES = []
SCALAR_DTYPES = []


@pto.jit(
    name='pldi_norm_kernel_2d',
    target="a5",
    backend="vpto",
    mode="explicit",
    source=KERNEL_SOURCE,
)
def kernel(
    arg0: pto.ptr(pto.ui8, "gm"),
    arg1: pto.ptr(pto.ui8, "gm"),
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
    name='legacy_micro_op_predicate_load_store_pldi_norm',
    grid=1,
    kernel=kernel,
    make_case=make_case,
    check=check_case,
)]

auto_main(globals())
