#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Migrated from test/vpto/cases/micro-op/compare-select/vsel-f32-plds-us-pintlv-pbitcast."""

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
    '\n// -----------------------------------------------------------------------------\n// case: micro-op/compar'
    'e-select/vsel-f32-plds-us-pintlv-pbitcast\n// family: compare-select\n// target_ops: pto.plds, pto.pintlv_b1'
    '6, pto.pbitcast, pto.vsel\n// scenarios: packed-us-mask-expand-to-b32, f32-select-from-compressed-mask\n// -'
    '----------------------------------------------------------------------------\nmodule attributes {pto.target_'
    'arch = "a5", pto.kernel_kind = #pto.kernel_kind<vector>} {\n  func.func @vsel_f32_plds_us_pintlv_pbitcast_ke'
    'rnel_2d(%arg0: !pto.ptr<f32, gm>,\n                                                        %arg1: !pto.ptr<f'
    '32, gm>,\n                                                        %arg2: !pto.ptr<ui8, gm>,\n               '
    '                                         %arg3: !pto.ptr<f32, gm>) attributes {pto.kernel} {\n    %c0 = arit'
    'h.constant 0 : index\n    %c64 = arith.constant 64 : index\n    %c0_i64 = arith.constant 0 : i64\n    %c32_i'
    '64 = arith.constant 32 : i64\n    %c128_i64 = arith.constant 128 : i64\n    %c512_i64 = arith.constant 512 :'
    ' i64\n    %c1024_i64 = arith.constant 1024 : i64\n    %c1056_i64 = arith.constant 1056 : i64\n\n    %ub_lhs '
    '= pto.castptr %c0_i64 : i64 -> !pto.ptr<f32, ub>\n    %ub_rhs = pto.castptr %c512_i64 : i64 -> !pto.ptr<f32,'
    ' ub>\n    %ub_mask = pto.castptr %c1024_i64 : i64 -> !pto.ptr<ui8, ub>\n    %ub_out = pto.castptr %c1056_i64'
    ' : i64 -> !pto.ptr<f32, ub>\n\n    pto.set_loop1_stride_outtoub %c128_i64, %c128_i64 : i64, i64\n    pto.set'
    '_loop2_stride_outtoub %c128_i64, %c128_i64 : i64, i64\n    pto.mte_gm_ub %arg0, %ub_lhs, %c0_i64, %c128_i64'
    '\n      nburst(%c32_i64, %c128_i64, %c128_i64)\n      : !pto.ptr<f32, gm>, !pto.ptr<f32, ub>, i64, i64, i64,'
    ' i64, i64\n    pto.mte_gm_ub %arg1, %ub_rhs, %c0_i64, %c128_i64\n      nburst(%c32_i64, %c128_i64, %c128_i64'
    ')\n      : !pto.ptr<f32, gm>, !pto.ptr<f32, ub>, i64, i64, i64, i64, i64\n    pto.mte_gm_ub %arg2, %ub_mask,'
    ' %c0_i64, %c32_i64\n      nburst(%c32_i64, %c32_i64, %c32_i64)\n      : !pto.ptr<ui8, gm>, !pto.ptr<ui8, ub>'
    ', i64, i64, i64, i64, i64\n\n    pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n    pto.wait_flag["PIPE_M'
    'TE2", "PIPE_V", "EVENT_ID0"]\n\n    pto.vecscope {\n      %mask_b8 = pto.plds %ub_mask[%c0], "US" : !pto.ptr'
    '<ui8, ub>, index -> !pto.mask<b8>\n      %mask_b16 = pto.pbitcast %mask_b8 : !pto.mask<b8> -> !pto.mask<b16>'
    '\n      %all_b16 = pto.pset_b16 "PAT_ALL" : !pto.mask<b16>\n      %all_b32 = pto.pset_b32 "PAT_ALL" : !pto.m'
    'ask<b32>\n      %mask0_b16, %mask1_b16 = pto.pintlv_b16 %mask_b16, %all_b16 : !pto.mask<b16>, !pto.mask<b16>'
    ' -> !pto.mask<b16>, !pto.mask<b16>\n      %mask0_b32 = pto.pbitcast %mask0_b16 : !pto.mask<b16> -> !pto.mask'
    '<b32>\n      %mask1_b32 = pto.pbitcast %mask1_b16 : !pto.mask<b16> -> !pto.mask<b32>\n      %lhs0 = pto.vlds'
    ' %ub_lhs[%c0] : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>\n      %rhs0 = pto.vlds %ub_rhs[%c0] : !pto.ptr<f32, '
    'ub>, index -> !pto.vreg<64xf32>\n      %lhs1 = pto.vlds %ub_lhs[%c64] : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>\n   '
    '   %rhs1 = pto.vlds %ub_rhs[%c64] : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>\n      %out0 = pto.vsel %lhs0, %r'
    'hs0, %mask0_b32 : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>\n      %out1 = p'
    'to.vsel %lhs1, %rhs1, %mask1_b32 : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>'
    '\n      pto.vsts %out0, %ub_out[%c0], %all_b32 : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, index, !pto.mask<b32>\n     '
    ' pto.vsts %out1, %ub_out[%c64], %all_b32 : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, index, !pto.mask<b32>\n    }\n\n  '
    '  pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n    pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n '
    '   pto.set_loop1_stride_ubtoout %c128_i64, %c128_i64 : i64, i64\n    pto.set_loop2_stride_ubtoout %c128_i64,'
    ' %c128_i64 : i64, i64\n    pto.mte_ub_gm %ub_out, %arg3, %c128_i64\n      nburst(%c32_i64, %c128_i64, %c128_'
    'i64)\n      : !pto.ptr<f32, ub>, !pto.ptr<f32, gm>, i64, i64, i64, i64\n    pto.barrier #pto.pipe<PIPE_ALL>'
    '\n    return\n  }\n}\n'
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


    LANES = 128
    MASK_BYTES = 32
    SEED = 19


    def load_us_source_bits(packed: np.ndarray) -> np.ndarray:
        # `plds ..., "US"` on this path consumes only the first VL/16 bytes.
        bits = np.unpackbits(packed[:16], bitorder="little")
        return bits.astype(np.bool_, copy=False)


    def plds_us_to_mask_b8(src_bits: np.ndarray) -> np.ndarray:
        # "US": duplicate each loaded bit once.
        return np.repeat(src_bits, 2).astype(np.bool_, copy=False)


    def pbitcast_b8_to_b16(mask_b8: np.ndarray) -> np.ndarray:
        # Reinterpret the same predicate image at b16 granularity.
        # For the duplicated "US" image, each b16 lane observes the first bit of
        # the corresponding 2-bit pair, which reconstructs the original 128 bits.
        return mask_b8[::2].astype(np.bool_, copy=False)


    def pintlv_b16_with_all(mask_b16: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Interleave `%mask_b16` with `all_b16`, then split into low/high outputs.
        interleaved = np.empty((256,), dtype=np.bool_)
        interleaved[0::2] = mask_b16
        interleaved[1::2] = True
        return interleaved[:128], interleaved[128:]


    def pbitcast_b16_to_b32(mask_b16_image: np.ndarray) -> np.ndarray:
        # Reinterpret the same predicate image at b32 granularity.
        # The b32 lanes read back the even-positioned b16 lanes.
        return mask_b16_image[0::2][:64].astype(np.bool_, copy=False)


    def build_vsel_lanes_from_mask_pipeline(packed: np.ndarray) -> np.ndarray:
        src_bits = load_us_source_bits(packed)
        mask_b8 = plds_us_to_mask_b8(src_bits)
        mask_b16 = pbitcast_b8_to_b16(mask_b8)
        mask0_b16, mask1_b16 = pintlv_b16_with_all(mask_b16)
        mask0_b32 = pbitcast_b16_to_b32(mask0_b16)
        mask1_b32 = pbitcast_b16_to_b32(mask1_b16)
        return np.concatenate([mask0_b32, mask1_b32]).astype(np.bool_, copy=False)


    def _generate(seed: int) -> dict:
        rng = np.random.default_rng(seed)
        v1 = rng.uniform(-3.0, 3.0, size=(LANES,)).astype(np.float32)
        v2 = rng.uniform(-3.0, 3.0, size=(LANES,)).astype(np.float32)
        packed = rng.integers(0, 256, size=(MASK_BYTES,), dtype=np.uint8)
        lanes = build_vsel_lanes_from_mask_pipeline(packed)
        v4 = np.zeros((LANES,), dtype=np.float32)
        golden_v4 = np.where(lanes, v1, v2).astype(np.float32, copy=False)

        _files["v1.bin"] = v1
        _files["v2.bin"] = v2
        _files["v3.bin"] = packed
        _files["v4.bin"] = v4
        _files["golden_v4.bin"] = golden_v4


    def main() -> None:
      parser = argparse.ArgumentParser(
          description="Generate inputs/golden for VPTO vsel-f32-plds-us-pintlv-pbitcast."
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


    def compare_bin(golden_path, output_path, dtype, eps):
        if not os.path.exists(golden_path) or not os.path.exists(output_path):
            return False
        golden = np.fromfile(golden_path, dtype=dtype)
        output = np.fromfile(output_path, dtype=dtype)
        if golden.shape != output.shape:
            print(f"[ERROR] Shape mismatch: {golden.shape} vs {output.shape}")
            return False
        if not np.allclose(golden, output, atol=eps, rtol=eps, equal_nan=True):
            diff = np.abs(golden.astype(np.float64) - output.astype(np.float64))
            idx = int(np.argmax(diff))
            print(f"[ERROR] Mismatch: idx={idx} golden={golden[idx]} out={output[idx]}")
            return False
        return True


    def main():
        strict = os.getenv("COMPARE_STRICT", "1") != "0"
        ok = compare_bin("golden_v4.bin", "v4.bin", np.float32, 1e-6)
        if not ok:
            if strict:
                print("[ERROR] compare failed")
                sys.exit(2)
            print("[WARN] compare failed (non-gating)")
            return
        print("[INFO] compare passed")


    return locals()

HELPERS = {}
SEED_EXPR = 'SEED'
COMPARE_PAIRS = [
    (
        'golden_v4.bin',
        'v4.bin'
    )
]

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
        'f32',
        'arg0'
    ),
    (
        'v2',
        'f32',
        'arg1'
    ),
    (
        'v3',
        'ui8',
        'arg2'
    ),
    (
        'v4',
        'f32',
        'arg3'
    )
]
SCALARS = []
SCALAR_SOURCES = []
SCALAR_DTYPES = []


@pto.jit(
    name='vsel_f32_plds_us_pintlv_pbitcast_kernel_2d',
    target="a5",
    backend="vpto",
    mode="explicit",
    source=KERNEL_SOURCE,
)
def kernel(
    arg0: pto.ptr(pto.f32, "gm"),
    arg1: pto.ptr(pto.f32, "gm"),
    arg2: pto.ptr(pto.ui8, "gm"),
    arg3: pto.ptr(pto.f32, "gm"),
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
    name='legacy_micro_op_compare_select_vsel_f32_plds_us_pintlv_pbitcast',
    grid=1,
    kernel=kernel,
    make_case=make_case,
    check=check_case,
)]

auto_main(globals())
