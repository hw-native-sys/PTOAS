#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Migrated from test/vpto/cases/vmi_new/quant-bf16x2-to-f4x2-rounding."""

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
    '\n// -----------------------------------------------------------------------------\n// case: vmi_new/quant-b'
    'f16x2-to-f4x2-rounding\n// family: conversion\n// target_ops: pto.vmi.vcvt (bf16x2 -> f4E1M2x2)\n// scenario'
    's: rounding {R,A,Z} exercise tie-breaking differences on\n//            f4-boundary input values. Three chun'
    'ks\n//            of the same 128-bf16 boundary input are converted with rounding\n//            R / A / Z a'
    'nd written to dst regions 0/1/2, so the golden can\n//            verify the hardware honors each rounding m'
    'ode (RNE/RTA/RTZ).\n//            H is rejected by the bf16x2 -> f4x2 VMI contract.\n// --------------------'
    '---------------------------------------------------------\nmodule attributes {pto.target_arch = "a5", pto.ke'
    'rnel_kind = #pto.kernel_kind<vector>} {\n  func.func @vmi_quant_bf16x2_to_f4x2_rounding_kernel(%src_gm: !pto'
    '.ptr<bf16, gm>,\n                                                      %dst_gm: !pto.ptr<ui8, gm>) attribute'
    's {pto.kernel} {\n    %c0 = arith.constant 0 : index\n    %c64 = arith.constant 64 : index\n    %c128 = arit'
    'h.constant 128 : index\n    %c256 = arith.constant 256 : index\n    %c0_i64 = arith.constant 0 : i64\n    %c'
    '1_i64 = arith.constant 1 : i64\n    %c192_i64 = arith.constant 192 : i64\n    %c768_i64 = arith.constant 768'
    ' : i64\n    %c1024_i64 = arith.constant 1024 : i64\n\n    %ub_src = pto.castptr %c0_i64 : i64 -> !pto.ptr<bf'
    '16, ub>\n    %ub_dst_u8 = pto.castptr %c1024_i64 : i64 -> !pto.ptr<ui8, ub>\n    %ub_dst_f4 = pto.castptr %c'
    '1024_i64 : i64 -> !pto.ptr<!pto.f4E1M2x2, ub>\n\n    // src: 384 bf16 = 768 bytes (3 chunks of 128); dst: 19'
    '2 bytes (3 x 64 f4x2).\n    pto.mte_gm_ub %src_gm, %ub_src, %c0_i64, %c768_i64\n      nburst(%c1_i64, %c768_'
    'i64, %c768_i64)\n      : !pto.ptr<bf16, gm>, !pto.ptr<bf16, ub>, i64, i64, i64, i64, i64\n    pto.mte_gm_ub '
    '%dst_gm, %ub_dst_u8, %c0_i64, %c192_i64\n      nburst(%c1_i64, %c192_i64, %c192_i64)\n      : !pto.ptr<ui8, '
    'gm>, !pto.ptr<ui8, ub>, i64, i64, i64, i64, i64\n\n    pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n   '
    ' pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n\n    pto.vecscope {\n      // chunk 0: rounding R (RNE)'
    '\n      %bf0 = pto.vmi.vload %ub_src[%c0] : !pto.ptr<bf16, ub>, index -> !pto.vmi.vreg<128xbf16>\n      %p0 = pto.'
    'vmi.vinterpret_cast %bf0\n          : !pto.vmi.vreg<128xbf16> -> !pto.vmi.vreg<64x!pto.bf16x2>\n      %r0 = p'
    'to.vmi.vcvt %p0 {rounding = "R"}\n          : !pto.vmi.vreg<64x!pto.bf16x2> -> !pto.vmi.vreg<64x!pto.f4E1M2x'
    '2>\n      pto.vmi.vstore %r0, %ub_dst_f4[%c0]\n          : !pto.vmi.vreg<64x!pto.f4E1M2x2>, !pto.ptr<!pto.f4'
    'E1M2x2, ub>, index\n\n      // chunk 1: rounding A (RTA)\n      %bf1 = pto.vmi.vload %ub_src[%c128] : !pto.ptr<bf1'
    '6, ub>, index -> !pto.vmi.vreg<128xbf16>\n      %p1 = pto.vmi.vinterpret_cast %bf1\n          : !pto.vmi.vreg<128x'
    'bf16> -> !pto.vmi.vreg<64x!pto.bf16x2>\n      %r1 = pto.vmi.vcvt %p1 {rounding = "A"}\n          : !pto.vmi.vr'
    'eg<64x!pto.bf16x2> -> !pto.vmi.vreg<64x!pto.f4E1M2x2>\n      pto.vmi.vstore %r1, %ub_dst_f4[%c64]\n         '
    ' : !pto.vmi.vreg<64x!pto.f4E1M2x2>, !pto.ptr<!pto.f4E1M2x2, ub>, index\n\n      // chunk 2: rounding Z (RTZ)\n    '
    '  %bf2 = pto.vmi.vload %ub_src[%c256] : !pto.ptr<bf16, ub>, index -> !pto.vmi.vreg<128xbf16>\n      %p2 = pto.vmi.'
    'vinterpret_cast %bf2\n          : !pto.vmi.vreg<128xbf16> -> !pto.vmi.vreg<64x!pto.bf16x2>\n      %r2 = pto.vm'
    'i.vcvt %p2 {rounding = "Z"}\n          : !pto.vmi.vreg<64x!pto.bf16x2> -> !pto.vmi.vreg<64x!pto.f4E1M2x2>\n '
    '     pto.vmi.vstore %r2, %ub_dst_f4[%c128]\n          : !pto.vmi.vreg<64x!pto.f4E1M2x2>, !pto.ptr<!pto.f4E1M'
    '2x2, ub>, index\n    }\n\n    pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n    pto.wait_flag["PIPE_V", "PIPE_'
    'MTE3", "EVENT_ID0"]\n    pto.mte_ub_gm %ub_dst_u8, %dst_gm, %c192_i64\n      nburst(%c1_i64, %c192_i64, %c192'
    '_i64)\n      : !pto.ptr<ui8, ub>, !pto.ptr<ui8, gm>, i64, i64, i64, i64\n    pto.barrier #pto.pipe<PIPE_ALL>'
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

    # Reference for bf16x2 -> f4E1M2x2 rounding {R,A,Z} on f4-boundary values.
    # Three chunks of 128 bf16 are quantized with RNE / RTA / RTZ respectively
    # (the same input set, so the output regions differ only by rounding mode).
    # Calibration note: the E1M2 table, R/A/Z interpretation, tie behavior, and
    # low/high nibble order below are provisional oracles. Confirm them on the first
    # A5 NPU run before treating a strict golden mismatch as a compiler regression.

    import argparse
    from pathlib import Path

    import numpy as np

    SRC_ELEMS = 384          # bf16 input lanes (3 chunks of 128)
    DST_ELEMS = 192          # f4x2 output bytes (3 regions of 64)

    # Magnitude values for E1M2 code 0..7 (code bit3 = sign). IEEE-style table.
    E1M2_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
    PAIR_HI_FIRST = True

    # f4-boundary values (exactly halfway between two representable E1M2 values),
    # so RNE / RTA / RTZ exercise tie-breaking differences.
    VALUES = np.array([0.125, 0.375, 0.625, 1.125, 1.375, 1.625, -0.125, -0.625],
                      dtype=np.float32)
    CHUNK = 128


    def f32_to_bf16_bits(values: np.ndarray) -> np.ndarray:
        f32 = values.astype(np.float32)
        bits = f32.view(np.uint32)
        lsb = (bits >> 16) & 1
        rounding = 0x7FFF + lsb
        bf16 = (bits + rounding) >> 16
        return bf16.astype(np.uint16)


    def bf16_to_f32(bits: np.ndarray) -> np.ndarray:
        return (bits.astype(np.uint32) << 16).view(np.float32)


    def quantize_f4e1m2(values: np.ndarray, mode: str) -> np.ndarray:
        """Nearest f4E1M2 code; ties (halfway) resolved by rounding mode."""
        table = np.array(E1M2_VALUES, dtype=np.float32)
        out = np.zeros(len(values), dtype=np.int32)
        for i, v in enumerate(values):
            sign = 1 if v < 0 else 0
            mag = abs(float(v))
            d = np.abs(table - mag)
            cands = np.where(np.isclose(d, d.min()))[0]
            if len(cands) == 1:
                idx = int(cands[0])
            else:
                lo, hi = int(cands[0]), int(cands[1])
                if mode == "rne":
                    idx = lo if lo % 2 == 0 else hi
                elif mode == "rta":
                    idx = hi
                elif mode == "rtz":
                    idx = lo
                else:
                    raise ValueError(mode)
            out[i] = (sign << 3) | idx
        return out.astype(np.uint8)


    def pack_pairs(f4: np.ndarray) -> np.ndarray:
        f4 = f4.astype(np.int32)
        even = f4[0::2]
        odd = f4[1::2]
        if PAIR_HI_FIRST:
            return ((odd << 4) | even).astype(np.uint8)
        return ((even << 4) | odd).astype(np.uint8)


    def _generate(seed: int) -> dict:
        repeats = (SRC_ELEMS + len(VALUES) - 1) // len(VALUES)
        src_f32 = np.tile(VALUES, repeats)[:SRC_ELEMS].astype(np.float32)
        src_bf16 = f32_to_bf16_bits(src_f32)

        modes = ["rne", "rta", "rtz"]
        dst = np.concatenate([
            pack_pairs(quantize_f4e1m2(bf16_to_f32(src_bf16)[c * CHUNK:(c + 1) * CHUNK], m))
            for c, m in enumerate(modes)
        ])
        assert dst.size == DST_ELEMS

        src_buf = src_bf16.astype("<u2")
        dst_buf = np.full(DST_ELEMS, 0xA5, dtype=np.uint8)

        _files["v1.bin"] = src_buf
        _files["v2.bin"] = dst_buf
        _files["golden_v2.bin"] = dst


    def main() -> None:
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

    import sys

    import numpy as np


    def main() -> None:
        golden = np.fromfile("golden_v2.bin", dtype=np.uint8)
        output = np.fromfile("v2.bin", dtype=np.uint8)
        if golden.shape != output.shape:
            print(f"[ERROR] size mismatch golden={golden.size} output={output.size}")
            sys.exit(2)
        if not np.array_equal(golden, output):
            diff = np.nonzero(golden != output)[0]
            idx = int(diff[0])
            print(
                f"[ERROR] compare failed idx={idx} "
                f"golden=0x{int(golden[idx]):02x} "
                f"output=0x{int(output[idx]):02x}"
            )
            sys.exit(2)
        print("[INFO] compare passed")


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
        'bf16',
        'src_gm'
    ),
    (
        'v2',
        'ui8',
        'dst_gm'
    )
]
SCALARS = []
SCALAR_SOURCES = []
SCALAR_DTYPES = []


@pto.jit(
    name='vmi_quant_bf16x2_to_f4x2_rounding_kernel',
    target="a5",
    backend="vpto",
    mode="explicit",
    source=KERNEL_SOURCE,
)
def kernel(
    src_gm: pto.ptr(pto.bf16, "gm"),
    dst_gm: pto.ptr(pto.ui8, "gm"),
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
    name='legacy_vmi_new_quant_bf16x2_to_f4x2_rounding',
    grid=1,
    kernel=kernel,
    make_case=make_case,
    check=check_case,
)]

auto_main(globals())
