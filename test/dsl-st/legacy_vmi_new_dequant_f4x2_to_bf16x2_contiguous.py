#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Migrated from test/vpto/cases/vmi_new/dequant-f4x2-to-bf16x2-contiguous."""

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
    '\n// -----------------------------------------------------------------------------\n// case: vmi_new/dequant'
    '-f4x2-to-bf16x2-contiguous\n// family: conversion\n// target_ops: pto.vmi.vcvt (f4E1M2x2 -> bf16x2)\n// scen'
    'arios: f4x2 contiguous load -> vcvt (widen, Packed4 P0-P3) -> bf16x2 d4 store\n// --------------------------'
    '---------------------------------------------------\nmodule attributes {pto.target_arch = "a5", pto.kernel_k'
    'ind = #pto.kernel_kind<vector>} {\n  func.func @vmi_dequant_f4x2_to_bf16x2_contiguous_kernel(%src_gm: !pto.p'
    'tr<!pto.f4E1M2x2, gm>,\n                                                          %dst_gm: !pto.ptr<bf16, gm'
    '>) attributes {pto.kernel} {\n    %c0 = arith.constant 0 : index\n    %c0_i64 = arith.constant 0 : i64\n    '
    '%c1_i64 = arith.constant 1 : i64\n    %c256_i64 = arith.constant 256 : i64\n    %c1024_i64 = arith.constant '
    '1024 : i64\n\n    %ub_src = pto.castptr %c0_i64 : i64 -> !pto.ptr<!pto.f4E1M2x2, ub>\n    %ub_dst_bf16 = pto'
    '.castptr %c1024_i64 : i64 -> !pto.ptr<bf16, ub>\n    %ub_dst = pto.castptr %c1024_i64 : i64 -> !pto.ptr<!pto'
    '.bf16x2, ub>\n\n    // src: 256 f4x2 = 256 bytes; dst: 256 bf16x2 = 1024 bytes.\n    pto.mte_gm_ub %src_gm, '
    '%ub_src, %c0_i64, %c256_i64\n      nburst(%c1_i64, %c256_i64, %c256_i64)\n      : !pto.ptr<!pto.f4E1M2x2, gm'
    '>, !pto.ptr<!pto.f4E1M2x2, ub>, i64, i64, i64, i64, i64\n    pto.mte_gm_ub %dst_gm, %ub_dst_bf16, %c0_i64, %'
    'c1024_i64\n      nburst(%c1_i64, %c1024_i64, %c1024_i64)\n      : !pto.ptr<bf16, gm>, !pto.ptr<bf16, ub>, i6'
    '4, i64, i64, i64, i64\n\n    pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n    pto.wait_flag["PIPE_MTE2"'
    ', "PIPE_V", "EVENT_ID0"]\n\n    pto.vecscope {\n      %packed = pto.vmi.vload %ub_src[%c0] : !pto.ptr<!pto.f'
    '4E1M2x2, ub>, index -> !pto.vmi.vreg<256x!pto.f4E1M2x2>\n      %wide = pto.vmi.vcvt %packed\n          : !pto.vmi.'
    'vreg<256x!pto.f4E1M2x2> -> !pto.vmi.vreg<256x!pto.bf16x2>\n      pto.vmi.vstore %wide, %ub_dst[%c0]\n        '
    '  : !pto.vmi.vreg<256x!pto.bf16x2>, !pto.ptr<!pto.bf16x2, ub>, index\n    }\n\n    pto.set_flag["PIPE_V", "PIPE_MT'
    'E3", "EVENT_ID0"]\n    pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n    pto.mte_ub_gm %ub_dst_bf16, %ds'
    't_gm, %c1024_i64\n      nburst(%c1_i64, %c1024_i64, %c1024_i64)\n      : !pto.ptr<bf16, ub>, !pto.ptr<bf16, '
    'gm>, i64, i64, i64, i64\n    pto.barrier #pto.pipe<PIPE_ALL>\n    return\n  }\n}\n'
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

    # Reference for f4E1M2x2 -> bf16x2 (contiguous path).
    # Each f4E1M2x2 byte packs two 4-bit FP4 E1M2 codes: low nibble = even bf16
    # lane, high nibble = odd bf16 lane (mirrors the quant case PAIR_HI_FIRST).
    # Each code dequantizes to one bf16 value; the pair becomes one bf16x2.
    #
    # Calibration note: the E1M2 table and nibble order below are provisional
    # oracles mirroring quant-bf16x2-to-f4x2-contiguous. Confirm them on the first
    # A5 NPU run before treating a strict golden mismatch as a compiler regression.

    import argparse
    from pathlib import Path

    import numpy as np

    SRC_ELEMS = 256          # f4x2 input bytes
    DST_ELEMS = 512          # bf16 output lanes

    # Magnitude values for E1M2 code 0..7 (code bit3 = sign). IEEE-style table.
    E1M2_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
    PAIR_HI_FIRST = True     # True: byte = (f4(odd) << 4) | f4(even); False: reversed

    # Input f4 codes (all 16 codes, values exactly representable in the table).
    CODES = np.array(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], dtype=np.uint8
    )


    def f32_to_bf16_bits(values: np.ndarray) -> np.ndarray:
        """Round f32 to nearest bf16 and return its 16-bit pattern."""
        f32 = values.astype(np.float32)
        bits = f32.view(np.uint32)
        # round-to-nearest-even on the low 16 bits
        lsb = (bits >> 16) & 1
        rounding = 0x7FFF + lsb
        bf16 = (bits + rounding) >> 16
        return bf16.astype(np.uint16)


    def dequantize_f4e1m2(codes: np.ndarray) -> np.ndarray:
        """f4E1M2 code (0..15) -> f32 value."""
        sign = ((codes >> 3) & 1).astype(np.int32)
        mag = np.array(E1M2_VALUES, dtype=np.float32)[(codes & 0x7).astype(np.int32)]
        values = np.where(sign == 1, -mag, mag).astype(np.float32)
        return values


    def _generate(seed: int) -> dict:
        # SRC_ELEMS f4x2 bytes pack 2 * SRC_ELEMS f4 codes.
        num_codes = SRC_ELEMS * 2
        repeats = (num_codes + len(CODES) - 1) // len(CODES)
        codes = np.tile(CODES, repeats)[:num_codes].astype(np.uint8)

        # Pack into f4x2 bytes: low nibble = even lane code, high nibble = odd.
        even_codes = codes[0::2]
        odd_codes = codes[1::2]
        assert even_codes.size == odd_codes.size == SRC_ELEMS
        if PAIR_HI_FIRST:
            src_bytes = ((odd_codes.astype(np.uint8) << 4) | even_codes).astype(np.uint8)
        else:
            src_bytes = ((even_codes.astype(np.uint8) << 4) | odd_codes).astype(np.uint8)

        # Dequantize: bf16[2j] = low nibble, bf16[2j+1] = high nibble.
        vals_even = dequantize_f4e1m2(even_codes)
        vals_odd = dequantize_f4e1m2(odd_codes)
        out_f32 = np.empty(DST_ELEMS, dtype=np.float32)
        out_f32[0::2] = vals_even
        out_f32[1::2] = vals_odd
        dst_bf16 = f32_to_bf16_bits(out_f32)

        src_buf = src_bytes.astype(np.uint8)
        dst_buf = np.full(DST_ELEMS, 0xA5, dtype=np.uint16)

        _files["v1.bin"] = src_buf
        _files["v2.bin"] = dst_buf
        _files["golden_v2.bin"] = dst_bf16


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
        golden = np.fromfile("golden_v2.bin", dtype=np.uint16)
        output = np.fromfile("v2.bin", dtype=np.uint16)
        if golden.shape != output.shape:
            print(f"[ERROR] size mismatch golden={golden.size} output={output.size}")
            sys.exit(2)
        if not np.array_equal(golden, output):
            diff = np.nonzero(golden != output)[0]
            idx = int(diff[0])
            print(
                f"[ERROR] compare failed idx={idx} "
                f"golden=0x{int(golden[idx]):04x} "
                f"output=0x{int(output[idx]):04x}"
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
        'f4E1M2x2',
        'src_gm'
    ),
    (
        'v2',
        'bf16',
        'dst_gm'
    )
]
SCALARS = []
SCALAR_SOURCES = []
SCALAR_DTYPES = []


@pto.jit(
    name='vmi_dequant_f4x2_to_bf16x2_contiguous_kernel',
    target="a5",
    backend="vpto",
    mode="explicit",
    source=KERNEL_SOURCE,
)
def kernel(
    src_gm: pto.ptr(pto.f4e1m2x2, "gm"),
    dst_gm: pto.ptr(pto.bf16, "gm"),
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
    name='legacy_vmi_new_dequant_f4x2_to_bf16x2_contiguous',
    grid=1,
    kernel=kernel,
    make_case=make_case,
    check=check_case,
)]

auto_main(globals())
