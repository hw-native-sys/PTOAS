#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Migrated from test/vpto/cases/vmi_new/short-extend-store-f8-to-f32."""

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
    '\n// Short dense f8e4m3 reads widened into short dense f32 stores at four sizes\n// and offsets.  The narrow'
    ' source is packed four to a 32-bit carrier lane\n// (lane_stride = 4) while the widened result is a contiguo'
    'us group packet, so a\n// value only lands in the right lanes when the widening picks the semantic\n// lanes'
    ' of its packed source.  Each shape writes a disjoint region; the rest of\n// the destination buffer keeps it'
    's sentinel, so a wrong lane, a wrong region or\n// an overrun shows up as a mismatch.\n\nmodule attributes {'
    'pto.target_arch = "a5", pto.kernel_kind = #pto.kernel_kind<vector>} {\n  func.func @vmi_short_extend_store_f'
    '8_kernel(%src_gm: !pto.ptr<ui8, gm>,\n                                              %dst_gm: !pto.ptr<f32, g'
    'm>) attributes {pto.kernel} {\n    %c0 = arith.constant 0 : index\n    %c0_i64 = arith.constant 0 : i64\n   '
    ' %c1_i64 = arith.constant 1 : i64\n    %c256_i64 = arith.constant 256 : i64\n    %c1024_i64 = arith.constant'
    ' 1024 : i64\n    %c2048_i64 = arith.constant 2048 : i64\n    %s0 = arith.constant 0 : index\n    %d0 = arith'
    '.constant 0 : index\n    %s32 = arith.constant 32 : index\n    %d32 = arith.constant 32 : index\n    %s64 = '
    'arith.constant 64 : index\n    %d64 = arith.constant 64 : index\n    %s96 = arith.constant 96 : index\n    %'
    'd96 = arith.constant 96 : index\n\n    %ub_src_u8 = pto.castptr %c0_i64 : i64 -> !pto.ptr<ui8, ub>\n    %ub_'
    'src_f8 = pto.castptr %c0_i64 : i64 -> !pto.ptr<f8E4M3FN, ub>\n    %ub_dst = pto.castptr %c2048_i64 : i64 -> '
    '!pto.ptr<f32, ub>\n\n    pto.mte_gm_ub %src_gm, %ub_src_u8, %c0_i64, %c256_i64\n      nburst(%c1_i64, %c256_'
    'i64, %c256_i64)\n      : !pto.ptr<ui8, gm>, !pto.ptr<ui8, ub>, i64, i64, i64, i64, i64\n    pto.mte_gm_ub %d'
    'st_gm, %ub_dst, %c0_i64, %c1024_i64\n      nburst(%c1_i64, %c1024_i64, %c1024_i64)\n      : !pto.ptr<f32, gm'
    '>, !pto.ptr<f32, ub>, i64, i64, i64, i64, i64\n\n    pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n    p'
    'to.wait_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n\n    pto.vecscope {\n      %v1 = pto.vmi.vload %ub_src_f8'
    '[%s0] : !pto.ptr<f8E4M3FN, ub>, index -> !pto.vmi.vreg<1xf8E4M3FN>\n      %w1 = pto.vmi.vcvt %v1 : !pto.vmi.vreg<1'
    'xf8E4M3FN> -> !pto.vmi.vreg<1xf32>\n      pto.vmi.vstore %w1, %ub_dst[%d0] : !pto.vmi.vreg<1xf32>, !pto.ptr<f'
    '32, ub>, index\n      %v2 = pto.vmi.vload %ub_src_f8[%s32] : !pto.ptr<f8E4M3FN, ub>, index -> !pto.vmi.vreg<2xf8E4'
    'M3FN>\n      %w2 = pto.vmi.vcvt %v2 : !pto.vmi.vreg<2xf8E4M3FN> -> !pto.vmi.vreg<2xf32>\n      pto.vmi.vstore %w2,'
    ' %ub_dst[%d32] : !pto.vmi.vreg<2xf32>, !pto.ptr<f32, ub>, index\n      %v4 = pto.vmi.vload %ub_src_f8[%s64] : !pto'
    '.ptr<f8E4M3FN, ub>, index -> !pto.vmi.vreg<4xf8E4M3FN>\n      %w4 = pto.vmi.vcvt %v4 : !pto.vmi.vreg<4xf8E4M3FN> -'
    '> !pto.vmi.vreg<4xf32>\n      pto.vmi.vstore %w4, %ub_dst[%d64] : !pto.vmi.vreg<4xf32>, !pto.ptr<f32, ub>, index\n'
    '      %v8 = pto.vmi.vload %ub_src_f8[%s96] : !pto.ptr<f8E4M3FN, ub>, index -> !pto.vmi.vreg<8xf8E4M3FN>\n      %w8'
    ' = pto.vmi.vcvt %v8 : !pto.vmi.vreg<8xf8E4M3FN> -> !pto.vmi.vreg<8xf32>\n      pto.vmi.vstore %w8, %ub_dst[%d96] :'
    ' !pto.vmi.vreg<8xf32>, !pto.ptr<f32, ub>, index\n    }\n\n    pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n  '
    '  pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n    pto.mte_ub_gm %ub_dst, %dst_gm, %c1024_i64\n      nb'
    'urst(%c1_i64, %c1024_i64, %c1024_i64)\n      : !pto.ptr<f32, ub>, !pto.ptr<f32, gm>, i64, i64, i64, i64\n   '
    ' pto.barrier #pto.pipe<PIPE_ALL>\n    return\n  }\n}\n'
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

    """Golden for the short packed f8e4m3 -> f32 widen stores with sentinel gaps."""

    import argparse
    from pathlib import Path

    import numpy as np

    ELEMS = 256
    SEED = 23
    SENTINEL = np.float32(-123.25)

    # f8e4m3fn bit patterns for values that are exactly representable in f32, so the
    # widening is exact and the comparison can be exact.
    F8_VALUES = {
        0x00: 0.0,
        0x38: 1.0,
        0xB8: -1.0,
        0x30: 0.5,
        0x40: 2.0,
        0xC0: -2.0,
        0x48: 4.0,
        0xC8: -4.0,
    }
    # (n, source element offset, destination element offset); every offset is 32B
    # aligned for its element type (see the kernel's bound reads and stores).
    SHAPES = ((1, 0, 0), (2, 32, 32), (4, 64, 64), (8, 96, 96))


    def decode(byte: int) -> np.float32:
        if byte not in F8_VALUES:
            raise ValueError(f"unexpected f8e4m3fn byte 0x{byte:02x} in the source")
        return np.float32(F8_VALUES[byte])


    def _generate(seed: int) -> dict:
        del seed  # the pattern is deterministic; the seed keeps the CLI uniform
        pattern = np.array(sorted(F8_VALUES), dtype=np.uint8)
        src = np.tile(pattern, (ELEMS + pattern.size - 1) // pattern.size)[:ELEMS]
        dst = np.full(ELEMS, SENTINEL, dtype=np.float32)
        golden = dst.copy()
        for n, src_offset, dst_offset in SHAPES:
            for k in range(n):
                golden[dst_offset + k] = decode(int(src[src_offset + k]))

        _files["v1.bin"] = src
        _files["v2.bin"] = dst
        _files["golden_v2.bin"] = golden


    def main() -> None:
        parser = argparse.ArgumentParser()
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

    """Exact comparison of the widened short f8 store against its golden."""

    import sys

    import numpy as np


    def main() -> None:
        golden = np.fromfile("golden_v2.bin", dtype=np.float32)
        output = np.fromfile("v2.bin", dtype=np.float32)
        if golden.shape != output.shape or not np.array_equal(golden, output):
            diff = np.nonzero(golden != output)[0]
            idx = int(diff[0]) if diff.size else -1
            print(
                f"[ERROR] compare failed idx={idx} "
                f"golden={golden[idx] if idx >= 0 else 'n/a'} "
                f"output={output[idx] if idx >= 0 else 'n/a'}"
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

BUFFERS = [
    (
        'v1',
        'ui8',
        'src_gm'
    ),
    (
        'v2',
        'f32',
        'dst_gm'
    )
]
SCALARS = []
SCALAR_SOURCES = []
SCALAR_DTYPES = []


@pto.jit(
    name='vmi_short_extend_store_f8_kernel',
    target="a5",
    backend="vpto",
    mode="explicit",
    source=KERNEL_SOURCE,
)
def kernel(
    src_gm: pto.ptr(pto.ui8, "gm"),
    dst_gm: pto.ptr(pto.f32, "gm"),
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
    name='legacy_vmi_new_short_extend_store_f8_to_f32',
    grid=1,
    kernel=kernel,
    make_case=make_case,
    check=check_case,
)]

auto_main(globals())
