#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Migrated from test/vpto/cases/micro-op/vector-load-store/vsts-packed-dist-mask-granularity."""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

from common import auto_main
from ptodsl import pto

KERNEL_SOURCE = (
    '// -----------------------------------------------------------------------------\n// case: micro-op/vector-l'
    'oad-store/vsts-packed-dist-mask-granularity\n// family: micro-op/vector-load-store\n// target_ops: pto.vlds,'
    ' pto.vsts\n// scenarios: packed-store, pk-b64, pk4-b32, b32-mask-granularity\n// ---------------------------'
    '--------------------------------------------------\nmodule attributes {pto.target_arch = "a5", pto.kernel_ki'
    'nd = #pto.kernel_kind<vector>} {\n  func.func @vsts_packed_dist_mask_granularity_kernel(\n      %arg0: !pto.'
    'ptr<si64, gm>,\n      %arg1: !pto.ptr<f32, gm>,\n      %arg2: !pto.ptr<ui8, gm>,\n      %arg3: !pto.ptr<ui8,'
    ' gm>) attributes {pto.kernel} {\n    %c0 = arith.constant 0 : index\n    %c2 = arith.constant 2 : index\n   '
    ' %c16 = arith.constant 16 : index\n    %c512 = arith.constant 512 : index\n    %c0_i64 = arith.constant 0 : '
    'i64\n    %c1_i64 = arith.constant 1 : i64\n    %c32_i64 = arith.constant 32 : i64\n    %c64_i64 = arith.cons'
    'tant 64 : i64\n    %c256_i64 = arith.constant 256 : i64\n    %c8192_i64 = arith.constant 8192 : i64\n    %c1'
    '2288_i64 = arith.constant 12288 : i64\n    %c16384_i64 = arith.constant 16384 : i64\n\n    %ub_i64_in = pto.'
    'castptr %c0_i64 : i64 -> !pto.ptr<si64, ub>\n    %ub_f32_out = pto.castptr %c8192_i64 : i64 -> !pto.ptr<f32,'
    ' ub>\n    %ub_u8_in = pto.castptr %c12288_i64 : i64 -> !pto.ptr<ui8, ub>\n    %ub_u8_pk4 = pto.castptr %c163'
    '84_i64 : i64 -> !pto.ptr<ui8, ub>\n\n    pto.mte_gm_ub %arg0, %ub_i64_in, %c0_i64, %c256_i64\n      nburst(%'
    'c32_i64, %c256_i64, %c256_i64)\n      : !pto.ptr<si64, gm>, !pto.ptr<si64, ub>, i64, i64, i64, i64, i64\n   '
    ' pto.mte_gm_ub %arg2, %ub_u8_in, %c0_i64, %c256_i64\n      nburst(%c1_i64, %c256_i64, %c256_i64)\n      : !p'
    'to.ptr<ui8, gm>, !pto.ptr<ui8, ub>, i64, i64, i64, i64, i64\n\n    pto.set_flag["PIPE_MTE2", "PIPE_V", "EVEN'
    'T_ID0"]\n    pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n\n    pto.vecscope {\n      %mask_b32 = pto.'
    'pset_b32 "PAT_ALL" : !pto.mask<b32>\n\n      scf.for %store_offset = %c0 to %c512 step %c16 {\n        %inpu'
    't_offset = arith.muli %store_offset, %c2 : index\n        %loaded_i64 = pto.vlds %ub_i64_in[%input_offset] :'
    ' !pto.ptr<si64, ub>, index -> !pto.vreg<32xsi64>\n        %converted = pto.vcvt %loaded_i64, %mask_b32 {rnd = "R", '
    'part = "EVEN"}\n            : !pto.vreg<32xsi64>, !pto.mask<b32> -> !pto.vreg<64xf32>\n        pto.vsts %con'
    'verted, %ub_f32_out[%store_offset], %mask_b32 {dist = "PK_B64"}\n            : !pto.vreg<64xf32>, !pto.ptr<f'
    '32, ub>, index, !pto.mask<b32>\n      }\n\n      %packed_u8 = pto.vlds %ub_u8_in[%c0] : !pto.ptr<ui8, ub>, index -> !pto.v'
    'reg<256xui8>\n      pto.vsts %packed_u8, %ub_u8_pk4[%c0], %mask_b32 {dist = "PK4_B32"}\n          : !pto.vre'
    'g<256xui8>, !pto.ptr<ui8, ub>, index, !pto.mask<b32>\n    }\n\n    pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]'
    '\n    pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n    pto.mte_ub_gm %ub_f32_out, %arg1, %c64_i64\n   '
    '   nburst(%c32_i64, %c64_i64, %c64_i64)\n      : !pto.ptr<f32, ub>, !pto.ptr<f32, gm>, i64, i64, i64, i64\n '
    '   pto.mte_ub_gm %ub_u8_pk4, %arg3, %c64_i64\n      nburst(%c1_i64, %c64_i64, %c64_i64)\n      : !pto.ptr<ui'
    '8, ub>, !pto.ptr<ui8, gm>, i64, i64, i64, i64\n    pto.barrier #pto.pipe<PIPE_ALL>\n    return\n  }\n}\n'
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

    # case: micro-op/vector-load-store/vsts-packed-dist-mask-granularity
    # family: micro-op/vector-load-store
    # target_ops: pto.vlds, pto.vsts
    # scenarios: packed-store, pk-b64, pk4-b32, b32-mask-granularity

    import argparse
    from pathlib import Path

    import numpy as np


    I64_ELEMS = 1024
    F32_ELEMS = 512
    U8_ELEMS = 256
    SEED = 910


    def _generate(seed: int) -> dict:
        rng = np.random.default_rng(seed)
        edge = np.array(
            [
                -(1 << 31),
                -(1 << 24) - 3,
                -(1 << 24),
                -(1 << 24) + 1,
                -65537,
                -32768,
                -1,
                0,
                1,
                32767,
                65537,
                (1 << 24) - 1,
                1 << 24,
                (1 << 24) + 1,
                (1 << 31) - 2,
                (1 << 31) - 1,
            ],
            dtype=np.int32,
        )
        i64_base = rng.integers(
            np.iinfo(np.int32).min, np.iinfo(np.int32).max, size=I64_ELEMS, dtype=np.int32
        )
        i64_base[: edge.size] = edge
        v1 = i64_base.astype(np.int64)

        v3 = rng.integers(0, 256, size=U8_ELEMS, dtype=np.uint8)
        ramp = np.arange(U8_ELEMS, dtype=np.uint16)
        v3 = ((v3.astype(np.uint16) + ramp) & 0xFF).astype(np.uint8)

        golden_v2 = np.concatenate(
            [i64_base[offset : offset + 16] for offset in range(0, I64_ELEMS, 32)]
        ).astype(np.float32)
        golden_v4 = np.zeros(U8_ELEMS, dtype=np.uint8)
        golden_v4[: U8_ELEMS // 4] = v3[0::4]

        _files["v1.bin"] = v1
        _files["v2.bin"] = np.zeros(F32_ELEMS, dtype=np.float32)
        _files["v3.bin"] = v3
        _files["v4.bin"] = np.zeros(U8_ELEMS, dtype=np.uint8)
        _files["golden_v2.bin"] = golden_v2
        _files["golden_v4.bin"] = golden_v4


    def main() -> None:
        parser = argparse.ArgumentParser(
            description="Generate inputs/golden for packed vsts mask granularity validation."
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

    # case: micro-op/vector-load-store/vsts-packed-dist-mask-granularity
    # family: micro-op/vector-load-store
    # target_ops: pto.vlds, pto.vsts
    # scenarios: packed-store, pk-b64, pk4-b32, b32-mask-granularity

    import os
    import sys

    import numpy as np


    def compare_bin(golden_path: str, output_path: str, dtype) -> bool:
        if not os.path.exists(golden_path):
            print(f"[ERROR] Golden missing: {golden_path}")
            return False
        if not os.path.exists(output_path):
            print(f"[ERROR] Output missing: {output_path}")
            return False
        golden = np.fromfile(golden_path, dtype=dtype)
        output = np.fromfile(output_path, dtype=dtype)
        if golden.shape != output.shape:
            print(f"[ERROR] Shape mismatch for {output_path}: golden={golden.shape}, output={output.shape}")
            return False
        if np.array_equal(golden, output):
            return True
        diff = np.nonzero(golden != output)[0]
        idx = int(diff[0]) if diff.size else 0
        print(f"[ERROR] First mismatch for {output_path} at idx={idx}: golden={golden[idx]}, output={output[idx]}")
        return False


    def main() -> None:
        strict = os.getenv("COMPARE_STRICT", "1") != "0"
        ok = compare_bin("golden_v2.bin", "v2.bin", np.float32)
        ok = compare_bin("golden_v4.bin", "v4.bin", np.uint8) and ok
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
        'golden_v2.bin',
        'v2.bin'
    ),
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
        'si64',
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
        'ui8',
        'arg3'
    )
]
SCALARS = []
SCALAR_SOURCES = []
SCALAR_DTYPES = []


@pto.jit(
    name='vsts_packed_dist_mask_granularity_kernel',
    target="a5",
    backend="vpto",
    mode="explicit",
    source=KERNEL_SOURCE,
)
def kernel(
    arg0: pto.ptr(pto.si64, "gm"),
    arg1: pto.ptr(pto.f32, "gm"),
    arg2: pto.ptr(pto.ui8, "gm"),
    arg3: pto.ptr(pto.ui8, "gm"),
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
    name='legacy_micro_op_vector_load_store_vsts_packed_dist_mask_granularity',
    grid=1,
    kernel=kernel,
    make_case=make_case,
    check=check_case,
)]

auto_main(globals())
