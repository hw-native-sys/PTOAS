#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Migrated from test/vpto/cases/cast-spine/quant-dequant-bf16-f8-bf16.

The C++ host of this case launched more than one entry point, so the module
compiles each one and a run() hook issues the launches in order.
"""

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
    '\n// RUN: ptoas --pto-arch a5 --pto-backend=vpto --emit-vpto %s -o /dev/null\n//\n// Direction spine: <up, d'
    'own, up, down> closing on the starting element type,\n// exactly as in roundtrip-bf16-f8-bf16, but here the '
    'narrow f8E4M3FN result is\n// *materialised* (stored to UB and copied out to GM) the way a real\n// quantise'
    '/dequantise kernel does:\n//\n//   bf16 --vcvt--> f32 --vmul--> f32 --vcvt--> f8E4M3FN   (quantise, stored)'
    '\n//   f8E4M3FN --vcvt--> f32 --vmul--> f32 --vcvt--> bf16   (dequantise, stored)\n//\n// The four cast legs'
    ' are the same closed nested round trip as the lit template\n// test/lit/vmi_new/vmi_layout_assignment_cast_r'
    'oundtrip.pto, so the direction\n// spine fires for the same reason; the difference is that the deinterleaved'
    ' /\n// lane-stride choice for the narrow element type now also decides how the f8\n// value has to be reshuf'
    'fled to reach contiguous UB memory.\n\nmodule attributes {pto.target_arch = "a5", pto.kernel_kind = #pto.ker'
    'nel_kind<vector>} {\n  func.func @vmi_cast_spine_quant_dequant_bf16_f8_kernel(%src_gm: !pto.ptr<bf16, gm>,\n'
    '                                                         %q_gm: !pto.ptr<f8E4M3FN, gm>,\n                   '
    '                                      %y_gm: !pto.ptr<bf16, gm>) attributes {pto.kernel} {\n    %c0 = arith.'
    'constant 0 : index\n    %c1 = arith.constant 1 : index\n    %c_repeat = arith.constant 4 : index\n    %c16 ='
    ' arith.constant 16 : index\n    %c64 = arith.constant @@TILES@@ : index\n    %c256 = arith.constant 256 : in'
    'dex\n    %c0_i64 = arith.constant 0 : i64\n    %c1_i64 = arith.constant 1 : i64\n    %c16384_i64 = arith.con'
    'stant @@SZ_16384@@ : i64\n    %c32768_i64 = arith.constant @@SZ_32768@@ : i64\n    %c49152_i64 = arith.const'
    'ant @@SZ_49152@@ : i64\n    %quant_scale = arith.constant 1.500000e+00 : f32\n    %dequant_scale = arith.con'
    'stant 2.000000e+00 : f32\n\n    %ub_src = pto.castptr %c0_i64 : i64 -> !pto.ptr<bf16, ub>\n    %ub_q = pto.c'
    'astptr %c32768_i64 : i64 -> !pto.ptr<f8E4M3FN, ub>\n    %ub_y = pto.castptr %c49152_i64 : i64 -> !pto.ptr<bf'
    '16, ub>\n\n    pto.mte_gm_ub %src_gm, %ub_src, %c0_i64, %c32768_i64\n      nburst(%c1_i64, %c32768_i64, %c32'
    '768_i64)\n      : !pto.ptr<bf16, gm>, !pto.ptr<bf16, ub>, i64, i64, i64, i64, i64\n\n    pto.set_flag["PIPE_'
    'MTE2", "PIPE_V", "EVENT_ID0"]\n    pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n\n    pto.vecscope {\n'
    '      scf.for %rep = %c0 to %c_repeat step %c1 {\n        scf.for %tile = %c0 to %c64 step %c1 {\n          '
    '%elem = arith.muli %tile, %c256 : index\n          %x_bf16 = pto.vmi.vload %ub_src[%elem]\n              : !'
    'pto.ptr<bf16, ub>, index -> !pto.vmi.vreg<256xbf16>\n          %x = pto.vmi.vcvt %x_bf16\n              : !pto.vmi'
    '.vreg<256xbf16> -> !pto.vmi.vreg<256xf32>\n          %qscale = pto.vmi.vbrc %quant_scale\n              : f32'
    ' -> !pto.vmi.vreg<256xf32>\n          %scaled = pto.vmi.vmul %x, %qscale\n              : !pto.vmi.vreg<256x'
    'f32>, !pto.vmi.vreg<256xf32>\n              -> !pto.vmi.vreg<256xf32>\n          %q = pto.vmi.vcvt %scaled {'
    'saturate = "SAT"}\n              : !pto.vmi.vreg<256xf32> -> !pto.vmi.vreg<256xf8E4M3FN>\n          pto.vmi.'
    'vstore %q, %ub_q[%elem]\n              : !pto.vmi.vreg<256xf8E4M3FN>, !pto.ptr<f8E4M3FN, ub>, index\n          %r '
    '= pto.vmi.vcvt %q\n              : !pto.vmi.vreg<256xf8E4M3FN> -> !pto.vmi.vreg<256xf32>\n          %dscale ='
    ' pto.vmi.vbrc %dequant_scale\n              : f32 -> !pto.vmi.vreg<256xf32>\n          %restored = pto.vmi.v'
    'mul %r, %dscale\n              : !pto.vmi.vreg<256xf32>, !pto.vmi.vreg<256xf32>\n              -> !pto.vmi.v'
    'reg<256xf32>\n          %y = pto.vmi.vcvt %restored {saturate = "SAT"}\n              : !pto.vmi.vreg<256xf3'
    '2> -> !pto.vmi.vreg<256xbf16>\n          pto.vmi.vstore %y, %ub_y[%elem]\n              : !pto.vmi.vreg<256x'
    'bf16>, !pto.ptr<bf16, ub>, index\n        }\n      }\n    }\n\n    pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"'
    ']\n    pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n    pto.mte_ub_gm %ub_q, %q_gm, %c16384_i64\n      '
    'nburst(%c1_i64, %c16384_i64, %c16384_i64)\n      : !pto.ptr<f8E4M3FN, ub>, !pto.ptr<f8E4M3FN, gm>, i64, i64,'
    ' i64, i64\n    pto.mte_ub_gm %ub_y, %y_gm, %c32768_i64\n      nburst(%c1_i64, %c32768_i64, %c32768_i64)\n   '
    '   : !pto.ptr<bf16, ub>, !pto.ptr<bf16, gm>, i64, i64, i64, i64\n    pto.barrier #pto.pipe<PIPE_ALL>\n    re'
    'turn\n  }\n}\n'
)
_TOKENS = {
    '@@SZ_16384@@': (
        16384,
        256
    ),
    '@@SZ_32768@@': (
        32768,
        512
    ),
    '@@SZ_49152@@': (
        49152,
        768
    ),
    '@@TILES@@': (
        64,
        1
    )
}
for _token, (_big, _small) in _TOKENS.items():
    KERNEL_SOURCE = KERNEL_SOURCE.replace(
        _token, str(_big if os.environ.get("PTOAS_DSL_BIG") else _small))


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

    """Golden data for quant-dequant-bf16-f8-bf16.

    Chain:  bf16 --vcvt--> f32 --*1.5--> f8E4M3FN (stored)
            f8E4M3FN --vcvt--> f32 --*2.0--> bf16 (stored)

    Every input value multiplied by 1.5 stays exactly representable in f8E4M3FN
    (magnitude <= 8, mantissa <= 3 bits) and the dequantised product is exact in
    bf16, so both outputs are lossless for this input set and the oracle is
    independent of the hardware rounding mode.
    """

    import argparse
    from pathlib import Path

    import ml_dtypes
    import numpy as np

    ELEMS = 16384 if os.environ.get("PTOAS_DSL_BIG") else 256
    QUANT_SCALE = np.float32(1.5)
    DEQUANT_SCALE = np.float32(2.0)
    BF16 = ml_dtypes.bfloat16
    F8E4M3FN = ml_dtypes.float8_e4m3fn
    EXACT_VALUES = np.array(
        [0.0, 1.0, -1.0, 0.5, -0.5, 1.5, -1.5, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0],
        dtype=np.float32,
    )


    def _generate(seed: int) -> dict:
        rng = np.random.default_rng(1337)
        src = EXACT_VALUES[rng.integers(0, EXACT_VALUES.size, ELEMS)].astype(BF16)

        quantized = (src.astype(np.float32) * QUANT_SCALE).astype(F8E4M3FN)
        restored = (quantized.astype(np.float32) * DEQUANT_SCALE).astype(BF16)

        _files["v1.bin"] = src.view(np.uint16)
        _files["v2.bin"] = np.zeros(ELEMS, dtype=np.uint8)
        _files["v3.bin"] = np.zeros(ELEMS, dtype=np.uint16)
        _files["golden_v2.bin"] = quantized.view(np.uint8)
        _files["golden_v3.bin"] = restored.view(np.uint16)


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

    """Output check for quant-dequant-bf16-f8-bf16.

    v2.bin is the raw f8E4M3FN byte stream (bit-exact expected) and v3.bin the
    dequantised bf16 bit pattern (bit-exact expected): both chains are lossless for
    the chosen input set.
    """

    import sys

    import numpy as np


    def main() -> None:
        gold_q = np.fromfile("golden_v2.bin", dtype=np.uint8)
        out_q = np.fromfile("v2.bin", dtype=np.uint8)
        gold_y = np.fromfile("golden_v3.bin", dtype=np.uint16)
        out_y = np.fromfile("v3.bin", dtype=np.uint16)

        if gold_q.shape != out_q.shape or not np.array_equal(gold_q, out_q):
            bad = np.nonzero(gold_q != out_q)[0] if gold_q.shape == out_q.shape else []
            idx = int(bad[0]) if bad.size else -1
            print(f"[ERROR] f8 compare failed n_mismatch={bad.size} idx={idx} "
                  f"golden={gold_q[idx] if idx >= 0 else 'n/a'} "
                  f"output={out_q[idx] if idx >= 0 else 'n/a'}")
            sys.exit(2)

        if gold_y.shape != out_y.shape or not np.array_equal(gold_y, out_y):
            bad = np.nonzero(gold_y != out_y)[0] if gold_y.shape == out_y.shape else []
            idx = int(bad[0]) if bad.size else -1
            print(f"[ERROR] bf16 compare failed n_mismatch={bad.size} idx={idx} "
                  f"golden=0x{int(gold_y[idx]):04x} output=0x{int(out_y[idx]):04x}"
                  if idx >= 0 else "[ERROR] bf16 compare failed shape mismatch")
            sys.exit(2)

        print(f"[INFO] compare passed (bit-exact f8 n={gold_q.size}, bf16 n={gold_y.size})")


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
BUFFER_FILES = [
    (
        'v1',
        'bf16'
    ),
    (
        'v2',
        'f8e4m3'
    ),
    (
        'v3',
        'bf16'
    )
]
LAUNCHES = [
    (
        0,
        [
            0,
            1,
            2
        ],
        1,
        [],
        [
            'bf16',
            'f8e4m3',
            'bf16'
        ],
        [
            '0',
            '0',
            '0'
        ]
    ),
    (
        0,
        [
            0,
            1,
            2
        ],
        1,
        [],
        [
            'bf16',
            'f8e4m3',
            'bf16'
        ],
        [
            '0',
            '0',
            '0'
        ]
    )
]

def _dtype_map():
    import torch
    return {
        "int8": torch.int8, "int16": torch.int16, "int32": torch.int32, "int64": torch.int64,
        "uint8": torch.uint8, "uint16": torch.int16, "uint32": torch.int32, "uint64": torch.int64,
        "float16": torch.float16, "float32": torch.float32, "float64": torch.float64,
    }


@pto.jit(
    name='vmi_cast_spine_quant_dequant_bf16_f8_kernel',
    target="a5",
    backend="vpto",
    mode="explicit",
    source=KERNEL_SOURCE,
)
def kernel_0(
    src_gm: pto.ptr(pto.bf16, "gm"),
    q_gm: pto.ptr(pto.f8e4m3, "gm"),
    y_gm: pto.ptr(pto.bf16, "gm"),
):
    pass


KERNELS = [kernel_0]


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
    tensors = []
    for name, ty in BUFFER_FILES:
        key = "%s.bin" % name
        if key in produced:
            array = _as_buffer_array(produced[key], NUMPY_DTYPE[ty])
        else:
            golden_key = output_to_golden.get(key)
            if golden_key is None or golden_key not in produced:
                raise RuntimeError("golden did not produce %s" % key)
            array = np.ascontiguousarray(
                np.zeros_like(np.asarray(produced[golden_key]), dtype=np.dtype(NUMPY_DTYPE[ty])))
        tensors.append(torch.from_numpy(array).to("npu:0"))
    return tensors, produced


def run_case(device_inputs, launch_args, stream):
    import torch
    compiled = [kernel.compile() for kernel in KERNELS]
    for kernel_index, indices, grid, scalars, dtypes, offsets in LAUNCHES:
        args = []
        for index, ty, offset in zip(indices, dtypes, offsets):
            tensor = device_inputs[index]
            if offset not in ('', '0'):
                tensor = tensor[int(offset):]
            want = _dtype_map()[NUMPY_DTYPE[ty]]
            if tensor.dtype != want and tensor.numel() * tensor.element_size() % want.itemsize == 0:
                tensor = tensor.view(want)
            args.append(tensor)
        compiled[kernel_index][grid, stream](*args, *scalars)


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
    for index, (name, _ty) in enumerate(BUFFER_FILES):
        files["%s.bin" % name] = device_inputs[index].cpu().numpy()
    _apply_write_masks(files, produced)
    _run_compare(files)


CASES = [dict(
    name='legacy_cast_spine_quant_dequant_bf16_f8_bf16',
    grid=1,
    kernel=KERNELS[0],
    make_case=make_case,
    check=check_case,
    run=run_case,
)]

auto_main(globals())
