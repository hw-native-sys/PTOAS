#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Migrated from test/vpto/cases/vmi_new/kernels/block-mx-quant-f16-e4m3-2x256."""

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
    '\nmodule attributes {pto.target_arch = "a5", pto.kernel_kind = #pto.kernel_kind<vector>} {\n  func.func @vmi'
    '_block_mx_quant_f16_e4m3_2x256_kernel(%src_gm: !pto.ptr<f16, gm>,\n                                         '
    '             %out_gm: !pto.ptr<ui8, gm>,\n                                                      %scale1_gm: '
    '!pto.ptr<ui8, gm>,\n                                                      %scale2_gm: !pto.ptr<ui8, gm>) att'
    'ributes {pto.kernel} {\n    %c0 = arith.constant 0 : index\n    %c1 = arith.constant 1 : index\n    %c2 = ar'
    'ith.constant 2 : index\n    %c32 = arith.constant 32 : index\n    %c8 = arith.constant 8 : index\n    %c64 ='
    ' arith.constant 64 : index\n    %c128 = arith.constant 128 : index\n    %c256 = arith.constant 256 : index\n'
    '    %c0_i64 = arith.constant 0 : i64\n    %c1_i64 = arith.constant 1 : i64\n    %c2_i64 = arith.constant 2 :'
    ' i64\n    %c8_i64 = arith.constant 8 : i64\n    %c32_i64 = arith.constant 32 : i64\n    %c64_i64 = arith.con'
    'stant 64 : i64\n    %c256_i64 = arith.constant 256 : i64\n    %c512_i64 = arith.constant 512 : i64\n    %c10'
    '24_i64 = arith.constant 1024 : i64\n    %c2048_i64 = arith.constant 2048 : i64\n    %c4096_i64 = arith.const'
    'ant 4096 : i64\n    %c8192_i64 = arith.constant 8192 : i64\n    %c16384_i64 = arith.constant 16384 : i64\n  '
    '  %c32768_i64 = arith.constant 32768 : i64\n    %c49152_i64 = arith.constant 49152 : i64\n    %c53248_i64 = '
    'arith.constant 53248 : i64\n    %c2139095040_i32 = arith.constant 2139095040 : i32\n    %c23_i32 = arith.con'
    'stant 23 : i32\n    %c8_i32 = arith.constant 8 : i32\n    %c119_i32 = arith.constant 119 : i32\n    %c254_i3'
    '2 = arith.constant 254 : i32\n\n    %ub_src = pto.castptr %c0_i64 : i64 -> !pto.ptr<f16, ub>\n    %ub_out_u8'
    ' = pto.castptr %c32768_i64 : i64 -> !pto.ptr<ui8, ub>\n    %ub_out_f8 = pto.castptr %c32768_i64 : i64 -> !pt'
    'o.ptr<f8E4M3FN, ub>\n    %ub_scale1 = pto.castptr %c49152_i64 : i64 -> !pto.ptr<ui8, ub>\n    %ub_scale2 = p'
    'to.castptr %c53248_i64 : i64 -> !pto.ptr<ui8, ub>\n\n    pto.mte_gm_ub %src_gm, %ub_src, %c0_i64, %c1024_i64'
    '\n      nburst(%c1_i64, %c1024_i64, %c1024_i64)\n      : !pto.ptr<f16, gm>, !pto.ptr<f16, ub>, i64, i64, i64'
    ', i64, i64\n\n    pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n    pto.wait_flag["PIPE_MTE2", "PIPE_V",'
    ' "EVENT_ID0"]\n\n    pto.vecscope {\n      scf.for %row = %c0 to %c2 step %c1 {\n        %elem_off = arith.m'
    'uli %row, %c256 : index\n        %mask = pto.vmi.create_mask %c256 : index -> !pto.vmi.mask<256xpred>\n     '
    '   %x_f16 = pto.vmi.vload %ub_src[%elem_off]\n            : !pto.ptr<f16, ub>, index -> !pto.vmi.vreg<256xf16>\n  '
    '      %x = pto.vmi.vcvt %x_f16\n            : !pto.vmi.vreg<256xf16> -> !pto.vmi.vreg<256xf32>\n        %abs '
    '= pto.vmi.vabs %x\n            : !pto.vmi.vreg<256xf32> -> !pto.vmi.vreg<256xf32>\n        %amax = pto.vmi.v'
    'cmax %abs, %mask {group = 8}\n            : !pto.vmi.vreg<256xf32>, !pto.vmi.mask<256xpred>\n            -> '
    '!pto.vmi.vreg<8xf32>\n        %amax_bits = pto.vmi.vinterpret_cast %amax\n            : !pto.vmi.vreg<8xf32>'
    ' -> !pto.vmi.vreg<8xi32>\n        %exp_mask = pto.vmi.vbrc %c2139095040_i32\n            : i32 -> !pto.vmi.v'
    'reg<8xi32>\n        %shift = pto.vmi.vbrc %c23_i32\n            : i32 -> !pto.vmi.vreg<8xi32>\n        %emax'
    ' = pto.vmi.vbrc %c8_i32\n            : i32 -> !pto.vmi.vreg<8xi32>\n        %scale_exp_bias = pto.vmi.vbrc %'
    'c254_i32\n            : i32 -> !pto.vmi.vreg<8xi32>\n        %exp_bits = pto.vmi.vand %amax_bits, %exp_mask'
    '\n            : !pto.vmi.vreg<8xi32>, !pto.vmi.vreg<8xi32>\n            -> !pto.vmi.vreg<8xi32>\n        %ex'
    'p = pto.vmi.vshr %exp_bits, %shift\n            : !pto.vmi.vreg<8xi32>, !pto.vmi.vreg<8xi32>\n            ->'
    ' !pto.vmi.vreg<8xi32>\n        %e8m0_i32 = pto.vmi.vsub %exp, %emax\n            : !pto.vmi.vreg<8xi32>, !pt'
    'o.vmi.vreg<8xi32>\n            -> !pto.vmi.vreg<8xi32>\n        %scale_ub_off = arith.muli %row, %c32 : inde'
    'x\n        pto.vmi.vstore %e8m0_i32, %ub_scale1[%scale_ub_off], %c1 {group = 8}\n            : !pto.vmi.vreg'
    '<8xi32>, !pto.ptr<ui8, ub>, index\n        %scale_exp = pto.vmi.vsub %scale_exp_bias, %e8m0_i32\n            : !pt'
    'o.vmi.vreg<8xi32>, !pto.vmi.vreg<8xi32>\n            -> !pto.vmi.vreg<8xi32>\n        %scale_bits = pto.vmi.v'
    'shl %scale_exp, %shift\n            : !pto.vmi.vreg<8xi32>, !pto.vmi.vreg<8xi32>\n            -> !pto.vmi.vr'
    'eg<8xi32>\n        %scale = pto.vmi.vinterpret_cast %scale_bits\n            : !pto.vmi.vreg<8xi32> -> !pto.'
    'vmi.vreg<8xf32>\n        %scale_vec = pto.vmi.vbrc %scale {group = 8}\n            : !pto.vmi.vreg<8xf32> ->'
    ' !pto.vmi.vreg<256xf32>\n        %scaled = pto.vmi.vmul %x, %scale_vec\n            : !pto.vmi.vreg<256xf32>'
    ', !pto.vmi.vreg<256xf32>\n            -> !pto.vmi.vreg<256xf32>\n        %q8 = pto.vmi.vcvt %scaled {saturat'
    'e = "SAT"}\n            : !pto.vmi.vreg<256xf32> -> !pto.vmi.vreg<256xf8E4M3FN>\n        pto.vmi.vstore %q8,'
    ' %ub_out_f8[%elem_off]\n            : !pto.vmi.vreg<256xf8E4M3FN>, !pto.ptr<f8E4M3FN, ub>, index\n      }\n\n     '
    ' %scale2_i32 = pto.vmi.vbrc %c119_i32\n          : i32 -> !pto.vmi.vreg<256xi32>\n      %scale2_u8 = pto.vmi.'
    'vcvt %scale2_i32 {saturate = "SAT"}\n          : !pto.vmi.vreg<256xi32> -> !pto.vmi.vreg<256xui8>\n      pto'
    '.vmi.vstore %scale2_u8, %ub_scale2[%c0]\n          : !pto.vmi.vreg<256xui8>, !pto.ptr<ui8, ub>, index\n      pto.v'
    'mi.vstore %scale2_u8, %ub_scale2[%c256]\n          : !pto.vmi.vreg<256xui8>, !pto.ptr<ui8, ub>, index\n    }\n\n  '
    '  pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n    pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n   '
    ' pto.mte_ub_gm %ub_out_u8, %out_gm, %c512_i64\n      nburst(%c1_i64, %c512_i64, %c512_i64)\n      : !pto.ptr'
    '<ui8, ub>, !pto.ptr<ui8, gm>, i64, i64, i64, i64\n    pto.mte_ub_gm %ub_scale1, %scale1_gm, %c8_i64\n      n'
    'burst(%c2_i64, %c32_i64, %c8_i64)\n      : !pto.ptr<ui8, ub>, !pto.ptr<ui8, gm>, i64, i64, i64, i64\n    pto'
    '.mte_ub_gm %ub_scale2, %scale2_gm, %c512_i64\n      nburst(%c1_i64, %c512_i64, %c512_i64)\n      : !pto.ptr<'
    'ui8, ub>, !pto.ptr<ui8, gm>, i64, i64, i64, i64\n    pto.barrier #pto.pipe<PIPE_ALL>\n    return\n  }\n}\n'
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

    ROWS = 2
    COLS = 256
    SCALE1_BYTES = ROWS * (COLS // 32)
    SCALE2_BYTES = 512
    SENTINEL_U8 = np.uint8(0xA5)

    Q_VALUES = np.array(
        [0.0, 1.0, -1.0, 0.5, 2.0, -2.0, 4.0, -4.0, 448.0], dtype=np.float32
    )
    F8E4M3FN_BYTES = np.array(
        [0x00, 0x38, 0xB8, 0x30, 0x40, 0xC0, 0x48, 0xC8, 0x7E], dtype=np.uint8
    )


    def _generate(seed: int) -> dict:
        repeats = (COLS + len(Q_VALUES) - 1) // len(Q_VALUES)
        q_row = np.tile(Q_VALUES, repeats)[:COLS].astype(np.float32)
        f8_row = np.tile(F8E4M3FN_BYTES, repeats)[:COLS].astype(np.uint8)

        src = np.tile((q_row / np.float32(256.0)).astype(np.float16), (ROWS, 1))
        golden_out = np.tile(f8_row, (ROWS, 1)).astype(np.uint8)
        golden_scale1 = np.full(SCALE1_BYTES, np.uint8(0x77), dtype=np.uint8)
        golden_scale2 = np.full(SCALE2_BYTES, np.uint8(0x77), dtype=np.uint8)

        out = np.full((ROWS, COLS), SENTINEL_U8, dtype=np.uint8)
        scale1 = np.full(SCALE1_BYTES, SENTINEL_U8, dtype=np.uint8)
        scale2 = np.full(SCALE2_BYTES, SENTINEL_U8, dtype=np.uint8)

        _files["v1.bin"] = src.view(np.uint16)
        _files["v2.bin"] = out
        _files["v3.bin"] = scale1
        _files["v4.bin"] = scale2
        _files["golden_v2.bin"] = golden_out
        _files["golden_v3.bin"] = golden_scale1
        _files["golden_v4.bin"] = golden_scale2


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


    def check_u8(name: str) -> bool:
        golden = np.fromfile(f"golden_{name}.bin", dtype=np.uint8)
        output = np.fromfile(f"{name}.bin", dtype=np.uint8)
        if golden.shape == output.shape and np.array_equal(golden, output):
            return True
        diff = np.nonzero(golden != output)[0]
        idx = int(diff[0]) if diff.size else -1
        print(
            f"[ERROR] compare failed {name} idx={idx} "
            f"golden=0x{int(golden[idx]):02x} output=0x{int(output[idx]):02x}"
        )
        return False


    def main() -> None:
        if not check_u8("v2") or not check_u8("v3") or not check_u8("v4"):
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
        'f16',
        'src_gm'
    ),
    (
        'v2',
        'ui8',
        'out_gm'
    ),
    (
        'v3',
        'ui8',
        'scale1_gm'
    ),
    (
        'v4',
        'ui8',
        'scale2_gm'
    )
]
SCALARS = []
SCALAR_SOURCES = []
SCALAR_DTYPES = []


@pto.jit(
    name='vmi_block_mx_quant_f16_e4m3_2x256_kernel',
    target="a5",
    backend="vpto",
    mode="explicit",
    source=KERNEL_SOURCE,
)
def kernel(
    src_gm: pto.ptr(pto.f16, "gm"),
    out_gm: pto.ptr(pto.ui8, "gm"),
    scale1_gm: pto.ptr(pto.ui8, "gm"),
    scale2_gm: pto.ptr(pto.ui8, "gm"),
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
    name='legacy_vmi_new_kernels_block_mx_quant_f16_e4m3_2x256',
    grid=1,
    kernel=kernel,
    make_case=make_case,
    check=check_case,
)]

auto_main(globals())
