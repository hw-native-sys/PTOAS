#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Migrated from test/vpto/cases/vmi_new/vmula-bf16-vl8-unaligned-stateful-load."""

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
    'the "License").\n// Please refer to the License for details. You should not use this file except in complian'
    'ce with the License.\n// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITH'
    'ER EXPRESS OR IMPLIED,\n// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A '
    'PARTICULAR PURPOSE.\n// See LICENSE in the root of the software repository for the full text of the License.'
    '\n\n// -----------------------------------------------------------------------------\n// case: vmi_new/vmula'
    '-bf16-vl8-unaligned-stateful-load\n// family: memory load lowering (PtrType stateful fallback)\n// target_op'
    's: pto.vmi.vload (PtrType source, unaligned effective address)\n// regression for: #1374 (commits 343742bf4 '
    '+ 700a8fd82)\n//\n// Before 343742bf4 a PtrType load whose effective address could not be proven\n// 32-byte'
    ' aligned was rejected by checkSupportedContiguousLoadAddress with\n// VMI-UNSUPPORTED ("requires statically '
    'shaped memref source"), because\n// computeSafeStatefulReadProof only handles MemRefType.  The vldas+vldus\n'
    '// stateful stream (5eb87c2) could have lowered it, but the preflight check\n// ran first and never let it t'
    'hrough.  This reproduced as 54 vmi_vmula sim\n// failures for small VL (vl=2/4/8) across bf16/f16/f32/i16/i3'
    '2/i8.\n//\n// This kernel reproduces that failing shape -- vl=8 bf16 (16 bytes, half a\n// VCG block), the l'
    'oads sourced from non-32B-aligned UB addresses -- with\n// three distinct misalignment residue classes:\n//'
    '\n//   - acc: base 1024 (aligned) + element offset 4 → eff. addr 1032, 8 mod 32\n//   - lhs: base 2048 (alig'
    'ned) + element offset 1 → eff. addr 2050, 2 mod 32\n//   - rhs: base 3072 (aligned) + element offset 6 → eff'
    '. addr 3084, 12 mod 32\n//\n// The GM→UB staging destinations stay 32B-aligned (A5 MTE requires aligned\n// '
    'copy destinations); the misalignment is introduced only on the load side,\n// which is exactly the situation'
    ' the preflight change covers.\n//\n// Expected behaviour (verified in the VPTO IR and on the simulator):\n//'
    '   - all three loads pass the preflight and lower to pto.vldas + pto.vldus\n//     (no pto.vsldb anywhere),'
    '\n//   - the vmula result is bit-identical to the aligned case\n//     vmi_new/vmula-bf16-vl8 (same input sa'
    'mples, wide-domain accumulate,\n//     one RNE round to bf16).\n// -----------------------------------------'
    '------------------------------------\n\nmodule attributes {pto.target_arch = "a5", pto.kernel_kind = #pto.ke'
    'rnel_kind<vector>} {\n  func.func @vmi_vmula_bf16_vl8_unaligned_kernel(%acc_gm: !pto.ptr<bf16, gm>,\n       '
    '                                          %lhs_gm: !pto.ptr<bf16, gm>,\n                                    '
    '             %rhs_gm: !pto.ptr<bf16, gm>,\n                                                 %dst_gm: !pto.pt'
    'r<bf16, gm>) attributes {pto.kernel} {\n    %c0 = arith.constant 0 : index\n    %c8 = arith.constant 8 : ind'
    'ex\n    %c0_i64 = arith.constant 0 : i64\n    %c1_i64 = arith.constant 1 : i64\n    %c16_i64 = arith.constan'
    't 16 : i64\n    %c32_i64 = arith.constant 32 : i64\n    %c1024_i64 = arith.constant 1024 : i64\n    %c2048_i'
    '64 = arith.constant 2048 : i64\n    %c3072_i64 = arith.constant 3072 : i64\n    %c4096_i64 = arith.constant '
    '4096 : i64\n    %c1 = arith.constant 1 : index\n    %c4 = arith.constant 4 : index\n    %c6 = arith.constant'
    ' 6 : index\n\n    // 32B-aligned, non-overlapping UB staging slots; each holds the 8-lane\n    // payload at'
    ' a different element offset (see the loads below).\n    %ub_acc = pto.castptr %c1024_i64 : i64 -> !pto.ptr<b'
    'f16, ub>\n    %ub_lhs = pto.castptr %c2048_i64 : i64 -> !pto.ptr<bf16, ub>\n    %ub_rhs = pto.castptr %c3072'
    '_i64 : i64 -> !pto.ptr<bf16, ub>\n    %ub_dst = pto.castptr %c4096_i64 : i64 -> !pto.ptr<bf16, ub>\n\n    pt'
    'o.mte_gm_ub %acc_gm, %ub_acc, %c0_i64, %c32_i64\n      nburst(%c1_i64, %c32_i64, %c32_i64)\n      : !pto.ptr'
    '<bf16, gm>, !pto.ptr<bf16, ub>, i64, i64, i64, i64, i64\n    pto.mte_gm_ub %lhs_gm, %ub_lhs, %c0_i64, %c32_i'
    '64\n      nburst(%c1_i64, %c32_i64, %c32_i64)\n      : !pto.ptr<bf16, gm>, !pto.ptr<bf16, ub>, i64, i64, i64'
    ', i64, i64\n    pto.mte_gm_ub %rhs_gm, %ub_rhs, %c0_i64, %c32_i64\n      nburst(%c1_i64, %c32_i64, %c32_i64)'
    '\n      : !pto.ptr<bf16, gm>, !pto.ptr<bf16, ub>, i64, i64, i64, i64, i64\n\n    pto.set_flag["PIPE_MTE2", "'
    'PIPE_V", "EVENT_ID0"]\n    pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n\n    pto.vecscope {\n      //'
    ' Three PtrType loads with unprovable 32B alignment (residues 8/2/12\n      // mod 32).  All three must take '
    'the vldas+vldus stateful stream.\n      %acc = pto.vmi.vload %ub_acc[%c4] : !pto.ptr<bf16, ub>, index -> !pto.vmi.'
    'vreg<8xbf16>\n      %lhs = pto.vmi.vload %ub_lhs[%c1] : !pto.ptr<bf16, ub>, index -> !pto.vmi.vreg<8xbf16>\n      '
    '%rhs = pto.vmi.vload %ub_rhs[%c6] : !pto.ptr<bf16, ub>, index -> !pto.vmi.vreg<8xbf16>\n      %mask = pto.vmi.crea'
    'te_mask %c8 : index -> !pto.vmi.mask<8xpred>\n      %result = pto.vmi.vmula %acc, %lhs, %rhs, %mask\n          '
    ': !pto.vmi.vreg<8xbf16>, !pto.vmi.vreg<8xbf16>,\n            !pto.vmi.vreg<8xbf16>, !pto.vmi.mask<8xpred>\n '
    '         -> !pto.vmi.vreg<8xbf16>\n      // The store side stays on the ordinary aligned path.\n      pto.vm'
    'i.vstore %result, %ub_dst[%c0]\n          : !pto.vmi.vreg<8xbf16>, !pto.ptr<bf16, ub>, index\n    }\n\n    pto.set'
    '_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n    pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n    pto.mte'
    '_ub_gm %ub_dst, %dst_gm, %c16_i64\n      nburst(%c1_i64, %c16_i64, %c16_i64)\n      : !pto.ptr<bf16, ub>, !p'
    'to.ptr<bf16, gm>, i64, i64, i64, i64\n    pto.barrier #pto.pipe<PIPE_ALL>\n    return\n  }\n}\n'
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

    # Golden for vmi_new/vmula-bf16-vl8-unaligned-stateful-load.
    #
    # Same input samples as vmi_new/vmula-bf16-vl8 (the framework case
    # vmi_vmula_bf16_vl8_g1_a8_n_nomask_t01): the unaligned-stateful case exists
    # to prove the vldas+vldus lowering (#1374) reads the same bits the aligned
    # path would, so the numerical golden is identical.
    #
    # Each GM operand is a 32-byte staging image of its UB slot: the 16-byte
    # payload is written at the element offset the kernel later loads from
    # (acc at lane 4, lhs at lane 1, rhs at lane 6), so the misaligned effective
    # addresses (1032/2050/3084) land exactly on the payload.
    #
    # Semantics: vmula computes acc + lhs*rhs in a wide domain and rounds once to
    # bf16 (RNE) at the output -- it does NOT round the product to bf16 first.

    import argparse
    from pathlib import Path

    import numpy as np

    VL = 8
    SLOT_LANES = 16  # 32-byte staging image, in bf16 lanes

    ACC_BITS = np.array(
        [0xC087, 0x3F4F, 0x3F3D, 0xC00D, 0x3FE5, 0xC027, 0xC047, 0x401E],
        dtype=np.uint16,
    )
    LHS_BITS = np.array(
        [0xC024, 0xBFF9, 0x4092, 0xBF79, 0xC080, 0x3FDE, 0xBF95, 0xC006],
        dtype=np.uint16,
    )
    RHS_BITS = np.array(
        [0x4095, 0xC022, 0xC08A, 0xC07E, 0x409B, 0xBE68, 0xC057, 0xBFCB],
        dtype=np.uint16,
    )

    assert ACC_BITS.shape == (VL,) and LHS_BITS.shape == (VL,) and RHS_BITS.shape == (VL,), \
        f"input arrays must have {VL} lanes"


    def bf16_bits_to_f32(bits):
        u = bits.astype(np.uint32) << np.uint32(16)
        return u.view(np.float32)


    def to_bf16_bits(x):
        """f32 -> bfloat16 bit pattern (uint16), round-to-nearest-even."""
        x = np.asarray(x, dtype=np.float32)
        u = x.view(np.uint32)
        lsb = (u >> np.uint32(16)) & np.uint32(1)
        rounding_bias = np.uint32(0x7FFF) + lsb
        r = ((u + rounding_bias) >> np.uint32(16)).astype(np.uint16)
        r = np.where(np.isnan(x), np.uint16(0x7FC0), r)
        return r


    def staging_image(payload_bits, lane_offset):
        """32-byte UB slot image with the payload at the given lane offset."""
        slot = np.zeros(SLOT_LANES, dtype=np.uint16)
        slot[lane_offset:lane_offset + VL] = payload_bits
        return slot


    def _generate(seed: int) -> dict:
        acc = bf16_bits_to_f32(ACC_BITS)
        lhs = bf16_bits_to_f32(LHS_BITS)
        rhs = bf16_bits_to_f32(RHS_BITS)
        # Wide-domain accumulate: compute in f64, round once to bf16 at output.
        wide = acc.astype(np.float64) + lhs.astype(np.float64) * rhs.astype(np.float64)
        golden = to_bf16_bits(wide.astype(np.float32))

        # Sentinel so a completely-missed store shows up as garbage, not coincidence.
        dst = np.full(VL, 0xAAAA, dtype=np.uint16)

        _files["v1.bin"] = staging_image(ACC_BITS, 4)
        _files["v2.bin"] = staging_image(LHS_BITS, 1)
        _files["v3.bin"] = staging_image(RHS_BITS, 6)
        _files["v4.bin"] = dst
        _files["golden_v4.bin"] = golden


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

    # vmula bf16 output is bit-compared as uint16 bf16 bit patterns: the DUT
    # rounds once to bf16 (RNE), so the simulator/NPU result must match the
    # golden bits exactly.  A mismatch here means the vldas+vldus stateful load
    # (#1374) read wrong lanes from the misaligned address.

    import sys

    import numpy as np


    def main() -> None:
        golden = np.fromfile("golden_v4.bin", dtype=np.uint16)
        output = np.fromfile("v4.bin", dtype=np.uint16)
        if golden.shape != output.shape or not np.array_equal(golden, output):
            common_size = min(golden.size, output.size)
            diff = np.nonzero(golden[:common_size] != output[:common_size])[0]
            idx = int(diff[0]) if diff.size else common_size
            golden_value = int(golden[idx]) if idx < golden.size else "n/a"
            output_value = int(output[idx]) if idx < output.size else "n/a"
            print(f"[ERROR] compare failed idx={idx} golden={golden_value} output={output_value}")
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
        'acc_gm'
    ),
    (
        'v2',
        'bf16',
        'lhs_gm'
    ),
    (
        'v3',
        'bf16',
        'rhs_gm'
    ),
    (
        'v4',
        'bf16',
        'dst_gm'
    )
]
SCALARS = []
SCALAR_SOURCES = []
SCALAR_DTYPES = []


@pto.jit(
    name='vmi_vmula_bf16_vl8_unaligned_kernel',
    target="a5",
    backend="vpto",
    mode="explicit",
    source=KERNEL_SOURCE,
)
def kernel(
    acc_gm: pto.ptr(pto.bf16, "gm"),
    lhs_gm: pto.ptr(pto.bf16, "gm"),
    rhs_gm: pto.ptr(pto.bf16, "gm"),
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
    name='legacy_vmi_new_vmula_bf16_vl8_unaligned_stateful_load',
    grid=1,
    kernel=kernel,
    make_case=make_case,
    check=check_case,
)]

auto_main(globals())
