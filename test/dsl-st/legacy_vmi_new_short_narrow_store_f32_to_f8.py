#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Migrated from test/vpto/cases/vmi_new/short-narrow-store-f32-to-f8.

Fallback style: the case's own golden.py/compare.py run in a work directory,
exactly as the legacy runner did.
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
    '\n// Short dense f32 reads narrowed into short packed f8e4m3 stores at four sizes\n// and offsets.  The f8 r'
    'esult is packed four to a 32-bit carrier lane\n// (lane_stride = 4) while the short store carries a single-c'
    'arrier group\n// packet, so a value only lands in the right bytes when the radix-4 narrowing\n// and the car'
    'rier layout agree.  Each shape writes a disjoint region; the rest\n// of the destination buffer keeps its se'
    'ntinel, so a wrong byte, a wrong region\n// or an overrun shows up as a mismatch.\n\nmodule attributes {pto.'
    'target_arch = "a5", pto.kernel_kind = #pto.kernel_kind<vector>} {\n  func.func @vmi_short_narrow_store_f8_ke'
    'rnel(%src_gm: !pto.ptr<f32, gm>,\n                                              %dst_gm: !pto.ptr<ui8, gm>) '
    'attributes {pto.kernel} {\n    %c0 = arith.constant 0 : index\n    %c0_i64 = arith.constant 0 : i64\n    %c1'
    '_i64 = arith.constant 1 : i64\n    %c256_i64 = arith.constant 256 : i64\n    %c1024_i64 = arith.constant 102'
    '4 : i64\n    %c2048_i64 = arith.constant 2048 : i64\n    %s0 = arith.constant 0 : index\n    %d0 = arith.con'
    'stant 0 : index\n    %s32 = arith.constant 32 : index\n    %d32 = arith.constant 32 : index\n    %s64 = arit'
    'h.constant 64 : index\n    %d64 = arith.constant 64 : index\n    %s96 = arith.constant 96 : index\n    %d96 '
    '= arith.constant 96 : index\n\n    %ub_src = pto.castptr %c0_i64 : i64 -> !pto.ptr<f32, ub>\n    %ub_dst_u8 '
    '= pto.castptr %c2048_i64 : i64 -> !pto.ptr<ui8, ub>\n    %ub_dst_f8 = pto.castptr %c2048_i64 : i64 -> !pto.p'
    'tr<f8E4M3FN, ub>\n\n    pto.mte_gm_ub %src_gm, %ub_src, %c0_i64, %c1024_i64\n      nburst(%c1_i64, %c1024_i6'
    '4, %c1024_i64)\n      : !pto.ptr<f32, gm>, !pto.ptr<f32, ub>, i64, i64, i64, i64, i64\n    pto.mte_gm_ub %ds'
    't_gm, %ub_dst_u8, %c0_i64, %c256_i64\n      nburst(%c1_i64, %c256_i64, %c256_i64)\n      : !pto.ptr<ui8, gm>'
    ', !pto.ptr<ui8, ub>, i64, i64, i64, i64, i64\n\n    pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n    pt'
    'o.wait_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n\n    pto.vecscope {\n      %v1 = pto.vmi.vload %ub_src[%s0'
    '] : !pto.ptr<f32, ub>, index -> !pto.vmi.vreg<1xf32>\n      %n1 = pto.vmi.vcvt %v1 {saturate = "SAT"} : !pto.vmi.v'
    'reg<1xf32> -> !pto.vmi.vreg<1xf8E4M3FN>\n      pto.vmi.vstore %n1, %ub_dst_f8[%d0] : !pto.vmi.vreg<1xf8E4M3FN'
    '>, !pto.ptr<f8E4M3FN, ub>, index\n      %v2 = pto.vmi.vload %ub_src[%s32] : !pto.ptr<f32, ub>, index -> !pto.vmi.v'
    'reg<2xf32>\n      %n2 = pto.vmi.vcvt %v2 {saturate = "SAT"} : !pto.vmi.vreg<2xf32> -> !pto.vmi.vreg<2xf8E4M3FN>\n '
    '     pto.vmi.vstore %n2, %ub_dst_f8[%d32] : !pto.vmi.vreg<2xf8E4M3FN>, !pto.ptr<f8E4M3FN, ub>, index\n      %v4 = '
    'pto.vmi.vload %ub_src[%s64] : !pto.ptr<f32, ub>, index -> !pto.vmi.vreg<4xf32>\n      %n4 = pto.vmi.vcvt %v4 {satu'
    'rate = "SAT"} : !pto.vmi.vreg<4xf32> -> !pto.vmi.vreg<4xf8E4M3FN>\n      pto.vmi.vstore %n4, %ub_dst_f8[%d64] : '
    '!pto.vmi.vreg<4xf8E4M3FN>, !pto.ptr<f8E4M3FN, ub>, index\n      %v8 = pto.vmi.vload %ub_src[%s96] : !pto.ptr<f32, '
    'ub>, index -> !pto.vmi.vreg<8xf32>\n      %n8 = pto.vmi.vcvt %v8 {saturate = "SAT"} : !pto.vmi.vreg<8xf32> -> !pto'
    '.vmi.vreg<8xf8E4M3FN>\n      pto.vmi.vstore %n8, %ub_dst_f8[%d96] : !pto.vmi.vreg<8xf8E4M3FN>, !pto.ptr<f8E4M3'
    'FN, ub>, index\n    }\n\n    pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n    pto.wait_flag["PIPE_V", "PIPE_M'
    'TE3", "EVENT_ID0"]\n    pto.mte_ub_gm %ub_dst_u8, %dst_gm, %c256_i64\n      nburst(%c1_i64, %c256_i64, %c256_'
    'i64)\n      : !pto.ptr<ui8, ub>, !pto.ptr<ui8, gm>, i64, i64, i64, i64\n    pto.barrier #pto.pipe<PIPE_ALL>'
    '\n    return\n  }\n}\n'
)

def _golden_scope():
    # Copyright (c) 2026 Huawei Technologies Co., Ltd.
    # This program is free software, you can redistribute it and/or modify it under the terms and conditions of
    # CANN Open Software License Agreement Version 2.0 (the "License").
    # Please refer to the License for details. You may not use this file except in compliance with the License.
    # THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
    # INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
    # See LICENSE in the root of the software repository for the full text of the License.

    """Golden for the short f32 -> packed f8e4m3 narrow stores with sentinel gaps."""

    import argparse
    from pathlib import Path

    import numpy as np

    ELEMS = 256
    SEED = 23
    SENTINEL = np.uint8(0xA5)

    # Values that are exactly representable in f8e4m3fn, with their bit patterns, so
    # the narrowing is exact and the comparison can be exact.
    F8_VALUES = (
        (0.0, 0x00),
        (0.5, 0x30),
        (1.0, 0x38),
        (2.0, 0x40),
        (4.0, 0x48),
        (-1.0, 0xB8),
        (-2.0, 0xC0),
        (-4.0, 0xC8),
    )
    # (n, source element offset, destination element offset); every offset is 32B
    # aligned for its element type (see the kernel's bound reads and stores).
    SHAPES = ((1, 0, 0), (2, 32, 32), (4, 64, 64), (8, 96, 96))


    def main() -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--output-dir", type=Path, default=Path("."))
        parser.add_argument("--seed", type=int, default=SEED)
        args = parser.parse_args()
        del args.seed  # the pattern is deterministic; the seed keeps the CLI uniform

        values = np.array([value for value, _ in F8_VALUES], dtype=np.float32)
        patterns = np.array([pattern for _, pattern in F8_VALUES], dtype=np.uint8)
        repeats = (ELEMS + values.size - 1) // values.size
        src = np.tile(values, repeats)[:ELEMS]
        dst = np.full(ELEMS, SENTINEL, dtype=np.uint8)
        golden = dst.copy()
        for n, src_offset, dst_offset in SHAPES:
            for k in range(n):
                value = np.float32(src[src_offset + k])
                matches = np.nonzero(values == value)[0]
                if matches.size != 1:
                    raise ValueError(f"value {value} is not an exact f8e4m3fn pattern")
                golden[dst_offset + k] = patterns[int(matches[0])]

        args.output_dir.mkdir(parents=True, exist_ok=True)
        src.tofile(args.output_dir / "v1.bin")
        dst.tofile(args.output_dir / "v2.bin")
        golden.tofile(args.output_dir / "golden_v2.bin")


    if __name__ == "__main__":
        main()
    return locals()

def _compare_scope():
    # Copyright (c) 2026 Huawei Technologies Co., Ltd.
    # This program is free software, you can redistribute it and/or modify it under the terms and conditions of
    # CANN Open Software License Agreement Version 2.0 (the "License").
    # Please refer to the License for details. You may not use this file except in compliance with the License.
    # THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
    # INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
    # See LICENSE in the root of the software repository for the full text of the License.

    """Exact comparison of the narrowed short f8 store against its golden."""

    import sys

    import numpy as np


    def main() -> None:
        golden = np.fromfile("golden_v2.bin", dtype=np.uint8)
        output = np.fromfile("v2.bin", dtype=np.uint8)
        if golden.shape != output.shape or not np.array_equal(golden, output):
            diff = np.nonzero(golden != output)[0]
            idx = int(diff[0]) if diff.size else -1
            print(f"[ERROR] compare failed idx={idx} golden=0x{int(golden[idx]):02x} output=0x{int(output[idx]):02x}")
            sys.exit(2)
        print("[INFO] compare passed")


    return locals()

HELPERS = {}

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
        'src_gm'
    ),
    (
        'v2',
        'ui8',
        'dst_gm'
    )
]
SCALARS = []


@pto.jit(
    name='vmi_short_narrow_store_f8_kernel',
    target="a5",
    backend="vpto",
    mode="explicit",
    source=KERNEL_SOURCE,
)
def kernel(
    src_gm: pto.ptr(pto.f32, "gm"),
    dst_gm: pto.ptr(pto.ui8, "gm"),
):
    pass


def _run_legacy_script(scope, cwd):
    """Run one inlined legacy script in-process under the legacy context."""
    for name, text in HELPERS.items():
        (Path(cwd) / name).write_text(text)
    real_argv, real_path = list(sys.argv), list(sys.path)
    real_environ, real_cwd = dict(os.environ), os.getcwd()
    sys.argv = ["_legacy_script.py"]
    sys.path.insert(0, str(cwd))
    os.environ["COMPARE_STRICT"] = "1"
    os.chdir(str(cwd))
    try:
        namespace = scope()
        if "_generate" in namespace:
            namespace["_generate"](0)
        elif "main" in namespace:
            namespace["main"]()
    except SystemExit as exc:
        if exc.code not in (0, None):
            raise RuntimeError("legacy script exited with %r" % (exc.code,))
    finally:
        os.chdir(real_cwd)
        sys.argv = real_argv
        sys.path[:] = real_path
        os.environ.clear()
        os.environ.update(real_environ)


def make_case():
    work_dir = tempfile.TemporaryDirectory(
        prefix="legacy-case-", dir=os.environ.get("TMPDIR"))
    work = Path(work_dir.name)
    _run_legacy_script(_golden_scope, work)
    arrays = [np.fromfile(work / (name + ".bin"), dtype=np.dtype(NUMPY_DTYPE[ty]))
              for name, ty, _slot in BUFFERS]
    return [*arrays, *[np.asarray(v) for v in SCALARS]], work_dir


def check_case(device_inputs, work_dir):
    work = Path(work_dir.name)
    try:
        for (name, ty, _slot), tensor in zip(BUFFERS, device_inputs):
            tensor.cpu().numpy().astype(np.dtype(NUMPY_DTYPE[ty])).tofile(
                Path(work) / (name + ".bin"))
        _run_legacy_script(_compare_scope, work)
    finally:
        work_dir.cleanup()


GRID_FROM_VALUES = None


CASES = [dict(
    name='legacy_vmi_new_short_narrow_store_f32_to_f8',
    grid=1,
    kernel=kernel,
    make_case=make_case,
    check=check_case,
)]

auto_main(globals())
