#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Migrated from test/vpto/cases/micro-op/dsa-sfu/vmrgsort4.

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
    'module attributes {pto.target_arch = "a5", pto.kernel_kind = #pto.kernel_kind<vector>} {\n  func.func @vmrgs'
    'ort4_kernel_f32(%arg0: !pto.ptr<f32, gm>,\n                                  %arg1: !pto.ptr<f32, gm>,\n    '
    '                              %arg2: !pto.ptr<i16, gm>) attributes {pto.kernel} {\n    %c0_i64 = arith.const'
    'ant 0 : i64\n    %c1_i64 = arith.constant 1 : i64\n    %c0 = arith.constant 0 : index\n    %c1 = arith.const'
    'ant 1 : index\n    %c2 = arith.constant 2 : index\n    %c3 = arith.constant 3 : index\n    %c4 = arith.const'
    'ant 4 : index\n    %c6 = arith.constant 6 : index\n    %c32_i64 = arith.constant 32 : i64\n    %c_count = ar'
    'ith.constant 281479271743489 : i64\n    %c_config = arith.constant 7937 : i64\n\n    %ub_in = pto.castptr %c'
    '0_i64 : i64 -> !pto.ptr<f32, ub>\n    %ub_out = pto.castptr %c32_i64 : i64 -> !pto.ptr<f32, ub>\n    %src1 ='
    ' pto.addptr %ub_in, %c2 : !pto.ptr<f32, ub>, index -> !pto.ptr<f32, ub>\n    %src2 = pto.addptr %ub_in, %c4 : !pto.'
    'ptr<f32, ub>, index -> !pto.ptr<f32, ub>\n    %src3 = pto.addptr %ub_in, %c6 : !pto.ptr<f32, ub>, index -> !pto.ptr<f32, u'
    'b>\n\n    pto.mte_gm_ub %arg0, %ub_in, %c0_i64, %c32_i64\n      nburst(%c1_i64, %c32_i64, %c32_i64)\n      :'
    ' !pto.ptr<f32, gm>, !pto.ptr<f32, ub>, i64, i64, i64, i64, i64\n\n    pto.set_flag["PIPE_MTE2", "PIPE_V", "E'
    'VENT_ID0"]\n    pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]\n\n    pto.vmrgsort4 %ub_out, %ub_in, %src'
    '1, %src2, %src3, %c_count, %c_config\n      : !pto.ptr<f32, ub>, !pto.ptr<f32, ub>, !pto.ptr<f32, ub>, !pto.'
    'ptr<f32, ub>,\n        !pto.ptr<f32, ub>, i64, i64\n\n    pto.set_flag["PIPE_V", "PIPE_S", "EVENT_ID0"]\n   '
    ' pto.wait_flag["PIPE_V", "PIPE_S", "EVENT_ID0"]\n    %list0, %list1, %list2, %list3 = pto.get_vms4_sr : i16,'
    ' i16, i16, i16\n    pto.store %list0, %arg2[%c0] : !pto.ptr<i16, gm>, index, i16\n    pto.store %list1, %arg2[%c1] '
    ': !pto.ptr<i16, gm>, index, i16\n    pto.store %list2, %arg2[%c2] : !pto.ptr<i16, gm>, index, i16\n    pto.store %list3, %'
    'arg2[%c3] : !pto.ptr<i16, gm>, index, i16\n\n    pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n    pto.wait_fla'
    'g["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]\n\n    pto.mte_ub_gm %ub_out, %arg1, %c32_i64\n      nburst(%c1_i64, %'
    'c32_i64, %c32_i64)\n      : !pto.ptr<f32, ub>, !pto.ptr<f32, gm>, i64, i64, i64, i64\n    pto.barrier #pto.p'
    'ipe<PIPE_ALL>\n    return\n  }\n}\n'
)

def _golden_scope():
    # Copyright (c) 2026 Huawei Technologies Co., Ltd.
    # This program is free software, you can redistribute it and/or modify it under the terms and conditions of
    # CANN Open Software License Agreement Version 2.0 (the "License").
    # Please refer to the License for details. You may not use this file except in compliance with the License.
    # THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
    # INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
    # See LICENSE in the root of the software repository for the full text of the License.

    import argparse
    import struct
    from pathlib import Path


    PAIR_FMT = "fI"


    def write_pairs(path: Path, pairs) -> None:
        with path.open("wb") as f:
            for score, index in pairs:
                f.write(struct.pack(PAIR_FMT, score, index))


    def main() -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--output-dir", type=Path, default=Path("."))
        args = parser.parse_args()

        out = args.output_dir
        out.mkdir(parents=True, exist_ok=True)

        src = [(9.0, 90), (7.0, 70), (8.0, 80), (6.0, 60)]
        # Exhausted mode stops after emitting the only valid proposal. The
        # remaining output slots are not architecturally produced and are ignored by
        # compare.py.
        golden = [(9.0, 90), (0.0, 0), (0.0, 0), (0.0, 0)]

        write_pairs(out / "v1.bin", src)
        write_pairs(out / "v2.bin", [(0.0, 0)] * 4)
        write_pairs(out / "golden_v2.bin", golden)
        (out / "v3.bin").write_bytes(struct.pack("4h", 0, 0, 0, 0))
        (out / "golden_v3.bin").write_bytes(struct.pack("4h", 1, 0, 0, 0))


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

    import os
    import struct
    import sys

    import numpy as np


    PAIR_FMT = "fI"
    PAIR_SIZE = struct.calcsize(PAIR_FMT)
    PAIR_COUNT = 4


    def read_pairs(path: str):
        values = []
        indices = []
        with open(path, "rb") as f:
            for _ in range(PAIR_COUNT):
                data = f.read(PAIR_SIZE)
                if len(data) != PAIR_SIZE:
                    break
                value, index = struct.unpack(PAIR_FMT, data)
                values.append(value)
                indices.append(index)
        return np.array(values, dtype=np.float32), np.array(indices, dtype=np.uint32)


    def read_counts(path: str):
        with open(path, "rb") as f:
            data = f.read(8)
        return np.array(struct.unpack("4h", data), dtype=np.int16)


    def main() -> None:
        strict = os.getenv("COMPARE_STRICT", "1") != "0"
        golden_values, golden_indices = read_pairs("golden_v2.bin")
        output_values, output_indices = read_pairs("v2.bin")
        golden_counts = read_counts("golden_v3.bin")
        output_counts = read_counts("v3.bin")
        produced = int(golden_counts.sum())
        ok = (
            golden_values.shape == output_values.shape
            and golden_indices.shape == output_indices.shape
            and 0 <= produced <= PAIR_COUNT
            and np.allclose(golden_values[:produced], output_values[:produced])
            and np.array_equal(golden_indices[:produced], output_indices[:produced])
            and np.array_equal(golden_counts, output_counts)
        )
        if not ok:
            if strict:
                print("[ERROR] compare failed")
                sys.exit(2)
            print("[WARN] compare failed (non-gating)")
            return
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
        'arg0'
    ),
    (
        'v2',
        'f32',
        'arg1'
    ),
    (
        'v3',
        'i16',
        'arg2'
    )
]
SCALARS = []


@pto.jit(
    name='vmrgsort4_kernel_f32',
    target="a5",
    backend="vpto",
    mode="explicit",
    source=KERNEL_SOURCE,
)
def kernel(
    arg0: pto.ptr(pto.f32, "gm"),
    arg1: pto.ptr(pto.f32, "gm"),
    arg2: pto.ptr(pto.i16, "gm"),
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
    name='legacy_micro_op_dsa_sfu_vmrgsort4',
    grid=1,
    kernel=kernel,
    make_case=make_case,
    check=check_case,
)]

auto_main(globals())
