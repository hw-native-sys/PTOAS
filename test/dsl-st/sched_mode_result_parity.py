#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""A5 i8 row expand multiplication under the default and the batch scheduling mode.

`pto.sched_mode` only selects the CCE kernel scheduling mode; it must never change
the numbers a kernel produces. The two kernels below run the same computation as
`expandmul_i8.py`'s `rowexpandmul_i8_8x8` -- one leaves the attribute off (the
default, normal mode) and one declares `sched_mode=1` (batch mode) -- against one
golden.

Numbers alone would still match if the attribute were dropped on the way to the
device, so each case also asserts the device image its launch uses: the batch
kernel must carry the `.ScheduleMode_<kernel>` section and the default kernel
must not, which is the artifact the runtime takes the scheduling mode from. The
compile-time coverage lives in `test/lit/pto/pto_entry_sched_mode*.pto`,
`test/lit/vpto/pto_entry_sched_mode_annotations.pto` and
`ptodsl/tests/test_sched_mode.py`.
"""

import os
import tempfile
from pathlib import Path

import numpy as np

from common import auto_main, golden_output_case
from ptodsl import pto

# Prefix bisheng uses for the per-kernel CCE scheduling-mode ELF section.
_SCHED_MODE_SECTION = b".ScheduleMode_"

# Keep the two kernels below byte-for-byte identical apart from the decorator:
# any body difference would make the comparison measure something else.


@pto.jit(name="rowexpandmul_i8_8x8_sched_default", kernel_kind="vector", target="a5",
         backend="vpto", mode="auto")
def rowexpandmul_i8_8x8_sched_default(a: pto.ptr(pto.i8, "gm"), b: pto.ptr(pto.i8, "gm"),
                                      out: pto.ptr(pto.i8, "gm")):
    a_view = pto.make_tensor_view(a, shape=[8, 8], strides=[8, 1])
    b_view = pto.make_tensor_view(b, shape=[8, 1], strides=[1, 1])
    out_view = pto.make_tensor_view(out, shape=[8, 8], strides=[8, 1])
    lhs = pto.alloc_tile(shape=[8, 32], dtype=pto.i8, valid_shape=[8, 8], blayout="RowMajor")
    rhs = pto.alloc_tile(shape=[32, 1], dtype=pto.i8, valid_shape=[8, 1], blayout="ColMajor")
    dst = pto.alloc_tile(shape=[8, 32], dtype=pto.i8, valid_shape=[8, 8], blayout="RowMajor")
    pto.tile.load(a_view, lhs)
    pto.tile.load(b_view, rhs)
    pto.tile.rowexpandmul(lhs, rhs, dst)
    pto.tile.store(dst, out_view)


@pto.jit(name="rowexpandmul_i8_8x8_sched_batch", kernel_kind="vector", target="a5",
         backend="vpto", mode="auto", sched_mode=1)
def rowexpandmul_i8_8x8_sched_batch(a: pto.ptr(pto.i8, "gm"), b: pto.ptr(pto.i8, "gm"),
                                    out: pto.ptr(pto.i8, "gm")):
    a_view = pto.make_tensor_view(a, shape=[8, 8], strides=[8, 1])
    b_view = pto.make_tensor_view(b, shape=[8, 1], strides=[1, 1])
    out_view = pto.make_tensor_view(out, shape=[8, 8], strides=[8, 1])
    lhs = pto.alloc_tile(shape=[8, 32], dtype=pto.i8, valid_shape=[8, 8], blayout="RowMajor")
    rhs = pto.alloc_tile(shape=[32, 1], dtype=pto.i8, valid_shape=[8, 1], blayout="ColMajor")
    dst = pto.alloc_tile(shape=[8, 32], dtype=pto.i8, valid_shape=[8, 8], blayout="RowMajor")
    pto.tile.load(a_view, lhs)
    pto.tile.load(b_view, rhs)
    pto.tile.rowexpandmul(lhs, rhs, dst)
    pto.tile.store(dst, out_view)


def _device_image(compiled, name):
    """Build one kernel and return the device image ptoas produced for it.

    The native build is scratch-only unless PTODSL_CACHE_DIR is set, and its
    scratch tree is removed once the host library is mapped, so point it at a
    cache directory for the duration of the build to keep kernel.o readable.
    """
    previous = os.environ.get("PTODSL_CACHE_DIR")
    if previous is None:
        os.environ["PTODSL_CACHE_DIR"] = tempfile.mkdtemp(prefix="ptodsl-sched-mode-")
    try:
        artifacts = Path(compiled.native_library_path()).parent
    finally:
        if previous is None:
            os.environ.pop("PTODSL_CACHE_DIR", None)
    image = artifacts / "kernel.o"
    assert image.is_file(), (
        f"the native build of {name!r} kept no device image at {image}; this case "
        "asserts the CCE scheduling-mode section in that image"
    )
    return image.read_bytes()


def _launch_with_schedule_mode_assert(kernel, name, *, expect_sched_mode, grid):
    """Return a run hook that asserts the device image, then launches."""

    def run(device_inputs, launch_args, stream):
        compiled = kernel.compile()
        image = _device_image(compiled, name)
        section = _SCHED_MODE_SECTION + name.encode()
        found = section in image
        assert found is expect_sched_mode, (
            f"device image of {name!r} "
            f"{'is missing' if expect_sched_mode else 'unexpectedly has'} "
            f"{section.decode()}: the kernel would run with the wrong CCE scheduling "
            "mode while still producing the right numbers"
        )
        compiled[grid, stream](*device_inputs, *launch_args)

    return run


def _inputs():
    lhs = np.arange(-32, 32, dtype=np.int8).reshape(8, 8)
    rhs = np.array([-128, -17, -3, -1, 0, 3, 17, 127], dtype=np.int8).reshape(8, 1)
    return [lhs, rhs]


def _expected(lhs, rhs):
    return (lhs.astype(np.int16) * rhs.astype(np.int16)).astype(np.int8)


CASES = [
    golden_output_case("rowexpandmul_i8_8x8_sched_default", rowexpandmul_i8_8x8_sched_default,
                       inputs=_inputs, expected=_expected, rtol=0, atol=0),
    golden_output_case("rowexpandmul_i8_8x8_sched_batch", rowexpandmul_i8_8x8_sched_batch,
                       inputs=_inputs, expected=_expected, rtol=0, atol=0),
]

CASES[0]["run"] = _launch_with_schedule_mode_assert(
    CASES[0]["kernel"], CASES[0]["name"], expect_sched_mode=False, grid=CASES[0]["grid"])
CASES[1]["run"] = _launch_with_schedule_mode_assert(
    CASES[1]["kernel"], CASES[1]["name"], expect_sched_mode=True, grid=CASES[1]["grid"])


auto_main(globals())
