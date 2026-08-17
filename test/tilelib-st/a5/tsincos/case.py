#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""PTODSL ST coverage for the A5 f32 sin/cos SoftLib implementations."""

from pathlib import Path
import sys

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import auto_main, golden_output_case
from ptodsl import pto, scalar


LANES = 32


@pto.simt
def _sincos_body(
    inp: pto.ptr(pto.f32, "gm"),
    sin_out: pto.ptr(pto.f32, "gm"),
    cos_out: pto.ptr(pto.f32, "gm"),
):
    tid = pto.get_tid_x()
    idx = scalar.index_cast(tid)
    value = scalar.load(inp, idx)
    scalar.store(pto.sin(value), sin_out, idx)
    scalar.store(pto.cos(value), cos_out, idx)


@pto.jit(
    name="tsincos_f32_soft_1x32",
    kernel_kind="vector",
    target="a5",
    mode="explicit",
    insert_sync=False,
)
def _kernel(
    inp: pto.ptr(pto.f32, "gm"),
    sin_out: pto.ptr(pto.f32, "gm"),
    cos_out: pto.ptr(pto.f32, "gm"),
):
    _sincos_body[LANES, 1, 1](inp, sin_out, cos_out)
    pto.pipe_barrier(pto.Pipe.ALL)


def _make_inputs():
    # Include quadrant boundaries and a Box-Muller-like [0, 2*pi] range.
    values = np.array(
        [0.0, np.pi / 6, np.pi / 4, np.pi / 2, 2 * np.pi / 3, np.pi,
         3 * np.pi / 2, 2 * np.pi, -np.pi / 3, -np.pi / 2, 7.0, -7.0],
        dtype=np.float32,
    )
    return [np.resize(values, (LANES,))]


def _make_expected(values):
    return np.sin(values).astype(np.float32), np.cos(values).astype(np.float32)


def _make_case():
    values = _make_inputs()[0]
    sin_expected, cos_expected = _make_expected(values)
    return [values, np.zeros_like(values), np.zeros_like(values)], (sin_expected, cos_expected)


def _check_case(device_inputs, expected):
    np.testing.assert_allclose(device_inputs[1].cpu().numpy(), expected[0], rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(device_inputs[2].cpu().numpy(), expected[1], rtol=2e-5, atol=2e-5)


CASES = [{
    "name": "tsincos_f32_soft_1x32",
    "kernel": _kernel,
    "make_case": _make_case,
    "check": _check_case,
}]


auto_main(globals())
