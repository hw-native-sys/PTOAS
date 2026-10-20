#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See the License in the root of the software repository for the full text of the License.

"""Issue #1626: f16 vexpdif below 128 lanes through the PTODSL VMI API.

Below 128 lanes the EVEN/ODD interleave relation of the dedicated f16
lowering cannot materialize, so vmi-lower-unified-to-legacy widens x and max
to f32 first and lowers the op on the f32 path. These kernels mirror the
issue repro (mte loads, prefix masks, pmode="zero") at the failing lane
counts and compare exp(x - max) against a float32 golden.
"""

import numpy as np

from common import auto_main, golden_output_case
from ptodsl import pto


def _vexpdif_f16_kernel(name, vl):
    @pto.jit(name=name, kernel_kind="vector", target="a5", backend="vpto", mode="explicit")
    def kernel(
        x_ptr: pto.ptr(pto.f16, "gm"),
        max_ptr: pto.ptr(pto.f16, "gm"),
        out_ptr: pto.ptr(pto.f32, "gm"),
    ):
        in_nbytes = vl * 2
        out_nbytes = vl * 4
        x_ub = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.f16, "ub"))
        max_ub = pto.castptr(pto.const(512, dtype=pto.i64), pto.ptr(pto.f16, "ub"))
        out_ub = pto.castptr(pto.const(1024, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
        pto.mte_gm_ub(x_ptr, x_ub, 0, in_nbytes, nburst=(1, 0, 0))
        pto.mte_gm_ub(max_ptr, max_ub, 0, in_nbytes, nburst=(1, 0, 0))
        pto.set_flag("MTE2", "V", event_id=0)
        pto.wait_flag("MTE2", "V", event_id=0)
        x = pto.vmi.vload(x_ub, pto.const(0, dtype=pto.index), size=vl)
        max_value = pto.vmi.vload(max_ub, pto.const(0, dtype=pto.index), size=vl)
        mask = pto.vmi.create_mask(vl, size=vl)
        result = pto.vmi.vexpdif(x, max_value, mask, pmode="zero")
        pto.vmi.vstore(result, out_ub, pto.const(0, dtype=pto.index))
        pto.set_flag("V", "MTE3", event_id=0)
        pto.wait_flag("V", "MTE3", event_id=0)
        pto.mte_ub_gm(out_ub, out_ptr, out_nbytes, nburst=(1, 0, 0))
        pto.pipe_barrier(pto.Pipe.ALL)

    return kernel


def _case(vl):
    kernel = _vexpdif_f16_kernel(f"vexpdif_f16_vl{vl}_kernel", vl)

    def make_inputs(vl=vl):
        rng = np.random.default_rng(1626)
        x = rng.uniform(-8.0, 8.0, size=vl).astype(np.float16)
        mx = rng.uniform(-4.0, 4.0, size=vl).astype(np.float16)
        return [x, mx]

    def golden(x, mx):
        # Documented per-lane semantics: expf(x - max) computed in f32; the
        # small-VL lowering widens the f16 inputs exactly before the op.
        return np.exp(x.astype(np.float32) - mx.astype(np.float32))

    return golden_output_case(
        f"vexpdif_f16_vl{vl}",
        kernel,
        inputs=make_inputs,
        expected=golden,
        rtol=1e-3,
        atol=1e-6,
    )


CASES = [
    _case(1),
    _case(2),
    _case(8),
    _case(64),
]

auto_main(globals())
