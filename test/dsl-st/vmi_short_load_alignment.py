#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Aligned short loads and element-aligned scalar loads with output canaries."""

import numpy as np
from common import auto_main
from ptodsl import pto


def build(dtype, n, offset, name):
    @pto.jit(name=name, target="a5", backend="vpto", mode="explicit",
             kernel_kind="vector", insert_sync=False)
    def kernel(src: pto.ptr(dtype, "gm"), out: pto.ptr(dtype, "gm")):
        ub_src = pto.castptr(pto.i64(1024), pto.ptr(dtype, "ub"))
        ub_out = pto.castptr(pto.i64(2048), pto.ptr(dtype, "ub"))
        pto.mte_gm_ub(src, ub_src, 0, 128, nburst=(1, 128, 128))
        pto.mte_gm_ub(out, ub_out, 0, 128, nburst=(1, 128, 128))
        pto.set_flag("MTE2", "V", event_id=0)
        pto.wait_flag("MTE2", "V", event_id=0)
        value = pto.vmi.vload(ub_src, offset, size=n)
        pto.vmi.vstore(value, ub_out, 0)
        pto.set_flag("V", "MTE3", event_id=0)
        pto.wait_flag("V", "MTE3", event_id=0)
        pto.mte_ub_gm(ub_out, out, 128, nburst=(1, 128, 128))
        pto.pipe_barrier(pto.Pipe.ALL)
    return kernel


CASES = []
for dtype, npdtype, bits in ((pto.ui8, np.uint8, 8), (pto.f16, np.float16, 16), (pto.f32, np.float32, 32)):
    for kind, n, offset in (("aligned", 8, 0), ("aligned_offset", 8, 256 // bits),
                            ("scalar_unaligned", 1, 1)):
        name = f"short_load_b{bits}_{kind}"
        source = (np.arange(1024 // bits) + 1).astype(npdtype)
        output = np.full_like(source, 241)
        expected = output.copy()
        expected[:n] = source[offset:offset + n]
        def make_case(source=source, expected=expected, output=output):
            return [source.copy(), output.copy()], expected.copy(), []
        CASES.append(dict(name=name, kernel=build(dtype, n, offset, name), make_case=make_case))


auto_main(globals())
