#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Short dense load, narrowed, stored back as a short dense store.

The narrowing result is lane-strided while the short dense store carries a
single-carrier group packet, so the value only lands correctly when the
contiguous lane-stride normalization and the carrier bridge agree.  The output
buffer is preloaded with a canary, so a store that writes the wrong lanes or
runs past the vector shows up as a canary mismatch.
"""

import numpy as np
from common import auto_main
from ptodsl import pto

SRC_ELEMS = 256
OUT_ELEMS = 256
CANARY = 241


def build(n, offset, name):
    @pto.jit(name=name, target="a5", backend="vpto", mode="explicit",
             kernel_kind="vector", insert_sync=False)
    def kernel(src: pto.ptr(pto.f32, "gm"), out: pto.ptr(pto.f16, "gm")):
        ub_src = pto.castptr(pto.i64(1024), pto.ptr(pto.f32, "ub"))
        ub_out = pto.castptr(pto.i64(4096), pto.ptr(pto.f16, "ub"))
        pto.mte_gm_ub(src, ub_src, 0, SRC_ELEMS * 4, nburst=(1, SRC_ELEMS * 4, SRC_ELEMS * 4))
        pto.mte_gm_ub(out, ub_out, 0, OUT_ELEMS * 2, nburst=(1, OUT_ELEMS * 2, OUT_ELEMS * 2))
        pto.set_flag("MTE2", "V", event_id=0)
        pto.wait_flag("MTE2", "V", event_id=0)
        value = pto.vmi.vload(ub_src, offset, size=n)
        narrowed = pto.vmi.vcvt(value, to_dtype=pto.f16)
        pto.vmi.vstore(narrowed, ub_out, 0)
        pto.set_flag("V", "MTE3", event_id=0)
        pto.wait_flag("V", "MTE3", event_id=0)
        pto.mte_ub_gm(ub_out, out, OUT_ELEMS * 2, nburst=(1, OUT_ELEMS * 2, OUT_ELEMS * 2))
        pto.pipe_barrier(pto.Pipe.ALL)
    return kernel


CASES = []
for n in (1, 2, 4, 8, 64):
    for offset in (0, 3):
        name = f"short_narrow_store_f32_to_f16_n{n}_off{offset}"
        source = (np.arange(SRC_ELEMS, dtype=np.float32) + 1.0)
        output = np.full(OUT_ELEMS, CANARY, dtype=np.float16)
        expected = output.copy()
        expected[:n] = source[offset:offset + n].astype(np.float16)

        def make_case(source=source, expected=expected, output=output):
            return [source.copy(), output.copy()], expected.copy(), []

        CASES.append(dict(name=name, kernel=build(n, offset, name), make_case=make_case))


auto_main(globals())
