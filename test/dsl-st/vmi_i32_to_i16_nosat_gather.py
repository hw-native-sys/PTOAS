#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under
# the terms and conditions of CANN Open Software License Agreement Version 2.0
# (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Check signed NOSAT narrowing when a B16 gather consumes the result."""

import numpy as np

from common import auto_main, golden_output_case
from ptodsl import pto


@pto.jit(name="i32_to_i16_nosat_gather", target="a5", backend="vpto", mode="explicit", kernel_kind="vector")
def gather(src: pto.ptr(pto.si32, "gm"), lut: pto.ptr(pto.f16, "gm"), dst: pto.ptr(pto.f16, "gm")):
    pto.init_core()
    source = pto.castptr(pto.i64(0), pto.ptr(pto.si32, "ub"))
    table = pto.castptr(pto.i64(512), pto.ptr(pto.f16, "ub"))
    out = pto.castptr(pto.i64(544), pto.ptr(pto.f16, "ub"))
    pto.mte_gm_ub(src, source, 0, 512, nburst=(1, 512, 512))
    pto.mte_gm_ub(lut, table, 0, 32, nburst=(1, 32, 32))
    pto.set_flag("MTE2", "V", event_id=0)
    pto.wait_flag("MTE2", "V", event_id=0)
    with pto.vecscope():
        mask = pto.vmi.create_mask(128, size=128)
        wide = pto.vmi.vload(source, 0, size=128)
        narrow = pto.vmi.vcvt(wide, to_dtype=pto.si16, saturate="NOSAT")
        offsets = pto.vmi.vand(narrow, pto.vmi.vbrc(pto.si16(15), size=128), mask)
        values = pto.vmi.vgather(table, offsets, mask)
        pto.vmi.vstore(values, out, 0, mask)
    pto.set_flag("V", "MTE3", event_id=0)
    pto.wait_flag("V", "MTE3", event_id=0)
    pto.mte_ub_gm(out, dst, 256, nburst=(1, 256, 256))


def inputs():
    lanes = np.arange(128, dtype=np.int32)
    bases = np.asarray([32768, -65536, 0, -32768, 65536, -131072, 2**30, -(2**31)], dtype=np.int32)
    return [bases[lanes // 16] + (lanes * 7 % 16), np.arange(16, dtype=np.float16)]


CASES = [
    golden_output_case(
        "i32_to_i16_nosat_gather",
        gather,
        inputs=inputs,
        expected=lambda src, lut: lut[src & 15],
        rtol=0,
        atol=0,
    ),
]


auto_main(globals())
