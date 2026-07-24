# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Python DSL form of ``desired_vmi.pto`` — explicit ``dist_mode=\"unpack\"`` load.

Canonical IR: ``desired_vmi.pto``. Illegal / residual today.
PTODSL ``vload(..., dist_mode=\"unpack\")`` currently requires a one-step
``to_dtype`` widen; the IR keeps bf16 — see the ``.pto`` sibling.
"""

from ptodsl import pto


@pto.jit(target="a5", backend="vpto", mode="explicit")
def bf16_strip_unpack():
    src = pto.alloc_tile(shape=[1, 64], dtype=pto.bf16)
    dst = pto.alloc_tile(shape=[1, 64], dtype=pto.bf16)
    off = pto.const(0, dtype=pto.index)
    # Mirrors desired_vmi.pto: dist_mode="unpack" (illegal / residual today).
    x = pto.vmi.vload(src.as_ptr(), off, size=64, dist_mode="unpack")
    xf = pto.vmi.vcvt(x, pto.f32)
    scale = pto.vmi.vbrc(pto.f32(1.0), size=64)
    y = pto.vmi.vmul(xf, scale)
    yo = pto.vmi.vcvt(y, pto.bf16, saturate="SAT")
    pto.vmi.vstore(yo, dst.as_ptr(), off)


if __name__ == "__main__":
    try:
        print(bf16_strip_unpack.compile().mlir_text())
    except Exception as exc:  # noqa: BLE001 — document the gap
        print(f"desired emit failed (expected until unpack bf16 lands): {exc}")
