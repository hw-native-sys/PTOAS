# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Isolate cell: stages=1, block_k=1024 — wide tile without dual-buffer stages."""

from ptodsl import pto


@pto.jit(target="a5", backend="vpto", mode="explicit")
def persistent_quant_stages1_k1024(*, BLOCK_COUNT: pto.const_expr = 4):
    x_ub = pto.alloc_tile(shape=[1, 1024], dtype=pto.bf16)
    y_ub = pto.alloc_tile(shape=[1, 1024], dtype=pto.f8e4m3)
    scale_ub = pto.alloc_tile(shape=[1, 8], dtype=pto.f32)
    off0 = pto.const(0, dtype=pto.index)
    scale_slot = pto.vmi.vload(scale_ub.as_ptr(), off0, size=1)
    scale = pto.vmi.vbrc(scale_slot, size=1024)
    for i in range(BLOCK_COUNT):
        off = i * 1024
        x_bf16 = pto.vmi.vload(x_ub.as_ptr(), off, size=1024)
        x_f32 = pto.vmi.vcvt(x_bf16, pto.f32)
        scaled = pto.vmi.vmul(x_f32, scale)
        y = pto.vmi.vcvt(scaled, pto.f8e4m3, saturate="SAT")
        pto.vmi.vstore(y, y_ub.as_ptr(), off)


if __name__ == "__main__":
    try:
        print(persistent_quant_stages1_k1024.compile().mlir_text())
    except Exception as exc:  # noqa: BLE001
        print(f"emit/lower may fail (layout): {exc}")
