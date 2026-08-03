#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from ptodsl import pto


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


@pto.jit(target="a5", backend="vpto", mode="explicit")
def vmi_bf16_group8_carrier_probe():
    src_tile = pto.alloc_tile(shape=[1, 256], dtype=pto.bf16)
    offset = pto.const(0, dtype=pto.index)
    mask = pto.vmi.create_mask(256, size=256)
    zero = pto.vmi.vbrc(pto.bf16(0.0), size=256)
    source = pto.vmi.vload(src_tile.as_ptr(), offset, size=256)
    source_f32 = pto.vmi.vcvt(source, to_dtype=pto.f32)
    mask_f32 = pto.vmi.create_mask(256, size=256)
    maximum = pto.vmi.vcmax(source_f32, mask_f32, group=8)
    low, high = pto.vmi.vintlv(zero, source, mask)
    low_f32 = pto.vmi.vinterpret_cast(low, to_dtype=pto.f32)
    high_f32 = pto.vmi.vinterpret_cast(high, to_dtype=pto.f32)
    _ = maximum
    _ = low_f32
    _ = high_f32


def main() -> None:
    text = vmi_bf16_group8_carrier_probe.compile().mlir_text()
    expect(
        "!pto.vmi.vreg<256xbf16>" in text,
        "carrier probe must preserve the 256-lane BF16 source",
    )
    expect(
        "pto.vmi.vcmax" in text and "{group = 8 : i64}" in text,
        "carrier probe must emit one surface group=8 maximum",
    )
    expect(
        "-> !pto.vmi.vreg<8xf32>" in text,
        "explicit BF16-to-F32 conversion followed by group=8 maximum must "
        "infer an 8xf32 result",
    )
    expect(
        text.count("!pto.vmi.vreg<256xbf16> -> !pto.vmi.vreg<128xf32>") == 2,
        "both BF16 carrier halves must reinterpret to 128xf32 by total bits",
    )
    print("ptodsl_vmi_bf16_group8_carrier: PASS")


if __name__ == "__main__":
    main()
