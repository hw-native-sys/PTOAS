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
def vmi_vshr_signed_probe():
    lhs_tile = pto.alloc_tile(shape=[1, 128], dtype=pto.si32)
    rhs_tile = pto.alloc_tile(shape=[1, 128], dtype=pto.si32)
    offset = pto.const(0, dtype=pto.index)
    active_lanes = pto.const(128, dtype=pto.index)
    mask = pto.vmi.create_mask(active_lanes, size=128)
    lhs = pto.vmi.vload(lhs_tile.as_ptr(), offset, size=128)
    rhs = pto.vmi.vload(rhs_tile.as_ptr(), offset, size=128)
    shifted = pto.vmi.vshr(lhs, rhs, mask)
    shifted_scalar = pto.vmi.vshrs(lhs, pto.si32(3), mask)
    _ = shifted
    _ = shifted_scalar


@pto.jit(target="a5", backend="vpto", mode="explicit")
def vmi_vshr_unsigned_probe():
    lhs_tile = pto.alloc_tile(shape=[1, 128], dtype=pto.ui32)
    rhs_tile = pto.alloc_tile(shape=[1, 128], dtype=pto.ui32)
    offset = pto.const(0, dtype=pto.index)
    active_lanes = pto.const(128, dtype=pto.index)
    mask = pto.vmi.create_mask(active_lanes, size=128)
    lhs = pto.vmi.vload(lhs_tile.as_ptr(), offset, size=128)
    rhs = pto.vmi.vload(rhs_tile.as_ptr(), offset, size=128)
    shifted = pto.vmi.vshr(lhs, rhs, mask)
    shifted_scalar = pto.vmi.vshrs(lhs, pto.ui32(3), mask)
    _ = shifted
    _ = shifted_scalar


def main() -> None:
    signed_text = vmi_vshr_signed_probe.compile().mlir_text()
    expect("pto.vmi.vshr" in signed_text, "signed probe must emit pto.vmi.vshr")
    expect("pto.vmi.vshrs" in signed_text, "signed probe must emit pto.vmi.vshrs")
    expect(", i16," in signed_text, "signed probe must use an i16 shift amount")
    expect(
        "!pto.vmi.vreg<128xsi32>" in signed_text,
        "signed probe must preserve the explicit si32 VMI element type",
    )

    unsigned_text = vmi_vshr_unsigned_probe.compile().mlir_text()
    expect("pto.vmi.vshr" in unsigned_text, "unsigned probe must emit pto.vmi.vshr")
    expect("pto.vmi.vshrs" in unsigned_text, "unsigned probe must emit pto.vmi.vshrs")
    expect(", i16," in unsigned_text, "unsigned probe must use an i16 shift amount")
    expect(
        "!pto.vmi.vreg<128xui32>" in unsigned_text,
        "unsigned probe must preserve the explicit ui32 VMI element type",
    )
    print("ptodsl_vmi_vshr_signedness: PASS")


if __name__ == "__main__":
    main()
