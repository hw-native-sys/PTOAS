#!/usr/bin/env python3
# coding=utf-8
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
def vmi_vmull_inferred_probe():
    lhs_tile = pto.alloc_tile(shape=[1, 256], dtype=pto.i32)
    rhs_tile = pto.alloc_tile(shape=[1, 256], dtype=pto.i32)
    dst_tile = pto.alloc_tile(shape=[1, 256], dtype=pto.i32)
    offset = pto.const(0, dtype=pto.index)
    active_lanes = pto.const(256, dtype=pto.index)
    mask = pto.vmi.create_mask(active_lanes, size=256)
    lhs = pto.vmi.vload(lhs_tile.as_ptr(), offset, size=256)
    rhs = pto.vmi.vload(rhs_tile.as_ptr(), offset, size=256)
    low, high = pto.vmi.vmull(lhs, rhs, mask)
    pto.vmi.vstore(low, dst_tile.as_ptr(), offset, mask)
    _ = high


@pto.jit(target="a5", backend="vpto", mode="explicit")
def vmi_vmull_unsigned_zero_probe():
    lhs_tile = pto.alloc_tile(shape=[1, 128], dtype=pto.ui32)
    rhs_tile = pto.alloc_tile(shape=[1, 128], dtype=pto.ui32)
    offset = pto.const(0, dtype=pto.index)
    active_lanes = pto.const(128, dtype=pto.index)
    mask = pto.vmi.create_mask(active_lanes, size=128)
    lhs = pto.vmi.vload(lhs_tile.as_ptr(), offset, size=128)
    rhs = pto.vmi.vload(rhs_tile.as_ptr(), offset, size=128)
    low, high = pto.vmi.vmull(lhs, rhs, mask, pmode="zero")
    _ = low
    _ = high


def main() -> None:
    inferred_text = vmi_vmull_inferred_probe.compile().mlir_text()
    expect(inferred_text.count("pto.vmi.vmull") == 1, "inferred probe must emit one VMI VMULL")
    expect(
        "-> !pto.vmi.vreg<256xi32>, !pto.vmi.vreg<256xi32>" in inferred_text,
        "VMULL must infer both 256xi32 results",
    )

    unsigned_text = vmi_vmull_unsigned_zero_probe.compile().mlir_text()
    expect(unsigned_text.count("pto.vmi.vmull") == 1, "unsigned probe must emit one VMI VMULL")
    expect(
        "-> !pto.vmi.vreg<128xui32>, !pto.vmi.vreg<128xui32>" in unsigned_text,
        "VMULL must infer both 128xui32 results",
    )
    expect('pmode = "zero"' in unsigned_text, "explicit zero pmode must be preserved")
    print("ptodsl_vmi_vmull: PASS")


if __name__ == "__main__":
    main()
