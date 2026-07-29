#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Small case: ``pto.vmi.vcmax(..., return_index=True)`` → (1xf32, 1xi32)."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ptodsl"))

from ptodsl import pto


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


@pto.jit(target="a5", backend="vpto", mode="explicit")
def vmi_vcmax_return_index_probe():
    src_tile = pto.alloc_tile(shape=[1, 64], dtype=pto.f32)
    # 32B-aligned parks for one-point stores (8×f32 / 8×i32).
    mx_tile = pto.alloc_tile(shape=[1, 8], dtype=pto.f32)
    idx_tile = pto.alloc_tile(shape=[1, 8], dtype=pto.i32)
    offset = pto.const(0, dtype=pto.index)
    mask = pto.vmi.create_mask(pto.const(64, dtype=pto.index), size=64)
    mask1 = pto.vmi.create_mask(pto.const(1, dtype=pto.index), size=1)
    src = pto.vmi.vload(src_tile.as_ptr(), offset, size=64)
    mx, idx = pto.vmi.vcmax(src, mask, return_index=True)
    pto.vmi.vstore(mx, mx_tile.as_ptr(), offset, mask1)
    pto.vmi.vstore(idx, idx_tile.as_ptr(), offset, mask1)


@pto.jit(target="a5", backend="vpto", mode="explicit")
def vmi_vcmax_value_only_probe():
    src_tile = pto.alloc_tile(shape=[1, 64], dtype=pto.f32)
    mx_tile = pto.alloc_tile(shape=[1, 8], dtype=pto.f32)
    offset = pto.const(0, dtype=pto.index)
    mask = pto.vmi.create_mask(pto.const(64, dtype=pto.index), size=64)
    mask1 = pto.vmi.create_mask(pto.const(1, dtype=pto.index), size=1)
    src = pto.vmi.vload(src_tile.as_ptr(), offset, size=64)
    mx = pto.vmi.vcmax(src, mask)
    pto.vmi.vstore(mx, mx_tile.as_ptr(), offset, mask1)


def main() -> None:
    dual = vmi_vcmax_return_index_probe.compile()
    dual_text = dual.mlir_text()
    expect(dual_text.count("pto.vmi.vcmax") == 1, "dual probe must emit one VMI vcmax")
    expect(
        "-> !pto.vmi.vreg<1xf32>" in dual_text and "!pto.vmi.vreg<1xi32>" in dual_text,
        f"dual-output vcmax must yield 1xf32 + 1xi32, got:\n{dual_text}",
    )

    value_only = vmi_vcmax_value_only_probe.compile()
    value_text = value_only.mlir_text()
    expect(value_text.count("pto.vmi.vcmax") == 1, "value-only probe must emit one VMI vcmax")
    expect(
        "-> !pto.vmi.vreg<1xf32>" in value_text,
        "value-only vcmax must keep a single 1xf32 result",
    )
    expect(
        "!pto.vmi.vreg<1xi32>" not in value_text,
        "value-only vcmax must not invent an index result",
    )
    print("ptodsl_vmi_vcmax_return_index: PASS")


if __name__ == "__main__":
    main()
