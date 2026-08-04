#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Regression: pto.vmi.vci accepts a dynamic MLIR index base (loop IV).

Dynamic loop indices must coerce to an i32 sreg so ODS/verify accept the op
and lowering emits ``VCI Vd, Sn`` (matches Ascend ``S.vci(T.int32(offset))``).

Also covers ``group=2`` (VL128 group-periodic) with a dynamic base. Lit
``vmi_to_vpto_iota_group2.pto`` checks the share lowering:
``%idx = pto.vci %base`` then ``return %idx, %idx``.

Camodel share+compute probe (not this unit test):
  ``ptodsl/examples/vci_vadds_share_launch.py`` → ``vci(0)+vadds(1000)+vsts``.

Run:
  python3 ptodsl/tests/test_vmi_vci_dynamic_index.py
"""

from __future__ import annotations

from ptodsl import pto


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


@pto.jit(target="a5", backend="vpto", mode="explicit")
def vmi_vci_const_i32_probe():
    dst = pto.alloc_tile(shape=[1, 64], dtype=pto.i32)
    offset = pto.const(0, dtype=pto.index)
    idx = pto.vmi.vci(pto.i32(0), size=64)
    pto.vmi.vstore(idx, dst.as_ptr(), offset)


@pto.jit(target="a5", backend="vpto", mode="explicit")
def vmi_vci_dynamic_index_probe():
    """vci(pass_id * 64) where pass_id is an MLIR index loop IV."""
    dst = pto.alloc_tile(shape=[1, 128], dtype=pto.i32)
    # AST rewrite turns range(...) into a dynamic index IV (scf.for).
    for pass_id in range(2):
        base = pass_id * 64
        idx = pto.vmi.vci(base, size=64)
        pto.vmi.vstore(idx, dst.as_ptr(), base)


@pto.jit(target="a5", backend="vpto", mode="explicit")
def vmi_vci_dynamic_group2_probe():
    """Dynamic base + group=2 → VL128 group-periodic iota ([0..63|0..63]+base)."""
    dst = pto.alloc_tile(shape=[1, 128], dtype=pto.i32)
    for pass_id in range(2):
        base = pass_id * 64
        idx = pto.vmi.vci(base, size=128, group=2)
        pto.vmi.vstore(idx, dst.as_ptr(), pto.const(0, dtype=pto.index))


def main() -> None:
    const_text = vmi_vci_const_i32_probe.compile().mlir_text()
    expect("pto.vmi.vci" in const_text, "const probe must emit pto.vmi.vci")
    expect(
        ": i32 -> !pto.vmi.vreg" in const_text,
        f"const probe must use i32 vci:\n{const_text[:800]}",
    )

    dyn_text = vmi_vci_dynamic_index_probe.compile().mlir_text()
    expect("pto.vmi.vci" in dyn_text, "dynamic probe must emit pto.vmi.vci")
    expect(
        "index -> !pto.vmi.vreg" not in dyn_text,
        "dynamic vci must not keep index as result element type",
    )
    expect(
        ": i32 -> !pto.vmi.vreg" in dyn_text,
        f"dynamic probe must coerce index→i32 vci:\n{dyn_text[:1600]}",
    )
    expect(
        "arith.index_cast" in dyn_text or "index_cast" in dyn_text,
        f"dynamic probe must index_cast before vci:\n{dyn_text[:1600]}",
    )

    g2_text = vmi_vci_dynamic_group2_probe.compile().mlir_text()
    expect("pto.vmi.vci" in g2_text, "group=2 probe must emit pto.vmi.vci")
    expect(
        "group = 2" in g2_text or "{group = 2" in g2_text,
        f"group=2 probe must preserve group attr:\n{g2_text[:2000]}",
    )
    expect(
        ": i32 -> !pto.vmi.vreg" in g2_text,
        f"group=2 probe must coerce index→i32 vci:\n{g2_text[:2000]}",
    )
    expect(
        "arith.index_cast" in g2_text or "index_cast" in g2_text,
        f"group=2 probe must index_cast before vci:\n{g2_text[:2000]}",
    )
    print("ptodsl_vmi_vci_dynamic_index: PASS")


if __name__ == "__main__":
    main()
