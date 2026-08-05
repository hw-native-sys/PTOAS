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


@pto.jit(target="a5", backend="vpto", mode="explicit")
def vmi_vci_subvl_i32_g2_probe():
    """Sub-VL: i32 size=64 group=2 → [0..31|0..31] in one physical VL."""
    dst = pto.alloc_tile(shape=[1, 64], dtype=pto.i32)
    idx = pto.vmi.vci(pto.i32(0), size=64, group=2)
    pto.vmi.vstore(idx, dst.as_ptr(), pto.const(0, dtype=pto.index))


@pto.jit(target="a5", backend="vpto", mode="explicit")
def vmi_vci_subvl_i16_g2_probe():
    """Sub-VL: i16 size=128 group=2 → [0..63|0..63] in one physical VL."""
    dst = pto.alloc_tile(shape=[1, 128], dtype=pto.i16)
    idx = pto.vmi.vci(pto.i16(0), size=128, group=2)
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

    subvl32 = vmi_vci_subvl_i32_g2_probe.compile().mlir_text()
    expect(
        "group = 2" in subvl32 or "{group = 2" in subvl32,
        f"sub-VL i32 g2 must preserve group:\n{subvl32[:1500]}",
    )
    expect(
        "!pto.vmi.vreg<64xi32" in subvl32,
        f"sub-VL i32 g2 must be 64xi32:\n{subvl32[:1500]}",
    )

    subvl16 = vmi_vci_subvl_i16_g2_probe.compile().mlir_text()
    expect(
        "group = 2" in subvl16 or "{group = 2" in subvl16,
        f"sub-VL i16 g2 must preserve group:\n{subvl16[:1500]}",
    )
    expect(
        "!pto.vmi.vreg<128xi16" in subvl16,
        f"sub-VL i16 g2 must be 128xi16:\n{subvl16[:1500]}",
    )

    # Untileable: i32 size=48 group=2 → S=24 does not tile phys 64.
    from ptodsl._vmi_namespace import _check_vci_group_tiles_phys_vl
    from ptoas.mlir.ir import Context, IntegerType

    with Context():
        i32 = IntegerType.get_signless(32)
        try:
            _check_vci_group_tiles_phys_vl(
                i32, 48, 2, context="pto.vmi.vci(...)"
            )
            raise AssertionError(
                "expected ValueError for untileable group_size=24"
            )
        except ValueError as err:
            expect(
                "physical lanes" in str(err),
                f"untileable group must mention physical lanes, got: {err}",
            )

        # P2: group=1 is a single group → equivalent to ungrouped; legal even
        # when size does not tile physical VL (same as ungrouped size=100).
        _check_vci_group_tiles_phys_vl(
            i32, 100, 1, context="pto.vmi.vci(...)"
        )

    @pto.jit(target="a5", backend="vpto", mode="explicit")
    def vmi_vci_group1_tail_probe():
        dst = pto.alloc_tile(shape=[1, 128], dtype=pto.i32)
        idx = pto.vmi.vci(pto.i32(0), size=100, group=1)
        pto.vmi.vstore(
            idx, dst.as_ptr(), pto.const(0, dtype=pto.index)
        )

    g1_tail = vmi_vci_group1_tail_probe.compile().mlir_text()
    expect(
        "pto.vmi.vci" in g1_tail,
        f"group=1 size=100 must emit vci:\n{g1_tail[:1500]}",
    )
    expect(
        "!pto.vmi.vreg<100xi32" in g1_tail,
        f"group=1 size=100 must keep logical length 100:\n{g1_tail[:1500]}",
    )

    print("ptodsl_vmi_vci_dynamic_index: PASS")

if __name__ == "__main__":
    main()
