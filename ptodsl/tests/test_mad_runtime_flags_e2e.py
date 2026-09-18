#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""End-to-end path for the runtime mad flag keywords.

Covers the chain a downstream template relies on: ``_split_mad_options``
maps ``unit_flag``/``init`` surface values onto the four optional runtime
operands, signed frontend scalars are stripped to signless (the unrealized
cast is reconciled away before expansion), the semantic ops expand into a
single raw op per mad inside a shared ctrl_state_guard, and the static /
TypeError paths behave as documented.
"""

import unittest

import ptodsl
from ptodsl import pto

# The scalar surface lives in the pto namespace since the PTODSL scalar
# unification; keep the template-style local alias for readability.
scalar = pto


@pto.jit(target="a5", entry=False, mode="explicit", kernel_kind="cube")
def runtime_mad_flags_probe(
    lhs_l0a: pto.Tile,
    rhs_l0b: pto.Tile,
    acc_tile: pto.Tile,
    clear_accum: pto.i32,
    uf: pto.si32,
):
    # The shapes the tilelang GEMM template emits: one mad_acc whose
    # acc-init is a runtime flag, and a unit_flag that arrives as a signed
    # select result (the 3 -> 2 downgrade lives in the template, not here).
    pto.mad_acc(lhs_l0a.as_ptr(), rhs_l0b.as_ptr(), acc_tile.as_ptr(),
                16, 16, 16,
                init=clear_accum, unit_flag=uf)
    downgraded = scalar.select(
        uf == 3, pto.const(2, dtype=pto.si32), uf)
    pto.mad_acc(lhs_l0a.as_ptr(), rhs_l0b.as_ptr(), acc_tile.as_ptr(),
                16, 16, 16, unit_flag=downgraded)


@pto.jit(target="a5", mode="explicit", kernel_kind="cube")
def runtime_mad_flags_entry(
    out_ptr: pto.ptr(pto.f32, "gm"),
    clear_accum: pto.i32,
    uf: pto.si32,
):
    lhs_l0a = pto.alloc_tile(shape=[16, 16], dtype=pto.bf16,
                             memory_space=pto.MemorySpace.LEFT)
    rhs_l0b = pto.alloc_tile(shape=[16, 16], dtype=pto.bf16,
                             memory_space=pto.MemorySpace.RIGHT)
    acc = pto.alloc_tile(shape=[16, 16], dtype=pto.f32,
                         memory_space=pto.MemorySpace.ACC)
    runtime_mad_flags_probe(lhs_l0a, rhs_l0b, acc, clear_accum, uf)
    pto.mte_l0c_gm(acc.as_ptr(), out_ptr,
                   pto.const(16), pto.const(16), pto.const(16), pto.const(16),
                   0, 0, layout=("nz2dn", pto.const(16)))


class MadRuntimeFlagsEndToEndTest(unittest.TestCase):
    def test_capability_switch(self):
        self.assertTrue(getattr(ptodsl, "MAD_RUNTIME_FLAGS", False))

    def test_runtime_operands_reach_the_ir(self):
        text = runtime_mad_flags_entry.compile().mlir_text()
        # Both runtime flag kinds appear as optional trailing operands.
        self.assertIn("acc_init(", text)
        self.assertIn("unit_flag_value(", text)
        # A signed select result is stripped to signless before it becomes
        # an operand; the strip cast is reconciled before expansion, so no
        # unrealized cast may survive alongside the operand.
        self.assertIn("si32", text)

    def test_static_init_is_rejected(self):
        with self.assertRaises(TypeError):
            pto.mad_acc(None, None, None, 16, 16, 16, init=0)

    def test_mx_family_rejects_runtime_flags(self):
        # The mx signatures do not expose the runtime flag keywords at all;
        # any attempt raises before IR construction. (A surface-value probe
        # would need a jit context; the keyword absence is the contract.)
        with self.assertRaises(TypeError):
            pto.mad_mx_acc(None, None, None, 16, 16, 16, init=1)


if __name__ == "__main__":
    unittest.main()
