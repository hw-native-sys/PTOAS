# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""TileLib render abstraction-level test for pto.tadd.

Asserts the rendered MLIR is *on par* with the tilelang golden
(fixtures/tadd_a5_8x64_f32.golden.mlir): same structured abstraction, not byte-identical
(ptodsl differs in SSA naming, constant hoisting, index-vs-i32 carry, ptr typing).
"""

import json
import unittest
from pathlib import Path

from ptoas.mlir.dialects import pto as pto_dialect
from ptoas.mlir.ir import Context
from ptodsl.tilelib._compiler_runtime import materialize
from ptodsl.tilelib import TileSpec, f32
from TileOps.a5.tadd import template_tadd

FIXTURE = Path(__file__).parent / "fixtures" / "tadd_a5_8x64_f32.golden.mlir"

# The structured abstraction the migration must preserve (see golden fixture).
REQUIRED_OPS = [
    "pto.tile_buf_addr",
    "!pto.ptr<f32, ub>",
    "pto.tile_valid_rows",
    "pto.tile_valid_cols",
    "scf.for",
    "iter_args",  # the inner loop carries `remained` (AST-rewrite .carry path)
    "pto.plt_b32",
    "pto.vlds",
    "pto.vadd",
    "pto.vsts",
    "!pto.vreg<64xf32>",
]

# The low-level pointer style the team explicitly rejected for TileLib templates.
FORBIDDEN_OPS = ["pto.castptr", "pto.addptr"]


def _render():
    spec = TileSpec(shape=(8, 64), dtype=f32)
    return template_tadd.specialize(src0=spec, src1=spec, dst=spec).mlir_text()


def _materialize(context):
    tile_spec = {
        "kind": "tile",
        "shape": [8, 64],
        "valid_shape": [8, 64],
        "dtype": "f32",
        "memory_space": "ub",
        "config": {"b_layout": "row_major", "s_layout": "none_box"},
    }
    return materialize(
        "a5",
        "pto.tadd",
        json.dumps([tile_spec, tile_spec, tile_spec]),
        "{}",
        "template_tadd",
        context,
    )


class TileLibRenderTest(unittest.TestCase):
    def test_renders_structured_abstraction(self):
        text = _render()
        for op in REQUIRED_OPS:
            self.assertIn(op, text)
        for op in FORBIDDEN_OPS:
            self.assertNotIn(op, text)

    def test_func_is_a_tilelang_instance(self):
        text = _render()
        self.assertIn('pto.target_arch = "a5"', text)
        self.assertIn("pto.tilelang.instance", text)
        self.assertIn("#pto.kernel_kind<vector>", text)
        self.assertIn("func.func @template_tadd", text)

    def test_golden_fixture_uses_same_abstraction(self):
        self.assertTrue(FIXTURE.exists(), f"missing golden fixture {FIXTURE}")
        golden = FIXTURE.read_text(encoding="utf-8")
        for op in ("pto.tile_buf_addr", "!pto.ptr<f32, ub>", "pto.vlds", "pto.vadd", "pto.vsts", "pto.plt_b32"):
            self.assertIn(op, golden)

    def test_materialize_uses_borrowed_context_and_returns_fresh_modules(self):
        context = Context()
        pto_dialect.register_dialect(context, load=True)
        first, first_entry = _materialize(context)
        second, second_entry = _materialize(context)

        self.assertIs(first.context, context)
        self.assertIs(second.context, context)
        self.assertIsNot(first, second)
        self.assertEqual(first_entry, "template_tadd")
        self.assertEqual(second_entry, "template_tadd")
        self.assertTrue(first.operation.verify())
        self.assertTrue(second.operation.verify())
        self.assertIn("func.func @template_tadd", str(first))

    def test_materialized_surface_wrappers_release_without_cycle_collection(self):
        context = Context()
        pto_dialect.register_dialect(context, load=True)
        module, _ = _materialize(context)
        self.assertEqual(context._get_live_operation_count(), 0)

        del module
        self.assertEqual(context._get_live_operation_count(), 0)


if __name__ == "__main__":
    unittest.main()
