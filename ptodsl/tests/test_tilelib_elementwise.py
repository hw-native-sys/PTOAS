# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Phase-4 breadth test: each ported elementwise op selects + renders to the structured
abstraction, using the right vector op."""

import ast
import unittest
from pathlib import Path

from ptodsl import pto
import ptodsl.tilelib as tilelib
import ptodsl.tilelib.templates.a5._elementwise as elementwise
from ptodsl.tilelib import (
    ScalarSpec,
    ScalarType,
    TileSpec,
    legal_candidates,
    select,
)
from ptodsl.tilelib.templates.a5._elementwise import (
    emit_scalar_binary_1d,
    emit_scalar_binary_2d,
    emit_scalar_fill_1d,
    emit_scalar_fill_2d,
    emit_unary_1d,
    emit_unary_2d,
    register_binary,
)

# op -> (expected template name, expected vector op in the rendered MLIR)
ELEMENTWISE = {
    "pto.tsub": ("template_tsub", "pto.vsub"),
    "pto.tmul": ("template_tmul", "pto.vmul"),
    "pto.tmax": ("template_tmax", "pto.vmax"),
    "pto.tmin": ("template_tmin", "pto.vmin"),
    "pto.tdiv": ("template_tdiv", "pto.vdiv"),
}

PRODUCTION_UNARY_1D = {
    "pto.tabs": ("template_tabs_1d", "template_tabs", "pto.vabs", "f32"),
    "pto.texp": ("template_texp_1d", "template_texp", "pto.vexp", "f32"),
    "pto.tneg": ("template_tneg_1d", "template_tneg", "pto.vneg", "i8"),
    "pto.tnot": ("template_tnot_1d", "template_tnot", "pto.vnot", "ui16"),
    "pto.trelu": ("template_trelu_1d", "template_trelu", "pto.vrelu", "i32"),
    "pto.trsqrt": (
        "template_trsqrt_1d",
        "template_trsqrt",
        "pto.vsqrt",
        "f16",
    ),
    "pto.tsqrt": (
        "template_tsqrt_1d",
        "template_tsqrt",
        "pto.vsqrt",
        "f32",
    ),
}

# Structured abstraction every elementwise template must preserve.
SHARED_OPS = ["pto.tile_buf_addr", "!pto.ptr<f32, ub>", "scf.for", "iter_args",
              "pto.plt_b32", "pto.vlds", "pto.vsts", "pto.tilelang.instance"]


def _f32_specs():
    spec = TileSpec(shape=(8, 64), dtype=ScalarType("f32"))
    return {"src0": spec, "src1": spec, "dst": spec}


def _test_template(*, name, loop_depth):
    return tilelib.tile_template(
        op=f"test.{name}",
        target="a5",
        name=name,
        loop_depth=loop_depth,
        register=False,
    )


@_test_template(name="test_unary_1d", loop_depth=1)
def _test_unary_1d(src: pto.Tile, dst: pto.Tile):
    emit_unary_1d(src, dst, pto.vabs)


@_test_template(name="test_unary_2d", loop_depth=2)
def _test_unary_2d(src: pto.Tile, dst: pto.Tile):
    emit_unary_2d(src, dst, pto.vabs)


_test_binary_1d = register_binary(
    op="test.elementwise.binary",
    name="test_binary_1d",
    vector_op=pto.vadd,
    dtypes=[("f32", "f32", "f32")],
    traversal="1d",
    priority=10,
    candidate_id=99,
)


_test_binary_2d = register_binary(
    op="test.elementwise.binary",
    name="test_binary_2d",
    vector_op=pto.vadd,
    dtypes=[("f32", "f32", "f32")],
    traversal="2d",
    priority=0,
    candidate_id=0,
)


@_test_template(name="test_scalar_1d", loop_depth=1)
def _test_scalar_1d(src: pto.Tile, scalar, dst: pto.Tile):
    emit_scalar_binary_1d(src, scalar, dst, pto.vadds)


@_test_template(name="test_scalar_2d", loop_depth=2)
def _test_scalar_2d(src: pto.Tile, scalar, dst: pto.Tile):
    emit_scalar_binary_2d(src, scalar, dst, pto.vadds)


@_test_template(name="test_fill_1d", loop_depth=1)
def _test_fill_1d(scalar, dst: pto.Tile):
    emit_scalar_fill_1d(scalar, dst)


@_test_template(name="test_fill_2d", loop_depth=2)
def _test_fill_2d(scalar, dst: pto.Tile):
    emit_scalar_fill_2d(scalar, dst)


_FAMILY_FORMS = (
    (_test_unary_1d, 1, "pto.vabs", ("src", "dst")),
    (_test_unary_2d, 2, "pto.vabs", ("src", "dst")),
    (_test_binary_1d, 1, "pto.vadd", ("src0", "src1", "dst")),
    (_test_binary_2d, 2, "pto.vadd", ("src0", "src1", "dst")),
    (_test_scalar_1d, 1, "pto.vadds", ("src", "scalar", "dst")),
    (_test_scalar_2d, 2, "pto.vadds", ("src", "scalar", "dst")),
    (_test_fill_1d, 1, "pto.vdup", ("scalar", "dst")),
    (_test_fill_2d, 2, "pto.vdup", ("scalar", "dst")),
)


def _foundation_specs(param_names, *, shape=(4, 65), valid_shape=None):
    tile = TileSpec(
        shape=shape,
        valid_shape=valid_shape,
        dtype=ScalarType("f32"),
    )
    scalar = ScalarSpec(dtype=ScalarType("f32"))
    return {
        name: scalar if name == "scalar" else tile
        for name in param_names
    }


def _unary_specs(
    dtype_name,
    *,
    shape=(4, 65),
    valid_shape=None,
    compact_mode="null",
):
    tile = TileSpec(
        shape=shape,
        valid_shape=valid_shape,
        dtype=ScalarType(dtype_name),
        compact_mode=compact_mode,
    )
    return {"src": tile, "dst": tile}


class TileLibElementwiseTest(unittest.TestCase):
    def test_shared_traversals_use_native_python_for_syntax(self):
        tree = ast.parse(
            Path(elementwise.__file__).read_text(encoding="utf-8")
        )
        functions = {
            node.name: node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
        }

        for function_name, expected_loops in (
            ("emit_elementwise_1d", 1),
            ("emit_elementwise_2d", 2),
        ):
            with self.subTest(function=function_name):
                function = functions[function_name]
                self.assertEqual(
                    sum(
                        isinstance(node, ast.For)
                        for node in ast.walk(function)
                    ),
                    expected_loops,
                )
                self.assertFalse(
                    any(
                        isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "for_"
                        for node in ast.walk(function)
                    )
                )

    def test_each_op_selects_and_renders(self):
        for op, (name, vop) in ELEMENTWISE.items():
            with self.subTest(op=op):
                descriptor = select(op, "a5", _f32_specs(), candidate_id=name)
                self.assertEqual(descriptor.name, name)

                mlir = descriptor.specialize(**_f32_specs()).mlir_text()
                self.assertIn(vop, mlir)                 # the op's own vector instruction
                for shared in SHARED_OPS:
                    self.assertIn(shared, mlir)
                self.assertNotIn("pto.castptr", mlir)    # structured, not bare-pointer

    def test_shared_family_forms_expose_expected_loop_depth(self):
        for descriptor, expected_depth, vector_op, params in _FAMILY_FORMS:
            with self.subTest(template=descriptor.name):
                self.assertEqual(descriptor.metadata.loop_depth, expected_depth)
                mlir = descriptor.specialize(
                    **_foundation_specs(params)
                ).mlir_text()
                self.assertEqual(mlir.count("scf.for"), expected_depth)
                self.assertIn(vector_op, mlir)
                self.assertIn("iter_args", mlir)
                self.assertIn("pto.plt_b32", mlir)

    def test_flattened_form_uses_one_total_element_loop_for_aligned_and_tail_shapes(self):
        for shape in ((4, 64), (4, 65), (1, 65)):
            with self.subTest(shape=shape):
                mlir = _test_binary_1d.specialize(
                    **_foundation_specs(
                        ("src0", "src1", "dst"),
                        shape=shape,
                    )
                ).mlir_text()
                self.assertEqual(mlir.count("scf.for"), 1)
                self.assertIn("arith.muli", mlir)
                self.assertIn("iter_args", mlir)
                self.assertNotIn("memref.subview", mlir)

    def test_rowwise_form_retains_row_and_column_loops(self):
        mlir = _test_binary_2d.specialize(
            **_foundation_specs(
                ("src0", "src1", "dst"),
                shape=(4, 65),
                valid_shape=(4, 63),
            )
        ).mlir_text()
        self.assertEqual(mlir.count("scf.for"), 2)
        self.assertNotIn("arith.muli", mlir)
        self.assertIn("memref.subview", mlir)

    def test_registration_helper_selects_1d_only_for_legal_shapes(self):
        contiguous = _foundation_specs(("src0", "src1", "dst"), shape=(4, 65))
        partial = _foundation_specs(
            ("src0", "src1", "dst"),
            shape=(4, 65),
            valid_shape=(4, 63),
        )

        selected_1d = select("test.elementwise.binary", "a5", contiguous)
        selected_2d = select("test.elementwise.binary", "a5", partial)

        self.assertEqual(selected_1d.name, "test_binary_1d")
        self.assertEqual(selected_1d.metadata.loop_depth, 1)
        self.assertEqual(selected_2d.name, "test_binary_2d")
        self.assertEqual(selected_2d.metadata.loop_depth, 2)

    def test_production_unary_ops_register_preferred_1d_and_fallback_2d(self):
        for op, (
            name_1d,
            name_2d,
            vector_op,
            dtype_name,
        ) in PRODUCTION_UNARY_1D.items():
            with self.subTest(op=op):
                specs = _unary_specs(dtype_name)
                candidates = legal_candidates(op, "a5", specs)

                self.assertEqual(
                    [candidate.name for candidate in candidates],
                    [name_1d, name_2d],
                )
                self.assertEqual(
                    [candidate.metadata.id for candidate in candidates],
                    [1, 0],
                )
                self.assertEqual(
                    [candidate.metadata.loop_depth for candidate in candidates],
                    [1, 2],
                )

                mlir = candidates[0].specialize(**specs).mlir_text()
                self.assertEqual(mlir.count("scf.for"), 1)
                self.assertIn(vector_op, mlir)
                self.assertNotIn("memref.subview", mlir)

    def test_production_unary_selection_uses_shared_legality_rule(self):
        shapes = (
            ("contiguous multi-row", (4, 65), None, "null", True),
            ("contiguous single-row", (4, 65), (1, 63), "null", True),
            ("partial multi-row", (4, 65), (4, 63), "null", False),
            ("stride gap", (4, 65), None, 2, False),
        )
        for op, (
            name_1d,
            name_2d,
            _,
            dtype_name,
        ) in PRODUCTION_UNARY_1D.items():
            for label, shape, valid_shape, compact_mode, expect_1d in shapes:
                with self.subTest(op=op, case=label):
                    specs = _unary_specs(
                        dtype_name,
                        shape=shape,
                        valid_shape=valid_shape,
                        compact_mode=compact_mode,
                    )
                    selected = select(op, "a5", specs)
                    self.assertEqual(
                        selected.name,
                        name_1d if expect_1d else name_2d,
                    )
                    self.assertEqual(
                        selected.metadata.loop_depth,
                        1 if expect_1d else 2,
                    )

    def test_tlog_precision_modes_register_distinct_1d_and_2d_candidates(self):
        specs = _unary_specs("f32")
        cases = (
            (
                "default",
                ["template_tlog_1d", "template_tlog"],
                [2, 0],
                ("pto.vln",),
            ),
            (
                "high_precision",
                [
                    "template_tlog_high_precision_1d",
                    "template_tlog_high_precision",
                ],
                [3, 1],
                (
                    "pto.vcmps",
                    "pto.vmuls",
                    "pto.vsel",
                    "pto.vln",
                    "pto.vadds",
                ),
            ),
        )
        for precision_type, expected_names, expected_ids, vector_ops in cases:
            with self.subTest(precision_type=precision_type):
                context_attrs = {"precisionType": precision_type}
                candidates = legal_candidates(
                    "pto.tlog",
                    "a5",
                    specs,
                    context_attrs=context_attrs,
                )

                self.assertEqual(
                    [candidate.name for candidate in candidates],
                    expected_names,
                )
                self.assertEqual(
                    [candidate.metadata.id for candidate in candidates],
                    expected_ids,
                )
                self.assertEqual(
                    [candidate.metadata.loop_depth for candidate in candidates],
                    [1, 2],
                )

                mlir = candidates[0].specialize(
                    context_attrs=context_attrs,
                    **specs,
                ).mlir_text()
                self.assertEqual(mlir.count("scf.for"), 1)
                self.assertNotIn("memref.subview", mlir)
                for vector_op in vector_ops:
                    self.assertIn(vector_op, mlir)

    def test_trecip_registers_local_computation_with_shared_traversals(self):
        specs = _unary_specs("f16")
        candidates = legal_candidates("pto.trecip", "a5", specs)

        self.assertEqual(
            [candidate.name for candidate in candidates],
            ["template_trecip_1d", "template_trecip"],
        )
        self.assertEqual(
            [candidate.metadata.id for candidate in candidates],
            [1, 0],
        )
        self.assertEqual(
            [candidate.metadata.loop_depth for candidate in candidates],
            [1, 2],
        )

        mlir = candidates[0].specialize(**specs).mlir_text()
        self.assertEqual(mlir.count("scf.for"), 1)
        self.assertNotIn("memref.subview", mlir)
        self.assertIn("pto.vbr", mlir)
        self.assertIn("pto.vdiv", mlir)

    def test_specialized_unary_selection_uses_shared_legality_rule(self):
        shapes = (
            ("contiguous multi-row", (4, 65), None, "null", True),
            ("contiguous single-row", (4, 65), (1, 63), "null", True),
            ("partial multi-row", (4, 65), (4, 63), "null", False),
            ("stride gap", (4, 65), None, 2, False),
        )
        operations = (
            (
                "pto.tlog",
                "template_tlog_high_precision_1d",
                "template_tlog_high_precision",
                {"precisionType": "high_precision"},
            ),
            (
                "pto.trecip",
                "template_trecip_1d",
                "template_trecip",
                None,
            ),
        )
        for op, name_1d, name_2d, context_attrs in operations:
            for label, shape, valid_shape, compact_mode, expect_1d in shapes:
                with self.subTest(op=op, case=label):
                    specs = _unary_specs(
                        "f32",
                        shape=shape,
                        valid_shape=valid_shape,
                        compact_mode=compact_mode,
                    )
                    selected = select(
                        op,
                        "a5",
                        specs,
                        context_attrs=context_attrs,
                    )
                    self.assertEqual(
                        selected.name,
                        name_1d if expect_1d else name_2d,
                    )
                    self.assertEqual(
                        selected.metadata.loop_depth,
                        1 if expect_1d else 2,
                    )


if __name__ == "__main__":
    unittest.main()
