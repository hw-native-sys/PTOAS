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
from ptodsl.tilelib import ScalarSpec, ScalarType, TileSpec, select
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


if __name__ == "__main__":
    unittest.main()
