# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Element-wise template selection, traversal, and scope acceptance tests."""

import ast
import unittest
from pathlib import Path

from ptodsl import pto
import ptodsl.tilelib as tilelib
import ptodsl.tilelib.templates.a5._elementwise as elementwise
import ptodsl.tilelib.templates.a5.tsel as tsel_templates
import ptodsl.tilelib.templates.a5.tsels as tsels_templates
from ptodsl.tilelib import (
    ScalarSpec,
    ScalarType,
    TileSpec,
    legal_candidates,
    select,
)
from ptodsl.tilelib.templates import load_template
from ptodsl.tilelib.templates.a5._elementwise import (
    emit_scalar_binary_1d,
    emit_scalar_binary_2d,
    emit_scalar_fill_1d,
    emit_scalar_fill_2d,
    emit_unary_1d,
    emit_unary_2d,
    register_binary,
)


ELEMENTWISE_SCOPE_BY_FAMILY = {
    "unary": frozenset(
        {
            "pto.tabs",
            "pto.texp",
            "pto.tlog",
            "pto.tneg",
            "pto.tnot",
            "pto.trecip",
            "pto.trelu",
            "pto.trsqrt",
            "pto.tsqrt",
        }
    ),
    "tile_tile": frozenset(
        {
            "pto.tadd",
            "pto.tand",
            "pto.tdiv",
            "pto.tfmod",
            "pto.tmax",
            "pto.tmin",
            "pto.tmul",
            "pto.tor",
            "pto.tprelu",
            "pto.trem",
            "pto.tshl",
            "pto.tshr",
            "pto.tsub",
            "pto.txor",
        }
    ),
    "tile_scalar": frozenset(
        {
            "pto.tadds",
            "pto.tands",
            "pto.tdivs",
            "pto.tfmods",
            "pto.tlrelu",
            "pto.tmaxs",
            "pto.tmins",
            "pto.tmuls",
            "pto.tors",
            "pto.trems",
            "pto.tshls",
            "pto.tshrs",
            "pto.tsubs",
            "pto.txors",
        }
    ),
    "compare": frozenset({"pto.tcmp", "pto.tcmps"}),
    "select": frozenset({"pto.tsel", "pto.tsels"}),
    "conversion": frozenset({"pto.tcvt"}),
    "scalar_fill": frozenset({"pto.texpands"}),
}

ELEMENTWISE_SCOPE = frozenset().union(
    *ELEMENTWISE_SCOPE_BY_FAMILY.values()
)

# Map an operation to its reviewed reason if no flattened form is legal.
# Every scoped operation currently has at least one legal 1D candidate.
ELEMENTWISE_1D_EXCEPTIONS = {}

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

PRODUCTION_BINARY_1D = {
    "pto.tadd": ("template_tadd_1d", "template_tadd", "pto.vadd", "f32"),
    "pto.tand": ("template_tand_1d", "template_tand", "pto.vand", "i8"),
    "pto.tmax": ("template_tmax_1d", "template_tmax", "pto.vmax", "f16"),
    "pto.tmin": ("template_tmin_1d", "template_tmin", "pto.vmin", "i16"),
    "pto.tmul": ("template_tmul_1d", "template_tmul", "pto.vmul", "bf16"),
    "pto.tor": ("template_tor_1d", "template_tor", "pto.vor", "ui8"),
    "pto.tshl": ("template_tshl_1d", "template_tshl", "pto.vshl", "ui16"),
    "pto.tshr": ("template_tshr_1d", "template_tshr", "pto.vshr", "ui32"),
    "pto.tsub": ("template_tsub_1d", "template_tsub", "pto.vsub", "i32"),
}

PRODUCTION_TEMP_BINARY_1D = {
    "pto.tprelu": (
        "template_tprelu_1d",
        "template_tprelu",
        "pto.vprelu",
        "f32",
    ),
    "pto.trem": (
        "template_trem_1d",
        "template_trem",
        "pto.vtrc",
        "f32",
    ),
    "pto.txor": (
        "template_txor_1d",
        "template_txor",
        "pto.vxor",
        "i16",
    ),
}

PRODUCTION_SCALAR_1D = {
    "pto.tadds": (
        "template_tadds_1d",
        "template_tadds",
        "pto.vadds",
        "f32",
        "f32",
    ),
    "pto.tands": (
        "template_tands_1d",
        "template_tands",
        "pto.vand",
        "i8",
        "i8",
    ),
    "pto.tmaxs": (
        "template_tmaxs_1d",
        "template_tmaxs",
        "pto.vmaxs",
        "f16",
        "f16",
    ),
    "pto.tmins": (
        "template_tmins_1d",
        "template_tmins",
        "pto.vmins",
        "ui16",
        "ui16",
    ),
    "pto.tmuls": (
        "template_tmuls_1d",
        "template_tmuls",
        "pto.vmuls",
        "f32",
        "f32",
    ),
    "pto.tors": (
        "template_tors_1d",
        "template_tors",
        "pto.vor",
        "ui8",
        "ui8",
    ),
    "pto.tshls": (
        "template_tshls_1d",
        "template_tshls",
        "pto.vshls",
        "ui16",
        "i16",
    ),
    "pto.tshrs": (
        "template_tshrs_1d",
        "template_tshrs",
        "pto.vshrs",
        "ui32",
        "i16",
    ),
    "pto.tsubs": (
        "template_tsubs_1d",
        "template_tsubs",
        "pto.vsub",
        "i32",
        "i32",
    ),
}

PRODUCTION_SPECIAL_SCALAR_1D = {
    "pto.tfmods": (
        "template_tfmods_1d",
        "template_tfmods",
        "pto.vtrc",
        "f32",
        "f32",
        "scalar",
    ),
    "pto.tlrelu": (
        "template_tlrelu_1d",
        "template_tlrelu",
        "pto.vlrelu",
        "f16",
        "f32",
        "slope",
    ),
}

PRODUCTION_TEMP_SCALAR_1D = {
    "pto.trems": (
        "template_trems_1d",
        "template_trems",
        "pto.vtrc",
        "f32",
    ),
    "pto.txors": (
        "template_txors_1d",
        "template_txors",
        "pto.vxor",
        "i16",
    ),
}

PRODUCTION_FILL_DTYPES = ("i8", "i16", "i32", "f16", "bf16", "f32")

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


def _binary_specs(
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
    return {"src0": tile, "src1": tile, "dst": tile}


def _binary_tmp_specs(
    data_dtype,
    *,
    tmp_dtype=None,
    shape=(4, 65),
    valid_shape=None,
    compact_mode="null",
    tmp_compact_mode=None,
):
    data = TileSpec(
        shape=shape,
        valid_shape=valid_shape,
        dtype=ScalarType(data_dtype),
        compact_mode=compact_mode,
    )
    tmp = TileSpec(
        shape=shape,
        valid_shape=valid_shape,
        dtype=ScalarType(tmp_dtype or data_dtype),
        compact_mode=(
            compact_mode
            if tmp_compact_mode is None
            else tmp_compact_mode
        ),
    )
    return {"src0": data, "src1": data, "tmp": tmp, "dst": data}


def _scalar_specs(
    data_dtype,
    *,
    scalar_dtype=None,
    shape=(4, 65),
    valid_shape=None,
    compact_mode="null",
):
    tile = TileSpec(
        shape=shape,
        valid_shape=valid_shape,
        dtype=ScalarType(data_dtype),
        compact_mode=compact_mode,
    )
    scalar = ScalarSpec(
        dtype=ScalarType(scalar_dtype or data_dtype),
        value=1,
    )
    return {"src": tile, "scalar": scalar, "dst": tile}


def _named_scalar_specs(
    data_dtype,
    *,
    scalar_dtype=None,
    scalar_name="scalar",
    shape=(4, 65),
    valid_shape=None,
    compact_mode="null",
):
    specs = _scalar_specs(
        data_dtype,
        scalar_dtype=scalar_dtype,
        shape=shape,
        valid_shape=valid_shape,
        compact_mode=compact_mode,
    )
    specs[scalar_name] = specs.pop("scalar")
    return specs


def _scalar_tmp_specs(
    data_dtype,
    *,
    scalar_dtype=None,
    shape=(4, 65),
    valid_shape=None,
    compact_mode="null",
    tmp_compact_mode=None,
):
    specs = _scalar_specs(
        data_dtype,
        scalar_dtype=scalar_dtype,
        shape=shape,
        valid_shape=valid_shape,
        compact_mode=compact_mode,
    )
    specs["tmp"] = TileSpec(
        shape=shape,
        valid_shape=valid_shape,
        dtype=ScalarType(data_dtype),
        compact_mode=(
            compact_mode
            if tmp_compact_mode is None
            else tmp_compact_mode
        ),
    )
    return {
        "src": specs["src"],
        "scalar": specs["scalar"],
        "tmp": specs["tmp"],
        "dst": specs["dst"],
    }


def _fill_specs(
    dtype_name,
    *,
    shape=(4, 65),
    valid_shape=None,
    compact_mode="null",
):
    return {
        "scalar": ScalarSpec(
            dtype=ScalarType(dtype_name),
            value=1,
        ),
        "dst": TileSpec(
            shape=shape,
            valid_shape=valid_shape,
            dtype=ScalarType(dtype_name),
            compact_mode=compact_mode,
        ),
    }


def _compare_specs(
    op,
    dtype_name,
    *,
    src_shape=(1, 256),
    src_valid_shape=None,
    dst_shape=None,
    dst_valid_shape=None,
    src_compact_mode="null",
    dst_compact_mode="null",
):
    src = TileSpec(
        shape=src_shape,
        valid_shape=src_valid_shape,
        dtype=ScalarType(dtype_name),
        memory_space="vec",
        compact_mode=src_compact_mode,
    )
    predicate_dtype = "i8" if op == "pto.tcmp" else "ui8"
    if dst_shape is None:
        dst_shape = src_shape if op == "pto.tcmp" else (src_shape[0], 32)
    dst = TileSpec(
        shape=dst_shape,
        valid_shape=dst_valid_shape,
        dtype=ScalarType(predicate_dtype),
        memory_space="vec",
        compact_mode=dst_compact_mode,
    )
    if op == "pto.tcmp":
        return {"src0": src, "src1": src, "dst": dst}
    return {
        "src": src,
        "scalar": ScalarSpec(dtype=ScalarType(dtype_name), value=1),
        "dst": dst,
    }


def _select_specs(
    op,
    data_dtype,
    *,
    mask_dtype="i8",
    data_shape=(1, 256),
    data_valid_shape=None,
    mask_shape=None,
    mask_valid_shape=None,
    tmp_shape=(1, 64),
    data_compact_mode="null",
    mask_compact_mode="null",
    tmp_compact_mode="null",
):
    bytewidth = {"i8": 1, "i16": 2, "i32": 4}[mask_dtype]
    if mask_shape is None:
        mask_shape = (data_shape[0], 32 // bytewidth)
    mask = TileSpec(
        shape=mask_shape,
        valid_shape=mask_valid_shape,
        dtype=ScalarType(mask_dtype),
        compact_mode=mask_compact_mode,
    )
    data = TileSpec(
        shape=data_shape,
        valid_shape=data_valid_shape,
        dtype=ScalarType(data_dtype),
        compact_mode=data_compact_mode,
    )
    tmp = TileSpec(
        shape=tmp_shape,
        dtype=ScalarType(data_dtype),
        compact_mode=tmp_compact_mode,
    )
    if op == "pto.tsel":
        return {
            "mask": mask,
            "src0": data,
            "src1": data,
            "tmp": tmp,
            "dst": data,
        }
    return {
        "mask": mask,
        "src": data,
        "tmp": tmp,
        "scalar": ScalarSpec(dtype=ScalarType(data_dtype), value=1),
        "dst": data,
    }


class TileLibElementwiseTest(unittest.TestCase):
    def test_issue_scope_catalog_has_ranked_1d_and_2d_coverage(self):
        family_sizes = sum(
            len(operations)
            for operations in ELEMENTWISE_SCOPE_BY_FAMILY.values()
        )
        self.assertEqual(family_sizes, 43)
        self.assertEqual(len(ELEMENTWISE_SCOPE), family_sizes)
        self.assertLessEqual(
            set(ELEMENTWISE_1D_EXCEPTIONS),
            ELEMENTWISE_SCOPE,
        )

        for op in sorted(ELEMENTWISE_SCOPE):
            with self.subTest(op=op):
                self.assertTrue(load_template(op, "a5"))
                candidates = tilelib.default_registry().lookup(op, "a5")
                self.assertTrue(candidates)

                candidate_ids = [
                    candidate.metadata.id for candidate in candidates
                ]
                self.assertEqual(
                    len(candidate_ids),
                    len(set(candidate_ids)),
                    "candidate IDs must remain unique within an operation",
                )

                by_loop_depth = {
                    loop_depth: [
                        candidate
                        for candidate in candidates
                        if candidate.metadata.loop_depth == loop_depth
                    ]
                    for loop_depth in (1, 2)
                }
                self.assertTrue(
                    by_loop_depth[2],
                    "every scoped operation needs a general 2D fallback",
                )

                exception = ELEMENTWISE_1D_EXCEPTIONS.get(op)
                if exception is not None:
                    self.assertTrue(exception.strip())
                    self.assertFalse(by_loop_depth[1])
                    continue

                self.assertTrue(
                    by_loop_depth[1],
                    "operation needs a 1D candidate or a documented exception",
                )
                self.assertGreater(
                    min(
                        candidate.metadata.priority
                        for candidate in by_loop_depth[1]
                    ),
                    max(
                        candidate.metadata.priority
                        for candidate in by_loop_depth[2]
                    ),
                    "every legal 1D form must rank ahead of every 2D fallback",
                )

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
        # Tile slicing may lower either to memref.subview or to explicit
        # row-stride pointer arithmetic. Both retain the same 2D traversal.
        self.assertEqual(mlir.count("scf.for"), 2)
        self.assertIn("iter_args", mlir)

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

    def test_production_binary_ops_register_preferred_1d_and_fallback_2d(self):
        for op, (
            name_1d,
            name_2d,
            vector_op,
            dtype_name,
        ) in PRODUCTION_BINARY_1D.items():
            with self.subTest(op=op):
                specs = _binary_specs(dtype_name)
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
                self.assertEqual(
                    candidates[0].metadata.dtypes,
                    candidates[1].metadata.dtypes,
                )

                mlir = candidates[0].specialize(**specs).mlir_text()
                self.assertEqual(mlir.count("scf.for"), 1)
                self.assertIn(vector_op, mlir)
                self.assertNotIn("memref.subview", mlir)

    def test_production_binary_selection_uses_shared_legality_rule(self):
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
        ) in PRODUCTION_BINARY_1D.items():
            for label, shape, valid_shape, compact_mode, expect_1d in shapes:
                with self.subTest(op=op, case=label):
                    specs = _binary_specs(
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

    def test_tdiv_precision_modes_share_ranked_traversal_candidates(self):
        specs = _binary_specs("f32")
        cases = (
            ("default", ("pto.vdiv",)),
            (
                "high_precision",
                (
                    "pto.vdiv",
                    "pto.vbitcast",
                    "pto.vcmp",
                    "pto.vsel",
                ),
            ),
        )
        for precision_type, vector_ops in cases:
            with self.subTest(precision_type=precision_type):
                context_attrs = {"precisionType": precision_type}
                candidates = legal_candidates(
                    "pto.tdiv",
                    "a5",
                    specs,
                    context_attrs=context_attrs,
                )

                self.assertEqual(
                    [candidate.name for candidate in candidates],
                    ["template_tdiv_1d", "template_tdiv"],
                )
                self.assertEqual(
                    [candidate.metadata.id for candidate in candidates],
                    [1, 0],
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

    def test_tfmod_1d_preserves_dtype_specific_remainder_computation(self):
        cases = (
            ("f32", True),
            ("f16", True),
            ("i16", False),
            ("ui16", False),
        )
        for dtype_name, expects_truncation in cases:
            with self.subTest(dtype=dtype_name):
                specs = _binary_specs(dtype_name)
                candidates = legal_candidates("pto.tfmod", "a5", specs)

                self.assertEqual(
                    [candidate.name for candidate in candidates],
                    ["template_tfmod_1d", "template_tfmod"],
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
                for vector_op in ("pto.vdiv", "pto.vmul", "pto.vsub"):
                    self.assertIn(vector_op, mlir)
                if expects_truncation:
                    self.assertIn("pto.vtrc", mlir)
                else:
                    self.assertNotIn("pto.vtrc", mlir)

    def test_specialized_binary_selection_uses_shared_legality_rule(self):
        shapes = (
            ("contiguous multi-row", (4, 65), None, "null", True),
            ("contiguous single-row", (4, 65), (1, 63), "null", True),
            ("partial multi-row", (4, 65), (4, 63), "null", False),
            ("stride gap", (4, 65), None, 2, False),
        )
        operations = (
            ("pto.tdiv", "template_tdiv_1d", "template_tdiv"),
            ("pto.tfmod", "template_tfmod_1d", "template_tfmod"),
        )
        for op, name_1d, name_2d in operations:
            for label, shape, valid_shape, compact_mode, expect_1d in shapes:
                with self.subTest(op=op, case=label):
                    specs = _binary_specs(
                        "f32",
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

    def test_temporary_binary_ops_register_preferred_1d_and_fallback_2d(self):
        for op, (
            name_1d,
            name_2d,
            vector_op,
            dtype_name,
        ) in PRODUCTION_TEMP_BINARY_1D.items():
            with self.subTest(op=op):
                specs = _binary_tmp_specs(dtype_name)
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
                self.assertEqual(
                    candidates[0].metadata.dtypes,
                    candidates[1].metadata.dtypes,
                )

                mlir = candidates[0].specialize(**specs).mlir_text()
                self.assertEqual(mlir.count("scf.for"), 1)
                self.assertIn(vector_op, mlir)
                self.assertNotIn("memref.subview", mlir)

    def test_temporary_binary_selection_uses_shared_legality_rule(self):
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
        ) in PRODUCTION_TEMP_BINARY_1D.items():
            for label, shape, valid_shape, compact_mode, expect_1d in shapes:
                with self.subTest(op=op, case=label):
                    specs = _binary_tmp_specs(
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

    def test_temporary_tile_alone_can_disqualify_1d(self):
        for op, (
            _,
            name_2d,
            _,
            dtype_name,
        ) in PRODUCTION_TEMP_BINARY_1D.items():
            with self.subTest(op=op):
                specs = _binary_tmp_specs(
                    dtype_name,
                    tmp_compact_mode=2,
                )
                candidates = legal_candidates(op, "a5", specs)

                self.assertEqual(
                    [candidate.name for candidate in candidates],
                    [name_2d],
                )
                self.assertEqual(candidates[0].metadata.loop_depth, 2)

    def test_tprelu_1d_accepts_i8_temporary_representation(self):
        specs = _binary_tmp_specs("f32", tmp_dtype="i8")
        candidates = legal_candidates("pto.tprelu", "a5", specs)

        self.assertEqual(
            [candidate.name for candidate in candidates],
            ["template_tprelu_1d", "template_tprelu"],
        )
        mlir = candidates[0].specialize(**specs).mlir_text()
        self.assertEqual(mlir.count("scf.for"), 1)
        self.assertIn("pto.vprelu", mlir)

    def test_trem_1d_preserves_floor_remainder_computation(self):
        specs = _binary_tmp_specs("f32")
        descriptor = select("pto.trem", "a5", specs)
        mlir = descriptor.specialize(**specs).mlir_text()

        self.assertEqual(descriptor.name, "template_trem_1d")
        self.assertEqual(mlir.count("scf.for"), 1)
        for vector_op in (
            "pto.vdiv",
            "pto.vtrc",
            "pto.vmul",
            "pto.vsub",
        ):
            self.assertIn(vector_op, mlir)

    def test_production_scalar_ops_register_preferred_1d_and_fallback_2d(self):
        for op, (
            name_1d,
            name_2d,
            vector_op,
            data_dtype,
            scalar_dtype,
        ) in PRODUCTION_SCALAR_1D.items():
            with self.subTest(op=op):
                specs = _scalar_specs(
                    data_dtype,
                    scalar_dtype=scalar_dtype,
                )
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
                self.assertEqual(
                    candidates[0].metadata.dtypes,
                    candidates[1].metadata.dtypes,
                )

                for shape in ((4, 64), (4, 65)):
                    with self.subTest(op=op, shape=shape):
                        render_specs = _scalar_specs(
                            data_dtype,
                            scalar_dtype=scalar_dtype,
                            shape=shape,
                        )
                        mlir = candidates[0].specialize(
                            **render_specs
                        ).mlir_text()
                        self.assertEqual(mlir.count("scf.for"), 1)
                        self.assertIn(vector_op, mlir)
                        self.assertNotIn("memref.subview", mlir)

    def test_production_scalar_selection_uses_shared_legality_rule(self):
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
            data_dtype,
            scalar_dtype,
        ) in PRODUCTION_SCALAR_1D.items():
            for label, shape, valid_shape, compact_mode, expect_1d in shapes:
                with self.subTest(op=op, case=label):
                    specs = _scalar_specs(
                        data_dtype,
                        scalar_dtype=scalar_dtype,
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

    def test_tsubs_shared_traversal_preserves_broadcast_subtraction(self):
        specs = _scalar_specs("f32")
        candidates = legal_candidates("pto.tsubs", "a5", specs)

        for candidate, expected_loops in zip(candidates, (1, 2)):
            with self.subTest(candidate=candidate.name):
                mlir = candidate.specialize(**specs).mlir_text()
                self.assertEqual(mlir.count("scf.for"), expected_loops)
                self.assertIn("pto.vbr", mlir)
                self.assertIn("pto.vsub", mlir)

    def test_tdivs_operand_forms_preserve_candidate_ids(self):
        cases = (
            (
                _scalar_specs("f32"),
                (
                    ("template_tdivs_tile_scalar_1d", 2, 1),
                    ("template_tdivs_tile_scalar", 0, 2),
                ),
            ),
            (
                {
                    "scalar": ScalarSpec(
                        dtype=ScalarType("f32"),
                        value=1,
                    ),
                    "src": TileSpec(
                        shape=(4, 65),
                        dtype=ScalarType("f32"),
                    ),
                    "dst": TileSpec(
                        shape=(4, 65),
                        dtype=ScalarType("f32"),
                    ),
                },
                (
                    ("template_tdivs_scalar_tile_1d", 3, 1),
                    ("template_tdivs_scalar_tile", 1, 2),
                ),
            ),
        )
        for specs, candidates in cases:
            with self.subTest(operand_order=tuple(specs)):
                for name, candidate_id, loop_depth in candidates:
                    descriptor = select(
                        "pto.tdivs",
                        "a5",
                        specs,
                        candidate_id=name,
                    )
                    self.assertEqual(descriptor.metadata.id, candidate_id)
                    self.assertEqual(
                        descriptor.metadata.loop_depth,
                        loop_depth,
                    )

    def test_tdivs_1d_preserves_precision_and_operand_order(self):
        cases = (
            (
                _scalar_specs("f32"),
                "template_tdivs_tile_scalar_1d",
            ),
            (
                {
                    "scalar": ScalarSpec(
                        dtype=ScalarType("f32"),
                        value=1,
                    ),
                    "src": TileSpec(
                        shape=(4, 65),
                        dtype=ScalarType("f32"),
                    ),
                    "dst": TileSpec(
                        shape=(4, 65),
                        dtype=ScalarType("f32"),
                    ),
                },
                "template_tdivs_scalar_tile_1d",
            ),
        )
        for specs, expected_name in cases:
            for precision_type in ("default", "high_precision"):
                with self.subTest(
                    operand_order=tuple(specs),
                    precision_type=precision_type,
                ):
                    context_attrs = {"precisionType": precision_type}
                    descriptor = select(
                        "pto.tdivs",
                        "a5",
                        specs,
                        context_attrs=context_attrs,
                        candidate_id=expected_name,
                    )
                    mlir = descriptor.specialize(
                        context_attrs=context_attrs,
                        **specs,
                    ).mlir_text()

                    self.assertEqual(descriptor.name, expected_name)
                    self.assertEqual(mlir.count("scf.for"), 1)
                    self.assertIn("pto.vbr", mlir)
                    self.assertIn("pto.vdiv", mlir)
                    if precision_type == "high_precision":
                        self.assertIn("pto.vbitcast", mlir)
                        self.assertIn("pto.vcmp", mlir)

    def test_tdivs_ineligible_ranges_reject_1d_candidates(self):
        specs = _scalar_specs(
            "f32",
            shape=(4, 65),
            valid_shape=(4, 63),
        )
        candidates = legal_candidates("pto.tdivs", "a5", specs)

        self.assertNotIn(
            "template_tdivs_tile_scalar_1d",
            [candidate.name for candidate in candidates],
        )
        self.assertNotIn(
            "template_tdivs_scalar_tile_1d",
            [candidate.name for candidate in candidates],
        )
        self.assertTrue(
            all(
                candidate.metadata.loop_depth == 2
                for candidate in candidates
            )
        )

    def test_specialized_scalar_ops_use_shared_legality_rule(self):
        shapes = (
            ("contiguous multi-row", (4, 65), None, "null", True),
            ("contiguous single-row", (4, 65), (1, 63), "null", True),
            ("partial multi-row", (4, 65), (4, 63), "null", False),
            ("stride gap", (4, 65), None, 2, False),
        )
        for op, (
            name_1d,
            name_2d,
            vector_op,
            data_dtype,
            scalar_dtype,
            scalar_name,
        ) in PRODUCTION_SPECIAL_SCALAR_1D.items():
            for label, shape, valid_shape, compact_mode, expect_1d in shapes:
                with self.subTest(op=op, case=label):
                    specs = _named_scalar_specs(
                        data_dtype,
                        scalar_dtype=scalar_dtype,
                        scalar_name=scalar_name,
                        shape=shape,
                        valid_shape=valid_shape,
                        compact_mode=compact_mode,
                    )
                    candidates = legal_candidates(op, "a5", specs)
                    self.assertEqual(
                        candidates[0].name,
                        name_1d if expect_1d else name_2d,
                    )
                    self.assertEqual(
                        candidates[0].metadata.loop_depth,
                        1 if expect_1d else 2,
                    )

                    mlir = candidates[0].specialize(**specs).mlir_text()
                    self.assertIn(vector_op, mlir)

    def test_temporary_scalar_ops_include_tmp_in_1d_legality(self):
        for op, (
            name_1d,
            name_2d,
            vector_op,
            data_dtype,
        ) in PRODUCTION_TEMP_SCALAR_1D.items():
            with self.subTest(op=op, case="contiguous"):
                specs = _scalar_tmp_specs(data_dtype)
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

            with self.subTest(op=op, case="temporary stride gap"):
                specs = _scalar_tmp_specs(
                    data_dtype,
                    tmp_compact_mode=2,
                )
                candidates = legal_candidates(op, "a5", specs)
                self.assertEqual(
                    [candidate.name for candidate in candidates],
                    [name_2d],
                )
                self.assertEqual(candidates[0].metadata.loop_depth, 2)

    def test_scalar_remainder_1d_preserves_instruction_sequences(self):
        for op, specs in (
            ("pto.tfmods", _scalar_specs("f32")),
            ("pto.trems", _scalar_tmp_specs("f32")),
        ):
            with self.subTest(op=op):
                descriptor = select(op, "a5", specs)
                mlir = descriptor.specialize(**specs).mlir_text()

                self.assertEqual(descriptor.metadata.loop_depth, 1)
                for vector_op in (
                    "pto.vbr",
                    "pto.vdiv",
                    "pto.vtrc",
                    "pto.vmuls",
                    "pto.vsub",
                ):
                    self.assertIn(vector_op, mlir)

    def test_texpands_registers_preferred_1d_and_fallback_2d(self):
        for dtype_name in PRODUCTION_FILL_DTYPES:
            with self.subTest(dtype=dtype_name):
                specs = _fill_specs(dtype_name)
                candidates = legal_candidates("pto.texpands", "a5", specs)

                self.assertEqual(
                    [candidate.name for candidate in candidates],
                    ["template_texpands_1d", "template_texpands"],
                )
                self.assertEqual(
                    [candidate.metadata.id for candidate in candidates],
                    [1, 0],
                )
                self.assertEqual(
                    [candidate.metadata.loop_depth for candidate in candidates],
                    [1, 2],
                )
                self.assertEqual(
                    candidates[0].metadata.dtypes,
                    candidates[1].metadata.dtypes,
                )

                for shape in ((4, 64), (4, 65)):
                    with self.subTest(dtype=dtype_name, shape=shape):
                        mlir = candidates[0].specialize(
                            **_fill_specs(dtype_name, shape=shape)
                        ).mlir_text()
                        self.assertEqual(mlir.count("scf.for"), 1)
                        self.assertIn("pto.vdup", mlir)
                        self.assertIn("pto.vsts", mlir)
                        self.assertNotIn("pto.vlds", mlir)
                        self.assertNotIn("memref.subview", mlir)

    def test_texpands_selection_uses_destination_legality(self):
        cases = (
            ("contiguous multi-row", (4, 65), None, "null", True),
            ("contiguous single-row", (4, 65), (1, 63), "null", True),
            ("partial multi-row", (4, 65), (4, 63), "null", False),
            ("stride gap", (4, 65), None, 2, False),
        )
        for label, shape, valid_shape, compact_mode, expect_1d in cases:
            with self.subTest(case=label):
                specs = _fill_specs(
                    "f32",
                    shape=shape,
                    valid_shape=valid_shape,
                    compact_mode=compact_mode,
                )
                selected = select("pto.texpands", "a5", specs)

                self.assertEqual(
                    selected.name,
                    "template_texpands_1d"
                    if expect_1d
                    else "template_texpands",
                )
                self.assertEqual(
                    selected.metadata.loop_depth,
                    1 if expect_1d else 2,
                )

    def test_compare_family_registers_preferred_1d_and_fallback_2d(self):
        cases = (
            ("pto.tcmp", "template_tcmp_1d", "template_tcmp", "pto.vcmp"),
            (
                "pto.tcmps",
                "template_tcmps_1d",
                "template_tcmps",
                "pto.vcmps",
            ),
        )
        for op, name_1d, name_2d, vector_op in cases:
            for dtype_name, expected_dist in (
                ("f32", "PK"),
                ("f16", "PK"),
                ("i8", "NORM"),
            ):
                with self.subTest(op=op, dtype=dtype_name):
                    specs = _compare_specs(op, dtype_name)
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

                    mlir = candidates[0].specialize(
                        context_attrs={"cmp_mode": "lt"},
                        **specs,
                    ).mlir_text()
                    self.assertEqual(mlir.count("scf.for"), 1)
                    self.assertIn(vector_op, mlir)
                    self.assertIn(expected_dist, mlir)
                    self.assertIn('"lt"', mlir)

    def test_compare_family_uses_predicate_specific_fallbacks(self):
        cases = (
            (
                "tcmp multi-row predicate padding",
                "pto.tcmp",
                _compare_specs(
                    "pto.tcmp",
                    "f32",
                    src_shape=(4, 128),
                ),
                "template_tcmp",
            ),
            (
                "tcmps source row tail",
                "pto.tcmps",
                _compare_specs(
                    "pto.tcmps",
                    "i8",
                    src_shape=(4, 128),
                    dst_shape=(4, 32),
                ),
                "template_tcmps",
            ),
            (
                "tcmps predicate row padding",
                "pto.tcmps",
                _compare_specs(
                    "pto.tcmps",
                    "f32",
                    src_shape=(4, 256),
                    dst_shape=(4, 64),
                ),
                "template_tcmps",
            ),
            (
                "tcmps stride gap",
                "pto.tcmps",
                _compare_specs(
                    "pto.tcmps",
                    "f32",
                    src_shape=(4, 256),
                    dst_shape=(4, 32),
                    dst_compact_mode=2,
                ),
                "template_tcmps",
            ),
        )
        for label, op, specs, fallback_name in cases:
            with self.subTest(case=label):
                candidates = legal_candidates(op, "a5", specs)
                self.assertEqual(
                    [candidate.name for candidate in candidates],
                    [fallback_name],
                )
                self.assertEqual(candidates[0].metadata.loop_depth, 2)

    def test_tcmps_aligned_dense_multi_row_selects_flattened_body(self):
        specs = _compare_specs(
            "pto.tcmps",
            "f32",
            src_shape=(4, 256),
            dst_shape=(4, 32),
        )
        selected = select("pto.tcmps", "a5", specs)
        mlir = selected.specialize(**specs).mlir_text()

        self.assertEqual(selected.name, "template_tcmps_1d")
        self.assertEqual(selected.metadata.loop_depth, 1)
        self.assertEqual(mlir.count("scf.for"), 1)

    def test_tcmps_f32_fallback_is_genuinely_row_wise(self):
        specs = _compare_specs(
            "pto.tcmps",
            "f32",
            src_shape=(4, 64),
            dst_shape=(4, 32),
        )
        selected = select("pto.tcmps", "a5", specs)
        mlir = selected.specialize(**specs).mlir_text()

        self.assertEqual(selected.name, "template_tcmps")
        self.assertEqual(selected.metadata.loop_depth, 2)
        self.assertEqual(mlir.count("scf.for"), 2)

    def test_tcmps_fallback_keeps_flat_packed_predicate_rows(self):
        specs = _compare_specs(
            "pto.tcmps",
            "f16",
            src_shape=(4, 256),
            dst_shape=(4, 64),
            dst_valid_shape=(4, 32),
        )
        selected = select("pto.tcmps", "a5", specs)
        mlir = selected.specialize(**specs).mlir_text()

        self.assertEqual(selected.name, "template_tcmps")
        self.assertEqual(selected.metadata.loop_depth, 2)
        self.assertIn("arith.constant 16 : index", mlir)
        self.assertNotIn("arith.constant 64 : index", mlir)

    def test_select_family_registers_preferred_1d_and_fallback_2d(self):
        cases = (
            ("pto.tsel", "f32", "i8", "US"),
            ("pto.tsel", "f16", "i8", "US"),
            ("pto.tsel", "i8", "i8", "NORM"),
            ("pto.tsels", "f32", "i32", "US"),
            ("pto.tsels", "f16", "i16", "US"),
            ("pto.tsels", "i8", "i8", "NORM"),
        )
        for op, data_dtype, mask_dtype, expected_dist in cases:
            with self.subTest(
                op=op,
                data_dtype=data_dtype,
                mask_dtype=mask_dtype,
            ):
                specs = _select_specs(
                    op,
                    data_dtype,
                    mask_dtype=mask_dtype,
                )
                candidates = legal_candidates(op, "a5", specs)
                base_name = "template_tsel" if op == "pto.tsel" else "template_tsels"
                self.assertEqual(
                    [candidate.name for candidate in candidates],
                    [f"{base_name}_1d", base_name],
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
                self.assertIn("pto.plds", mlir)
                self.assertIn("pto.vsel", mlir)
                self.assertIn(expected_dist, mlir)
                if op == "pto.tsels":
                    self.assertIn("pto.vdup", mlir)

    def test_select_family_uses_predicate_specific_fallbacks(self):
        cases = (
            (
                "partial data row",
                _select_specs(
                    "pto.tsel",
                    "f32",
                    data_shape=(4, 256),
                    data_valid_shape=(4, 255),
                ),
            ),
            (
                "predicate row padding",
                _select_specs(
                    "pto.tsels",
                    "f32",
                    data_shape=(4, 256),
                    mask_shape=(4, 64),
                ),
            ),
            (
                "predicate stride gap",
                _select_specs(
                    "pto.tsels",
                    "f32",
                    data_shape=(4, 256),
                    mask_compact_mode=2,
                ),
            ),
            (
                "temporary stride gap",
                _select_specs(
                    "pto.tsel",
                    "f32",
                    data_shape=(4, 256),
                    tmp_compact_mode=2,
                ),
            ),
        )
        for label, specs in cases:
            with self.subTest(case=label):
                op = "pto.tsels" if "scalar" in specs else "pto.tsel"
                candidates = legal_candidates(op, "a5", specs)
                fallback = (
                    "template_tsels" if op == "pto.tsels" else "template_tsel"
                )
                self.assertEqual(
                    [candidate.name for candidate in candidates],
                    [fallback],
                )
                self.assertEqual(candidates[0].metadata.loop_depth, 2)

    def test_select_family_aligned_multi_row_selects_flattened_body(self):
        for op in ("pto.tsel", "pto.tsels"):
            with self.subTest(op=op):
                specs = _select_specs(
                    op,
                    "f32",
                    data_shape=(4, 256),
                )
                selected = select(op, "a5", specs)
                mlir = selected.specialize(**specs).mlir_text()

                self.assertEqual(selected.name, f"template_{op[4:]}_1d")
                self.assertEqual(selected.metadata.loop_depth, 1)
                self.assertEqual(mlir.count("scf.for"), 1)

    def test_select_f32_fallback_processes_a_65_element_row_as_a_pair(self):
        for op, module in (
            ("pto.tsel", tsel_templates),
            ("pto.tsels", tsels_templates),
        ):
            with self.subTest(op=op):
                self.assertEqual(module._f32_paired_cols(63, 64), 0)
                self.assertEqual(module._f32_paired_cols(65, 64), 128)
                self.assertEqual(module._f32_paired_cols(129, 64), 128)

                specs = _select_specs(
                    op,
                    "f32",
                    data_shape=(4, 128),
                    data_valid_shape=(4, 65),
                )
                fallback = "template_tsel" if op == "pto.tsel" else "template_tsels"
                selected = select(op, "a5", specs, candidate_id=fallback)
                mlir = selected.specialize(**specs).mlir_text()

                self.assertIn("pto.pintlv_b16", mlir)

    def test_single_row_tcmp_mask_is_legal_input_to_tsel_1d(self):
        data = TileSpec(
            shape=(1, 256),
            valid_shape=(1, 255),
            dtype=ScalarType("f32"),
        )
        mask = TileSpec(
            shape=(1, 256),
            valid_shape=(1, 255),
            dtype=ScalarType("i8"),
        )
        compare_specs = {"src0": data, "src1": data, "dst": mask}
        select_specs = {
            "mask": mask,
            "src0": data,
            "src1": data,
            "tmp": TileSpec(
                shape=(1, 64),
                dtype=ScalarType("f32"),
            ),
            "dst": data,
        }

        self.assertEqual(
            select("pto.tcmp", "a5", compare_specs).name,
            "template_tcmp_1d",
        )
        self.assertEqual(
            select("pto.tsel", "a5", select_specs).name,
            "template_tsel_1d",
        )

    def test_tdiv_mismatched_ranges_retain_existing_2d_candidate(self):
        specs = {
            "src0": TileSpec(shape=(4, 65), dtype=ScalarType("f32")),
            "src1": TileSpec(
                shape=(4, 65),
                valid_shape=(3, 65),
                dtype=ScalarType("f32"),
            ),
            "dst": TileSpec(shape=(4, 65), dtype=ScalarType("f32")),
        }

        candidates = legal_candidates("pto.tdiv", "a5", specs)
        self.assertEqual(
            [candidate.name for candidate in candidates],
            ["template_tdiv"],
        )
        self.assertEqual(candidates[0].metadata.loop_depth, 2)


if __name__ == "__main__":
    unittest.main()
