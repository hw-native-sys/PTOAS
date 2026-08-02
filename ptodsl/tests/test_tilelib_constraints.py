# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Constraint-driven selection test for the reduction op pto.tcolmax.

tcolmax is legal only for row-major / none-box operands with a 1-row output
(dst_valid_shape[0] == 1). Selection must accept the legal case and reject the others.
"""

import unittest

from ptodsl.tilelib import ScalarType, TileSpec, VectorSpec, ViewSpec, select
from ptodsl.tilelib.constraints import (
    build_context,
    passes,
    require_conversion_1d,
    require_elementwise_1d,
    require_predicate_compare_1d,
    require_predicate_select_1d,
)
from ptodsl.tilelib.registry import NoMatchingTemplate

F32 = ScalarType("f32")


def _specs(*, dst_valid=(1, 64), dst_blayout="row_major", dst_slayout="none_box"):
    src = TileSpec(shape=(8, 64), dtype=F32, valid_shape=(8, 64))
    dst = TileSpec(shape=(8, 64), dtype=F32, valid_shape=dst_valid,
                   b_layout=dst_blayout, s_layout=dst_slayout)
    return {"src": src, "dst": dst}


def _tile(
    *,
    shape=(8, 64),
    valid_shape=None,
    memory_space="ub",
    b_layout="row_major",
    s_layout="none_box",
    compact_mode="null",
):
    return TileSpec(
        shape=shape,
        dtype=F32,
        valid_shape=valid_shape if valid_shape is not None else shape,
        memory_space=memory_space,
        b_layout=b_layout,
        s_layout=s_layout,
        compact_mode=compact_mode,
    )


def _ordinary_elementwise_1d(specs, *operand_names):
    context = build_context(specs, "a5", "pto.example")
    return passes((require_elementwise_1d(*operand_names),), context)


def _conversion_1d(specs, *, source_elements_per_destination=1):
    context = build_context(specs, "a5", "pto.tcvt")
    return passes(
        (
            require_conversion_1d(
                source_elements_per_destination=(
                    source_elements_per_destination
                ),
            ),
        ),
        context,
    )


def _predicate_compare_1d(specs, *data_operand_names):
    context = build_context(specs, "a5", "pto.example")
    return passes(
        (
            require_predicate_compare_1d(
                *data_operand_names,
                predicate_operand="dst",
            ),
        ),
        context,
    )


def _predicate_select_1d(specs, *data_operand_names):
    context = build_context(specs, "a5", "pto.example")
    return passes(
        (
            require_predicate_select_1d(
                "mask",
                *data_operand_names,
                temporary_operand="tmp",
            ),
        ),
        context,
    )


class TileLibConstraintTest(unittest.TestCase):
    def test_legal_colmax_selected(self):
        chosen = select("pto.tcolmax", "a5", _specs())
        self.assertEqual(chosen.name, "template_tcolmax")

    def test_rejected_when_dst_not_single_row(self):
        with self.assertRaises(NoMatchingTemplate):
            select("pto.tcolmax", "a5", _specs(dst_valid=(8, 64)))

    def test_rejected_when_not_row_major(self):
        with self.assertRaises(NoMatchingTemplate):
            select("pto.tcolmax", "a5", _specs(dst_blayout="col_major"))

    def test_rejected_when_not_none_box(self):
        with self.assertRaises(NoMatchingTemplate):
            select("pto.tcolmax", "a5", _specs(dst_slayout="row_major"))

    def test_legal_colmax_renders_structured_mlir(self):
        chosen = select("pto.tcolmax", "a5", _specs())
        mlir = chosen.specialize(**_specs()).mlir_text()
        for op in ("pto.tile_valid_rows", "!pto.ptr<f32, ub>", "scf.for", "iter_args",
                   "pto.vmax", "pto.vsts", "pto.tilelang.instance"):
            self.assertIn(op, mlir)
        self.assertNotIn("pto.castptr", mlir)

    def test_context_tracks_view_and_vector_operands(self):
        context = build_context(
            {
                "src": ViewSpec(
                    shape=(16, 64),
                    dtype=F32,
                    memory_space="gm",
                    strides=(64, 1),
                    layout="ND",
                ),
                "aux": VectorSpec(shape=(4,), dtype=ScalarType("i32")),
            },
            "a5",
            "pto.example",
        )

        self.assertEqual(context["operand_kinds"], ("view", "vector"))
        self.assertEqual(context["src_shape"], (16, 64))
        self.assertEqual(context["src_strides"], (64, 1))
        self.assertEqual(context["src_memory_space"], "gm")
        self.assertEqual(context["src_layout"], "ND")
        self.assertEqual(context["aux_shape"], (4,))
        self.assertEqual(context["aux_size"], 4)

    def test_elementwise_1d_accepts_contiguous_multi_row_tiles(self):
        specs = {
            "src0": _tile(),
            "src1": _tile(memory_space="vec"),
            "dst": _tile(),
        }

        self.assertTrue(
            _ordinary_elementwise_1d(specs, "src0", "src1", "dst")
        )

    def test_elementwise_1d_accepts_a_contiguous_single_logical_row(self):
        specs = {
            "src": _tile(valid_shape=(1, 31)),
            "dst": _tile(valid_shape=(1, 31)),
        }

        self.assertTrue(_ordinary_elementwise_1d(specs, "src", "dst"))

    def test_elementwise_1d_rejects_partial_columns_across_rows(self):
        specs = {
            "src": _tile(valid_shape=(4, 63)),
            "dst": _tile(valid_shape=(4, 63)),
        }

        self.assertFalse(_ordinary_elementwise_1d(specs, "src", "dst"))

    def test_elementwise_1d_rejects_mismatched_logical_ranges(self):
        specs = {
            "src": _tile(valid_shape=(8, 64)),
            "dst": _tile(valid_shape=(7, 64)),
        }

        self.assertFalse(_ordinary_elementwise_1d(specs, "src", "dst"))

    def test_elementwise_1d_rejects_unknown_or_unsupported_metadata(self):
        cases = {
            "dynamic valid shape": {
                "src": _tile(valid_shape=(None, 64)),
                "dst": _tile(valid_shape=(None, 64)),
            },
            "non-local memory": {
                "src": _tile(memory_space="gm"),
                "dst": _tile(),
            },
            "column-major block layout": {
                "src": _tile(b_layout="col_major"),
                "dst": _tile(),
            },
            "boxed sub-layout": {
                "src": _tile(s_layout="row_major"),
                "dst": _tile(),
            },
            "stride gap": {
                "src": _tile(compact_mode=2),
                "dst": _tile(),
            },
            "unknown compact mode": {
                "src": _tile(compact_mode=None),
                "dst": _tile(),
            },
        }

        for label, specs in cases.items():
            with self.subTest(label=label):
                self.assertFalse(
                    _ordinary_elementwise_1d(specs, "src", "dst")
                )

    def test_elementwise_1d_includes_temporary_tiles(self):
        specs = {
            "src": _tile(),
            "tmp": _tile(shape=(8, 65), valid_shape=(8, 64)),
            "dst": _tile(),
        }

        self.assertTrue(_ordinary_elementwise_1d(specs, "src", "dst"))
        self.assertFalse(
            _ordinary_elementwise_1d(specs, "src", "tmp", "dst")
        )

    def test_conversion_1d_accepts_typed_contiguous_streams(self):
        specs = {
            "src": _tile(shape=(8, 65)),
            "dst": TileSpec(
                shape=(8, 65),
                dtype=ScalarType("i16"),
                valid_shape=(8, 65),
                compact_mode="normal",
            ),
        }
        self.assertTrue(_conversion_1d(specs))

    def test_conversion_1d_accepts_a_partial_single_logical_row(self):
        specs = {
            "src": _tile(shape=(4, 65), valid_shape=(1, 63)),
            "dst": TileSpec(
                shape=(4, 65),
                dtype=ScalarType("i16"),
                valid_shape=(1, 63),
            ),
        }
        self.assertTrue(_conversion_1d(specs))

    def test_conversion_1d_rejects_partial_multi_row_or_stride_gap(self):
        partial = {
            "src": _tile(shape=(4, 65), valid_shape=(4, 63)),
            "dst": TileSpec(
                shape=(4, 65),
                dtype=ScalarType("i16"),
                valid_shape=(4, 63),
            ),
        }
        stride_gap = {
            "src": _tile(shape=(4, 65), compact_mode="row_plus_one"),
            "dst": TileSpec(
                shape=(4, 65),
                dtype=ScalarType("i16"),
            ),
        }
        self.assertFalse(_conversion_1d(partial))
        self.assertFalse(_conversion_1d(stride_gap))

    def test_conversion_1d_models_bf16_to_fp4_packing(self):
        legal = {
            "src": TileSpec(
                shape=(8, 130),
                dtype=ScalarType("bf16"),
            ),
            "dst": TileSpec(
                shape=(8, 65),
                dtype=ScalarType("f4e1m2x2"),
            ),
        }
        wrong_ratio = {
            "src": TileSpec(
                shape=(8, 129),
                dtype=ScalarType("bf16"),
            ),
            "dst": legal["dst"],
        }
        self.assertTrue(
            _conversion_1d(
                legal,
                source_elements_per_destination=2,
            )
        )
        self.assertFalse(
            _conversion_1d(
                wrong_ratio,
                source_elements_per_destination=2,
            )
        )

    def test_predicate_compare_1d_accepts_single_row_with_rounded_capacity(self):
        specs = {
            "src": _tile(shape=(4, 128), valid_shape=(1, 63)),
            "dst": TileSpec(
                shape=(4, 32),
                valid_shape=(1, 8),
                dtype=ScalarType("ui8"),
            ),
        }

        self.assertTrue(_predicate_compare_1d(specs, "src"))

    def test_predicate_compare_1d_accepts_dense_block_aligned_rows(self):
        specs = {
            "src0": _tile(shape=(4, 256)),
            "src1": _tile(shape=(4, 256), memory_space="vec"),
            "dst": TileSpec(
                shape=(4, 32),
                dtype=ScalarType("i8"),
            ),
        }

        self.assertTrue(_predicate_compare_1d(specs, "src0", "src1"))

    def test_predicate_compare_1d_rejects_row_tail_or_predicate_padding(self):
        cases = {
            "source row tail": {
                "src": TileSpec(
                    shape=(4, 128),
                    dtype=ScalarType("i8"),
                ),
                "dst": TileSpec(
                    shape=(4, 32),
                    dtype=ScalarType("ui8"),
                ),
            },
            "predicate row padding": {
                "src": _tile(shape=(4, 256)),
                "dst": TileSpec(
                    shape=(4, 64),
                    dtype=ScalarType("ui8"),
                ),
            },
            "insufficient single-row capacity": {
                "src": _tile(shape=(1, 65), valid_shape=(1, 63)),
                "dst": TileSpec(
                    shape=(1, 32),
                    dtype=ScalarType("ui8"),
                ),
            },
            "predicate stride gap": {
                "src": _tile(shape=(4, 256)),
                "dst": TileSpec(
                    shape=(4, 32),
                    dtype=ScalarType("ui8"),
                    compact_mode=2,
                ),
            },
        }

        for label, specs in cases.items():
            with self.subTest(label=label):
                self.assertFalse(_predicate_compare_1d(specs, "src"))

    def test_predicate_select_1d_accepts_single_row_and_dense_multi_row(self):
        cases = {
            "single row": {
                "mask": TileSpec(
                    shape=(1, 32),
                    valid_shape=(1, 8),
                    dtype=ScalarType("i8"),
                ),
                "src": _tile(shape=(1, 128), valid_shape=(1, 63)),
                "tmp": _tile(shape=(1, 32)),
                "dst": _tile(shape=(1, 128), valid_shape=(1, 63)),
            },
            "dense multi row with i32 mask container": {
                "mask": TileSpec(
                    shape=(4, 8),
                    dtype=ScalarType("i32"),
                ),
                "src": _tile(shape=(4, 256)),
                "tmp": _tile(shape=(1, 32)),
                "dst": _tile(shape=(4, 256)),
            },
        }

        for label, specs in cases.items():
            with self.subTest(label=label):
                self.assertTrue(_predicate_select_1d(specs, "src", "dst"))

    def test_predicate_select_1d_rejects_ineligible_metadata(self):
        common = {
            "mask": TileSpec(
                shape=(4, 32),
                dtype=ScalarType("i8"),
            ),
            "src": _tile(shape=(4, 256)),
            "tmp": _tile(shape=(1, 32)),
            "dst": _tile(shape=(4, 256)),
        }
        cases = {
            "partial data row": {
                **common,
                "src": _tile(shape=(4, 256), valid_shape=(4, 255)),
                "dst": _tile(shape=(4, 256), valid_shape=(4, 255)),
            },
            "predicate row padding": {
                **common,
                "mask": TileSpec(
                    shape=(4, 64),
                    dtype=ScalarType("i8"),
                ),
            },
            "predicate stride gap": {
                **common,
                "mask": TileSpec(
                    shape=(4, 32),
                    dtype=ScalarType("i8"),
                    compact_mode=2,
                ),
            },
            "temporary stride gap": {
                **common,
                "tmp": _tile(shape=(1, 32), compact_mode=2),
            },
            "mismatched data range": {
                **common,
                "dst": _tile(shape=(4, 256), valid_shape=(3, 256)),
            },
        }

        for label, specs in cases.items():
            with self.subTest(label=label):
                self.assertFalse(_predicate_select_1d(specs, "src", "dst"))

    def test_context_carries_complete_tile_layout_metadata(self):
        context = build_context(
            {
                "src": TileSpec(
                    shape=(8, 64),
                    dtype=F32,
                    s_fractal_size=32,
                    compact_mode=1,
                )
            },
            "a5",
            "pto.example",
        )

        self.assertEqual(context["src_s_fractal_size"], 32)
        self.assertEqual(context["src_compact_mode"], 1)
        self.assertEqual(context["src_config"].s_fractal_size, 32)
        self.assertEqual(context["src_config"].compact_mode, 1)
        self.assertEqual(context["operand_s_fractal_sizes"], (32,))
        self.assertEqual(context["operand_compact_modes"], (1,))


if __name__ == "__main__":
    unittest.main()
