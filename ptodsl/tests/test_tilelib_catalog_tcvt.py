# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Selection and render coverage for the PTODSL TileLib tcvt catalog."""

from pathlib import Path
import unittest

import ptodsl.tilelib as tilelib
from ptodsl.tilelib import ScalarType, TileSpec, select


def _tcvt_specs_for(src_dtype, dst_dtype):
    """Build legal tile specs for one tcvt dtype signature.

    Packed fp4 pairs occupy one byte for two values, so those signatures need a
    2:1 shape ratio; every other pair is elementwise.
    """

    def _packed(name):
        return name.startswith("f4")

    src_shape, dst_shape = (8, 64), (8, 64)
    if _packed(src_dtype):
        src_shape, dst_shape = (8, 32), (8, 64)
    elif _packed(dst_dtype):
        src_shape, dst_shape = (8, 64), (8, 32)
    return {
        "src": TileSpec(shape=src_shape, dtype=ScalarType(src_dtype), valid_shape=src_shape),
        "dst": TileSpec(shape=dst_shape, dtype=ScalarType(dst_dtype), valid_shape=dst_shape),
    }


# (src, dst) element types -> the flattened (loop_depth 1) tcvt template that
# must be selected for each contiguous full-tile signature.
_TCVT_FLATTENED_SIGNATURES = {
    ("i32", "f32"): "template_tcvt_i32_to_f32",
    ("i16", "f16"): "template_tcvt_i16_to_f16",
    ("f16", "i16"): "template_tcvt_f16_to_i16",
    ("bf16", "f16"): "template_tcvt_bf16_to_f16",
    ("f32", "f16"): "template_tcvt_f32_to_f16",
    ("f32", "bf16"): "template_tcvt_f32_to_bf16",
    ("f32", "i16"): "template_tcvt_f32_to_i16",
    ("f32", "i64"): "template_tcvt_f32_to_i64",
    ("f32", "f32"): "template_tcvt_f32_to_f32",
    ("f16", "i32"): "template_tcvt_f16_to_i32",
    ("f16", "f32"): "template_tcvt_f16_to_f32",
    ("f16", "ui8"): "template_tcvt_f16_to_ui8",
    ("f16", "si8"): "template_tcvt_f16_to_si8",
    ("bf16", "f32"): "template_tcvt_bf16_to_f32",
    ("bf16", "i32"): "template_tcvt_bf16_to_i32",
    ("ui8", "f16"): "template_tcvt_ui8_to_f16",
    ("ui8", "ui16"): "template_tcvt_ui8_to_ui16",
    ("si8", "f16"): "template_tcvt_si8_to_f16",
    ("si8", "si16"): "template_tcvt_si8_to_si16",
    ("si8", "i32"): "template_tcvt_si8_to_i32",
    ("i16", "ui8"): "template_tcvt_i16_to_ui8",
    ("i16", "f32"): "template_tcvt_i16_to_f32",
    ("i16", "ui32"): "template_tcvt_i16_to_ui32",
    ("i16", "i32"): "template_tcvt_i16_to_i32",
    ("i32", "i64"): "template_tcvt_i32_to_i64",
    ("i32", "i16"): "template_tcvt_i32_to_i16",
    ("i32", "ui8"): "template_tcvt_i32_to_ui8",
    ("i32", "ui16"): "template_tcvt_i32_to_ui16",
    ("ui32", "i16"): "template_tcvt_ui32_to_i16",
    ("ui32", "ui16"): "template_tcvt_ui32_to_ui16",
    ("ui32", "ui8"): "template_tcvt_ui32_to_ui8",
    ("i64", "f32"): "template_tcvt_i64_to_f32",
    ("i64", "i32"): "template_tcvt_i64_to_i32",
    ("f32", "f8e4m3"): "template_tcvt_f32_to_fp8",
    ("f32", "f8e5m2"): "template_tcvt_f32_to_fp8",
    ("f32", "hif8"): "template_tcvt_f32_to_hif8",
    ("f16", "hif8"): "template_tcvt_f16_to_hif8",
}


class TileLibCatalogTcvtTest(unittest.TestCase):
    def test_tcvt_fp8_widening_pairs_and_signed_narrowing_gaps(self):
        # Issue #1612: f8e4m3/f8e5m2 -> f32 widened pairs are catalogued, while
        # i32/ui32 -> si8 has no A5 intrinsic and must stay unmatched.
        for src_dtype in ("f8e4m3", "f8e5m2"):
            specs = {
                "src": TileSpec(shape=(8, 64), dtype=ScalarType(src_dtype)),
                "dst": TileSpec(shape=(8, 64), dtype=ScalarType("f32")),
            }
            with self.subTest(signature=(src_dtype, "f32")):
                selected = select("pto.tcvt", "a5", specs)
                self.assertTrue(selected.name.startswith("template_tcvt_fp8_to_f32"))
        for src_dtype in ("i32", "ui32"):
            specs = {
                "src": TileSpec(shape=(8, 64), dtype=ScalarType(src_dtype)),
                "dst": TileSpec(shape=(8, 64), dtype=ScalarType("si8")),
            }
            with self.subTest(signature=(src_dtype, "si8")):
                with self.assertRaises(tilelib.NoMatchingTemplate):
                    select("pto.tcvt", "a5", specs)

    def test_tcvt_contiguous_versions_select_flattened_candidates(self):
        for (src_dtype, dst_dtype), expected_name in _TCVT_FLATTENED_SIGNATURES.items():
            with self.subTest(signature=(src_dtype, dst_dtype)):
                specs = {
                    "src": TileSpec(shape=(8, 64), dtype=ScalarType(src_dtype)),
                    "dst": TileSpec(shape=(8, 64), dtype=ScalarType(dst_dtype)),
                }
                selected = select("pto.tcvt", "a5", specs)
                self.assertEqual(selected.name, f"{expected_name}_1d")
                self.assertEqual(selected.metadata.loop_depth, 1)
                expected_op = "pto.vtrc" if expected_name == "template_tcvt_f32_to_f32" else "pto.vcvt"
                mlir = selected.specialize(**specs).mlir_text()
                self.assertEqual(mlir.count("scf.for"), 1)
                self.assertIn(expected_op, mlir)

    def test_tcvt_bf16_to_fp4_versions_render(self):
        for dst_dtype in ("f4e1m2x2", "f4e2m1x2"):
            with self.subTest(dst_dtype=dst_dtype):
                specs = {
                    "src": TileSpec(shape=(8, 128), dtype=ScalarType("bf16")),
                    "dst": TileSpec(shape=(8, 64), dtype=ScalarType(dst_dtype)),
                }
                selected = select("pto.tcvt", "a5", specs)
                self.assertEqual(selected.name, "template_tcvt_bf16_to_fp4_1d")
                self.assertEqual(selected.metadata.loop_depth, 1)
                mlir = selected.specialize(**specs).mlir_text()
                self.assertEqual(mlir.count("scf.for"), 1)
                self.assertIn("pto.vcvt", mlir)

    def test_tcvt_32_to_ui8_store_mask_matches_source_chunk(self):
        cases = (
            (
                "flattened",
                (1, 128),
                None,
                "template_tcvt_i32_to_ui8_1d",
                1,
            ),
            (
                "rowwise",
                (2, 128),
                (2, 65),
                "template_tcvt_i32_to_ui8",
                2,
            ),
        )
        for src_dtype in ("i32", "ui32"):
            for label, shape, valid_shape, expected_name, loop_depth in cases:
                with self.subTest(src_dtype=src_dtype, case=label):
                    specs = {
                        "src": TileSpec(
                            shape=shape,
                            valid_shape=valid_shape,
                            dtype=ScalarType(src_dtype),
                        ),
                        "dst": TileSpec(
                            shape=shape,
                            valid_shape=valid_shape,
                            dtype=ScalarType("ui8"),
                        ),
                    }
                    selected = select("pto.tcvt", "a5", specs)
                    mlir = selected.specialize(**specs).mlir_text()

                    self.assertEqual(
                        selected.name,
                        expected_name.replace("i32", src_dtype),
                    )
                    self.assertEqual(selected.metadata.loop_depth, loop_depth)
                    # Issue #1612: the b8 store mask counts the chunk's valid
                    # bytes, so it is built with plt_b8 from the source chunk
                    # count; folding the b32 element mask with pbitcast enabled
                    # only every fourth byte.
                    self.assertIn("pto.plt_b32", mlir)
                    self.assertIn("pto.pbitcast", mlir)
                    self.assertIn("pto.plt_b8", mlir)

    def test_tcvt_partial_multi_row_ranges_retain_2d_fallbacks(self):
        cases = (
            ("f32", "i32", "template_tcvt_f32_to_i32"),
            ("f16", "i16", "template_tcvt_f16_to_i16"),
            ("i64", "f32", "template_tcvt_i64_to_f32"),
        )
        for src_dtype, dst_dtype, expected_name in cases:
            with self.subTest(signature=(src_dtype, dst_dtype)):
                specs = {
                    "src": TileSpec(
                        shape=(8, 65),
                        valid_shape=(8, 63),
                        dtype=ScalarType(src_dtype),
                    ),
                    "dst": TileSpec(
                        shape=(8, 65),
                        valid_shape=(8, 63),
                        dtype=ScalarType(dst_dtype),
                    ),
                }
                selected = select("pto.tcvt", "a5", specs)
                self.assertEqual(selected.name, expected_name)
                self.assertEqual(selected.metadata.loop_depth, 2)
                self.assertEqual(
                    selected.specialize(**specs).mlir_text().count("scf.for"),
                    2,
                )

        fp4_specs = {
            "src": TileSpec(
                shape=(8, 130),
                valid_shape=(8, 126),
                dtype=ScalarType("bf16"),
            ),
            "dst": TileSpec(
                shape=(8, 65),
                valid_shape=(8, 63),
                dtype=ScalarType("f4e1m2x2"),
            ),
        }
        selected = select("pto.tcvt", "a5", fp4_specs)
        self.assertEqual(selected.name, "template_tcvt_bf16_to_fp4")
        self.assertEqual(selected.metadata.loop_depth, 2)

    def test_tcvt_single_row_and_stride_gap_selection(self):
        single_row = {
            "src": TileSpec(
                shape=(4, 65),
                valid_shape=(1, 63),
                dtype=ScalarType("f32"),
            ),
            "dst": TileSpec(
                shape=(4, 65),
                valid_shape=(1, 63),
                dtype=ScalarType("i32"),
            ),
        }
        stride_gap = {
            "src": TileSpec(
                shape=(4, 65),
                dtype=ScalarType("f32"),
                compact_mode="row_plus_one",
            ),
            "dst": TileSpec(
                shape=(4, 65),
                dtype=ScalarType("i32"),
            ),
        }

        self.assertEqual(
            select("pto.tcvt", "a5", single_row).name,
            "template_tcvt_f32_to_i32_1d",
        )
        self.assertEqual(
            select("pto.tcvt", "a5", stride_gap).name,
            "template_tcvt_f32_to_i32",
        )

    def test_tcvt_catalog_has_one_1d_pair_for_every_existing_candidate(self):
        specs = {
            "src": TileSpec(shape=(8, 64), dtype=ScalarType("f32")),
            "dst": TileSpec(shape=(8, 64), dtype=ScalarType("i32")),
        }
        select("pto.tcvt", "a5", specs)
        candidates = list(tilelib.default_registry().lookup("pto.tcvt", "a5"))
        by_id = {descriptor.metadata.id: descriptor for descriptor in candidates}

        self.assertEqual(len(candidates), 78)
        self.assertEqual(set(by_id), set(range(78)))
        for fallback_id in range(39):
            with self.subTest(fallback_id=fallback_id):
                fallback = by_id[fallback_id]
                flattened = by_id[39 + fallback_id]
                self.assertEqual(fallback.metadata.loop_depth, 2)
                self.assertEqual(fallback.metadata.priority, 0)
                self.assertEqual(flattened.metadata.loop_depth, 1)
                self.assertEqual(flattened.metadata.priority, 10)
                self.assertEqual(flattened.name, f"{fallback.name}_1d")

    def test_tcvt_every_registered_candidate_materializes(self):
        # Issue #1612: a misplaced @tilelib.tile_template decorator left two 2D
        # entry points uncallable and no numeric case noticed, because only the
        # flattened candidates had coverage.  Materializing every candidate keeps
        # that class of registration failure visible without a numeric case.
        # Loading is lazy, so trigger it explicitly: this test must not depend on
        # another test having selected a tcvt candidate first.
        select("pto.tcvt", "a5", _tcvt_specs_for("f32", "i32"))
        candidates = list(tilelib.default_registry().lookup("pto.tcvt", "a5"))
        self.assertTrue(candidates)
        for descriptor in candidates:
            src_dtype, dst_dtype = descriptor.metadata.dtypes[0]
            specs = _tcvt_specs_for(src_dtype, dst_dtype)
            with self.subTest(candidate=descriptor.name):
                selected = select("pto.tcvt", "a5", specs, candidate_id=descriptor.name)
                mlir = selected.specialize(**specs).mlir_text()
                self.assertIn(f"func.func @{descriptor.name}", mlir)

    def test_tcvt_templates_use_native_python_loop_syntax(self):
        from ptodsl.tilelib.templates.a5 import tcvt

        source = Path(tcvt.__file__).read_text(encoding="utf-8")
        self.assertNotIn("pto.for_(", source)


if __name__ == "__main__":
    unittest.main()
