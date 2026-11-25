# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Regression coverage for the canonical NZ geometry of GM<->UB/VEC loads and stores.

A 5D NZ view ``[g0, g1, g2, g3, g4]`` covers ``g2 * g3`` logical rows and
``g0 * g1 * g4`` logical columns (#1768): the tload/tstore constraints must
derive both extents from the full product, never from a single dimension.
"""

import unittest
from dataclasses import replace

from ptodsl.tilelib import NoMatchingTemplate, ScalarType, TileSpec, ViewSpec, select


# Element strides of the canonical f32 NZ view [2, 1, 2, 16, 8] (#1768).
NZ_F32_STRIDES = (256, 256, 128, 8, 1)
NZ_F16_GEOMETRY = ((1, 2, 2, 16, 16), (1024, 512, 256, 16, 1), (32, 32))
NZ_I8_GEOMETRY = ((1, 1, 2, 16, 32), (1024, 1024, 512, 32, 1), (32, 32))


def nz_view_specs(*, shape, strides, dtype, tile_shape, valid_shape=None, layout="nz",
                  memory_space="vec", compact_mode="null"):
    return {
        "src": ViewSpec(
            shape=shape,
            dtype=ScalarType(dtype),
            strides=strides,
            layout=layout,
        ),
        "dst": TileSpec(
            shape=tile_shape,
            valid_shape=valid_shape,
            dtype=ScalarType(dtype),
            memory_space=memory_space,
            b_layout="col_major",
            s_layout="row_major",
            compact_mode=compact_mode,
        ),
    }


def nz_load_specs(*, shape=(2, 1, 2, 16, 8), strides=NZ_F32_STRIDES, dtype="f32", tile_shape=(32, 16),
                  valid_shape=None, layout="nz", memory_space="vec", compact_mode="null"):
    return nz_view_specs(
        shape=shape,
        strides=strides,
        dtype=dtype,
        tile_shape=tile_shape,
        valid_shape=valid_shape,
        layout=layout,
        memory_space=memory_space,
        compact_mode=compact_mode,
    )


def nz_store_specs(*, shape=(2, 1, 2, 16, 8), strides=NZ_F32_STRIDES, dtype="f32", tile_shape=(32, 16),
                   valid_shape=None, layout="nz", memory_space="vec", compact_mode="null"):
    load_specs = nz_view_specs(
        shape=shape,
        strides=strides,
        dtype=dtype,
        tile_shape=tile_shape,
        valid_shape=valid_shape,
        layout=layout,
        memory_space=memory_space,
        compact_mode=compact_mode,
    )
    # The store swaps the load's operand roles: the tile moves into the view.
    return {"src": load_specs["dst"], "dst": load_specs["src"]}


class NzLoadStoreTest(unittest.TestCase):
    def test_reproducer_geometry_selects_nz_templates(self):
        # #1768: g2 * g3 = 32 rows and g0 * g1 * g4 = 16 columns must match
        # the [32, 16] tile; the buggy g2-only row count rejected the load.
        for layout in ("nz", None):
            for memory_space in ("ub", "vec"):
                with self.subTest(layout=layout, memory_space=memory_space):
                    load_specs = nz_load_specs(layout=layout, memory_space=memory_space)
                    self.assertEqual(
                        select("pto.tload", "a5", load_specs).name,
                        "template_tload_nz2nz",
                    )
                    store_specs = nz_store_specs(layout=layout, memory_space=memory_space)
                    self.assertEqual(
                        select("pto.tstore", "a5", store_specs).name,
                        "template_tstore_nz",
                    )

    def test_multi_dtype_canonical_nz_geometry(self):
        for dtype, (shape, strides, tile_shape) in (
            ("f16", NZ_F16_GEOMETRY),
            ("i8", NZ_I8_GEOMETRY),
        ):
            with self.subTest(dtype=dtype):
                self.assertEqual(
                    select(
                        "pto.tload",
                        "a5",
                        nz_load_specs(shape=shape, strides=strides, dtype=dtype, tile_shape=tile_shape),
                    ).name,
                    "template_tload_nz2nz",
                )
                self.assertEqual(
                    select(
                        "pto.tstore",
                        "a5",
                        nz_store_specs(shape=shape, strides=strides, dtype=dtype, tile_shape=tile_shape),
                    ).name,
                    "template_tstore_nz",
                )

    def test_padded_tile_and_partial_valid_window(self):
        # A padded physical tile still matches: the NZ geometry bounds only
        # the valid window.  Partial valid rows are legal for loads (the view
        # covers more than the window) but a store must cover exactly the
        # view's logical extent.
        load_cases = (
            ((64, 32), (32, 16)),  # padded physical tile, full valid window
            ((64, 32), (20, 16)),  # partial valid rows
        )
        for tile_shape, valid_shape in load_cases:
            with self.subTest(op="pto.tload", tile=tile_shape, valid=valid_shape):
                self.assertEqual(
                    select(
                        "pto.tload",
                        "a5",
                        nz_load_specs(tile_shape=tile_shape, valid_shape=valid_shape),
                    ).name,
                    "template_tload_nz2nz",
                )
        with self.subTest(op="pto.tstore", tile=(64, 32), valid=(32, 16)):
            self.assertEqual(
                select(
                    "pto.tstore",
                    "a5",
                    nz_store_specs(tile_shape=(64, 32), valid_shape=(32, 16)),
                ).name,
                "template_tstore_nz",
            )
        with self.subTest(op="pto.tstore", tile=(64, 32), valid=(20, 16)):
            with self.assertRaises(NoMatchingTemplate):
                select(
                    "pto.tstore",
                    "a5",
                    nz_store_specs(tile_shape=(64, 32), valid_shape=(20, 16)),
                )

    def test_dynamic_row_extents_follow_the_direction(self):
        # The load DMA length follows the tile's valid rows, so unknown row
        # extents (every "unknown" spelling) keep the load selectable and a
        # None row extent materializes end to end.  The store DMA length is
        # fixed at compile time, so an unknown destination extent cannot
        # statically prove equal row coverage and must be rejected.
        for rows in (None, -1, -(2**63)):
            with self.subTest(rows=rows):
                self.assertEqual(
                    select("pto.tload", "a5", nz_load_specs(shape=(2, 1, rows, 16, 8))).name,
                    "template_tload_nz2nz",
                )
                with self.assertRaises(NoMatchingTemplate):
                    select("pto.tstore", "a5", nz_store_specs(shape=(2, 1, rows, 16, 8)))
        load_specs = nz_load_specs(shape=(2, 1, None, 16, 8))
        rendered = select("pto.tload", "a5", load_specs).specialize(**load_specs).mlir_text()
        self.assertIn("pto.mte_gm_ub", rendered)

    def test_dynamic_column_extents_are_rejected(self):
        # The template bodies iterate g0 and multiply g1/g4 into the tile
        # stride, so an unknown column extent cannot be materialized;
        # selection must fail instead of crashing at specialize time.
        for shape in (
            (None, 1, 2, 16, 8),
            (2, None, 2, 16, 8),
            (2, 1, 2, 16, None),
        ):
            with self.subTest(shape=shape):
                with self.assertRaises(NoMatchingTemplate):
                    select("pto.tload", "a5", nz_load_specs(shape=shape))
                with self.assertRaises(NoMatchingTemplate):
                    select("pto.tstore", "a5", nz_store_specs(shape=shape))

    def test_unknown_group_strides_are_rejected(self):
        # The templates step the g0 groups by strides[0]/strides[1], so
        # unknown values there -- including a view without strides at all --
        # cannot be materialized, while the unused inner strides may stay
        # dynamic.
        for strides in (None, (None, 256, 128, 8, 1), (256, None, 128, 8, 1)):
            with self.subTest(strides=strides):
                with self.assertRaises(NoMatchingTemplate):
                    select("pto.tload", "a5", nz_load_specs(strides=strides))
                with self.assertRaises(NoMatchingTemplate):
                    select("pto.tstore", "a5", nz_store_specs(strides=strides))
        load_specs = nz_load_specs(strides=(256, 256, 128, 8, None))
        rendered = select("pto.tload", "a5", load_specs).specialize(**load_specs).mlir_text()
        self.assertIn("pto.mte_gm_ub", rendered)

    def test_taller_source_prefix_load(self):
        # The load DMA copies only the tile's valid rows, so the source may
        # be taller than the tile's physical rows; only the valid prefix is
        # read and no source-to-tile row capacity requirement applies.
        load_cases = (
            ((2, 1, 4, 16, 8), (32, 16), (32, 16)),  # 64-row source, full tile
            ((2, 1, 2, 16, 8), (16, 16), (2, 16)),  # 32-row source, 2 valid rows
        )
        for shape, tile_shape, valid_shape in load_cases:
            with self.subTest(shape=shape, tile=tile_shape, valid=valid_shape):
                load_specs = nz_load_specs(shape=shape, tile_shape=tile_shape, valid_shape=valid_shape)
                self.assertEqual(
                    select("pto.tload", "a5", load_specs).name,
                    "template_tload_nz2nz",
                )
                rendered = select("pto.tload", "a5", load_specs).specialize(**load_specs).mlir_text()
                self.assertIn("pto.mte_gm_ub", rendered)
        # The store must still cover the tile's valid rows exactly, so a
        # taller source view cannot serve a shorter tile's store.
        with self.assertRaises(NoMatchingTemplate):
            select("pto.tstore", "a5", nz_store_specs(shape=(2, 1, 4, 16, 8)))

    def test_nd_labelled_single_column_block_views_stay_selectable(self):
        # Layout inference labels unannotated 5D views that are not
        # NZ-canonical (single-fractal shapes) as ND.  With a single C0-wide
        # column block (g0 * g1 == 1, g4 == C0) and dense row-major inner
        # strides, ND and NZ memory coincide, so those views keep using the
        # NZ templates.
        single_block = (
            ((1, 1, 1, 16, 8), (128, 128, 128, 8, 1), (16, 8)),
            ((1, 1, 2, 16, 8), (256, 256, 128, 8, 1), (32, 8)),
        )
        for shape, strides, tile_shape in single_block:
            for layout in ("nd", "row_major"):
                with self.subTest(shape=shape, layout=layout):
                    load_specs = nz_load_specs(shape=shape, strides=strides, tile_shape=tile_shape, layout=layout)
                    self.assertEqual(
                        select("pto.tload", "a5", load_specs).name,
                        "template_tload_nz2nz",
                    )
                    rendered = select("pto.tload", "a5", load_specs).specialize(**load_specs).mlir_text()
                    self.assertIn("pto.mte_gm_ub", rendered)
                    store_specs = nz_store_specs(shape=shape, strides=strides, tile_shape=tile_shape, layout=layout)
                    self.assertEqual(
                        select("pto.tstore", "a5", store_specs).name,
                        "template_tstore_nz",
                    )
                    rendered = select("pto.tstore", "a5", store_specs).specialize(**store_specs).mlir_text()
                    self.assertIn("pto.mte_ub_gm", rendered)

    def test_nd_labelled_degenerate_views_are_rejected(self):
        # The template's len_burst moves rows * C0 elements in one contiguous
        # burst, so an ND view wider than one C0 block leaves the tile
        # under-filled (round-4 #510 review: [1,1,1,16,64] into a full 16x64
        # tile moved only 512 of the 4096 bytes), and a gapped inner stride
        # misaligns even a C0-wide view.  Neither can reach the templates.
        inadmissible = (
            ((1, 1, 1, 16, 64), (1024, 1024, 64, 64, 1), (16, 64)),  # g4 = 8 * C0
            ((1, 1, 8, 1, 64), (512, 512, 64, 64, 1), (8, 64)),  # C0-mismatched degenerate
            ((1, 1, 1, 16, 8), (256, 256, 256, 16, 2), (16, 8)),  # gapped inner strides
        )
        for shape, strides, tile_shape in inadmissible:
            for layout in ("nd", "row_major"):
                with self.subTest(shape=shape, layout=layout):
                    self._assert_direction_rejected(
                        "pto.tload",
                        nz_load_specs(shape=shape, strides=strides, tile_shape=tile_shape, layout=layout),
                    )
                    self._assert_direction_rejected(
                        "pto.tstore",
                        nz_store_specs(shape=shape, strides=strides, tile_shape=tile_shape, layout=layout),
                    )

    def test_tile_fractal_width_must_match_the_dma_block(self):
        for fractal_size in (32, 1024):
            with self.subTest(dtype="f32", fractal_size=fractal_size):
                load_specs = nz_load_specs()
                load_specs["dst"] = replace(load_specs["dst"], s_fractal_size=fractal_size)
                store_specs = {"src": load_specs["dst"], "dst": load_specs["src"]}
                self._assert_direction_rejected("pto.tload", load_specs)
                self._assert_direction_rejected("pto.tstore", store_specs)

        shape, strides, tile_shape = NZ_F16_GEOMETRY
        load_specs = nz_load_specs(shape=shape, strides=strides, dtype="f16", tile_shape=tile_shape)
        load_specs["dst"] = replace(load_specs["dst"], s_fractal_size=1024)
        store_specs = {"src": load_specs["dst"], "dst": load_specs["src"]}
        self.assertEqual(select("pto.tload", "a5", load_specs).name, "template_tload_nz2nz")
        self.assertEqual(select("pto.tstore", "a5", store_specs).name, "template_tstore_nz")

    def test_row_plus_one_tiles_are_rejected(self):
        # compact=2 (row_plus_one) pads each C0 block with one extra row, so
        # the physical pitch is (rows + 1) * 32 bytes while the templates step
        # by the dense rows * 32 pitch; both directions must fail selection
        # instead of silently corrupting the tile.
        for compact_mode in ("row_plus_one", "RowPlusOne", 2):
            with self.subTest(compact_mode=compact_mode):
                with self.assertRaises(NoMatchingTemplate):
                    select("pto.tload", "a5", nz_load_specs(compact_mode=compact_mode))
                with self.assertRaises(NoMatchingTemplate):
                    select("pto.tstore", "a5", nz_store_specs(compact_mode=compact_mode))

    def test_load_rejects_geometry_violations(self):
        for shape in (
            (2, 1, 1, 16, 8),  # g2 * g3 rows cannot cover the valid rows
            (1, 1, 2, 16, 4),  # g0 * g1 * g4 cols cannot cover the valid cols
            (4, 1, 2, 16, 8),  # cols exceed the physical tile columns
        ):
            with self.subTest(shape=shape):
                with self.assertRaises(NoMatchingTemplate):
                    select("pto.tload", "a5", nz_load_specs(shape=shape))

    def test_store_rejects_geometry_violations(self):
        for shape in (
            (2, 1, 1, 16, 8),
            (2, 1, 4, 16, 8),
            (1, 1, 2, 16, 4),
            (4, 1, 2, 16, 8),
        ):
            with self.subTest(shape=shape):
                with self.assertRaises(NoMatchingTemplate):
                    select("pto.tstore", "a5", nz_store_specs(shape=shape))

    def test_explicit_conflicting_layouts_are_rejected(self):
        for layout in ("nd", "dn"):
            with self.subTest(layout=layout):
                with self.assertRaises(NoMatchingTemplate):
                    select("pto.tload", "a5", nz_load_specs(layout=layout))
                with self.assertRaises(NoMatchingTemplate):
                    select("pto.tstore", "a5", nz_store_specs(layout=layout))

    def test_non_rank5_views_are_rejected(self):
        with self.assertRaises(NoMatchingTemplate):
            select("pto.tload", "a5", nz_load_specs(shape=(32, 16), strides=(16, 1)))
        with self.assertRaises(NoMatchingTemplate):
            select("pto.tstore", "a5", nz_store_specs(shape=(32, 16), strides=(16, 1)))

    def _assert_direction_rejected(self, op, specs):
        with self.assertRaises(NoMatchingTemplate):
            select(op, "a5", specs)


if __name__ == "__main__":
    unittest.main()
