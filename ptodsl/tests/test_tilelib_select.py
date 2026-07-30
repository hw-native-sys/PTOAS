# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""TileLib selection test: metadata-driven legality and descriptor metadata."""

import unittest
from types import SimpleNamespace

from ptodsl.tilelib import (
    AmbiguousTemplate,
    ScalarSpec,
    ScalarType,
    TemplateMetadata,
    TileSpec,
    TileTemplateRegistry,
    legal_candidates,
    select,
)
from ptodsl.tilelib.registry import NoMatchingTemplate


def _f32_specs():
    spec = TileSpec(shape=(8, 64), dtype=ScalarType("f32"))
    return {"src0": spec, "src1": spec, "dst": spec}


def _plain_specs(*, dtype="f32", memory_space="ub", b_layout="row_major"):
    spec = SimpleNamespace(
        shape=(8, 64),
        valid_shape=(8, 64),
        dtype=ScalarType(dtype),
        memory_space=memory_space,
        b_layout=b_layout,
        s_layout="none_box",
    )
    return {"src0": spec, "src1": spec, "dst": spec}


class TileLibSelectTest(unittest.TestCase):
    def test_priority_order_is_independent_of_registration_and_candidate_id(self):
        def descriptor(name, priority, candidate_id):
            return SimpleNamespace(
                op="pto.test_order",
                target="a5",
                name=name,
                param_names=("src0", "src1", "dst"),
                metadata=TemplateMetadata.build(
                    op="pto.test_order",
                    target="a5",
                    name=name,
                    priority=priority,
                    id=candidate_id,
                ),
            )

        preferred = descriptor("preferred_1d", priority=10, candidate_id=99)
        fallback = descriptor("fallback_2d", priority=0, candidate_id=0)
        for registration_order in (
            (fallback, preferred),
            (preferred, fallback),
        ):
            with self.subTest(
                registration_order=[
                    candidate.name for candidate in registration_order
                ]
            ):
                registry = TileTemplateRegistry()
                for candidate in registration_order:
                    registry.register(candidate)

                legal = registry.legal_candidates(
                    "pto.test_order",
                    "a5",
                    _f32_specs(),
                )

                self.assertEqual(
                    [candidate.name for candidate in legal],
                    ["preferred_1d", "fallback_2d"],
                )
                self.assertEqual(
                    registry.select(
                        "pto.test_order",
                        "a5",
                        _f32_specs(),
                    ).name,
                    "preferred_1d",
                )

    def test_equal_top_priority_is_ambiguous_independent_of_registration_order(self):
        registry = TileTemplateRegistry()
        for name in ("second", "first"):
            registry.register(
                SimpleNamespace(
                    op="pto.test_tie",
                    target="a5",
                    name=name,
                    param_names=("src0", "src1", "dst"),
                    metadata=TemplateMetadata.build(
                        op="pto.test_tie",
                        target="a5",
                        name=name,
                        priority=1,
                    ),
                )
            )

        legal = registry.legal_candidates(
            "pto.test_tie",
            "a5",
            _f32_specs(),
        )
        self.assertEqual(
            [candidate.name for candidate in legal],
            ["first", "second"],
        )
        with self.assertRaises(AmbiguousTemplate):
            registry.select("pto.test_tie", "a5", _f32_specs())

    def test_hard_metadata_legality_is_centralized(self):
        registry = TileTemplateRegistry()
        registry.register(SimpleNamespace(
            op="pto.test_metadata",
            target="a5",
            name="metadata_candidate",
            param_names=("src0", "src1", "dst"),
            metadata=TemplateMetadata.build(
                op="pto.test_metadata",
                target="a5",
                name="metadata_candidate",
                dtypes=(("f32", "f32", "f32"),),
                layouts=("row_major",),
                memory_spaces=("ub",),
            ),
        ))

        legal = registry.legal_candidates(
            "pto.test_metadata",
            "a5",
            _plain_specs(),
        )
        self.assertEqual([candidate.name for candidate in legal], ["metadata_candidate"])

        for specs in (
            _plain_specs(dtype="i8"),
            _plain_specs(memory_space="gm"),
            _plain_specs(b_layout="col_major"),
        ):
            with self.subTest(specs=specs):
                with self.assertRaises(NoMatchingTemplate):
                    registry.legal_candidates(
                        "pto.test_metadata",
                        "a5",
                        specs,
                    )

    def test_context_attributes_are_available_to_constraints(self):
        registry = TileTemplateRegistry()
        registry.register(SimpleNamespace(
            op="pto.test_context",
            target="a5",
            name="context_candidate",
            param_names=("src0", "src1", "dst"),
            metadata=TemplateMetadata.build(
                op="pto.test_context",
                target="a5",
                name="context_candidate",
                constraints=(lambda mode: mode == "enabled",),
            ),
        ))

        legal = registry.legal_candidates(
            "pto.test_context",
            "a5",
            _f32_specs(),
            context_attrs={"mode": "enabled"},
        )
        self.assertEqual([candidate.name for candidate in legal], ["context_candidate"])

        with self.assertRaises(NoMatchingTemplate):
            registry.legal_candidates(
                "pto.test_context",
                "a5",
                _f32_specs(),
                context_attrs={"mode": "disabled"},
            )

    def test_scalar_operand_dtypes_participate_in_legality(self):
        registry = TileTemplateRegistry()
        registry.register(SimpleNamespace(
            op="pto.test_scalar",
            target="a5",
            name="scalar_candidate",
            param_names=("src", "scalar", "dst"),
            metadata=TemplateMetadata.build(
                op="pto.test_scalar",
                target="a5",
                name="scalar_candidate",
                dtypes=(("f32", "f32", "f32"),),
                layouts=("row_major",),
                memory_spaces=("ub",),
            ),
        ))
        tile = TileSpec(shape=(8, 64), dtype=ScalarType("f32"))

        legal = registry.legal_candidates(
            "pto.test_scalar",
            "a5",
            {
                "src": tile,
                "scalar": ScalarSpec(dtype=ScalarType("f32"), value=1.0),
                "dst": tile,
            },
        )
        self.assertEqual([candidate.name for candidate in legal], ["scalar_candidate"])

        with self.assertRaises(NoMatchingTemplate):
            registry.legal_candidates(
                "pto.test_scalar",
                "a5",
                {
                    "src": tile,
                    "scalar": ScalarSpec(dtype=ScalarType("i32"), value=1),
                    "dst": tile,
                },
            )

    def test_tadd_prefers_1d_and_retains_2d_fallback(self):
        candidates = legal_candidates("pto.tadd", "a5", _f32_specs())
        self.assertEqual(
            [candidate.name for candidate in candidates],
            ["template_tadd_1d", "template_tadd"],
        )

        chosen = select("pto.tadd", "a5", _f32_specs())
        self.assertEqual(chosen.name, "template_tadd_1d")
        self.assertEqual(
            chosen.metadata.dtypes,
            (
                ("i8", "i8", "i8"),
                ("i16", "i16", "i16"),
                ("i32", "i32", "i32"),
                ("ui8", "ui8", "ui8"),
                ("ui16", "ui16", "ui16"),
                ("ui32", "ui32", "ui32"),
                ("f16", "f16", "f16"),
                ("bf16", "bf16", "bf16"),
                ("f32", "f32", "f32"),
            ),
        )
        self.assertFalse(chosen.metadata.is_post_update)
        self.assertEqual(chosen.metadata.loop_depth, 1)
        self.assertIsNone(chosen.metadata.Tail)
        self.assertEqual(chosen.metadata.iteration_axis, "none")
        self.assertEqual(chosen.metadata.op_engine, "vector")
        self.assertEqual(chosen.metadata.op_class, "elementwise")
        self.assertEqual(chosen.metadata.tags, ("elementwise", "binary"))

    def test_can_select_named_legal_candidate(self):
        chosen = select(
            "pto.tadd",
            "a5",
            _f32_specs(),
            candidate_id="template_tadd",
        )
        self.assertEqual(chosen.name, "template_tadd")

    def test_no_matching_dtype_raises(self):
        i8 = TileSpec(shape=(8, 64), dtype=ScalarType("i8"))
        f32 = TileSpec(shape=(8, 64), dtype=ScalarType("f32"))
        with self.assertRaises(NoMatchingTemplate):
            select("pto.tadd", "a5", {"src0": i8, "src1": f32, "dst": i8})

    def test_unknown_op_raises(self):
        with self.assertRaises(NoMatchingTemplate):
            select("pto.tnope", "a5", _f32_specs())


if __name__ == "__main__":
    unittest.main()
