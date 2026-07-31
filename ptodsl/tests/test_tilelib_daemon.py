# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""End-to-end tests for the PTODSL TileLib daemon's Unix-socket RPC."""

import os
import socket
import stat
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from ptodsl.tilelib import AmbiguousTemplate, TemplateMetadata
from ptodsl.tilelib.serving.client import DaemonClient, DaemonError
from ptodsl.tilelib.serving.daemon import (
    TileLibDaemonServer,
    _build_tile_specs,
    metadata_request,
    _remove_socket_path,
)
from ptodsl.tilelib.serving.wire import MAX_MESSAGE_SIZE, recv_message


def _tile_spec(dtype="f32", shape=(8, 64)):
    return {
        "kind": "tile",
        "dtype": dtype,
        "shape": list(shape),
        "valid_shape": list(shape),
        "memory_space": "ub",
        "config": {
            "b_layout": "row_major",
            "s_layout": "none_box",
            "s_fractal_size": 512,
            "pad_value": "0x0",
        },
    }


def _view_spec(dtype="f32", shape=(1, 1, 1, 8, 64), strides=(512, 512, 512, 64, 1)):
    return {
        "kind": "view",
        "dtype": dtype,
        "shape": list(shape),
        "strides": list(strides),
        "memory_space": "gm",
    }


# ExpandTileOp sends tadd as ins(src0, src1), outs(dst), matching the
# template parameter order (src0, src1, dst).
TADD_OPERANDS = [_tile_spec(), _tile_spec(), _tile_spec()]
TADD = "template_tadd"
TADD_1D = "template_tadd_1d"


def _metadata_descriptor(name, priority, candidate_id):
    return SimpleNamespace(
        op="pto.tadd",
        target="a5",
        name=name,
        param_names=("src0", "src1", "dst"),
        metadata=TemplateMetadata.build(
            op="pto.tadd",
            target="a5",
            name=name,
            priority=priority,
            id=candidate_id,
            loop_depth=1 if priority > 0 else 2,
        ),
    )


class TileLibMetadataRequestTest(unittest.TestCase):
    def test_metadata_candidates_preserve_priority_not_id_or_registration_order(self):
        preferred = _metadata_descriptor(
            "preferred_1d",
            priority=10,
            candidate_id=99,
        )
        fallback = _metadata_descriptor(
            "fallback_2d",
            priority=0,
            candidate_id=0,
        )

        for registered in ((fallback, preferred), (preferred, fallback)):
            with self.subTest(
                registered=[candidate.name for candidate in registered]
            ):
                with patch(
                    "ptodsl.tilelib.serving.daemon._registered_candidates",
                    return_value=registered,
                ):
                    metadata = metadata_request(
                        "a5",
                        "pto.tadd",
                        TADD_OPERANDS,
                    )

                self.assertEqual(
                    [
                        (candidate["name"], candidate["id"])
                        for candidate in metadata["candidates"]
                    ],
                    [("preferred_1d", 99), ("fallback_2d", 0)],
                )

    def test_metadata_rejects_equal_top_priority(self):
        tied = (
            _metadata_descriptor("second", priority=1, candidate_id=1),
            _metadata_descriptor("first", priority=1, candidate_id=0),
        )
        with patch(
            "ptodsl.tilelib.serving.daemon._registered_candidates",
            return_value=tied,
        ):
            with self.assertRaises(AmbiguousTemplate):
                metadata_request("a5", "pto.tadd", TADD_OPERANDS)


class TileLibDaemonTest(unittest.TestCase):
    def setUp(self):
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.socket_path = os.path.join(
            self._temporary_directory.name,
            "ptodsl_tilelib.sock",
        )
        self.server = TileLibDaemonServer(self.socket_path)
        self._thread = threading.Thread(
            target=self.server.serve_forever,
            daemon=True,
        )
        self._thread.start()
        self.client = DaemonClient(self.socket_path)

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self._thread.join()
        self._temporary_directory.cleanup()

    def test_ping(self):
        self.assertEqual(self.client.ping(), "pong")

    def test_socket_is_accessible_only_by_owner(self):
        mode = stat.S_IMODE(os.stat(self.socket_path).st_mode)
        self.assertEqual(mode, 0o600)

    def test_instantiate_named_candidate_returns_structured_mlir(self):
        mlir = self.client.instantiate(
            "a5",
            "pto.tadd",
            TADD_OPERANDS,
            candidate_id=TADD,
        )
        self.assertIn(f"func.func @{TADD}", mlir)
        for operation in (
            "pto.tile_buf_addr",
            "!pto.ptr<f32, ub>",
            "pto.vlds",
            "pto.vadd",
            "pto.vsts",
            "pto.plt_b32",
            "pto.tilelang.instance",
        ):
            self.assertIn(operation, mlir)
        self.assertNotIn("pto.castptr", mlir)

    def test_instantiate_uses_preferred_tadd_candidate_without_explicit_id(self):
        mlir = self.client.instantiate("a5", "pto.tadd", TADD_OPERANDS)
        self.assertIn(f"func.func @{TADD_1D}", mlir)

    def test_get_metadata_returns_legal_candidates(self):
        metadata = self.client.get_metadata("a5", "pto.tadd", TADD_OPERANDS)
        candidates = metadata["candidates"]
        self.assertEqual(
            [candidate["name"] for candidate in candidates],
            [TADD_1D, TADD],
        )

        selected = candidates[0]
        self.assertEqual(selected["loop_depth"], 1)
        self.assertIsNone(selected["Tail"])
        self.assertFalse(selected["has_tail"])
        self.assertFalse(selected["is_post_update"])
        self.assertEqual(selected["iteration_axis"], "none")
        self.assertEqual(selected["op_engine"], "vector")
        self.assertEqual(selected["op_class"], "elementwise")
        self.assertEqual(selected["tags"], ["elementwise", "binary"])

    def test_tdivs_metadata_preserves_operand_order_and_ranked_traversal(self):
        scalar = {"kind": "scalar", "dtype": "f32", "value": 1.0}
        cases = (
            (
                [_tile_spec(), scalar, _tile_spec()],
                [
                    ("template_tdivs_tile_scalar_1d", 2, 1),
                    ("template_tdivs_tile_scalar", 0, 2),
                ],
            ),
            (
                [scalar, _tile_spec(), _tile_spec()],
                [
                    ("template_tdivs_scalar_tile_1d", 3, 1),
                    ("template_tdivs_scalar_tile", 1, 2),
                ],
            ),
        )
        for operands, expected in cases:
            with self.subTest(
                operand_order=[
                    operand["kind"]
                    for operand in operands
                ]
            ):
                metadata = self.client.get_metadata(
                    "a5",
                    "pto.tdivs",
                    operands,
                )
                self.assertEqual(
                    [
                        (
                            candidate["name"],
                            candidate["id"],
                            candidate["loop_depth"],
                        )
                        for candidate in metadata["candidates"]
                    ],
                    expected,
                )

    def test_tile_spec_reconstruction_preserves_complete_config(self):
        operand = _tile_spec()
        operand["config"]["s_fractal_size"] = 32
        operand["config"]["compact_mode"] = 2
        descriptor = SimpleNamespace(
            name="template_example",
            param_names=("src",),
        )

        spec = _build_tile_specs(descriptor, [operand])["src"]

        self.assertEqual(spec.s_fractal_size, 32)
        self.assertEqual(spec.compact_mode, 2)

    def test_render_preserves_fractal_and_compact_mode(self):
        operands = [_tile_spec(), _tile_spec(), _tile_spec()]
        for operand in operands:
            operand["config"]["s_fractal_size"] = 32
            operand["config"]["compact_mode"] = 2

        mlir = self.client.instantiate(
            "a5",
            "pto.tadd",
            operands,
            candidate_id=TADD,
        )

        self.assertIn("fractal=32", mlir)
        self.assertIn("compact=2", mlir)

    def test_cache_stats_and_clear_are_available_over_rpc(self):
        arguments = (
            "a5",
            "pto.tadd",
            TADD_OPERANDS,
        )
        self.client.instantiate(
            *arguments,
            candidate_id=TADD,
        )
        self.client.instantiate(
            *arguments,
            candidate_id=TADD,
        )

        stats = self.client.get_stats()
        self.assertEqual(stats["misses"], 1)
        self.assertEqual(stats["hits"], 1)
        self.assertEqual(stats["entries"], 1)

        self.assertEqual(self.client.clear(), {"cleared": True})
        self.assertEqual(self.client.get_stats()["entries"], 0)

    def test_cache_key_includes_context_attributes(self):
        self.client.instantiate(
            "a5",
            "pto.tadd",
            TADD_OPERANDS,
            context_attrs={"variant": 0},
            candidate_id=TADD,
        )
        self.client.instantiate(
            "a5",
            "pto.tadd",
            TADD_OPERANDS,
            context_attrs={"variant": 1},
            candidate_id=TADD,
        )
        self.assertEqual(self.client.get_stats()["misses"], 2)

    def test_oversized_wire_message_is_rejected_before_payload_read(self):
        receiver, sender = socket.socketpair()
        self.addCleanup(receiver.close)
        self.addCleanup(sender.close)
        sender.sendall((MAX_MESSAGE_SIZE + 1).to_bytes(4, byteorder="big"))

        with self.assertRaisesRegex(ValueError, "exceeds limit"):
            recv_message(receiver)

    def test_socket_cleanup_removes_broken_symlink(self):
        missing_target = os.path.join(
            self._temporary_directory.name,
            "missing.sock",
        )
        broken_link = os.path.join(
            self._temporary_directory.name,
            "broken.sock",
        )
        os.symlink(missing_target, broken_link)

        _remove_socket_path(broken_link)

        self.assertFalse(os.path.lexists(broken_link))

    def test_scalar_operand_template_instantiates(self):
        operands = [
            _tile_spec(),
            {"kind": "scalar", "dtype": "f32", "value": 1.0},
            _tile_spec(),
        ]

        mlir = self.client.instantiate("a5", "pto.tadds", operands)

        self.assertIn("func.func @template_tadds_1d", mlir)
        self.assertIn("pto.vadds", mlir)

    def test_render_passes_context_attributes_into_template_body(self):
        operands = [
            _tile_spec(dtype="f32", shape=(8, 64)),
            _tile_spec(dtype="f32", shape=(8, 64)),
            _tile_spec(dtype="i8", shape=(8, 64)),
        ]

        mlir = self.client.instantiate(
            "a5",
            "pto.tcmp",
            operands,
            context_attrs={"cmp_mode": "gt"},
            candidate_id="template_tcmp",
        )

        self.assertIn('"gt"', mlir)
        self.assertNotIn('"eq"', mlir)

    def test_vector_operand_metadata_is_accepted(self):
        operands = [
            _tile_spec(),
            _tile_spec(),
            _tile_spec(),
            _tile_spec(),
            {"kind": "vector", "dtype": "i16", "shape": [4]},
        ]

        metadata = self.client.get_metadata("a5", "pto.tmrgsort", operands)

        self.assertIn(
            "template_tmrgsort_multi_list2",
            [candidate["name"] for candidate in metadata["candidates"]],
        )

    def test_view_operand_template_instantiates(self):
        operands = [_view_spec(), _tile_spec()]

        mlir = self.client.instantiate(
            "a5",
            "pto.tload",
            operands,
            candidate_id="template_tload_nd2nd",
        )

        self.assertIn("func.func @template_tload_nd2nd", mlir)
        self.assertIn("pto.tensor_view_addr", mlir)
        self.assertIn("pto.mte_gm_ub", mlir)

    def test_unsupported_operand_kind_is_rejected_explicitly(self):
        operands = list(TADD_OPERANDS)
        operands[0] = {"kind": "mystery", "dtype": "f32", "shape": [64]}

        with self.assertRaisesRegex(DaemonError, "supports tile, scalar, view, and vector operands"):
            self.client.instantiate(
                "a5",
                "pto.tadd",
                operands,
                candidate_id=TADD,
            )

    def test_unknown_op_errors(self):
        with self.assertRaises(DaemonError):
            self.client.instantiate("a5", "pto.tnope", TADD_OPERANDS)


if __name__ == "__main__":
    unittest.main()
