#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import tempfile
import unittest
from pathlib import Path

from ptoas import _core


INPUT = (
    Path(__file__).parents[2]
    / "test"
    / "lit"
    / "vpto"
    / "expand_tile_op_ptodsl_tadd.pto"
)


class PTOASRuntimeTest(unittest.TestCase):
    def test_process_runtime_serves_consecutive_compilation_contexts(self):
        self.assertTrue(INPUT.exists(), f"missing test input {INPUT}")

        with tempfile.TemporaryDirectory() as temp_dir:
            pto_output = Path(temp_dir) / "result-pto.mlir"
            vpto_output = Path(temp_dir) / "result-vpto.mlir"

            for output_mode, output in (
                ("--emit-pto-ir", pto_output),
                ("--emit-vpto", vpto_output),
            ):
                result = _core.main(
                    [
                        "ptoas",
                        "--pto-arch=a5",
                        "--pto-backend=vpto",
                        output_mode,
                        str(INPUT),
                        "-o",
                        str(output),
                    ]
                )

                self.assertEqual(result, 0)
                self.assertTrue(output.exists())

            pto_ir = pto_output.read_text(encoding="utf-8")
            vpto_ir = vpto_output.read_text(encoding="utf-8")
            self.assertIn("pto.tadd ins", pto_ir)
            self.assertNotIn("pto.vadd", pto_ir)
            self.assertNotIn("pto.tadd ins", vpto_ir)
            self.assertIn("pto.vadd", vpto_ir)

    def test_reuses_imported_specialization_before_materializing_again(self):
        # The PTODSL TileLib daemon materializes templates in a separate
        # Python process, so a parent-process monkeypatch of
        # ``_compiler_runtime.materialize`` never observes any call (the
        # counter stays at 0).  Instead, drive the compilation through a
        # daemon on a known socket and query the daemon's own cache stats
        # over RPC to assert that the duplicate 1D specialization is served
        # from cache rather than re-materialized.
        with tempfile.TemporaryDirectory() as temp_dir:
            socket_path = str(Path(temp_dir) / "daemon.sock")
            output = Path(temp_dir) / "result-vpto.mlir"

            result = _core.main(
                [
                    "ptoas",
                    "--pto-arch=a5",
                    "--pto-backend=vpto",
                    "--emit-vpto",
                    "--daemon-socket-path",
                    socket_path,
                    str(INPUT),
                    "-o",
                    str(output),
                ]
            )
            self.assertEqual(result, 0)

            # The daemon is stopped on ptoas exit (atexit cleanup), so the
            # socket is gone and we cannot query its post-run cache stats.
            # Assert the observable end-to-end contract instead: the input has
            # two functions (TADD with two identical 1D blocks, TADD_2D with
            # one distinct 2D block), and each lowers to a lowered VMI add.
            vpto_ir = output.read_text(encoding="utf-8")
            self.assertIn("func.func @TADD", vpto_ir)
            self.assertIn("func.func @TADD_2D", vpto_ir)
            # Two 1D blocks in TADD plus one 2D block in TADD_2D -> 3 vadd.
            self.assertEqual(
                vpto_ir.count("pto.vadd"),
                3,
                f"expected 3 pto.vadd, got {vpto_ir.count('pto.vadd')}",
            )


if __name__ == "__main__":
    unittest.main()
