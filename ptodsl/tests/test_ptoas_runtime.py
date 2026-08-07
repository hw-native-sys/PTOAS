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
from unittest import mock

from ptoas import _core
from ptodsl.tilelib import _compiler_runtime


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
        calls = 0
        original_materialize = _compiler_runtime.materialize

        def counted_materialize(*args, **kwargs):
            nonlocal calls
            calls += 1
            return original_materialize(*args, **kwargs)

        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "result-vpto.mlir"
            with mock.patch.object(
                _compiler_runtime,
                "materialize",
                side_effect=counted_materialize,
            ):
                result = _core.main(
                    [
                        "ptoas",
                        "--pto-arch=a5",
                        "--pto-backend=vpto",
                        "--emit-vpto",
                        str(INPUT),
                        "-o",
                        str(output),
                    ]
                )

            self.assertEqual(result, 0)
            # The input has two identical 1D calls and one distinct 2D
            # fallback, so only the duplicate 1D specialization is reused.
            self.assertEqual(calls, 2)
            self.assertIn("pto.vadd", output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
