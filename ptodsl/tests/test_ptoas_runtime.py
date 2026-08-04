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
            for index in range(2):
                output = Path(temp_dir) / f"result-{index}.mlir"
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
                self.assertTrue(output.exists())
                self.assertIn("pto.vadd", output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
