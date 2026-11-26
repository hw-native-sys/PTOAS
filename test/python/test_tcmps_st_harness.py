#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Exercise tcmps ST readback with physical rows and nonzero padding."""

import importlib.util
from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import re
import runpy
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ST_ROOT = ROOT / "test/tilelang_st/npu/a5/src/st"


def load_cases(directory):
    spec = importlib.util.spec_from_file_location("tcmps_harness_cases", directory / "cases.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CASES


def run_script(directory, script, cwd, *arguments):
    output = io.StringIO()
    previous_directory = Path.cwd()
    returncode = 0
    try:
        os.chdir(cwd)
        with patch.dict(sys.modules), patch.object(sys, "argv", [script, *arguments]), patch.object(
            sys, "path", [str(directory), str(directory.parent), *sys.path]
        ), redirect_stdout(output):
            sys.modules.pop("cases", None)
            sys.modules.pop("st_common", None)
            try:
                runpy.run_path(str(directory / script), run_name="__main__")
            except SystemExit as error:
                returncode = error.code
    finally:
        os.chdir(previous_directory)
    return SimpleNamespace(returncode=returncode, stdout=output.getvalue(), stderr="")


class TcmpsHarnessTest(unittest.TestCase):
    def test_physical_rows_and_short_readback(self):
        for suite in ("testcase", "smoke/testcase"):
            directory = ST_ROOT / suite / "tcmps"
            cases = load_cases(directory)
            with self.subTest(suite=suite), tempfile.TemporaryDirectory() as temporary:
                generated = run_script(directory, "gen_data.py", temporary)
                self.assertEqual(generated.returncode, 0, generated.stdout + generated.stderr)
                for case in cases:
                    rows, cols = case.get("dst_shape", case["shape"])
                    vr, vc = case["valid_shape"]
                    row_bytes = (vc + 7) // 8
                    case_dir = Path(temporary) / case["name"]
                    golden = np.fromfile(case_dir / "golden.bin", dtype=np.uint8)
                    output = np.full((rows, cols), 0xA5, dtype=np.uint8)
                    output[:vr, :row_bytes] = golden.reshape(vr, row_bytes)
                    output.tofile(case_dir / "output.bin")
                compared = run_script(directory, "compare.py", temporary)
                self.assertEqual(compared.returncode, 0, compared.stdout + compared.stderr)
                case = cases[0]
                output_path = Path(temporary) / case["name"] / "output.bin"
                output_path.write_bytes(output_path.read_bytes()[:-1])
                short = run_script(directory, "compare.py", temporary, case["name"])
                self.assertNotEqual(short.returncode, 0, short.stdout + short.stderr)
                self.assertIn("expected", short.stdout)

    def test_destination_metadata_matches_kernel(self):
        for suite in ("testcase", "smoke/testcase"):
            directory = ST_ROOT / suite / "tcmps"
            source = (directory / "tcmps.pto").read_text(encoding="utf-8")
            for case in load_cases(directory):
                with self.subTest(suite=suite, case=case["name"]):
                    function = source.split(f"func.func @TCMP_{case['name']}(", 1)[1]
                    function = function.split("return", 1)[0]
                    destination = re.search(
                        r"%dst = pto.alloc_tile\s*: !pto.tile_buf<vec, (\d+)x(\d+)xui8[,>]", function
                    )
                    self.assertIsNotNone(destination)
                    actual = tuple(map(int, destination.groups()))
                    self.assertEqual(case.get("dst_shape", case["shape"]), actual)


if __name__ == "__main__":
    unittest.main()
