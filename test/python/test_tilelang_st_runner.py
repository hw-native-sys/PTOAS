#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("tilelang_run_st", ROOT / "test/tilelang_st/script/run_st.py")
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class SimulatorRunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="tileop runner ")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        previous = Path.cwd()
        os.chdir(self.root)
        self.addCleanup(os.chdir, previous)
        self.binary = self.root / "test binary"
        self.binary.write_text(
            f"#!{sys.executable}\nimport json, sys\nfrom pathlib import Path\n"
            "Path('arguments.json').write_text(json.dumps(sys.argv[1:]))\n"
        )
        self.binary.chmod(0o700)
        self.msprof = self.root / "fake msprof"
        patcher = mock.patch.object(RUNNER.shutil, "which", return_value=str(self.msprof))
        patcher.start()
        self.addCleanup(patcher.stop)

    def profiler(self, *, execute=True, status=0, message=""):
        self.msprof.write_text(
            f"#!{sys.executable}\nimport subprocess, sys\n"
            + ("subprocess.run([sys.argv[-1]], check=False)\n" if execute else "")
            + f"print({message!r})\nsys.exit({status})\n"
        )
        self.msprof.chmod(0o700)

    def test_success_preserves_arguments_and_isolates_outputs(self):
        self.profiler()
        args = ["case with spaces", "literal $HOME"]
        for _ in range(2):
            RUNNER._run_simulator_binary([str(self.binary), *args], "Ascend950PR_9599")
        self.assertEqual(json.loads((self.root / "arguments.json").read_text()), args)
        outputs = list(self.root.glob("msprof-*"))
        self.assertEqual(len(outputs), 2)
        for output in outputs:
            self.assertEqual(output.stat().st_mode & 0o777, 0o700)
            self.assertEqual((output / "binary.exitcode").read_text().strip(), "0")

    def test_binary_failure_is_not_hidden_by_profiler_success(self):
        self.profiler()
        with self.binary.open("a") as handle:
            handle.write("sys.exit(7)\n")
        with self.assertRaises(subprocess.CalledProcessError) as error:
            RUNNER._run_simulator_binary([str(self.binary)], "Ascend950PR_9599")
        self.assertEqual(error.exception.returncode, 7)

    def test_missing_binary_status_is_failure(self):
        self.profiler(execute=False)
        with self.assertRaisesRegex(RuntimeError, "did not record"):
            RUNNER._run_simulator_binary([str(self.binary)], "Ascend950PR_9599")

    def test_profiler_failure_is_not_hidden_by_binary_success(self):
        self.profiler(status=9)
        with self.assertRaises(subprocess.CalledProcessError) as error:
            RUNNER._run_simulator_binary([str(self.binary)], "Ascend950PR_9599")
        self.assertEqual(error.exception.returncode, 9)

    def test_trace_export_failure_warns_and_allows_numerical_comparison(self):
        self.profiler(message="[ERROR] Profiling data parse failed. Please check")
        with mock.patch("builtins.print") as output:
            RUNNER._run_simulator_binary([str(self.binary)], "Ascend950PR_9599")
        self.assertTrue(any("[WARN]" in str(call) and "numerical comparison" in str(call)
                            for call in output.call_args_list))

    def test_npu_execution_does_not_use_simulator(self):
        directory = self.root / "build/testcase/tadd"
        directory.mkdir(parents=True)
        with mock.patch.object(RUNNER, "run_command") as direct:
            with mock.patch.object(RUNNER, "_run_simulator_binary") as simulator:
                RUNNER.run_binary("tadd", ["shape"], run_mode="npu")
        direct.assert_called_once_with([os.path.join("..", "..", "bin", "tadd"), "shape"])
        simulator.assert_not_called()


if __name__ == "__main__":
    unittest.main()
