#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import subprocess
import sys
import tempfile
from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]


class ReleaseVersionScriptTests(unittest.TestCase):
    def test_ptoas_version_script_accepts_ptoas_tag_prefix(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            cmake_file = temp_root / "CMakeLists.txt"
            cmake_file.write_text("project(ptoas VERSION 0.51)\n", encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / ".github/scripts/compute_ptoas_version.py"),
                    "--cmake-file",
                    str(cmake_file),
                    "--check-tag",
                    "ptoas-v0.51",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.stdout.strip(), "0.51")

    def test_vmi_version_script_accepts_vmi_tag_prefix(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            version_file = temp_root / "VMI_VERSION"
            version_file.write_text("0.1.0\n", encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / ".github/scripts/compute_vmi_version.py"),
                    "--version-file",
                    str(version_file),
                    "--check-tag",
                    "vmi-v0.1.0",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.stdout.strip(), "0.1.0")

    def test_ptoas_version_bump_rejects_vmi_tag(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            cmake_file = temp_root / "CMakeLists.txt"
            cmake_file.write_text("project(ptoas VERSION 0.51)\n", encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / ".github/scripts/update_ptoas_base_version.py"),
                    "--cmake-file",
                    str(cmake_file),
                    "--version",
                    "vmi-v0.1.0",
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
