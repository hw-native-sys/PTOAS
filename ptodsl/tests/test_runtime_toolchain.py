#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ptodsl._runtime import native_build, toolchain


class RuntimeToolchainTest(unittest.TestCase):
    def test_include_flag_skips_inaccessible_directory(self):
        flags = []
        inaccessible = Path("/inaccessible/include")
        error = PermissionError(13, "Permission denied", str(inaccessible))

        with mock.patch.object(Path, "is_dir", side_effect=error):
            toolchain._append_include_flag(flags, inaccessible)

        self.assertEqual(flags, [])

    def test_include_flag_adds_accessible_directory_once(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            include_dir = Path(temp_dir)
            flags = []

            toolchain._append_include_flag(flags, include_dir)
            toolchain._append_include_flag(flags, include_dir)

            self.assertEqual(flags, [f"-I{include_dir}"])

    def test_a2_a3_use_c220_aicore_arch(self):
        self.assertEqual(
            toolchain.aicore_arch_for_kernel_kind("vector", "a3"), "dav-c220-vec"
        )
        self.assertEqual(toolchain.aicore_arch_for_kernel_kind(None, "a3"), "dav-c220")
        self.assertEqual(
            toolchain.aicore_arch_for_kernel_kind("cube", "a3"), "dav-c220-cube"
        )
        self.assertEqual(
            toolchain.aicore_arch_for_kernel_kind("vector", "a2"), "dav-c220-vec"
        )

    def test_a5_uses_c310_aicore_arch(self):
        self.assertEqual(
            toolchain.aicore_arch_for_kernel_kind("vector", "a5"), "dav-c310-vec"
        )
        self.assertEqual(
            toolchain.aicore_arch_for_kernel_kind("cube", "a5"), "dav-c310-cube"
        )
        self.assertEqual(toolchain.aicore_arch_for_kernel_kind(None, "a5"), "dav-c310")

    def test_native_launch_flags_use_target_arch(self):
        with mock.patch.object(native_build, "common_include_flags", return_value=[]):
            self.assertIn(
                "--cce-aicore-arch=dav-c220-vec",
                native_build._kernel_compile_flags("vector", "a3"),
            )
            self.assertIn(
                "--cce-aicore-arch=dav-c310-cube",
                native_build._kernel_compile_flags("cube", "a5"),
            )


class ResolvePtoasBinaryTests(unittest.TestCase):
    def test_env_override_wins_over_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            env_ptoas = temp_root / "env-ptoas"
            env_ptoas.write_text("", encoding="utf-8")

            path_ptoas = temp_root / "path-ptoas"

            with mock.patch.dict(
                os.environ, {"PTOAS_BIN": str(env_ptoas)}, clear=True
            ), mock.patch.object(
                toolchain.shutil, "which", return_value=str(path_ptoas)
            ) as which:
                resolved = toolchain.resolve_ptoas_binary()

            self.assertEqual(resolved, env_ptoas)
            which.assert_not_called()

    def test_path_is_used_without_env_override(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path_ptoas = Path(temp_dir) / "path-ptoas"
            with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(
                toolchain.shutil, "which", return_value=str(path_ptoas)
            ) as which:
                resolved = toolchain.resolve_ptoas_binary()

            self.assertEqual(resolved, path_ptoas)
            which.assert_called_once_with("ptoas")

    def test_invalid_env_override_raises(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            missing = Path(temp_dir) / "missing-ptoas"
            with mock.patch.dict(
                os.environ, {"PTOAS_BIN": str(missing)}, clear=True
            ), mock.patch.object(toolchain.shutil, "which", return_value=None):
                with self.assertRaises(FileNotFoundError):
                    toolchain.resolve_ptoas_binary()


if __name__ == "__main__":
    unittest.main()
