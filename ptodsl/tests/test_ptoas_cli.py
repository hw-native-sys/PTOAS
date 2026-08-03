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
from types import SimpleNamespace
from unittest import mock

from ptoas import _cli


class PTOASCLITests(unittest.TestCase):
    def _make_native_module(self, package_root: Path):
        return SimpleNamespace(
            __file__=str(package_root / "_core.fake.so"),
            main=mock.Mock(return_value=0),
        )

    def test_launch_uses_standard_native_module_and_packaged_resources(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            package_root = Path(temp_dir) / "site-packages" / "ptoas"
            tileops_dir = package_root / "_runtime" / "share" / "ptoas" / "TileOps"
            wrapper = Path(temp_dir) / "bin" / "ptoas"
            tileops_dir.mkdir(parents=True)
            wrapper.parent.mkdir(parents=True)
            wrapper.write_text("", encoding="utf-8")
            native_module = self._make_native_module(package_root)

            with mock.patch.object(
                _cli, "_load_native_module", return_value=native_module
            ), mock.patch.dict(
                _cli.os.environ,
                {"PATH": "/usr/bin"},
                clear=True,
            ):
                exit_code = _cli.launch(["--version"], wrapper=wrapper)
                environment = dict(_cli.os.environ)

        self.assertEqual(exit_code, 0)
        native_module.main.assert_called_once_with(
            [str(wrapper.resolve()), "--version"]
        )
        self.assertEqual(environment["PTOAS_BIN"], str(wrapper.resolve()))
        self.assertNotIn("PTOAS_PYTHON_EXE", environment)
        self.assertEqual(environment["PATH"], "/usr/bin")

    def test_user_arguments_are_forwarded_unchanged(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            package_root = Path(temp_dir) / "install" / "ptoas"
            (package_root / "_runtime" / "share" / "ptoas" / "TileOps").mkdir(
                parents=True
            )
            wrapper = Path(temp_dir) / "install" / "bin" / "ptoas"
            wrapper.parent.mkdir(parents=True)
            wrapper.write_text("", encoding="utf-8")
            native_module = self._make_native_module(package_root)
            arguments = [
                "--pto-arch=a5",
                "--version",
            ]

            with mock.patch.object(
                _cli, "_load_native_module", return_value=native_module
            ):
                _cli.launch(arguments, wrapper=wrapper)

        native_module.main.assert_called_once_with(
            [str(wrapper.resolve()), *arguments]
        )

    def test_build_tree_uses_the_same_packaged_resource_layout(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            build_root = Path(temp_dir) / "build"
            package_root = build_root / "python" / "ptoas"
            tileops_dir = package_root / "_runtime" / "share" / "ptoas" / "TileOps"
            tileops_dir.mkdir(parents=True)
            native_module = self._make_native_module(package_root)

            resolved_tileops = _cli._resolve_tileops_dir(native_module)

        self.assertEqual(resolved_tileops, tileops_dir.resolve())

    def test_missing_tileops_resources_is_an_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            package_root = Path(temp_dir) / "site-packages" / "ptoas"
            native_module = self._make_native_module(package_root)

            with self.assertRaisesRegex(SystemExit, "TileOps"):
                _cli._resolve_tileops_dir(native_module)


if __name__ == "__main__":
    unittest.main()
