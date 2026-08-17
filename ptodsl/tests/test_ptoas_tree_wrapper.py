#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
WRAPPER_SOURCE = REPO_ROOT / "tools" / "ptoas" / "ptoas_wrapper.py"

FAKE_CLI = """
calls = []

def launch(user_args, *, wrapper=None):
    calls.append((list(user_args), wrapper))
    return 17
""".lstrip()


class TreeWrapperTests(unittest.TestCase):
    def test_wrapper_disables_editable_import_redirects(self):
        class EditableFinder:
            pass

        editable_finder = EditableFinder()
        regular_finder = object()
        saved_meta_path = list(sys.meta_path)
        try:
            sys.meta_path[:] = [editable_finder, regular_finder]
            module = self._load_wrapper(
                Path("/tmp/build/tools/ptoas/ptoas"),
                "test_ptoas_editable_finder_filter",
                python_root_mode="absolute",
                python_root=Path("/tmp/build/python"),
            )
            module._disable_editable_import_redirects()
            self.assertEqual(sys.meta_path, [regular_finder])
        finally:
            sys.meta_path[:] = saved_meta_path

    def _load_wrapper(
        self,
        wrapper_path: Path,
        module_name: str,
        *,
        python_root_mode: str,
        python_root: Path,
    ):
        spec = importlib.util.spec_from_file_location(module_name, WRAPPER_SOURCE)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.__file__ = str(wrapper_path)
        module._PYTHON_ROOT_MODE = python_root_mode
        module._PYTHON_ROOT = python_root
        return module

    def _run_wrapper(self, wrapper_module, python_root: Path, wrapper: Path):
        saved_sys_path = list(sys.path)
        saved_argv = list(sys.argv)
        saved_modules = {
            name: sys.modules.pop(name, None)
            for name in ("ptoas", "ptoas._cli")
        }
        try:
            sys.path = [entry for entry in sys.path if entry != str(python_root)]
            sys.argv = [str(wrapper), "--version"]
            with self.assertRaises(SystemExit) as exc:
                wrapper_module.main()
            self.assertEqual(exc.exception.code, 17)
            return sys.modules["ptoas._cli"]
        finally:
            sys.path = saved_sys_path
            sys.argv = saved_argv
            for name in ("ptoas._cli", "ptoas"):
                sys.modules.pop(name, None)
                if saved_modules[name] is not None:
                    sys.modules[name] = saved_modules[name]

    def _make_package(self, python_root: Path) -> None:
        package_root = python_root / "ptoas"
        package_root.mkdir(parents=True)
        (package_root / "__init__.py").write_text("", encoding="utf-8")
        (package_root / "_cli.py").write_text(FAKE_CLI, encoding="utf-8")

    def test_build_tree_wrapper_delegates_to_cli_module(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            build_root = Path(temp_dir) / "build"
            python_root = build_root / "python"
            wrapper = build_root / "tools" / "ptoas" / "ptoas"
            self._make_package(python_root)
            wrapper.parent.mkdir(parents=True)
            wrapper.write_text("", encoding="utf-8")

            module = self._load_wrapper(
                wrapper,
                "test_ptoas_build_wrapper",
                python_root_mode="absolute",
                python_root=python_root,
            )
            cli = self._run_wrapper(module, python_root, wrapper)

        self.assertEqual(cli.calls, [(["--version"], wrapper.resolve())])

    def test_install_tree_wrapper_delegates_to_cli_module(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            install_root = Path(temp_dir) / "install"
            wrapper = install_root / "bin" / "ptoas"
            self._make_package(install_root)
            wrapper.parent.mkdir(parents=True)
            wrapper.write_text("", encoding="utf-8")

            module = self._load_wrapper(
                wrapper,
                "test_ptoas_install_wrapper",
                python_root_mode="wrapper-relative",
                python_root=Path(".."),
            )
            cli = self._run_wrapper(module, install_root, wrapper)

        self.assertEqual(cli.calls, [(["--version"], wrapper.resolve())])

    def test_wrapper_requires_cli_module(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            python_root = Path(temp_dir) / "python"
            python_root.mkdir()
            module = self._load_wrapper(
                Path(temp_dir) / "build" / "tools" / "ptoas" / "ptoas",
                "test_ptoas_missing_cli",
                python_root_mode="absolute",
                python_root=python_root,
            )
            with self.assertRaisesRegex(SystemExit, "ptoas/_cli.py"):
                module._require_python_root(python_root, context="test")

    def test_archive_wrapper_rejects_a_different_python_minor(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            python_root = Path(temp_dir) / "archive"
            wrapper = python_root / "bin" / "ptoas"
            self._make_package(python_root)
            wrapper.parent.mkdir(parents=True)
            wrapper.write_text("", encoding="utf-8")
            (python_root / ".ptoas-python-version").write_text(
                "0.0\n", encoding="utf-8"
            )

            module = self._load_wrapper(
                wrapper,
                "test_ptoas_archive_python_version",
                python_root_mode="wrapper-relative",
                python_root=Path(".."),
            )
            saved_argv = list(sys.argv)
            try:
                sys.argv = [str(wrapper)]
                with self.assertRaisesRegex(SystemExit, "requires CPython 0.0"):
                    module.main()
            finally:
                sys.argv = saved_argv


if __name__ == "__main__":
    unittest.main()
