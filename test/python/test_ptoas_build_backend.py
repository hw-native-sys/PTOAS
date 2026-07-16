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

import _ptoas_build_backend as build_backend


class PtoasBuildBackendTests(unittest.TestCase):
    def test_cmake_build_driver_is_used_for_build_and_install(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            repo = temp_root / "repo"
            build_dir = temp_root / "build"
            llvm_build = temp_root / "llvm-build"
            install_dir = temp_root / "install"
            (repo / "cmake").mkdir(parents=True)

            check_call_args = []

            def fake_check_call(args):
                check_call_args.append(args)

            with mock.patch.object(build_backend, "_REPO", repo), mock.patch.object(
                build_backend, "_BUILD_DIR", build_dir
            ), mock.patch.object(
                build_backend, "_LLVM_BUILD_DIR", llvm_build
            ), mock.patch.object(
                build_backend, "_PTO_INSTALL_DIR", install_dir
            ), mock.patch.object(
                build_backend,
                "_should_use_linux_hardening_cache",
                return_value=False,
            ), mock.patch.object(
                build_backend.subprocess,
                "check_output",
                return_value="/tmp/pybind11-cmake",
            ), mock.patch.object(
                build_backend.subprocess,
                "check_call",
                side_effect=fake_check_call,
            ), mock.patch.object(
                build_backend,
                "_assert_installed_ptodsl_payload",
            ):
                build_backend._cmake_configure_and_build()

        self.assertEqual(
            check_call_args[-2],
            ["cmake", "--build", str(build_dir)],
        )
        self.assertEqual(
            check_call_args[-1],
            ["cmake", "--build", str(build_dir), "--target", "install"],
        )

    def test_linux_hardening_cache_keeps_x86_64_baseline(self):
        cache_path = (
            Path(__file__).resolve().parents[2] / "cmake" / "LinuxHardeningCache.cmake"
        )
        cache_text = cache_path.read_text(encoding="utf-8")
        self.assertIn("uname -m", cache_text)
        self.assertIn('MATCHES "^(x86_64|amd64|i[3-6]86)$"', cache_text)
        self.assertIn("-march=x86-64", cache_text)
        self.assertIn("-mtune=generic", cache_text)


if __name__ == "__main__":
    unittest.main()
