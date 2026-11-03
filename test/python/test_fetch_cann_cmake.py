#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import shutil
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CMAKE = shutil.which("cmake")
GIT = shutil.which("git")


@unittest.skipUnless(CMAKE, "cmake is required")
class FetchCannCmakeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="cann bootstrap ")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.fixture = self.root / "dependency"
        (self.fixture / "function").mkdir(parents=True)
        (self.fixture / "function/prepare.cmake").write_text("set(CANN_BOOTSTRAP_LOADED TRUE)\n")
        self.source = self.root / "source"
        self.source.mkdir()
        (self.source / "CMakeLists.txt").write_text(
            "cmake_minimum_required(VERSION 3.20)\n"
            f'include("{ROOT / "cmake/fetch_cann_cmake.cmake"}")\n'
            'if(NOT CANN_BOOTSTRAP_LOADED)\nmessage(FATAL_ERROR "prepare.cmake was not loaded")\nendif()\n'
            "project(bootstrap_test NONE)\n"
        )

    def configure(self, build, *args):
        result = subprocess.run(
            [CMAKE, "-S", str(self.source), "-B", str(build), "-Werror=dev", *args],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertTrue((build / "cann-3rd-lib/cann-cmake/function/prepare.cmake").is_file())

    def test_archive_bootstrap_and_reconfigure(self):
        build = self.root / "archive-build"
        dependency_root = build / "cann-3rd-lib"
        dependency_root.mkdir(parents=True)
        with tarfile.open(dependency_root / "cmake-master-055.tar.gz", "w:gz") as archive:
            archive.add(self.fixture, arcname="cann-cmake")
        self.configure(build)
        self.configure(build)

    @unittest.skipUnless(GIT, "git is required")
    def test_git_bootstrap_and_reconfigure(self):
        for command in (
            [GIT, "init", "-q"],
            [GIT, "add", "."],
            [GIT, "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "fixture"],
            [GIT, "tag", "fixture"],
        ):
            subprocess.run(command, cwd=self.fixture, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        build = self.root / "git-build"
        args = (f"-DCANN_CMAKE_GIT_URL={self.fixture.as_uri()}", "-DCANN_CMAKE_GIT_REF=fixture")
        self.configure(build, *args)
        self.configure(build, *args)


if __name__ == "__main__":
    unittest.main()
