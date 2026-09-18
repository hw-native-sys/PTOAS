# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Exercise build.sh with real CMake/Ninja and two small translation units."""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


SOURCE = Path(sys.argv[1]).resolve()
BASH = shutil.which("bash")


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def prepare(root):
    source = (SOURCE / "build.sh").read_text(encoding="utf-8")
    write(root / "functions.sh", source.removesuffix('main "$@"\n'))
    write(root / "entry.sh", '''source ./functions.sh
ensure_llvm_build() { :; }
resolve_compiler_rt() { :; }
resolve_devtoolset_toolchain() { DEVTOOLSET_TOOLCHAIN_FLAGS=""; }
main "$@"
''')
    write(root / "CMakeLists.txt", '''cmake_minimum_required(VERSION 3.20)
project(incremental_fixture LANGUAGES CXX)
add_library(fixture STATIC first.cpp second.cpp)
install(TARGETS fixture DESTINATION lib)
''')
    write(root / "value.h", "#define FIXTURE_VALUE 1\n")
    write(root / "first.cpp", '#include "value.h"\nint first() { return FIXTURE_VALUE; }\n')
    write(root / "second.cpp", "int second() { return 2; }\n")
    launcher = root / "launcher"
    write(launcher, '''#!/usr/bin/env bash
set -e
printf '%s\\n' "$*" >> "$TEST_COMPILE_LOG"
exec "$@"
''')
    launcher.chmod(0o755)
    env = dict(os.environ)
    for name in ("SMOKE_TYPE", "ST_PART", "task_name", "PTOAS_WHEEL_FILE", "LLVM_BUILD_DIR",
                 "CMAKE_ARGS", "CMAKE_C_COMPILER_LAUNCHER", "CMAKE_CXX_COMPILER_LAUNCHER"):
        env.pop(name, None)
    env.update(PTOAS_CC=shutil.which("cc"), PTOAS_CXX=shutil.which("c++"),
               CMAKE_CXX_COMPILER_LAUNCHER=str(launcher), TEST_COMPILE_LOG=str(root / "compile.log"))
    return env


def build(root, env, *extra):
    result = subprocess.run([BASH, str(root / "entry.sh"), "--build", "-j", "2", *extra],
                            cwd=root, env=env, capture_output=True, text=True, timeout=60)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    return result.stdout


def objects(root):
    return {path.name: path.stat().st_mtime_ns for path in (root / "build").rglob("*.cpp.o")}


def check_incremental():
    with tempfile.TemporaryDirectory(prefix="ptoas incremental ") as directory:
        root = Path(directory)
        env = prepare(root)
        build(root, env)
        initial = objects(root)
        assert set(initial) == {"first.cpp.o", "second.cpp.o"}, initial
        calls = (root / "compile.log").read_text(encoding="utf-8")
        assert "first.cpp" in calls and "second.cpp" in calls
        build(root, env)
        assert objects(root) == initial, "Unchanged build recompiled objects"
        assert (root / "compile.log").read_text(encoding="utf-8") == calls
        write(root / "value.h", "#define FIXTURE_VALUE 3\n")
        build(root, env)
        updated = objects(root)
        assert updated["first.cpp.o"] != initial["first.cpp.o"]
        assert updated["second.cpp.o"] == initial["second.cpp.o"]
        print("PASS: launcher is used; unchanged build is a no-op; header edit rebuilds its consumer only")

        # A nested wheel tree must survive normal native configuration, while
        # explicit --clean clears it and rebuilds both native objects.
        marker = root / "build/wheel/preserved"
        write(marker, "wheel build state\n")
        build(root, env)
        assert marker.is_file() and objects(root) == updated
        llvm_marker = root / "third_party/lib_cache/preserved"
        write(llvm_marker, "shared dependency\n")
        build(root, env, "--clean")
        assert not marker.exists() and llvm_marker.is_file()
        cleaned = objects(root)
        assert all(cleaned[name] != updated[name] for name in updated)
        print("PASS: normal builds preserve wheel state; --clean rebuilds PTOAS without deleting LLVM")


check_incremental()
