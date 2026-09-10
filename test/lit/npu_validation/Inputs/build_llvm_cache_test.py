# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Check LLVM cache reuse and rejection without building LLVM or PTOAS."""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

BASH = shutil.which("bash")
SOURCE = Path(sys.argv[1]).resolve()
CACHE_TEXT = """LLVM_BSPUB_NPU_DATA_TYPE:BOOL=ON
CMAKE_C_FLAGS:STRING=-DBSPUB_NPU_DATA_TYPE
CMAKE_CXX_FLAGS:STRING=-DBSPUB_NPU_DATA_TYPE -D_GLIBCXX_USE_CXX11_ABI=0
BUILD_SHARED_LIBS:BOOL=ON
LLVM_BUILD_LLVM_DYLIB:BOOL=OFF
LLVM_LINK_LLVM_DYLIB:BOOL=OFF
LLVM_LLVMVectorize_LINKER_FLAGS:STRING=-lLLVMTargetParser
"""


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def executable(path, body):
    write(path, f"#!{BASH}\nset -e\n{body}\n")
    path.chmod(0o755)


def seed_cache(cache):
    for name in ("lib/cmake/llvm/LLVMConfig.cmake", "lib/cmake/mlir/MLIRConfig.cmake",
                 "lib/libLLVMSupport.so.19.1", "lib/libLLVMVectorize.so.19.1"):
        write(cache / name, "fixture\n")
    write(cache / "CMakeCache.txt", CACHE_TEXT)
    write(cache / "include/llvm/IR/CallingConv.h", "SimtEntry\n")
    write(cache / "preserved", "must survive cache lookup\n")


def check_case(keyed=False, invalid=None):
    with tempfile.TemporaryDirectory(prefix="ptoas llvm cache ") as directory:
        root = Path(directory)
        text = (SOURCE / "build.sh").read_text()
        assert text.endswith('main "$@"\n')
        write(root / "build-functions.sh", text.removesuffix('main "$@"\n'))
        default = root / "cache/lib_cache/llvm_19.1.7/build-shared"
        cache = Path(str(default) + "-ptoas-abi0-bspub-shared") if keyed else default
        seed_cache(cache)
        write(root / "cache/llvm-19/llvm/CMakeLists.txt", "# cached source\n")
        write(root / "cache/llvm-19/llvm/include/llvm/IR/CallingConv.h", "SimtEntry\n")
        env = dict(os.environ)
        for key in ("LLVM_BUILD_DIR", "PTOAS_LLVM_BUILD_DIR_EXPLICIT", "ASCEND_3RD_LIB_PATH"):
            env.pop(key, None)
        env.update(PATH=str(root / "tools") + os.pathsep + env.get("PATH", ""), PTOAS_GLIBCXX_ABI="0")
        # Emit far more than a pipe buffer after the match. grep -q closes its
        # input early; pipefail must not mistake that SIGPIPE for a cache miss.
        symbols = "_ZNK4llvm5Twine3strB5cxx11Ev" if invalid == "abi" else "_ZNK4llvm5Twine3strEv"
        dependency = "libOther.so" if invalid == "dependency" else "libLLVMTargetParser.so.19.1"
        for tool, first_line in (("nm", symbols), ("readelf", dependency)):
            status = 9 if invalid == tool + "-error" else 0
            executable(root / "tools" / tool,
                       f"printf '%s\\n' '{first_line}'\nprintf '%02000d\\n' {{1..1000}}\nexit {status}")
        for tool in ("cmake", "ninja"):
            executable(root / "tools" / tool, "echo 'Unexpected LLVM build' >&2; exit 97")
        if invalid == "bspub":
            write(cache / "CMakeCache.txt", CACHE_TEXT.replace("-DBSPUB_NPU_DATA_TYPE", ""))
        if invalid == "cmake":
            (cache / "lib/cmake/mlir/MLIRConfig.cmake").unlink()
        if invalid == "simt":
            write(cache / "include/llvm/IR/CallingConv.h", "UpstreamOnly\n")
        if invalid == "shared":
            write(cache / "CMakeCache.txt",
                  CACHE_TEXT.replace("BUILD_SHARED_LIBS:BOOL=ON", "BUILD_SHARED_LIBS:BOOL=OFF"))
        if invalid == "linker":
            write(cache / "CMakeCache.txt", CACHE_TEXT.replace("-lLLVMTargetParser", ""))
        check = ('LLVM_BUILD_DIR="$TEST_CACHE"; if llvm_build_cache_is_usable; then exit 1; fi'
                 if invalid else 'ensure_llvm_build')
        write(root / "check.sh", 'source ./build-functions.sh\nCANN_3RD_LIB_PATH="${BASE_PATH}/cache"\n'
              'prepare_llvm_cache_layout\n' + check + '\n[[ "$LLVM_BUILD_DIR" == "$TEST_CACHE" ]]\n')
        env["TEST_CACHE"] = str(cache)
        result = subprocess.run([BASH, str(root / "check.sh")], cwd=root, env=env,
                                text=True, capture_output=True, timeout=20)
        output = result.stdout + result.stderr
        assert result.returncode == 0, output
        assert (cache / "preserved").read_text() == "must survive cache lookup\n"
        if invalid:
            assert "LLVM/MLIR cache miss at" in output, output
        else:
            assert f"Reusing cached LLVM/MLIR build at {cache}" in output, output
            assert "Building LLVM/MLIR" not in output, output
        print(f"PASS: LLVM cache keyed={keyed}, invalid={invalid}")


check_case()
check_case(keyed=True)
for invalid_case in ("abi", "dependency", "nm-error", "readelf-error", "bspub", "cmake", "simt", "shared", "linker"):
    check_case(invalid=invalid_case)
