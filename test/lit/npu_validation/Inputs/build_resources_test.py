# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Check LLVM parallelism and the limit on every PTOAS build stage."""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

BASH = shutil.which("bash")
SOURCE = Path(sys.argv[1]).resolve()


def check_jobs(ncpu, requested, expected):
    with tempfile.TemporaryDirectory(prefix="ptoas build jobs ") as directory:
        root = Path(directory)
        source = (SOURCE / "build.sh").read_text(encoding="utf-8")
        (root / "functions.sh").write_text(source.removesuffix('main "$@"\n'), encoding="utf-8")
        script = root / "check.sh"
        script.write_text('''source ./functions.sh
nproc() { echo "$TEST_NCPU"; }
resolve_build_jobs
[[ "$JOBS" == "$TEST_EXPECTED" ]]
[[ "$CMAKE_BUILD_PARALLEL_LEVEL" == "$TEST_EXPECTED" ]]
''', encoding="utf-8")
        env = dict(os.environ, TEST_NCPU=str(ncpu), JOBS=requested, TEST_EXPECTED=str(expected))
        result = subprocess.run([BASH, str(script)], cwd=root, env=env, text=True,
                                capture_output=True, timeout=10)
        if expected is None:
            assert result.returncode != 0 and "positive integer" in result.stderr, result
        else:
            assert result.returncode == 0, result.stdout + result.stderr
        print(f"PASS: ncpu={ncpu}, requested={requested!r}, jobs={expected}")


PIPELINE_CHECK = r'''source ./functions.sh
nproc() { echo 128; }
ensure_llvm_build() { echo "llvm:$JOBS:$CMAKE_BUILD_PARALLEL_LEVEL" >> "$TEST_LOG"; }
configure_ptoas() { mkdir -p "$BUILD_PATH"; }
pip_install_runtime_deps() { :; }
resolve_compiler_rt() { echo "compiler-rt:$JOBS:$CMAKE_BUILD_PARALLEL_LEVEL" >> "$TEST_LOG"; }
patchelf() { :; }
cmake() {
  if [[ "$1" == --build ]]; then
    echo "native:${@: -1}" >> "$TEST_LOG"
  fi
}
python3() {
  if [[ "$1" == -c ]]; then
    [[ "$2" != *sysconfig* ]] || echo "$BASE_PATH"
    return 0
  fi
  [[ "$1" == -m ]] || return 0
  if [[ "$2" == pip && "$*" == *--help* ]]; then
    return 1
  fi
  if [[ "$2" == pip ]]; then
    echo "wheel:$CMAKE_BUILD_PARALLEL_LEVEL" >> "$TEST_LOG"
  fi
  while [[ $# -gt 0 ]]; do
    if [[ "$1" == --wheel-dir ]]; then
      mkdir -p "$2"
      touch "$2/ptoas-test.whl"
      return 0
    fi
    shift
  done
  return 1
}
resolve_build_jobs
DEVTOOLSET_TOOLCHAIN_FLAGS="$TEST_TOOLCHAIN_FLAGS"
"$TEST_ENTRY"
echo "after:$JOBS:$CMAKE_BUILD_PARALLEL_LEVEL" >> "$TEST_LOG"
'''


def check_pipeline(entry, requested, expected_llvm, expected_ptoas, toolchain_flags=""):
    with tempfile.TemporaryDirectory(prefix="ptoas pipeline jobs ") as directory:
        root = Path(directory)
        source = (SOURCE / "build.sh").read_text(encoding="utf-8")
        (root / "functions.sh").write_text(source.removesuffix('main "$@"\n'), encoding="utf-8")
        script = root / "check.sh"
        script.write_text(PIPELINE_CHECK, encoding="utf-8")
        log = root / "calls.log"
        env = dict(os.environ, JOBS=requested, TEST_ENTRY=entry, TEST_LOG=str(log),
                   TEST_TOOLCHAIN_FLAGS=toolchain_flags)
        result = subprocess.run([BASH, str(script)], cwd=root, env=env, text=True,
                                capture_output=True, timeout=10)
        assert result.returncode == 0, result.stdout + result.stderr
        expected = [f"llvm:{expected_llvm}:{expected_llvm}", f"native:{expected_ptoas}"]
        if entry == "package":
            if toolchain_flags:
                expected.append(f"compiler-rt:{expected_llvm}:{expected_llvm}")
            expected.extend([f"wheel:{expected_ptoas}"] + [f"native:{expected_ptoas}"] * 2)
        expected.append(f"after:{expected_llvm}:{expected_llvm}")
        actual = log.read_text(encoding="utf-8").splitlines()
        assert actual == expected, f"{entry}: expected {expected}, got {actual}"
        print(f"PASS: {entry}, LLVM jobs={expected_llvm}, PTOAS jobs={expected_ptoas}, "
              f"devtoolset={bool(toolchain_flags)}")


# The default follows nproc with no cap: 128 CPUs yield 128 jobs.
# An explicit -j is honored verbatim, above or below the CPU count.
for ncpu, requested, expected in [(128, "", 128), (32, "", 32),
                                  (128, "64", 64), (16, "64", 64),
                                  (128, "4", 4)]:
    check_jobs(ncpu, requested, expected)
for invalid in ("0", "-1", "hello", "01", "99999999999999999999"):
    check_jobs(64, invalid, None)

for requested, llvm_jobs, ptoas_jobs in [("", 128, 16), ("64", 64, 16), ("4", 4, 4)]:
    check_pipeline("build_only", requested, llvm_jobs, ptoas_jobs)
    for flags in ("", "--sysroot=/test/devtoolset-7"):
        check_pipeline("package", requested, llvm_jobs, ptoas_jobs, flags)
