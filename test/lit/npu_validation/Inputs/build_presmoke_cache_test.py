# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Exercise build.sh's PreSmoke package handoff without compiling LLVM or PTOAS."""

import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

BASH = shutil.which("bash")
SOURCE = Path(sys.argv[1]).resolve()
ARCH = {"arm64": "aarch64", "amd64": "x86_64"}.get(platform.machine(), platform.machine())
PACKAGE_NAME = f"cann-pto-as_linux-{ARCH}.run"
PAYLOAD = "#!/usr/bin/env bash\nprintf 'cached installer invoked\\n'\n"


def run(command, root, env, expect_success=True):
    result = subprocess.run(command, cwd=root, env=env, text=True, capture_output=True, timeout=20)
    if (result.returncode == 0) != expect_success:
        raise RuntimeError(f"Unexpected exit {result.returncode}: {result.stdout}{result.stderr}")
    return result.stdout + result.stderr


def prepare(root):
    shutil.copyfile(SOURCE / "build.sh", root / "build.sh")
    samples = root / "test/samples"
    samples.mkdir(parents=True)
    shutil.copyfile(SOURCE / "test/samples/runop.sh", samples / "runop.sh")
    board_script = root / "test/npu_validation/scripts/run_remote_npu_validation.sh"
    board_script.parent.mkdir(parents=True)
    shutil.copyfile(SOURCE / "test/npu_validation/scripts/run_remote_npu_validation.sh", board_script)
    tools = root / "guard-tools"
    tools.mkdir()
    for name in ("cmake", "clang", "clang++", "gcc", "g++", "ninja"):
        guard = tools / name
        guard.write_text(f"#!{BASH}\nprintf '%s\\n' build-reached > \"$GUARD_LOG\"\nexit 97\n")
        guard.chmod(0o755)
    env = dict(os.environ)
    for key in ("SMOKE_TYPE", "ST_PART", "download_path", "ASCEND_3RD_LIB_PATH", "LLVM_BUILD_DIR",
                "PTOAS_PRESMOKE_SKIP_RUNOP_MARKER", "PTOAS_CXX", "PTOAS_CC", "PACKAGE_TYPE", "obs_path", "JOBS",
                "GIT_PR_NUMBER", "pr_id", "task_name"):
        env.pop(key, None)
    env.update(PATH=str(tools) + os.pathsep + env.get("PATH", ""), GUARD_LOG=str(root / "guard.log"),
               PTOAS_CC=str(tools / "clang"), PTOAS_CXX=str(tools / "clang++"),
               PTOAS_PRESMOKE_LOG_DIR=str(root / "ascend/log"))
    return env


def check_case(location, smoke_key="SMOKE_TYPE", mode="--pkg", valid=True, normal=False):
    with tempfile.TemporaryDirectory(prefix="ptoas presmoke ") as directory:
        root = Path(directory)
        env = prepare(root)
        cache = root / "cache/lib_cache/ptoas-presmoke" / ARCH
        folders = {"output": root / "build_out", "workspace": root, "download": root / "download", "cache": cache}
        package_dir = folders[location]
        package_dir.mkdir(parents=True, exist_ok=True)
        package = package_dir / PACKAGE_NAME
        package.write_text(PAYLOAD if valid else "")
        if location == "download":
            env["download_path"] = str(package_dir)
        marker = root / "build/.skip-presmoke-runop"
        marker.parent.mkdir()
        if normal:
            marker.touch()
            env.update(task_name="Compile_Ascend_ARM", GIT_PR_NUMBER="48")
        else:
            env[smoke_key] = "1" if smoke_key == "ST_PART" else "pre"
        hit = valid and not normal
        output = run([BASH, "build.sh", mode, "--cann_3rd_lib_path", str(root / "cache")], root, env, hit)
        if hit:
            assert "skipping LLVM/PTOAS build and packaging" in output
            assert "execute samples success (skipped in PreSmoke" in output
            assert not (root / "guard.log").exists()
            assert (root / "build_out" / PACKAGE_NAME).read_text() == PAYLOAD
            assert marker.is_file()
            # A new child shell receives no build.sh exports, just as in PreSmoke.
            smoke = run([BASH, str(root / "test/samples/runop.sh"), "all"], root, env)
            assert "PreSmoke runop smoke skipped" in smoke and "OK=0  FAIL=0  SKIP=0" in smoke
            board = run([BASH, str(root / "test/npu_validation/scripts/run_remote_npu_validation.sh")], root, env)
            assert "Skipping remote NPU validation in PreSmoke" in board
        else:
            assert (root / "guard.log").is_file() == normal, output
            assert "execute samples success" not in output
            assert marker.exists() != normal
        print(f"PASS: {location}, {smoke_key}, {mode}, valid={valid}, normal={normal}")


def check_cache_refresh():
    with tempfile.TemporaryDirectory(prefix="ptoas cache refresh ") as directory:
        root = Path(directory)
        env = prepare(root)
        # Source the actual functions, then model a successful package operation.
        # main() remains real, including cache publication and marker restoration.
        text = (root / "build.sh").read_text()
        assert text.endswith('main "$@"\n')
        (root / "build-functions.sh").write_text(text.removesuffix('main "$@"\n'))
        script = root / "refresh.sh"
        script.write_text('''source ./build-functions.sh
resolve_devtoolset_toolchain() { :; }
resolve_ptoas_toolchain() { :; }
package() {
  rm -rf "${BUILD_PATH}"
  mkdir -p "${BUILD_OUT_PATH}"
  printf '#!/usr/bin/env bash\\nprintf "fresh installer\\n"\\n' > "${BUILD_OUT_PATH}/${TEST_PACKAGE_NAME}"
}
main --pkg --cann_3rd_lib_path "${BASE_PATH}/cache"
''')
        env["TEST_PACKAGE_NAME"] = PACKAGE_NAME
        run([BASH, str(script)], root, env)
        cache = root / "cache/lib_cache/ptoas-presmoke" / ARCH
        assert (cache / PACKAGE_NAME).read_text() == (root / "build_out" / PACKAGE_NAME).read_text()
        assert not list(cache.glob(".package.*"))
        shutil.rmtree(root / "build_out")
        env["ST_PART"] = "1"
        output = run([BASH, "build.sh", "--pkg", "--cann_3rd_lib_path", str(root / "cache")], root, env)
        assert "skipping LLVM/PTOAS build and packaging" in output
        assert (root / "build/.skip-presmoke-runop").exists()
        assert not (root / "guard.log").exists()
        print("PASS: successful package refreshes persistent cache for the next PreSmoke")


def check_invalid_packages():
    with tempfile.TemporaryDirectory(prefix="ptoas invalid cache ") as directory:
        root = Path(directory)
        env = prepare(root)
        output_dir = root / "build_out"
        output_dir.mkdir()
        (output_dir / "cann-pto-as_linux-wrongarch.run").write_text(PAYLOAD)
        env["ST_PART"] = "1"
        command = [BASH, "build.sh", "--pkg", "--cann_3rd_lib_path", str(root / "cache")]
        run(command, root, env, False)
        assert not (root / "guard.log").exists()
        (output_dir / PACKAGE_NAME).write_text(PAYLOAD)
        run(command, root, env, False)
        assert not (root / "guard.log").exists()
        print("PASS: wrong-architecture and ambiguous packages fail without a source build")


def check_toolchain_pipefail():
    with tempfile.TemporaryDirectory(prefix="ptoas toolchain probe ") as directory:
        root = Path(directory)
        env = prepare(root)
        text = (root / "build.sh").read_text()
        (root / "build-functions.sh").write_text(text.removesuffix('main "$@"\n'))
        ldd = root / "guard-tools/ldd"
        ldd.write_text(f"#!{BASH}\nprintf 'ldd (GNU libc) 2.35\\n'\nprintf '%02000d\\n' {{1..1000}}\n")
        ldd.chmod(0o755)
        script = root / "probe.sh"
        script.write_text('''source ./build-functions.sh
devtoolset7_tree_is_usable() { return 0; }
resolve_devtoolset_toolchain
[[ "${DEVTOOLSET_TOOLCHAIN_FLAGS}" == *--sysroot=* ]]
''')
        run([BASH, str(script)], root, env)
        print("PASS: verbose ldd output cannot terminate toolchain resolution with SIGPIPE")


def check_ci_download(ci, failure=None, mode="--pkg"):
    with tempfile.TemporaryDirectory(prefix="ptoas ci package ") as directory:
        root = Path(directory)
        env = prepare(root)
        variants = {
            "actions": {"ST_PART": "1", "obs_path": "CANN/pto-as/ci/package/48"},
            "robot": {"SMOKE_TYPE": "pre", "GIT_PR_NUMBER": "48"},
            "robot_task": {"task_name": "PreSmoke_A900", "GIT_PR_NUMBER": "48"},
            "robot_pr_id": {"SMOKE_TYPE": "pre", "pr_id": "48"},
            "robot_obs": {"SMOKE_TYPE": "pre", "obs_path": "pto-as/package/48"},
        }
        env.update(variants[ci], TEST_PAYLOAD=PAYLOAD, DOWNLOAD_LOG=str(root / "download.log"),
                   DOWNLOAD_FAILURE=failure or "")
        if failure == "path":
            env["obs_path"] = "pto-as/package/../../other"
        elif failure == "missing_id":
            env.pop("GIT_PR_NUMBER", None)
            env.pop("obs_path", None)
        elif failure == "invalid_id":
            env["GIT_PR_NUMBER"] = "48/../../other"
        curl = root / "guard-tools/curl"
        curl.write_text(f"#!{sys.executable}\n" + '''import os
import sys
from pathlib import Path
Path(os.environ["DOWNLOAD_LOG"]).write_text("\\n".join(sys.argv[1:]))
output = Path(sys.argv[sys.argv.index("--output") + 1])
failure = os.environ["DOWNLOAD_FAILURE"]
output.write_text("<html>not an installer</html>\\n" if failure == "html" else os.environ["TEST_PAYLOAD"])
sys.exit(22 if failure == "http" else 0)
''')
        curl.chmod(0o755)
        output = run([BASH, "build.sh", mode, "--cann_3rd_lib_path", str(root / "cache")],
                     root, env, failure is None)
        assert not (root / "guard.log").exists(), output
        assert not list((root / "build_out").glob(".package.*"))
        package = root / "build_out" / PACKAGE_NAME
        if failure:
            assert not package.exists()
            assert "execute samples success" not in output
            assert not (root / "ascend/log").exists()
        else:
            assert package.read_text() == PAYLOAD
            assert "skipping LLVM/PTOAS build and packaging" in output
            assert "execute samples success" in output
            assert "board validation are disabled" in (root / "ascend/log/ptoas-presmoke.log").read_text()
            installed = run([BASH, str(package), "--full", "--quiet", "--install-path=" + str(root / "install")],
                            root, env)
            assert "cached installer invoked" in installed
            smoke = run([BASH, str(root / "test/samples/runop.sh"), "all"], root, env)
            assert "PreSmoke runop smoke skipped" in smoke
            board = run([BASH, str(root / "test/npu_validation/scripts/run_remote_npu_validation.sh")], root, env)
            assert "Skipping remote NPU validation in PreSmoke" in board
            tar = shutil.which("tar")
            run([tar, "-zcf", str(root / "slog.tar.gz"), "-C", str(root / "ascend"), "log"], root, env)
        if failure in ("path", "missing_id", "invalid_id"):
            assert not (root / "download.log").exists()
        else:
            download = (root / "download.log").read_text()
            prefix = "CANN/pto-as/ci" if ci == "actions" else "pto-as"
            assert f"/{prefix}/package/48/{PACKAGE_NAME}" in download
            assert "--max-time\n300" in download and "--proto\n=https" in download
        print(f"PASS: {ci} installer download, failure={failure}, mode={mode}")


for location in ("output", "workspace", "download", "cache"):
    check_case(location)
check_case("cache", smoke_key="ST_PART")
check_case("cache", valid=False)
check_case("cache", normal=True)
check_case("cache", mode="--build", normal=True)
check_case("cache", mode="--build")
check_cache_refresh()
check_invalid_packages()
check_toolchain_pipefail()
for ci_variant in ("actions", "robot", "robot_task", "robot_pr_id", "robot_obs"):
    check_ci_download(ci_variant)
for ci_variant in ("actions", "robot"):
    for download_failure in ("http", "html", "path", "missing_id"):
        check_ci_download(ci_variant, download_failure)
check_ci_download("robot", "invalid_id")
check_ci_download("robot_task", mode="--build")
