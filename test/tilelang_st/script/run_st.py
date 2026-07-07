#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

"""
TileLang ST runner — validates TileLang DSL template library on NPU / simulator.

Usage:
    python3 test/tilelang_st/script/run_st.py -r npu -v a5 -t tadd
    python3 test/tilelang_st/script/run_st.py -r sim -v a5 -t tadd
"""

import os
import sys
import subprocess
import shutil
import argparse


def run_command(command, cwd=None, check=True):
    try:
        print(f"run command: {' '.join(command)}")
        subprocess.run(command, cwd=cwd, check=check, stdout=None, stderr=None, text=True)
    except subprocess.CalledProcessError as e:
        print(f"run command failed with return code {e.returncode}")
        raise


def _normalize_case_filters(case_filters):
    if not case_filters:
        return []
    if isinstance(case_filters, str):
        return [case_filters]
    normalized = []
    seen = set()
    for case_filter in case_filters:
        if not case_filter or case_filter in seen:
            continue
        normalized.append(case_filter)
        seen.add(case_filter)
    return normalized


def find_ptoas_bin():
    """Locate the ptoas binary by walking up from this script to the repo root."""
    env_bin = os.environ.get("PTOAS_BIN")
    if env_bin and os.path.isfile(env_bin):
        return os.path.abspath(env_bin)

    search_dir = os.path.dirname(os.path.abspath(__file__))
    for _ in range(8):
        candidate = os.path.join(search_dir, "build", "tools", "ptoas", "ptoas")
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
        parent = os.path.dirname(search_dir)
        if parent == search_dir:
            break
        search_dir = parent
    return None


def set_env_variables(run_mode, soc_version):
    if run_mode == "sim":
        ld_lib_path = os.environ.get("LD_LIBRARY_PATH", "")
        if ld_lib_path:
            filtered_paths = [
                path for path in ld_lib_path.split(":")
                if "/runtime/lib64" not in path
            ]
            os.environ["LD_LIBRARY_PATH"] = ":".join(filtered_paths)

        ascend_home = os.environ.get("ASCEND_HOME_PATH")
        if not ascend_home:
            raise EnvironmentError("ASCEND_HOME_PATH is not set")

        os.environ["LD_LIBRARY_PATH"] = (
            f"{ascend_home}/runtime/lib64/stub:{os.environ.get('LD_LIBRARY_PATH', '')}"
        )
        setenv_path = os.path.join(ascend_home, "bin", "setenv.bash")
        if os.path.exists(setenv_path):
            print(f"run env shell: {setenv_path}")
            result = subprocess.run(
                f"source {setenv_path} && env",
                shell=True,
                executable=shutil.which("bash") or "bash",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            for line in result.stdout.splitlines():
                if "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key] = value
        else:
            print(f"warning: not found {setenv_path}")

        simulator_lib_path = os.path.join(
            ascend_home, "tools", "simulator", soc_version, "lib"
        )
        sim_lib_dir = os.environ.get("SIM_LIB_DIR")
        if sim_lib_dir:
            simulator_lib_path = sim_lib_dir
            os.environ["TILELANG_ST_SIM_LIB_DIR"] = simulator_lib_path
            print(f"[INFO] SIM_LIB_DIR={simulator_lib_path}")
        os.environ["LD_LIBRARY_PATH"] = (
            f"{simulator_lib_path}:{os.environ.get('LD_LIBRARY_PATH', '')}"
        )


def get_testcase_work_dir(testcase):
    return os.path.join("build", "testcase", testcase)


def build_project(run_mode, soc_version, testcase, ptoas_bin):
    build_dir = "build"
    if os.path.exists(build_dir):
        print(f"clean build: {build_dir}")
        shutil.rmtree(build_dir)
    os.makedirs(build_dir, exist_ok=True)

    try:
        cmake_cmd = [
            "cmake",
            f"-DRUN_MODE={run_mode}",
            f"-DSOC_VERSION={soc_version}",
            f"-DTEST_CASE={testcase}",
            f"-DPTOAS_BIN={ptoas_bin}",
            "..",
        ]
        subprocess.run(
            cmake_cmd,
            cwd=build_dir,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        cpu_count = os.cpu_count() or 4
        make_cmd = ["make", "VERBOSE=1", "-j", str(cpu_count)]
        result = subprocess.run(
            make_cmd,
            cwd=build_dir,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        print("compile process:\n", result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"build failed: {e.stdout}")
        raise


def _write_filtered_cases_module(src_path, dst_path, selected_case_names):
    selected_case_names = _normalize_case_filters(selected_case_names)
    wrapper = f"""#!/usr/bin/python3
# Auto-generated by run_st.py; do not edit.

import importlib.util as _importlib_util

_SOURCE_PATH = {src_path!r}
_SELECTED_CASE_NAMES = {selected_case_names!r}

_SPEC = _importlib_util.spec_from_file_location("_tilelang_st_source_cases", _SOURCE_PATH)
_MODULE = _importlib_util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

for _name, _value in vars(_MODULE).items():
    if not _name.startswith("_"):
        globals()[_name] = _value

if _SELECTED_CASE_NAMES:
    _selected = set(_SELECTED_CASE_NAMES)
    _all_case_names = [case.get("name") for case in CASES]
    _missing = sorted(name for name in _selected if name not in _all_case_names)
    if _missing:
        raise RuntimeError("unknown case filter(s): " + ", ".join(_missing))
    CASES = [case for case in CASES if case.get("name") in _selected]
"""
    with open(dst_path, "w", encoding="utf-8") as handle:
        handle.write(wrapper)


def _copy_testcase_scripts(testcase, case_filters=None):
    """Copy shared and per-testcase Python scripts into the build work directory."""
    work_dir = get_testcase_work_dir(testcase)
    os.makedirs(work_dir, exist_ok=True)
    # Shared scripts (testcase/ level).
    for name in ("st_common.py",):
        src = os.path.join("testcase", name)
        if os.path.isfile(src):
            run_command(["cp", src, os.path.join(work_dir, name)])
    # Per-testcase scripts.
    testcase_src = f"testcase/{testcase}"
    for name in ("cases.py", "gen_data.py", "compare.py"):
        src = os.path.join(testcase_src, name)
        if os.path.isfile(src):
            dst = os.path.join(work_dir, name)
            if name == "cases.py":
                selected_case_names = _normalize_case_filters(case_filters)
                if selected_case_names:
                    _write_filtered_cases_module(os.path.abspath(src), dst, selected_case_names)
                    continue
            run_command(["cp", src, dst])


def run_gen_data(testcase, case_filters=None):
    original_dir = os.getcwd()
    try:
        work_dir = get_testcase_work_dir(testcase)
        _copy_testcase_scripts(testcase, case_filters)
        os.chdir(work_dir)
        run_command([sys.executable, "gen_data.py"])
    except Exception as e:
        print(f"gen golden failed: {e}")
        raise
    finally:
        os.chdir(original_dir)


def run_binary(testcase, case_filters=None):
    original_dir = os.getcwd()
    try:
        os.chdir(get_testcase_work_dir(testcase))
        binary = os.path.join("..", "..", "bin", testcase)
        normalized_case_filters = _normalize_case_filters(case_filters)
        if not normalized_case_filters:
            run_command([binary])
        else:
            for case_filter in normalized_case_filters:
                run_command([binary, case_filter])
    except Exception as e:
        print(f"run binary failed: {e}")
        raise
    finally:
        os.chdir(original_dir)


def run_compare(testcase, case_filters=None):
    original_dir = os.getcwd()
    try:
        work_dir = get_testcase_work_dir(testcase)
        _copy_testcase_scripts(testcase, case_filters)
        os.chdir(work_dir)
        cmd = [sys.executable, "compare.py"]
        run_command(cmd)
    except Exception as e:
        print(f"compare failed: {e}")
        raise
    finally:
        os.chdir(original_dir)


def main():
    parser = argparse.ArgumentParser(description="TileLang ST runner")
    parser.add_argument("-r", "--run-mode", required=True,
                        help="Run mode: sim or npu")
    parser.add_argument("-v", "--soc-version", required=True,
                        help="SoC version: a5")
    parser.add_argument("-t", "--testcase", required=True,
                        help="Test case name (e.g. tadd)")
    parser.add_argument("-p", "--ptoas-bin", required=False,
                        help="Path to ptoas binary (auto-detected if omitted)")
    parser.add_argument("-c", "--case", action="append", default=[],
                        help="Run one or more specific cases within the testcase. Can be passed multiple times.")
    parser.add_argument("-w", "--without-build", action="store_true",
                        help="Skip build (requires prior build)")
    parser.add_argument("--target-dir", required=False,
                        help="TileLang ST target directory. Defaults to npu/<soc>/src/st.")

    args = parser.parse_args()

    if args.soc_version == "a5":
        default_soc_version = "Ascend950PR_9599"
    else:
        print(f"[ERROR] Unsupported soc-version: {args.soc_version}, only a5 is supported",
              file=sys.stderr)
        sys.exit(1)

    testcase = args.testcase

    ptoas_bin = args.ptoas_bin or find_ptoas_bin()
    if not ptoas_bin:
        print("[ERROR] Cannot find ptoas binary. "
              "Set PTOAS_BIN env or use -p flag.", file=sys.stderr)
        sys.exit(1)
    ptoas_bin = os.path.abspath(ptoas_bin)
    print(f"[INFO] ptoas: {ptoas_bin}")

    original_dir = os.getcwd()
    try:
        script_path = os.path.abspath(__file__)
        tilelang_st_root = os.path.dirname(os.path.dirname(script_path))
        target_dir = args.target_dir or os.environ.get("TILELANG_ST_TARGET_DIR")
        if target_dir:
            target_dir = os.path.abspath(target_dir)
        else:
            target_dir = os.path.join(tilelang_st_root, "npu", args.soc_version, "src", "st")

        if not os.path.isdir(target_dir):
            print(f"[ERROR] Target dir not found: {target_dir}", file=sys.stderr)
            sys.exit(1)

        print(f"target_dir: {target_dir}")
        os.chdir(target_dir)

        set_env_variables(args.run_mode, default_soc_version)

        if not args.without_build:
            build_project(args.run_mode, default_soc_version, testcase, ptoas_bin)

        # gen golden → run binary → compare
        run_gen_data(testcase, args.case)
        run_binary(testcase, args.case)
        run_compare(testcase, args.case)

    except Exception as e:
        print(f"run failed: {str(e)}", file=sys.stderr)
        sys.exit(1)
    finally:
        os.chdir(original_dir)


if __name__ == "__main__":
    main()
