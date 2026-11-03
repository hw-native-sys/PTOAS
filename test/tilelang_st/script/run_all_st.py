#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Batch runner for TileLang ST, suitable for CI/self-hosted runner usage."""

import argparse
import concurrent.futures
import importlib.util
import os
import subprocess
import sys
import traceback

import run_st


SOC_VERSION_MAP = {
    "a5": "Ascend950PR_9599",
}

def discover_testcases(testcase_root):
    testcases = []
    for entry in sorted(os.listdir(testcase_root)):
        testcase_dir = os.path.join(testcase_root, entry)
        if not os.path.isdir(testcase_dir):
            continue
        pto_file = os.path.join(testcase_dir, f"{entry}.pto")
        if os.path.isfile(pto_file):
            testcases.append(entry)
    return testcases


def load_case_names(testcase_root, testcase):
    cases_path = os.path.join(testcase_root, testcase, "cases.py")
    if not os.path.isfile(cases_path):
        raise FileNotFoundError(f"cases.py not found: {cases_path}")

    spec = importlib.util.spec_from_file_location(f"_tilelang_{testcase}_cases", cases_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return [case["name"] for case in module.CASES]


def resolve_case_filters(testcase_root, testcase, smoke_mode):
    if not smoke_mode:
        return []
    case_names = load_case_names(testcase_root, testcase)
    if not case_names:
        raise ValueError(f"no cases found for smoke testcase: {testcase}")
    return []


def resolve_smoke_case_names(testcase_root, testcase, smoke_mode):
    if not smoke_mode:
        return []
    case_names = load_case_names(testcase_root, testcase)
    if not case_names:
        raise ValueError(f"no cases found for smoke testcase: {testcase}")
    return case_names


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run all TileLang ST testcases for CI or local batch validation."
    )
    parser.add_argument(
        "-r", "--run-mode", default="sim",
        help="Run mode: sim or npu (default: sim)",
    )
    parser.add_argument(
        "-v", "--soc-version", default="a5",
        help="SoC version: a5 (default: a5)",
    )
    parser.add_argument(
        "-p", "--ptoas-bin", default=None,
        help="Path to ptoas binary (auto-detected if omitted)",
    )
    parser.add_argument(
        "-t", "--testcase", action="append", default=[],
        help="Run only selected testcase(s). Can be passed multiple times.",
    )
    parser.add_argument(
        "-w", "--without-build", action="store_true",
        help="Skip build and reuse the existing build directory.",
    )
    parser.add_argument(
        "--fail-fast", action="store_true",
        help="Stop immediately after the first failed testcase.",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List discovered testcases and exit.",
    )
    parser.add_argument(
        "-j", "--jobs", type=int, default=1,
        help="Number of testcases to run in parallel after the shared build (default: 1).",
    )
    parser.add_argument(
        "--build-jobs", type=int, default=None,
        help="Maximum parallel jobs for the shared CMake build (default: host CPU count).",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Run only a representative smoke subset of cases for each testcase.",
    )
    return parser.parse_args()


def resolve_selected_testcases(all_testcases, requested):
    if not requested:
        return all_testcases

    requested_set = []
    seen = set()
    for testcase in requested:
        if testcase not in seen:
            requested_set.append(testcase)
            seen.add(testcase)

    missing = [testcase for testcase in requested_set if testcase not in all_testcases]
    if missing:
        raise ValueError(
            f"Unsupported testcase(s): {', '.join(missing)}; "
            f"supported: {', '.join(all_testcases)}"
        )
    return requested_set


def run_testcase_subprocess(
    run_st_script_path, run_mode, soc_version, ptoas_bin, target_dir, testcase, case_filters=None
):
    command = [
        sys.executable,
        run_st_script_path,
        "-r", run_mode,
        "-v", soc_version,
        "-t", testcase,
        "-p", ptoas_bin,
        "--target-dir", target_dir,
        "-w",
    ]
    for case_filter in case_filters or []:
        command.extend(["-c", case_filter])
    env = os.environ.copy()
    result = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    return testcase, result.returncode, result.stdout


def main():
    args = parse_args()

    if args.soc_version not in SOC_VERSION_MAP:
        print(
            f"[ERROR] Unsupported soc-version: {args.soc_version}, "
            f"supported: {', '.join(sorted(SOC_VERSION_MAP))}",
            file=sys.stderr,
        )
        sys.exit(1)
    if args.jobs < 1:
        print("[ERROR] --jobs must be >= 1", file=sys.stderr)
        sys.exit(1)
    if args.build_jobs is not None and args.build_jobs < 1:
        print("[ERROR] --build-jobs must be >= 1", file=sys.stderr)
        sys.exit(1)

    batch_script_path = os.path.abspath(__file__)
    run_st_script_path = os.path.abspath(run_st.__file__)
    tilelang_st_root = os.path.dirname(os.path.dirname(batch_script_path))
    st_root = os.path.join(tilelang_st_root, "npu", args.soc_version, "src", "st")
    full_testcase_root = os.path.join(st_root, "testcase")
    smoke_testcase_root = os.path.join(st_root, "smoke", "testcase")
    if args.smoke:
        testcase_root = smoke_testcase_root
        target_dir = os.path.dirname(smoke_testcase_root)
    else:
        testcase_root = full_testcase_root
        target_dir = st_root

    if not os.path.isdir(testcase_root):
        print(f"[ERROR] Testcase root not found: {testcase_root}", file=sys.stderr)
        sys.exit(1)

    all_testcases = discover_testcases(testcase_root)
    if not all_testcases:
        print(f"[ERROR] No testcases found in: {testcase_root}", file=sys.stderr)
        sys.exit(1)

    if args.list:
        for testcase in all_testcases:
            print(testcase)
        return

    try:
        selected_testcases = resolve_selected_testcases(all_testcases, args.testcase)
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    ptoas_bin = args.ptoas_bin or run_st.find_ptoas_bin()
    if not ptoas_bin:
        print(
            "[ERROR] Cannot find ptoas binary. Set PTOAS_BIN env or use -p flag.",
            file=sys.stderr,
        )
        sys.exit(1)
    ptoas_bin = os.path.abspath(ptoas_bin)

    default_soc_version = SOC_VERSION_MAP[args.soc_version]
    print(f"[INFO] run_mode={args.run_mode}")
    print(f"[INFO] soc_version={args.soc_version} ({default_soc_version})")
    print(f"[INFO] ptoas={ptoas_bin}")
    print(f"[INFO] target_dir={target_dir}")
    print(f"[INFO] selected_testcases={', '.join(selected_testcases)}")
    print(f"[INFO] smoke={args.smoke}")
    print(f"[INFO] jobs={args.jobs}")
    print(f"[INFO] build_jobs={args.build_jobs or os.cpu_count() or 4}")

    original_dir = os.getcwd()
    failures = []
    try:
        os.chdir(target_dir)
        run_st.set_env_variables(args.run_mode, default_soc_version)

        if not args.without_build:
            if len(selected_testcases) == 1:
                build_target = selected_testcases[0]
            else:
                build_target = "all"
            print(f"[INFO] build requested for {build_target}")
            run_st.build_project(
                args.run_mode,
                default_soc_version,
                build_target,
                ptoas_bin,
                args.build_jobs,
            )

        total = len(selected_testcases)
        if args.jobs == 1:
            for index, testcase in enumerate(selected_testcases, start=1):
                case_filters = resolve_case_filters(testcase_root, testcase, args.smoke)
                smoke_case_names = resolve_smoke_case_names(testcase_root, testcase, args.smoke)
                print(f"[INFO] [{index}/{total}] running testcase: {testcase}")
                if smoke_case_names:
                    print(f"[INFO] smoke cases: {', '.join(smoke_case_names)}")
                try:
                    run_st.run_gen_data(testcase, case_filters)
                    run_st.run_binary(testcase, case_filters)
                    run_st.run_compare(testcase, case_filters)
                except Exception as exc:  # pragma: no cover - CI-side aggregation path
                    failures.append((testcase, str(exc)))
                    print(f"[ERROR] testcase failed: {testcase}")
                    traceback.print_exc()
                    if args.fail_fast:
                        break
        else:
            print(f"[INFO] running testcases in parallel with jobs={args.jobs}")
            max_workers = min(args.jobs, total)
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_testcase = {}
                for index, testcase in enumerate(selected_testcases, start=1):
                    case_filters = resolve_case_filters(testcase_root, testcase, args.smoke)
                    smoke_case_names = resolve_smoke_case_names(testcase_root, testcase, args.smoke)
                    print(f"[INFO] [{index}/{total}] queue testcase: {testcase}")
                    if smoke_case_names:
                        print(f"[INFO] smoke cases: {', '.join(smoke_case_names)}")
                    future = executor.submit(
                        run_testcase_subprocess,
                        run_st_script_path,
                        args.run_mode,
                        args.soc_version,
                        ptoas_bin,
                        target_dir,
                        testcase,
                        case_filters,
                    )
                    future_to_testcase[future] = testcase

                for future in concurrent.futures.as_completed(future_to_testcase):
                    testcase = future_to_testcase[future]
                    try:
                        _, returncode, output = future.result()
                    except Exception as exc:  # pragma: no cover - executor/host failure
                        failures.append((testcase, str(exc)))
                        print(f"[ERROR] testcase runner crashed: {testcase}")
                        traceback.print_exc()
                        if args.fail_fast:
                            break
                        continue

                    print(f"[INFO] ===== testcase {testcase} output begin =====")
                    if output:
                        print(output, end="" if output.endswith("\n") else "\n")
                    print(f"[INFO] ===== testcase {testcase} output end =====")

                    if returncode != 0:
                        failures.append((testcase, f"subprocess exited with {returncode}"))
                        print(f"[ERROR] testcase failed: {testcase}")
                        if args.fail_fast:
                            break

    except Exception as exc:
        print(f"[ERROR] batch run failed: {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        os.chdir(original_dir)

    passed = len(selected_testcases) - len(failures)
    print("[INFO] TileLang ST summary")
    print(f"[INFO] passed={passed} failed={len(failures)} total={len(selected_testcases)}")
    if failures:
        for testcase, reason in failures:
            print(f"[INFO] failed testcase: {testcase} ({reason})")
        sys.exit(1)


if __name__ == "__main__":
    main()
