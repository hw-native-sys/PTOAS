#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Run TileLang ST testcases with PTODSL TileLib and per-test logs."""

import argparse
import concurrent.futures
import contextlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import run_all_st
import run_st


def _repo_root():
    return Path(__file__).resolve().parents[3]


def _timestamp():
    return time.strftime("%Y%m%d-%H%M%S")


def _default_ptoas_bin(repo_root):
    candidate = repo_root / "build-llvm21" / "tools" / "ptoas" / "ptoas"
    if candidate.is_file():
        return candidate
    found = run_st.find_ptoas_bin()
    if found:
        return Path(found)
    return candidate


def _testcase_root(repo_root, soc_version, smoke):
    st_root = repo_root / "test" / "tilelang_st" / "npu" / soc_version / "src" / "st"
    if smoke:
        return st_root / "smoke" / "testcase"
    return st_root / "testcase"


def _target_dir_from_testcase_root(testcase_root):
    return testcase_root.parent


def _log_dir(repo_root, requested):
    if requested:
        return Path(requested).resolve()
    return repo_root / "build-llvm21" / "tilelang_st_ptodsl_logs" / _timestamp()


def _write_json(path, payload):
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _run_build(args, default_soc_version, target_dir, log_dir, ptoas_bin):
    build_log = log_dir / "build.log"
    started = time.time()
    with build_log.open("w", encoding="utf-8") as handle:
        handle.write(f"# cwd: {target_dir}\n")
        handle.write(f"# ptoas: {ptoas_bin}\n")
        handle.write("# command: run_st.build_project(..., testcase='all', ...)\n\n")
        handle.flush()
        original_dir = os.getcwd()
        try:
            os.chdir(target_dir)
            with contextlib.redirect_stdout(handle), contextlib.redirect_stderr(handle):
                run_st.set_env_variables(args.run_mode, default_soc_version)
                run_st.build_project(
                    args.run_mode,
                    default_soc_version,
                    "all",
                    str(ptoas_bin),
                    args.jobs,
                )
        finally:
            os.chdir(original_dir)
    return {
        "name": "build",
        "returncode": 0,
        "seconds": time.time() - started,
        "log": str(build_log),
    }


def _run_one_testcase(args, testcase, target_dir, log_dir, ptoas_bin):
    log_path = log_dir / f"{testcase}.log"
    command = [
        sys.executable,
        str(Path(run_all_st.__file__).resolve()),
        "-r",
        args.run_mode,
        "-v",
        args.soc_version,
        "-p",
        str(ptoas_bin),
        "-t",
        testcase,
        "-j",
        "1",
        "-w",
    ]
    if args.smoke:
        command.append("--smoke")

    env = os.environ.copy()

    started = time.time()
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"# testcase: {testcase}\n")
        handle.write(f"# cwd: {target_dir}\n")
        handle.write("# command: " + " ".join(command) + "\n\n")
        handle.flush()
        proc = subprocess.Popen(
            command,
            cwd=target_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            handle.write(line)
        proc.wait()

    return {
        "name": testcase,
        "returncode": proc.returncode,
        "seconds": time.time() - started,
        "log": str(log_path),
    }


def _parse_args():
    repo_root = _repo_root()
    parser = argparse.ArgumentParser(
        description="Run TileLang ST testcases in parallel and save one log per testcase."
    )
    parser.add_argument("-r", "--run-mode", default="sim", help="Run mode: sim or npu.")
    parser.add_argument("-v", "--soc-version", default="a5", help="SoC version key, default: a5.")
    parser.add_argument(
        "-p",
        "--ptoas-bin",
        default=str(_default_ptoas_bin(repo_root)),
        help="Path to ptoas binary.",
    )
    parser.add_argument(
        "-t",
        "--testcase",
        action="append",
        default=[],
        help="Run selected testcase(s). Can be passed more than once.",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Number of testcases to run in parallel. Each testcase uses run_all_st -j 1.",
    )
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Directory for build.log, one <testcase>.log per testcase, and summary files.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Use full ST testcase directory instead of smoke/testcase.",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip the shared build and run testcases with -w.",
    )
    parser.add_argument("--fail-fast", action="store_true", help="Stop queuing after the first failure.")
    parser.add_argument("--list", action="store_true", help="List selected testcases and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Print selected testcases and commands only.")
    return parser.parse_args()


def main():
    args = _parse_args()
    repo_root = _repo_root()
    args.smoke = not args.full

    if args.soc_version not in run_all_st.SOC_VERSION_MAP:
        print(
            f"[ERROR] Unsupported soc-version: {args.soc_version}; "
            f"supported: {', '.join(sorted(run_all_st.SOC_VERSION_MAP))}",
            file=sys.stderr,
        )
        return 1
    if args.jobs < 1:
        print("[ERROR] --jobs must be >= 1", file=sys.stderr)
        return 1

    testcase_root = _testcase_root(repo_root, args.soc_version, args.smoke)
    target_dir = _target_dir_from_testcase_root(testcase_root)
    if not testcase_root.is_dir():
        print(f"[ERROR] Testcase root not found: {testcase_root}", file=sys.stderr)
        return 1

    all_testcases = run_all_st.discover_testcases(str(testcase_root))
    try:
        selected = run_all_st.resolve_selected_testcases(all_testcases, args.testcase)
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    ptoas_bin = Path(args.ptoas_bin).resolve()
    if not ptoas_bin.is_file():
        print(f"[ERROR] ptoas binary not found: {ptoas_bin}", file=sys.stderr)
        return 1

    if args.list:
        for testcase in selected:
            print(testcase)
        return 0

    log_dir = _log_dir(repo_root, args.log_dir)
    if not args.dry_run:
        log_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] selected={len(selected)} smoke={args.smoke} jobs={args.jobs}")
    print(f"[INFO] target_dir={target_dir}")
    print(f"[INFO] ptoas={ptoas_bin}")
    print(f"[INFO] logs={log_dir}")

    if args.dry_run:
        for testcase in selected:
            print(f"[DRY-RUN] {testcase}")
        return 0

    default_soc_version = run_all_st.SOC_VERSION_MAP[args.soc_version]
    results = []

    try:
        if not args.skip_build:
            print("[INFO] building all selected smoke/full ST targets once")
            build_result = _run_build(args, default_soc_version, target_dir, log_dir, ptoas_bin)
            results.append(build_result)
            print(f"[PASS] build ({build_result['seconds']:.1f}s) {build_result['log']}")
    except Exception as exc:
        build_result = {
            "name": "build",
            "returncode": 1,
            "seconds": 0.0,
            "log": str(log_dir / "build.log"),
            "error": str(exc),
        }
        results.append(build_result)
        _write_json(log_dir / "summary.json", {"results": results})
        print(f"[FAIL] build: {exc}")
        print(f"[INFO] build log: {build_result['log']}")
        return 1

    failures = []
    max_workers = min(args.jobs, len(selected))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_testcase = {
            executor.submit(_run_one_testcase, args, testcase, target_dir, log_dir, ptoas_bin): testcase
            for testcase in selected
        }
        for future in concurrent.futures.as_completed(future_to_testcase):
            testcase = future_to_testcase[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "name": testcase,
                    "returncode": 1,
                    "seconds": 0.0,
                    "log": str(log_dir / f"{testcase}.log"),
                    "error": str(exc),
                }
            results.append(result)
            status = "PASS" if result["returncode"] == 0 else "FAIL"
            print(f"[{status}] {testcase} ({result['seconds']:.1f}s) {result['log']}")
            if result["returncode"] != 0:
                failures.append(result)
                if args.fail_fast:
                    for queued in future_to_testcase:
                        queued.cancel()
                    break

    summary = {
        "run_mode": args.run_mode,
        "soc_version": args.soc_version,
        "smoke": args.smoke,
        "jobs": args.jobs,
        "target_dir": str(target_dir),
        "ptoas": str(ptoas_bin),
        "log_dir": str(log_dir),
        "passed": sum(1 for item in results if item["name"] != "build" and item["returncode"] == 0),
        "failed": len(failures),
        "total": len(selected),
        "results": results,
    }
    _write_json(log_dir / "summary.json", summary)
    with (log_dir / "summary.tsv").open("w", encoding="utf-8") as handle:
        handle.write("testcase\treturncode\tseconds\tlog\n")
        for result in results:
            handle.write(
                f"{result['name']}\t{result['returncode']}\t"
                f"{result['seconds']:.3f}\t{result['log']}\n"
            )

    print(
        f"[INFO] summary: passed={summary['passed']} failed={summary['failed']} "
        f"total={summary['total']}"
    )
    print(f"[INFO] summary files: {log_dir / 'summary.tsv'} {log_dir / 'summary.json'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
