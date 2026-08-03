#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Run all A5 TileLang ST smoke and full testcases in isolated parallel jobs."""

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import run_all_st
import run_st


SOC_VERSION = "a5"
DEFAULT_SOC_VERSION = run_all_st.SOC_VERSION_MAP[SOC_VERSION]


def _repo_root():
    return Path(__file__).resolve().parents[3]


def _timestamp():
    return time.strftime("%Y%m%d-%H%M%S")


def _default_ptoas_bin(repo_root):
    found = run_st.find_ptoas_bin()
    if found:
        return Path(found)
    return repo_root / "build" / "tools" / "ptoas" / "ptoas"


def _default_output_root(repo_root):
    return repo_root / "build" / "tilelang_st_a5_all_parallel" / _timestamp()


def _kind_paths(repo_root):
    st_root = repo_root / "test" / "tilelang_st" / "npu" / SOC_VERSION / "src" / "st"
    return {
        "full": {
            "target_dir": st_root,
            "testcase_root": st_root / "testcase",
        },
        "smoke": {
            "target_dir": st_root / "smoke",
            "testcase_root": st_root / "smoke" / "testcase",
        },
    }


def _discover_jobs(repo_root, kinds, requested):
    jobs = []
    paths = _kind_paths(repo_root)
    for kind in kinds:
        testcase_root = paths[kind]["testcase_root"]
        all_testcases = run_all_st.discover_testcases(str(testcase_root))
        selected = run_all_st.resolve_selected_testcases(all_testcases, requested)
        for testcase in selected:
            jobs.append(
                {
                    "kind": kind,
                    "testcase": testcase,
                    "target_dir": paths[kind]["target_dir"],
                    "testcase_root": testcase_root,
                }
            )
    return jobs


def _run_logged(command, log_handle, cwd, env):
    log_handle.write(f"# cwd: {cwd}\n")
    log_handle.write("# command: " + " ".join(str(item) for item in command) + "\n\n")
    log_handle.flush()
    proc = subprocess.Popen(
        [str(item) for item in command],
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        log_handle.write(line)
    return proc.wait()


def _copy_case_scripts(testcase_root, testcase, case_work_dir):
    case_work_dir.mkdir(parents=True, exist_ok=True)
    shared = testcase_root / "st_common.py"
    if shared.is_file():
        shutil.copy2(shared, case_work_dir / "st_common.py")
    testcase_dir = testcase_root / testcase
    for name in ("cases.py", "gen_data.py", "compare.py"):
        src = testcase_dir / name
        if src.is_file():
            shutil.copy2(src, case_work_dir / name)


def _run_one(job, args, ptoas_bin, output_root, base_env):
    kind = job["kind"]
    testcase = job["testcase"]
    job_name = f"{kind}_{testcase}"
    job_root = output_root / "work" / job_name
    build_dir = job_root / "build"
    tmp_dir = job_root / "tmp"
    log_path = output_root / "logs" / f"{job_name}.log"
    started = time.time()

    env = base_env.copy()
    env["TMPDIR"] = str(tmp_dir)
    env["PTODSL_CACHE_DIR"] = str(job_root / "ptodsl-cache")

    tmp_dir.mkdir(parents=True, exist_ok=True)
    (output_root / "logs").mkdir(parents=True, exist_ok=True)

    result = {
        "name": job_name,
        "kind": kind,
        "testcase": testcase,
        "returncode": 0,
        "seconds": 0.0,
        "log": str(log_path),
        "build_dir": str(build_dir),
    }

    with log_path.open("w", encoding="utf-8") as log_handle:
        log_handle.write(f"# kind: {kind}\n")
        log_handle.write(f"# testcase: {testcase}\n")
        log_handle.write(f"# source: {job['target_dir']}\n")
        log_handle.write(f"# build: {build_dir}\n")
        log_handle.write(f"# PTODSL_CACHE_DIR={env['PTODSL_CACHE_DIR']}\n")
        log_handle.write("\n")

        cmake_cmd = [
            "cmake",
            "-S",
            job["target_dir"],
            "-B",
            build_dir,
            f"-DRUN_MODE={args.run_mode}",
            f"-DSOC_VERSION={DEFAULT_SOC_VERSION}",
            f"-DTEST_CASE={testcase}",
            f"-DPTOAS_BIN={ptoas_bin}",
        ]

        rc = _run_logged(cmake_cmd, log_handle, output_root, env)
        if rc == 0:
            rc = _run_logged(
                ["cmake", "--build", build_dir, "--parallel", str(args.build_jobs)],
                log_handle,
                output_root,
                env,
            )

        if rc != 0:
            result["returncode"] = rc
            result["phase"] = "build"
            result["seconds"] = time.time() - started
            return result

        case_work_dir = build_dir / "testcase" / testcase
        _copy_case_scripts(job["testcase_root"], testcase, case_work_dir)

        for phase, command in (
            ("gen_data", [sys.executable, "gen_data.py"]),
            ("run", [build_dir / "bin" / testcase]),
            ("compare", [sys.executable, "compare.py"]),
        ):
            rc = _run_logged(command, log_handle, case_work_dir, env)
            if rc != 0:
                result["returncode"] = rc
                result["phase"] = phase
                break

    result["seconds"] = time.time() - started
    return result


def _write_summary(output_root, summary):
    with (output_root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with (output_root / "summary.tsv").open("w", encoding="utf-8") as handle:
        handle.write("name\tkind\ttestcase\treturncode\tphase\tseconds\tlog\n")
        for item in summary["results"]:
            handle.write(
                f"{item['name']}\t{item['kind']}\t{item['testcase']}\t"
                f"{item['returncode']}\t{item.get('phase', '')}\t"
                f"{item['seconds']:.3f}\t{item['log']}\n"
            )


def _result_name_list(results):
    return [item["name"] for item in sorted(results, key=lambda result: result["name"])]


def _print_result_group(title, results):
    names = _result_name_list(results)
    print(f"[INFO] {title}: {len(names)}")
    for item in sorted(results, key=lambda result: result["name"]):
        phase = f" phase={item.get('phase')}" if item.get("phase") else ""
        print(f"[INFO]   {item['name']}{phase} log={item['log']}")


def _parse_args():
    repo_root = _repo_root()
    parser = argparse.ArgumentParser(
        description="Run all A5 TileLang ST full and smoke testcases in parallel."
    )
    parser.add_argument("-r", "--run-mode", default="sim", help="Run mode: sim or npu.")
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
        help="Run only selected testcase(s), for both selected kinds.",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Number of testcases to run in parallel.",
    )
    parser.add_argument(
        "--build-jobs",
        type=int,
        default=1,
        help="Parallel build jobs inside each testcase build. Default: 1.",
    )
    parser.add_argument(
        "--output-root",
        default=str(_default_output_root(repo_root)),
        help="Directory for logs, summaries, and per-testcase build trees.",
    )
    parser.add_argument("--full-only", action="store_true", help="Run only non-smoke ST cases.")
    parser.add_argument("--smoke-only", action="store_true", help="Run only smoke ST cases.")
    parser.add_argument("--list", action="store_true", help="List selected jobs and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Print selected jobs and exit.")
    return parser.parse_args()


def main():
    args = _parse_args()
    repo_root = _repo_root()

    if args.full_only and args.smoke_only:
        print("[ERROR] --full-only and --smoke-only are mutually exclusive", file=sys.stderr)
        return 1
    if args.jobs < 1:
        print("[ERROR] --jobs must be >= 1", file=sys.stderr)
        return 1
    if args.build_jobs < 1:
        print("[ERROR] --build-jobs must be >= 1", file=sys.stderr)
        return 1

    kinds = ["full", "smoke"]
    if args.full_only:
        kinds = ["full"]
    if args.smoke_only:
        kinds = ["smoke"]

    try:
        jobs = _discover_jobs(repo_root, kinds, args.testcase)
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    ptoas_bin = Path(args.ptoas_bin).resolve()
    if not ptoas_bin.is_file():
        print(f"[ERROR] ptoas binary not found: {ptoas_bin}", file=sys.stderr)
        return 1

    if args.list or args.dry_run:
        for job in jobs:
            print(f"{job['kind']}\t{job['testcase']}")
        return 0

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "logs").mkdir(parents=True, exist_ok=True)

    base_env = os.environ.copy()
    original_env = os.environ.copy()
    try:
        try:
            run_st.set_env_variables(args.run_mode, DEFAULT_SOC_VERSION)
        except Exception as exc:
            print(f"[ERROR] failed to set TileLang ST environment: {exc}", file=sys.stderr)
            return 1
        base_env.update(os.environ)
    finally:
        os.environ.clear()
        os.environ.update(original_env)

    print(f"[INFO] jobs={len(jobs)} parallel={args.jobs} build_jobs={args.build_jobs}")
    print(f"[INFO] run_mode={args.run_mode} soc={SOC_VERSION} ({DEFAULT_SOC_VERSION})")
    print(f"[INFO] ptoas={ptoas_bin}")
    print(f"[INFO] output_root={output_root}")
    print("[INFO] each testcase uses its own build dir, PTODSL cache, and TMPDIR")

    results = []
    max_workers = min(args.jobs, len(jobs))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_job = {
            executor.submit(_run_one, job, args, ptoas_bin, output_root, base_env): job
            for job in jobs
        }
        for future in concurrent.futures.as_completed(future_to_job):
            job = future_to_job[future]
            try:
                result = future.result()
            except Exception as exc:  # pragma: no cover - host-side failure aggregation
                name = f"{job['kind']}_{job['testcase']}"
                result = {
                    "name": name,
                    "kind": job["kind"],
                    "testcase": job["testcase"],
                    "returncode": 1,
                    "phase": "runner",
                    "seconds": 0.0,
                    "log": str(output_root / "logs" / f"{name}.log"),
                    "error": str(exc),
                }
            results.append(result)
            status = "PASS" if result["returncode"] == 0 else "FAIL"
            phase = f" phase={result.get('phase')}" if result.get("phase") else ""
            print(f"[{status}] {result['name']} ({result['seconds']:.1f}s){phase} {result['log']}")

    passed_results = [item for item in results if item["returncode"] == 0]
    build_failed_results = [
        item for item in results
        if item["returncode"] != 0 and item.get("phase") == "build"
    ]
    run_failed_results = [
        item for item in results
        if item["returncode"] != 0 and item.get("phase") != "build"
    ]
    failures = build_failed_results + run_failed_results
    summary = {
        "run_mode": args.run_mode,
        "soc_version": SOC_VERSION,
        "soc_full_version": DEFAULT_SOC_VERSION,
        "ptoas": str(ptoas_bin),
        "output_root": str(output_root),
        "jobs": args.jobs,
        "build_jobs": args.build_jobs,
        "passed": len(passed_results),
        "build_failed": len(build_failed_results),
        "run_failed": len(run_failed_results),
        "failed": len(failures),
        "total": len(results),
        "passed_tests": _result_name_list(passed_results),
        "build_failed_tests": _result_name_list(build_failed_results),
        "run_failed_tests": _result_name_list(run_failed_results),
        "results": sorted(results, key=lambda item: item["name"]),
    }
    _write_summary(output_root, summary)

    print(
        f"[INFO] summary: passed={summary['passed']} "
        f"build_failed={summary['build_failed']} run_failed={summary['run_failed']} "
        f"failed={summary['failed']} "
        f"total={summary['total']}"
    )
    _print_result_group("passed testcases", passed_results)
    _print_result_group("build-failed testcases", build_failed_results)
    _print_result_group("run-failed testcases", run_failed_results)
    print(f"[INFO] summary files: {output_root / 'summary.tsv'} {output_root / 'summary.json'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
