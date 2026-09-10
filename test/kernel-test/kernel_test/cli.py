# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Command-line interface for the kernel-test framework."""

from __future__ import annotations

import argparse
import os
from typing import Sequence

from .cases import select_cases
from .registry import RegistryError, load_registry
from .runners import run_artifact_suite, run_correctness_suite, run_cycle_probe


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified kernel-test CLI")
    parser.add_argument(
        "--kernel-dir",
        help=(
            "Kernel package root to discover. Accepts either a directory that contains "
            "kernel subdirectories such as docs/ or one kernel package directory such as docs/rope. "
            "Defaults to test/kernel-test/kernels or $KERNEL_TEST_KERNEL_DIR."
        ),
    )
    parser.add_argument("--list-ops", action="store_true", help="List registered kernels")
    parser.add_argument("--op", help="Kernel name to run")
    parser.add_argument(
        "--workflow",
        choices=("correctness", "cycle"),
        default="correctness",
        help="Workflow name",
    )
    parser.add_argument("--backend", help="Backend name")
    parser.add_argument("--case", action="append", default=[], help="Case id to run")
    parser.add_argument("--case-filter", help="Substring filter for cases")
    parser.add_argument("--list-cases", action="store_true", help="List cases for one kernel")
    parser.add_argument(
        "--emit-mlir",
        action="store_true",
        help="Generate PTO artifacts such as vmi.pto/mi.pto under kernels/**/generated",
    )
    return parser


def _print_lines(lines: Sequence[str]) -> None:
    for line in lines:
        print(line)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    kernel_dir = args.kernel_dir or os.environ.get("KERNEL_TEST_KERNEL_DIR")

    try:
        registry = load_registry(kernel_dir=kernel_dir)
    except RegistryError as exc:
        raise SystemExit(str(exc)) from exc

    if args.list_ops:
        names = registry.list_names()
        if not names:
            print("No kernels registered yet.")
            return 0
        _print_lines(names)
        return 0

    if args.list_cases:
        if not args.op:
            parser.error("--list-cases requires --op")
        spec = registry.get(args.op)
        if spec is None:
            parser.error(f"unknown kernel: {args.op}")
        case_ids = sorted(spec.list_cases(args.workflow).keys())
        if not case_ids:
            print(f"No cases registered for kernel {spec.name} workflow={args.workflow}.")
            return 0
        _print_lines(case_ids)
        return 0

    if not args.op:
        parser.error("one of --list-ops or --op is required")

    if args.emit_mlir and args.workflow != "correctness":
        parser.error("--emit-mlir currently requires --workflow correctness")

    spec = registry.get(args.op)
    if spec is None:
        parser.error(f"unknown kernel: {args.op}")

    backend_name = args.backend or spec.default_backend
    if spec.backend_names and backend_name not in spec.backend_names:
        parser.error(
            f"unsupported backend {backend_name!r} for kernel {spec.name}; "
            f"choices: {', '.join(spec.backend_names)}"
        )

    try:
        cases = select_cases(
            spec.list_cases(args.workflow),
            case_ids=args.case,
            case_filter=args.case_filter,
            require_single=args.workflow == "cycle",
        )
    except ValueError as exc:
        parser.error(str(exc))

    backend = spec.create_backend(backend_name)

    if args.emit_mlir:
        summary = run_artifact_suite(cases, backend=backend)
        print(
            f"SUMMARY total={summary.total} passed={summary.passed} "
            f"failed={summary.failed} skipped={summary.skipped}"
        )
        return 0 if summary.all_passed else 1

    if args.workflow == "correctness":
        summary = run_correctness_suite(cases, backend=backend, verify_case=spec.verify)
        print(
            f"SUMMARY total={summary.total} passed={summary.passed} "
            f"failed={summary.failed} skipped={summary.skipped}"
        )
        return 0 if summary.all_passed else 1

    case_id, case = next(iter(cases.items()))
    marker_fields = {
        "op": spec.name,
        "backend": backend.name,
        **dict(spec.cycle_fields(case_id, case, backend)),
    }
    return run_cycle_probe(case_id=case_id, case=case, backend=backend, marker_fields=marker_fields)
