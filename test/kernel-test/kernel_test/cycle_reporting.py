# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Framework contract and runner for per-kernel cycle reporting."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass

from .cycle_metrics import RunMetrics, format_run_summary, format_table, parse_run_metrics


@dataclass(frozen=True)
class CycleReporterSpec:
    """Framework-owned contract for one kernel cycle-report adapter."""

    name: str
    default_out_dirs: Callable[[str | None], list[str]]
    missing_message: str
    parse_metrics: Callable[[str], RunMetrics] = parse_run_metrics
    format_summary: Callable[[RunMetrics, str | None], str] = format_run_summary
    format_table: Callable[[list[tuple[str, RunMetrics]]], str] = format_table
    label_for_dir: Callable[[str], str] = lambda path: os.path.basename(path.rstrip("/"))


def run_cycle_report(spec: CycleReporterSpec, argv: list[str] | None = None) -> int:
    """Run one kernel-local cycle report with shared CLI behavior."""

    parser = argparse.ArgumentParser(description=f"Parse {spec.name} cycle metrics from kernel-test outputs")
    parser.add_argument(
        "out_dirs",
        nargs="*",
        help="case output dirs such as sim_outputs/<op>/<backend>/<case>",
    )
    parser.add_argument("--table", action="store_true", help="Print compact table")
    args = parser.parse_args(argv)

    dirs = args.out_dirs if args.out_dirs else spec.default_out_dirs(None)
    if not dirs:
        print(spec.missing_message, file=sys.stderr)
        return 1

    rows: list[tuple[str, RunMetrics]] = []
    for path in dirs:
        try:
            metrics = spec.parse_metrics(path)
        except FileNotFoundError as exc:
            print(str(exc), file=sys.stderr)
            continue
        label = spec.label_for_dir(path)
        rows.append((label, metrics))
        if not args.table:
            print(spec.format_summary(metrics, label))
            print()

    if args.table and rows:
        print(spec.format_table(rows))
    return 0 if rows else 1
