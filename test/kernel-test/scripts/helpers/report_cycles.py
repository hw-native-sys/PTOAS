#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Internal helper: dispatch cycle reporting to one kernel-local analyzer."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

KT_ROOT = Path(__file__).resolve().parents[2]
if str(KT_ROOT) not in sys.path:
    sys.path.insert(0, str(KT_ROOT))

from kernel_test.registry import import_kernel_module


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Report cycle metrics for one kernel-test operator")
    parser.add_argument(
        "--kernel-dir",
        help=(
            "Kernel package root to discover. Accepts either a directory that contains "
            "kernel subdirectories or one kernel package directory. "
            "Defaults to test/kernel-test/kernels or $KERNEL_TEST_KERNEL_DIR."
        ),
    )
    parser.add_argument("--op", required=True, help="Kernel name")
    parser.add_argument("--table", action="store_true", help="Print compact table output")
    parser.add_argument("out_dirs", nargs="*", help="Per-case sim output directories")
    args = parser.parse_args(argv)
    kernel_dir = args.kernel_dir or os.environ.get("KERNEL_TEST_KERNEL_DIR")

    try:
        module = import_kernel_module(args.op, kernel_dir=kernel_dir, submodule="cycle_metrics")
    except ModuleNotFoundError as exc:
        print(f"no cycle metrics analyzer for kernel {args.op}: {exc}", file=sys.stderr)
        return 1

    if hasattr(module, "get_cycle_reporter"):
        from kernel_test.cycle_reporting import run_cycle_report

        reporter = module.get_cycle_reporter()
        forwarded: list[str] = []
        if args.table:
            forwarded.append("--table")
        forwarded.extend(args.out_dirs)
        return int(run_cycle_report(reporter, forwarded))

    forwarded = []
    if args.table:
        forwarded.append("--table")
    forwarded.extend(args.out_dirs)
    return int(module.main(forwarded))


if __name__ == "__main__":
    raise SystemExit(main())
