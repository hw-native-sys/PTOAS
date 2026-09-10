# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Rope-specific cycle-report entrypoint built on shared kernel-test contracts."""

from __future__ import annotations

import glob
import os

from kernel_test.cycle_reporting import CycleReporterSpec, run_cycle_report


def default_cycle_out_dirs(sim_root: str | None = None) -> list[str]:
    """Return kernel-test rope sim output dirs in canonical case order."""

    root = sim_root or os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "sim_outputs",
        "rope",
    )
    try:
        from .tile_config import DTYPES, MODES

        ordered = [
            os.path.join(root, backend, f"{dtype}_{mode}")
            for backend in ("cce", "vmi", "mi")
            for dtype in DTYPES
            for mode in MODES
            if os.path.isdir(os.path.join(root, backend, f"{dtype}_{mode}"))
        ]
        if ordered:
            return ordered
    except ImportError:
        pass
    return sorted(
        path
        for path in glob.glob(os.path.join(root, "*", "*"))
        if os.path.isdir(path)
    )


CYCLE_REPORTER = CycleReporterSpec(
    name="rope",
    default_out_dirs=default_cycle_out_dirs,
    missing_message="No rope cycle output dirs found. Run kernel-test/scripts/run_cycle.sh first.",
)


def get_cycle_reporter() -> CycleReporterSpec:
    """Return the rope cycle reporter registration."""

    return CYCLE_REPORTER


def main(argv: list[str] | None = None) -> int:
    return run_cycle_report(CYCLE_REPORTER, argv)


if __name__ == "__main__":
    raise SystemExit(main())
