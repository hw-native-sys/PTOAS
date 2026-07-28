# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Run each TileLib ST case in an isolated simulator subprocess."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest


_TEST_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _TEST_ROOT.parents[1]
_CASE_ROOT = _TEST_ROOT / "a5"

sys.path.insert(0, str(_TEST_ROOT))
from common import discover_cases  # noqa: E402


def _case_names() -> tuple[str, ...]:
    return tuple(case["name"] for case in discover_cases(_CASE_ROOT))


def _output_root() -> Path:
    override = os.environ.get("TILELIB_ST_OUTPUT_ROOT")
    if override:
        return Path(override).resolve()
    return _REPO_ROOT / "build" / "tilelib-st"


def _safe_name(case_name: str) -> str:
    return case_name.replace("/", "_").replace("\\", "_")


@pytest.mark.parametrize("case_name", _case_names(), ids=lambda name: name)
def test_tilelib_case(case_name: str) -> None:
    output_root = _output_root()
    safe_name = _safe_name(case_name)
    case_output = output_root / "cases" / safe_name
    log_path = output_root / "logs" / f"{safe_name}.log"
    msprof_root = output_root / ".msprof"
    cache_root = case_output / "ptodsl-cache"
    tmp_root = case_output / "tmp"

    log_path.parent.mkdir(parents=True, exist_ok=True)
    msprof_root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)
    tmp_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHON_BIN"] = sys.executable
    env["PTO_PYTHON_BIN"] = sys.executable
    env["PTOAS_MSPROF_PRIVATE_ROOT"] = str(msprof_root)
    env["PTODSL_CACHE_DIR"] = str(cache_root)
    env["TMPDIR"] = str(tmp_root)

    command = [
        str(_REPO_ROOT / "scripts" / "sim_dsl.sh"),
        "--output",
        str(case_output),
        str(_TEST_ROOT / "run_tilelib_st.py"),
        "--",
        str(_CASE_ROOT),
        "--case",
        case_name,
    ]
    completed = subprocess.run(
        command,
        cwd=_REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    log_path.write_text(completed.stdout, encoding="utf-8")

    assert completed.returncode == 0, (
        f"TileLib ST case {case_name!r} failed with exit code {completed.returncode}.\n"
        f"Log: {log_path}\n\n{completed.stdout}"
    )
