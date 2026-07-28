# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Run each TileLib ST operator suite in an isolated simulator subprocess.

All cases declared by one ``case.py`` run in the same simulator session.  This
keeps operator suites independently schedulable while avoiding one CA model
startup per parameterized case.
"""

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
from common import discover_case_modules  # noqa: E402


def _suite_params() -> tuple:
    params = []
    seen_names = set()
    for module in discover_case_modules(_CASE_ROOT):
        case_path = Path(module.__file__).resolve()
        relative_path = case_path.relative_to(_CASE_ROOT)
        suite_name = relative_path.parent.as_posix()
        if suite_name == ".":
            suite_name = relative_path.stem
        if suite_name in seen_names:
            raise RuntimeError(f"Duplicate TileLib ST suite name {suite_name!r}")
        seen_names.add(suite_name)
        case_names = tuple(case["name"] for case in module.CASES)
        params.append(pytest.param(suite_name, case_path, case_names, id=suite_name))
    return tuple(params)


def _output_root() -> Path:
    override = os.environ.get("TILELIB_ST_OUTPUT_ROOT")
    if override:
        return Path(override).resolve()
    return _REPO_ROOT / "build" / "tilelib-st"


def _safe_name(suite_name: str) -> str:
    return suite_name.replace("/", "_").replace("\\", "_")


@pytest.mark.parametrize(("suite_name", "case_path", "case_names"), _suite_params())
def test_tilelib_suite(suite_name: str, case_path: Path, case_names: tuple[str, ...]) -> None:
    output_root = _output_root()
    safe_name = _safe_name(suite_name)
    suite_output = output_root / "suites" / safe_name
    log_path = output_root / "logs" / f"{safe_name}.log"
    msprof_root = output_root / ".msprof"
    cache_root = suite_output / "ptodsl-cache"
    tmp_root = suite_output / "tmp"

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
        str(suite_output),
        str(case_path),
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
        f"TileLib ST suite {suite_name!r} failed with exit code {completed.returncode}.\n"
        f"Cases: {', '.join(case_names)}\n"
        f"Log: {log_path}\n\n{completed.stdout}"
    )
