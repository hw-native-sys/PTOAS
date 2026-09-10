#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Regression tests for PTOAS wheel cleanup during a run-package reinstall."""

from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import tempfile


REPOSITORY = Path(__file__).resolve().parents[2]
INSTALL_SCRIPT = REPOSITORY / "scripts/package/pto_as/scripts/install.sh"
COMMON_SCRIPT = REPOSITORY / "scripts/package/pto_as/scripts/pto_common.sh"


def extract_function(source: str, name: str) -> str:
    match = re.search(r"^" + re.escape(name) + r"\(\) \{\n.*?^\}", source, re.MULTILINE | re.DOTALL)
    if match is None:
        raise AssertionError(f"missing shell function: {name}")
    return match.group(0)


def run_prepare_reinstall(
    payload_wheels: list[str], installed_wheels: list[str], with_metadata: bool
) -> tuple[int, str, list[str], bool]:
    install_source = INSTALL_SCRIPT.read_text(encoding="utf-8")
    common_source = COMMON_SCRIPT.read_text(encoding="utf-8")
    shell = "\n".join(
        (
            extract_function(common_source, "pto_find_wheel"),
            extract_function(install_source, "get_installed_info"),
            extract_function(install_source, "clean_before_reinstall"),
            extract_function(install_source, "prepare_reinstall"),
        )
    )

    with tempfile.TemporaryDirectory(prefix="ptoas-reinstall-test-") as directory:
        root = Path(directory)
        package_root = root / "payload"
        package_wheels = package_root / "tools/ptoas/wheels"
        package_script = package_root / "share/info/pto_as/script"
        install_root = root / "installed/cann"
        install_info_dir = install_root / "share/info/pto_as"
        installed_wheel_dir = install_root / "tools/ptoas/wheels"
        package_wheels.mkdir(parents=True)
        package_script.mkdir(parents=True)
        install_info_dir.mkdir(parents=True)
        installed_wheel_dir.mkdir(parents=True)
        for wheel in payload_wheels:
            (package_wheels / wheel).touch()
        for wheel in installed_wheels:
            (installed_wheel_dir / wheel).touch()

        marker = root / "uninstall-called"
        if with_metadata:
            (install_info_dir / "ascend_install.info").write_text(
                f"PTO_AS_INSTALL_PATH_VAL={install_root}\nPTO_AS_VERSION=9.2.0\n",
                encoding="utf-8",
            )
            (install_info_dir / "script").mkdir(parents=True, exist_ok=True)
            uninstaller = install_info_dir / "script/pto_uninstall.sh"
            uninstaller.write_text(f"#!/bin/bash\ntouch '{marker}'\n", encoding="utf-8")
            uninstaller.chmod(0o755)

        command = shell + "\nprepare_reinstall\n"
        environment = dict(
            os.environ,
            CURR_PATH=str(package_script),
            TARGET_VERSION_DIR=str(install_root),
            TARGET_MOULDE_DIR=str(install_info_dir),
            UNINSTALL_SHELL_FILE=str(install_info_dir / "script/pto_uninstall.sh"),
            INSTALL_INFO_FILE=str(install_info_dir / "ascend_install.info"),
            KEY_INSTALLED_PATH="PTO_AS_INSTALL_PATH_VAL",
            KEY_INSTALLED_VERSION="PTO_AS_VERSION",
            IS_QUIET="y",
            IN_FEATURE="all",
            IS_DOCKER_INSTALL="n",
            DOCKER_ROOT="",
            pkg_version_dir="cann",
        )
        result = subprocess.run(
            ["bash", "-c", command],
            cwd=REPOSITORY,
            env=environment,
            text=True,
            capture_output=True,
            timeout=10,
        )
        remaining = sorted(path.name for path in installed_wheel_dir.glob("ptoas*.whl"))
        return result.returncode, result.stdout + result.stderr, remaining, marker.exists()


def main() -> None:
    result = run_prepare_reinstall(
        ["ptoas-0.61-cp37-abi3-manylinux_2_17_x86_64.whl"],
        ["ptoas-0.60-cp37-abi3-manylinux_2_17_x86_64.whl"],
        with_metadata=True,
    )
    assert result[0] == 0, result[1]
    assert result[2] == [], result
    assert result[3], result

    result = run_prepare_reinstall(
        ["ptoas-0.61-cp37-abi3-manylinux_2_17_x86_64.whl"],
        ["ptoas-0.60-cp37-abi3-manylinux_2_17_x86_64.whl"],
        with_metadata=False,
    )
    assert result[0] == 0, result[1]
    assert result[2] == [], result
    assert not result[3], result

    result = run_prepare_reinstall(
        [
            "ptoas-0.60-cp37-abi3-manylinux_2_17_x86_64.whl",
            "ptoas-0.61-cp37-abi3-manylinux_2_17_x86_64.whl",
        ],
        ["ptoas-0.59-cp37-abi3-manylinux_2_17_x86_64.whl"],
        with_metadata=True,
    )
    assert result[0] != 0, result
    assert result[2] == ["ptoas-0.59-cp37-abi3-manylinux_2_17_x86_64.whl"], result
    assert not result[3], result

    result = run_prepare_reinstall(
        [],
        ["ptoas-0.59-cp37-abi3-manylinux_2_17_x86_64.whl"],
        with_metadata=True,
    )
    assert result[0] != 0, result
    assert result[2] == ["ptoas-0.59-cp37-abi3-manylinux_2_17_x86_64.whl"], result
    assert not result[3], result
    print("PTOAS reinstall wheel cleanup regression tests passed")


if __name__ == "__main__":
    main()
