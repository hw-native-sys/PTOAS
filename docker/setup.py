# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import os
import pathlib
import re

from setuptools import setup, find_namespace_packages


_PROJECT_VERSION_RE = re.compile(
    r"project\s*\(\s*ptoas\s+VERSION\s+([0-9]+\.[0-9]+)\s*\)"
)


def read_package_version() -> str:
    version = os.environ.get("PTOAS_PYTHON_PACKAGE_VERSION", "").strip()
    if version:
        return version
    cmake_file = pathlib.Path(__file__).resolve().parents[1] / "CMakeLists.txt"
    match = _PROJECT_VERSION_RE.search(cmake_file.read_text(encoding="utf-8"))
    if not match:
        raise RuntimeError(f"could not find PTOAS version in {cmake_file}")
    return match.group(1)

setup(
    name="ptoas",
    version=read_package_version(),
    description="PTO Assembler & Optimizer",
    # NOTE: find_namespace_packages detects folders even without __init__.py
    packages=find_namespace_packages(),
    # NOTE: The * at the end captures .so.22, .so.22.1, etc.
    package_data={
            "mlir": [
                "**/*.so*",
                "**/*.pyd",
                "**/*.py",
                "_mlir_libs/*.so*", 
            ],
        },
    include_package_data=True,
    zip_safe=False,
    python_requires=">=3.9",
)
