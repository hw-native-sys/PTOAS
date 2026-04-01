# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import os

from setuptools import setup, find_namespace_packages


def read_package_version() -> str:
    return os.environ.get("PTOAS_PYTHON_PACKAGE_VERSION", "0.1.1")

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
