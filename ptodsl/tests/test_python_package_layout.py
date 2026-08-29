# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import unittest
from pathlib import Path

from ptodsl import _ops


class PythonPackageLayoutTest(unittest.TestCase):
    def test_tileops_is_declared_as_an_editable_python_package(self):
        project_root = Path(__file__).resolve().parents[2]
        pyproject = (project_root / "pyproject.toml").read_text(encoding="utf-8")

        self.assertIn("[tool.scikit-build.wheel.packages]", pyproject)
        self.assertIn('TileOps = "lib/TileOps"', pyproject)
        self.assertTrue((project_root / "lib/TileOps/__init__.py").is_file())

    def test_internal_ops_exports_reference_existing_symbols(self):
        missing = [name for name in _ops.__all__ if not hasattr(_ops, name)]
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
