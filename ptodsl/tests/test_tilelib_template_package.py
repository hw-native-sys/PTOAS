#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Tests for TileOps package resolution outside the PTOAS CLI launcher."""

import unittest
from unittest.mock import Mock, patch

from ptodsl.tilelib import _template_package


class TileOpsPackageResolutionTest(unittest.TestCase):
    def setUp(self):
        _template_package.tileops_package.cache_clear()

    def tearDown(self):
        _template_package.tileops_package.cache_clear()

    def test_prefers_top_level_source_package(self):
        source_package = Mock()
        with patch.object(
            _template_package,
            "import_module",
            return_value=source_package,
        ) as import_module:
            self.assertIs(_template_package.tileops_package(), source_package)

        import_module.assert_called_once_with("TileOps")

    def test_falls_back_to_packaged_runtime_resources(self):
        packaged = Mock()
        missing_source = ModuleNotFoundError(
            "No module named 'TileOps'",
            name="TileOps",
        )
        with patch.object(
            _template_package,
            "import_module",
            side_effect=(missing_source, packaged),
        ) as import_module:
            self.assertIs(_template_package.tileops_package(), packaged)

        self.assertEqual(
            [call.args[0] for call in import_module.call_args_list],
            ["TileOps", "ptoas._runtime.share.ptoas.TileOps"],
        )

    def test_does_not_hide_source_package_dependency_errors(self):
        missing_dependency = ModuleNotFoundError(
            "No module named 'dependency'",
            name="dependency",
        )
        with patch.object(
            _template_package,
            "import_module",
            side_effect=missing_dependency,
        ) as import_module:
            with self.assertRaises(ModuleNotFoundError) as raised:
                _template_package.tileops_package()

        self.assertIs(raised.exception, missing_dependency)
        import_module.assert_called_once_with("TileOps")


if __name__ == "__main__":
    unittest.main()
