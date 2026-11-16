#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You should not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Unit regression for the ``pto.ttrans`` b8 16-bit chunk-span guard (#1604).

A b8 row stride whose single 128-lane chunk needs more element offsets than
the 16-bit vgather2/vscatter offset field holds (``128 * 544 = 69632``) must
raise instead of silently wrapping: no chunk splitting can shrink one chunk's
span, so the shape is unsupported on this surface (upstream ``TTrans.hpp``
rejects it with a ``static_assert``). Every tile that over-spans the field
also overflows the vec buffer, and the buffer check in the expansion pass
fires first, so the guard is exercised directly here.
"""

import importlib.util
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TTRANS = _REPO_ROOT / "lib" / "TileOps" / "a5" / "ttrans.py"


def _load_ttrans():
    spec = importlib.util.spec_from_file_location("ttrans_chunk_span_test", _TTRANS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TtransUi16ChunkSpanTest(unittest.TestCase):
    def test_over_span_stride_raises(self):
        ttrans = _load_ttrans()
        with self.assertRaises(ValueError) as ctx:
            ttrans._ensure_ui16_chunk_span(128, 544)
        message = str(ctx.exception)
        self.assertIn("128-lane chunk spans 69632 element offsets", message)
        self.assertIn("65535", message)
        self.assertIn("no chunking can bring this back in range", message)

    def test_span_bound_is_inclusive(self):
        ttrans = _load_ttrans()
        # 128 * 512 = 65536: the widest offset is exactly 65535, still in range.
        self.assertIsNone(ttrans._ensure_ui16_chunk_span(128, 512))


if __name__ == "__main__":
    unittest.main()
