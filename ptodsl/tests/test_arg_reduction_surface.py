# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import inspect
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from ptoas.mlir import ir
from ptoas.mlir.dialects import func

from ptodsl import pto
from ptodsl._context import make_context
import ptodsl._ops as _ops


class ArgReductionSurfaceTest(unittest.TestCase):
    def test_tile_arg_reductions_forward_optional_outputs_and_tmp(self):
        for name, low_level_name in (
            ("rowargmax", "trowargmax"),
            ("rowargmin", "trowargmin"),
            ("colargmax", "tcolargmax"),
            ("colargmin", "tcolargmin"),
        ):
            with self.subTest(operation=name):
                self._assert_arg_reduction_wrapper(name, low_level_name)

    def test_generated_bindings_accept_optional_dst_val(self):
        with make_context(), ir.Location.unknown():
            tile_types = [
                ir.Type.parse("!pto.tile_buf<vec, 8x16xf32>"),
                ir.Type.parse("!pto.tile_buf<vec, 8x16xi32>"),
                ir.Type.parse("!pto.tile_buf<vec, 8x16xf32>"),
                ir.Type.parse("!pto.tile_buf<vec, 8x16xf32>"),
            ]
            module = ir.Module.create()
            with ir.InsertionPoint(module.body):
                function = func.FuncOp("arg_reduction_probe", (tile_types, []))
            block = function.add_entry_block()
            src, dst_idx, dst_val, tmp = block.arguments
            with ir.InsertionPoint(block):
                self._emit_all_arg_reduction_forms(src, dst_idx, dst_val, tmp)
                func.ReturnOp([])

        self._assert_optional_dst_val_segments(str(module))

    def _emit_all_arg_reduction_forms(self, src, dst_idx, dst_val, tmp):
        for wrapper_name in (
            "trowargmax",
            "trowargmin",
            "tcolargmax",
            "tcolargmin",
        ):
            wrapper = getattr(_ops, wrapper_name)
            wrapper(src, dst_idx)
            wrapper(src, dst_idx, dst_val, tmp=tmp)

    def _assert_optional_dst_val_segments(self, module_text):
        for op_name in (
            "trowargmax",
            "trowargmin",
            "tcolargmax",
            "tcolargmin",
        ):
            with self.subTest(operation=op_name):
                op_lines = [
                    line
                    for line in module_text.splitlines()
                    if f'"pto.{op_name}"' in line
                ]
                self.assertEqual(len(op_lines), 2)
                self.assertIn("array<i32: 1, 0, 1, 0>", op_lines[0])
                self.assertIn("array<i32: 1, 1, 1, 1>", op_lines[1])

    def _assert_arg_reduction_wrapper(self, name, low_level_name):
        self._assert_signature(name)
        src = SimpleNamespace(type="src_ty")
        dst_idx = object()
        dst_val = object()
        tmp = object()
        with patch.object(_ops, low_level_name) as low_level_op:
            getattr(pto.tile, name)(src, dst_idx)
        low_level_op.assert_called_once_with(src, dst_idx, None, tmp=None)

        with patch.object(_ops, low_level_name) as low_level_op:
            getattr(pto.tile, name)(src, dst_idx, dst_val, tmp=tmp)
        low_level_op.assert_called_once_with(src, dst_idx, dst_val, tmp=tmp)

    def _assert_signature(self, name):
        signature = inspect.signature(getattr(pto.tile, name))
        self.assertEqual(
            list(signature.parameters), ["src", "dst_idx", "dst_val", "tmp"]
        )
        self.assertIsNone(signature.parameters["dst_val"].default)
        self.assertEqual(signature.parameters["tmp"].kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertIsNone(signature.parameters["tmp"].default)


if __name__ == "__main__":
    unittest.main()
