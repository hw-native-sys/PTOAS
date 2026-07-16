#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import ast
from pathlib import Path
import sys
import textwrap
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "ptodsl"))

from ptodsl._ast_rewrite import (
    PTODSLAstRewriteError,
    _ControlFlowRewriter,
    _name_info,
)


def rewrite_source(source):
    tree = ast.parse(textwrap.dedent(source))
    function = tree.body[0]
    function.body = _ControlFlowRewriter().rewrite_block(function.body, live_after=set())
    ast.fix_missing_locations(tree)
    compile(tree, "<ast-rewrite-test>", "exec")
    return tree


class AstRewriteRegressionTest(unittest.TestCase):
    def test_sibling_runtime_loop_target_reuse_is_not_live_out(self):
        rewrite_source(
            """
            def probe(rows, cols):
                for t in range(rows):
                    for c in range(cols):
                        col = c * 64
                        mask = col
                    for mo in pto.static_range(4):
                        mix_acc = 0
                        for c in range(cols):
                            col = c * 64
                            mask = col
                            mix_acc = mix_acc + mask
                        sink = mix_acc
            """
        )

    def test_sibling_loop_temporary_reuse_is_not_last_iteration_value(self):
        rewrite_source(
            """
            def probe(rows, cols):
                for t in range(rows):
                    for mo in pto.static_range(4):
                        mix_acc = 0
                        for dpost_chunk in range(cols):
                            col = dpost_chunk * 64
                            cnt = cols - col
                            mask = cnt
                            d_reg = mask
                            mix_acc = mix_acc + d_reg
                        sink = mix_acc
                    for mi in pto.static_range(4):
                        for mo in pto.static_range(4):
                            mix_acc = 0
                            for dcomb_chunk in range(cols):
                                col = dcomb_chunk * 64
                                cnt = cols - col
                                mask = cnt
                                d_reg = mask
                                mix_acc = mix_acc + d_reg
                            sink = mix_acc
            """
        )

    def test_real_runtime_loop_escapes_remain_rejected(self):
        with self.assertRaisesRegex(PTODSLAstRewriteError, "loop induction variable"):
            rewrite_source(
                """
                def probe(rows):
                    for i in range(rows):
                        value = i
                    sink = i
                """
            )
        with self.assertRaisesRegex(PTODSLAstRewriteError, "last-iteration-only"):
            rewrite_source(
                """
                def probe(rows):
                    for i in range(rows):
                        last = i
                    sink = last
                """
            )

    def test_comprehension_targets_do_not_escape_the_comprehension_scope(self):
        for expression in (
            "[h for h in pto.static_range(4)]",
            "{h for h in pto.static_range(4)}",
            "{h: h for h in pto.static_range(4)}",
            "(h for h in pto.static_range(4))",
        ):
            info = _name_info(ast.parse(f"values = {expression}").body[0])
            self.assertNotIn("h", info.loads)
            self.assertNotIn("h", info.stores)

        tree = rewrite_source(
            """
            def probe(rows, cols):
                for t in range(rows):
                    for chunk in range(cols):
                        values = [h for h in pto.static_range(4)]
                        sink = values[0]
            """
        )
        self.assertNotIn("carry(h=h)", ast.unparse(tree))

    def test_direct_assignment_conditional_expression_rewrites_to_a_branch(self):
        tree = rewrite_source(
            """
            def probe(rows, cols):
                for t in range(rows):
                    cnt = cols - t if t < 64 else 64
                    sink = cnt
            """
        )
        self.assertFalse(any(isinstance(node, ast.IfExp) for node in ast.walk(tree)))
        self.assertIn("pto.if_", ast.unparse(tree))

    def test_nested_trace_time_conditional_expressions_are_preserved(self):
        tree = rewrite_source(
            """
            def probe(enabled, cols):
                scaled = (cols if enabled else 0) * 2
                selected = 1 if (enabled if True else False) else 0
                if (enabled if True else False):
                    scaled = scaled + selected
                for index in range(1 if enabled else 0):
                    scaled = scaled + index
                return scaled
            """
        )
        rewritten = ast.unparse(tree)
        self.assertIn("(cols if enabled else 0) * 2", rewritten)
        self.assertIn("enabled if True else False", rewritten)
        self.assertIn("pto.for_(0, 1 if enabled else 0, step=1)", rewritten)


if __name__ == "__main__":
    unittest.main()
