#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""End-to-end coverage for the PTODSL `@pto.jit(sched_mode=...)` surface.

The frontend only stamps the signless-i32 `pto.sched_mode` attribute on the
generated entry function; PTOAS owns value and placement validation. These
tests start from a DSL kernel, drive the real `ptoas` binary, and check both
backend spellings plus the verifier diagnostic, so a frontend attribute name or
type drift cannot pass unnoticed.
"""

import subprocess
import tempfile
import unittest
from pathlib import Path

from ptodsl import pto
from ptodsl._runtime.toolchain import resolve_ptoas_binary


@pto.jit(target="a5", backend="emitc", sched_mode=1)
def sched_mode_batch_entry(ptr: pto.ptr(pto.f32, "gm")):
    return


@pto.jit(target="a5", backend="emitc")
def sched_mode_default_entry(ptr: pto.ptr(pto.f32, "gm")):
    return


@pto.jit(target="a5", backend="vpto", kernel_kind="vector", sched_mode=1)
def sched_mode_batch_vpto_entry(ptr: pto.ptr(pto.f32, "gm")):
    return


@pto.jit(target="a5", backend="emitc", sched_mode=2)
def sched_mode_invalid_entry(ptr: pto.ptr(pto.f32, "gm")):
    return


class SchedModeEndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ptoas = resolve_ptoas_binary()

    def run_ptoas(self, kernel, *arguments):
        with tempfile.NamedTemporaryFile(
            "w", suffix=".mlir", delete=False, encoding="utf-8"
        ) as handle:
            handle.write(kernel.compile().mlir_text())
            input_path = Path(handle.name)
        try:
            return subprocess.run(
                [str(self.ptoas), str(input_path), *arguments],
                capture_output=True,
                text=True,
                check=False,
            )
        finally:
            input_path.unlink(missing_ok=True)

    def test_emitc_emits_schedmode_before_global(self):
        mlir_text = sched_mode_batch_entry.compile().mlir_text()
        self.assertIn(
            "pto.sched_mode = 1 : i32",
            mlir_text,
            "the DSL must stamp the signless-i32 pto.sched_mode attribute",
        )

        result = self.run_ptoas(
            sched_mode_batch_entry, "--pto-backend=emitc", "-o", "-"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            "__schedmode__(1) __global__ AICORE void sched_mode_batch_entry(",
            result.stdout,
        )

    def test_emitc_unset_kernel_emits_no_schedmode(self):
        mlir_text = sched_mode_default_entry.compile().mlir_text()
        self.assertNotIn("pto.sched_mode", mlir_text)

        result = self.run_ptoas(
            sched_mode_default_entry, "--pto-backend=emitc", "-o", "-"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("__schedmode__", result.stdout)
        self.assertIn(
            "__global__ AICORE void sched_mode_default_entry(",
            result.stdout,
        )

    def test_vpto_carries_launch_schedule_mode_i32(self):
        result = self.run_ptoas(
            sched_mode_batch_vpto_entry,
            "--pto-arch=a5",
            "--pto-backend=vpto",
            "--emit-vpto-llvm-ir",
            "-o",
            "-",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertRegex(
            result.stdout,
            r'!\{ptr @sched_mode_batch_vpto_entry\w*, !"launch-schedule-mode", i32 1\}',
        )

    def test_invalid_value_is_rejected_with_the_verifier_diagnostic(self):
        mlir_text = sched_mode_invalid_entry.compile().mlir_text()
        self.assertIn(
            "pto.sched_mode = 2 : i32",
            mlir_text,
            "the DSL must not silently drop or rewrite the value",
        )

        result = self.run_ptoas(
            sched_mode_invalid_entry, "--pto-backend=emitc", "-o", "-"
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "`pto.sched_mode` must be 0 (normal mode) or 1 (batch mode), got 2",
            result.stderr,
        )


class SchedModeOptionValidationTests(unittest.TestCase):
    """The Python-side contract the IR verifier cannot express.

    PTOAS owns the accepted value set and the entry-placement rule; these tests
    pin the two things it cannot see from the IR: the argument has to be an int
    in the signless-i32 range (so nothing wraps or coerces into a legal mode),
    and it has to reach the IR at all (``entry=False`` and ``source=`` cannot
    carry it).
    """

    def assert_option_error(self, error, pattern, **options):
        with self.assertRaisesRegex(error, pattern):
            pto.jit(target="a5", **options)

    def test_bool_is_rejected_even_though_true_equals_one(self):
        self.assert_option_error(TypeError, "must be an int", sched_mode=True)
        self.assert_option_error(TypeError, "must be an int", sched_mode=False)

    def test_non_integer_values_are_rejected(self):
        for value in (1.0, "1", [1], (1,), object()):
            with self.subTest(value=value):
                self.assert_option_error(TypeError, "must be an int", sched_mode=value)

    def test_values_that_cannot_fit_i32_are_rejected_before_they_wrap(self):
        # 2**31 wraps to a negative mode and 2**32 + 1 wraps to the valid 1;
        # neither may reach the attribute.
        for value in (2**31, -(2**31) - 1, 2**32 + 1):
            with self.subTest(value=value):
                self.assert_option_error(
                    ValueError, "does not fit in the signless-i32", sched_mode=value
                )

    def test_source_backed_kernel_rejects_sched_mode(self):
        # A source-backed kernel takes its attributes from the IR file, so the
        # argument used to be dropped silently.
        self.assert_option_error(
            TypeError, "do not support sched_mode", source="kernel.pto", sched_mode=1
        )

    def test_non_entry_kernel_rejects_sched_mode(self):
        # The attribute is kernel-only; failing at decoration beats failing at
        # codegen with a placement diagnostic about an attribute the user never
        # wrote by hand.
        self.assert_option_error(
            TypeError, "do not support sched_mode", entry=False, sched_mode=1
        )

    def test_legal_values_still_reach_the_attribute(self):
        def kernel(ptr: pto.ptr(pto.f32, "gm")):
            return

        for value in (0, 1):
            with self.subTest(value=value):
                handle = pto.jit(target="a5", backend="emitc", sched_mode=value)(kernel)
                self.assertIn(
                    f"pto.sched_mode = {value} : i32", handle.compile().mlir_text()
                )

    def test_numpy_integer_scalars_are_accepted(self):
        # Host code routinely carries numpy scalars; they are integral, and they
        # used to reach IntegerAttr.get unchanged.
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy is not installed")

        def kernel(ptr: pto.ptr(pto.f32, "gm")):
            return

        for value in (np.int8(1), np.int32(1), np.int64(0), np.uint32(1), np.uint64(1)):
            with self.subTest(dtype=type(value).__name__, value=value):
                handle = pto.jit(target="a5", backend="emitc", sched_mode=value)(kernel)
                self.assertIn(
                    f"pto.sched_mode = {int(value)} : i32", handle.compile().mlir_text()
                )

    def test_numpy_values_outside_i32_are_still_rejected(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy is not installed")

        # np.uint64 holds 2**32 + 1 exactly; narrowing it must not reach the IR.
        self.assert_option_error(
            ValueError, "does not fit in the signless-i32", sched_mode=np.uint64(2**32 + 1)
        )
        # np.bool_ is not integral, so it takes the type diagnostic.
        self.assert_option_error(TypeError, "must be an int", sched_mode=np.bool_(True))

    def test_omitting_sched_mode_leaves_the_attribute_off(self):
        def kernel(ptr: pto.ptr(pto.f32, "gm")):
            return

        handle = pto.jit(target="a5", backend="emitc")(kernel)
        self.assertNotIn("pto.sched_mode", handle.compile().mlir_text())


if __name__ == "__main__":
    unittest.main()
