# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Check the exceptional-value predicate of the A5 fmod lowering."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import ptodsl.tilelib.templates.a5._remainder as remainder
from ptodsl.tilelib.templates.a5.elementwise.tfmods import template_tfmods


def compare_equal(lhs, rhs, mask, mode):
    return np.logical_and(np.equal(lhs, rhs), mask)


def predicate_and(lhs, rhs, mask):
    return np.logical_and(np.logical_and(lhs, rhs), mask)


def predicate_not(value, mask):
    return np.logical_and(np.logical_not(value), mask)


def compare(lhs, rhs, mask, mode):
    if mode == "le":
        return np.logical_and(np.less_equal(lhs, rhs), mask)
    return compare_equal(lhs, rhs, mask, mode)


class TileLibRemainderTest(unittest.TestCase):
    def test_zero_sign_is_restored_only_for_nonzero_non_nan_divisor(self):
        operations = SimpleNamespace(
            f32=np.float32,
            CmpMode=SimpleNamespace(EQ="eq", LE="le"),
            vcmps=compare,
            vcmp=compare_equal,
            pand=predicate_and,
            pnot=predicate_not,
            por=lambda lhs, rhs, mask: np.logical_and(np.logical_or(lhs, rhs), mask),
            vabs=lambda value, mask: np.abs(value),
            vmul=lambda lhs, rhs, mask: np.multiply(lhs, rhs),
            vmuls=lambda lhs, rhs, mask: np.multiply(lhs, rhs),
            vdup=lambda value, mask: value,
            vsel=lambda yes, no, mask: np.where(mask, yes, no),
            pset_b16=lambda pattern: True,
            pset_b32=lambda pattern: True,
            PAT=SimpleNamespace(ALL="all"),
            VcvtPartMode=SimpleNamespace(EVEN="even", ODD="odd"),
            VcvtSatMode=SimpleNamespace(NOSAT="nosat"),
            bf16="bf16",
            vbr=lambda value: value,
            vcvt=lambda value, dtype, mask, **kwargs: value,
            vor=lambda even, odd, mask: even,
        )
        dividend = np.array([0.0, -0.0], dtype=np.float32)
        initial = np.full(2, np.nan, dtype=np.float32)
        for divisor in (3.0, -3.0, np.inf, -np.inf, 0.0, -0.0, np.nan):
            with self.subTest(divisor=divisor), patch.object(remainder, "pto", operations), patch.object(
                remainder, "_f32_solve_scale", return_value=initial
            ), patch.object(remainder, "_f32_sign_multiplier", return_value=np.ones(2, dtype=np.float32)):
                source = SimpleNamespace(values=dividend)
                destination = SimpleNamespace(dtype="bf16", values=None)
                with patch.object(remainder, "emit_scalar_binary_2d", side_effect=emit_scalar_chunk):
                    template_tfmods.py_fn(source, divisor, destination)
                actual = destination.values
                with np.errstate(invalid="ignore"):
                    expected = np.fmod(dividend, np.float32(divisor))
                np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
                finite = np.isfinite(expected)
                np.testing.assert_array_equal(actual[finite].view(np.uint32), expected[finite].view(np.uint32))


def emit_scalar_chunk(source, scalar, destination, operation):
    destination.values = operation(source.values, scalar, True)


if __name__ == "__main__":
    unittest.main()
