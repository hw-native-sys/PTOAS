#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Unsigned SoftLib ABI and tile restoring-division regressions."""

import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from SoftOps import div_uint

from ptodsl._context import make_context
from ptodsl.softlib._compiler_runtime import materialize


def _lookup(table, key, what):
    """Fetch from a dispatch table with an explicit error instead of KeyError."""
    value = table.get(key)
    if value is None:
        raise ValueError(f"unsupported {what}: {key!r}")
    return value


def _masked(operation, lhs, rhs, mask):
    result = operation(lhs, rhs).astype(np.asarray(lhs).dtype)
    return np.where(mask, result, np.zeros_like(result))


_COMPARES = {"lt": np.less, "ge": np.greater_equal, "ne": np.not_equal, "eq": np.equal}


class _ArrayOps:
    """Execute the restoring division's vector operations with NumPy semantics.

    This checks the loop's quotient and remainder bookkeeping, its truncation
    back to the element width and its mask flow against exact integer division,
    not device instruction behavior. ``_masked`` casts every result back to the
    operand dtype, which is what makes the deliberate overflow of
    ``remainder << 1`` wrap the way the hardware does.
    """

    ui8 = np.uint8
    ui16 = np.uint16
    ui32 = np.uint32
    CmpMode = SimpleNamespace(LT="lt", GE="ge", NE="ne", EQ="eq")

    @staticmethod
    def vbr(value):
        return np.asarray(value)

    @staticmethod
    def vsel(lhs, rhs, mask):
        return np.where(mask, lhs, rhs)

    @staticmethod
    def vcmp(lhs, rhs, mask, mode):
        return _lookup(_COMPARES, mode, "compare mode")(lhs, rhs) & mask

    @staticmethod
    def vcmps(lhs, rhs, mask, mode):
        return _lookup(_COMPARES, mode, "compare mode")(lhs, rhs) & mask

    @staticmethod
    def por(lhs, rhs, mask):
        return (lhs | rhs) & mask

    @staticmethod
    def vor(lhs, rhs, mask):
        return _masked(np.bitwise_or, lhs, rhs, mask)

    @staticmethod
    def vands(lhs, rhs, mask):
        return _masked(np.bitwise_and, lhs, rhs, mask)

    @staticmethod
    def vshls(lhs, rhs, mask):
        return _masked(np.left_shift, lhs, rhs, mask)

    @staticmethod
    def vshrs(lhs, rhs, mask):
        return _masked(np.right_shift, lhs, rhs, mask)

    @staticmethod
    def vsub(lhs, rhs, mask):
        return _masked(np.subtract, lhs, rhs, mask)


class UnsignedSoftLibTest(unittest.TestCase):
    def test_unsigned_helpers_materialize_with_preserved_abi(self):
        for dtype, lanes, mask in (("ui16", 128, "b16"), ("ui32", 64, "b32")):
            with self.subTest(dtype=dtype):
                module, entry = materialize("a5", "pto.vdiv", json.dumps(
                    {"dtype": dtype, "lanes": lanes, "mask": mask}), make_context())
                text = str(module)
                self.assertEqual(entry, "div_" + dtype + "_soft")
                self.assertIn("-> !pto.vreg<" + str(lanes) + "x" + dtype + ">", text)
                self.assertIn("pto.vdiv", text)
                self.assertIn("pto.softlib.instance", text)

    def test_tile_restoring_division_full_range_and_sparse_masks(self):
        rng = np.random.default_rng(1607)
        for bits, dtype in ((16, np.uint16), (32, np.uint32)):
            top = 1 << (bits - 1)
            edges = np.array([0, 1, 2, 3, top - 2, top - 1, top, top + 1,
                              2 * top - 2, 2 * top - 1], dtype=dtype)
            lhs, rhs = np.meshgrid(edges, edges)
            lhs = np.concatenate((lhs.ravel(), rng.integers(0, 2 * top, 100000, dtype=dtype)))
            rhs = np.concatenate((rhs.ravel(), rng.integers(1, 2 * top, 100000, dtype=dtype)))
            for sparse in (False, True):
                mask = np.ones(lhs.shape, dtype=bool)
                if sparse:
                    mask[::3] = False
                expected = np.zeros_like(lhs)
                np.floor_divide(lhs, rhs, out=expected, where=mask & (rhs != 0))
                # A zero divisor is undefined; the loop sets every quotient bit.
                expected[mask & (rhs == 0)] = 2 * top - 1
                with self.subTest(bits=bits, sparse=sparse), \
                        patch.object(div_uint, "pto", _ArrayOps()):
                    result = div_uint.div_uint_soft(lhs, rhs, mask, f"ui{bits}")
                    np.testing.assert_array_equal(result, expected)


if __name__ == "__main__":
    unittest.main()
