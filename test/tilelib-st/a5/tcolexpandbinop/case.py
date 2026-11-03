#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""PTODSL TileLib ST coverage for the A5 column-expand binary family.

The seven column-expand binary TileOps (``tcolexpandadd/sub/mul/div/max/min/expdif``)
share the TileLib template ``lib/TileOps/a5/_expand_binary.py``'s
``_emit_column_expand_body`` (``tcolexpanddiv`` also has an integer variant).  The
broadcast row may come from either input -- the full-sized operand is the one whose
valid shape matches ``dst``, with ``src0`` taking precedence::

    src1 expanded:  dst[r, c] = src0[r, c] OP src1[0, c]
    src0 expanded:  dst[r, c] = src0[0, c] OP src1[r, c]

Coverage in this suite: all seven interfaces in both directions on an f32 16x64 dst
with a 1x64 broadcast row, a 16-bit case per direction, and a dst width below one
vector register (partial-mask chunk) in the src0-expanded direction.

This suite is PTODSL auto mode, so it compiles through
``ptoas --pto-backend=vpto --enable-tile-op-expand`` -- the path that lowers the
column-expand template.  See ``test/tilelib-st/README.md``.
"""

from pathlib import Path
import sys
import zlib

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import auto_main, golden_output_case
from ptodsl import pto


_DTYPE = {
    "f32": (pto.f32, np.float32),
    "f16": (pto.f16, np.float16),
    "i32": (pto.i32, np.int32),
    "i16": (pto.i16, np.int16),
}


def _exp_dif(lhs, rhs):
    """``tcolexpandexpdif`` semantics: ``exp(src0 - src1)``."""
    return np.exp(lhs - rhs)


def _divide(lhs, rhs, dtype_key):
    """Division matching the kernel: signed integer division truncates toward zero."""
    if np.issubdtype(_DTYPE[dtype_key][1], np.integer):
        lhs64 = lhs.astype(np.int64)
        rhs64 = rhs.astype(np.int64)
        quotient = np.abs(lhs64) // np.abs(rhs64)
        return np.where((lhs64 < 0) ^ (rhs64 < 0), -quotient, quotient)
    return np.divide(lhs, rhs)


# op name -> (PTODSL tile op, NumPy reference, floating-point only?)
_OPS = {
    "add": (pto.tile.colexpandadd, np.add, False),
    "sub": (pto.tile.colexpandsub, np.subtract, False),
    "mul": (pto.tile.colexpandmul, np.multiply, False),
    "div": (pto.tile.colexpanddiv, np.divide, True),
    "max": (pto.tile.colexpandmax, np.maximum, False),
    "min": (pto.tile.colexpandmin, np.minimum, False),
    "expdif": (pto.tile.colexpandexpdif, _exp_dif, True),
}

INTERFACES = tuple(_OPS)


# ── kernel factory ────────────────────────────────────────────────────────────

def _make_kernel(case_name, op_name, dtype_key, rows, cols, expanded):
    """Build one ``@pto.jit`` kernel for one (op, dtype, shape, direction).

    ``expanded`` says which operand carries the broadcast row (``"src0"`` or
    ``"src1"``); the other one matches ``dst``.  Column expand requires row-major
    tiles for all three operands.
    """
    pto_dtype, _np_dtype = _DTYPE[dtype_key]
    op_fn = _OPS[op_name][0]
    src0_is_full = expanded == "src1"
    full_shape, row_shape = [rows, cols], [1, cols]
    src0_shape = full_shape if src0_is_full else row_shape
    src1_shape = row_shape if src0_is_full else full_shape

    @pto.jit(name=case_name, target="a5")
    def _kernel(
        src0_ptr: pto.ptr(pto_dtype, "gm"),
        src1_ptr: pto.ptr(pto_dtype, "gm"),
        dst_ptr: pto.ptr(pto_dtype, "gm"),
    ):
        src0_view = pto.make_tensor_view(src0_ptr, shape=src0_shape, strides=[cols, 1])
        src1_view = pto.make_tensor_view(src1_ptr, shape=src1_shape, strides=[cols, 1])
        dst_view = pto.make_tensor_view(dst_ptr, shape=[rows, cols], strides=[cols, 1])

        src0 = pto.alloc_tile(shape=src0_shape, dtype=pto_dtype, valid_shape=src0_shape)
        src1 = pto.alloc_tile(shape=src1_shape, dtype=pto_dtype, valid_shape=src1_shape)
        dst = pto.alloc_tile(shape=[rows, cols], dtype=pto_dtype, valid_shape=[rows, cols])

        pto.tile.load(src0_view, src0)
        pto.tile.load(src1_view, src1)
        op_fn(src0, src1, dst)
        pto.tile.store(dst, dst_view)

    return _kernel


# ── inputs / golden ───────────────────────────────────────────────────────────

def _make_inputs(case_name, op_name, dtype_key, rows, cols, expanded):
    """Deterministic per-case inputs, keeping every op in a well-conditioned range."""
    _pto_dtype, np_dtype = _DTYPE[dtype_key]
    rng = np.random.default_rng(zlib.crc32(case_name.encode("utf-8")) & 0xFFFFFFFF)

    def _draw(shape, kind):
        if np.issubdtype(np_dtype, np.integer):
            # Small positive values: integer division stays exact and well-defined.
            return rng.integers(1, 5, size=shape).astype(np_dtype)
        if op_name == "div":
            # The divisor stays bounded away from zero whichever operand it is.
            low, high = (0.5, 2.0) if kind == "expanded" else (1.0, 4.0)
            return rng.uniform(low, high, size=shape).astype(np_dtype)
        if op_name == "expdif":
            # Keep exp() in range.
            return rng.uniform(-1.0, 1.0, size=shape).astype(np_dtype)
        return rng.uniform(0.25, 2.0, size=shape).astype(np_dtype)

    full_shape, row_shape = (rows, cols), (1, cols)
    if expanded == "src0":
        return [_draw(row_shape, "expanded"), _draw(full_shape, "full")]
    return [_draw(full_shape, "full"), _draw(row_shape, "expanded")]


def _make_expected(op_name, dtype_key, expanded, src0, src1):
    """Golden for both broadcast directions."""
    _pto_dtype, np_dtype = _DTYPE[dtype_key]
    full, row = (src0, src1) if expanded == "src1" else (src1, src0)
    broadcast = np.broadcast_to(row[:1, :], full.shape)
    # Keep the authored operand order: dst = src0 OP src1.
    if op_name == "div":
        result = (
            _divide(full, broadcast, dtype_key)
            if expanded == "src1"
            else _divide(broadcast, full, dtype_key)
        )
    else:
        op_fn = _OPS[op_name][1]
        result = op_fn(full, broadcast) if expanded == "src1" else op_fn(broadcast, full)
    return result.astype(np_dtype)


def _tolerances(op_name, dtype_key):
    if np.issubdtype(_DTYPE[dtype_key][1], np.integer):
        return 0.0, 0.0
    if dtype_key == "f16":
        return 1e-2, 1e-2
    if op_name == "expdif":
        return 1e-4, 1e-4
    if op_name == "div":
        return 1e-5, 1e-5
    return 1e-6, 1e-6


# ── case matrix ───────────────────────────────────────────────────────────────

# (op, dtype, rows, cols, expanded)
# Primary shape: f32 16x64 dst with a 1x64 broadcast row.
_PRIMARY_SHAPE = (16, 64)

CASE_SPECS = []

# 1) Every interface with src1 carrying the broadcast row ...
for _op in INTERFACES:
    CASE_SPECS.append((_op, "f32", _PRIMARY_SHAPE[0], _PRIMARY_SHAPE[1], "src1"))
# 2) ... and again with src0 carrying it, which also pins the authored operand order
#    for the non-commutative interfaces.
for _op in INTERFACES:
    CASE_SPECS.append((_op, "f32", _PRIMARY_SHAPE[0], _PRIMARY_SHAPE[1], "src0"))

# 3) Extra scenarios.
CASE_SPECS += [
    # 16-bit elements in both directions.
    ("add", "f16", 16, 64, "src1"),
    ("add", "f16", 16, 64, "src0"),
    # Integer division, both directions: the inlined integer body shares the same
    # direction selection as the floating-point one.
    ("div", "i32", 16, 64, "src1"),
    ("div", "i32", 16, 64, "src0"),
    ("div", "i16", 16, 64, "src0"),
    # dst width below one vector register of f32 (64 lanes): a single masked chunk.
    ("sub", "f32", 16, 40, "src0"),
]


def _case_name(op, dtype_key, rows, cols, expanded):
    return f"tcolexpand{op}_{dtype_key}_{rows}x{cols}_{expanded}"


def _build_cases():
    cases = []
    for op, dtype_key, rows, cols, expanded in CASE_SPECS:
        name = _case_name(op, dtype_key, rows, cols, expanded)
        kernel = _make_kernel(name, op, dtype_key, rows, cols, expanded)
        rtol, atol = _tolerances(op, dtype_key)

        def _inputs(_name=name, _op=op, _dt=dtype_key, _r=rows, _c=cols, _expanded=expanded):
            return _make_inputs(_name, _op, _dt, _r, _c, _expanded)

        def _expected(src0, src1, _op=op, _dt=dtype_key, _expanded=expanded):
            return _make_expected(_op, _dt, _expanded, src0, src1)

        cases.append(
            golden_output_case(
                name,
                kernel,
                inputs=_inputs,
                expected=_expected,
                rtol=rtol,
                atol=atol,
            )
        )
    return cases


CASES = _build_cases()


auto_main(globals())
