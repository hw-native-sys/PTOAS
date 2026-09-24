#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Precision requests must survive tracing and materialize scalar SoftLib."""

import json

from ptoas.mlir.ir import InsertionPoint, Location, Module
from ptodsl import pto
from ptodsl._context import make_context
from ptodsl.softlib._compiler_runtime import materialize


def test_scalar_div_precision_attribute():
    with make_context(), Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            lhs, rhs = pto.f32(1.0), pto.f32(7.0)
            pto.div(lhs, rhs, precision=pto.DivPrecision.HighPrecision)
        assert "precisionType = #pto<div_precision high_precision>" in str(module)
        assert module.operation.verify()


def test_scalar_div_precision_normalizes_names_and_rejects_unknown_values():
    for value in ("HIGH_PRECISION", pto.DivPrecision.HighPrecision):
        with make_context(), Location.unknown():
            module = Module.create()
            with InsertionPoint(module.body):
                pto.div(pto.f32(1.0), pto.f32(7.0), precision=value)
            assert "precisionType = #pto<div_precision high_precision>" in str(module)
            assert module.operation.verify()

    with make_context(), Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            try:
                pto.div(pto.f32(1.0), pto.f32(7.0), precision="high")
            except ValueError as error:
                assert "div precision" in str(error)
            else:
                raise AssertionError("invalid scalar division precision was accepted")


def test_scalar_div_softlib_materializes():
    module, entry = materialize(
        "a5", "pto.divf", '{"dtype":"f32","precision":"high_precision"}', make_context()
    )
    assert entry == "div_f32_soft"
    assert module.operation.verify()
    text = str(module)
    assert "pto.divi" in text
    assert "pto.select" in text
    assert "pto.divf" in text


def test_reject_integer_precision():
    with make_context(), Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            try:
                pto.div(pto.i32(1), pto.i32(7), precision=pto.DivPrecision.HighPrecision)
            except TypeError as error:
                assert "floating-point" in str(error)
            else:
                raise AssertionError("integer precision request was accepted")


def test_vector_precision_attribute():
    with make_context(), Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            lhs = pto.vbr(pto.f32(1.0))
            rhs = pto.vbr(pto.f32(7.0))
            mask = pto.pset_b32("PAT_ALL")
            pto.vdiv(lhs, rhs, mask, precision=pto.DivPrecision.HighPrecision)
        assert "precisionType = #pto<div_precision high_precision>" in str(module)
        assert module.operation.verify()


def test_vector_div_precision_normalizes_names_and_rejects_unknown_values():
    with make_context(), Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            lhs = pto.vbr(pto.f32(1.0))
            rhs = pto.vbr(pto.f32(7.0))
            mask = pto.pset_b32("PAT_ALL")
            pto.vdiv(lhs, rhs, mask, precision="HIGH_PRECISION")
            assert "precisionType = #pto<div_precision high_precision>" in str(module)
            assert module.operation.verify()

    with make_context(), Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            lhs = pto.vbr(pto.f32(1.0))
            rhs = pto.vbr(pto.f32(7.0))
            mask = pto.pset_b32("PAT_ALL")
            try:
                pto.vdiv(lhs, rhs, mask, precision="high")
            except ValueError as error:
                assert "vdiv precision" in str(error)
            else:
                raise AssertionError("invalid vector division precision was accepted")


def test_vector_precision_materializes():
    module, _ = materialize("a5", "pto.vdiv", json.dumps({
        "dtype": "f32", "lanes": 64, "mask": "b32",
        "precision": "high_precision",
    }), make_context())
    assert module.operation.verify()
    text = str(module)
    assert text.count("pto.vmula") == 3
    assert "pto.vbitcast" in text
    assert text.count("pto.vdiv") == 1


def test_vector_precision_residual_uses_rhs_times_quotient_minus_lhs():
    module, _ = materialize("a5", "pto.vdiv", json.dumps({
        "dtype": "f32", "lanes": 64, "mask": "b32",
        "precision": "high_precision",
    }), make_context())
    function = list(module.body.operations)[0]
    block = function.regions[0].blocks[0]
    vmula_ops = [op for op in block.operations if op.name == "pto.vmula"]
    assert len(vmula_ops) == 3

    for op in vmula_ops:
        negative_lhs = op.operands[0]
        assert negative_lhs.owner.name == "pto.vmuls"
        assert negative_lhs.owner.operands[0] == block.arguments[0]
        assert op.operands[1] == block.arguments[1]
        assert op.operands[2] != block.arguments[1]


def test_vector_precision_zeroes_inactive_lanes():
    module, _ = materialize("a5", "pto.vdiv", json.dumps({
        "dtype": "f32", "lanes": 64, "mask": "b32",
        "precision": "high_precision",
    }), make_context())
    function = list(module.body.operations)[0]
    block = function.regions[0].blocks[0]
    vsel_ops = [op for op in block.operations if op.name == "pto.vsel"]
    final = vsel_ops[-1]
    assert final.operands[2] == block.arguments[2]
    assert final.operands[1].owner.name == "pto.vbr"


def test_vector_precision_computes_with_full_mask_before_zeroing():
    module, _ = materialize("a5", "pto.vdiv", json.dumps({
        "dtype": "f32", "lanes": 64, "mask": "b32",
        "precision": "high_precision",
    }), make_context())
    function = list(module.body.operations)[0]
    block = function.regions[0].blocks[0]
    assert any(op.name == "pto.pset_b32" for op in block.operations)
    masked_ops = [
        op for op in block.operations
        if op.name in {"pto.vdiv", "pto.vadds", "pto.vmuls", "pto.vmula", "pto.vcmp", "pto.vcmps"}
    ]
    assert masked_ops
    assert all(op.operands[-1] != block.arguments[2] for op in masked_ops)

def test_reject_unsupported_softlib_precision():
    requests = [
        ("a3", "pto.divf", '{"dtype":"f32","precision":"high_precision"}'),
        ("a3", "pto.vdiv", '{"dtype":"f32","mask":"b32","lanes":64,"precision":"high_precision"}'),
        ("a5", "pto.vdiv", '{"dtype":"f32","mask":"b16","lanes":64,"precision":"high_precision"}'),
    ]
    for target, op, specs in requests:
        try:
            materialize(target, op, specs, make_context())
        except ValueError as error:
            assert "requires" in str(error)
        else:
            raise AssertionError(f"unsupported precision request accepted: {target}, {specs}")


def test_vector_precision_uses_ties_to_even_for_equal_residuals():
    module, _ = materialize("a5", "pto.vdiv", json.dumps({
        "dtype": "f32", "lanes": 64, "mask": "b32",
        "precision": "high_precision",
    }), make_context())
    text = str(module)
    assert text.count('"lt"') == 2
    assert text.count('"eq"') == 5
    assert text.count("pto.vand") == 2
    assert text.count("pto.pand") == 2
    assert text.count("pto.por") == 3

    # The first tie uses original's LSB; the second tie must use the LSB of
    # the candidate selected by the first comparison.
    lines = text.splitlines()
    vsel_lines = [i for i, line in enumerate(lines) if "pto.vsel" in line]
    best_bitcast = next(i for i, line in enumerate(lines) if "pto.vbitcast" in line and i > vsel_lines[0])
    assert any(i > best_bitcast for i in vsel_lines)


if __name__ == "__main__":
    test_scalar_div_precision_attribute()
    test_scalar_div_precision_normalizes_names_and_rejects_unknown_values()
    test_scalar_div_softlib_materializes()
    test_reject_integer_precision()
    test_vector_precision_attribute()
    test_vector_div_precision_normalizes_names_and_rejects_unknown_values()
    test_vector_precision_materializes()
    test_reject_unsupported_softlib_precision()
    test_vector_precision_uses_ties_to_even_for_equal_residuals()
