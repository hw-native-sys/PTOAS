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
from collections import Counter

import pytest
from ptoas.mlir.ir import InsertionPoint, Location, MLIRError, Module
from ptodsl import pto
from ptodsl._context import make_context
from ptodsl.softlib._compiler_runtime import (
    _resolve_softlib_request,
    materialize,
)


def test_scalar_div_precision_attribute():
    with make_context(), Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            lhs, rhs = pto.f32(1.0), pto.f32(7.0)
            pto.div(lhs, rhs, precision=pto.DivPrecision.Precision0UlpFtzTrue)
        assert "precisionType = #pto<div_precision precision_0ulp_ftz_true>" in str(module)
        assert module.operation.verify()


def test_scalar_div_precision_normalizes_names_and_rejects_unknown_values():
    for value in ("0ULP_FTZ_TRUE", pto.DivPrecision.Precision0UlpFtzTrue):
        with make_context(), Location.unknown():
            module = Module.create()
            with InsertionPoint(module.body):
                pto.div(pto.f32(1.0), pto.f32(7.0), precision=value)
            assert "precisionType = #pto<div_precision precision_0ulp_ftz_true>" in str(module)
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
        "a5", "pto.divf", '{"dtype":"f32","precision":"precision_0ulp_ftz_true"}', make_context()
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
                pto.div(pto.i32(1), pto.i32(7), precision=pto.DivPrecision.Precision0UlpFtzTrue)
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
            pto.vdiv(lhs, rhs, mask, precision=pto.DivPrecision.Precision0UlpFtzTrue)
        assert "precisionType = #pto<div_precision precision_0ulp_ftz_true>" in str(module)
        assert module.operation.verify()


def test_vector_div_precision_normalizes_names_and_rejects_unknown_values():
    with make_context(), Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            lhs = pto.vbr(pto.f32(1.0))
            rhs = pto.vbr(pto.f32(7.0))
            mask = pto.pset_b32("PAT_ALL")
            pto.vdiv(lhs, rhs, mask, precision="0ULP_FTZ_TRUE")
            assert "precisionType = #pto<div_precision precision_0ulp_ftz_true>" in str(module)
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


# Tokens AscendC maps to the native vector divide, so no SoftLib expansion.
NATIVE_DIV_ALGO_TOKENS = ("intrinsic", "precision_1ulp_ftz_true")

# f32 tokens that expand through SoftLib, and the SoftOps helper they land on.
SOFTWARE_DIV_ALGO_HELPERS = {
    "diff_compensation": "div_vf32_soft",
    "precision_0ulp_ftz_true": "div_vf32_soft",
    "precision_0ulp_ftz_false": "div_vf32_ulp0_ftz_false",
    "precision_1ulp_ftz_false": "div_vf32_ulp1_ftz_false",
}


def _vector_div_softop(token="precision_0ulp_ftz_true"):
    module, entry = materialize("a5", "pto.vdiv", json.dumps({
        "dtype": "f32", "lanes": 64, "mask": "b32",
        "precision": token,
    }), make_context())
    assert entry == SOFTWARE_DIV_ALGO_HELPERS[token]
    assert module.operation.verify()
    function = list(module.body.operations)[0]
    return module, function.regions[0].blocks[0]


def test_vector_precision_materializes():
    _module, block = _vector_div_softop()
    names = [op.name for op in block.operations]
    assert names.count("pto.vdiv") == 1
    assert names.count("pto.vmula") == 3
    assert "pto.vbitcast" in names


def test_vector_precision_matches_the_cann_sequence():
    """Mirror AscendC ``DivPrecisionImpl`` operation for operation."""
    _module, block = _vector_div_softop()
    counts = Counter(op.name for op in block.operations)
    assert counts["pto.vdiv"] == 1  # hardware divide
    assert counts["pto.vmuls"] == 1  # y = -src1
    assert counts["pto.vor"] == 1  # infNan = z | 0x80000000
    assert counts["pto.vcmps"] == 2  # zero, inf/nan bound
    assert counts["pto.por"] == 1  # infNanCmp |= zeroCmp
    assert counts["pto.vmula"] == 3  # r, rPre, rNext
    assert counts["pto.vabs"] == 3
    assert counts["pto.vadds"] == 2  # bit neighbours z-1 / z+1
    assert counts["pto.vcmp"] == 2  # r < rPre, rNext < r
    assert counts["pto.vsel"] == 4  # residual, quotient, following, special
    # The unscaled sequence is deliberate: the dav_3510 exponent pre-scale costs
    # 11 instructions per division and is not implemented here.
    assert counts["pto.vand"] == 0
    assert counts["pto.vshrs"] == 0
    assert counts["pto.vsub"] == 0
    assert counts["pto.vshls"] == 0
    assert counts["pto.vmul"] == 0
    # No round-half-even machinery and no full-width staging mask.
    assert counts["pto.pand"] == 0
    assert counts["pto.pset_b32"] == 0


def test_vector_precision_residual_uses_cann_vmula_form():
    """CANN accumulates the residual as src0 + z * (-src1) with fused vmula."""
    _module, block = _vector_div_softop()
    lhs, rhs = block.arguments[0], block.arguments[1]
    quotient = next(op for op in block.operations if op.name == "pto.vdiv").results[0]
    residuals = [op for op in block.operations if op.name == "pto.vmula"]
    assert len(residuals) == 3
    for op in residuals:
        assert op.operands[0] == lhs
        negated = op.operands[2]
        assert negated.owner.name == "pto.vmuls"
        assert negated.owner.operands[0] == rhs
    assert residuals[0].operands[1] == quotient
    # Every residual is absolute-valued before it feeds a comparison.
    for op in residuals:
        assert any(use.owner.name == "pto.vabs" for use in op.results[0].uses)


def test_vector_precision_predicates_every_op_with_the_caller_mask():
    """CANN predicates each instruction; there is no PAT_ALL staging pass."""
    _module, block = _vector_div_softop()
    mask = block.arguments[2]
    predicated_names = {
        "pto.vdiv", "pto.vmuls", "pto.vor", "pto.vcmps", "pto.por", "pto.vmula",
        "pto.vabs", "pto.vadds", "pto.vcmp",
    }
    predicated = [op for op in block.operations if op.name in predicated_names]
    assert predicated
    assert all(op.operands[-1] == mask for op in predicated)
    # vsel carries the computed predicate in its last slot (CANN's asc_select).
    selects = [op for op in block.operations if op.name == "pto.vsel"]
    assert len(selects) == 4
    assert all(str(op.operands[-1].type).startswith("!pto.mask") for op in selects)
    # No full-width staging mask and no zero-fill of inactive lanes.
    assert not any(op.name == "pto.pset_b32" for op in block.operations)
    assert selects[-1].operands[1].owner.name != "pto.vbr"


def _vmi_div_operands(lanes=64):
    lhs = pto.vmi.vbrc(pto.f32(1.0), size=lanes)
    rhs = pto.vmi.vbrc(pto.f32(7.0), size=lanes)
    mask = pto.vmi.create_mask(lanes, size=lanes)
    return lhs, rhs, mask


def test_vmi_vdiv_precision_attribute():
    for value in ("0ULP_FTZ_TRUE", pto.DivPrecision.Precision0UlpFtzTrue):
        with make_context(), Location.unknown():
            module = Module.create()
            with InsertionPoint(module.body):
                lhs, rhs, mask = _vmi_div_operands()
                pto.vmi.vdiv(lhs, rhs, mask, precision=value)
            assert "precisionType = #pto<div_precision precision_0ulp_ftz_true>" in str(module)
            assert module.operation.verify()


def test_vmi_vdiv_default_precision_has_no_attribute():
    with make_context(), Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            lhs, rhs, mask = _vmi_div_operands()
            pto.vmi.vdiv(lhs, rhs, mask)
        assert "precisionType" not in str(module)
        assert module.operation.verify()


def test_vmi_vdiv_precision_rejects_unknown_values():
    with make_context(), Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            lhs, rhs, mask = _vmi_div_operands()
            try:
                pto.vmi.vdiv(lhs, rhs, mask, precision="high")
            except ValueError as error:
                assert "vdiv precision" in str(error)
            else:
                raise AssertionError("invalid VMI division precision was accepted")


DIV_ALGO_TOKENS = (
    "intrinsic",
    "diff_compensation",
    "precision_1ulp_ftz_true",
    "precision_0ulp_ftz_true",
    "precision_0ulp_ftz_false",
    "precision_1ulp_ftz_false",
)


def _vdiv_with_algo(token):
    with make_context(), Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            lhs = pto.vbr(pto.f32(1.0))
            rhs = pto.vbr(pto.f32(7.0))
            mask = pto.pset_b32("PAT_ALL")
            pto.vdiv(lhs, rhs, mask, precision=token)
        return module


def test_div_algo_tokens_round_trip_through_the_ir():
    """All six AscendC DivAlgo spellings are accepted and printed verbatim."""
    for token in DIV_ALGO_TOKENS:
        module = _vdiv_with_algo(token)
        text = str(module)
        mnemonic = {
            "intrinsic": None,  # default selector: elided when printed
            "diff_compensation": "diff_compensation",
            "precision_1ulp_ftz_true": "precision_1ulp_ftz_true",
            "precision_0ulp_ftz_true": "precision_0ulp_ftz_true",
            "precision_0ulp_ftz_false": "precision_0ulp_ftz_false",
            "precision_1ulp_ftz_false": "precision_1ulp_ftz_false",
        }[token]
        if mnemonic is None:
            assert "precisionType" not in text
        else:
            assert f"precisionType = #pto<div_precision {mnemonic}>" in text
        assert module.operation.verify()


def test_div_algo_rejects_the_retired_high_precision_token():
    with make_context(), Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            lhs = pto.vbr(pto.f32(1.0))
            rhs = pto.vbr(pto.f32(7.0))
            mask = pto.pset_b32("PAT_ALL")
            try:
                pto.vdiv(lhs, rhs, mask, precision="high_precision")
            except ValueError as error:
                assert "vdiv precision" in str(error)
            else:
                raise AssertionError("retired high_precision token was accepted")


def test_diff_compensation_uses_the_compensated_implementation():
    """DIFF_COMPENSATION is a software algorithm, not a native divide."""
    _module, block = _vector_div_softop("diff_compensation")
    counts = Counter(op.name for op in block.operations)
    assert counts["pto.vmula"] == 3
    assert counts["pto.vmul"] == 0
    assert counts["pto.vdiv"] == 1


def test_native_div_algos_have_no_softlib_expansion():
    """intrinsic / diff_compensation / 1ulp_ftz_true stay a single vdiv."""
    for token in NATIVE_DIV_ALGO_TOKENS:
        with make_context():
            try:
                materialize("a5", "pto.vdiv", json.dumps({
                    "dtype": "f32", "lanes": 64, "mask": "b32", "precision": token,
                }), make_context())
            except ValueError as error:
                assert "no SoftOps implementation" in str(error)
            else:
                raise AssertionError(f"{token} must not expand through SoftLib")


def _ieee754_block(token):
    module, entry = materialize("a5", "pto.vdiv", json.dumps({
        "dtype": "f32", "lanes": 64, "mask": "b32", "precision": token,
    }), make_context())
    assert module.operation.verify()
    # Keep the module alive: the block operations die with its MLIR context.
    return entry, module, module.body.operations[0].regions[0].blocks[0]


def test_ieee754_divisions_share_the_normalisation_prologue():
    """Both FTZ-false algorithms normalize subnormals before dividing."""
    for token, entry in (("precision_0ulp_ftz_false", "div_vf32_ulp0_ftz_false"),
                         ("precision_1ulp_ftz_false", "div_vf32_ulp1_ftz_false")):
        got_entry, _module, block = _ieee754_block(token)
        assert got_entry == entry
        counts = Counter(op.name for op in block.operations)
        assert counts["pto.vdiv"] == 1
        # 2^23 scaling of the subnormal operands; the 0-ULP variant adds one
        # more vmuls/vadds for the residual-search negation and the sticky
        # direction (which the 1-ULP variant does not need).
        zero_ulp = token == "precision_0ulp_ftz_false"
        assert counts["pto.vmuls"] == (4 if zero_ulp else 2)
        assert counts["pto.vadds"] == (5 if zero_ulp else 3)
        assert counts["pto.vshr"] == 1  # shift by a vector amount
        assert counts["pto.vshl"] == 2
        assert counts["pto.vshrs"] == 3
        assert counts["pto.vshls"] == 1


def test_ieee754_0ulp_adds_the_residual_search():
    """The 0-ULP variant threads the quotient through DivPrecisionImpl."""
    _entry, _module, block = _ieee754_block("precision_0ulp_ftz_false")
    counts = Counter(op.name for op in block.operations)
    assert counts["pto.vmula"] == 4  # 3 residual candidates + sticky direction
    assert counts["pto.vabs"] == 5

    _entry, _module, block = _ieee754_block("precision_1ulp_ftz_false")
    counts = Counter(op.name for op in block.operations)
    # No residual search and no sticky direction: AscendC's compiler drops the
    # unused helper, so the emitted stream has no fused multiply-add at all.
    assert counts["pto.vmula"] == 0
    assert counts["pto.vabs"] == 2


def test_ieee754_subnormal_rounding_uses_sticky_bits():
    """Only the 0-ULP variant consults the sticky-zero tie rule."""
    for token, expected_pand, expected_pnot in (
        ("precision_0ulp_ftz_false", 4, 7),
        ("precision_1ulp_ftz_false", 1, 6),
    ):
        _entry, _module, block = _ieee754_block(token)
        counts = Counter(op.name for op in block.operations)
        assert counts["pto.pand"] == expected_pand
        assert counts["pto.pnot"] == expected_pnot


SCALAR_DIV_ALGO_TOKENS = ("intrinsic", "precision_0ulp_ftz_true")
VECTOR_ONLY_DIV_ALGO_TOKENS = (
    "diff_compensation",
    "precision_1ulp_ftz_true",
    "precision_0ulp_ftz_false",
    "precision_1ulp_ftz_false",
)


def _scalar_divf_with_algo(token):
    with make_context(), Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            lhs, rhs = pto.f32(1.0), pto.f32(7.0)
            pto.div(lhs, rhs, precision=token)
        return module


def test_scalar_division_accepts_only_the_supported_algorithms():
    """Scalar pto.divf honours intrinsic / 0-ULP-FTZ-TRUE and rejects the rest."""
    for token in SCALAR_DIV_ALGO_TOKENS:
        assert _scalar_divf_with_algo(token).operation.verify()

    for token in VECTOR_ONLY_DIV_ALGO_TOKENS:
        module = _scalar_divf_with_algo(token)
        with pytest.raises(MLIRError):
            module.operation.verify()


def test_scalar_division_forwards_the_requested_algorithm():
    """The SoftOps request carries the scalar algorithm instead of a constant."""
    name, _helper, arg_types, result_types = _resolve_softlib_request(
        "a5", "pto.divf", {"dtype": "f32", "precision": "precision_0ulp_ftz_true"}
    )
    assert name == "div_f32_soft"
    assert len(arg_types) == 2 and len(result_types) == 1

    for token in VECTOR_ONLY_DIV_ALGO_TOKENS:
        with pytest.raises(ValueError, match="not supported for scalar division"):
            _resolve_softlib_request(
                "a5", "pto.divf", {"dtype": "f32", "precision": token}
            )


def test_reject_unsupported_softlib_precision():
    requests = [
        ("a3", "pto.divf", '{"dtype":"f32","precision":"precision_0ulp_ftz_true"}'),
        ("a3", "pto.vdiv", '{"dtype":"f32","mask":"b32","lanes":64,"precision":"precision_0ulp_ftz_true"}'),
        ("a5", "pto.vdiv", '{"dtype":"f32","mask":"b16","lanes":64,"precision":"precision_0ulp_ftz_true"}'),
    ]
    for target, op, specs in requests:
        try:
            materialize(target, op, specs, make_context())
        except ValueError as error:
            assert "requires" in str(error)
        else:
            raise AssertionError(f"unsupported precision request accepted: {target}, {specs}")


if __name__ == "__main__":
    test_scalar_div_precision_attribute()
    test_scalar_div_precision_normalizes_names_and_rejects_unknown_values()
    test_scalar_div_softlib_materializes()
    test_reject_integer_precision()
    test_vector_precision_attribute()
    test_vector_div_precision_normalizes_names_and_rejects_unknown_values()
    test_vector_precision_materializes()
    test_vector_precision_matches_the_cann_sequence()
    test_vector_precision_residual_uses_cann_vmula_form()
    test_vector_precision_predicates_every_op_with_the_caller_mask()
    test_vmi_vdiv_precision_attribute()
    test_vmi_vdiv_default_precision_has_no_attribute()
    test_vmi_vdiv_precision_rejects_unknown_values()
    test_div_algo_tokens_round_trip_through_the_ir()
    test_div_algo_rejects_the_retired_high_precision_token()
    test_native_div_algos_have_no_softlib_expansion()
    test_ieee754_divisions_share_the_normalisation_prologue()
    test_ieee754_0ulp_adds_the_residual_search()
    test_ieee754_subnormal_rounding_uses_sticky_bits()
    test_scalar_division_accepts_only_the_supported_algorithms()
    test_scalar_division_forwards_the_requested_algorithm()
    test_reject_unsupported_softlib_precision()
