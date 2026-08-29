# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from ptodsl import pto


@pto.jit(target="a5")
def scalar_mapping_gaps_probe():
    floats = pto.alloc_tile(shape=[1, 16], dtype=pto.f32, valid_shape=[1, 2])
    signed = pto.alloc_tile(shape=[1, 16], dtype=pto.si32, valid_shape=[1, 1])
    unsigned = pto.alloc_tile(shape=[1, 16], dtype=pto.ui32, valid_shape=[1, 1])
    lhs = pto.load(floats[0, 0])
    rhs = pto.load(floats[0, 1])
    si = pto.load(signed[0, 0])
    ui = pto.load(unsigned[0, 0])
    _ = lhs % rhs
    _ = pto.cast(si, pto.f32)
    _ = pto.cast(ui, pto.f32)
    _ = pto.cast(lhs, pto.si32)
    _ = pto.cast(rhs, pto.ui32)
    _ = pto.bitcast(lhs, pto.ui32)
    _ = pto.cmp(lhs, rhs, "uno")


@pto.jit(target="a5")
def vector_mapping_gaps_probe():
    floats = pto.alloc_tile(shape=[1, 16], dtype=pto.f32, valid_shape=[1, 8])
    unsigned = pto.alloc_tile(shape=[1, 16], dtype=pto.ui32, valid_shape=[1, 4])
    lhs = pto.load(floats.as_ptr(), 0, contiguous=4)
    rhs = pto.load(floats.as_ptr(), 4, contiguous=4)
    ui = pto.load(unsigned.as_ptr(), 0, contiguous=4)
    _ = lhs % rhs
    _ = pto.cast(ui, pto.f32)
    _ = pto.cast(lhs, pto.ui32)
    _ = pto.bitcast(ui, pto.f32)
    _ = pto.cmp(lhs, rhs, "ord")


@pto.jit(target="a5")
def integer_float_predicate_probe():
    values = pto.alloc_tile(shape=[1, 16], dtype=pto.i32, valid_shape=[1, 2])
    lhs = pto.load(values[0, 0])
    rhs = pto.load(values[0, 1])
    _ = pto.cmp(lhs, rhs, "uno")


@pto.jit(target="a5")
def integer_div_fastmath_probe():
    values = pto.alloc_tile(shape=[1, 16], dtype=pto.i32, valid_shape=[1, 2])
    lhs = pto.load(values[0, 0])
    rhs = pto.load(values[0, 1])
    _ = pto.div(lhs, rhs, fastmath="nnan")


@pto.jit(target="a5")
def integer_max_fastmath_probe():
    values = pto.alloc_tile(shape=[1, 16], dtype=pto.i32, valid_shape=[1, 2])
    lhs = pto.load(values[0, 0])
    rhs = pto.load(values[0, 1])
    _ = pto.max(lhs, rhs, fastmath="nnan")


@pto.jit(target="a5")
def scalar_attribute_probe():
    floats = pto.alloc_tile(shape=[1, 16], dtype=pto.f32, valid_shape=[1, 2])
    signed = pto.alloc_tile(shape=[1, 16], dtype=pto.si32, valid_shape=[1, 2])
    lhs = pto.load(floats[0, 0])
    rhs = pto.load(floats[0, 1])
    si_lhs = pto.load(signed[0, 0])
    si_rhs = pto.load(signed[0, 1])
    _ = pto.add(si_lhs, si_rhs, overflow=("nsw", "nuw"))
    _ = pto.add(lhs, rhs, fastmath=("nnan", "ninf"))
    _ = pto.cmp(lhs, rhs, "olt", fastmath="nnan")
    _ = pto.max(lhs, rhs, fastmath="nnan")
    _ = pto.maximum(lhs, rhs)
    _ = pto.minimum(lhs, rhs)
    _ = pto.cast(lhs, pto.f16, rounding="toward_zero", fastmath="nnan")
    _, _ = pto.addui_extended(si_lhs, si_rhs)
    _, _ = pto.mul_extended(si_lhs, si_rhs, signedness="signed")
    _, _ = pto.mul_extended(si_lhs, si_rhs, signedness="unsigned")


def test_mapping_gap_helpers_are_public():
    assert hasattr(pto, "bitcast")
    assert hasattr(pto, "cmp")
    assert all(hasattr(pto, name) for name in ("add", "sub", "mul", "div", "rem"))
    assert hasattr(pto, "maximum")
    assert hasattr(pto, "addui_extended")
    assert hasattr(pto, "mul_extended")
    assert not hasattr(__import__("ptodsl"), "scalar")


def test_scalar_mapping_gaps_emit_pto_ops():
    text = scalar_mapping_gaps_probe.compile().mlir_text()
    assert "pto.remf " in text
    assert all(op in text for op in ("pto.itof ", "pto.ftoi "))
    assert "pto.bitcast " in text
    assert "pto.cmpf uno " in text


def test_vector_mapping_gaps_preserve_shape():
    text = vector_mapping_gaps_probe.compile().mlir_text()
    assert "pto.remf " in text and "vector<4xf32>" in text
    assert "unsigned : vector<4xi32> -> vector<4xf32>" in text
    assert "unsigned : vector<4xf32> -> vector<4xi32>" in text
    assert "pto.bitcast " in text
    assert "pto.cmpf ord " in text


def test_float_only_predicate_rejects_integer_values():
    try:
        integer_float_predicate_probe.compile()
    except TypeError as error:
        assert "not supported" in str(error)
    else:
        raise AssertionError("pto.cmp must reject float-only predicates on integers")


def test_category_specific_attributes_reject_wrong_operand_kinds():
    for probe, helper in (
        (integer_div_fastmath_probe, "truediv"),
        (integer_max_fastmath_probe, "pto.max"),
    ):
        try:
            probe.compile()
        except TypeError as error:
            assert helper in str(error)
        else:
            raise AssertionError(f"{helper} must reject integer values")


def test_explicit_attributes_and_extended_ops_emit_pto_ir():
    text = scalar_attribute_probe.compile().mlir_text()
    assert "overflow<nsw, nuw>" in text
    assert "fastmath<nnan,ninf>" in text
    assert "pto.cmpf olt " in text and "fastmath<nnan>" in text
    assert "pto.maxf " in text
    assert "pto.maximum " in text
    assert "pto.minimum " in text
    assert "round(z) fastmath<nnan>" in text
    assert "pto.addui_extended " in text
    assert "pto.mul_extended " in text
    assert " signed : i32" in text
    assert " unsigned : i32" in text


def main():
    test_mapping_gap_helpers_are_public()
    test_scalar_mapping_gaps_emit_pto_ops()
    test_vector_mapping_gaps_preserve_shape()
    test_float_only_predicate_rejects_integer_values()
    test_category_specific_attributes_reject_wrong_operand_kinds()
    test_explicit_attributes_and_extended_ops_emit_pto_ir()
    print("ptodsl_scalar_mapping_gaps: PASS")


if __name__ == "__main__":
    main()
