# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from ptodsl import pto, scalar


@pto.jit(target="a5")
def scalar_float_cast_probe():
    data = pto.alloc_tile(shape=[1, 16], dtype=pto.f16, valid_shape=[1, 1])
    value = scalar.load(data[0, 0])
    widened = scalar.cast(value, pto.f32)
    narrowed = scalar.cast(widened, pto.f16)
    scalar.store(narrowed, data[0, 0])


@pto.jit(target="a5")
def vector_float_cast_probe():
    data = pto.alloc_tile(shape=[1, 16], dtype=pto.f16, valid_shape=[1, 8])
    ptr = data.as_ptr()
    value = scalar.load(ptr, 0, contiguous=4)
    widened = scalar.cast(value, pto.f32)
    narrowed = scalar.cast(widened, pto.Vec(pto.f16, 4))
    scalar.store(narrowed, ptr, 4)


@pto.jit(target="a5")
def vector_float_cast_shape_mismatch_probe():
    data = pto.alloc_tile(shape=[1, 16], dtype=pto.f16, valid_shape=[1, 8])
    value = scalar.load(data.as_ptr(), 0, contiguous=4)
    _ = scalar.cast(value, pto.Vec(pto.f32, 8))


@pto.jit(target="a5")
def scalar_bool_store_probe(dst: pto.ptr(pto.i8, "gm")):
    tid = pto.get_tid_x()
    pred = tid == pto.i32(0)
    scalar.store(pred, dst, 0)


@pto.jit(target="a5")
def scalar_bool_cast_probe():
    data = pto.alloc_tile(shape=[1, 32], dtype=pto.i8, valid_shape=[1, 1])
    tid = pto.get_tid_x()
    pred = tid == pto.i32(0)
    widened = scalar.cast(pred, pto.i8)
    scalar.store(widened, data[0, 0])


def test_scalar_cast_is_public():
    assert "cast" in scalar.__all__


def test_scalar_float_cast():
    text = scalar_float_cast_probe.compile().mlir_text()
    assert "arith.extf" in text
    assert "arith.truncf" in text
    assert "f16 to f32" in text
    assert "f32 to f16" in text


def test_vector_float_cast_preserves_shape():
    text = vector_float_cast_probe.compile().mlir_text()
    assert "arith.extf" in text
    assert "arith.truncf" in text
    assert "vector<4xf16> to vector<4xf32>" in text
    assert "vector<4xf32> to vector<4xf16>" in text


def test_vector_float_cast_rejects_shape_mismatch():
    try:
        vector_float_cast_shape_mismatch_probe.compile()
    except TypeError as error:
        assert "destination vector shape" in str(error)
    else:
        raise AssertionError(
            "scalar.cast must reject a mismatched destination vector shape"
        )


def test_scalar_bool_store_zero_extends_i1_to_i8():
    text = scalar_bool_store_probe.compile().mlir_text()
    assert "arith.extui" in text
    assert "i1 to i8" in text
    assert "arith.extsi" not in text


def test_scalar_bool_cast_zero_extends_i1_to_i8():
    text = scalar_bool_cast_probe.compile().mlir_text()
    assert "arith.extui" in text
    assert "i1 to i8" in text
    assert "arith.extsi" not in text


def main():
    test_scalar_cast_is_public()
    test_scalar_float_cast()
    test_vector_float_cast_preserves_shape()
    test_vector_float_cast_rejects_shape_mismatch()
    test_scalar_bool_store_zero_extends_i1_to_i8()
    test_scalar_bool_cast_zero_extends_i1_to_i8()
    print("ptodsl_scalar_cast: PASS")

if __name__ == "__main__":
    main()
