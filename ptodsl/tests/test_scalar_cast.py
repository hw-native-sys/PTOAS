# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from ptodsl import pto


@pto.jit(target="a5")
def scalar_float_cast_probe():
    data = pto.alloc_tile(shape=[1, 16], dtype=pto.f16, valid_shape=[1, 1])
    value = pto.load(data[0, 0])
    widened = pto.cast(value, pto.f32)
    narrowed = pto.cast(widened, pto.f16)
    pto.store(narrowed, data[0, 0])


@pto.jit(target="a5")
def vector_float_cast_probe():
    data = pto.alloc_tile(shape=[1, 16], dtype=pto.f16, valid_shape=[1, 8])
    ptr = data.as_ptr()
    value = pto.load(ptr, 0, contiguous=4)
    widened = pto.cast(value, pto.f32)
    narrowed = pto.cast(widened, pto.Vec(pto.f16, 4))
    pto.store(narrowed, ptr, 4)


@pto.jit(target="a5")
def vector_float_cast_shape_mismatch_probe():
    data = pto.alloc_tile(shape=[1, 16], dtype=pto.f16, valid_shape=[1, 8])
    value = pto.load(data.as_ptr(), 0, contiguous=4)
    _ = pto.cast(value, pto.Vec(pto.f32, 8))


@pto.simt
def simt_saturating_cast_probe():
    value = pto.cast(pto.get_laneid(), pto.f32, rounding="to_nearest_even")
    _ = pto.cast(value, pto.i32, rounding="toward_zero", saturation="sat")


@pto.jit(target="a5", mode="explicit")
def simt_saturating_cast_kernel():
    simt_saturating_cast_probe[1, 1, 1]()


@pto.jit(target="a5")
def scalar_bool_store_probe(dst: pto.ptr(pto.i8, "gm")):
    tid = pto.get_tid_x()
    pred = tid == pto.i32(0)
    pto.store(pred, dst, 0)


@pto.jit(target="a5")
def scalar_bool_cast_probe():
    data = pto.alloc_tile(shape=[1, 32], dtype=pto.i8, valid_shape=[1, 1])
    tid = pto.get_tid_x()
    pred = tid == pto.i32(0)
    widened = pto.cast(pred, pto.i8)
    pto.store(widened, data[0, 0])


def _make_issue_1179_simt_probe(name, source_dtype, target_dtype):
    @pto.simt(name=f"{name}_body")
    def body(
        source: pto.ptr(source_dtype, "gm"),
        destination: pto.ptr(target_dtype, "gm"),
    ):
        tid = pto.get_tid_x()
        value = pto.ldg(source, tid)
        converted = pto.cast(value, target_dtype)
        pto.stg(converted, destination, tid)

    @pto.jit(
        name=name,
        kernel_kind="vector",
        target="a5",
        mode="explicit",
        insert_sync=False,
    )
    def probe(
        source: pto.ptr(source_dtype, "gm"),
        destination: pto.ptr(target_dtype, "gm"),
    ):
        body[1, 1, 1](source, destination)

    return probe


def _make_issue_1179_ordinary_probe(name, source_dtype, target_dtype):
    @pto.jit(
        name=name,
        kernel_kind="vector",
        target="a5",
        mode="explicit",
        insert_sync=False,
    )
    def probe(
        source: pto.ptr(source_dtype, "gm"),
        destination: pto.ptr(target_dtype, "gm"),
    ):
        value = pto.load(source, 0)
        converted = pto.cast(value, target_dtype)
        pto.store(converted, destination, 0)

    return probe


ISSUE_1179_CAST_DIRECTIONS = (
    ("f32_to_i32", pto.f32, pto.i32, "pto.ftoi", "signed"),
    ("f32_to_ui32", pto.f32, pto.ui32, "pto.ftoi", "unsigned"),
    ("i32_to_f32", pto.i32, pto.f32, "pto.itof", "signed"),
    ("ui32_to_f32", pto.ui32, pto.f32, "pto.itof", "unsigned"),
)

ISSUE_1179_SIMT_PROBES = tuple(
    (name, _make_issue_1179_simt_probe(f"issue_1179_simt_{name}", source, target), op, sign)
    for name, source, target, op, sign in ISSUE_1179_CAST_DIRECTIONS
)
ISSUE_1179_ORDINARY_PROBES = tuple(
    (name, _make_issue_1179_ordinary_probe(f"issue_1179_ordinary_{name}", source, target), op, sign)
    for name, source, target, op, sign in ISSUE_1179_CAST_DIRECTIONS
)


@pto.jit(target="a5")
def index_to_i32_probe(value: pto.index, destination: pto.ptr(pto.i32, "gm")):
    pto.store(pto.cast(value, pto.i32), destination, 0)


@pto.jit(target="a5")
def index_to_ui32_probe(value: pto.index, destination: pto.ptr(pto.ui32, "gm")):
    pto.store(pto.cast(value, pto.ui32), destination, 0)


@pto.jit(target="a5")
def integer_to_index_probe(
    value: pto.i32,
    source: pto.ptr(pto.i32, "gm"),
    destination: pto.ptr(pto.i32, "gm"),
):
    offset = pto.cast(value, pto.index)
    loaded = pto.load(source, offset)
    pto.store(loaded, destination, offset)


@pto.jit(target="a5")
def unsigned_integer_to_index_probe(
    value: pto.ui32,
    source: pto.ptr(pto.i32, "gm"),
    destination: pto.ptr(pto.i32, "gm"),
):
    offset = pto.cast(value, pto.index)
    loaded = pto.load(source, offset)
    pto.store(loaded, destination, offset)


@pto.jit(target="a5")
def vector_index_integer_cast_probe():
    values = pto.Vec(pto.ui32, 2, init=(1, 2))
    indices = pto.cast(values, pto.index)
    _ = pto.cast(indices, pto.Vec(pto.ui32, 2))


@pto.jit(target="a5")
def automatic_index_consumer_probe(
    offset: pto.i32,
    source: pto.ptr(pto.i32, "gm"),
    destination: pto.ptr(pto.i32, "gm"),
):
    loaded = pto.load(source, offset)
    pto.store(loaded, destination, offset)


def test_scalar_cast_is_public():
    assert hasattr(pto, "cast")
    assert not hasattr(pto, "index_cast")
    assert not hasattr(pto, "ftof")
    assert not hasattr(pto, "ftoi")
    assert not hasattr(pto, "itof")
    assert not hasattr(__import__("ptodsl"), "scalar")


def test_scalar_float_cast():
    text = scalar_float_cast_probe.compile().mlir_text()
    assert text.count("pto.ftof ") == 2
    assert "f16 -> f32" in text
    assert "f32 -> f16" in text


def test_vector_float_cast_preserves_shape():
    text = vector_float_cast_probe.compile().mlir_text()
    assert text.count("pto.ftof ") == 2
    assert "vector<4xf16> -> vector<4xf32>" in text
    assert "vector<4xf32> -> vector<4xf16>" in text
    assert "arith.extf" not in text
    assert "arith.truncf" not in text


def test_vector_float_cast_rejects_shape_mismatch():
    try:
        vector_float_cast_shape_mismatch_probe.compile()
    except TypeError as error:
        assert "destination vector shape" in str(error)
    else:
        raise AssertionError(
            "pto.cast must reject a mismatched destination vector shape"
        )


def test_simt_saturating_cast_uses_category_ir():
    text = simt_saturating_cast_kernel.compile().mlir_text()
    assert "pto.itof " in text
    assert "pto.ftoi " in text
    assert "signed round(z) sat : f32 -> i32" in text


def test_scalar_bool_store_zero_extends_i1_to_i8():
    text = scalar_bool_store_probe.compile().mlir_text()
    assert "pto.exti" in text
    assert "unsigned : i1 -> i8" in text


def test_scalar_bool_cast_zero_extends_i1_to_i8():
    text = scalar_bool_cast_probe.compile().mlir_text()
    assert "pto.exti" in text
    assert "unsigned : i1 -> i8" in text


def test_issue_1179_scalar_float_integer_casts():
    for _, probe, operation, signedness in ISSUE_1179_SIMT_PROBES + ISSUE_1179_ORDINARY_PROBES:
        text = probe.compile().mlir_text()
        assert operation in text
        assert f"{signedness} :" in text


def test_index_integer_casts_use_pto_ir():
    index_to_i32_text = index_to_i32_probe.compile().mlir_text()
    assert "pto.index_cast" in index_to_i32_text
    assert "signed : index -> i32" in index_to_i32_text

    index_to_ui32_text = index_to_ui32_probe.compile().mlir_text()
    assert "pto.index_cast" in index_to_ui32_text
    assert "unsigned : index -> i32" in index_to_ui32_text

    integer_to_index_text = integer_to_index_probe.compile().mlir_text()
    assert "pto.index_cast" in integer_to_index_text
    assert "signed : i32 -> index" in integer_to_index_text

    unsigned_to_index_text = unsigned_integer_to_index_probe.compile().mlir_text()
    assert "pto.index_cast" in unsigned_to_index_text
    assert "unsigned : i32 -> index" in unsigned_to_index_text

    vector_text = vector_index_integer_cast_probe.compile().mlir_text()
    assert vector_text.count("pto.index_cast") == 2
    assert "signed : vector<2xi32> -> vector<2xindex>" in vector_text
    assert "unsigned : vector<2xindex> -> vector<2xi32>" in vector_text


def test_index_consumers_adapt_integer_offsets():
    text = automatic_index_consumer_probe.compile().mlir_text()
    assert "pto.index_cast" in text
    assert "signed : i32 -> index" in text


def main():
    test_scalar_cast_is_public()
    test_scalar_float_cast()
    test_vector_float_cast_preserves_shape()
    test_vector_float_cast_rejects_shape_mismatch()
    test_simt_saturating_cast_uses_category_ir()
    test_scalar_bool_store_zero_extends_i1_to_i8()
    test_scalar_bool_cast_zero_extends_i1_to_i8()
    test_issue_1179_scalar_float_integer_casts()
    test_index_integer_casts_use_pto_ir()
    test_index_consumers_adapt_integer_offsets()
    print("ptodsl_scalar_cast: PASS")

if __name__ == "__main__":
    main()
