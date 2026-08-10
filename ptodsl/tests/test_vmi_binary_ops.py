#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import warnings

from ptodsl import pto
from ptodsl._diagnostics import PTODSLDeprecationWarning


@pto.jit(target="a5", backend="vpto", mode="explicit")
def vmi_binary_vector_vector_probe():
    lhs_tile = pto.alloc_tile(shape=[1, 64], dtype=pto.f32)
    rhs_tile = pto.alloc_tile(shape=[1, 64], dtype=pto.f32)
    mask = pto.vmi.create_mask(64, size=64)
    lhs = pto.vmi.vload(lhs_tile.as_ptr(), 0, size=64)
    rhs = pto.vmi.vload(rhs_tile.as_ptr(), 0, size=64)
    _ = pto.vmi.vadd(lhs, rhs, mask)


@pto.jit(target="a5", backend="vpto", mode="explicit")
def vmi_binary_add_vector_scalar_probe():
    source_tile = pto.alloc_tile(shape=[1, 64], dtype=pto.f32)
    mask = pto.vmi.create_mask(64, size=64)
    source = pto.vmi.vload(source_tile.as_ptr(), 0, size=64)
    _ = pto.vmi.vadd(source, 1.0, mask)


@pto.jit(target="a5", backend="vpto", mode="explicit")
def vmi_binary_vector_scalar_probe():
    source_tile = pto.alloc_tile(shape=[1, 64], dtype=pto.f32)
    integer_tile = pto.alloc_tile(shape=[1, 64], dtype=pto.i32)
    mask = pto.vmi.create_mask(64, size=64)
    source = pto.vmi.vload(source_tile.as_ptr(), 0, size=64)
    integer_source = pto.vmi.vload(integer_tile.as_ptr(), 0, size=64)
    _ = pto.vmi.vmul(source, 2.0, mask)
    _ = pto.vmi.vmax(source, 1.0, mask)
    _ = pto.vmi.vmin(source, 1.0, mask)
    _ = pto.vmi.vshl(integer_source, 1, mask)
    _ = pto.vmi.vshr(integer_source, 1, mask)


@pto.jit(target="a5", backend="vpto", mode="explicit")
def vmi_binary_scalar_vector_probe():
    source_tile = pto.alloc_tile(shape=[1, 64], dtype=pto.f32)
    mask = pto.vmi.create_mask(64, size=64)
    source = pto.vmi.vload(source_tile.as_ptr(), 0, size=64)
    _ = pto.vmi.vadd(1.0, source, mask)
    _ = pto.vmi.vmul(2.0, source, mask)
    _ = pto.vmi.vmax(1.0, source, mask)
    _ = pto.vmi.vmin(1.0, source, mask)


@pto.jit(target="a5", backend="vpto", mode="explicit")
def vmi_deprecated_binary_scalar_probe():
    source_tile = pto.alloc_tile(shape=[1, 64], dtype=pto.f32)
    integer_tile = pto.alloc_tile(shape=[1, 64], dtype=pto.i32)
    mask = pto.vmi.create_mask(64, size=64)
    source = pto.vmi.vload(source_tile.as_ptr(), 0, size=64)
    integer_source = pto.vmi.vload(integer_tile.as_ptr(), 0, size=64)
    _ = pto.vmi.vadds(source, 1.0, mask)
    _ = pto.vmi.vmuls(source, 2.0, mask)
    _ = pto.vmi.vmaxs(source, 1.0, mask)
    _ = pto.vmi.vmins(source, 1.0, mask)
    _ = pto.vmi.vshls(integer_source, 1, mask)
    _ = pto.vmi.vshrs(integer_source, 1, mask)


@pto.jit(target="a5", backend="vpto", mode="explicit")
def vmi_binary_add_compatibility_probe():
    source_tile = pto.alloc_tile(shape=[1, 64], dtype=pto.f32)
    mask = pto.vmi.create_mask(64, size=64)
    source = pto.vmi.vload(source_tile.as_ptr(), 0, size=64)
    _ = pto.vmi.vadds(source, 1.0, mask)


@pto.jit(target="a5", backend="vpto", mode="explicit")
def vmi_add_carry_probe():
    lhs_tile = pto.alloc_tile(shape=[1, 64], dtype=pto.ui32)
    rhs_tile = pto.alloc_tile(shape=[1, 64], dtype=pto.ui32)
    mask = pto.vmi.create_mask(64, size=64)
    lhs = pto.vmi.vload(lhs_tile.as_ptr(), 0, size=64)
    rhs = pto.vmi.vload(rhs_tile.as_ptr(), 0, size=64)
    sum_value, carry = pto.vmi.vaddc(lhs, rhs, mask)
    _, _ = pto.vmi.vaddcs(sum_value, rhs, carry, mask)


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    vector_vector_text = vmi_binary_vector_vector_probe.compile().mlir_text()
    expect(
        "pto.vmi.vadd" in vector_vector_text,
        "vmi.vadd(vector, vector, mask) should emit pto.vmi.vadd",
    )
    expect(
        "pto.vmi.vadds" not in vector_vector_text,
        "vector-vector vmi.vadd should not emit pto.vmi.vadds",
    )

    vector_scalar_text = vmi_binary_add_vector_scalar_probe.compile().mlir_text()
    expect(
        "pto.vmi.vadds" in vector_scalar_text,
        "vmi.vadd(vector, scalar, mask) should emit pto.vmi.vadds",
    )

    carry_text = vmi_add_carry_probe.compile().mlir_text()
    expect("pto.vmi.vaddc" in carry_text, "vmi.vaddc should emit pto.vmi.vaddc")
    expect("pto.vmi.vaddcs" in carry_text, "vmi.vaddcs should emit pto.vmi.vaddcs")

    vector_scalar_text = vmi_binary_vector_scalar_probe.compile().mlir_text()
    for op_name in ("vmuls", "vmaxs", "vmins", "vshls", "vshrs"):
        expect(
            f"pto.vmi.{op_name}" in vector_scalar_text,
            f"unified VMI scalar form should emit pto.vmi.{op_name}",
        )

    scalar_vector_text = vmi_binary_scalar_vector_probe.compile().mlir_text()
    for op_name in ("vadds", "vmuls", "vmaxs", "vmins"):
        expect(
            f"pto.vmi.{op_name}" in scalar_vector_text,
            f"commutative VMI scalar-vector form should emit pto.vmi.{op_name}",
        )

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", PTODSLDeprecationWarning)
        compatibility_text = vmi_binary_add_compatibility_probe.compile().mlir_text()

    expect("pto.vmi.vadds" in compatibility_text, "vmi.vadds compatibility should preserve the VMI op")
    deprecation_warnings = [
        warning for warning in captured if warning.category is PTODSLDeprecationWarning
    ]
    expect(len(deprecation_warnings) == 1, "vmi.vadds should emit one deprecation warning")
    expect(
        "pto.vmi.vadd(vector, scalar, mask)" in str(deprecation_warnings[0].message),
        "vmi.vadds warning should name the replacement API",
    )

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", PTODSLDeprecationWarning)
        deprecated_text = vmi_deprecated_binary_scalar_probe.compile().mlir_text()

    for op_name in ("vadds", "vmuls", "vmaxs", "vmins", "vshls", "vshrs"):
        expect(
            f"pto.vmi.{op_name}" in deprecated_text,
            f"deprecated VMI {op_name} should preserve the underlying operation",
        )
    deprecation_warnings = [
        warning for warning in captured if warning.category is PTODSLDeprecationWarning
    ]
    expect(len(deprecation_warnings) == 6, "all deprecated VMI scalar compatibility APIs should warn")

    print("ptodsl_vmi_binary_ops: PASS")


if __name__ == "__main__":
    main()
