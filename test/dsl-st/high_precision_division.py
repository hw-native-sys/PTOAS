#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""PTODSL ST coverage for high-precision f32 division and related rounding."""

import numpy as np

from common import auto_main
from ptodsl import pto


LANES = 64
VECTOR_ELEMENTS = 256
SCALAR_LANES = VECTOR_ELEMENTS
SENTINEL_BITS = 0xC2F68000  # -123.25f


def _repeat_to_lanes(bits):
    return np.resize(np.asarray(bits, dtype=np.uint32), LANES).view(np.float32)


def _repeat_to_elements(bits):
    return np.resize(np.asarray(bits, dtype=np.uint32), VECTOR_ELEMENTS).view(np.float32)


def exact_halfway_vectors():
    """Return exact halfway divisions around signed zero and min-subnormal."""
    lhs_bits = np.array([0x00000001, 0x80000001, 0x00000003, 0x80000003], dtype=np.uint32)
    rhs_bits = np.array([0x40000000, 0x40000000, 0x40000000, 0x40000000], dtype=np.uint32)
    expected_bits = np.array([0x00000000, 0x80000000, 0x00000002, 0x80000002], dtype=np.uint32)
    return lhs_bits.view(np.float32), rhs_bits.view(np.float32), expected_bits.view(np.float32)


def _finite_vector_bits():
    lhs_bits = np.array([
        0x00000001, 0x80000001, 0x00000003, 0x80000003,
        0x007FFFFF, 0x807FFFFF, 0x00800000, 0x80800000,
        0x3F800000, 0x40E00000, 0x3F800000, 0x40A00000,
        0x41200000, 0xBF800000, 0xC0E00000, 0xBF800000,
    ], dtype=np.uint32)
    rhs_bits = np.array([
        0x40000000, 0x40000000, 0x40000000, 0x40000000,
        0x40000000, 0x40000000, 0x40000000, 0x40000000,
        0x40E00000, 0x40400000, 0x40400000, 0x40400000,
        0x40400000, 0x40E00000, 0x40400000, 0x40400000,
    ], dtype=np.uint32)
    # Vector high-precision division corrects finite nonzero native results.
    # Native A5 vector vdiv flushes these underflow/subnormal outputs to signed zero.
    expected_bits = np.array([
        0x00000000, 0x80000000, 0x00000000, 0x80000000,
        0x00000000, 0x80000000, 0x00000000, 0x80000000,
        0x3E124925, 0x40155555, 0x3EAAAAAB, 0x3FD55555,
        0x40555555, 0xBE124925, 0xC0155555, 0xBEAAAAAB,
    ], dtype=np.uint32)
    return lhs_bits, rhs_bits, expected_bits


def _native_rounding_vector_bits():
    lhs_bits = np.array([
        0x3F800000, 0x40E00000, 0x3F800000, 0x40E00000,
    ], dtype=np.uint32)
    rhs_bits = np.array([
        0x40E00000, 0x40400000, 0x40400000, 0x40E00000,
    ], dtype=np.uint32)
    expected_bits = np.array([
        0x3E124925, 0x40155555, 0x3EAAAAAB, 0x3F800000,
    ], dtype=np.uint32)
    return lhs_bits, rhs_bits, expected_bits


def _nan_vector_bits():
    lhs_bits = np.array([
        0x7FC00001, 0x7FC12345, 0xFFC54321, 0x7F800001,
        0xFF800123, 0x3F800000, 0xBF800000, 0x3F800000,
    ], dtype=np.uint32)
    rhs_bits = np.array([
        0x3F800000, 0x40000000, 0x40400000, 0x3F800000,
        0x40000000, 0x7FC00001, 0x7F800001, 0xFF800123,
    ], dtype=np.uint32)
    return lhs_bits, rhs_bits


def _special_scalar_bits():
    pairs = np.array([
        [0x3F800000, 0x40E00000], [0x40E00000, 0x40400000],
        [0x00000001, 0x3F800000], [0x00000001, 0x40000000],
        [0x00000003, 0x40000000], [0x007FFFFF, 0x3F800000],
        [0x00800000, 0x40000000], [0x00FFFFFF, 0x40000000],
        [0x7F7FFFFF, 0x3F000000], [0xFF7FFFFF, 0x3F000000],
        [0x80000000, 0x3F800000], [0x3F800000, 0x80000000],
        [0x7F800000, 0x3F800000], [0xBF800000, 0x7F800000],
        [0x00000000, 0x00000000], [0x7F800000, 0x7F800000],
        [0x7FC12345, 0x3F800000], [0x3F800000, 0xFFC12345],
    ], dtype=np.uint32)
    return pairs[:, 0], pairs[:, 1]


@pto.simt
def scalar_division_body(
    lhs_ptr: pto.ptr(pto.f32, "gm"),
    rhs_ptr: pto.ptr(pto.f32, "gm"),
    out_ptr: pto.ptr(pto.f32, "gm"),
):
    tid = pto.get_tid_x()
    lhs = pto.ldg(lhs_ptr, tid)
    rhs = pto.ldg(rhs_ptr, tid)
    quotient = pto.div(lhs, rhs, precision=pto.DivPrecision.HighPrecision)
    pto.stg(quotient, out_ptr, tid)


@pto.jit(
    name="high_precision_scalar_division_kernel",
    kernel_kind="vector",
    target="a5",
    mode="explicit",
    insert_sync=False,
)
def high_precision_scalar_division_kernel(
    lhs_ptr: pto.ptr(pto.f32, "gm"),
    rhs_ptr: pto.ptr(pto.f32, "gm"),
    out_ptr: pto.ptr(pto.f32, "gm"),
):
    scalar_division_body[SCALAR_LANES, 1, 1](lhs_ptr, rhs_ptr, out_ptr)
    pto.pipe_barrier(pto.Pipe.ALL)


def _vector_load_store(lhs_ptr, rhs_ptr, out_ptr, total, offset):
    view_shape = [1, 1, 1, 1, total]
    view_strides = [total, total, total, total, 1]
    offsets = [0, 0, 0, 0, offset]
    tile_shape = [1, LANES]
    tile_sizes = [1, 1, 1, 1, LANES]

    lhs_view = pto.make_tensor_view(lhs_ptr, shape=view_shape, strides=view_strides)
    rhs_view = pto.make_tensor_view(rhs_ptr, shape=view_shape, strides=view_strides)
    out_view = pto.make_tensor_view(out_ptr, shape=view_shape, strides=view_strides)
    lhs_part = pto.partition_view(lhs_view, offsets=offsets, sizes=tile_sizes)
    rhs_part = pto.partition_view(rhs_view, offsets=offsets, sizes=tile_sizes)
    out_part = pto.partition_view(out_view, offsets=offsets, sizes=tile_sizes)

    lhs_tile = pto.alloc_tile(shape=tile_shape, dtype=pto.f32, addr=0,
                              valid_shape=tile_shape, blayout="RowMajor")
    rhs_tile = pto.alloc_tile(shape=tile_shape, dtype=pto.f32, addr=1024,
                              valid_shape=tile_shape, blayout="RowMajor")
    out_tile = pto.alloc_tile(shape=tile_shape, dtype=pto.f32, addr=2048,
                              valid_shape=tile_shape, blayout="RowMajor")

    pto.tile.load(lhs_part, lhs_tile)
    pto.tile.load(rhs_part, rhs_tile)
    pto.set_flag("MTE2", "V", event_id=0)
    pto.wait_flag("MTE2", "V", event_id=0)
    return lhs_tile, rhs_tile, out_tile, out_part


@pto.jit(
    name="high_precision_vector_division_kernel",
    kernel_kind="vector",
    target="a5",
    mode="explicit",
    insert_sync=False,
)
def high_precision_vector_division_kernel(
    lhs_ptr: pto.ptr(pto.f32, "gm"),
    rhs_ptr: pto.ptr(pto.f32, "gm"),
    out_ptr: pto.ptr(pto.f32, "gm"),
):
    for offset in pto.static_range(0, VECTOR_ELEMENTS, LANES):
        lhs_tile, rhs_tile, out_tile, out_part = _vector_load_store(
            lhs_ptr, rhs_ptr, out_ptr, VECTOR_ELEMENTS, offset
        )
        with pto.tileop():
            mask = pto.pset_b32(pto.MaskPattern.ALL)
            lhs = pto.vlds(lhs_tile[0, 0:])
            rhs = pto.vlds(rhs_tile[0, 0:])
            quotient = pto.vdiv(lhs, rhs, mask, precision=pto.DivPrecision.HighPrecision)
            pto.vsts(quotient, out_tile.as_ptr(), 0, mask, dist="NORM_B32")

        pto.set_flag("V", "MTE3", event_id=0)
        pto.wait_flag("V", "MTE3", event_id=0)
        pto.tile.store(out_tile, out_part)


@pto.jit(
    name="high_precision_vector_division_partial_mask_kernel",
    kernel_kind="vector",
    target="a5",
    mode="explicit",
    insert_sync=False,
)
def high_precision_vector_division_partial_mask_kernel(
    lhs_ptr: pto.ptr(pto.f32, "gm"),
    rhs_ptr: pto.ptr(pto.f32, "gm"),
    out_ptr: pto.ptr(pto.f32, "gm"),
):
    lhs_tile, rhs_tile, out_tile, out_part = _vector_load_store(
        lhs_ptr, rhs_ptr, out_ptr, LANES, 0
    )
    with pto.tileop():
        full_mask = pto.pset_b32(pto.MaskPattern.ALL)
        active_mask = pto.pset_b32(pto.MaskPattern.VL2)
        sentinel = pto.vbr(pto.f32(-123.25))
        pto.vsts(sentinel, out_tile.as_ptr(), 0, full_mask, dist="NORM_B32")
        lhs = pto.vlds(lhs_tile[0, 0:])
        rhs = pto.vlds(rhs_tile[0, 0:])
        quotient = pto.vdiv(lhs, rhs, active_mask, precision=pto.DivPrecision.HighPrecision)
        pto.vsts(quotient, out_tile.as_ptr(), 0, active_mask, dist="NORM_B32")

    pto.set_flag("V", "MTE3", event_id=0)
    pto.wait_flag("V", "MTE3", event_id=0)
    pto.tile.store(out_tile, out_part)


@pto.jit(
    name="high_precision_vector_division_holey_mask_kernel",
    kernel_kind="vector",
    target="a5",
    mode="explicit",
    insert_sync=False,
)
def high_precision_vector_division_holey_mask_kernel(
    lhs_ptr: pto.ptr(pto.f32, "gm"),
    rhs_ptr: pto.ptr(pto.f32, "gm"),
    out_ptr: pto.ptr(pto.f32, "gm"),
):
    for offset in pto.static_range(0, VECTOR_ELEMENTS, LANES):
        lhs_tile, rhs_tile, out_tile, out_part = _vector_load_store(
            lhs_ptr, rhs_ptr, out_ptr, VECTOR_ELEMENTS, offset
        )
        with pto.tileop():
            all_mask = pto.pset_b32(pto.MaskPattern.ALL)
            lt48, _ = pto.plt_b32(pto.const(48, dtype=pto.i32))
            lt16, _ = pto.plt_b32(pto.const(16, dtype=pto.i32))
            not_lt16 = pto.pnot(lt16, all_mask)
            holey_mask = pto.pand(lt48, not_lt16, all_mask)
            lhs = pto.vlds(lhs_tile[0, 0:])
            rhs = pto.vlds(rhs_tile[0, 0:])
            quotient = pto.vdiv(lhs, rhs, holey_mask, precision=pto.DivPrecision.HighPrecision)
            # Store every lane so inactive results are observable.
            pto.vsts(quotient, out_tile.as_ptr(), 0, all_mask, dist="NORM_B32")

        pto.set_flag("V", "MTE3", event_id=0)
        pto.wait_flag("V", "MTE3", event_id=0)
        pto.tile.store(out_tile, out_part)


@pto.jit(
    name="high_precision_vector_division_native_comparison_kernel",
    kernel_kind="vector",
    target="a5",
    mode="explicit",
    insert_sync=False,
)
def high_precision_vector_division_native_comparison_kernel(
    lhs_ptr: pto.ptr(pto.f32, "gm"),
    rhs_ptr: pto.ptr(pto.f32, "gm"),
    native_out_ptr: pto.ptr(pto.f32, "gm"),
    precise_out_ptr: pto.ptr(pto.f32, "gm"),
):
    lhs_tile, rhs_tile, native_tile, native_part = _vector_load_store(
        lhs_ptr, rhs_ptr, native_out_ptr, LANES, 0
    )
    precise_view = pto.make_tensor_view(
        precise_out_ptr, shape=[1, 1, 1, 1, LANES], strides=[LANES, LANES, LANES, LANES, 1]
    )
    precise_part = pto.partition_view(
        precise_view, offsets=[0, 0, 0, 0, 0], sizes=[1, 1, 1, 1, LANES]
    )
    precise_tile = pto.alloc_tile(shape=[1, LANES], dtype=pto.f32, addr=3072,
                                  valid_shape=[1, LANES], blayout="RowMajor")

    with pto.tileop():
        mask = pto.pset_b32(pto.MaskPattern.ALL)
        lhs = pto.vlds(lhs_tile[0, 0:])
        rhs = pto.vlds(rhs_tile[0, 0:])
        native = pto.vdiv(lhs, rhs, mask)
        precise = pto.vdiv(lhs, rhs, mask, precision=pto.DivPrecision.HighPrecision)
        pto.vsts(native, native_tile.as_ptr(), 0, mask, dist="NORM_B32")
        pto.vsts(precise, precise_tile.as_ptr(), 0, mask, dist="NORM_B32")

    pto.set_flag("V", "MTE3", event_id=0)
    pto.wait_flag("V", "MTE3", event_id=0)
    pto.tile.store(native_tile, native_part)
    pto.set_flag("V", "MTE3", event_id=1)
    pto.wait_flag("V", "MTE3", event_id=1)
    pto.tile.store(precise_tile, precise_part)


def _scalar_inputs():
    lhs, rhs, _ = exact_halfway_vectors()
    return [np.resize(lhs, (SCALAR_LANES,)), np.resize(rhs, (SCALAR_LANES,))]


def _special_scalar_inputs():
    lhs_bits, rhs_bits = _special_scalar_bits()
    return [_repeat_to_elements(lhs_bits), _repeat_to_elements(rhs_bits)]


def _vector_inputs():
    lhs_bits, rhs_bits, _ = _finite_vector_bits()
    return [_repeat_to_elements(lhs_bits), _repeat_to_elements(rhs_bits)]


def _vector_tile_inputs():
    lhs_bits, rhs_bits, _ = _finite_vector_bits()
    return [_repeat_to_lanes(lhs_bits), _repeat_to_lanes(rhs_bits)]


def _holey_mask_inputs():
    rng = np.random.default_rng(20260823)
    lhs = (rng.random(VECTOR_ELEMENTS, dtype=np.float32) * np.float32(2.0) + np.float32(0.5)).astype(np.float32)
    rhs = (rng.random(VECTOR_ELEMENTS, dtype=np.float32) * np.float32(8.0) + np.float32(0.25)).astype(np.float32)
    return [lhs, rhs]


def _holey_mask_expected(lhs, rhs):
    with np.errstate(all="ignore"):
        expected = np.divide(lhs, rhs, dtype=np.float32)
    inactive = np.ones((VECTOR_ELEMENTS,), dtype=bool)
    for offset in range(0, VECTOR_ELEMENTS, LANES):
        inactive[offset + 16:offset + 48] = False
    expected[inactive] = np.float32(0.0)
    return expected


def _vector_expected(_lhs, _rhs):
    _, _, expected_bits = _finite_vector_bits()
    return _repeat_to_elements(expected_bits)


def _native_rounding_inputs():
    lhs_bits, rhs_bits, _ = _native_rounding_vector_bits()
    return [_repeat_to_lanes(lhs_bits), _repeat_to_lanes(rhs_bits)]


def _native_rounding_expected(_lhs, _rhs):
    _, _, expected_bits = _native_rounding_vector_bits()
    return _repeat_to_lanes(expected_bits)


def _exact_halfway_expected(_lhs, _rhs):
    _, _, expected = exact_halfway_vectors()
    return np.resize(expected, (SCALAR_LANES,))


def _special_scalar_expected(lhs, rhs):
    with np.errstate(all="ignore"):
        return np.divide(lhs, rhs, dtype=np.float32)


def _scalar_special_case():
    def make_case():
        lhs, rhs = _special_scalar_inputs()
        out = np.zeros((SCALAR_LANES,), dtype=np.float32)
        return [lhs, rhs, out], _special_scalar_expected(lhs, rhs)

    def check_case(device_inputs, golden):
        actual = device_inputs[-1].cpu().numpy()
        equal = actual.view(np.uint32) == golden.view(np.uint32)
        equal |= np.isnan(actual) & np.isnan(golden)
        np.testing.assert_array_equal(equal, np.ones_like(equal, dtype=bool))

    return {
        "name": "high_precision_scalar_division_special_values",
        "kernel": high_precision_scalar_division_kernel,
        "make_case": make_case,
        "check": check_case,
    }


def _partial_expected(_lhs, _rhs):
    expected = _repeat_to_lanes([SENTINEL_BITS])
    expected[:2] = _vector_expected(_lhs, _rhs)[:2]
    return expected


def _nan_inputs():
    lhs_bits, rhs_bits = _nan_vector_bits()
    return [_repeat_to_lanes(lhs_bits), _repeat_to_lanes(rhs_bits)]


def _nan_expected(_lhs, _rhs):
    return np.ones((LANES,), dtype=bool)


def _check_nan_case(device_inputs, expected):
    actual = device_inputs[-1].cpu().numpy()
    np.testing.assert_array_equal(np.isnan(actual), expected)


def _bitwise_case(name, kernel, inputs, expected):
    def make_case():
        host_inputs = [np.array(value, copy=True) for value in inputs()]
        golden = np.array(expected(*host_inputs), copy=True)
        output = np.zeros(golden.shape, dtype=golden.dtype)
        return [*host_inputs, output], golden

    def check_case(device_inputs, golden):
        actual = device_inputs[-1].cpu().numpy()
        actual_bits = actual.view(np.uint32)
        golden_bits = golden.view(np.uint32)
        mismatch = np.flatnonzero(actual_bits != golden_bits)
        if mismatch.size:
            # Keep CI output actionable: NumPy truncates the arrays and hides
            # whether failures are active lanes or masked lanes.
            sample = mismatch[:64]
            rows = []
            lhs = device_inputs[0].cpu().numpy()
            rhs = device_inputs[1].cpu().numpy()
            for index in sample:
                rows.append(
                    f"idx={index} tile={index // LANES} lane={index % LANES} "
                    f"lhs={int(lhs[index].view(np.uint32)):08x} "
                    f"rhs={int(rhs[index].view(np.uint32)):08x} "
                    f"actual={int(actual_bits[index]):08x} golden={int(golden_bits[index]):08x}"
                )
            raise AssertionError(
                f"{mismatch.size} mismatches; first {sample.size}:\n" + "\n".join(rows)
            )

    return {
        "name": name,
        "kernel": kernel,
        "make_case": make_case,
        "check": check_case,
    }


def _native_rounding_case():
    def make_case():
        lhs, rhs = _native_rounding_inputs()
        native_out = np.zeros((LANES,), dtype=np.float32)
        precise_out = np.zeros((LANES,), dtype=np.float32)
        return [lhs, rhs, native_out, precise_out], _native_rounding_expected(lhs, rhs)

    def check_case(device_inputs, golden):
        native = device_inputs[-2].cpu().numpy()
        precise = device_inputs[-1].cpu().numpy()
        expected_bits = golden.view(np.uint32)
        native_bits = native.view(np.uint32)
        precise_bits = precise.view(np.uint32)
        np.testing.assert_array_equal(precise_bits, expected_bits)
        native_delta = native_bits.astype(np.int64) - expected_bits.astype(np.int64)
        # Native vdiv is permitted to be exact.  Its implementation may differ
        # from the high-precision path by at most one ULP, but the selected
        # inputs do not have to exercise both rounding directions (or either
        # direction) on every simulator/device version.
        if np.any(np.abs(native_delta) > 1):
            raise AssertionError(
                "native vdiv exceeded the one-ULP tolerance: "
                f"delta values={np.unique(native_delta)}"
            )

    return {
        "name": "high_precision_vector_division_native_rounding",
        "kernel": high_precision_vector_division_native_comparison_kernel,
        "make_case": make_case,
        "check": check_case,
    }


def _nan_case():
    def make_case():
        lhs, rhs = _nan_inputs()
        out = np.zeros((LANES,), dtype=np.float32)
        return [lhs, rhs, out], _nan_expected(lhs, rhs)

    return {
        "name": "high_precision_vector_division_nan_category",
        "kernel": high_precision_vector_division_kernel,
        "make_case": make_case,
        "check": _check_nan_case,
    }


@pto.simt
def scalar_precision_comparison_body(
    lhs_ptr: pto.ptr(pto.f32, "gm"),
    rhs_ptr: pto.ptr(pto.f32, "gm"),
    mul_lhs_ptr: pto.ptr(pto.f32, "gm"),
    mul_rhs_ptr: pto.ptr(pto.f32, "gm"),
    acc_ptr: pto.ptr(pto.f32, "gm"),
    out_ptr: pto.ptr(pto.f32, "gm"),
):
    tid = pto.get_tid_x()
    lhs = pto.ldg(lhs_ptr, tid)
    rhs = pto.ldg(rhs_ptr, tid)
    mul_lhs = pto.ldg(mul_lhs_ptr, tid)
    mul_rhs = pto.ldg(mul_rhs_ptr, tid)
    acc = pto.ldg(acc_ptr, tid)
    separate = pto.add(pto.mul(mul_lhs, mul_rhs), acc)
    fused = pto.fma(mul_lhs, mul_rhs, acc)
    ordinary = pto.div(lhs, rhs)
    precise = pto.div(lhs, rhs, precision=pto.DivPrecision.HighPrecision)
    pto.stg(separate, out_ptr, tid)
    pto.stg(fused, out_ptr, tid + VECTOR_ELEMENTS)
    pto.stg(ordinary, out_ptr, tid + 2 * VECTOR_ELEMENTS)
    pto.stg(precise, out_ptr, tid + 3 * VECTOR_ELEMENTS)


@pto.jit(
    name="scalar_precision_comparison_kernel",
    kernel_kind="vector",
    target="a5",
    mode="explicit",
    insert_sync=False,
)
def scalar_precision_comparison_kernel(
    lhs_ptr: pto.ptr(pto.f32, "gm"),
    rhs_ptr: pto.ptr(pto.f32, "gm"),
    mul_lhs_ptr: pto.ptr(pto.f32, "gm"),
    mul_rhs_ptr: pto.ptr(pto.f32, "gm"),
    acc_ptr: pto.ptr(pto.f32, "gm"),
    out_ptr: pto.ptr(pto.f32, "gm"),
):
    scalar_precision_comparison_body[SCALAR_LANES, 1, 1](
        lhs_ptr, rhs_ptr, mul_lhs_ptr, mul_rhs_ptr, acc_ptr, out_ptr
    )
    pto.pipe_barrier(pto.Pipe.ALL)


def _precision_inputs():
    rng = np.random.default_rng(20260803)
    lhs = (rng.random(VECTOR_ELEMENTS, dtype=np.float32) * np.float32(2.0)).astype(np.float32)
    rhs = (rng.random(VECTOR_ELEMENTS, dtype=np.float32) * np.float32(128.0) + np.float32(0.25)).astype(np.float32)
    mul_lhs = rng.uniform(-2, 2, VECTOR_ELEMENTS).astype(np.float32)
    mul_rhs = rng.uniform(-2, 2, VECTOR_ELEMENTS).astype(np.float32)
    acc = -np.multiply(mul_lhs, mul_rhs, dtype=np.float32)
    mul_lhs[0] = np.float32(1 + 2**-23)
    mul_rhs[0] = np.float32(1 - 2**-23)
    acc[0] = np.float32(-1)
    return [lhs, rhs, mul_lhs, mul_rhs, acc]


def _precision_expected(lhs, rhs, mul_lhs, mul_rhs, acc):
    separate = np.add(np.multiply(mul_lhs, mul_rhs, dtype=np.float32), acc, dtype=np.float32)
    fused = (mul_lhs.astype(np.float64) * mul_rhs.astype(np.float64) + acc.astype(np.float64)).astype(np.float32)
    with np.errstate(all="ignore"):
        ordinary = np.divide(lhs, rhs, dtype=np.float32)
        precise = np.divide(lhs, rhs, dtype=np.float32)
    return np.concatenate((separate, fused, ordinary, precise))


def _precision_case():
    def make_case():
        inputs = _precision_inputs()
        expected = _precision_expected(*inputs)
        return [*inputs, np.zeros(expected.shape, dtype=np.float32)], expected

    def check_case(device_inputs, golden):
        actual = device_inputs[-1].cpu().numpy()
        for index in (0, 1, 3):
            begin = index * VECTOR_ELEMENTS
            end = begin + VECTOR_ELEMENTS
            np.testing.assert_array_equal(actual[begin:end].view(np.uint32), golden[begin:end].view(np.uint32))

    return {
        "name": "simt_fp32_precision_fma_and_division",
        "kernel": scalar_precision_comparison_kernel,
        "make_case": make_case,
        "check": check_case,
    }


CASES = [
    _bitwise_case(
        "high_precision_scalar_division_exact_halfway",
        high_precision_scalar_division_kernel,
        _scalar_inputs,
        _exact_halfway_expected,
    ),
    _scalar_special_case(),
    _bitwise_case(
        "high_precision_vector_division_boundaries",
        high_precision_vector_division_kernel,
        _vector_inputs,
        _vector_expected,
    ),
    _bitwise_case(
        "high_precision_vector_division_partial_mask",
        high_precision_vector_division_partial_mask_kernel,
        _vector_tile_inputs,
        _partial_expected,
    ),
    _bitwise_case(
        "high_precision_vector_division_holey_mask_zero_inactive",
        high_precision_vector_division_holey_mask_kernel,
        _holey_mask_inputs,
        _holey_mask_expected,
    ),
    _native_rounding_case(),
    _nan_case(),
    _precision_case(),
]


auto_main(globals())
