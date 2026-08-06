#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Runtime equivalence checks through element-wise 1D/2D TileOp selection."""

from pathlib import Path
import sys

import numpy as np


def _bootstrap_dsl_st_common() -> None:
    here = Path(__file__).resolve()
    for candidate in here.parents:
        common_dir = candidate / "test" / "dsl-st"
        if (common_dir / "common.py").exists():
            sys.path.insert(0, str(common_dir))
            return
    raise RuntimeError("Unable to locate test/dsl-st/common.py")


_bootstrap_dsl_st_common()

from common import assert_close, auto_main
from ptodsl import pto
from ptodsl.tilelib.templates.a5.tabs import (
    template_tabs,
    template_tabs_1d,
)
from ptodsl.tilelib.templates.a5.tadd import (
    template_tadd,
    template_tadd_1d,
)
from ptodsl.tilelib.templates.a5.tadds import (
    template_tadds,
    template_tadds_1d,
)
from ptodsl.tilelib.templates.a5.tcmps import (
    template_tcmps,
    template_tcmps_1d,
)
from ptodsl.tilelib.templates.a5.tcvt import (
    template_tcvt_f32_to_i16,
    template_tcvt_f32_to_i16_1d,
)
from ptodsl.tilelib.templates.a5.texpand import (
    template_texpands,
    template_texpands_1d,
)
from ptodsl.tilelib.templates.a5.tsels import (
    template_tsels,
    template_tsels_1d,
)


SEED = 20260802
GUARD_ELEMENTS = 32

TABS_SHAPE = (2, 72)
TABS_PADDED_SHAPE = (2, 80)
TADD_SHAPE = (4, 80)
TADD_PADDED_SHAPE = (4, 96)
TADDS_SHAPE = (4, 96)
TADDS_PADDED_SHAPE = (4, 128)
TCMPS_DATA_SHAPE = (4, 256)
TCMPS_MASK_SHAPE = (4, 32)
TCMPS_PADDED_DATA_SHAPE = (4, 512)
TSELS_DATA_SHAPE = (4, 256)
TSELS_MASK_SHAPE = (4, 32)
TSELS_PADDED_DATA_SHAPE = (4, 512)
TSELS_PADDED_MASK_SHAPE = (4, 64)
TCVT_SHAPE = (4, 80)
TCVT_PADDED_SHAPE = (4, 96)
TEXPANDS_SHAPE = (4, 72)
TEXPANDS_PADDED_SHAPE = (4, 80)

# Each compact shape fills its physical column axis and selects 1D. The
# corresponding padded shape keeps the same valid region but introduces a row
# stride gap, which conservatively selects the 2D fallback.


_TEMPLATE_PAIRS = (
    (template_tabs_1d, template_tabs),
    (template_tadd_1d, template_tadd),
    (template_tadds_1d, template_tadds),
    (template_tcmps_1d, template_tcmps),
    (template_tsels_1d, template_tsels),
    (template_tcvt_f32_to_i16_1d, template_tcvt_f32_to_i16),
    (template_texpands_1d, template_texpands),
)

for _flattened, _rowwise in _TEMPLATE_PAIRS:
    if _flattened.metadata.loop_depth != 1:
        raise AssertionError(f"{_flattened.name} is not a 1D candidate")
    if _rowwise.metadata.loop_depth != 2:
        raise AssertionError(f"{_rowwise.name} is not a 2D candidate")


def _row_major_view(ptr, rows, cols, *, offset=0):
    if offset:
        ptr = pto.addptr(ptr, offset)
    return pto.make_tensor_view(
        ptr,
        shape=[rows, cols],
        strides=[cols, 1],
    )


def _equivalence_jit(op_name):
    return pto.jit(
        name=f"elementwise_{op_name}_1d_2d_equivalence",
        target="a5",
        kernel_kind="vector",
        mode="explicit",
        insert_sync=True,
    )


# Only cases with the same one-input ABI and tile topology use this factory.
# Packed predicates, temporaries, and other distinct operand forms stay explicit.
def _make_single_input_equivalence_kernel(
    op_name,
    *,
    shape,
    padded_shape,
    src_dtype,
    dst_dtype,
    operation,
):
    @_equivalence_jit(op_name)
    def _kernel(
        src_ptr: pto.ptr(src_dtype, "gm"),
        out_1d_ptr: pto.ptr(dst_dtype, "gm"),
        out_2d_ptr: pto.ptr(dst_dtype, "gm"),
    ):
        rows, cols = shape
        src_view = _row_major_view(src_ptr, rows, cols)
        out_1d_view = _row_major_view(
            out_1d_ptr, rows, cols, offset=GUARD_ELEMENTS
        )
        out_2d_view = _row_major_view(
            out_2d_ptr, rows, cols, offset=GUARD_ELEMENTS
        )

        src_1d = pto.alloc_tile(
            shape=[rows, cols], dtype=src_dtype, addr=0
        )
        src_2d = pto.alloc_tile(
            shape=list(padded_shape),
            dtype=src_dtype,
            valid_shape=[rows, cols],
            addr=4096,
        )
        out_1d = pto.alloc_tile(
            shape=[rows, cols], dtype=dst_dtype, addr=8192
        )
        out_2d = pto.alloc_tile(
            shape=list(padded_shape),
            dtype=dst_dtype,
            valid_shape=[rows, cols],
            addr=12288,
        )

        pto.tile.load(src_view, src_1d)
        pto.tile.load(src_view, src_2d)
        operation(src_1d, out_1d)
        operation(src_2d, out_2d)
        pto.tile.store(out_1d, out_1d_view)
        pto.tile.store(out_2d, out_2d_view)

    return _kernel


def _tabs_operation(src, dst):
    pto.tile.abs(src, dst)


def _tadds_operation(src, dst):
    pto.tile.adds(src, 7, dst)


def _tcvt_operation(src, dst):
    pto.tile.cvt(src, dst)


elementwise_tabs_1d_2d_equivalence = _make_single_input_equivalence_kernel(
    "tabs",
    shape=TABS_SHAPE,
    padded_shape=TABS_PADDED_SHAPE,
    src_dtype=pto.f32,
    dst_dtype=pto.f32,
    operation=_tabs_operation,
)


@_equivalence_jit("tadd")
def elementwise_tadd_1d_2d_equivalence(
    lhs_ptr: pto.ptr(pto.i16, "gm"),
    rhs_ptr: pto.ptr(pto.i16, "gm"),
    out_1d_ptr: pto.ptr(pto.i16, "gm"),
    out_2d_ptr: pto.ptr(pto.i16, "gm"),
):
    rows, cols = TADD_SHAPE

    lhs_view = _row_major_view(lhs_ptr, rows, cols)
    rhs_view = _row_major_view(rhs_ptr, rows, cols)
    out_1d_view = _row_major_view(
        out_1d_ptr, rows, cols, offset=GUARD_ELEMENTS
    )
    out_2d_view = _row_major_view(
        out_2d_ptr, rows, cols, offset=GUARD_ELEMENTS
    )

    lhs_1d = pto.alloc_tile(shape=[rows, cols], dtype=pto.i16, addr=0)
    rhs_1d = pto.alloc_tile(shape=[rows, cols], dtype=pto.i16, addr=4096)
    lhs_2d = pto.alloc_tile(
        shape=list(TADD_PADDED_SHAPE),
        dtype=pto.i16,
        valid_shape=[rows, cols],
        addr=8192,
    )
    rhs_2d = pto.alloc_tile(
        shape=list(TADD_PADDED_SHAPE),
        dtype=pto.i16,
        valid_shape=[rows, cols],
        addr=12288,
    )
    out_1d = pto.alloc_tile(shape=[rows, cols], dtype=pto.i16, addr=16384)
    out_2d = pto.alloc_tile(
        shape=list(TADD_PADDED_SHAPE),
        dtype=pto.i16,
        valid_shape=[rows, cols],
        addr=20480,
    )

    pto.tile.load(lhs_view, lhs_1d)
    pto.tile.load(rhs_view, rhs_1d)
    pto.tile.load(lhs_view, lhs_2d)
    pto.tile.load(rhs_view, rhs_2d)
    pto.tile.add(lhs_1d, rhs_1d, out_1d)
    pto.tile.add(lhs_2d, rhs_2d, out_2d)
    pto.tile.store(out_1d, out_1d_view)
    pto.tile.store(out_2d, out_2d_view)


elementwise_tadds_1d_2d_equivalence = _make_single_input_equivalence_kernel(
    "tadds",
    shape=TADDS_SHAPE,
    padded_shape=TADDS_PADDED_SHAPE,
    src_dtype=pto.i8,
    dst_dtype=pto.i8,
    operation=_tadds_operation,
)


@_equivalence_jit("tcmps")
def elementwise_tcmps_1d_2d_equivalence(
    src_ptr: pto.ptr(pto.i8, "gm"),
    out_1d_ptr: pto.ptr(pto.ui8, "gm"),
    out_2d_ptr: pto.ptr(pto.ui8, "gm"),
):
    data_rows, data_cols = TCMPS_DATA_SHAPE
    mask_rows, mask_cols = TCMPS_MASK_SHAPE

    src_view = _row_major_view(src_ptr, data_rows, data_cols)
    out_1d_view = _row_major_view(
        out_1d_ptr, mask_rows, mask_cols, offset=GUARD_ELEMENTS
    )
    out_2d_view = _row_major_view(
        out_2d_ptr, mask_rows, mask_cols, offset=GUARD_ELEMENTS
    )

    src_1d = pto.alloc_tile(
        shape=[data_rows, data_cols], dtype=pto.i8, addr=0
    )
    src_2d = pto.alloc_tile(
        shape=list(TCMPS_PADDED_DATA_SHAPE),
        dtype=pto.i8,
        valid_shape=[data_rows, data_cols],
        addr=4096,
    )
    out_1d = pto.alloc_tile(
        shape=[mask_rows, mask_cols],
        dtype=pto.ui8,
        addr=8192,
    )
    out_2d = pto.alloc_tile(
        shape=[mask_rows, mask_cols],
        dtype=pto.ui8,
        addr=12288,
    )

    pto.tile.load(src_view, src_1d)
    pto.tile.load(src_view, src_2d)
    pto.tile.cmps(src_1d, 5, out_1d)
    pto.tile.cmps(src_2d, 5, out_2d)
    pto.tile.store(out_1d, out_1d_view)
    pto.tile.store(out_2d, out_2d_view)


@_equivalence_jit("tsels")
def elementwise_tsels_1d_2d_equivalence(
    mask_ptr: pto.ptr(pto.i8, "gm"),
    src_ptr: pto.ptr(pto.i8, "gm"),
    out_1d_ptr: pto.ptr(pto.i8, "gm"),
    out_2d_ptr: pto.ptr(pto.i8, "gm"),
):
    data_rows, data_cols = TSELS_DATA_SHAPE
    mask_rows, mask_cols = TSELS_MASK_SHAPE

    mask_view = _row_major_view(mask_ptr, mask_rows, mask_cols)
    src_view = _row_major_view(src_ptr, data_rows, data_cols)
    out_1d_view = _row_major_view(
        out_1d_ptr, data_rows, data_cols, offset=GUARD_ELEMENTS
    )
    out_2d_view = _row_major_view(
        out_2d_ptr, data_rows, data_cols, offset=GUARD_ELEMENTS
    )

    mask_1d = pto.alloc_tile(
        shape=[mask_rows, mask_cols], dtype=pto.i8, addr=0
    )
    src_1d = pto.alloc_tile(
        shape=[data_rows, data_cols],
        dtype=pto.i8,
        addr=4096,
    )
    tmp_1d = pto.alloc_tile(shape=[1, 256], dtype=pto.i8, addr=8192)
    out_1d = pto.alloc_tile(
        shape=[data_rows, data_cols],
        dtype=pto.i8,
        addr=12288,
    )
    mask_2d = pto.alloc_tile(
        shape=list(TSELS_PADDED_MASK_SHAPE),
        dtype=pto.i8,
        valid_shape=[mask_rows, mask_cols],
        addr=16384,
    )
    src_2d = pto.alloc_tile(
        shape=list(TSELS_PADDED_DATA_SHAPE),
        dtype=pto.i8,
        valid_shape=[data_rows, data_cols],
        addr=20480,
    )
    tmp_2d = pto.alloc_tile(shape=[1, 256], dtype=pto.i8, addr=24576)
    out_2d = pto.alloc_tile(
        shape=list(TSELS_PADDED_DATA_SHAPE),
        dtype=pto.i8,
        valid_shape=[data_rows, data_cols],
        addr=28672,
    )

    pto.tile.load(mask_view, mask_1d)
    pto.tile.load(src_view, src_1d)
    pto.tile.load(mask_view, mask_2d)
    pto.tile.load(src_view, src_2d)
    pto.tile.sels(mask_1d, src_1d, -3, out_1d, tmp=tmp_1d)
    pto.tile.sels(mask_2d, src_2d, -3, out_2d, tmp=tmp_2d)
    pto.tile.store(out_1d, out_1d_view)
    pto.tile.store(out_2d, out_2d_view)


elementwise_tcvt_1d_2d_equivalence = _make_single_input_equivalence_kernel(
    "tcvt",
    shape=TCVT_SHAPE,
    padded_shape=TCVT_PADDED_SHAPE,
    src_dtype=pto.f32,
    dst_dtype=pto.i16,
    operation=_tcvt_operation,
)


@_equivalence_jit("texpands")
def elementwise_texpands_1d_2d_equivalence(
    out_1d_ptr: pto.ptr(pto.i32, "gm"),
    out_2d_ptr: pto.ptr(pto.i32, "gm"),
):
    rows, cols = TEXPANDS_SHAPE

    out_1d_view = _row_major_view(
        out_1d_ptr, rows, cols, offset=GUARD_ELEMENTS
    )
    out_2d_view = _row_major_view(
        out_2d_ptr, rows, cols, offset=GUARD_ELEMENTS
    )

    out_1d = pto.alloc_tile(shape=[rows, cols], dtype=pto.i32, addr=0)
    out_2d = pto.alloc_tile(
        shape=list(TEXPANDS_PADDED_SHAPE),
        dtype=pto.i32,
        valid_shape=[rows, cols],
        addr=4096,
    )

    pto.tile.expands(23, out_1d)
    pto.tile.expands(23, out_2d)
    pto.tile.store(out_1d, out_1d_view)
    pto.tile.store(out_2d, out_2d_view)


def _equivalence_case(
    name,
    kernel,
    *,
    inputs,
    expected,
    output_dtype,
    guard_value,
    rtol=0.0,
    atol=0.0,
):
    def make_case():
        host_inputs = [np.array(value, copy=True) for value in inputs()]
        golden = np.array(expected(*host_inputs), dtype=output_dtype, copy=True)
        guarded_size = golden.size + 2 * GUARD_ELEMENTS
        out_1d = np.full(guarded_size, guard_value, dtype=output_dtype)
        out_2d = np.full(guarded_size, guard_value, dtype=output_dtype)
        return [*host_inputs, out_1d, out_2d], golden

    def check(device_inputs, golden):
        outputs = [
            device_inputs[-2].cpu().numpy(),
            device_inputs[-1].cpu().numpy(),
        ]
        expected_guard = np.full(
            GUARD_ELEMENTS,
            guard_value,
            dtype=output_dtype,
        )
        logical_outputs = []
        for output in outputs:
            assert_close(
                output[:GUARD_ELEMENTS],
                expected_guard,
                rtol=0.0,
                atol=0.0,
            )
            assert_close(
                output[-GUARD_ELEMENTS:],
                expected_guard,
                rtol=0.0,
                atol=0.0,
            )
            logical = output[
                GUARD_ELEMENTS:-GUARD_ELEMENTS
            ].reshape(golden.shape)
            assert_close(logical, golden, rtol=rtol, atol=atol)
            logical_outputs.append(logical)
        assert_close(
            logical_outputs[0],
            logical_outputs[1],
            rtol=rtol,
            atol=atol,
        )

    return {
        "name": name,
        "kernel": kernel,
        "make_case": make_case,
        "check": check,
    }


def _tabs_inputs():
    rng = np.random.default_rng(SEED + 1)
    return [rng.uniform(-20.0, 20.0, size=TABS_SHAPE).astype(np.float32)]


def _tadd_inputs():
    rng = np.random.default_rng(SEED + 2)
    return [
        rng.integers(-100, 100, size=TADD_SHAPE, dtype=np.int16),
        rng.integers(-100, 100, size=TADD_SHAPE, dtype=np.int16),
    ]


def _tadds_inputs():
    rng = np.random.default_rng(SEED + 3)
    return [rng.integers(-50, 50, size=TADDS_SHAPE, dtype=np.int8)]


def _tcmps_inputs():
    rng = np.random.default_rng(SEED + 4)
    src = rng.integers(0, 10, size=TCMPS_DATA_SHAPE, dtype=np.int8)
    return [src]


def _tsels_inputs():
    rng = np.random.default_rng(SEED + 5)
    predicate = rng.integers(
        0,
        2,
        size=TSELS_DATA_SHAPE,
        dtype=np.uint8,
    )
    mask = np.packbits(predicate, axis=1, bitorder="little").view(np.int8)
    src = rng.integers(-20, 20, size=TSELS_DATA_SHAPE, dtype=np.int8)
    return [mask, src]


def _tcvt_inputs():
    rng = np.random.default_rng(SEED + 6)
    src = rng.integers(-1000, 1000, size=TCVT_SHAPE).astype(np.float32)
    return [src]


CASES = [
    _equivalence_case(
        "elementwise_tabs_f32_multirow_tail_1d_2d_equivalence",
        elementwise_tabs_1d_2d_equivalence,
        inputs=_tabs_inputs,
        expected=lambda src: np.abs(src),
        output_dtype=np.float32,
        guard_value=np.float32(12345.0),
    ),
    _equivalence_case(
        "elementwise_tadd_i16_tail_1d_2d_equivalence",
        elementwise_tadd_1d_2d_equivalence,
        inputs=_tadd_inputs,
        expected=lambda lhs, rhs: (lhs + rhs).astype(np.int16),
        output_dtype=np.int16,
        guard_value=np.int16(12345),
    ),
    _equivalence_case(
        "elementwise_tadds_i8_tail_1d_2d_equivalence",
        elementwise_tadds_1d_2d_equivalence,
        inputs=_tadds_inputs,
        expected=lambda src: (src.astype(np.int16) + 7).astype(np.int8),
        output_dtype=np.int8,
        guard_value=np.int8(101),
    ),
    _equivalence_case(
        "elementwise_tcmps_i8_aligned_1d_2d_equivalence",
        elementwise_tcmps_1d_2d_equivalence,
        inputs=_tcmps_inputs,
        expected=lambda src: np.packbits(
            src == np.int8(5),
            axis=1,
            bitorder="little",
        ),
        output_dtype=np.uint8,
        guard_value=np.uint8(0xA5),
    ),
    _equivalence_case(
        "elementwise_tsels_i8_aligned_1d_2d_equivalence",
        elementwise_tsels_1d_2d_equivalence,
        inputs=_tsels_inputs,
        expected=lambda mask, src: np.where(
            np.unpackbits(
                mask.view(np.uint8),
                axis=1,
                count=TSELS_DATA_SHAPE[1],
                bitorder="little",
            ).astype(bool),
            src,
            np.int8(-3),
        ),
        output_dtype=np.int8,
        guard_value=np.int8(101),
    ),
    _equivalence_case(
        "elementwise_tcvt_f32_i16_tail_1d_2d_equivalence",
        elementwise_tcvt_1d_2d_equivalence,
        inputs=_tcvt_inputs,
        expected=lambda src: src.astype(np.int16),
        output_dtype=np.int16,
        guard_value=np.int16(12345),
    ),
    _equivalence_case(
        "elementwise_texpands_i32_tail_1d_2d_equivalence",
        elementwise_texpands_1d_2d_equivalence,
        inputs=lambda: [],
        expected=lambda: np.full(TEXPANDS_SHAPE, 23, dtype=np.int32),
        output_dtype=np.int32,
        guard_value=np.int32(123456789),
    ),
]


auto_main(globals())
