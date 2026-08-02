#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Runtime equivalence checks for registered element-wise 1D/2D templates."""

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
from ptodsl._ast_rewrite import rewrite_jit_function
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

TABS_SHAPE = (1, 72)
TADD_SHAPE = (4, 80)
TADDS_SHAPE = (4, 96)
TCMPS_DATA_SHAPE = (4, 256)
TCMPS_MASK_SHAPE = (4, 32)
TSELS_DATA_SHAPE = (4, 256)
TSELS_MASK_SHAPE = (4, 32)
TCVT_SHAPE = (4, 80)
TEXPANDS_SHAPE = (4, 72)


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


_tcvt_f32_to_i16_2d_body = rewrite_jit_function(
    template_tcvt_f32_to_i16.py_fn
)


def _ub_ptr(dtype, addr):
    return pto.castptr(
        pto.const(addr, dtype=pto.i64),
        pto.ptr(dtype, "ub"),
    )


def _load_to_ub(source, dtype, addr, byte_count):
    destination = _ub_ptr(dtype, addr)
    pto.get_buf(pto.Pipe.MTE2, 0)
    pto.mte_gm_ub(
        source,
        destination,
        0,
        byte_count,
        nburst=(1, byte_count, byte_count),
    )
    pto.rls_buf(pto.Pipe.MTE2, 0)


def _store_from_ub(source_addr, dtype, destination, byte_count):
    source = _ub_ptr(dtype, source_addr)
    logical_destination = pto.addptr(destination, GUARD_ELEMENTS)
    pto.get_buf(pto.Pipe.MTE3, 0)
    pto.mte_ub_gm(
        source,
        logical_destination,
        byte_count,
        nburst=(1, byte_count, byte_count),
    )
    pto.rls_buf(pto.Pipe.MTE3, 0)


def _wait_for_loads():
    pto.set_flag("MTE2", "V", event_id=0)
    pto.wait_flag("MTE2", "V", event_id=0)


def _wait_for_stores():
    pto.set_flag("V", "MTE3", event_id=0)
    pto.wait_flag("V", "MTE3", event_id=0)


@pto.jit(
    name="elementwise_tabs_1d_2d_equivalence",
    target="a5",
    kernel_kind="vector",
    mode="explicit",
)
def elementwise_tabs_1d_2d_equivalence(
    src_ptr: pto.ptr(pto.f32, "gm"),
    out_1d_ptr: pto.ptr(pto.f32, "gm"),
    out_2d_ptr: pto.ptr(pto.f32, "gm"),
):
    rows, cols = TABS_SHAPE
    byte_count = rows * cols * 4

    src = pto.alloc_tile(shape=[rows, cols], dtype=pto.f32, addr=0)
    out_1d = pto.alloc_tile(shape=[rows, cols], dtype=pto.f32, addr=1024)
    out_2d = pto.alloc_tile(shape=[rows, cols], dtype=pto.f32, addr=2048)

    _load_to_ub(src_ptr, pto.f32, 0, byte_count)
    _wait_for_loads()
    template_tabs_1d.py_fn(src, out_1d)
    template_tabs.py_fn(src, out_2d)
    _wait_for_stores()
    _store_from_ub(1024, pto.f32, out_1d_ptr, byte_count)
    _store_from_ub(2048, pto.f32, out_2d_ptr, byte_count)
    pto.pipe_barrier(pto.Pipe.ALL)


@pto.jit(
    name="elementwise_tadd_1d_2d_equivalence",
    target="a5",
    kernel_kind="vector",
    mode="explicit",
)
def elementwise_tadd_1d_2d_equivalence(
    lhs_ptr: pto.ptr(pto.i16, "gm"),
    rhs_ptr: pto.ptr(pto.i16, "gm"),
    out_1d_ptr: pto.ptr(pto.i16, "gm"),
    out_2d_ptr: pto.ptr(pto.i16, "gm"),
):
    rows, cols = TADD_SHAPE
    byte_count = rows * cols * 2

    lhs = pto.alloc_tile(shape=[rows, cols], dtype=pto.i16, addr=0)
    rhs = pto.alloc_tile(shape=[rows, cols], dtype=pto.i16, addr=1024)
    out_1d = pto.alloc_tile(shape=[rows, cols], dtype=pto.i16, addr=2048)
    out_2d = pto.alloc_tile(shape=[rows, cols], dtype=pto.i16, addr=3072)

    _load_to_ub(lhs_ptr, pto.i16, 0, byte_count)
    _load_to_ub(rhs_ptr, pto.i16, 1024, byte_count)
    _wait_for_loads()
    template_tadd_1d.py_fn(lhs, rhs, out_1d)
    template_tadd.py_fn(lhs, rhs, out_2d)
    _wait_for_stores()
    _store_from_ub(2048, pto.i16, out_1d_ptr, byte_count)
    _store_from_ub(3072, pto.i16, out_2d_ptr, byte_count)
    pto.pipe_barrier(pto.Pipe.ALL)


@pto.jit(
    name="elementwise_tadds_1d_2d_equivalence",
    target="a5",
    kernel_kind="vector",
    mode="explicit",
)
def elementwise_tadds_1d_2d_equivalence(
    src_ptr: pto.ptr(pto.i8, "gm"),
    out_1d_ptr: pto.ptr(pto.i8, "gm"),
    out_2d_ptr: pto.ptr(pto.i8, "gm"),
):
    rows, cols = TADDS_SHAPE
    byte_count = rows * cols

    src = pto.alloc_tile(shape=[rows, cols], dtype=pto.i8, addr=0)
    out_1d = pto.alloc_tile(shape=[rows, cols], dtype=pto.i8, addr=512)
    out_2d = pto.alloc_tile(shape=[rows, cols], dtype=pto.i8, addr=1024)

    _load_to_ub(src_ptr, pto.i8, 0, byte_count)
    _wait_for_loads()
    template_tadds_1d.py_fn(src, 7, out_1d)
    template_tadds.py_fn(src, 7, out_2d)
    _wait_for_stores()
    _store_from_ub(512, pto.i8, out_1d_ptr, byte_count)
    _store_from_ub(1024, pto.i8, out_2d_ptr, byte_count)
    pto.pipe_barrier(pto.Pipe.ALL)


@pto.jit(
    name="elementwise_tcmps_1d_2d_equivalence",
    target="a5",
    kernel_kind="vector",
    mode="explicit",
)
def elementwise_tcmps_1d_2d_equivalence(
    src_ptr: pto.ptr(pto.i8, "gm"),
    out_1d_ptr: pto.ptr(pto.ui8, "gm"),
    out_2d_ptr: pto.ptr(pto.ui8, "gm"),
):
    data_rows, data_cols = TCMPS_DATA_SHAPE
    mask_rows, mask_cols = TCMPS_MASK_SHAPE
    src_byte_count = data_rows * data_cols
    dst_byte_count = mask_rows * mask_cols

    src = pto.alloc_tile(shape=[data_rows, data_cols], dtype=pto.i8, addr=0)
    out_1d = pto.alloc_tile(
        shape=[mask_rows, mask_cols],
        dtype=pto.ui8,
        addr=4096,
    )
    out_2d = pto.alloc_tile(
        shape=[mask_rows, mask_cols],
        dtype=pto.ui8,
        addr=4352,
    )

    _load_to_ub(src_ptr, pto.i8, 0, src_byte_count)
    _wait_for_loads()
    template_tcmps_1d.py_fn(src, 5, out_1d)
    template_tcmps.py_fn(src, 5, out_2d)
    _wait_for_stores()
    _store_from_ub(4096, pto.ui8, out_1d_ptr, dst_byte_count)
    _store_from_ub(4352, pto.ui8, out_2d_ptr, dst_byte_count)
    pto.pipe_barrier(pto.Pipe.ALL)


@pto.jit(
    name="elementwise_tsels_1d_2d_equivalence",
    target="a5",
    kernel_kind="vector",
    mode="explicit",
)
def elementwise_tsels_1d_2d_equivalence(
    mask_ptr: pto.ptr(pto.i8, "gm"),
    src_ptr: pto.ptr(pto.i8, "gm"),
    out_1d_ptr: pto.ptr(pto.i8, "gm"),
    out_2d_ptr: pto.ptr(pto.i8, "gm"),
):
    data_rows, data_cols = TSELS_DATA_SHAPE
    mask_rows, mask_cols = TSELS_MASK_SHAPE
    mask_byte_count = mask_rows * mask_cols
    data_byte_count = data_rows * data_cols

    mask = pto.alloc_tile(shape=[mask_rows, mask_cols], dtype=pto.i8, addr=0)
    src = pto.alloc_tile(
        shape=[data_rows, data_cols],
        dtype=pto.i8,
        addr=256,
    )
    tmp = pto.alloc_tile(shape=[1, 256], dtype=pto.i8, addr=1280)
    out_1d = pto.alloc_tile(
        shape=[data_rows, data_cols],
        dtype=pto.i8,
        addr=2048,
    )
    out_2d = pto.alloc_tile(
        shape=[data_rows, data_cols],
        dtype=pto.i8,
        addr=3072,
    )

    _load_to_ub(mask_ptr, pto.i8, 0, mask_byte_count)
    _load_to_ub(src_ptr, pto.i8, 256, data_byte_count)
    _wait_for_loads()
    template_tsels_1d.py_fn(mask, src, tmp, -3, out_1d)
    template_tsels.py_fn(mask, src, tmp, -3, out_2d)
    _wait_for_stores()
    _store_from_ub(2048, pto.i8, out_1d_ptr, data_byte_count)
    _store_from_ub(3072, pto.i8, out_2d_ptr, data_byte_count)
    pto.pipe_barrier(pto.Pipe.ALL)


@pto.jit(
    name="elementwise_tcvt_1d_2d_equivalence",
    target="a5",
    kernel_kind="vector",
    mode="explicit",
)
def elementwise_tcvt_1d_2d_equivalence(
    src_ptr: pto.ptr(pto.f32, "gm"),
    out_1d_ptr: pto.ptr(pto.i16, "gm"),
    out_2d_ptr: pto.ptr(pto.i16, "gm"),
):
    rows, cols = TCVT_SHAPE
    src_byte_count = rows * cols * 4
    dst_byte_count = rows * cols * 2

    src = pto.alloc_tile(shape=[rows, cols], dtype=pto.f32, addr=0)
    out_1d = pto.alloc_tile(shape=[rows, cols], dtype=pto.i16, addr=2048)
    out_2d = pto.alloc_tile(shape=[rows, cols], dtype=pto.i16, addr=3072)

    _load_to_ub(src_ptr, pto.f32, 0, src_byte_count)
    _wait_for_loads()
    template_tcvt_f32_to_i16_1d.py_fn(src, out_1d)
    _tcvt_f32_to_i16_2d_body(src, out_2d)
    _wait_for_stores()
    _store_from_ub(2048, pto.i16, out_1d_ptr, dst_byte_count)
    _store_from_ub(3072, pto.i16, out_2d_ptr, dst_byte_count)
    pto.pipe_barrier(pto.Pipe.ALL)


@pto.jit(
    name="elementwise_texpands_1d_2d_equivalence",
    target="a5",
    kernel_kind="vector",
    mode="explicit",
)
def elementwise_texpands_1d_2d_equivalence(
    out_1d_ptr: pto.ptr(pto.i32, "gm"),
    out_2d_ptr: pto.ptr(pto.i32, "gm"),
):
    rows, cols = TEXPANDS_SHAPE
    byte_count = rows * cols * 4

    out_1d = pto.alloc_tile(shape=[rows, cols], dtype=pto.i32, addr=0)
    out_2d = pto.alloc_tile(shape=[rows, cols], dtype=pto.i32, addr=2048)

    template_texpands_1d.py_fn(23, out_1d)
    template_texpands.py_fn(23, out_2d)
    _wait_for_stores()
    _store_from_ub(0, pto.i32, out_1d_ptr, byte_count)
    _store_from_ub(2048, pto.i32, out_2d_ptr, byte_count)
    pto.pipe_barrier(pto.Pipe.ALL)


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
        "elementwise_tabs_f32_single_row_tail_1d_2d_equivalence",
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
