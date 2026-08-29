#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

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
    raise RuntimeError("Unable to locate test/dsl-st/common.py from allreduce.py")


_bootstrap_dsl_st_common()

from common import auto_main, golden_output_case  # noqa: E402
from ptodsl import pto  # noqa: E402


WARP_THREADS = 32
CROSS_THREADS = 128
SEED = 20260707


@pto.simt
def allreduce_sum_body(
    inp: pto.ptr(pto.f32, "gm"),
    out: pto.ptr(pto.f32, "gm"),
    scratch: pto.ptr(pto.f32, "ub"),
    *,
    threads: pto.const_expr,
):
    tid = pto.get_tid_x()
    idx = pto.index_cast(tid)
    value = pto.load(inp, idx)
    reduced = pto.simt_allreduce_sum(
        value,
        threads=threads,
        scale=1,
        thread_offset=0,
        scratch=scratch,
    )
    pto.store(reduced, out, idx)


@pto.simt
def allreduce_max_body(
    inp: pto.ptr(pto.f32, "gm"),
    out: pto.ptr(pto.f32, "gm"),
    scratch: pto.ptr(pto.f32, "ub"),
    *,
    threads: pto.const_expr,
):
    tid = pto.get_tid_x()
    idx = pto.index_cast(tid)
    value = pto.load(inp, idx)
    reduced = pto.simt_allreduce_max(
        value,
        threads=threads,
        scale=1,
        thread_offset=0,
        scratch=scratch,
    )
    pto.store(reduced, out, idx)


@pto.simt
def allreduce_min_body(
    inp: pto.ptr(pto.f32, "gm"),
    out: pto.ptr(pto.f32, "gm"),
    scratch: pto.ptr(pto.f32, "ub"),
    *,
    threads: pto.const_expr,
):
    tid = pto.get_tid_x()
    idx = pto.index_cast(tid)
    value = pto.load(inp, idx)
    reduced = pto.simt_allreduce_min(
        value,
        threads=threads,
        scale=1,
        thread_offset=0,
        scratch=scratch,
    )
    pto.store(reduced, out, idx)


@pto.jit(name="allreduce_warp_sum_kernel", kernel_kind="vector", target="a5", mode="explicit", insert_sync=False)
def allreduce_warp_sum_kernel(inp: pto.ptr(pto.f32, "gm"), out: pto.ptr(pto.f32, "gm")):
    scratch = pto.castptr(pto.const(0, dtype=pto.ui64), pto.ptr(pto.f32, "ub"))
    allreduce_sum_body[WARP_THREADS, 1, 1](inp, out, scratch, threads=WARP_THREADS)
    pto.pipe_barrier(pto.Pipe.ALL)


@pto.jit(name="allreduce_warp_max_kernel", kernel_kind="vector", target="a5", mode="explicit", insert_sync=False)
def allreduce_warp_max_kernel(inp: pto.ptr(pto.f32, "gm"), out: pto.ptr(pto.f32, "gm")):
    scratch = pto.castptr(pto.const(0, dtype=pto.ui64), pto.ptr(pto.f32, "ub"))
    allreduce_max_body[WARP_THREADS, 1, 1](inp, out, scratch, threads=WARP_THREADS)
    pto.pipe_barrier(pto.Pipe.ALL)


@pto.jit(name="allreduce_warp_min_kernel", kernel_kind="vector", target="a5", mode="explicit", insert_sync=False)
def allreduce_warp_min_kernel(inp: pto.ptr(pto.f32, "gm"), out: pto.ptr(pto.f32, "gm")):
    scratch = pto.castptr(pto.const(0, dtype=pto.ui64), pto.ptr(pto.f32, "ub"))
    allreduce_min_body[WARP_THREADS, 1, 1](inp, out, scratch, threads=WARP_THREADS)
    pto.pipe_barrier(pto.Pipe.ALL)


@pto.jit(name="allreduce_cross_sum_kernel", kernel_kind="vector", target="a5", mode="explicit", insert_sync=False)
def allreduce_cross_sum_kernel(inp: pto.ptr(pto.f32, "gm"), out: pto.ptr(pto.f32, "gm")):
    scratch = pto.castptr(pto.const(0, dtype=pto.ui64), pto.ptr(pto.f32, "ub"))
    allreduce_sum_body[CROSS_THREADS, 1, 1](inp, out, scratch, threads=CROSS_THREADS)
    pto.pipe_barrier(pto.Pipe.ALL)


@pto.jit(name="allreduce_cross_max_kernel", kernel_kind="vector", target="a5", mode="explicit", insert_sync=False)
def allreduce_cross_max_kernel(inp: pto.ptr(pto.f32, "gm"), out: pto.ptr(pto.f32, "gm")):
    scratch = pto.castptr(pto.const(0, dtype=pto.ui64), pto.ptr(pto.f32, "ub"))
    allreduce_max_body[CROSS_THREADS, 1, 1](inp, out, scratch, threads=CROSS_THREADS)
    pto.pipe_barrier(pto.Pipe.ALL)


@pto.jit(name="allreduce_cross_min_kernel", kernel_kind="vector", target="a5", mode="explicit", insert_sync=False)
def allreduce_cross_min_kernel(inp: pto.ptr(pto.f32, "gm"), out: pto.ptr(pto.f32, "gm")):
    scratch = pto.castptr(pto.const(0, dtype=pto.ui64), pto.ptr(pto.f32, "ub"))
    allreduce_min_body[CROSS_THREADS, 1, 1](inp, out, scratch, threads=CROSS_THREADS)
    pto.pipe_barrier(pto.Pipe.ALL)


def make_inputs(threads: int, reducer: str):
    def materialize():
        rng = np.random.default_rng(SEED + threads + len(reducer))
        values = rng.uniform(-8.0, 8.0, size=threads).astype(np.float32)
        if reducer == "max":
            values[threads // 2 + 1] = np.float32(16.0)
        elif reducer == "min":
            values[threads // 2 + 1] = np.float32(-16.0)
        return [values]

    return materialize


def make_expected(reducer: str):
    def materialize(values):
        if reducer == "sum":
            result = np.sum(values, dtype=np.float32)
        elif reducer == "max":
            result = np.max(values)
        elif reducer == "min":
            result = np.min(values)
        else:
            raise ValueError(f"unsupported allreduce reducer: {reducer}")
        return np.full((values.shape[0],), result, dtype=np.float32)

    return materialize


def case(name: str, kernel, *, threads: int, reducer: str):
    return golden_output_case(
        name,
        kernel,
        inputs=make_inputs(threads, reducer),
        expected=make_expected(reducer),
        rtol=1e-5,
        atol=1e-5,
    )


CASES = [
    case("allreduce_warp_sum", allreduce_warp_sum_kernel, threads=WARP_THREADS, reducer="sum"),
    case("allreduce_warp_max", allreduce_warp_max_kernel, threads=WARP_THREADS, reducer="max"),
    case("allreduce_warp_min", allreduce_warp_min_kernel, threads=WARP_THREADS, reducer="min"),
    case("allreduce_cross_sum", allreduce_cross_sum_kernel, threads=CROSS_THREADS, reducer="sum"),
    case("allreduce_cross_max", allreduce_cross_max_kernel, threads=CROSS_THREADS, reducer="max"),
    case("allreduce_cross_min", allreduce_cross_min_kernel, threads=CROSS_THREADS, reducer="min"),
]


auto_main(globals())
