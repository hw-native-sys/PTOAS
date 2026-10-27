#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Issue #1690: store_acc2gm nz2nd / nz2nz with a short valid region.

Two template bugs corrupt the acc2gm flow once the valid rows are shorter
than the tile's physical rows:

- the nd2nz GM->MAT load wrote the tile's VALID rows into the NZ loop3
  destination stride, so C0 fractals overlapped in L1 whenever the inner
  extent crossed a C0 boundary (template_tload_gm_to_mat_nd2nz);
- the nz2nz ACC->GM store hardcoded the destination stride as the column
  count instead of the 16-row-aligned row count times C0 (channel split
  keeps C0 at 8 for an f32 C0=8 view), so every column block after the
  first landed at the wrong pitch (template_tstore_acc_to_gm_nz2nz).

Golden: nz2nd is lhs @ rhs; nz2nz is the same product packed as NZ with
C0 = 8.
"""

import numpy as np

from common import auto_main, golden_output_case
from ptoas.mlir.dialects import arith
from ptoas.mlir.ir import IndexType, IntegerAttr
from ptodsl import pto

C0 = 8  # f32 channel-split NZ width


def idx(values):
    ty = IndexType.get()
    return [arith.ConstantOp(ty, IntegerAttr.get(ty, int(v))).result for v in values]


def nd_view(rows, cols):
    span = rows * cols
    return [1, 1, 1, rows, cols], [span, span, span, cols, 1]


def nz_view(rows, cols):
    n_tiles, k_tiles = rows // 16, cols // C0
    stride2 = 16 * C0
    stride1 = n_tiles * stride2
    return [1, k_tiles, n_tiles, 16, C0], [k_tiles * stride1, stride1, stride2, C0, 1]


def _acc2gm_kernel(name, m, n, k, mat_m, mat_k, mat_n, form):
    @pto.jit(name=name, kernel_kind="cube", target="a5", backend="vpto", mode="auto")
    def kernel(
        lhs_ptr: pto.ptr(pto.f32, "gm"),
        rhs_ptr: pto.ptr(pto.f32, "gm"),
        out_ptr: pto.ptr(pto.f32, "gm"),
    ):
        lhs_shape, lhs_strides = nd_view(mat_m, mat_k)
        rhs_shape, rhs_strides = nd_view(mat_k, mat_n)
        lhs_view = pto.make_tensor_view(lhs_ptr, shape=lhs_shape, strides=lhs_strides)
        rhs_view = pto.make_tensor_view(rhs_ptr, shape=rhs_shape, strides=rhs_strides)
        if form == "nz2nz":
            out_shape, out_strides = nz_view(m, n)
            out_view = pto.make_tensor_view(
                out_ptr, shape=out_shape, strides=out_strides, layout="NZ"
            )
        else:
            out_shape = None
            out_view = pto.make_tensor_view(out_ptr, shape=[m, n], strides=[n, 1])
        lhs_mat = pto.alloc_tile(
            shape=[mat_m, mat_k], dtype=pto.f32, memory_space=pto.MemorySpace.MAT,
            valid_shape=[m, k], blayout="ColMajor", slayout="RowMajor", fractal_size=512,
        )
        rhs_mat = pto.alloc_tile(
            shape=[mat_k, mat_n], dtype=pto.f32, memory_space=pto.MemorySpace.MAT,
            valid_shape=[k, n], blayout="ColMajor", slayout="RowMajor", fractal_size=512,
        )
        pto.tile.load(lhs_view, lhs_mat, offsets=idx([0, 0, 0, 0, 0]), sizes=idx(lhs_shape))
        pto.tile.load(rhs_view, rhs_mat, offsets=idx([0, 0, 0, 0, 0]), sizes=idx(rhs_shape))
        lhs = pto.alloc_tile(
            shape=[mat_m, mat_k], dtype=pto.f32, memory_space=pto.MemorySpace.LEFT,
            valid_shape=[m, k], blayout="ColMajor", slayout="RowMajor",
        )
        rhs = pto.alloc_tile(
            shape=[mat_k, mat_n], dtype=pto.f32, memory_space=pto.MemorySpace.RIGHT,
            valid_shape=[k, n], blayout="RowMajor", slayout="ColMajor",
        )
        dst = pto.alloc_tile(
            shape=[mat_m, mat_n], dtype=pto.f32, memory_space=pto.MemorySpace.ACC,
            valid_shape=[m, n], blayout="ColMajor", slayout="RowMajor",
        )
        pto.tile.mov(lhs_mat, lhs)
        pto.tile.mov(rhs_mat, rhs)
        pto.tile.matmul(lhs, rhs, dst)
        if form == "nz2nz":
            pto.tile.store(dst, out_view, offsets=idx([0, 0, 0, 0, 0]), sizes=idx(out_shape))
        else:
            pto.tile.store(dst, out_view)

    return kernel


def _inputs(m, n, k, mat_m, mat_k, mat_n, form):
    rng = np.random.default_rng(1690)
    lhs = rng.uniform(-1.0, 1.0, size=(mat_m, mat_k)).astype(np.float32)
    rhs = rng.uniform(-1.0, 1.0, size=(mat_k, mat_n)).astype(np.float32)
    # The valid region outside [m, k] / [k, n] is zero-padded GM.
    lhs[m:, :] = 0.0
    lhs[:, k:] = 0.0
    rhs[k:, :] = 0.0
    rhs[:, n:] = 0.0
    out_size = m * n if form == "nz2nd" else (m // 16) * (n // C0) * 16 * C0
    out = np.zeros(out_size, dtype=np.float32)
    return [lhs.reshape(-1), rhs.reshape(-1), out]


def _golden_nd(product, m, n):
    return product[:m, :n].reshape(-1)


def _golden_nz(product, m, n):
    # Pack as NZ with C0 = 8, matching nz_view's [1, k_tiles, n_tiles, 16, C0]
    # memory order: column blocks outermost, then 16-row tiles.
    n_tiles, k_tiles = m // 16, n // C0
    packed = np.zeros((k_tiles, n_tiles, 16, C0), dtype=np.float32)
    for kt in range(k_tiles):
        for nt in range(n_tiles):
            packed[kt, nt] = product[nt * 16:(nt + 1) * 16, kt * C0:(kt + 1) * C0]
    return packed.reshape(-1)


def _case(name, m, n, k, mat_m, mat_k, mat_n, form):
    kernel = _acc2gm_kernel(f"{name}_kernel", m, n, k, mat_m, mat_k, mat_n, form)
    lhs, rhs, out = _inputs(m, n, k, mat_m, mat_k, mat_n, form)
    product = lhs.reshape(mat_m, mat_k) @ rhs.reshape(mat_k, mat_n)

    if form == "nz2nd":
        def make_inputs(lhs=lhs, rhs=rhs):
            return [lhs.copy(), rhs.copy()]

        def golden(lhs, rhs, m=m, n=n):
            return _golden_nd(product, m, n)

        return golden_output_case(
            name, kernel, inputs=make_inputs, expected=golden,
            rtol=1e-4, atol=1e-4)

    # nz2nz: the NZ-packed buffer is the kernel's own out argument, so the
    # launch receives exactly the three kernel arguments and the compare
    # reads it back directly.
    expected = _golden_nz(product, m, n)

    def make_case(lhs=lhs, rhs=rhs, out=out, expected=expected):
        return [lhs.copy(), rhs.copy(), out.copy()], expected.copy(), []

    def check(device_inputs, golden):
        np.testing.assert_allclose(
            device_inputs[-1].cpu().numpy(), golden, rtol=1e-4, atol=1e-4)

    return {
        "name": name,
        "kernel": kernel,
        "make_case": make_case,
        "check": check,
    }


CASES = [
    # nz2nd, valid rows shorter than the physical tile rows: the load used to
    # overlap C0 fractals (K = 15 and N = 32 both cross C0 = 8 boundaries).
    _case("store_acc2gm_f32_nz2nd_case2", 31, 32, 15, 32, 16, 32, "nz2nd"),
    # nz2nd control, valid shape equal to the static shape.
    _case("store_acc2gm_f32_nz2nd_aligned", 32, 64, 32, 32, 32, 64, "nz2nd"),
    # nz2nz, B still has a short valid region and the store used to pitch
    # column blocks at n instead of align16(rows) * C0 = 32 * 8 = 256.
    _case("store_acc2gm_f32_nz2nz_case7", 32, 64, 25, 32, 32, 64, "nz2nz"),
]

auto_main(globals())
