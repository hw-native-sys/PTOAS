# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Shared e2e test harness: kernel builders, launch, and reference functions."""

# NOTE: do NOT add "from __future__ import annotations" here.
# make_binary_kernel uses dynamic annotation expressions (pto.ptr(pto_dtype, "gm"))
# which must be evaluated at definition time, not stored as strings.

import time
from typing import Callable

import numpy as np

from ptodsl import pto

BINARY_OPS = {
    "add": (pto.tile.add, lambda x, y: x + y),
    "addrelu": (pto.tile.addrelu, lambda x, y: np.maximum(x + y, 0)),
    "sub": (pto.tile.sub, lambda x, y: x - y),
    "mul": (pto.tile.mul, lambda x, y: x * y),
    "div": (pto.tile.div, lambda x, y: x / (y + 1e-8)),
    "max": (pto.tile.max, lambda x, y: np.maximum(x, y)),
    "min": (pto.tile.min, lambda x, y: np.minimum(x, y)),
}

INT_OPS = {
    "bit_and": (pto.tile.bit_and, lambda x, y: x & y),
    "bit_or":  (pto.tile.bit_or,  lambda x, y: x | y),
    "bit_xor": (pto.tile.bit_xor, lambda x, y: x ^ y),
}

SHIFT_OPS = {
    "bit_shls": (pto.tile.bit_shls, lambda x, n: np.left_shift(x, n)),
    "bit_shrs": (pto.tile.bit_shrs, lambda x, n: np.right_shift(x, n)),
}

UNARY_OPS = {
    "abs":  (pto.tile.abs,  lambda x: np.abs(x)),
    "relu": (pto.tile.relu, lambda x: np.maximum(x, 0)),
    "neg":  (pto.tile.neg,  lambda x: np.negative(x)),
    "exp":  (pto.tile.exp,  lambda x: np.exp(x)),
    "log":  (pto.tile.log,  lambda x: np.log(x)),
    "sqrt": (pto.tile.sqrt, lambda x: np.sqrt(np.abs(x))),
    "rsqrt":(pto.tile.rsqrt,lambda x: 1.0 / np.sqrt(np.abs(x))),
    "recip":(pto.tile.recip,lambda x: 1.0 / x),
}

POSITIVE_INPUT_OPS = {"log", "sqrt", "rsqrt", "recip"}

SCALAR_OPS = {
    "adds": (pto.tile.adds, lambda x, s: x + s),
    "muls": (pto.tile.muls, lambda x, s: x * s),
    "maxs": (pto.tile.maxs, lambda x, s: np.maximum(x, s)),
    "mins": (pto.tile.mins, lambda x, s: np.minimum(x, s)),
}


def _npu_stream(torch):
    return torch.npu.current_stream()._as_parameter_  # noqa: SLF001


def _torch_dtype(torch, dtype_str: str):
    return getattr(torch, dtype_str)


def make_input_int(shape, torch, seed=42):
    """Return an NPU i16 tensor filled with small positive integers."""
    rng = np.random.RandomState(seed)
    x = rng.randint(0, 100, size=shape).astype(np.int16)
    return torch.from_numpy(x).to(device="npu:0", dtype=torch.int16)


def launch_and_check_int(
    *,
    kernel_handle,
    ref_fn: Callable,
    shape: tuple[int, int],
    torch,
    seed: int = 42,
):
    """Compile, launch, and numerical-check one i16 kernel specialization."""
    x = make_input_int(shape, torch, seed=seed)
    y = make_input_int(shape, torch, seed=seed + 1)
    z = torch.empty(shape, dtype=torch.int16, device="npu:0")
    ref = ref_fn(x.cpu().numpy(), y.cpu().numpy()).astype(np.int16)
    stream = _npu_stream(torch)

    t0 = time.perf_counter()
    compiled = kernel_handle.compile()
    compile_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    compiled[1, stream](x.data_ptr(), y.data_ptr(), z.data_ptr())
    torch.npu.synchronize()
    launch_s = time.perf_counter() - t0

    actual = z.cpu().numpy()
    np.testing.assert_array_equal(actual, ref)
    return compile_s, launch_s


def launch_and_check_shift(
    *,
    kernel_handle,
    ref_fn: Callable,
    shape: tuple[int, int],
    shift_val: int,
    torch,
    seed: int = 42,
):
    """Compile, launch, and numerical-check one i16 scalar-shift kernel."""
    x = make_input_int(shape, torch, seed=seed)
    z = torch.empty(shape, dtype=torch.int16, device="npu:0")
    ref = ref_fn(x.cpu().numpy(), shift_val).astype(np.int16)
    stream = _npu_stream(torch)

    t0 = time.perf_counter()
    compiled = kernel_handle.compile()
    compile_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    compiled[1, stream](x.data_ptr(), z.data_ptr())
    torch.npu.synchronize()
    launch_s = time.perf_counter() - t0

    actual = z.cpu().numpy()
    np.testing.assert_array_equal(actual, ref)
    return compile_s, launch_s


def make_binary_kernel(
    op_name: str,
    rows: int,
    cols: int,
    dtype_str: str = "float32",
    target: str = "a3",
    backend: str = "vpto",
    kernel_kind: str = "vector",
):
    """Return a ``@pto.jit`` KernelHandle for an elementwise binary op.

    The generated kernel uses the same 5-D tile-buffer pattern as the
    ``tadd_launch_a3.py`` / ``bop_launch_a3.py`` examples.
    """
    tile_op_fn = (BINARY_OPS.get(op_name) or INT_OPS.get(op_name))[0]
    pto_dtype = getattr(pto, dtype_str)
    fn_name = f"bin_{op_name}_{dtype_str}_{rows}x{cols}"

    def kernel_body(
        A_ptr: pto.ptr(pto_dtype, "gm"),
        B_ptr: pto.ptr(pto_dtype, "gm"),
        C_ptr: pto.ptr(pto_dtype, "gm"),
    ) -> None:
        c0 = pto.const(0)
        c1 = pto.const(1)
        c_rows = pto.const(rows)
        c_cols = pto.const(cols)
        c_elems = pto.const(rows * cols)

        shape = [c1, c1, c1, c_rows, c_cols]
        strides = [c_elems, c_elems, c_elems, c_cols, c1]
        off = [c0, c0, c0, c0, c0]

        a_view = pto.make_tensor_view(A_ptr, shape=shape, strides=strides)
        b_view = pto.make_tensor_view(B_ptr, shape=shape, strides=strides)
        c_view = pto.make_tensor_view(C_ptr, shape=shape, strides=strides)

        a_part = pto.partition_view(a_view, offsets=off, sizes=shape)
        b_part = pto.partition_view(b_view, offsets=off, sizes=shape)
        c_part = pto.partition_view(c_view, offsets=off, sizes=shape)

        a_tile = pto.alloc_tile(shape=[rows, cols], dtype=pto_dtype)
        b_tile = pto.alloc_tile(shape=[rows, cols], dtype=pto_dtype)
        c_tile = pto.alloc_tile(shape=[rows, cols], dtype=pto_dtype)

        pto.tile.load(a_part, a_tile)
        pto.tile.load(b_part, b_tile)
        if op_name == "bit_xor":
            tmp_tile = pto.alloc_tile(shape=[rows, cols], dtype=pto_dtype)
            tile_op_fn(a_tile, b_tile, tmp_tile, c_tile)
        else:
            tile_op_fn(a_tile, b_tile, c_tile)
        pto.tile.store(c_tile, c_part)

    kernel_body.__name__ = fn_name
    return pto.jit(
        name=fn_name,
        kernel_kind=kernel_kind,
        target=target,
        backend=backend,
    )(kernel_body)


def make_shift_kernel(
    op_name: str,
    rows: int,
    cols: int,
    shift_val: int = 3,
    target: str = "a3",
    backend: str = "vpto",
    kernel_kind: str = "vector",
):
    """Return a ``@pto.jit`` KernelHandle for a scalar shift op (tshls/tshrs)."""
    tile_op_fn = SHIFT_OPS[op_name][0]
    pto_dtype = pto.int16
    fn_name = f"shift_{op_name}_int16_{rows}x{cols}_s{shift_val}"

    def kernel_body(
        A_ptr: pto.ptr(pto_dtype, "gm"),
        C_ptr: pto.ptr(pto_dtype, "gm"),
    ) -> None:
        c0 = pto.const(0)
        c1 = pto.const(1)
        c_rows = pto.const(rows)
        c_cols = pto.const(cols)
        c_elems = pto.const(rows * cols)

        shape = [c1, c1, c1, c_rows, c_cols]
        strides = [c_elems, c_elems, c_elems, c_cols, c1]
        off = [c0, c0, c0, c0, c0]

        a_view = pto.make_tensor_view(A_ptr, shape=shape, strides=strides)
        c_view = pto.make_tensor_view(C_ptr, shape=shape, strides=strides)

        a_part = pto.partition_view(a_view, offsets=off, sizes=shape)
        c_part = pto.partition_view(c_view, offsets=off, sizes=shape)

        a_tile = pto.alloc_tile(shape=[rows, cols], dtype=pto_dtype)
        c_tile = pto.alloc_tile(shape=[rows, cols], dtype=pto_dtype)

        pto.tile.load(a_part, a_tile)
        tile_op_fn(a_tile, shift_val, c_tile)
        pto.tile.store(c_tile, c_part)

    kernel_body.__name__ = fn_name
    return pto.jit(
        name=fn_name,
        kernel_kind=kernel_kind,
        target=target,
        backend=backend,
    )(kernel_body)


def make_input(shape, dtype, torch, seed=42):
    """Return an NPU tensor filled with seeded random small integers.

    Small integers ensure exact fp results, avoiding rounding differences
    on different hardware paths.
    """
    rng = np.random.RandomState(seed)
    x = rng.randint(1, 10, size=shape).astype(np.float32)
    return torch.from_numpy(x).to(device="npu:0", dtype=dtype)


def launch_and_check(
    *,
    op_name: str | None = None,
    kernel_handle,
    ref_fn: Callable,
    shape: tuple[int, int],
    dtype_str: str,
    torch,
    rtol: float = 1e-6,
    atol: float = 1e-6,
    seed: int = 42,
):
    """Compile, launch, and numerical-check one kernel specialization."""
    torch_dt = _torch_dtype(torch, dtype_str)

    if op_name == "addrelu":
        x = make_input_signed(shape, torch_dt, torch, seed=seed)
        y = make_input_signed(shape, torch_dt, torch, seed=seed + 1)
    else:
        x = make_input(shape, torch_dt, torch, seed=seed)
        y = make_input(shape, torch_dt, torch, seed=seed + 1)
    z = torch.empty(shape, dtype=torch_dt, device="npu:0")
    ref = ref_fn(x.cpu().numpy(), y.cpu().numpy())
    stream = _npu_stream(torch)

    t0 = time.perf_counter()
    compiled = kernel_handle.compile()
    compile_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    compiled[1, stream](x.data_ptr(), y.data_ptr(), z.data_ptr())
    torch.npu.synchronize()
    launch_s = time.perf_counter() - t0

    actual = z.cpu().numpy()
    # VRSQRT uses a hardware fast approximation; relax tolerance.
    eff_rtol = 1e-2 if op_name == "rsqrt" else rtol
    eff_atol = 1e-2 if op_name == "rsqrt" else atol
    np.testing.assert_allclose(actual, ref, rtol=eff_rtol, atol=eff_atol)
    return compile_s, launch_s


def make_unary_kernel(
    op_name: str,
    rows: int,
    cols: int,
    dtype_str: str = "float32",
    target: str = "a3",
    backend: str = "vpto",
    kernel_kind: str = "vector",
):
    """Return a ``@pto.jit`` KernelHandle for an elementwise unary op."""
    tile_op_fn = UNARY_OPS[op_name][0]
    pto_dtype = getattr(pto, dtype_str)
    fn_name = f"un_{op_name}_{dtype_str}_{rows}x{cols}"

    def kernel_body(
        A_ptr: pto.ptr(pto_dtype, "gm"),
        C_ptr: pto.ptr(pto_dtype, "gm"),
    ) -> None:
        c0 = pto.const(0)
        c1 = pto.const(1)
        c_rows = pto.const(rows)
        c_cols = pto.const(cols)
        c_elems = pto.const(rows * cols)

        shape = [c1, c1, c1, c_rows, c_cols]
        strides = [c_elems, c_elems, c_elems, c_cols, c1]
        off = [c0, c0, c0, c0, c0]

        a_view = pto.make_tensor_view(A_ptr, shape=shape, strides=strides)
        c_view = pto.make_tensor_view(C_ptr, shape=shape, strides=strides)

        a_part = pto.partition_view(a_view, offsets=off, sizes=shape)
        c_part = pto.partition_view(c_view, offsets=off, sizes=shape)

        a_tile = pto.alloc_tile(shape=[rows, cols], dtype=pto_dtype)
        c_tile = pto.alloc_tile(shape=[rows, cols], dtype=pto_dtype)

        pto.tile.load(a_part, a_tile)
        tile_op_fn(a_tile, c_tile)
        pto.tile.store(c_tile, c_part)

    kernel_body.__name__ = fn_name
    return pto.jit(
        name=fn_name,
        kernel_kind=kernel_kind,
        target=target,
        backend=backend,
    )(kernel_body)


def make_input_signed(shape, dtype, torch, seed=42):
    """Return an NPU tensor filled with signed random small integers.

    Includes negative values so abs/relu are meaningful.
    """
    rng = np.random.RandomState(seed)
    x = rng.randint(-10, 10, size=shape).astype(np.float32)
    return torch.from_numpy(x).to(device="npu:0", dtype=dtype)


def launch_and_check_unary(
    *,
    op_name: str,
    kernel_handle,
    ref_fn: Callable,
    shape: tuple[int, int],
    dtype_str: str,
    torch,
    rtol: float = 1e-6,
    atol: float = 1e-6,
    seed: int = 42,
):
    """Compile, launch, and numerical-check one unary kernel specialization."""
    torch_dt = _torch_dtype(torch, dtype_str)

    if op_name in POSITIVE_INPUT_OPS:
        x = make_input(shape, torch_dt, torch, seed=seed)
        ref = ref_fn(x.cpu().numpy())
    else:
        x = make_input_signed(shape, torch_dt, torch, seed=seed)
        ref = ref_fn(x.cpu().numpy())
    z = torch.empty(shape, dtype=torch_dt, device="npu:0")
    stream = _npu_stream(torch)

    t0 = time.perf_counter()
    compiled = kernel_handle.compile()
    compile_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    compiled[1, stream](x.data_ptr(), z.data_ptr())
    torch.npu.synchronize()
    launch_s = time.perf_counter() - t0

    actual = z.cpu().numpy()
    eff_rtol = 1e-2 if op_name == "rsqrt" else rtol
    eff_atol = 1e-2 if op_name == "rsqrt" else atol
    np.testing.assert_allclose(actual, ref, rtol=eff_rtol, atol=eff_atol)
    return compile_s, launch_s


def make_scalar_kernel(
    op_name: str,
    rows: int,
    cols: int,
    scalar_val: float,
    dtype_str: str = "float32",
    target: str = "a3",
    backend: str = "vpto",
    kernel_kind: str = "vector",
):
    """Return a ``@pto.jit`` KernelHandle for a scalar-tile binary op."""
    tile_op_fn = SCALAR_OPS[op_name][0]
    pto_dtype = getattr(pto, dtype_str)
    fn_name = f"scl_{op_name}_{dtype_str}_{rows}x{cols}_s{str(scalar_val).replace('.', 'p')}"

    def kernel_body(
        A_ptr: pto.ptr(pto_dtype, "gm"),
        C_ptr: pto.ptr(pto_dtype, "gm"),
    ) -> None:
        c0 = pto.const(0)
        c1 = pto.const(1)
        c_rows = pto.const(rows)
        c_cols = pto.const(cols)
        c_elems = pto.const(rows * cols)

        shape = [c1, c1, c1, c_rows, c_cols]
        strides = [c_elems, c_elems, c_elems, c_cols, c1]
        off = [c0, c0, c0, c0, c0]

        a_view = pto.make_tensor_view(A_ptr, shape=shape, strides=strides)
        c_view = pto.make_tensor_view(C_ptr, shape=shape, strides=strides)

        a_part = pto.partition_view(a_view, offsets=off, sizes=shape)
        c_part = pto.partition_view(c_view, offsets=off, sizes=shape)

        a_tile = pto.alloc_tile(shape=[rows, cols], dtype=pto_dtype)
        c_tile = pto.alloc_tile(shape=[rows, cols], dtype=pto_dtype)

        pto.tile.load(a_part, a_tile)
        tile_op_fn(a_tile, scalar_val, c_tile)
        pto.tile.store(c_tile, c_part)

    kernel_body.__name__ = fn_name
    return pto.jit(
        name=fn_name,
        kernel_kind=kernel_kind,
        target=target,
        backend=backend,
    )(kernel_body)


def launch_and_check_scalar(
    *,
    op_name: str,
    kernel_handle,
    ref_fn: Callable,
    shape: tuple[int, int],
    scalar_val: float,
    dtype_str: str,
    torch,
    rtol: float = 1e-6,
    atol: float = 1e-6,
    seed: int = 42,
):
    """Compile, launch, and numerical-check one scalar-tile kernel."""
    torch_dt = _torch_dtype(torch, dtype_str)

    x = make_input(shape, torch_dt, torch, seed=seed)
    z = torch.empty(shape, dtype=torch_dt, device="npu:0")
    ref = ref_fn(x.cpu().numpy(), scalar_val)
    stream = _npu_stream(torch)

    t0 = time.perf_counter()
    compiled = kernel_handle.compile()
    compile_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    compiled[1, stream](x.data_ptr(), z.data_ptr())
    torch.npu.synchronize()
    launch_s = time.perf_counter() - t0

    actual = z.cpu().numpy()
    np.testing.assert_allclose(actual, ref, rtol=rtol, atol=atol)
    return compile_s, launch_s


# ---------------------------------------------------------------------------
# Gather e2e helpers
# ---------------------------------------------------------------------------

def make_gatherb_kernel(
    rows: int,
    cols: int,
    dtype_str: str = "float32",
    target: str = "a3",
    backend: str = "vpto",
    kernel_kind: str = "vector",
):
    """Return a ``@pto.jit`` KernelHandle for pto.tile.gatherb (block gather).

    Kernel signature: (Src_ptr f32 GM, Off_ptr i32 GM, Dst_ptr f32 GM).
    Loads src and compact 32B block addresses into UB tiles, calls gatherb,
    and stores the gathered blocks.
    """
    pto_dtype = getattr(pto, dtype_str)
    off_dtype = pto.i32
    fn_name = f"gatherb_{dtype_str}_{rows}x{cols}"

    def kernel_body(
        Src_ptr: pto.ptr(pto_dtype, "gm"),
        Off_ptr: pto.ptr(off_dtype, "gm"),
        Dst_ptr: pto.ptr(pto_dtype, "gm"),
    ) -> None:
        c0 = pto.const(0)
        c1 = pto.const(1)
        c_rows = pto.const(rows)
        c_cols = pto.const(cols)
        c_elems = pto.const(rows * cols)
        block_elems = 8 if dtype_str == "float32" else 16
        block_cols = (cols + block_elems - 1) // block_elems
        block_cols = ((block_cols + 7) // 8) * 8
        block_total = rows * block_cols
        c_block_cols = pto.const(block_cols)
        c_block_total = pto.const(block_total)

        shape = [c1, c1, c1, c_rows, c_cols]
        strides = [c_elems, c_elems, c_elems, c_cols, c1]
        off_shape = [c1, c1, c1, c_rows, c_block_cols]
        off_strides = [c_block_total, c_block_total, c_block_total,
                       c_block_cols, c1]
        off = [c0, c0, c0, c0, c0]

        src_view = pto.make_tensor_view(Src_ptr, shape=shape, strides=strides)
        off_view = pto.make_tensor_view(Off_ptr, shape=off_shape,
                                        strides=off_strides)
        dst_view = pto.make_tensor_view(Dst_ptr, shape=shape, strides=strides)

        src_part = pto.partition_view(src_view, offsets=off, sizes=shape)
        off_part = pto.partition_view(off_view, offsets=off, sizes=off_shape)
        dst_part = pto.partition_view(dst_view, offsets=off, sizes=shape)

        src_tile = pto.alloc_tile(shape=[rows, cols], dtype=pto_dtype)
        off_tile = pto.alloc_tile(shape=[rows, block_cols], dtype=off_dtype)
        dst_tile = pto.alloc_tile(shape=[rows, cols], dtype=pto_dtype)

        pto.tile.load(src_part, src_tile)
        pto.tile.load(off_part, off_tile)
        pto.tile.gatherb(src_tile, off_tile, dst_tile)
        pto.tile.store(dst_tile, dst_part)

    kernel_body.__name__ = fn_name
    return pto.jit(
        name=fn_name,
        kernel_kind=kernel_kind,
        target=target,
        backend=backend,
    )(kernel_body)


def launch_and_check_gather(
    *,
    rows: int,
    cols: int,
    dtype_str: str,
    torch,
    kernel_handle,
    seed: int = 42,
    rtol: float = 1e-6,
    atol: float = 1e-6,
):
    """Compile, launch, and numerical-check a gatherb kernel.

    ``vgatherb`` is a 32B block gather: each i32 offset entry selects one
    32B source block, and the instruction copies that whole block to dst.
    """
    rng = np.random.RandomState(seed)
    total = rows * cols
    elem_size = 4 if dtype_str == "float32" else 2
    block_elems = 32 // elem_size
    valid_block_cols = (cols + block_elems - 1) // block_elems
    block_cols = ((valid_block_cols + 7) // 8) * 8
    valid_blocks = rows * valid_block_cols
    total_blocks = rows * block_cols
    np_dtype = np.float32 if dtype_str == "float32" else np.float16

    # Random source data (small integers for exact results).
    src_flat = rng.randint(1, 10, size=(total,)).astype(np_dtype)
    # Random valid 32B-aligned byte offsets, one per output 32B block.
    valid_src_blocks = valid_blocks
    block_idx = rng.randint(0, valid_src_blocks, size=(total_blocks,))
    off_flat = (block_idx * 32).astype(np.int32)

    golden_flat = np.empty((total,), dtype=np_dtype)
    for row in range(rows):
        for col_block in range(valid_block_cols):
            out_block = row * valid_block_cols + col_block
            src_block = block_idx[row * block_cols + col_block]
            out_start = out_block * block_elems
            src_start = src_block * block_elems
            count = min(block_elems, total - out_start, total - src_start)
            golden_flat[out_start:out_start + count] = src_flat[src_start:src_start + count]
    golden = golden_flat.reshape(rows, cols)

    torch_dt = _torch_dtype(torch, dtype_str)
    src_dev = torch.from_numpy(src_flat).to(device="npu:0", dtype=torch_dt)
    off_dev = torch.from_numpy(off_flat).to(device="npu:0", dtype=torch.int32)
    dst_dev = torch.empty(total, dtype=torch_dt, device="npu:0")
    stream = _npu_stream(torch)

    t0 = time.perf_counter()
    compiled = kernel_handle.compile()
    compile_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    compiled[1, stream](src_dev.data_ptr(), off_dev.data_ptr(), dst_dev.data_ptr())
    torch.npu.synchronize()
    launch_s = time.perf_counter() - t0

    actual = dst_dev.cpu().numpy().reshape(rows, cols)
    np.testing.assert_allclose(actual, golden, rtol=rtol, atol=atol)
    return compile_s, launch_s


def make_gather_index_kernel(
    rows: int,
    cols: int,
    dtype_str: str = "float32",
    target: str = "a3",
    backend: str = "vpto",
    kernel_kind: str = "vector",
):
    """Return a ``@pto.jit`` KernelHandle for scalar index ``pto.tile.gather``."""
    pto_dtype = getattr(pto, dtype_str)
    idx_dtype = pto.i32
    fn_name = f"gather_index_{dtype_str}_{rows}x{cols}"

    def kernel_body(
        Src_ptr: pto.ptr(pto_dtype, "gm"),
        Idx_ptr: pto.ptr(idx_dtype, "gm"),
        Dst_ptr: pto.ptr(pto_dtype, "gm"),
    ) -> None:
        c0 = pto.const(0)
        c1 = pto.const(1)
        c_rows = pto.const(rows)
        c_cols = pto.const(cols)
        c_elems = pto.const(rows * cols)

        shape = [c1, c1, c1, c_rows, c_cols]
        strides = [c_elems, c_elems, c_elems, c_cols, c1]
        off = [c0, c0, c0, c0, c0]

        src_view = pto.make_tensor_view(Src_ptr, shape=shape, strides=strides)
        idx_view = pto.make_tensor_view(Idx_ptr, shape=shape, strides=strides)
        dst_view = pto.make_tensor_view(Dst_ptr, shape=shape, strides=strides)

        src_part = pto.partition_view(src_view, offsets=off, sizes=shape)
        idx_part = pto.partition_view(idx_view, offsets=off, sizes=shape)
        dst_part = pto.partition_view(dst_view, offsets=off, sizes=shape)

        src_tile = pto.alloc_tile(shape=[rows, cols], dtype=pto_dtype)
        idx_tile = pto.alloc_tile(shape=[rows, cols], dtype=idx_dtype)
        tmp_tile = pto.alloc_tile(shape=[rows, cols], dtype=idx_dtype)
        dst_tile = pto.alloc_tile(shape=[rows, cols], dtype=pto_dtype)

        pto.tile.load(src_part, src_tile)
        pto.tile.load(idx_part, idx_tile)
        pto.tile.gather(src_tile, dst_tile, indices=idx_tile, tmp=tmp_tile)
        pto.tile.store(dst_tile, dst_part)

    kernel_body.__name__ = fn_name
    return pto.jit(
        name=fn_name,
        kernel_kind=kernel_kind,
        target=target,
        backend=backend,
    )(kernel_body)


def launch_and_check_gather_index(
    *,
    rows: int,
    cols: int,
    dtype_str: str,
    torch,
    kernel_handle,
    seed: int = 42,
    rtol: float = 1e-6,
    atol: float = 1e-6,
):
    """Compile, launch, and numerical-check scalar index ``pto.tile.gather``."""
    rng = np.random.RandomState(seed)
    total = rows * cols
    np_dtype = np.float32 if dtype_str == "float32" else np.float16

    src_flat = rng.randint(1, 10, size=(total,)).astype(np_dtype)
    idx_flat = rng.randint(0, total, size=(total,)).astype(np.int32)
    golden = src_flat[idx_flat].reshape(rows, cols)

    torch_dt = _torch_dtype(torch, dtype_str)
    src_dev = torch.from_numpy(src_flat).to(device="npu:0", dtype=torch_dt)
    idx_dev = torch.from_numpy(idx_flat).to(device="npu:0", dtype=torch.int32)
    dst_dev = torch.empty(total, dtype=torch_dt, device="npu:0")
    stream = _npu_stream(torch)

    t0 = time.perf_counter()
    compiled = kernel_handle.compile()
    compile_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    compiled[1, stream](src_dev.data_ptr(), idx_dev.data_ptr(), dst_dev.data_ptr())
    torch.npu.synchronize()
    launch_s = time.perf_counter() - t0

    actual = dst_dev.cpu().numpy().reshape(rows, cols)
    np.testing.assert_allclose(actual, golden, rtol=rtol, atol=atol)
    return compile_s, launch_s


def make_mgather_kernel(
    *,
    src_rows: int,
    src_cols: int,
    dst_rows: int,
    dst_cols: int,
    coalesce: str,
    gather_oob: str,
    dtype_str: str = "float32",
    target: str = "a3",
    backend: str = "vpto",
    kernel_kind: str = "vector",
):
    """Return a ``@pto.jit`` KernelHandle for GM-to-UB ``pto.tile.mgather``."""
    pto_dtype = getattr(pto, dtype_str)
    idx_dtype = pto.i32
    idx_rows = 1 if coalesce == "row" else dst_rows
    idx_cols = dst_rows if coalesce == "row" else dst_cols
    idx_physical_cols = ((idx_cols + 7) // 8) * 8
    fn_name = (
        f"mgather_{coalesce}_{gather_oob}_{dtype_str}_"
        f"{src_rows}x{src_cols}_{dst_rows}x{dst_cols}"
    )

    def kernel_body(
        Src_ptr: pto.ptr(pto_dtype, "gm"),
        Idx_ptr: pto.ptr(idx_dtype, "gm"),
        Dst_ptr: pto.ptr(pto_dtype, "gm"),
    ) -> None:
        c0 = pto.const(0)
        c1 = pto.const(1)
        c_src_rows = pto.const(src_rows)
        c_src_cols = pto.const(src_cols)
        c_src_elems = pto.const(src_rows * src_cols)
        c_idx_rows = pto.const(idx_rows)
        c_idx_cols = pto.const(idx_physical_cols)
        c_idx_elems = pto.const(idx_rows * idx_physical_cols)
        c_dst_rows = pto.const(dst_rows)
        c_dst_cols = pto.const(dst_cols)
        c_dst_elems = pto.const(dst_rows * dst_cols)

        src_shape = [c1, c1, c1, c_src_rows, c_src_cols]
        src_strides = [c_src_elems, c_src_elems, c_src_elems, c_src_cols, c1]
        idx_shape = [c1, c1, c1, c_idx_rows, c_idx_cols]
        idx_strides = [c_idx_elems, c_idx_elems, c_idx_elems, c_idx_cols, c1]
        dst_shape = [c1, c1, c1, c_dst_rows, c_dst_cols]
        dst_strides = [c_dst_elems, c_dst_elems, c_dst_elems, c_dst_cols, c1]
        off = [c0, c0, c0, c0, c0]

        src_view = pto.make_tensor_view(Src_ptr, shape=src_shape, strides=src_strides)
        idx_view = pto.make_tensor_view(Idx_ptr, shape=idx_shape, strides=idx_strides)
        dst_view = pto.make_tensor_view(Dst_ptr, shape=dst_shape, strides=dst_strides)
        src_part = pto.partition_view(src_view, offsets=off, sizes=src_shape)
        idx_part = pto.partition_view(idx_view, offsets=off, sizes=idx_shape)
        dst_part = pto.partition_view(dst_view, offsets=off, sizes=dst_shape)

        idx_tile = pto.alloc_tile(
            shape=[idx_rows, idx_physical_cols],
            valid_shape=[idx_rows, idx_cols],
            dtype=idx_dtype,
        )
        dst_tile = pto.alloc_tile(shape=[dst_rows, dst_cols], dtype=pto_dtype)
        pto.tile.load(idx_part, idx_tile)
        pto.tile.mgather(
            src_part,
            idx_tile,
            dst_tile,
            coalesce,
            gather_oob=gather_oob,
        )
        pto.tile.store(dst_tile, dst_part)

    kernel_body.__name__ = fn_name
    return pto.jit(
        name=fn_name,
        kernel_kind=kernel_kind,
        target=target,
        backend=backend,
    )(kernel_body)


def launch_and_check_mgather(
    *,
    kernel_handle,
    src_rows: int,
    src_cols: int,
    dst_rows: int,
    dst_cols: int,
    coalesce: str,
    gather_oob: str,
    dtype_str: str,
    torch,
    seed: int = 42,
):
    """Compile, launch, and numerical-check one GM-to-UB mgather kernel."""
    rng = np.random.RandomState(seed)
    np_dtype = np.float32 if dtype_str == "float32" else np.float16
    src = rng.randint(1, 10, size=(src_rows, src_cols)).astype(np_dtype)
    idx_rows = 1 if coalesce == "row" else dst_rows
    idx_cols = dst_rows if coalesce == "row" else dst_cols
    idx_physical_cols = ((idx_cols + 7) // 8) * 8
    idx_count = idx_rows * idx_cols
    limit = src_rows if coalesce == "row" else src_rows * src_cols
    if gather_oob == "undefined":
        idx = rng.randint(0, limit, size=(idx_count,), dtype=np.int32)
    else:
        pattern = np.array([-2, -1, 0, limit - 1, limit, limit + 1], dtype=np.int32)
        idx = np.resize(pattern, idx_count)

    if gather_oob == "clamp":
        normalized = np.clip(idx, 0, limit - 1)
    elif gather_oob == "wrap":
        normalized = idx % limit
    else:
        normalized = idx

    if coalesce == "row":
        golden = np.zeros((dst_rows, dst_cols), dtype=np_dtype)
        valid = (idx >= 0) & (idx < limit)
        if gather_oob in {"clamp", "wrap"}:
            valid[:] = True
        golden[valid] = src[normalized[valid], :dst_cols]
    else:
        src_flat = src.reshape(-1)
        golden_flat = np.zeros((idx_count,), dtype=np_dtype)
        valid = (idx >= 0) & (idx < limit)
        if gather_oob in {"clamp", "wrap"}:
            valid[:] = True
        golden_flat[valid] = src_flat[normalized[valid]]
        golden = golden_flat.reshape(dst_rows, dst_cols)

    torch_dt = _torch_dtype(torch, dtype_str)
    idx_storage = np.zeros((idx_rows, idx_physical_cols), dtype=np.int32)
    if coalesce == "row":
        idx_storage[0, :idx_cols] = idx
    else:
        idx_storage[:, :dst_cols] = idx.reshape(dst_rows, dst_cols)
    src_dev = torch.from_numpy(src).to(device="npu:0", dtype=torch_dt)
    idx_dev = torch.from_numpy(idx_storage).to(device="npu:0", dtype=torch.int32)
    dst_dev = torch.empty((dst_rows, dst_cols), dtype=torch_dt, device="npu:0")
    stream = _npu_stream(torch)

    t0 = time.perf_counter()
    compiled = kernel_handle.compile()
    compile_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    compiled[1, stream](src_dev.data_ptr(), idx_dev.data_ptr(), dst_dev.data_ptr())
    torch.npu.synchronize()
    launch_s = time.perf_counter() - t0

    np.testing.assert_array_equal(dst_dev.cpu().numpy(), golden)
    return compile_s, launch_s
