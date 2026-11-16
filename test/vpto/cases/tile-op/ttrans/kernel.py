#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Runtime numeric regression for the ``pto.ttrans`` tile templates (issue #1604).

Shapes cover all four b8/b16 ttrans templates on the VPTO path:

- i8  [32,32] tall: b8 colwise (vgather2 zero-extend + PK_B16 packed store)
- ui8 [32,32] tall: b8 colwise
- f16 [16,32] wide: b16 rowwise (vlds + 16-bit-offset vscatter)
- i8  [32,64] wide: b8 rowwise (UNPK_B8 load + even-byte vscatter)

When a tile holds more than 65536 elements (rows * cols > 2**16, i.e.
the largest linear offset row * stride + col exceeds 65535), the
transpose result must still match the golden (src.T) element by element.

The host side mirrors the other PTODSL VPTO cases (e.g. micro-op/
cube-matmul/mad_mx_e8m0_stage_abi): torch_npu owns device memory and the
stream, so the case shares the runtime exit path the CI simulator image has
already validated.
"""

import zlib

import numpy as np

from ptodsl import pto


_DEVICE = "npu:0"


def _ttrans_body(src_ptr, dst_ptr, *, rows, cols, dtype):
    src_view = pto.make_tensor_view(src_ptr, shape=[rows, cols], strides=[cols, 1])
    dst_view = pto.make_tensor_view(dst_ptr, shape=[cols, rows], strides=[rows, 1])
    src = pto.alloc_tile(shape=[rows, cols], dtype=dtype, valid_shape=[rows, cols], blayout="RowMajor")
    tmp = pto.alloc_tile(shape=[rows, cols], dtype=dtype, valid_shape=[rows, cols], blayout="RowMajor")
    dst = pto.alloc_tile(shape=[cols, rows], dtype=dtype, valid_shape=[cols, rows], blayout="RowMajor")
    pto.tile.load(src_view, src)
    pto.tile.transpose(src, tmp, dst)
    pto.tile.store(dst, dst_view)


def _make_ttrans_kernel(name, dtype, rows, cols):
    @pto.jit(name=name, target="a5")
    def _kernel(
        src_ptr: pto.ptr(dtype, "gm"),
        dst_ptr: pto.ptr(dtype, "gm"),
    ):
        _ttrans_body(src_ptr, dst_ptr, rows=rows, cols=cols, dtype=dtype)

    return _kernel


SHAPES = (
    ("ttrans_i8_tall_32x32", pto.i8, np.int8, 32, 32),
    ("ttrans_ui8_tall_32x32", pto.ui8, np.uint8, 32, 32),
    ("ttrans_f16_wide_16x32", pto.f16, np.float16, 16, 32),
    ("ttrans_i8_wide_32x64", pto.i8, np.int8, 32, 64),
    # issue #1604 reopen: 83968-element b8 tiles exceed the 16-bit offset span.
    ("ttrans_i8_tall_ubmax_2624x32", pto.i8, np.int8, 2624, 32),
    ("ttrans_ui8_tall_ubmax_2624x32", pto.ui8, np.uint8, 2624, 32),
    ("ttrans_i8_wide_ubmax_32x2624", pto.i8, np.int8, 32, 2624),
    ("ttrans_ui8_wide_ubmax_32x2624", pto.ui8, np.uint8, 32, 2624),
    # boundary: exactly 2**16 elements, max offset 65535 of the old single-base form.
    ("ttrans_i8_tall_2048x32", pto.i8, np.int8, 2048, 32),
)

KERNELS = {
    name: _make_ttrans_kernel(name, dtype, rows, cols)
    for name, dtype, _np_dtype, rows, cols in SHAPES
}


def _make_source(name, np_dtype, rows, cols):
    rng = np.random.default_rng(zlib.crc32(name.encode("utf-8")))
    if np_dtype == np.float16:
        return rng.uniform(-4.0, 4.0, size=(rows, cols)).astype(np.float16)
    if np_dtype == np.uint8:
        return rng.integers(0, 256, size=(rows, cols)).astype(np.uint8)
    return rng.integers(-100, 101, size=(rows, cols)).astype(np.int8)


def _init_runtime():
    import torch
    import torch_npu  # noqa: F401

    torch.npu.config.allow_internal_format = False
    torch_npu.npu.set_compile_mode(jit_compile=False)
    torch.npu.set_device(_DEVICE)
    return torch


def main():
    torch = _init_runtime()
    stream = torch.npu.current_stream()._as_parameter_  # noqa: SLF001
    failed = False
    for name, _dtype, np_dtype, rows, cols in SHAPES:
        src = _make_source(name, np_dtype, rows, cols)
        golden = np.ascontiguousarray(src.T)

        src_dev = torch.from_numpy(src).to(_DEVICE)
        dst_dev = torch.from_numpy(np.zeros((cols, rows), dtype=np_dtype)).to(_DEVICE)

        compiled = KERNELS[name].compile()
        compiled[1, stream](src_dev.data_ptr(), dst_dev.data_ptr())
        torch.npu.synchronize()

        out = dst_dev.cpu().numpy()
        mismatched = int((out != golden).sum())
        ok = mismatched == 0
        print(
            f"{'PASS' if ok else 'FAIL'} {name} shape={rows}x{cols} "
            f"mismatched={mismatched}/{out.size}"
        )
        if not ok:
            print(f"  golden row0: {golden[0, :8].tolist()}")
            print(f"  actual row0: {out[0, :8].tolist()}")
            failed = True

    if failed:
        return 1
    print("All ttrans cases passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
