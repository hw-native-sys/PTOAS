# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Issue #1604 repro kernels (PTODSL tile transpose on VPTO/A5).

Kernels are copied verbatim from the issue report; the only adjustment is the
decorator spelling (``@pto.jit`` instead of the issue's ``@ptodsl.jit``,
which does not exist in this package).

Problem 1 (i8/ui8, rows >= cols): simulation output interleaves correct bytes
with zeros.  Representative case: transpose_i8_s32x32 (src [32, 32]).
Problem 2 (b8/b16, rows < cols): no template matches (NoMatchingTemplate).
Representative cases: transpose_f16_s16x32 (f16 [16, 32]) and
transpose_i8_s32x64 (i8 [32, 64]).

Running this file as a script dumps one ``.pto`` per kernel into
``test/dsl/`` for the ptoas repro command.
"""

from pathlib import Path

from ptodsl import pto


@pto.jit(target="a5")
def transpose_i8_s32x32(
    src_ptr: pto.ptr(pto.i8, "gm"),
    out_ptr: pto.ptr(pto.i8, "gm"),
):
    src_view = pto.make_tensor_view(src_ptr, shape=[32, 32], strides=[32, 1])
    dst_view = pto.make_tensor_view(out_ptr, shape=[32, 32], strides=[32, 1])
    src = pto.alloc_tile(shape=[32, 32], dtype=pto.i8, valid_shape=[32, 32], blayout="RowMajor")
    tmp = pto.alloc_tile(shape=[32, 32], dtype=pto.i8, valid_shape=[32, 32], blayout="RowMajor")
    dst = pto.alloc_tile(shape=[32, 32], dtype=pto.i8, valid_shape=[32, 32], blayout="RowMajor")
    pto.tile.load(src_view, src)
    pto.tile.transpose(src, tmp, dst)
    pto.tile.store(dst, dst_view)


@pto.jit(target="a5")
def transpose_ui8_s32x32(
    src_ptr: pto.ptr(pto.ui8, "gm"),
    out_ptr: pto.ptr(pto.ui8, "gm"),
):
    src_view = pto.make_tensor_view(src_ptr, shape=[32, 32], strides=[32, 1])
    dst_view = pto.make_tensor_view(out_ptr, shape=[32, 32], strides=[32, 1])
    src = pto.alloc_tile(shape=[32, 32], dtype=pto.ui8, valid_shape=[32, 32], blayout="RowMajor")
    tmp = pto.alloc_tile(shape=[32, 32], dtype=pto.ui8, valid_shape=[32, 32], blayout="RowMajor")
    dst = pto.alloc_tile(shape=[32, 32], dtype=pto.ui8, valid_shape=[32, 32], blayout="RowMajor")
    pto.tile.load(src_view, src)
    pto.tile.transpose(src, tmp, dst)
    pto.tile.store(dst, dst_view)


@pto.jit(target="a5")
def transpose_f16_s16x32(
    src_ptr: pto.ptr(pto.f16, "gm"),
    out_ptr: pto.ptr(pto.f16, "gm"),
):
    src_view = pto.make_tensor_view(src_ptr, shape=[16, 32], strides=[32, 1])
    dst_view = pto.make_tensor_view(out_ptr, shape=[32, 16], strides=[16, 1])
    src = pto.alloc_tile(shape=[16, 32], dtype=pto.f16, valid_shape=[16, 32], blayout="RowMajor")
    tmp = pto.alloc_tile(shape=[16, 32], dtype=pto.f16, valid_shape=[16, 32], blayout="RowMajor")
    dst = pto.alloc_tile(shape=[32, 16], dtype=pto.f16, valid_shape=[32, 16], blayout="RowMajor")
    pto.tile.load(src_view, src)
    pto.tile.transpose(src, tmp, dst)
    pto.tile.store(dst, dst_view)


@pto.jit(target="a5")
def transpose_i8_s32x64(
    src_ptr: pto.ptr(pto.i8, "gm"),
    out_ptr: pto.ptr(pto.i8, "gm"),
):
    src_view = pto.make_tensor_view(src_ptr, shape=[32, 64], strides=[64, 1])
    dst_view = pto.make_tensor_view(out_ptr, shape=[64, 32], strides=[32, 1])
    src = pto.alloc_tile(shape=[32, 64], dtype=pto.i8, valid_shape=[32, 64], blayout="RowMajor")
    tmp = pto.alloc_tile(shape=[32, 64], dtype=pto.i8, valid_shape=[32, 64], blayout="RowMajor")
    dst = pto.alloc_tile(shape=[64, 32], dtype=pto.i8, valid_shape=[64, 32], blayout="RowMajor")
    pto.tile.load(src_view, src)
    pto.tile.transpose(src, tmp, dst)
    pto.tile.store(dst, dst_view)


_KERNELS = {
    "issue_1604_tall_i8_s32x32": transpose_i8_s32x32,
    "issue_1604_tall_ui8_s32x32": transpose_ui8_s32x32,
    "issue_1604_wide_f16_s16x32": transpose_f16_s16x32,
    "issue_1604_wide_i8_s32x64": transpose_i8_s32x64,
}


def main():
    out_dir = Path(__file__).resolve().parents[2] / "test" / "dsl"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, kernel in _KERNELS.items():
        path = out_dir / f"{name}.pto"
        path.write_text(str(kernel) + "\n")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
