# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See the License in the root of the software repository for the full text of the License.

"""Issue #1599 repro: PTODSL rowmax / rowmin with i8 / ui8 tiles on A5.

pto-isa A5 TRowMax / TRowMin accept int8_t / uint8_t. The PTODSL entry points
and the shared row-reduction verifier whitelist only allow i16/i32/f16/f32, so
these kernels fail at MLIR verification before the VPTO backend runs.
"""

import ptodsl
from ptodsl import pto


@ptodsl.jit
def rowmax_i8_anchor_uniform(src_ptr: pto.ptr(pto.i8, "gm"), out_ptr: pto.ptr(pto.i8, "gm")):
    src_view = pto.make_tensor_view(src_ptr, shape=[8, 32], strides=[32, 1])
    dst_view = pto.make_tensor_view(out_ptr, shape=[8, 1], strides=[1, 1])
    src = pto.alloc_tile(shape=[8, 32], dtype=pto.i8, valid_shape=[8, 32], blayout="RowMajor")
    dst = pto.alloc_tile(shape=[8, 32], dtype=pto.i8, valid_shape=[8, 1], blayout="RowMajor")
    pto.tile.load(src_view, src)
    pto.tile.rowmax(src, dst)
    pto.tile.store(dst, dst_view)


@ptodsl.jit
def rowmax_ui8_anchor_uniform(src_ptr: pto.ptr(pto.ui8, "gm"), out_ptr: pto.ptr(pto.ui8, "gm")):
    src_view = pto.make_tensor_view(src_ptr, shape=[8, 32], strides=[32, 1])
    dst_view = pto.make_tensor_view(out_ptr, shape=[8, 1], strides=[1, 1])
    src = pto.alloc_tile(shape=[8, 32], dtype=pto.ui8, valid_shape=[8, 32], blayout="RowMajor")
    dst = pto.alloc_tile(shape=[8, 32], dtype=pto.ui8, valid_shape=[8, 1], blayout="RowMajor")
    pto.tile.load(src_view, src)
    pto.tile.rowmax(src, dst)
    pto.tile.store(dst, dst_view)


@ptodsl.jit
def rowmin_i8_anchor_uniform(src_ptr: pto.ptr(pto.i8, "gm"), out_ptr: pto.ptr(pto.i8, "gm")):
    src_view = pto.make_tensor_view(src_ptr, shape=[8, 32], strides=[32, 1])
    dst_view = pto.make_tensor_view(out_ptr, shape=[8, 1], strides=[1, 1])
    src = pto.alloc_tile(shape=[8, 32], dtype=pto.i8, valid_shape=[8, 32], blayout="RowMajor")
    dst = pto.alloc_tile(shape=[8, 32], dtype=pto.i8, valid_shape=[8, 1], blayout="RowMajor")
    pto.tile.load(src_view, src)
    pto.tile.rowmin(src, dst)
    pto.tile.store(dst, dst_view)


@ptodsl.jit
def rowmin_ui8_anchor_uniform(src_ptr: pto.ptr(pto.ui8, "gm"), out_ptr: pto.ptr(pto.ui8, "gm")):
    src_view = pto.make_tensor_view(src_ptr, shape=[8, 32], strides=[32, 1])
    dst_view = pto.make_tensor_view(out_ptr, shape=[8, 1], strides=[1, 1])
    src = pto.alloc_tile(shape=[8, 32], dtype=pto.ui8, valid_shape=[8, 32], blayout="RowMajor")
    dst = pto.alloc_tile(shape=[8, 32], dtype=pto.ui8, valid_shape=[8, 1], blayout="RowMajor")
    pto.tile.load(src_view, src)
    pto.tile.rowmin(src, dst)
    pto.tile.store(dst, dst_view)
