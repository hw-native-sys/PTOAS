#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Focused tracing coverage for the public ``@pto.tileop`` surface."""

from ptodsl import pto
from ptodsl._context import make_context
from ptoas.mlir.ir import Module


@pto.simt(max_threads=64, ast_rewrite=False)
def tileop_simt_epilogue(dst: pto.ptr(pto.f32, "vec"), columns: pto.i32):
    pass


@pto.tileop(ast_rewrite=False)
def tileop_simt_stage(dst: pto.Tile, columns: pto.i32):
    tileop_simt_epilogue[64, 1, 1](dst.as_ptr(), columns)


@pto.jit(target="a5", ast_rewrite=False)
def tileop_entry(*, TRACE_TOKEN: pto.const_expr = 0):
    dst = pto.alloc_tile(shape=[1, 16], dtype=pto.f32, valid_shape=[1, 16])
    columns = pto.const(16, dtype=pto.i32)
    tileop_simt_stage(dst, columns)
    tileop_simt_stage(dst, columns)


@pto.jit(target="a5", mode="explicit", kernel_kind="vector", ast_rewrite=False)
def explicit_named_tileop_entry(*, TRACE_TOKEN: pto.const_expr = 0):
    dst = pto.alloc_tile(shape=[1, 16], dtype=pto.f32, valid_shape=[1, 16])
    columns = pto.const(16, dtype=pto.i32)
    tileop_simt_stage(dst, columns)


@pto.jit(target="a5", ast_rewrite=False)
def inline_tileop_entry(*, TRACE_TOKEN: pto.const_expr = 0):
    dst = pto.alloc_tile(shape=[1, 16], dtype=pto.f32, valid_shape=[1, 16])
    columns = pto.const(16, dtype=pto.i32)
    with pto.tileop():
        mask, _ = pto.make_mask(pto.f32, columns)
        value = pto.vlds(dst[0, 0:])
        pto.vsts(value, dst[0, 0:], mask)


@pto.jit(target="a5", ast_rewrite=False)
def invalid_inline_tileop_pointer_capture_entry(
    src: pto.ptr(pto.f32, "gm"),
    *,
    TRACE_TOKEN: pto.const_expr = 0,
):
    with pto.tileop():
        pto.addptr(src, 1)


@pto.tileop(ast_rewrite=False)
def tileop_direct_simt_call(dst: pto.Tile):
    tileop_simt_epilogue(dst.as_ptr())


@pto.jit(target="a5", ast_rewrite=False)
def invalid_direct_simt_entry(*, TRACE_TOKEN: pto.const_expr = 0):
    dst = pto.alloc_tile(shape=[1, 16], dtype=pto.f32, valid_shape=[1, 16])
    tileop_direct_simt_call(dst)


@pto.jit(target="a5", ast_rewrite=False)
def invalid_scalar_argument_entry(*, TRACE_TOKEN: pto.const_expr = 0):
    dst = pto.alloc_tile(shape=[1, 16], dtype=pto.f32, valid_shape=[1, 16])
    wrong_type = pto.const(16.0, dtype=pto.f32)
    tileop_simt_stage(dst, wrong_type)


@pto.jit(
    target="a5",
    entry=False,
    mode="explicit",
    kernel_kind="vector",
    ast_rewrite=False,
)
def tileop_forbidden_kernel_module(dst: pto.Tile):
    pass


@pto.tileop(ast_rewrite=False)
def tileop_kernel_module_call(dst: pto.Tile):
    tileop_forbidden_kernel_module(dst)


@pto.jit(target="a5", ast_rewrite=False)
def invalid_kernel_module_call_entry(*, TRACE_TOKEN: pto.const_expr = 0):
    dst = pto.alloc_tile(shape=[1, 16], dtype=pto.f32, valid_shape=[1, 16])
    tileop_kernel_module_call(dst)


def define_tileop_result_annotation():
    @pto.tileop(ast_rewrite=False)
    def tileop_result_annotation(dst: pto.Tile) -> pto.i32:
        tileop_simt_epilogue[64, 1, 1](dst.as_ptr(), pto.const(1, dtype=pto.i32))

    return tileop_result_annotation


def define_postponed_none_tileop():
    namespace = {"pto": pto}
    exec(
        "from __future__ import annotations\n"
        "@pto.tileop(ast_rewrite=False)\n"
        "def postponed_none_tileop(dst: pto.Tile) -> None:\n"
        "    pass\n",
        namespace,
    )
    return namespace["postponed_none_tileop"]


@pto.tileop(ast_rewrite=False)
def tileop_return_value(dst: pto.Tile, columns: pto.i32):
    tileop_simt_epilogue[64, 1, 1](dst.as_ptr(), columns)
    return columns


@pto.jit(target="a5", ast_rewrite=False)
def invalid_scalar_result_entry(*, TRACE_TOKEN: pto.const_expr = 0):
    dst = pto.alloc_tile(shape=[1, 16], dtype=pto.f32, valid_shape=[1, 16])
    columns = pto.const(16, dtype=pto.i32)
    tileop_return_value(dst, columns)


def main() -> None:
    text = tileop_entry.compile(TRACE_TOKEN=1).mlir_text()
    assert "pto.tileop.helper" in text
    assert "pto.simt_entry" in text
    assert "pto.simt_launch @tileop_simt_epilogue__simt_" in text
    assert text.count("call @tileop_simt_stage__ptodsl_") == 2
    assert "pto.section.vector" not in text
    with make_context() as context:
        module = Module.parse(text, context)
        module.operation.verify()

    explicit_named_text = explicit_named_tileop_entry.compile(TRACE_TOKEN=1).mlir_text()
    assert "pto.kernel_kind = #pto.kernel_kind<vector>" in explicit_named_text
    assert "pto.tileop.helper" in explicit_named_text
    assert "call @tileop_simt_stage__ptodsl_" in explicit_named_text
    with make_context() as context:
        module = Module.parse(explicit_named_text, context)
        module.operation.verify()

    inline_text = inline_tileop_entry.compile(TRACE_TOKEN=1).mlir_text()
    assert "func.func private @inline_tileop_0" in inline_text
    assert "pto.tileop.helper" in inline_text
    assert "call @inline_tileop_0" in inline_text
    assert "!pto.tile_buf<" in inline_text
    with make_context() as context:
        module = Module.parse(inline_text, context)
        module.operation.verify()

    try:
        invalid_inline_tileop_pointer_capture_entry.compile(TRACE_TOKEN=1)
    except TypeError as exc:
        message = str(exc)
        assert "with pto.tileop(): captured boundary value #1" in message
        assert "!pto.ptr<" in message
        assert "only pto.Tile and PTO scalar values" in message
        assert "matching the @pto.tileop parameter ABI" in message
    else:
        raise AssertionError("inline TileOp pointer captures must be rejected")

    try:
        invalid_direct_simt_entry.compile(TRACE_TOKEN=1)
    except RuntimeError as exc:
        message = str(exc)
        assert "only invoke @pto.simt through an explicit launch" in message
        assert "helper[dim_x, dim_y, dim_z](...)" in message
    else:
        raise AssertionError("direct @pto.simt call inside @pto.tileop must be rejected")

    try:
        invalid_scalar_argument_entry.compile(TRACE_TOKEN=1)
    except TypeError as exc:
        message = str(exc)
        assert "Expected a PTO scalar of MLIR type i32, got 'f32'" in message
    else:
        raise AssertionError("mismatched @pto.tileop scalar argument must be rejected")

    try:
        invalid_kernel_module_call_entry.compile(TRACE_TOKEN=1)
    except RuntimeError as exc:
        message = str(exc)
        assert "cannot call @pto.jit(entry=False)" in message
        assert "enclosing @pto.jit kernel" in message
    else:
        raise AssertionError("@pto.tileop calls to @pto.jit(entry=False) must be rejected")

    try:
        define_tileop_result_annotation()
    except TypeError as exc:
        message = str(exc)
        assert "@pto.tileop helpers must return None" in message
        assert "mutable Tile parameters" in message
    else:
        raise AssertionError("@pto.tileop result annotations must be rejected")

    postponed_none_tileop = define_postponed_none_tileop()
    assert postponed_none_tileop.spec.role.value == "tileop"

    try:
        invalid_scalar_result_entry.compile(TRACE_TOKEN=1)
    except TypeError as exc:
        message = str(exc)
        assert "@pto.tileop helpers must return None" in message
        assert "mutable Tile parameters" in message
    else:
        raise AssertionError("@pto.tileop return values must be rejected")
    print("ptodsl_tileop: PASS")


if __name__ == "__main__":
    main()
