# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Compiler regressions for A5 expand dtype coverage and legal VPTO lowering."""

import re
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

from ptoas import _cli
from ptodsl import pto
from ptodsl._runtime.toolchain import resolve_ptoas_binary


def _binary_kernel(op, dtype, row):
    @pto.jit(name=f"expand_{op}_{dtype}", target="a5")
    def kernel(a: pto.ptr(dtype, "gm"), b: pto.ptr(dtype, "gm"), out: pto.ptr(dtype, "gm")):
        src0_view = pto.make_tensor_view(a, shape=[8, 8], strides=[8, 1])
        src1_view = pto.make_tensor_view(b, shape=[8, 1] if row else [1, 8],
                                         strides=[1, 1] if row else [8, 1])
        dst_view = pto.make_tensor_view(out, shape=[8, 8], strides=[8, 1])
        cols = 32 if dtype in (pto.i8, pto.ui8) else 16 if dtype in (pto.i16, pto.ui16, pto.bf16) else 8
        src0 = pto.alloc_tile(shape=[8, cols], dtype=dtype, valid_shape=[8, 8], blayout="RowMajor")
        src1 = pto.alloc_tile(shape=[32, 1] if row else [8, cols], dtype=dtype,
                              valid_shape=[8, 1] if row else [1, 8],
                              blayout="ColMajor" if row else "RowMajor")
        dst = pto.alloc_tile(shape=[8, cols], dtype=dtype, valid_shape=[8, 8], blayout="RowMajor")
        pto.tile.load(src0_view, src0)
        pto.tile.load(src1_view, src1)
        getattr(pto.tile, op)(src0, src1, dst)
        pto.tile.store(dst, dst_view)
    return kernel


def _unary_kernel(op, dtype, row=True):
    @pto.jit(name=f"expand_{op}_{dtype}", target="a5")
    def kernel(a: pto.ptr(dtype, "gm"), out: pto.ptr(dtype, "gm")):
        src_view = pto.make_tensor_view(a, shape=[8, 1] if row else [1, 8],
                                        strides=[1, 1] if row else [8, 1])
        dst_view = pto.make_tensor_view(out, shape=[8, 8], strides=[8, 1])
        cols = 32 if dtype == pto.ui8 else 16 if dtype == pto.ui16 else 8
        src = pto.alloc_tile(shape=[8, cols], dtype=dtype,
                             valid_shape=[8, 1] if row else [1, 8],
                             blayout="RowMajor")
        dst = pto.alloc_tile(shape=[8, cols], dtype=dtype, valid_shape=[8, 8], blayout="RowMajor")
        pto.tile.load(src_view, src)
        getattr(pto.tile, op)(src, dst)
        pto.tile.store(dst, dst_view)
    return kernel


def _expands_kernel(dtype):
    @pto.jit(name=f"expand_scalar_{dtype}", target="a5")
    def kernel(out: pto.ptr(dtype, "gm")):
        dst_view = pto.make_tensor_view(out, shape=[8, 8], strides=[8, 1])
        cols = 32 if dtype == pto.ui8 else 16 if dtype == pto.ui16 else 8
        dst = pto.alloc_tile(shape=[8, cols], dtype=dtype, valid_shape=[8, 8], blayout="RowMajor")
        pto.tile.expands(dtype(3), dst)
        pto.tile.store(dst, dst_view)
    return kernel


def _compile(kernel, tmp_path, *, llvm_ir=False):
    source = tmp_path / "input.pto"
    output = tmp_path / ("output.ll" if llvm_ir else "output.vpto")
    source.write_text(kernel.compile().mlir_text(), encoding="utf-8")
    result = _cli.launch(
        ["--pto-arch=a5", "--pto-backend=vpto",
         "--emit-vpto-llvm-ir" if llvm_ir else "--emit-vpto", str(source), "-o", str(output)],
        wrapper=resolve_ptoas_binary(),
    )
    assert result == 0
    return output.read_text(encoding="utf-8")


@pytest.mark.parametrize("op,row", [
    ("rowexpandadd", True), ("rowexpandsub", True), ("rowexpandmul", True),
    ("rowexpandmax", True), ("rowexpandmin", True),
    ("colexpandadd", False), ("colexpandsub", False), ("colexpandmul", False),
    ("colexpandmax", False), ("colexpandmin", False),
])
def test_bf16_binary_expand_compiles(op, row, tmp_path):
    _compile(_binary_kernel(op, pto.bf16, row), tmp_path)


@pytest.mark.parametrize("dtype", [pto.ui8, pto.ui16, pto.ui32])
def test_unsigned_expands_compiles(dtype, tmp_path):
    _compile(_expands_kernel(dtype), tmp_path)


@pytest.mark.parametrize("dtype", [pto.ui8, pto.ui16, pto.ui32])
@pytest.mark.parametrize("op,row", [("rowexpand", True), ("colexpand", False)])
def test_unsigned_unary_expand_compiles(dtype, op, row, tmp_path):
    _compile(_unary_kernel(op, dtype, row), tmp_path)


@pytest.mark.parametrize("dtype", [pto.ui8, pto.ui16, pto.ui32])
def test_unsigned_rowexpandadd_compiles(dtype, tmp_path):
    _compile(_binary_kernel("rowexpandadd", dtype, True), tmp_path)


@pytest.mark.parametrize("dtype", [pto.ui8, pto.ui16, pto.ui32])
@pytest.mark.parametrize("op,row", [
    ("rowexpandsub", True), ("rowexpandmul", True), ("rowexpanddiv", True),
    ("rowexpandmax", True), ("rowexpandmin", True),
    ("colexpandadd", False), ("colexpandsub", False), ("colexpandmul", False),
    ("colexpanddiv", False), ("colexpandmax", False), ("colexpandmin", False),
])
def test_unsigned_binary_expand_compiles(dtype, op, row, tmp_path):
    _compile(_binary_kernel(op, dtype, row), tmp_path)


@pytest.mark.parametrize("op,row", [("rowexpanddiv", True), ("colexpanddiv", False)])
def test_bf16_div_expand_compiles(op, row, tmp_path):
    _compile(_binary_kernel(op, pto.bf16, row), tmp_path)


@pytest.mark.parametrize("op,row", [("rowexpandmul", True), ("colexpandmul", False)])
def test_i8_multiply_uses_legal_vector_operations(op, row, tmp_path):
    llvm_ir = _compile(_binary_kernel(op, pto.i8, row), tmp_path, llvm_ir=True)
    assert "llvm.hivm.vmul.u.x.v128i16" in llvm_ir
    assert "llvm.hivm.vmul.s.x.v256i8" not in llvm_ir
    if row:
        # Packed INT8 row values are loaded with a one-byte post-update and
        # broadcast from the vector register.
        assert re.search(
            r"call .* @llvm\.hivm\.vldsx1\.post\.v256i8\(ptr addrspace\(6\) [^,]+, i32 1,",
            llvm_ir,
        )
        assert "call <256 x i8> @llvm.hivm.vdup.z.v256i8(" in llvm_ir


def test_rowexpand_wide_smoke_compiles(tmp_path):
    source = (Path(__file__).resolve().parents[2] / "test/tilelang_st/npu/a5/src/st/smoke"
              / "testcase/trowexpand/trowexpand.pto")
    output = tmp_path / "rowexpand.ll"
    result = _cli.launch(
        ["--pto-arch=a5", "--pto-backend=vpto", "--enable-insert-sync",
         "--enable-tile-op-expand", "--emit-vpto-llvm-ir",
         str(source), "-o", str(output)],
        wrapper=resolve_ptoas_binary(),
    )
    assert result == 0
    llvm_ir = output.read_text(encoding="utf-8")
    assert "call <64 x float> @llvm.hivm.vdup.z.v64f32(" in llvm_ir
    assert "call void @llvm.hivm.vstsx1.v64f32(" in llvm_ir


@pytest.mark.parametrize("module_name,kernel_name", [
    ("expandmul_i8", "rowexpandmul_i8"),
    ("expandmul_i8", "colexpandmul_i8"),
    ("expand_unsigned", "expands_ui8"),
    ("expand_unsigned", "rowexpandadd_ui16"),
    ("expand_unsigned", "colexpanddiv_ui32"),
    ("expanddiv_bf16", "rowexpanddiv_bf16"),
])
def test_expand_st_kernel_uses_auto_allocation(module_name, kernel_name, tmp_path, monkeypatch):
    dsl_root = Path(__file__).resolve().parents[2] / "test/dsl-st"
    monkeypatch.syspath_prepend(str(dsl_root))
    spec = spec_from_file_location(f"st_{module_name}", dsl_root / f"{module_name}.py")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    kernel = getattr(module, kernel_name)
    assert kernel.compile().mode == "auto"
    _compile(kernel, tmp_path)
