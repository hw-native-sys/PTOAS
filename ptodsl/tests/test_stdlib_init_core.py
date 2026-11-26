#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import operator
import re
import sys
import tempfile
from pathlib import Path
from types import MappingProxyType

from ptoas import _core
from ptodsl import pto
from ptodsl._func import FuncTemplate
from ptodsl.stdlib import _exports


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def expect_raises(exc_type, func, message_substring=None) -> Exception:
    try:
        func()
    except exc_type as exc:
        if message_substring is not None and message_substring not in str(exc):
            raise AssertionError(
                f"expected {exc_type.__name__} containing {message_substring!r}, got {exc!r}"
            ) from exc
        return exc
    except Exception as exc:
        raise AssertionError(
            f"expected {exc_type.__name__}, got {exc.__class__.__name__}: {exc}"
        ) from exc
    raise AssertionError(f"expected {exc_type.__name__} to be raised")


def catalog_lists_init_core() -> None:
    expect("init_core" in _exports.EXPORTS, "catalog must list init_core")
    expect(_exports.EXPORTS["init_core"] == ("ptodsl.stdlib.init_core", "init_core"),
           "catalog entry must map to the implementation module and attribute")


def import_does_not_eagerly_load_implementation() -> None:
    sys.modules.pop("ptodsl.stdlib.init_core", None)
    sys.modules.pop("ptodsl.pto", None)
    import importlib
    importlib.import_module("ptodsl.pto")
    expect("ptodsl.stdlib.init_core" not in sys.modules,
           "importing ptodsl.pto must not eagerly load the init_core implementation")
    expect("init_core" in pto.__all__, "pto.__all__ must include init_core")
    expect("init_core" in dir(pto), "dir(pto) must include init_core")


def resolution_returns_func_template() -> None:
    impl = _exports.resolve_export("init_core")
    expect(isinstance(impl, FuncTemplate),
           "resolved init_core must be a @pto.func FuncTemplate")
    expect(impl.spec.symbol_name == "init_core",
           "resolved FuncTemplate symbol name must match the public name")


def cached_on_pto_module() -> None:
    first = pto.init_core
    second = pto.init_core
    expect(first is second, "pto.init_core must be cached on the module after first resolution")


def calling_outside_tracing_is_rejected() -> None:
    expect_raises(
        RuntimeError,
        pto.init_core,
        "while tracing",
    )


def auto_mode_is_rejected() -> None:
    @pto.jit(target="a5", mode="auto")
    def auto_kernel(inp: pto.ptr(pto.f32, "gm")):
        pto.init_core()

    expect_raises(
        RuntimeError,
        lambda: auto_kernel.mlir_text(),
        "explicit",
    )


def non_a5_target_is_rejected() -> None:
    @pto.jit(target="a3", mode="explicit")
    def a3_kernel(inp: pto.ptr(pto.f32, "gm")):
        pto.init_core()

    expect_raises(
        ValueError,
        lambda: a3_kernel.mlir_text(),
        "target",
    )


def emitc_backend_is_rejected() -> None:
    @pto.jit(target="a5", backend="emitc", mode="explicit", kernel_kind="vector")
    def emitc_kernel(inp: pto.ptr(pto.f32, "gm"), out: pto.ptr(pto.f32, "gm")):
        pto.init_core()

    expect_raises(
        ValueError,
        lambda: emitc_kernel.mlir_text(),
        "backend",
    )


def unknown_export_is_rejected() -> None:
    expect_raises(
        KeyError,
        lambda: _exports.resolve_export("not_an_export"),
        "not_an_export",
    )


def name_mismatch_is_rejected() -> None:
    saved = _exports.EXPORTS
    _exports.EXPORTS = MappingProxyType(
        {"mismatched": ("ptodsl.stdlib.init_core", "init_core")}
    )
    try:
        expect_raises(
            TypeError,
            lambda: _exports.resolve_export("mismatched"),
            "match",
        )
    finally:
        _exports.EXPORTS = saved


def production_catalog_is_read_only() -> None:
    expect_raises(
        TypeError,
        lambda: operator.setitem(
            _exports.EXPORTS, "probe", ("ptodsl.stdlib.init_core", "init_core")
        ),
    )


def star_import_covers_public_surface() -> None:
    public_names = [name for name in dir(pto) if not name.startswith("_")]
    exported = set(pto.__all__)
    missing = [name for name in public_names if name not in exported]
    expect(
        not missing,
        "from ptodsl.pto import * must export every public pto member "
        f"(star import resolves __all__); missing: {missing}",
    )
    for name in pto.__all__:
        expect(hasattr(pto, name),
               f"star import of {name!r} must resolve on the pto module")
    expect("init_core" in exported,
           "star export list must include stdlib names such as init_core")


def standard_interfaces_are_exposed() -> None:
    for name in (
        "get_ctrl",
        "set_ctrl",
        "set_mov_pad_val",
        "set_loop_size_ubtoout",
        "set_loop_size_outtoub",
        "set_store_atomic_cfg",
    ):
        expect(hasattr(pto, name), f"pto.{name} standard interface must be exposed")


def set_mov_pad_val_accepts_typed_scalars() -> None:
    @pto.jit(target="a5", mode="explicit", kernel_kind="cube")
    def typed_probe(inp: pto.ptr(pto.f32, "gm"), out: pto.ptr(pto.f32, "gm")):
        pto.set_mov_pad_val(pto.const(1, dtype=pto.i16))
        pto.set_mov_pad_val(pto.const(0.5, dtype=pto.f16))

    text = typed_probe.mlir_text()
    expect(": i16" in text,
           "set_mov_pad_val must preserve i16 runtime scalars")
    expect(": f16" in text,
           "set_mov_pad_val must preserve f16 runtime scalars")


def set_mov_pad_val_literal_defaults() -> None:
    @pto.jit(target="a5", mode="explicit", kernel_kind="cube")
    def literal_probe(inp: pto.ptr(pto.f32, "gm"), out: pto.ptr(pto.f32, "gm")):
        pto.set_mov_pad_val(0)
        pto.set_mov_pad_val(0.5)

    text = literal_probe.mlir_text()
    expect(": i32" in text,
           "set_mov_pad_val must default int literals to i32")
    expect(": f32" in text,
           "set_mov_pad_val must default float literals to f32")


def set_mov_pad_val_rejects_unsupported_types() -> None:
    @pto.jit(target="a5", mode="explicit", kernel_kind="cube")
    def bad_probe(inp: pto.ptr(pto.f32, "gm"), out: pto.ptr(pto.f32, "gm")):
        pto.set_mov_pad_val(pto.const(2, dtype=pto.i64))

    expect_raises(
        TypeError,
        lambda: bad_probe.mlir_text(),
        "i8/i16/i32 or f16/bf16/f32",
    )


def vector_kernel_emits_full_init_sequence() -> None:
    @pto.jit(target="a5", mode="explicit", kernel_kind="vector")
    def vector_probe(inp: pto.ptr(pto.f32, "gm"), out: pto.ptr(pto.f32, "gm")):
        pto.init_core()

    text = vector_probe.mlir_text()
    for needle in (
        "pto.get_ctrl",
        "pto.and",
        "pto.or",
        "pto.set_ctrl",
        "pto.set_loop_size_ubtoout",
        "pto.set_loop_size_outtoub",
        "pto.set_store_atomic_cfg",
    ):
        expect(needle in text, f"vector init_core must emit {needle!r}")
    expect("pto.set_mov_pad_val" not in text,
           "vector init_core must not emit the cube-only set_mov_pad_val")


def cube_kernel_emits_cube_branch() -> None:
    @pto.jit(target="a5", mode="explicit", kernel_kind="cube")
    def cube_probe(inp: pto.ptr(pto.f32, "gm"), out: pto.ptr(pto.f32, "gm")):
        pto.init_core()

    text = cube_probe.mlir_text()
    for needle in (
        "pto.get_ctrl",
        "pto.set_ctrl",
        "pto.set_mov_pad_val",
        "pto.set_store_atomic_cfg",
    ):
        expect(needle in text, f"cube init_core must emit {needle!r}")
    expect("pto.ub.set_mask" not in text,
           "init_core must not emit internal ub.set_mask operations")
    expect("pto.set_loop_size_" not in text,
           "cube init_core must not emit the vector-only set_loop_size")


def repeated_calls_reuse_one_helper() -> None:
    @pto.jit(target="a5", mode="explicit", kernel_kind="vector")
    def twice_probe(inp: pto.ptr(pto.f32, "gm"), out: pto.ptr(pto.f32, "gm")):
        pto.init_core()
        pto.init_core()

    text = twice_probe.mlir_text()
    helper_defs = re.findall(r"func\.func private @init_core__ptodsl_\w+\(", text)
    call_sites = text.count("call @init_core__ptodsl_")
    expect(len(helper_defs) == 1, "repeated calls must reuse one helper definition")
    expect(call_sites == 2, "repeated calls must emit one call site per invocation")


def implicit_vector_kernel_keeps_top_level_init() -> None:
    @pto.jit(target="a5", mode="explicit")
    def implicit_vector_probe():
        pto.init_core()

    text = implicit_vector_probe.mlir_text()
    expect("pto.set_loop_size_ubtoout" in text and
           "pto.set_loop_size_outtoub" in text,
           "an implicit Vector kernel must keep the historical Vector initialization")
    expect("pto.set_mov_pad_val" not in text,
           "an implicit Vector kernel must not emit Cube initialization")


def mixed_kernel_specializes_init_by_physical_section() -> None:
    @pto.jit(target="a5", mode="explicit", ast_rewrite=False)
    def mixed_probe():
        with pto.section("cube"):
            pto.init_core()
        with pto.section("vector"):
            pto.init_core()

    text = mixed_probe.mlir_text()
    cube_call = re.search(r"pto\.section\.cube \{\s+func\.call @([^\(]+)", text)
    vector_call = re.search(r"pto\.section\.vector \{\s+func\.call @([^\(]+)", text)
    expect(cube_call is not None, "Cube section must call an init_core helper")
    expect(vector_call is not None, "Vector section must call an init_core helper")
    expect(cube_call.group(1) != vector_call.group(1),
           "Cube and Vector sections must use distinct init_core specializations")
    expect("pto.set_mov_pad_val" in text,
           "Cube init_core specialization must set the move padding value")
    expect("pto.set_loop_size_ubtoout" in text and
           "pto.set_loop_size_outtoub" in text,
           "Vector init_core specialization must restore DMA loop sizes")


def mixed_kernel_survives_full_vpto_pipeline() -> None:
    @pto.jit(target="a5", mode="explicit", ast_rewrite=False)
    def mixed_pipeline_probe():
        with pto.section("cube"):
            pto.init_core()
        with pto.section("vector"):
            pto.init_core()

    with tempfile.TemporaryDirectory() as temp_dir:
        input_path = Path(temp_dir) / "mixed-init-core.mlir"
        output_path = Path(temp_dir) / "mixed-init-core-vpto.mlir"
        input_path.write_text(mixed_pipeline_probe.mlir_text(), encoding="utf-8")
        result = _core.main(
            [
                "ptoas",
                "--emit-vpto",
                "--pto-arch=a5",
                "--pto-backend=vpto",
                "--vpto-scheduler=off",
                str(input_path),
                "-o",
                str(output_path),
            ]
        )
        expect(result == 0, "mixed init_core must survive the full VPTO pipeline")
        text = output_path.read_text(encoding="utf-8")

    modules = re.split(r"(?=  module attributes)", text)
    vector_module = next(
        part for part in modules if "#pto.kernel_kind<vector>" in part
    )
    cube_module = next(
        part for part in modules if "#pto.kernel_kind<cube>" in part
    )
    entry_pattern = re.compile(
        r"func\.func @mixed_pipeline_probe\(\).*?^    }",
        re.MULTILINE | re.DOTALL,
    )
    vector_entry = entry_pattern.search(vector_module)
    cube_entry = entry_pattern.search(cube_module)
    expect(vector_entry is not None, "split Vector module must retain the kernel entry")
    expect(cube_entry is not None, "split Cube module must retain the kernel entry")
    expect("pto.set_loop_size_ubtoout" in vector_entry.group(0),
           "split Vector entry must contain Vector core initialization")
    expect("pto.set_mov_pad_val" not in vector_entry.group(0),
           "split Vector entry must not contain Cube core initialization")
    expect("pto.set_mov_pad_val" in cube_entry.group(0),
           "split Cube entry must contain Cube core initialization")
    expect("pto.set_loop_size_" not in cube_entry.group(0),
           "split Cube entry must not contain Vector core initialization")


def mixed_kernel_rejects_unscoped_init_before_sections() -> None:
    @pto.jit(target="a5", mode="explicit", ast_rewrite=False)
    def mixed_top_level_probe():
        pto.init_core()
        with pto.section("cube"):
            pass
        with pto.section("vector"):
            pass

    expect_raises(
        RuntimeError,
        mixed_top_level_probe.mlir_text,
        "once inside each physical section",
    )


def mixed_kernel_rejects_unscoped_init_after_sections() -> None:
    @pto.jit(target="a5", mode="explicit", ast_rewrite=False)
    def mixed_top_level_probe():
        with pto.section("cube"):
            pass
        with pto.section("vector"):
            pass
        pto.init_core()

    expect_raises(
        RuntimeError,
        mixed_top_level_probe.mlir_text,
        "once inside each physical section",
    )


def _merged_init_kernels():
    @pto.jit(name="merged_init_a", target="a5", mode="explicit", kernel_kind="vector")
    def merged_init_a(inp: pto.ptr(pto.f32, "gm")):
        pto.init_core()

    @pto.jit(name="merged_init_b", target="a5", mode="explicit", kernel_kind="vector")
    def merged_init_b(inp: pto.ptr(pto.f32, "gm")):
        pto.init_core()

    return pto.merge_jit_modules(merged_init_a, merged_init_b)


def _child_entry_and_init_helper(child):
    entries = []
    helpers = []
    for op in child.body.operations:
        if op.operation.name != "func.func":
            continue
        attributes = op.operation.attributes
        if "pto.entry" in attributes:
            entries.append(op)
        if "pto.ptodsl.logical_name" not in attributes:
            continue
        logical_name = str(attributes["pto.ptodsl.logical_name"]).strip('"')
        if logical_name == "init_core":
            helpers.append(op)
    expect(len(entries) == 1,
           "each merged child must contain one exported kernel entry")
    expect(len(helpers) == 1,
           "each merged child must contain one init_core helper definition")
    return entries[0], helpers[0]


def _check_child_helper_visibility(child):
    entry, helper = _child_entry_and_init_helper(child)
    entry_symbol = str(entry.operation.attributes["sym_name"]).strip('"')
    helper_symbol = str(helper.operation.attributes["sym_name"]).strip('"')

    expect("sym_visibility" not in entry.operation.attributes,
           "kernel entries must retain public symbol visibility")
    expect(
        str(helper.operation.attributes["sym_visibility"]).strip('"') == "private",
        "init_core helpers must be private in each owning child",
    )
    expect(
        "pto.entry" not in helper.operation.attributes
        and "pto.visibility" not in helper.operation.attributes,
        "ordinary helpers must not carry entry or external artifact visibility",
    )
    expect(f"call @{helper_symbol}" in str(child),
           "each merged kernel must call its own child-local init_core helper")
    return entry_symbol, helper_symbol


def merged_kernels_keep_init_core_helpers_private() -> None:
    merged = _merged_init_kernels()
    children = []
    for op in merged.body.operations:
        if op.operation.name == "builtin.module":
            children.append(op)
    expect(len(children) == 2,
           "merged kernels must retain two independent backend children")

    helper_symbols = []
    entry_symbols = []
    for child in children:
        entry_symbol, helper_symbol = _check_child_helper_visibility(child)
        entry_symbols.append(entry_symbol)
        helper_symbols.append(helper_symbol)

    expect(set(entry_symbols) == {"merged_init_a", "merged_init_b"},
           "merged children must retain both authored kernel entries")
    expect(len(set(helper_symbols)) == 1,
           "equal init_core specializations should keep their stable symbol name")


def constants_match_reference() -> None:
    @pto.jit(target="a5", mode="explicit", kernel_kind="vector")
    def const_probe(inp: pto.ptr(pto.f32, "gm"), out: pto.ptr(pto.f32, "gm")):
        pto.init_core()

    text = const_probe.mlir_text()
    expect("281474976710656" in text, "ctrl keep mask 0x1000000000000 must be emitted")
    expect("pto.constant 8 : i64" in text, "ctrl preset bits 0x8 must be emitted")
    expect("36" in text, "default st_atomic_cfg 0b00100100 == 36 must be emitted")


def main() -> None:
    catalog_lists_init_core()
    import_does_not_eagerly_load_implementation()
    resolution_returns_func_template()
    cached_on_pto_module()
    calling_outside_tracing_is_rejected()
    auto_mode_is_rejected()
    non_a5_target_is_rejected()
    emitc_backend_is_rejected()
    unknown_export_is_rejected()
    name_mismatch_is_rejected()
    production_catalog_is_read_only()
    star_import_covers_public_surface()
    standard_interfaces_are_exposed()
    set_mov_pad_val_accepts_typed_scalars()
    set_mov_pad_val_literal_defaults()
    set_mov_pad_val_rejects_unsupported_types()
    vector_kernel_emits_full_init_sequence()
    cube_kernel_emits_cube_branch()
    repeated_calls_reuse_one_helper()
    implicit_vector_kernel_keeps_top_level_init()
    mixed_kernel_specializes_init_by_physical_section()
    mixed_kernel_survives_full_vpto_pipeline()
    mixed_kernel_rejects_unscoped_init_before_sections()
    mixed_kernel_rejects_unscoped_init_after_sections()
    merged_kernels_keep_init_core_helpers_private()
    constants_match_reference()


if __name__ == "__main__":
    main()
