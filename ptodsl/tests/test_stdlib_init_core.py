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
from types import MappingProxyType

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
    helper_defs = re.findall(r"func\.func @init_core__ptodsl_\w+\(", text)
    call_sites = text.count("call @init_core__ptodsl_")
    expect(len(helper_defs) == 1, "repeated calls must reuse one helper definition")
    expect(call_sites == 2, "repeated calls must emit one call site per invocation")


def constants_match_reference() -> None:
    @pto.jit(target="a5", mode="explicit", kernel_kind="vector")
    def const_probe(inp: pto.ptr(pto.f32, "gm"), out: pto.ptr(pto.f32, "gm")):
        pto.init_core()

    text = const_probe.mlir_text()
    expect("281474976710656" in text, "ctrl keep mask 0x1000000000000 must be emitted")
    expect("1152921504606846984" in text, "ctrl preset bits 0x1000000000000008 must be emitted")
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
    constants_match_reference()


if __name__ == "__main__":
    main()
