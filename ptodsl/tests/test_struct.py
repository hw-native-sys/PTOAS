#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import unittest

from ptodsl import _types, pto
from ptodsl._context import make_context
from ptoas.mlir.ir import InsertionPoint, Location, Module


STRUCT_TYPE = pto.struct_type(pto.i32, pto.struct_type(pto.f32, pto.i16))


@pto.jit(target="a5")
def struct_surface_kernel(x: pto.i32, y: pto.f32):
    state = pto.declare_struct(STRUCT_TYPE)
    pto.struct_set(state, 0, x)
    pto.struct_set(state, (1, 0), y)
    pto.struct_set(state, (1, 1), 7)
    count = pto.struct_get(state, [0])
    pto.struct_set(state, 0, 1)
    value = pto.struct_get(state, (1, 0))
    _ = count
    _ = value


@pto.jit(target="a5")
def struct_outer_loop_kernel():
    state = pto.declare_struct(STRUCT_TYPE)
    with pto.for_(0, 2, step=1):
        pto.struct_set(state, 0, 1)


@pto.jit(target="a5")
def struct_carry_kernel():
    state = pto.declare_struct(STRUCT_TYPE)
    with pto.for_(0, 2, step=1).carry(state=state) as loop:
        loop.update(state=loop.state)


class StructSurfaceTest(unittest.TestCase):
    def test_public_namespace_exports_struct_surface(self):
        for name in ("struct_type", "struct", "declare_struct", "struct_get", "struct_set"):
            with self.subTest(name=name):
                self.assertTrue(hasattr(pto, name), name)
        self.assertIn("struct_type", _types.__all__)
        self.assertIn("struct", _types.__all__)

    def test_struct_type_is_lazy_and_exposes_fields(self):
        descriptor = pto.struct_type(pto.i32, pto.struct_type(pto.f32, pto.i16))
        self.assertEqual(len(descriptor.field_descriptors), 2)

        with make_context() as ctx, Location.unknown(ctx):
            resolved = descriptor.resolve()
            self.assertEqual(str(resolved), "!pto.struct<i32, !pto.struct<f32, i16>>")
            self.assertEqual([str(field) for field in resolved.field_types], [
                "i32",
                "!pto.struct<f32, i16>",
            ])

    def test_struct_surface_emits_nested_get_and_set(self):
        text = struct_surface_kernel.compile().mlir_text()
        self.assertIn("pto.declare_struct", text)
        self.assertIn("pto.struct_set", text)
        self.assertIn("pto.struct_get", text)
        self.assertIn("[1, 0]", text)
        self.assertIn("[1, 1]", text)
        self.assertLess(
            text.index("pto.struct_get"),
            text.rindex("pto.struct_set"),
            "a field read must remain an SSA value when the field is subsequently written",
        )

        with make_context() as ctx:
            module = Module.parse(text, ctx)
            module.operation.verify()

    def test_literal_and_ssa_type_rules(self):
        with make_context() as ctx, Location.unknown(ctx):
            module = Module.create()
            with InsertionPoint(module.body):
                state = pto.declare_struct(STRUCT_TYPE)
                self.assertIsNone(pto.struct_set(state, 0, 3))
                self.assertIsNone(pto.struct_set(state, (1, 0), 3))
                self.assertIsNone(pto.struct_set(state, (1, 0), 3.5))
                self.assertIsNone(pto.struct_set(state, (1, 1), pto.i16(7)))

                with self.assertRaisesRegex(TypeError, "bool literals"):
                    pto.struct_set(state, 0, True)
                with self.assertRaisesRegex(TypeError, "Python int/float literal or an SSA value"):
                    pto.struct_set(state, 0, "7")
                with self.assertRaisesRegex(TypeError, "floating-point literal"):
                    pto.struct_set(state, 0, 3.5)
                with self.assertRaisesRegex(TypeError, "must exactly match"):
                    pto.struct_set(state, (1, 1), pto.i32(7))

            module.operation.verify()

    def test_rejects_invalid_field_types_and_paths(self):
        self.assertRaisesRegex(ValueError, "at least one field", pto.struct_type)

        with make_context() as ctx, Location.unknown(ctx):
            invalid_fields = (
                pto.i1,
                pto.f8e4m3,
                pto.ptr(pto.i32, "ub"),
            )
            for field in invalid_fields:
                with self.subTest(field=field), self.assertRaisesRegex(TypeError, "not supported"):
                    pto.struct_type(field).resolve()

            module = Module.create()
            with InsertionPoint(module.body):
                state = pto.declare_struct(STRUCT_TYPE)
                invalid_paths = (
                    ((), ValueError, "must not be empty"),
                    (True, TypeError, "static int"),
                    ((0, True), TypeError, "static Python int"),
                    ((-1,), ValueError, "non-negative"),
                    ((2,), ValueError, "out of range"),
                    ((0, 0), TypeError, "cannot descend further"),
                    ((1,), TypeError, "must end at a scalar field"),
                )
                for path, error_type, message in invalid_paths:
                    with self.subTest(path=path), self.assertRaisesRegex(error_type, message):
                        pto.struct_get(state, path)

                with self.assertRaisesRegex(TypeError, "must be a value of !pto.struct"):
                    pto.struct_get(pto.i32(1), 0)

    def test_struct_is_rejected_as_entry_abi_annotation(self):
        def bad_entry(state: STRUCT_TYPE):
            del state

        with self.assertRaisesRegex(TypeError, "Stack-local structs must be created inside"):
            pto.jit(target="a5")(bad_entry)

    def test_struct_is_rejected_as_kernel_module_abi_annotation(self):
        def bad_helper(state: STRUCT_TYPE):
            del state

        with self.assertRaisesRegex(TypeError, "Stack-local structs must be created inside"):
            pto.jit(target="a5", entry=False)(bad_helper)

    def test_struct_is_rejected_as_subkernel_abi_annotation(self):
        def bad_subkernel(state: STRUCT_TYPE):
            del state

        for decorator in (pto.tileop, pto.simt):
            with self.subTest(decorator=decorator), self.assertRaisesRegex(
                TypeError,
                "unsupported subkernel annotation",
            ):
                decorator(bad_subkernel)

    def test_struct_carry_is_rejected_but_outer_loop_mutation_is_legal(self):
        legal_text = struct_outer_loop_kernel.compile().mlir_text()
        self.assertIn("scf.for", legal_text)
        self.assertIn("pto.struct_set", legal_text)

        with self.assertRaisesRegex(TypeError, "does not accept stack-local struct values"):
            struct_carry_kernel.compile()

        with self.assertRaisesRegex(TypeError, "does not accept pto.struct_type"):
            pto.for_(0, 2, step=1).carry(state=STRUCT_TYPE)


NAMED_STRUCT = pto.struct({"n": pto.i32, "sum": pto.f32})
NAMED_NESTED = pto.struct({
    "id": pto.i32,
    "pt": pto.struct({"x": pto.i32, "y": pto.f32}),
})


@pto.jit(target="a5")
def named_member_kernel(x: pto.i32, y: pto.f32):
    state = pto.declare_struct(NAMED_STRUCT)
    state.n = x
    state.sum = y
    count = state.n
    total = state.sum
    _ = count
    _ = total


@pto.jit(target="a5")
def named_nested_kernel():
    s = pto.declare_struct(NAMED_NESTED)
    s.pt.x = 1
    v = s.pt.y
    _ = v


@pto.jit(target="a5")
def named_augassign_kernel(x: pto.i32):
    state = pto.declare_struct(NAMED_STRUCT)
    state.n += x
    _ = state.n


@pto.jit(target="a5")
def named_local_alias_kernel():
    Inner = pto.struct({"x": pto.i32, "y": pto.f32})
    Alias = Inner
    Outer = pto.struct({"id": pto.i32, "inner": Alias})
    s = pto.declare_struct(Outer)
    s.inner.x = 1
    v = s.inner.y
    _ = v


@pto.jit(target="a5")
def named_ann_mismatch_kernel():
    S = pto.struct({"x": pto.f32})
    state = pto.declare_struct(S)
    state.x: pto.i32 = 1
    _ = state.x


@pto.jit(target="a5")
def named_dup_key_kernel():
    S = pto.struct({"x": pto.i32, "x": pto.f32})
    state = pto.declare_struct(S)
    state.x = 1
    _ = state.x


class NamedStructMemberAccessTest(unittest.TestCase):
    def test_named_descriptor_exposes_fields(self):
        descriptor = pto.struct({"n": pto.i32, "sum": pto.f32})
        self.assertEqual(descriptor.field_names, ("n", "sum"))
        self.assertTrue(descriptor.is_named)
        self.assertEqual(descriptor.field_index("n"), 0)
        self.assertEqual(descriptor.field_index("sum"), 1)
        self.assertEqual(descriptor.field_descriptor_at(1)[0], "sum")

    def test_named_and_positional_resolve_to_same_type(self):
        named = pto.struct({"n": pto.i32, "sum": pto.f32})
        positional = pto.struct_type(pto.i32, pto.f32)
        with make_context() as ctx, Location.unknown(ctx):
            self.assertEqual(
                str(named.resolve()),
                str(positional.resolve()),
            )

    def test_named_field_rules(self):
        with self.assertRaisesRegex(ValueError, "valid Python identifiers"):
            pto.struct({"a-b": pto.i32})
        with self.assertRaisesRegex(ValueError, "Python keyword"):
            pto.struct({"class": pto.i32})
        with self.assertRaisesRegex(ValueError, "underscore"):
            pto.struct({"_x": pto.i32})
        with self.assertRaisesRegex(ValueError, "reserved"):
            pto.struct({"value": pto.i32})
        with self.assertRaisesRegex(TypeError, "dict"):
            pto.struct([pto.i32, pto.f32])
        with self.assertRaisesRegex(ValueError, "at least one"):
            pto.struct({})

    def test_member_access_rewrites_to_canonical_ops(self):
        text = named_member_kernel.compile().mlir_text()
        self.assertIn("pto.struct_set", text)
        self.assertIn("pto.struct_get", text)
        # Field names ("n", "sum") must not leak into the IR.
        for name in ("n", "sum"):
            self.assertNotRegex(text, rf"\b{name}\b")

    def test_nested_member_access_rewrites_to_path(self):
        text = named_nested_kernel.compile().mlir_text()
        self.assertIn("pto.struct_set", text)
        self.assertIn("pto.struct_get", text)
        # Nested member access resolves to position path [1, 0] / [1, 1].
        self.assertIn("[1, 0]", text)
        self.assertIn("[1, 1]", text)

    def test_augassign_member_rewrites_to_read_write(self):
        text = named_augassign_kernel.compile().mlir_text()
        self.assertIn("pto.struct_get", text)
        self.assertIn("pto.struct_set", text)
        self.assertLess(
            text.index("pto.struct_get"),
            text.index("pto.struct_set"),
            "state.n += x should read then write",
        )

    def test_local_descriptor_alias_nesting(self):
        text = named_local_alias_kernel.compile().mlir_text()
        self.assertIn("[1, 0]", text)  # s.inner.x -> path [1, 0]
        self.assertIn("[1, 1]", text)  # s.inner.y -> path [1, 1]

    def test_annassign_type_mismatch_rejected(self):
        with self.assertRaisesRegex(SyntaxError, "does not match field type"):
            named_ann_mismatch_kernel.compile()

    def test_duplicate_literal_key_rejected(self):
        with self.assertRaisesRegex(SyntaxError, "duplicate field name"):
            named_dup_key_kernel.compile()


if __name__ == "__main__":
    unittest.main()
