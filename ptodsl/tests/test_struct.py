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


@pto.jit(target="a5")
def named_unknown_field_kernel():
    state = pto.declare_struct(NAMED_STRUCT)
    state.missing = 1


@pto.jit(target="a5")
def named_positional_layer_kernel():
    S = pto.struct({"pt": pto.struct_type(pto.i32, pto.f32)})
    state = pto.declare_struct(S)
    state.pt.x = 1


@pto.jit(target="a5")
def named_nested_bare_read_kernel():
    s = pto.declare_struct(NAMED_NESTED)
    v = s.pt
    _ = v


@pto.jit(target="a5")
def named_del_field_kernel():
    state = pto.declare_struct(NAMED_STRUCT)
    del state.n


@pto.jit(target="a5")
def named_branch_conflict_kernel(x: pto.i32):
    if x > 0:
        S = pto.struct({"a": pto.i32})
    else:
        S = pto.struct({"b": pto.i32})


@pto.jit(target="a5")
def named_rebind_kernel(x: pto.i32):
    state = pto.declare_struct(NAMED_STRUCT)
    state.n = 1
    state = x
    state.n = 2


@pto.jit(target="a5")
def named_static_loop_member_kernel():
    state = pto.declare_struct(NAMED_STRUCT)
    for i in pto.static_range(2):
        state.n = i
    v = state.n
    _ = v


@pto.jit(target="a5")
def named_static_loop_rebind_kernel(x: pto.i32):
    state = pto.declare_struct(NAMED_STRUCT)
    state.n = 1
    for i in pto.static_range(2):
        state = x
    state.n = 2


@pto.jit(target="a5")
def named_multi_target_kernel(x: pto.i32, y: pto.f32):
    a = b = pto.declare_struct(NAMED_STRUCT)
    a.n = x
    b.sum = y
    _ = a.n
    _ = b.sum


@pto.jit(target="a5")
def named_multi_target_alias_kernel():
    Inner = pto.struct({"x": pto.i32})
    A = B = Inner
    s = pto.declare_struct(A)
    s.x = 1
    t = pto.declare_struct(B)
    t.x = 2


@pto.jit(target="a5")
def named_annassign_decl_kernel():
    S: object = pto.struct({"x": pto.i32, "y": pto.f32})
    state = pto.declare_struct(S)
    state.x = 1
    v = state.y
    _ = v


@pto.jit(target="a5", ast_rewrite=False)
def named_member_no_rewrite_kernel():
    state = pto.declare_struct(NAMED_STRUCT)
    state.n = 1


@pto.struct
class ClassPoint:
    x: pto.i32
    y: pto.f32


@pto.struct
class ClassInner:
    x: pto.i32
    y: pto.f32


@pto.struct
class ClassOuter:
    id: pto.i32
    inner: ClassInner


@pto.jit(target="a5")
def class_member_kernel(x: pto.i32, y: pto.f32):
    state = ClassPoint(x, y)
    count = state.x
    total = state.y
    _ = count
    _ = total


@pto.jit(target="a5")
def class_kwargs_kernel(x: pto.i32):
    state = ClassPoint(y=2.5, x=x)
    _ = state.x


@pto.jit(target="a5")
def class_declare_only_kernel():
    state = ClassPoint()
    state.x = 1
    _ = state.x


@pto.jit(target="a5")
def class_nested_kernel():
    s = ClassOuter()
    s.inner.x = 1
    v = s.inner.y
    _ = v


@pto.jit(target="a5")
def class_local_kernel(x: pto.i32):
    @pto.struct
    class Local:
        x: pto.i32
        y: pto.f32

    s = Local(x, 2.5)
    v = s.y
    _ = v


@pto.jit(target="a5")
def class_multi_target_kernel(x: pto.i32, y: pto.f32):
    a = b = ClassPoint(x, y)
    _ = a.x
    b.y = y


# Regression for the future-flag fix in rewrite_jit_function: the rewritten
# module must compile with the kernel's own future flags (dont_inherit=True).
# If compile() inherited ``from __future__ import annotations`` from
# _ast_rewrite, ``v: T`` below would stringify and from_class could not
# resolve the kernel-local name ``T`` in module globals.
@pto.jit(target="a5")
def class_local_dtype_alias_kernel(x: pto.i32):
    T = pto.i32

    @pto.struct
    class Local:
        v: T

    s = Local(x)
    _ = s.v


@pto.jit(target="a5", ast_rewrite=False)
def class_no_rewrite_kernel(x: pto.i32, y: pto.f32):
    state = ClassPoint(x, y)
    state.x = 1


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

    def test_reserved_field_name_set_rejected(self):
        for name in sorted(_types._STRUCT_RESERVED_FIELD_NAMES):
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, "reserved"):
                    pto.struct({name: pto.i32})

    def test_unknown_field_rejected(self):
        with self.assertRaisesRegex(SyntaxError, "unknown struct field 'missing'"):
            named_unknown_field_kernel.compile()

    def test_positional_layer_member_access_rejected(self):
        with self.assertRaisesRegex(SyntaxError, "positional"):
            named_positional_layer_kernel.compile()

    def test_nested_struct_bare_read_rejected(self):
        with self.assertRaisesRegex(SyntaxError, "nested struct member as a value"):
            named_nested_bare_read_kernel.compile()

    def test_del_field_rejected(self):
        with self.assertRaisesRegex(SyntaxError, "does not support del"):
            named_del_field_kernel.compile()

    def test_branch_merge_conflicting_types_rejected(self):
        with self.assertRaisesRegex(SyntaxError, "incompatible struct types"):
            named_branch_conflict_kernel.compile()

    def test_rebinding_cancels_struct_identity(self):
        # After ``state = x`` the name is no longer a struct value, so the
        # second member write is not rewritten (documented plain-Python
        # attribute behavior); only the first write reaches the IR.
        text = named_rebind_kernel.compile().mlir_text()
        self.assertEqual(text.count("pto.struct_set"), 1)

    def test_static_loop_member_access_survives(self):
        text = named_static_loop_member_kernel.compile().mlir_text()
        self.assertIn("pto.struct_set", text)
        self.assertIn("pto.struct_get", text)

    def test_static_loop_rebinding_cancels_struct_identity(self):
        # The loop body rebinds ``state`` to a scalar, so the member write
        # after the loop must not be rewritten to struct_set.
        text = named_static_loop_rebind_kernel.compile().mlir_text()
        self.assertEqual(text.count("pto.struct_set"), 1)

    def test_annassign_declaration_binds_struct_type(self):
        text = named_annassign_decl_kernel.compile().mlir_text()
        self.assertIn("pto.struct_set", text)
        self.assertIn("pto.struct_get", text)

    def test_multi_target_assignment_binds_all_names(self):
        # ``a = b = declare_struct(...)`` evaluates the RHS once; both names
        # alias the same struct value and their member access is rewritten.
        text = named_multi_target_kernel.compile().mlir_text()
        self.assertEqual(text.count("pto.declare_struct"), 1)
        self.assertIn("pto.struct_set", text)
        self.assertIn("pto.struct_get", text)
        self.assertIn("[0]", text)
        self.assertIn("[1]", text)

    def test_multi_target_assignment_binds_type_aliases(self):
        # ``A = B = Inner`` binds the struct type on both names so each can
        # back a declare_struct(...) call.
        text = named_multi_target_alias_kernel.compile().mlir_text()
        self.assertEqual(text.count("pto.declare_struct"), 2)
        self.assertEqual(text.count("pto.struct_set"), 2)

    def test_ast_rewrite_disabled_member_access_diagnostic(self):
        with self.assertRaisesRegex(AttributeError, "AST rewriting"):
            named_member_no_rewrite_kernel.compile()

    def test_unresolvable_dotted_field_type_does_not_crash(self):
        import ast as _ast

        from ptodsl import _ast_rewrite

        src = (
            "def k():\n"
            "    S = pto.struct({'x': a.b.c})\n"
            "    state = pto.declare_struct(S)\n"
            "    state.x = 1\n"
        )
        fn = _ast.parse(src).body[0]
        rewriter = _ast_rewrite._StructMemberRewriter(static_env={})
        # Unresolvable field types must leave the statements unrewritten
        # instead of crashing with a raw AttributeError.
        rewriter.rewrite_block(fn.body)


class StructClassFormTest(unittest.TestCase):
    def test_class_form_resolves_same_type_as_dict(self):
        with make_context() as ctx, Location.unknown(ctx):
            self.assertEqual(
                str(ClassPoint.resolve()),
                str(pto.struct({"x": pto.i32, "y": pto.f32}).resolve()),
            )

    def test_class_form_field_metadata(self):
        self.assertEqual(ClassPoint.field_names, ("x", "y"))
        self.assertTrue(ClassPoint.is_named)
        self.assertEqual(ClassPoint.field_index("y"), 1)
        self.assertEqual(ClassPoint.field_descriptor_at(0)[0], "x")

    def test_class_form_ctor_validation(self):
        with self.assertRaisesRegex(TypeError, "positional"):
            ClassPoint(1, 2, 3)
        with self.assertRaisesRegex(TypeError, "unexpected field"):
            ClassPoint(x=1, z=2)
        with self.assertRaisesRegex(TypeError, "multiple values"):
            ClassPoint(1, x=2)
        with self.assertRaisesRegex(TypeError, "missing"):
            ClassPoint(1)

    def test_class_form_declaration_errors(self):
        with self.assertRaisesRegex(TypeError, "default value"):
            @pto.struct
            class _WithDefault:
                x: pto.i32 = 0

        with self.assertRaisesRegex(TypeError, "unsupported class attribute"):
            @pto.struct
            class _WithMethod:
                x: pto.i32

                def foo(self):
                    pass

        with self.assertRaisesRegex(ValueError, "underscore"):
            @pto.struct
            class _WithUnderscore:
                _x: pto.i32

        with self.assertRaisesRegex(ValueError, "at least one"):
            @pto.struct
            class _Empty:
                pass

    def test_class_form_rejects_inheritance(self):
        class _PlainBase:
            pass

        with self.assertRaisesRegex(TypeError, "inheritance"):
            @pto.struct
            class _Sub(_PlainBase):
                x: pto.i32

    def test_positional_descriptor_not_callable(self):
        with self.assertRaisesRegex(TypeError, "not callable"):
            pto.struct_type(pto.i32, pto.f32)(1, 2)

    def test_class_member_access_rewrites_to_canonical_ops(self):
        text = class_member_kernel.compile().mlir_text()
        self.assertIn("pto.declare_struct", text)
        # Constructor initializes both fields; reads become struct_get.
        self.assertEqual(text.count("pto.struct_set"), 2)
        self.assertEqual(text.count("pto.struct_get"), 2)
        self.assertNotIn("ClassPoint", text)

    def test_class_kwargs_ctor(self):
        text = class_kwargs_kernel.compile().mlir_text()
        self.assertEqual(text.count("pto.struct_set"), 2)
        self.assertIn("pto.struct_get", text)

    def test_class_declare_only_ctor(self):
        text = class_declare_only_kernel.compile().mlir_text()
        # Point() declares without initializing; only the explicit member
        # write produces struct_set.
        self.assertEqual(text.count("pto.struct_set"), 1)
        self.assertIn("pto.struct_get", text)

    def test_class_nested_member_access(self):
        text = class_nested_kernel.compile().mlir_text()
        self.assertIn("[1, 0]", text)
        self.assertIn("[1, 1]", text)

    def test_class_form_function_local(self):
        text = class_local_kernel.compile().mlir_text()
        self.assertEqual(text.count("pto.struct_set"), 2)
        self.assertIn("pto.struct_get", text)

    def test_class_form_construction_without_ast_rewrite(self):
        # Construction itself needs no AST rewrite (the descriptor __call__
        # emits declare+set at trace time); only member access does.
        with self.assertRaisesRegex(AttributeError, "AST rewriting"):
            class_no_rewrite_kernel.compile()

    def test_class_multi_target_assignment_binds_all_names(self):
        # ``a = b = ClassPoint(...)`` evaluates the constructor once; both
        # names alias the same struct value and support member access.  The
        # constructor initializes both fields (2 struct_set) and the explicit
        # member write adds a third.
        text = class_multi_target_kernel.compile().mlir_text()
        self.assertEqual(text.count("pto.declare_struct"), 1)
        self.assertEqual(text.count("pto.struct_set"), 3)
        self.assertIn("pto.struct_get", text)

    def test_class_form_local_dtype_alias_annotation(self):
        # Regression for the future-flag fix in rewrite_jit_function: the
        # rewritten module must compile with the kernel's own future flags.
        # If compile() inherited ``from __future__ import annotations`` from
        # _ast_rewrite, ``v: T`` would stringify and from_class could not
        # resolve the kernel-local name ``T`` in module globals.
        text = class_local_dtype_alias_kernel.compile().mlir_text()
        self.assertIn("pto.declare_struct", text)
        self.assertIn("pto.struct_set", text)
        self.assertIn("pto.struct_get", text)

    def test_class_form_resolves_string_annotations(self):
        # PEP 563 stringifies class-body annotations; from_class resolves
        # them in the defining module's namespace (here this test module).
        @pto.struct
        class _StringAnnotated:
            x: "pto.i32"
            y: "pto.f32"

        with make_context() as ctx, Location.unknown(ctx):
            self.assertEqual(
                str(_StringAnnotated.resolve()),
                str(pto.struct({"x": pto.i32, "y": pto.f32}).resolve()),
            )


if __name__ == "__main__":
    unittest.main()
