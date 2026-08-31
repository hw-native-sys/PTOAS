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


@pto.jit(target="a5", mode="explicit")
def sdma_gm_gm_surface_kernel(
    src: pto.ptr(pto.i8, "gm"),
    sess_gm: pto.ptr(pto.i8, "gm"),
    dst: pto.ptr(pto.i8, "gm"),
    nbytes: pto.i64,
):
    sess = pto.declare_struct(pto.async_session_type())
    pto.session_init(sess, sess_gm)
    pto.sdma_gm_gm(dst, src, nbytes, session=sess)
    pto.sdma_gm_gm(dst, src, nbytes, session=sess, soft_put=True, block_bytes=64, channel_idx=0)


class AsyncCommSurfaceTest(unittest.TestCase):
    def test_public_namespace_exports_the_surface(self):
        for name in ("async_session_type", "session_init", "sdma_gm_gm"):
            with self.subTest(name=name):
                self.assertTrue(hasattr(pto, name), name)
        self.assertIn("async_session_type", _types.__all__)

    def test_async_session_type_matches_the_isa_layout(self):
        with make_context() as ctx, Location.unknown(ctx):
            resolved = pto.async_session_type().resolve()
            self.assertEqual(
                str(resolved),
                "!pto.struct<i64, i64, i32, i32, i32, i32, i64, i64, i32, i32, i32, i32, i32>",
            )

    def test_surface_emits_session_init_and_sdma_gm_gm(self):
        text = sdma_gm_gm_surface_kernel.compile().mlir_text()
        self.assertIn("pto.session_init", text)
        self.assertIn("pto.sdma_gm_gm", text)
        self.assertIn("soft_put", text)
        self.assertIn("block_bytes = 64", text)
        self.assertIn("channel_idx = 0", text)

        with make_context() as ctx:
            module = Module.parse(text, ctx)
            module.operation.verify()

    def test_rejects_wrong_session_and_pointer_spaces(self):
        with make_context() as ctx, Location.unknown(ctx):
            module = Module.create()
            with InsertionPoint(module.body):
                sess = pto.declare_struct(pto.async_session_type())
                other = pto.declare_struct(pto.struct_type(pto.i32))
                gm = pto.ptr(pto.i8, "gm")
                ub = pto.ptr(pto.i8, "ub")
                # Materialize dummy SSA pointers via castptr so the helpers see values.
                zero = pto.const(0, dtype=pto.i64)
                gm_ptr = pto.castptr(zero, gm)
                ub_ptr = pto.castptr(zero, ub)

                with self.assertRaisesRegex(TypeError, "13 fields"):
                    pto.session_init(other, gm_ptr)
                with self.assertRaisesRegex(TypeError, "GM pointer"):
                    pto.session_init(sess, ub_ptr)
                with self.assertRaisesRegex(TypeError, "GM pointer"):
                    pto.sdma_gm_gm(ub_ptr, gm_ptr, 64, session=sess)
                with self.assertRaisesRegex(ValueError, "multiple of 64"):
                    pto.sdma_gm_gm(gm_ptr, gm_ptr, 64, session=sess, block_bytes=32)
                with self.assertRaisesRegex(ValueError, "<= 39"):
                    pto.sdma_gm_gm(gm_ptr, gm_ptr, 64, session=sess, channel_idx=40)
                with self.assertRaisesRegex(TypeError, "soft_put"):
                    pto.sdma_gm_gm(gm_ptr, gm_ptr, 64, session=sess, soft_put=1)


if __name__ == "__main__":
    unittest.main()
