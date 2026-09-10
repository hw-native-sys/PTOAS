#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from ptoas.mlir.ir import Context, InsertionPoint, Location, Module
from ptoas.mlir.dialects import pto


def assert_contains(text: str, needle: str) -> None:
    if needle not in text:
        raise AssertionError(f"missing {needle!r} in:\n{text}")


def main() -> None:
    with Context() as ctx, Location.unknown(ctx):
        pto.register_dialect(ctx, load=True)
        module = Module.create()
        with InsertionPoint(module.body):
            pto.CmoCacheInvalidOp(pto.AddressSpace.GM)
            pto.FenceBarrierAllOp(pto.FenceScope.GM)

        text = str(module)
        assert_contains(text, "pto.cmo.cacheinvalid all #pto.address_space<gm>")
        assert_contains(text, "pto.fence.barrier_all <gm>")

        single_line = Module.parse(
            """
            module {
              func.func @single_line_parse(%payload_ptr: !pto.ptr<i32>)
                  attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
                pto.cmo.cacheinvalid %payload_ptr single_cache_line : !pto.ptr<i32>
                return
              }
            }
            """
        )
        assert_contains(str(single_line), "single_cache_line")

        single_line_ctor = Module.parse(
            """
            module {
              func.func @single_line_ctor(%payload_ptr: !pto.ptr<i32>)
                  attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
                return
              }
            }
            """
        )
        func = single_line_ctor.body.operations[0]
        block = func.regions[0].blocks[0]
        with InsertionPoint.at_block_begin(block):
            pto.CmoCacheInvalidOp(pto.AddressSpace.GM, addr=block.arguments[0])
        assert_contains(str(single_line_ctor), "single_cache_line")

    print("memory_consistency_bindings: PASS")


if __name__ == "__main__":
    main()
