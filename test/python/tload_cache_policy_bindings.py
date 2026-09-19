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


def check_tload_offsets(policy) -> None:
    """Check optional offset builders, operand segments, and IR round trips."""
    module = Module.parse("""
        module {
          func.func @loads(%src: !pto.partition_tensor_view<16x16xf32>,
                           %dst: !pto.tile_buf<vec, 16x16xf32>,
                           %bytes: i64, %index_bytes: index) {
            return
          }
        }
    """)
    block = module.body.operations[0].regions[0].blocks[0]
    src, dst, byte_offset, index_offset = block.arguments
    with InsertionPoint.at_block_begin(block):
        ordinary = pto.TLoadOp(None, src, dst)
        bypass = pto.TLoadOp(None, src, dst, cache_policy=policy)
        load_i64 = pto.TLoadOp(None, src, dst, cache_policy=policy, offset=byte_offset)
        load_index = pto.TLoadOp(None, src, dst, cache_policy=policy, offset=index_offset)
    if ordinary.offset is not None or bypass.offset is not None:
        raise RuntimeError("tload offset must be absent by default")
    if load_i64.offset != byte_offset or load_index.offset != index_offset:
        raise RuntimeError("tload builder did not preserve offset operands")
    if not module.operation.verify():
        raise RuntimeError("tload offset module did not verify")
    round_trip = Module.parse(str(module))
    if not round_trip.operation.verify():
        raise RuntimeError("tload offset module did not survive a round trip")


def main() -> None:
    with Context() as ctx, Location.unknown(ctx):
        pto.register_dialect(ctx, load=True)
        policy = pto.LoadCachePolicyAttr.get(pto.LoadCachePolicy.L2Bypass)
        if str(policy) != "#pto.load_cache_policy<l2_bypass>":
            raise RuntimeError(f"unexpected load cache policy: {policy}")
        if policy.value != pto.LoadCachePolicy.L2Bypass.value:
            raise RuntimeError(f"unexpected load cache policy value: {policy.value}")
        check_tload_offsets(policy)

    print("tload_cache_policy_bindings: PASS")


if __name__ == "__main__":
    main()
