#!/usr/bin/env python3
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
        module = Module.parse(
            """
            module {
              func.func @binding_smoke(
                  %src: !pto.tile_buf<vec, 2x16xui8>,
                  %tmp: !pto.tile_buf<vec, 1x1xui8>,
                  %dst: !pto.tile_buf<vec, 2x16xui8, slayout=row_major>,
                  %fp_src: !pto.tile_buf<acc, 32x32xf32, blayout=col_major, slayout=row_major>,
                  %fp: !pto.tile_buf<scaling, 1x32xf16>,
                  %fp_dst: !pto.tile_buf<mat, 32x32xf32>) {
                return
              }
            }
            """
        )
        func = module.body.operations[0]
        block = func.regions[0].blocks[0]
        src, tmp, dst, fp_src, fp, fp_dst = block.arguments
        with InsertionPoint.at_block_begin(block):
            pto.TMovOp(
                None,
                src,
                dst,
                fp=tmp,
                grpAxis=pto.MxGroupAxisAttr.get(pto.MxGroupAxis.Axis0),
            )
            pto.TMovOp(None, fp_src, fp_dst, fp=fp)

        text = str(module)
        assert_contains(text, "grpAxis = #pto<mx_group_axis axis0>")
        assert_contains(text, "!pto.tile_buf<vec, 1x1xui8>")
        assert_contains(text, "!pto.tile_buf<scaling, 1x32xf16>")
        if "aux" in text:
            raise AssertionError(f"unexpected aux API spelling in:\n{text}")

    print("mx_grp_axis_bindings: PASS")


if __name__ == "__main__":
    main()
