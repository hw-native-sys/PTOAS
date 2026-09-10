#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from ptoas.mlir.ir import Context, F32Type, MLIRError, Module
from ptoas.mlir.dialects import pto


def expect_equal(actual: str, expected: str, label: str) -> None:
    if actual != expected:
        raise AssertionError(
            f"{label} mismatch\nexpected: {expected}\nactual:   {actual}"
        )


def expect_contains(text: str, needle: str, label: str) -> None:
    if needle not in text:
        raise AssertionError(
            f"{label} missing substring\nneedle: {needle}\ntext:\n{text}"
        )


def expect_not_contains(text: str, needle: str, label: str) -> None:
    if needle in text:
        raise AssertionError(
            f"{label} unexpectedly contained substring\nneedle: {needle}\ntext:\n{text}"
        )


def expect_parse_error(ctx: Context, asm: str, needle: str, label: str) -> None:
    try:
        Module.parse(asm, ctx)
    except MLIRError as err:
        if needle not in str(err):
            raise AssertionError(
                f"{label} error mismatch\nexpected substring: {needle}\nactual: {err}"
            ) from err
        return
    raise AssertionError(f"{label} unexpectedly parsed successfully")


def main() -> None:
    with Context() as ctx:
        pto.register_dialect(ctx)

        vec = pto.AddressSpaceAttr.get(pto.AddressSpace.VEC, ctx)
        col_major = pto.BLayoutAttr.get(pto.BLayout.ColMajor, ctx)
        row_major = pto.SLayoutAttr.get(pto.SLayout.RowMajor, ctx)
        zero_pad = pto.PadValueAttr.get(pto.PadValue.Zero, ctx)
        cfg = pto.TileBufConfigAttr.get(col_major, row_major, 1024, zero_pad, ctx)

        default_ty = pto.TileBufType.get([1, 16], F32Type.get(ctx), vec, context=ctx)
        expect_equal(
            str(default_ty),
            "!pto.tile_buf<vec, 1x16xf32>",
            "default compact print",
        )

        valid_ty = pto.TileBufType.get(
            [16, 128],
            F32Type.get(ctx),
            vec,
            valid_shape=[16, 1],
            context=ctx,
        )
        expect_equal(
            str(valid_ty),
            "!pto.tile_buf<vec, 16x128xf32, valid=16x1>",
            "valid suffix print",
        )

        non_default_cfg_ty = pto.TileBufType.get(
            [8, 8],
            F32Type.get(ctx),
            vec,
            config=cfg,
            context=ctx,
        )
        expect_equal(
            str(non_default_cfg_ty),
            "!pto.tile_buf<vec, 8x8xf32, blayout=col_major, slayout=row_major, fractal=1024, pad=1>",
            "non-default config suffix print",
        )

        compact_cfg_ty = pto.TileBufType.get(
            [8, 8],
            F32Type.get(ctx),
            vec,
            config=pto.TileBufConfigAttr.get(
                col_major,
                row_major,
                1024,
                zero_pad,
                ctx,
                compact_mode=pto.CompactMode.RowPlusOne,
            ),
            context=ctx,
        )
        expect_equal(
            str(compact_cfg_ty),
            "!pto.tile_buf<vec, 8x8xf32, blayout=col_major, slayout=row_major, fractal=1024, pad=1, compact=2>",
            "non-default compact suffix print",
        )

        legacy_module = Module.parse(
            """
module {
  func.func @legacy(
      %arg0: !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=128,
                           v_row=16, v_col=1, blayout=col_major,
                           slayout=none_box, fractal=512, pad=0>) {
    return
  }
}
""",
            ctx,
        )
        legacy_text = str(legacy_module)
        expect_contains(
            legacy_text,
            "!pto.tile_buf<vec, 16x128xf32, valid=16x1, blayout=col_major>",
            "legacy parse reprint",
        )
        expect_not_contains(legacy_text, "loc=", "legacy parse reprint")
        expect_not_contains(legacy_text, "v_row=", "legacy parse reprint")
        expect_not_contains(legacy_text, "v_col=", "legacy parse reprint")

        compact_module = Module.parse(
            """
module {
  func.func @compact(
      %arg0: !pto.tile_buf<vec, 16x128xf32, valid=16x1, blayout=col_major>) {
    return
  }
}
""",
            ctx,
        )
        expect_contains(
            str(compact_module),
            "!pto.tile_buf<vec, 16x128xf32, valid=16x1, blayout=col_major>",
            "compact parse roundtrip",
        )

        compact_mode_module = Module.parse(
            """
module {
  func.func @compact_mode(
      %arg0: !pto.tile_buf<vec, 16x128xf32, compact=2, blayout=col_major>) {
    return
  }
}
""",
            ctx,
        )
        expect_contains(
            str(compact_mode_module),
            "!pto.tile_buf<vec, 16x128xf32, blayout=col_major, compact=2>",
            "compact mode parse roundtrip",
        )

        expect_parse_error(
            ctx,
            """
module {
  func.func @dup_valid(
      %arg0: !pto.tile_buf<vec, 16x128xf32, valid=16x1, valid=16x2>) {
    return
  }
}
""",
            "duplicate valid in tile_buf compact syntax",
            "duplicate valid rejection",
        )
        expect_parse_error(
            ctx,
            """
module {
  func.func @dynamic_base(%arg0: !pto.tile_buf<vec, ?x16xf32>) {
    return
  }
}
""",
            "expected static shape",
            "dynamic base shape rejection",
        )
        expect_parse_error(
            ctx,
            """
module {
  func.func @bad_valid_rank(
      %arg0: !pto.tile_buf<vec, 16x128xf32, valid=16x1x2>) {
    return
  }
}
""",
            "tile_buf valid must have exactly two dims",
            "valid rank rejection",
        )

    print("compact_tile_buf_asm: PASS")


if __name__ == "__main__":
    main()
