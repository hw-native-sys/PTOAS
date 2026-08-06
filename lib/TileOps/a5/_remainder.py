# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Shared PTODSL implementations for remainder/fmod-style TileOps."""

from ptodsl import pto
import ptodsl.tilelib as tilelib

from ._common import ub_row_major_constraints
from ._elementwise import (
    emit_binary_1d,
    emit_binary_2d,
    emit_scalar_binary_1d,
    emit_scalar_binary_2d,
    traversal_metadata,
)


FMOD_DTYPES = [("f32", "f32", "f32"), ("f16", "f16", "f16"), ("i16", "i16", "i16"), ("ui16", "ui16", "ui16")]
FMODS_DTYPES = [("f32", "f32", "f32"), ("f16", "f16", "f16"), ("i32", "i32", "i32"), ("i16", "i16", "i16")]
REM_DTYPES = [("f32", "f32", "f32", "f32"), ("f16", "f16", "f16", "f16"), ("i32", "i32", "i32", "i32")]
REMS_DTYPES = [("f32", "f32", "f32", "f32"), ("f16", "f16", "f16", "f16")]


def _remainder(lhs, rhs, mask, *, round_mode, dtype):
    quotient = pto.vdiv(lhs, rhs, mask)
    if str(dtype) in {"f16", "bf16", "f32"}:
        quotient = pto.vtrc(quotient, mask, rnd=round_mode)
    product = pto.vmul(quotient, rhs, mask)
    result = pto.vsub(lhs, product, mask)
    if str(dtype) == "f32":
        sign_diff_mask = pto.vcmps(
            pto.vmul(rhs, result, mask), pto.f32(0.0), mask, pto.CmpMode.LT
        )
        corrected = pto.vadd(result, rhs, sign_diff_mask)
        result = pto.vsel(corrected, result, sign_diff_mask)
    return result


def _scalar_remainder(lhs, scalar, mask, *, round_mode, dtype):
    scalar_vec = pto.vbr(scalar)
    quotient = pto.vdiv(lhs, scalar_vec, mask)
    if str(dtype) in {"f16", "bf16", "f32"}:
        quotient = pto.vtrc(quotient, mask, rnd=round_mode)
    product = pto.vmuls(quotient, scalar, mask)
    return pto.vsub(lhs, product, mask)


def register_binary_remainder(*, op, name, dtypes, round_mode, has_tmp=False,
                              traversal="2d", priority=None,
                              candidate_id=None):
    if traversal not in {"1d", "2d"}:
        raise ValueError(
            f"unsupported remainder traversal {traversal!r}; "
            "expected '1d' or '2d'"
        )
    loop_depth, priority, candidate_id = traversal_metadata(
        traversal,
        priority=priority,
        candidate_id=candidate_id,
    )
    constraints = ub_row_major_constraints("src0", "src1", "dst")
    if has_tmp:
        constraints = ub_row_major_constraints("src0", "src1", "tmp", "dst")
        if traversal == "1d":
            constraints.append(
                tilelib.require_elementwise_1d(
                    "src0",
                    "src1",
                    "tmp",
                    "dst",
                )
            )

        @tilelib.tile_template(
            op=op,
            target="a5",
            name=name,
            dtypes=dtypes,
            iteration_axis="none",
            op_engine="vector",
            op_class="elementwise",
            constraints=constraints,
            priority=priority,
            id=candidate_id,
            loop_depth=loop_depth,
            is_post_update=False,
            tags=("elementwise", "remainder"),
        )
        def template(src0: pto.Tile, src1: pto.Tile, tmp: pto.Tile, dst: pto.Tile):
            _ = tmp
            _emit_binary(src0, src1, dst, round_mode, traversal)

        return template

    if traversal == "1d":
        constraints.append(
            tilelib.require_elementwise_1d("src0", "src1", "dst")
        )

    @tilelib.tile_template(
        op=op,
        target="a5",
        name=name,
        dtypes=dtypes,
        iteration_axis="none",
        op_engine="vector",
        op_class="elementwise",
        constraints=constraints,
        priority=priority,
        id=candidate_id,
        loop_depth=loop_depth,
        is_post_update=False,
        tags=("elementwise", "remainder"),
    )
    def template(src0: pto.Tile, src1: pto.Tile, dst: pto.Tile):
        _emit_binary(src0, src1, dst, round_mode, traversal)

    return template


def _emit_binary(src0, src1, dst, round_mode, traversal):
    dtype = dst.dtype

    def remainder(lhs, rhs, mask):
        return _remainder(
            lhs,
            rhs,
            mask,
            round_mode=round_mode,
            dtype=dtype,
        )

    if traversal == "1d":
        emit_binary_1d(src0, src1, dst, remainder)
    else:
        emit_binary_2d(src0, src1, dst, remainder)


def register_scalar_remainder(*, op, name, dtypes, round_mode, has_tmp=False,
                              traversal="2d", priority=None,
                              candidate_id=None):
    if traversal not in {"1d", "2d"}:
        raise ValueError(
            f"unsupported remainder traversal {traversal!r}; "
            "expected '1d' or '2d'"
        )
    loop_depth, priority, candidate_id = traversal_metadata(
        traversal,
        priority=priority,
        candidate_id=candidate_id,
    )
    constraints = ub_row_major_constraints("src", "dst")
    if has_tmp:
        constraints = ub_row_major_constraints("src", "tmp", "dst")
        if traversal == "1d":
            constraints.append(
                tilelib.require_elementwise_1d("src", "tmp", "dst")
            )

        @tilelib.tile_template(
            op=op,
            target="a5",
            name=name,
            dtypes=dtypes,
            iteration_axis="none",
            op_engine="vector",
            op_class="elementwise",
            constraints=constraints,
            priority=priority,
            id=candidate_id,
            loop_depth=loop_depth,
            is_post_update=False,
            tags=("elementwise", "scalar", "remainder"),
        )
        def template(src: pto.Tile, scalar, tmp: pto.Tile, dst: pto.Tile):
            _ = tmp
            _emit_scalar(src, scalar, dst, round_mode, traversal)

        return template

    if traversal == "1d":
        constraints.append(tilelib.require_elementwise_1d("src", "dst"))

    @tilelib.tile_template(
        op=op,
        target="a5",
        name=name,
        dtypes=dtypes,
        iteration_axis="none",
        op_engine="vector",
        op_class="elementwise",
        constraints=constraints,
        priority=priority,
        id=candidate_id,
        loop_depth=loop_depth,
        is_post_update=False,
        tags=("elementwise", "scalar", "remainder"),
    )
    def template(src: pto.Tile, scalar, dst: pto.Tile):
        _emit_scalar(src, scalar, dst, round_mode, traversal)

    return template


def _emit_scalar(src, scalar, dst, round_mode, traversal):
    dtype = dst.dtype

    def remainder(value, scalar_value, mask):
        return _scalar_remainder(
            value,
            scalar_value,
            mask,
            round_mode=round_mode,
            dtype=dtype,
        )

    if traversal == "1d":
        emit_scalar_binary_1d(src, scalar, dst, remainder)
    else:
        emit_scalar_binary_2d(src, scalar, dst, remainder)


__all__ = [
    "FMOD_DTYPES",
    "FMODS_DTYPES",
    "REM_DTYPES",
    "REMS_DTYPES",
    "register_binary_remainder",
    "register_scalar_remainder",
]
