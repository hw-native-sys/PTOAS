# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Shared TileLib constraints for row and column arg-reductions."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ArgReductionConstraintContext:
    """Operand metadata used to select an arg-reduction template."""

    src_config: object
    tmp_config: object
    dst_idx_config: object
    src_dtype: object
    tmp_dtype: object
    src_valid_shape: object
    dst_idx_valid_shape: object
    dst_idx_dtype: object = None
    dst_val_config: object = None
    dst_val_dtype: object = None
    dst_val_valid_shape: object = None

    @classmethod
    def from_kwargs(cls, kwargs):
        return cls(
            src_config=kwargs.get("src_config"),
            tmp_config=kwargs.get("tmp_config"),
            dst_idx_config=kwargs.get("dst_idx_config"),
            src_dtype=kwargs.get("src_dtype"),
            tmp_dtype=kwargs.get("tmp_dtype"),
            src_valid_shape=kwargs.get("src_valid_shape"),
            dst_idx_valid_shape=kwargs.get("dst_idx_valid_shape"),
            dst_idx_dtype=kwargs.get("dst_idx_dtype"),
            dst_val_config=kwargs.get("dst_val_config"),
            dst_val_dtype=kwargs.get("dst_val_dtype"),
            dst_val_valid_shape=kwargs.get("dst_val_valid_shape"),
        )


def _known_eq(lhs, rhs):
    return lhs is None or rhs is None or lhs == rhs


def _shape_matches(lhs, rhs):
    return (
        lhs is not None
        and rhs is not None
        and len(lhs) == len(rhs)
        and all(_known_eq(lhs_dim, rhs_dim) for lhs_dim, rhs_dim in zip(lhs, rhs))
    )


def _uses_supported_layout(config):
    return (
        config is not None
        and getattr(config, "b_layout", None) == "row_major"
        and getattr(config, "s_layout", None) == "none_box"
    )


def _common_constraints(context, axis):
    if not all(
        _uses_supported_layout(config)
        for config in (context.src_config, context.tmp_config, context.dst_idx_config)
    ):
        return False
    if context.src_dtype != context.tmp_dtype:
        return False

    src_shape = context.src_valid_shape
    idx_shape = context.dst_idx_valid_shape
    if src_shape is None or idx_shape is None:
        return False
    if len(src_shape) != 2 or len(idx_shape) != 2:
        return False
    if axis == "row":
        return _known_eq(src_shape[0], idx_shape[0]) and idx_shape[1] == 1
    if axis == "col":
        return idx_shape[0] == 1 and _known_eq(src_shape[1], idx_shape[1])
    return False


def index_output_constraints(context, axis, index_dtypes):
    return _common_constraints(context, axis) and context.dst_idx_dtype in index_dtypes


def value_output_constraints(
    context, axis, index_dtypes, source_dtypes=None
):
    if not _common_constraints(context, axis):
        return False
    return (
        _uses_supported_layout(context.dst_val_config)
        and context.dst_idx_dtype in index_dtypes
        and context.dst_val_dtype == context.src_dtype
        and (source_dtypes is None or context.src_dtype in source_dtypes)
        and _shape_matches(context.dst_idx_valid_shape, context.dst_val_valid_shape)
    )
