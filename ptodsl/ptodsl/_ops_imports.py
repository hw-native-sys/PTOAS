# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Shared import surface for the PTODSL op modules.

``_ops_common``, ``_ops_core`` and ``_ops_vmem`` all depend on the same set of
diagnostics, scalar helpers, surface-value wrappers, type constructors and MLIR
dialect/IR symbols. Consolidating those imports here lets each sibling module
star-import a single, authoritative block instead of repeating the identical
list (which the duplicate-code checker rightly flagged). ``__all__`` defines the
re-export surface so ``from ._ops_imports import *`` also brings in the
underscore-prefixed helpers.
"""

from __future__ import annotations

from functools import wraps
import warnings
from ._diagnostics import (
    PTODSLDeprecationWarning,
    explicit_mode_required_with_context_error,
    make_tensor_view_invalid_layout_error,
    make_tensor_view_missing_metadata_error,
    tile_row_alignment_error,
)
from ._host_tensors import resolve_tensor_data_entry
from ._scalar_coercion import coerce_scalar_to_type, materialize_scalar_literal
from ._scalar_adaptation import (
    classify_runtime_scalar_type,
    coerce_runtime_i1_value,
    coerce_runtime_index_value,
    coerce_runtime_integer_value,
)
from ._runtime_scalar_ops import emit_runtime_binary_op
from ._surface_values import (
    AllocatedBufferValue,
    MaskResultValue,
    PartitionTensorViewValue,
    TensorViewValue,
    TileSliceValue,
    TileValue,
    _coerce_index_value,
    _static_index_dims,
    _unwrap_sequence,
    compose_partition_spec,
    emit_as_ptr,
    infer_tile_element_type,
    is_runtime_scalar_ir_type,
    parse_tile_type_metadata,
    resolve_address_access,
    unwrap_surface_value,
    wrap_surface_value,
)
from ._types import (
    _is_struct_type,
    _isinstance_pto_type,
    _materialize_integer_literal,
    _normalize_address_space,
    _resolve,
    _strip_integer_signedness,
    mask_type,
    part_tensor_view_type,
    part_tensor_view_type_from_dims,
    ptr,
    tensor_view_type,
    tensor_view_type_from_dims,
    vreg_type,
)
from ptoas.mlir.dialects import arith, pto as _pto
from ptoas.mlir.ir import (
    Attribute,
    BF16Type,
    F16Type,
    F32Type,
    Float8E4M3FNType,
    Float8E5M2Type,
    FloatAttr,
    IndexType,
    IntegerAttr,
    IntegerType,
    MemRefType,
    Operation,
    Type,
    TypeAttr,
    UnitAttr,
    VectorType,
)

__all__ = [
    "wraps",
    "warnings",
    "PTODSLDeprecationWarning",
    "explicit_mode_required_with_context_error",
    "make_tensor_view_invalid_layout_error",
    "make_tensor_view_missing_metadata_error",
    "tile_row_alignment_error",
    "resolve_tensor_data_entry",
    "coerce_scalar_to_type",
    "materialize_scalar_literal",
    "classify_runtime_scalar_type",
    "coerce_runtime_i1_value",
    "coerce_runtime_index_value",
    "coerce_runtime_integer_value",
    "emit_runtime_binary_op",
    "AllocatedBufferValue",
    "MaskResultValue",
    "PartitionTensorViewValue",
    "TensorViewValue",
    "TileSliceValue",
    "TileValue",
    "_coerce_index_value",
    "_static_index_dims",
    "_unwrap_sequence",
    "compose_partition_spec",
    "emit_as_ptr",
    "infer_tile_element_type",
    "is_runtime_scalar_ir_type",
    "parse_tile_type_metadata",
    "resolve_address_access",
    "unwrap_surface_value",
    "wrap_surface_value",
    "_is_struct_type",
    "_isinstance_pto_type",
    "_materialize_integer_literal",
    "_normalize_address_space",
    "_resolve",
    "_strip_integer_signedness",
    "mask_type",
    "part_tensor_view_type",
    "part_tensor_view_type_from_dims",
    "ptr",
    "tensor_view_type",
    "tensor_view_type_from_dims",
    "vreg_type",
    "arith",
    "_pto",
    "Attribute",
    "BF16Type",
    "F16Type",
    "F32Type",
    "Float8E4M3FNType",
    "Float8E5M2Type",
    "FloatAttr",
    "IndexType",
    "IntegerAttr",
    "IntegerType",
    "MemRefType",
    "Operation",
    "Type",
    "TypeAttr",
    "UnitAttr",
    "VectorType",
]
