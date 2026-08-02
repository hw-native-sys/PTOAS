# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""PTODSL TileLib: ptodsl-native templates for ExpandTileOp (migration of tilelang-dsl).

Layers:
  - metadata     : TileSpec + dtypes + TemplateMetadata
  - decorator    : @tile_template (registers a version + its metadata)
  - registry     : constraint/priority selection among registered versions
  - render       : render_best(op, target, specs) + CLI  (the ExpandTileOp seam)
  - templates/   : the ported per-arch template bodies

Template bodies use the public ``from ptodsl import pto`` operation surface.
"""

from .constraints import (
    BLayout,
    SLayout,
    check_layout,
    check_memory_space,
    check_s_layout,
    check_type,
    require_contiguous,
    require_conversion_1d,
    require_elementwise_1d,
    require_predicate_compare_1d,
    require_predicate_select_1d,
    require_same_valid_shape,
    require_valid_rows,
)
from .decorator import SpecializedTileTemplate, TileTemplate, tile_template
from .metadata import (
    ScalarSpec,
    ScalarType,
    TemplateMetadata,
    TileSpec,
    VectorSpec,
    ViewSpec,
    bf16,
    f4e1m2x2,
    f4e2m1x2,
    f16,
    f32,
    f8e4m3,
    f8e5m2,
    hif8,
    i8,
    i16,
    i32,
    i64,
    si8,
    si16,
    si32,
    si64,
    ui8,
    ui16,
    ui32,
    ui64,
)
from .registry import (
    AmbiguousTemplate,
    NoMatchingTemplate,
    TileTemplateRegistry,
    default_registry,
    legal_candidates,
    register,
    select,
)
from .render import render_best, select_and_specialize

__all__ = [
    # template registration
    "tile_template",
    # specs / metadata
    "TileSpec",
    "ScalarSpec",
    "ViewSpec",
    "VectorSpec",
    "ScalarType",
    "TemplateMetadata",
    "BLayout",
    "SLayout",
    "check_type",
    "check_memory_space",
    "check_layout",
    "check_s_layout",
    "require_contiguous",
    "require_conversion_1d",
    "require_elementwise_1d",
    "require_predicate_compare_1d",
    "require_predicate_select_1d",
    "require_same_valid_shape",
    "require_valid_rows",
    "f32",
    "f16",
    "bf16",
    "f8e4m3",
    "f8e5m2",
    "hif8",
    "f4e1m2x2",
    "f4e2m1x2",
    "i32",
    "i16",
    "i8",
    "i64",
    "si32",
    "si16",
    "si8",
    "si64",
    "ui32",
    "ui16",
    "ui8",
    "ui64",
    # descriptors
    "TileTemplate",
    "SpecializedTileTemplate",
    # registry / selection
    "TileTemplateRegistry",
    "default_registry",
    "legal_candidates",
    "register",
    "select",
    "NoMatchingTemplate",
    "AmbiguousTemplate",
    # rendering
    "render_best",
    "select_and_specialize",
]
