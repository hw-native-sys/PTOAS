# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""PTODSL TileLib template for pto.tabs."""

from ptodsl import pto

from ._elementwise import register_unary


_DTYPES = [("f16", "f16"), ("f32", "f32")]


template_tabs = register_unary(
    op="pto.tabs",
    name="template_tabs",
    vector_op=pto.vabs,
    dtypes=_DTYPES,
)


template_tabs_1d = register_unary(
    op="pto.tabs",
    name="template_tabs_1d",
    vector_op=pto.vabs,
    dtypes=_DTYPES,
    traversal="1d",
)


from ._vmi_common import (  # noqa: E402
    _abs as _vmi_abs,
    canonical_vmi_template,
    emit_elementwise_vmi,
    f16,
    f32,
)


# Note: bf16 is intentionally not in the VMI tabs candidate. The ordinary
# template_tabs above only covers f16/f32 (bf16 vabs is not validated on A5);
# bf16 tabs conservatively falls back to the ordinary PTODSL path per ADR-0003
# (未验完保守回退). allowed_dtypes is pinned to (f32, f16) to match.
@canonical_vmi_template(
    target="a5",
    op="tabs",
    name="vmi_tabs",
    dtypes=(
        ("f32", "f32"),
        ("f16", "f16"),
    ),
)
def vmi_tabs(src: pto.Tile, dst: pto.Tile):
    emit_elementwise_vmi(dst, (src,), _vmi_abs, allowed_dtypes=(f32, f16))
