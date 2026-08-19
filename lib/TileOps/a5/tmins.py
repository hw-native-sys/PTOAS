# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""PTODSL TileLib template for pto.tmins."""

from ptodsl import pto

from ._common import same_dtype_signatures
from ._elementwise import register_scalar_binary


_DTYPES = same_dtype_signatures(3)


template_tmins = register_scalar_binary(
    op="pto.tmins",
    name="template_tmins",
    vector_op=pto.vmins,
    dtypes=_DTYPES,
)

template_tmins_1d = register_scalar_binary(
    op="pto.tmins",
    name="template_tmins_1d",
    vector_op=pto.vmins,
    dtypes=_DTYPES,
    traversal="1d",
)


from ._vmi_common import (  # noqa: E402
    _vmins as _vmi_vmins,
    bf16,
    canonical_vmi_template,
    emit_elementwise_vmi,
    f16,
    f32,
)


@canonical_vmi_template(
    target="a5",
    op="tmins",
    name="vmi_tmins",
    dtypes=(("f32", "f32", "f32"),),
)
def vmi_tmins(src: pto.Tile, scalar: f32, dst: pto.Tile):
    emit_elementwise_vmi(
        dst,
        (src,),
        lambda values, mask: _vmi_vmins(values[0], scalar, mask),
        allowed_dtypes=(f32,),
    )


# Per-dtype vector-scalar candidates (texpand pattern). Float-only: A5 vmins
# lowering rewrites signed-int vmin to `pto.vmi.minf` (float-only), so int
# tmins fails at VMI lowering (same root cause as tcolmin int). bf16/f16/f32
# are supported. See ADR-0003 PR2.


@canonical_vmi_template(
    target="a5",
    op="tmins",
    name="vmi_tmins_f16",
    dtypes=(("f16", "f16", "f16"),),
)
def vmi_tmins_f16(src: pto.Tile, scalar: f16, dst: pto.Tile):
    emit_elementwise_vmi(
        dst,
        (src,),
        lambda values, mask: _vmi_vmins(values[0], scalar, mask),
        allowed_dtypes=(f16,),
    )


@canonical_vmi_template(
    target="a5",
    op="tmins",
    name="vmi_tmins_bf16",
    dtypes=(("bf16", "bf16", "bf16"),),
)
def vmi_tmins_bf16(src: pto.Tile, scalar: bf16, dst: pto.Tile):
    emit_elementwise_vmi(
        dst,
        (src,),
        lambda values, mask: _vmi_vmins(values[0], scalar, mask),
        allowed_dtypes=(bf16,),
    )
