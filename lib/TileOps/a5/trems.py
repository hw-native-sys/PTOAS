# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""PTODSL TileLib template for pto.trems."""

from ._remainder import REMS_DTYPES, register_scalar_remainder


template_trems = register_scalar_remainder(
    op="pto.trems",
    name="template_trems",
    dtypes=REMS_DTYPES,
    round_mode="F",
    has_tmp=True,
)

template_trems_1d = register_scalar_remainder(
    op="pto.trems",
    name="template_trems_1d",
    dtypes=REMS_DTYPES,
    round_mode="F",
    has_tmp=True,
    traversal="1d",
)
