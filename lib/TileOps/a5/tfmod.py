# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""PTODSL TileLib template for pto.tfmod."""

from ._remainder import FMOD_DTYPES, register_binary_remainder


template_tfmod = register_binary_remainder(
    op="pto.tfmod",
    name="template_tfmod",
    dtypes=FMOD_DTYPES,
    round_mode="Z",
)


template_tfmod_1d = register_binary_remainder(
    op="pto.tfmod",
    name="template_tfmod_1d",
    dtypes=FMOD_DTYPES,
    round_mode="Z",
    traversal="1d",
)
