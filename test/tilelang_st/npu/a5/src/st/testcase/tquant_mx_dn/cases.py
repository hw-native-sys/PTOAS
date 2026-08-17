# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Single source of truth for the supported MX TQUANT + exponent TMOV cases."""

FULL_CHAIN_CASES = [
    {"name": "dn_fp8_ocp_64x64", "m": 64, "n": 64, "grp_axis": 0, "alg": "ocp", "src_type": "f32", "fp4": False, "eps": 0.0},
    {"name": "dn_fp8_ocp_128x32", "m": 128, "n": 32, "grp_axis": 0, "alg": "ocp", "src_type": "f32", "fp4": False, "eps": 0.0},
    {"name": "dn_fp8_nv_64x64", "m": 64, "n": 64, "grp_axis": 0, "alg": "nv", "src_type": "f32", "fp4": False, "eps": 0.0},
    {"name": "dn_fp4_bf16_ocp_64x64", "m": 64, "n": 64, "grp_axis": 0, "alg": "ocp", "src_type": "bf16", "fp4": True, "eps": 0.0},
    {"name": "dn_fp8_ocp_64x96", "m": 64, "n": 96, "grp_axis": 0, "alg": "ocp", "src_type": "f32", "fp4": False, "eps": 0.0},
    {"name": "nd_fp8_f32_ocp_16x128", "m": 16, "n": 128, "grp_axis": 1, "alg": "ocp", "src_type": "f32", "fp4": False, "eps": 0.0},
    {"name": "nd_fp8_f32_ocp_20x128", "m": 20, "n": 128, "grp_axis": 1, "alg": "ocp", "src_type": "f32", "fp4": False, "eps": 0.0},
]

# The shared TileLang ST runner expects this conventional name and may replace
# it with a filtered list for ``-c`` invocations.
CASES = FULL_CHAIN_CASES
