# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import os
import numpy as np
from ml_dtypes import bfloat16, float4_e2m1fn, float8_e4m3fn

from cases import CASES
from st_common import setup_case_rng


def fp32_to_bf16_storage(data):
    return np.asarray(data, dtype=np.float32).astype(bfloat16)


def exp_scaling_f32(max_values, alg):
    max_values = np.asarray(max_values, dtype=np.float32)
    bits = max_values.view(np.uint32)
    exponent = ((bits & np.uint32(0x7F800000)) >> np.uint32(23)).astype(np.int32)
    mantissa = bits & np.uint32(0x007FFFFF)
    if alg == "ocp":
        exp = np.clip(exponent - 8, 0, 254).astype(np.uint8)
        scale_exp = np.clip(254 - exp.astype(np.int32), 0, 255).astype(np.uint32)
        scaling = (scale_exp << 23).view(np.float32)
        low = exponent <= 8
        exp[low] = 0
        scaling[low] = np.float32(2.0 ** -127)
    else:
        scaled_max = (max_values * np.float32(1.0 / 448.0)).astype(np.float32)
        scaled_bits = scaled_max.view(np.uint32)
        scaled_exp = ((scaled_bits & np.uint32(0x7F800000)) >> np.uint32(23)).astype(np.int32)
        scaled_mantissa = scaled_bits & np.uint32(0x007FFFFF)
        round_up = (scaled_exp > 0) & (scaled_exp < 254) & (scaled_mantissa != 0)
        exp = np.clip(scaled_exp + round_up.astype(np.int32), 0, 254).astype(np.uint8)
        scale_exp = np.clip(254 - exp.astype(np.int32), 0, 255).astype(np.uint32)
        scaling = (scale_exp << 23).view(np.float32)
    return exp, scaling


def exp_scaling_fp4_bf16(max_values):
    bits = np.asarray(max_values, dtype=np.float32).view(np.uint32)
    bf16_bits = (bits >> np.uint32(16)).astype(np.uint16)
    exp_bits = bf16_bits & np.uint16(0x7F80)
    clamped = np.maximum(exp_bits, np.uint16(0x0100))
    shared = clamped - np.uint16(0x0100)
    exp = ((shared >> np.uint16(7)) & np.uint16(0xFF)).astype(np.uint8)
    scaling_bits = np.uint16(0x7F00) - shared
    scaling = (scaling_bits.astype(np.uint32) << np.uint32(16)).view(np.float32)
    return exp, scaling


def fp4_pack(values):
    codes = np.asarray(values).astype(bfloat16).astype(float4_e2m1fn).view(np.uint8)
    return ((codes[:, 0::2] & 0x0F) | ((codes[:, 1::2] & 0x0F) << 4)).reshape(-1)


def dn_to_zz(exp):
    rows, cols = exp.shape
    return exp.reshape(rows // 2, 2, cols // 16, 16).transpose(2, 0, 3, 1).reshape(-1)


def nd_to_zz(exp):
    rows, cols = exp.shape
    padded_rows = (rows + 15) // 16 * 16
    padded = np.zeros((padded_rows, cols), dtype=np.uint8)
    padded[:rows, :] = exp
    return padded.reshape(padded_rows // 16, 16, cols // 2, 2).transpose(0, 2, 1, 3).reshape(-1)


def generate(case):
    m, n = case["m"], case["n"]
    raw = np.random.uniform(-10, 10, size=(m, n)).astype(np.float32)
    src = fp32_to_bf16_storage(raw) if case["src_type"] == "bf16" else raw
    src_f32 = src.astype(np.float32)
    if case["grp_axis"] == 0:
        group_max = np.max(np.abs(src_f32.reshape(m // 32, 32, n)), axis=1)
    else:
        group_max = np.max(np.abs(src_f32.reshape(m, n // 32, 32)), axis=2)
    if case["fp4"]:
        exp, scaling_f32 = exp_scaling_fp4_bf16(group_max)
        scaling = fp32_to_bf16_storage(scaling_f32)
        scaled = (src.astype(bfloat16) * np.repeat(scaling.astype(bfloat16), 32, axis=0)).astype(bfloat16).astype(np.float32)
        dst = fp4_pack(scaled)
    else:
        exp, scaling_f32 = exp_scaling_f32(group_max, case["alg"])
        scaling = scaling_f32.astype(np.float32)
        if case["grp_axis"] == 0:
            scaled = src_f32 * np.repeat(scaling_f32, 32, axis=0)
        else:
            scaled = src_f32 * np.repeat(scaling_f32, 32, axis=1)
        dst = np.clip(scaled, -448, 448).astype(float8_e4m3fn).view(np.uint8).reshape(-1)

    if case["grp_axis"] == 0:
        exp_zz = dn_to_zz(exp)
    else:
        exp_zz = nd_to_zz(exp)
    return src, dst, exp.reshape(-1), group_max.reshape(-1), scaling.reshape(-1), exp_zz


for case in CASES:
    setup_case_rng(case)
    case_dir = case["name"]
    os.makedirs(case_dir, exist_ok=True)
    src, dst, exp, max_values, scaling, exp_zz = generate(case)
    src.tofile(os.path.join(case_dir, "input.bin"))
    dst.tofile(os.path.join(case_dir, "golden_dst.bin"))
    exp.tofile(os.path.join(case_dir, "golden_exp.bin"))
    max_values.astype(src.dtype).tofile(os.path.join(case_dir, "golden_max.bin"))
    scaling.astype(src.dtype).tofile(os.path.join(case_dir, "golden_scaling.bin"))
    exp_zz.tofile(os.path.join(case_dir, "golden_exp_zz.bin"))
    print(f"[INFO] generated {case_dir}: {case['m']}x{case['n']} {case['src_type']} {case['alg']}")
