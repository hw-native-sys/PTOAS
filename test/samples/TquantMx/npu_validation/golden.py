#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Board golden generator for the axis1 MXFP8 plus ND-to-ZZ sample.

Replicates the OCP MXFP8 quantization contract implemented by pto-isa
TQuant.hpp (AbsReduceMax -> ExtractB8ExponentAndScaling -> CalcQuantizedFP8Values):

  src f32 [M, K] (group_size=32 along K)
    -> per-group absmax
    -> e8m0 shared exponent (emax=8 for e4m3)
    -> fp32 reciprocal scaling = 2^(254 - e8m0)
    -> fp8 e4m3 output = clamp(src * scaling, -448, 448)

Auxiliary tiles:
  exp     : ui8  [M, K/32]         (e8m0 exponent per group)
  max     : f32  [M, K/32]         (per-group absmax)
  scaling : f32  [M, K/32]         (reciprocal scale per group)
"""

import numpy as np

M = 16
K = 64
GROUP_SIZE = 32
EMAX = 8  # e4m3 max exponent
GROUP_COLS = K // GROUP_SIZE


def fp32_to_fp8e4m3fn_bytes(arr):
    """Convert fp32 array to fp8 e4m3fn packed as int8 bytes (round-to-nearest-even).

    fp8 e4m3fn: sign(1) | exp(4, bias 7) | mantissa(3). No Inf; 0x7F is NaN.
    Max normal = 0x7E (448). Subnormals: exp_field=0, value = mantissa * 2^(-6).
    Matches OCP e4m3 cast (vcvt fp32->fp8, round-to-nearest) used by pto-isa.
    """
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    u32 = arr.view(np.uint32)
    sign = ((u32 >> np.uint32(31)) & np.uint32(1)).astype(np.int32)
    exp_f32 = ((u32 >> np.uint32(23)) & np.uint32(0xFF)).astype(np.int32)
    mant_f32 = (u32 & np.uint32(0x7FFFFF)).astype(np.int32)

    result = np.zeros(arr.shape, dtype=np.int32)

    # Finite non-zero: compute unbiased exponent of fp32 value.
    # For normals (exp_f32 in 1..254): value = 1.mantissa * 2^(exp_f32-127)
    # For subnormals (exp_f32==0): value = 0.mantissa * 2^(-126)
    is_normal = (exp_f32 >= 1) & (exp_f32 <= 254)
    is_subnormal_f32 = (exp_f32 == 0) & (mant_f32 != 0)

    # For fp32 normals: unbiased exp = exp_f32 - 127. Target fp8 exp bias = 7,
    # so fp8 biased exp = (exp_f32 - 127) + 7 = exp_f32 - 120.
    fp8_exp = exp_f32 - 120  # for normals

    # For fp32 subnormals: value = mant * 2^(-149). fp8 exp would be very negative -> 0.
    fp8_exp = np.where(is_subnormal_f32, -999, fp8_exp)

    # Overflow (fp8 exp_field > 15): clamp to max normal 0x7E (448).
    # Note: e4m3fn has no Inf; exp_field 0..15 are all valid. Max normal = 0x7E (448).
    overflow = is_normal & (fp8_exp > 15)
    result = np.where(overflow, (sign << np.int32(7)) | np.int32(0x7E), result)

    # Underflow (fp8_exp < 0 for normals, or subnormal): 0 (ignore subnormal fp8 for simplicity,
    # since source range [-10,10]*scaling rarely produces fp8 subnormals).
    inrange = is_normal & (fp8_exp >= 0) & (fp8_exp <= 15)

    # For in-range normals: mantissa fp32 (23 bits, with implicit 1) -> fp8 (3 bits).
    # Value = 1.mantissa_f32 * 2^(fp8_exp - 7). Round 23-bit mantissa to 3 bits RNE.
    # Top 3 mantissa bits + rounding.
    m_high = mant_f32 >> np.int32(20)  # top 3 bits of mantissa (0..7)
    round_bit = (mant_f32 >> np.int32(19)) & np.int32(1)
    sticky = np.where((mant_f32 & np.int32(0x7FFFF)) != 0, np.int32(1), np.int32(0))
    round_up = (round_bit & (sticky | (m_high & np.int32(1)))).astype(bool)
    m_rounded = m_high + np.where(round_up, np.int32(1), np.int32(0))
    # Carry: m_rounded == 8 -> exp+1, mantissa 0.
    carry = (m_rounded >> np.int32(3)) & np.int32(1)
    m_rounded = m_rounded & np.int32(0x7)
    fp8_exp_r = fp8_exp + carry
    # After carry, could overflow to exp=16 -> clamp to max normal.
    over_after_carry = inrange & (fp8_exp_r > 15)
    result = np.where(over_after_carry, (sign << np.int32(7)) | np.int32(0x7E), result)
    inrange = inrange & (fp8_exp_r <= 15)
    assembled = (sign << np.int32(7)) | (fp8_exp_r << np.int32(3)) | m_rounded
    result = np.where(inrange, assembled, result)

    # NaN: fp32 NaN -> fp8 NaN (0x7F). Inf -> max normal (e4m3fn has no Inf).
    is_nan = (exp_f32 == 0xFF) & (mant_f32 != 0)
    result = np.where(is_nan, np.int32(0x7F), result)
    is_inf = (exp_f32 == 0xFF) & (mant_f32 == 0)
    result = np.where(is_inf, (sign << np.int32(7)) | np.int32(0x7E), result)

    # Zero (fp32 zero): 0.
    return result.astype(np.uint8).view(np.int8)


def fp32_to_fp8_element(data_abs_max, emax=EMAX):
    bits = np.uint32(np.frombuffer(np.float32(data_abs_max).tobytes(), dtype=np.uint32)[0])
    exponent_b32 = int((bits & np.uint32(0x7F800000)) >> np.uint32(23))
    mantissa_b32 = int(bits & np.uint32(0x007FFFFF))
    if exponent_b32 == 0xFF and mantissa_b32 != 0:
        return 0xFF, np.uint32(0x7FC00000).view(np.float32)
    if exponent_b32 <= emax:
        return 0x00, np.uint32(0x7F000000).view(np.float32)
    e8m0 = exponent_b32 - emax
    scale_exp = 254 - e8m0
    scaling = np.uint32(scale_exp << 23).view(np.float32)
    if scaling == 0.0:
        scaling = np.float32(2.0 ** -127)
    return e8m0, scaling


def nd_to_zz(exp):
    rows, cols = exp.shape
    padded_rows = ((rows + 15) // 16) * 16
    padded = np.zeros((padded_rows, cols), dtype=np.uint8)
    padded[:rows, :] = exp
    return padded.reshape(padded_rows // 16, 16, cols // 2, 2).transpose(0, 2, 1, 3).reshape(-1)


def main():
    np.random.seed(23)
    src = np.random.uniform(-10, 10, [M, K]).astype(np.float32)
    src.tofile("input.bin")

    # Per-group absmax along last dim (group_size=32).
    src_abs = np.abs(src)
    group_max = np.max(src_abs.reshape(M, GROUP_COLS, GROUP_SIZE), axis=2)

    e8m0s = []
    scalings = []
    for v in group_max.reshape(-1).tolist():
        e, s = fp32_to_fp8_element(v, emax=EMAX)
        e8m0s.append(e)
        scalings.append(s)
    e8m0 = np.array(e8m0s).astype(np.uint8)
    scaling_per_group = np.array(scalings).astype(np.float32)

    # exp tile: canonical ui8 [M, K/32].
    exp_tile = e8m0.reshape(M, GROUP_COLS)
    exp_tile.tofile("exp.bin")

    max_tile = group_max.reshape(M, GROUP_COLS).astype(np.float32)
    max_tile.tofile("max.bin")

    scaling_tile = scaling_per_group.reshape(M, GROUP_COLS).astype(np.float32)
    scaling_tile.tofile("scaling.bin")

    nd_to_zz(exp_tile).tofile("exp_zz.bin")

    # dst: fp8 e4m3 = clamp(src * scaling, -448, 448), packed as int8 bytes.
    # Pure-numpy fp32 -> fp8 e4m3fn (round-to-nearest-even), no ml_dtypes needed.
    scaling_broadcast = np.repeat(scaling_tile, GROUP_SIZE, axis=1).astype(np.float32)
    scaled = src.astype(np.float64) * scaling_broadcast.astype(np.float64)
    scaled = np.clip(scaled, -448.0, 448.0).astype(np.float32)
    dst_bytes = fp32_to_fp8e4m3fn_bytes(scaled)
    dst_bytes.tofile("golden.bin")


if __name__ == "__main__":
    main()
