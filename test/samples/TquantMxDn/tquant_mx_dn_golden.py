#!/usr/bin/python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Reference for DN MXFP8 quantization followed by DN-to-ZZ exponent packing."""

from pathlib import Path
import sys
import numpy as np

for search_root in (Path(__file__).resolve().parent, Path(__file__).resolve().parents[1]):
    if (search_root / "validation_runtime.py").is_file():
        sys.path.insert(0, str(search_root))
        break

from validation_runtime import default_buffers, load_case_meta, rng, write_buffers, write_golden

M, N, GROUP = 64, 64, 32
GROUP_ROWS = M // GROUP


def fp8_bytes(values):
    values = np.ascontiguousarray(values, dtype=np.float32)
    bits = values.view(np.uint32)
    sign = ((bits >> np.uint32(31)) & 1).astype(np.int32)
    exp = ((bits >> np.uint32(23)) & 0xFF).astype(np.int32)
    mant = (bits & 0x7FFFFF).astype(np.int32)
    out = np.zeros(values.shape, dtype=np.int32)
    normal = (exp >= 1) & (exp <= 254)
    fp8_exp = exp - 120
    overflow = normal & (fp8_exp > 15)
    out = np.where(overflow, (sign << 7) | 0x7E, out)
    inrange = normal & (fp8_exp >= 0) & (fp8_exp <= 15)
    high = mant >> 20
    round_bit = (mant >> 19) & 1
    sticky = ((mant & 0x7FFFF) != 0).astype(np.int32)
    rounded = high + ((round_bit & (sticky | (high & 1))) != 0)
    carry = rounded >> 3
    rounded &= 7
    exp_rounded = fp8_exp + carry
    inrange &= exp_rounded <= 15
    out = np.where(inrange, (sign << 7) | (exp_rounded << 3) | rounded, out)
    out = np.where((exp == 0xFF) & (mant != 0), 0x7F, out)
    out = np.where((exp == 0xFF) & (mant == 0), (sign << 7) | 0x7E, out)
    return out.astype(np.uint8)


def ocp_exp_scaling(max_values):
    bits = np.asarray(max_values, dtype=np.float32).view(np.uint32)
    exponent = ((bits & np.uint32(0x7F800000)) >> np.uint32(23)).astype(np.int32)
    e8m0 = np.clip(exponent - 8, 0, 254).astype(np.uint8)
    scale_exp = np.clip(254 - e8m0.astype(np.int32), 0, 255).astype(np.uint32)
    scaling = (scale_exp << 23).view(np.float32)
    low = exponent <= 8
    e8m0[low] = 0
    scaling[low] = np.float32(2.0 ** -127)
    return e8m0, scaling


def dn_to_zz(exp):
    rows, cols = exp.shape
    return exp.reshape(rows // 2, 2, cols // 16, 16).transpose(2, 0, 3, 1).reshape(-1)


def pack(meta, name, values):
    out = np.zeros(meta.elem_counts[name], dtype=meta.np_types[name])
    flat = np.asarray(values, dtype=meta.np_types[name]).reshape(-1)
    out[:flat.size] = flat
    return out


def main():
    meta = load_case_meta()
    generator = rng()
    names = meta.outputs
    src_name = meta.inputs[0] if meta.inputs else "v1"
    dst_name, exp_name, max_name, scaling_name, exp_zz_name = (names + ["v2", "v3", "v4", "v5", "v6"])[:5]
    src = generator.uniform(-10, 10, size=(M, N)).astype(np.float32)
    group_max = np.max(np.abs(src.reshape(GROUP_ROWS, GROUP, N)), axis=1)
    exp, scaling = ocp_exp_scaling(group_max)
    scaled = np.repeat(scaling, GROUP, axis=0) * src
    buffers = default_buffers(meta)
    buffers[src_name] = src.reshape(-1)
    for name in names:
        buffers[name] = np.zeros(meta.elem_counts[name], dtype=meta.np_types[name])
    write_buffers(meta, buffers)
    outputs = {
        dst_name: pack(meta, dst_name, fp8_bytes(np.clip(scaled, -448, 448))),
        exp_name: pack(meta, exp_name, exp),
        max_name: pack(meta, max_name, group_max),
        scaling_name: pack(meta, scaling_name, scaling),
        exp_zz_name: pack(meta, exp_zz_name, dn_to_zz(exp)),
    }
    write_golden(meta, outputs)


if __name__ == "__main__":
    main()
