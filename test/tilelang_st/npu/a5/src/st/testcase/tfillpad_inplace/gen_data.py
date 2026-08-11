# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

"""Generate golden data for tfillpad_inplace test cases.

For tfillpad_inplace:
  - Source and destination tile handles alias the same UB storage
  - Input: full tile shape (rows x cols), random values in valid region, zeros in padding
  - Golden: full tile shape with valid region copied and padding filled with MAX (PadValue.Max)
"""

import os
import numpy as np
import struct

from cases import CASES

# FLT_MAX for float (matching DSL PadValue.MAX)
def _float32_from_bits(bits: int) -> float:
    return struct.unpack(">f", bits.to_bytes(4, byteorder="big", signed=False))[0]

_FLT_MAX = _float32_from_bits(0x7F7FFFFF)  # ~3.4028235e+38


def get_pad_value(dtype, padval_name):
    """Get the actual pad value for a dtype based on PadValue enum."""
    if padval_name == "Max":
        if np.issubdtype(dtype, np.floating):
            return np.float32(_FLT_MAX)
        else:
            return np.iinfo(dtype).max
    elif padval_name == "Min":
        if np.issubdtype(dtype, np.floating):
            return np.float32(-_FLT_MAX)
        else:
            return np.iinfo(dtype).min
    elif padval_name == "Zero":
        return dtype(0)
    else:
        return dtype(0)


def setup_case_rng(case):
    """Set a per-case deterministic random seed."""
    np.random.seed(hash(case["name"]) & 0xFFFFFFFF)


def save_case_data(case_name, data_dict):
    """Create case directory and write {name}.bin for each entry."""
    os.makedirs(case_name, exist_ok=True)
    for name, arr in data_dict.items():
        arr.tofile(os.path.join(case_name, f"{name}.bin"))


for case in CASES:
    setup_case_rng(case)

    dtype = case["dtype"]
    src_shape = case["src_shape"]
    src_valid = case["src_valid"]
    dst_shape = case["dst_shape"]
    dst_valid = case["dst_valid"]
    fill_padval = case.get("fill_padval", "Max")

    src_vr, src_vc = src_valid
    dst_r, dst_c = dst_shape
    dst_vr, dst_vc = dst_valid

    # Input occupies the full physical buffer; only src_valid contains data.
    input_data = np.zeros(src_shape, dtype=dtype)
    input_data[:src_vr, :src_vc] = np.random.uniform(
        1.0, 10.0, size=(src_vr, src_vc)
    ).astype(dtype)

    # Golden: dst full region
    # Copy src.valid region to dst[:src_vr, :src_vc]
    # Fill cols src_vc to dst_vc with FillPadVal
    # Fill rows src_vr to dst_vr with FillPadVal (row expansion, if any)
    golden = np.zeros(dst_shape, dtype=dtype)
    golden[:src_vr, :src_vc] = input_data[:src_vr, :src_vc]

    # Fill column padding (cols src_vc to dst_vc)
    if dst_vc > src_vc:
        fill_val = get_pad_value(dtype, fill_padval)
        golden[:dst_vr, src_vc:dst_vc] = fill_val

    # Fill row padding (rows src_vr to dst_vr)
    if dst_vr > src_vr:
        fill_val = get_pad_value(dtype, fill_padval)
        golden[src_vr:dst_vr, :dst_vc] = fill_val

    save_case_data(case["name"], {"input": input_data, "golden": golden})
    print(f"[INFO] gen_data: {case['name']} "
          f"src_valid={src_valid} dst_shape={dst_shape} "
          f"fill_pad={fill_padval} dtype={dtype.__name__}")
