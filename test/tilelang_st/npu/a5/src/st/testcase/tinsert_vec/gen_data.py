# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

import numpy as np

from cases import CASES
from st_common import setup_case_rng, save_case_data


for case in CASES:
    setup_case_rng(case)
    dtype = case["dtype"]
    src_rows, src_cols = case["src_shape"]
    dst_rows, dst_cols = case["dst_shape"]
    idx_row, idx_col = case["index_row"], case["index_col"]

    src = np.random.uniform(-1.0, 1.0, size=(src_rows, src_cols)).astype(dtype)
    dst = np.random.uniform(-1.0, 1.0, size=(dst_rows, dst_cols)).astype(dtype)

    golden = dst.copy()
    golden[idx_row:idx_row + src_rows, idx_col:idx_col + src_cols] = src

    data = {"input1": src, "input2": dst, "golden": golden}
    save_case_data(case["name"], data)
    print(
        f"[INFO] gen_data: {case['name']} src=({src_rows},{src_cols}) dst=({dst_rows},{dst_cols}) "
        f"idx=({idx_row},{idx_col}) dtype={dtype.__name__}"
    )