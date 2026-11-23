# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Build inputs and an independent golden for two K128 MX matmul steps."""

import numpy as np

from validation_runtime import default_buffers, load_case_meta, single_output, write_buffers, write_golden


def main():
    """Use unit FP8 payloads to make each E8M0 group contribution observable."""
    meta = load_case_meta()
    a, b, a_scale, b_scale = meta.inputs
    m = n = 32
    k_step = 128
    group_size = 32
    groups = 2 * k_step // group_size
    e8m0_bias = 127
    fp8_one = 0x38

    rows, a_groups = np.indices((m, groups))
    b_groups, cols = np.indices((groups, n))
    a_exp = (rows + a_groups + a_groups // 4) % 4
    b_exp = (cols + 2 * b_groups + b_groups // 4) % 5
    a_bits = (e8m0_bias + a_exp).astype(np.uint8)
    b_bits = (e8m0_bias + b_exp).astype(np.uint8)

    buffers = default_buffers(meta)
    buffers[a][:m * k_step] = fp8_one
    buffers[b][:k_step * n] = fp8_one
    # The runner can reserve a larger GM buffer than the packed scale view.
    buffers[a_scale][:a_bits.size] = a_bits.reshape(m // 16, 16, groups // 2, 2).transpose(0, 2, 1, 3).reshape(-1)
    buffers[b_scale][:b_bits.size] = b_bits.reshape(groups // 2, 2, n // 16, 16).transpose(2, 0, 3, 1).reshape(-1)
    write_buffers(meta, buffers)

    # Each column/row of the scale matrices represents 32 unit FP8 products.
    expected = group_size * (np.exp2(a_exp) @ np.exp2(b_exp))
    write_golden(meta, {single_output(meta): expected.astype(np.float32)})


if __name__ == "__main__":
    main()
