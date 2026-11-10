# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Exact integer-valued FP16 matmul reference for all 128 output tiles."""

import numpy as np


def main():
    rng = np.random.default_rng(1694)
    lhs = rng.integers(-3, 4, size=(32, 32)).astype(np.float16)
    rhs = rng.integers(-3, 4, size=(32, 32)).astype(np.float16)
    lhs.tofile("v1.bin")
    rhs.tofile("v2.bin")
    np.zeros((4096, 32), dtype=np.float32).tofile("v3.bin")
    tile = lhs.astype(np.float32) @ rhs.astype(np.float32)
    np.tile(tile, (128, 1)).tofile("golden_v3.bin")


if __name__ == "__main__":
    main()
