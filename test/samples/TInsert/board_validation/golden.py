#!/usr/bin/python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

import numpy as np


def main():
    np.random.seed(19)

    src_a = np.random.random(size=(32, 32)).astype(np.float32)
    src_b = np.random.random(size=(32, 32)).astype(np.float32)
    # Identity matrix used by the post-tinsert matmul path.
    rhs_identity = np.eye(32, dtype=np.float16)
    out_init = np.zeros((32, 32), dtype=np.float32)

    src_a.tofile("v1.bin")
    src_b.tofile("v2.bin")
    rhs_identity.tofile("v3.bin")
    out_init.tofile("v4.bin")


if __name__ == "__main__":
    main()
