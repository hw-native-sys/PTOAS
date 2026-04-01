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


def generate_golden(v1, v2, v3):
    outputs = []
    outputs.append(np.full(1024, -1, dtype=np.float32))
    return outputs


def main():
    np.random.seed(19)
    v1 = np.random.random(size=(1024,)).astype(np.float32)
    v1.tofile("v1.bin")
    v2 = np.random.random(size=(1024,)).astype(np.float32)
    v2.tofile("v2.bin")
    v3 = np.random.random(size=(1024,)).astype(np.float32)
    v3.tofile("v3.bin")
    outputs = generate_golden(v1, v2, v3)
    outputs[0].tofile("golden_v4.bin")


if __name__ == "__main__":
    main()
