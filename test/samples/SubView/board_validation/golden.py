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
    src = np.arange(16 * 16, dtype=np.float32).reshape(16, 16)

    out0 = src[0:8, 0:8].copy()
    out1 = src[0:8, 8:16].copy()
    out2 = src[8:16, 0:8].copy()
    out3 = src[8:16, 8:16].copy()

    src.tofile("src.bin")
    np.zeros((8, 8), dtype=np.float32).tofile("out0.bin")
    np.zeros((8, 8), dtype=np.float32).tofile("out1.bin")
    np.zeros((8, 8), dtype=np.float32).tofile("out2.bin")
    np.zeros((8, 8), dtype=np.float32).tofile("out3.bin")

    out0.tofile("golden_out0.bin")
    out1.tofile("golden_out1.bin")
    out2.tofile("golden_out2.bin")
    out3.tofile("golden_out3.bin")


if __name__ == "__main__":
    main()
