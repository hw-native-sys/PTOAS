#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import numpy as np


ELEMENT_COUNT = 1024
VECTOR_ELEMENTS = 64
SENTINEL = np.float32(-999.0)


def main() -> None:
    input_data = np.arange(ELEMENT_COUNT, dtype=np.float32) + np.float32(0.25)
    output = np.full((ELEMENT_COUNT,), SENTINEL, dtype=np.float32)
    golden = output.copy()
    for offset in (0, 64, 136, 216):
        end = offset + VECTOR_ELEMENTS
        golden[offset:end] = input_data[offset:end]
    for offset in (512, 576, 640, 704):
        end = offset + VECTOR_ELEMENTS
        golden[offset:end] = input_data[offset:end]
    golden[800:888] = input_data[800:888]

    input_data.tofile("input.bin")
    output.tofile("output.bin")
    golden.tofile("golden_output.bin")


if __name__ == "__main__":
    main()
