#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import numpy as np


def main() -> None:
    data = (np.arange(64, dtype=np.uint32) * 17) + 3
    output_i = np.zeros((72,), dtype=np.uint32)
    output_s = np.zeros((72,), dtype=np.uint32)
    golden_i = output_i.copy()
    golden_s = output_s.copy()
    golden_i[1:65] = data
    golden_s[2:66] = data

    data.tofile("input.bin")
    output_i.tofile("output_i.bin")
    output_s.tofile("output_s.bin")
    golden_i.tofile("golden_output_i.bin")
    golden_s.tofile("golden_output_s.bin")


if __name__ == "__main__":
    main()
