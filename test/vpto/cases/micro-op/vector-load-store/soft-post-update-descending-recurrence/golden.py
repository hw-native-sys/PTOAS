#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import numpy as np


def main() -> None:
    data = (np.arange(4096, dtype=np.uint16) * 23 + 3).astype(np.uint8)
    golden = np.full((4096,), 0xA5, dtype=np.uint8)
    golden[0:256] = data[1024:1280]
    golden[256:512] = data[992:1248]
    data.tofile("input.bin")
    np.full((4096,), 0xA5, dtype=np.uint8).tofile("output.bin")
    golden.tofile("golden_output.bin")


if __name__ == "__main__":
    main()
