#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import numpy as np


BUFFER_BYTES = 4096
SENTINEL = 0xA5


def main() -> None:
    data_f32 = (np.arange(BUFFER_BYTES // 4, dtype=np.float32) * 0.25) + 1.0
    data = data_f32.view(np.uint8)
    golden = np.full((BUFFER_BYTES,), SENTINEL, dtype=np.uint8)

    # vldsx2 DINTLV_B32: low receives even f32 elements, high receives odd.
    vldsx2 = np.empty((256,), dtype=np.float32)
    for iteration, base in enumerate((0, 128)):
        dst = iteration * 128
        vldsx2[dst : dst + 64] = data_f32[base : base + 128 : 2]
        vldsx2[dst + 64 : dst + 128] = data_f32[base + 1 : base + 128 : 2]
    golden[0:1024] = vldsx2.view(np.uint8)

    # vsldb starts each access repeat_stride=8 blocks (64 f32 elements) past
    # the current base, and the loop recurrence advances by that same amount.
    vsldb_source = 2048 // 4
    vsldb = np.concatenate(
        (
            data_f32[vsldb_source + 64 : vsldb_source + 128],
            data_f32[vsldb_source + 128 : vsldb_source + 192],
        )
    )
    golden[1024:1536] = vsldb.view(np.uint8)

    # Two NORM predicate images advance by 32 bytes each.
    golden[2048:2112] = data[1024:1088]

    # Each sprsts writes one 32-bit SPR value. The returned-base recurrence is
    # 32 bytes, so the two zero words are separated by 28 untouched bytes.
    golden[3072:3076] = 0
    golden[3104:3108] = 0

    data.tofile("input.bin")
    np.full((BUFFER_BYTES,), SENTINEL, dtype=np.uint8).tofile("output.bin")
    golden.tofile("golden_output.bin")


if __name__ == "__main__":
    main()
