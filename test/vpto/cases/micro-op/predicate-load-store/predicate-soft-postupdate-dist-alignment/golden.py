#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import numpy as np


BUFFER_BYTES = 2048
NORM_BASE = 0
US_BASE = 512
DS_BASE = 1024


def downsample_predicate(payload: np.ndarray) -> np.ndarray:
    bits = np.unpackbits(payload, bitorder="little")
    return np.packbits(bits[::2], bitorder="little")


def main() -> None:
    indices = np.arange(BUFFER_BYTES, dtype=np.uint32)
    data = ((indices * 73 + 19) & 0xFF).astype(np.uint8)
    golden = np.zeros((BUFFER_BYTES,), dtype=np.uint8)

    # Two normal loop iterations must advance by 32 bytes in NORM mode.
    golden[NORM_BASE : NORM_BASE + 64] = data[NORM_BASE : NORM_BASE + 64]

    # US expands a 16-byte packed image and PK packs it back to 16 bytes.
    golden[US_BASE : US_BASE + 32] = data[US_BASE : US_BASE + 32]

    # DS consumes 64 bytes and keeps every other bit. The two source windows
    # begin 32 bytes apart, exactly matching the A5 DS alignment unit.
    for iteration in range(2):
        src = DS_BASE + iteration * 32
        dst = DS_BASE + iteration * 32
        golden[dst : dst + 32] = downsample_predicate(data[src : src + 64])

    data.tofile("input.bin")
    np.zeros((BUFFER_BYTES,), dtype=np.uint8).tofile("output.bin")
    golden.tofile("golden_output.bin")


if __name__ == "__main__":
    main()
