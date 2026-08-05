#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import numpy as np


LOAD_BEGIN = 0
LOAD_END = 64
LOAD_INPUT_BEGIN = 128
STORE_BEGIN = 256


def fail(message: str) -> None:
    print(f"[ERROR] {message}")
    raise SystemExit(2)


def main() -> None:
    input_data = np.fromfile("input.bin", dtype=np.float32)
    initial = np.fromfile("initial.bin", dtype=np.float32)
    explicit = np.fromfile("explicit_output.bin", dtype=np.float32)
    rewritten = np.fromfile("rewritten_output.bin", dtype=np.float32)

    if not (input_data.shape == initial.shape == explicit.shape == rewritten.shape):
        fail(
            "shape mismatch: "
            f"input={input_data.shape}, initial={initial.shape}, "
            f"explicit={explicit.shape}, rewritten={rewritten.shape}"
        )

    mismatch = np.flatnonzero(explicit != rewritten)
    if mismatch.size:
        idx = int(mismatch[0])
        fail(
            "explicit and rewritten post-update paths differ at "
            f"index {idx}: explicit={explicit[idx]}, rewritten={rewritten[idx]}"
        )

    expected_load = input_data[LOAD_INPUT_BEGIN : LOAD_INPUT_BEGIN + LOAD_END]
    if not np.array_equal(explicit[LOAD_BEGIN:LOAD_END], expected_load):
        mismatch = np.flatnonzero(
            explicit[LOAD_BEGIN:LOAD_END] != expected_load
        )
        idx = int(mismatch[0]) if mismatch.size else LOAD_BEGIN
        fail(
            f"vldus load result mismatch at index {idx}: "
            f"expected={expected_load[idx]}, output={explicit[idx]}"
        )

    store_changes = np.flatnonzero(
        explicit[STORE_BEGIN:] != initial[STORE_BEGIN:]
    )
    if not store_changes.size:
        fail("vstus/vstar did not change the store probe region")
    if int(store_changes[0]) != 0:
        fail(
            "vstus first write was pre-offset: first changed store-relative "
            f"index is {int(store_changes[0])}, expected 0"
        )

    print(
        "[INFO] compare passed: explicit and rewritten paths match; "
        f"vstus first changed store-relative index={int(store_changes[0])}"
    )


if __name__ == "__main__":
    main()
