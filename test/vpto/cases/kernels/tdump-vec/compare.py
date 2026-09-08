#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# case: kernels/tdump-vec
# family: kernels
# target_ops: pto.mte_gm_ub, pto.tdump
# scenarios: tdump-vec-f32-4x16, header-plus-row-major-payload

import os
import sys

import numpy as np


MAGIC = 0x50544450
VERSION = 1
ELEM_SIZE = 4
ELEM_TYPE_CODE = 3  # f32
ROWS = 4
COLS = 16
DATA_OFFSET = 64
HEADER_WORDS = 16


def decode_header(raw: np.ndarray):
    if raw.nbytes < DATA_OFFSET:
        print(f"[ERROR] dump too small for header: {raw.nbytes} bytes")
        return None
    words = np.frombuffer(raw[:DATA_OFFSET].tobytes(), dtype="<u4")
    if words[0] != MAGIC:
        print(f"[ERROR] bad magic: 0x{int(words[0]):08x}, want 0x{MAGIC:08x}")
        return None
    if words[1] != VERSION:
        print(f"[ERROR] bad version: {int(words[1])}, want {VERSION}")
        return None
    if words[2] != ELEM_SIZE:
        print(f"[ERROR] bad elem_size: {int(words[2])}, want {ELEM_SIZE}")
        return None
    if words[3] != ELEM_TYPE_CODE:
        print(f"[ERROR] bad elem_type_code: {int(words[3])}, want {ELEM_TYPE_CODE}")
        return None
    shape = [int(words[5]), int(words[6])]
    valid = [int(words[7]), int(words[8])]
    if shape != [ROWS, COLS]:
        print(f"[ERROR] bad shape: {shape}, want {[ROWS, COLS]}")
        return None
    if valid != [ROWS, COLS]:
        print(f"[ERROR] bad valid_shape: {valid}, want {[ROWS, COLS]}")
        return None
    if int(words[9]) != DATA_OFFSET:
        print(f"[ERROR] bad data_offset: {int(words[9])}, want {DATA_OFFSET}")
        return None
    payload = np.frombuffer(raw[DATA_OFFSET:].tobytes(), dtype="<f4")
    return payload


def main():
    strict = os.environ.get("COMPARE_STRICT") == "1"
    eps = 1e-5 if strict else 1e-3

    golden_path = "./golden_v2.bin"
    output_path = "./v2.bin"
    if not os.path.exists(output_path):
        print(f"[ERROR] Output missing: {output_path}")
        return 1
    if not os.path.exists(golden_path):
        print(f"[ERROR] Golden missing: {golden_path}")
        return 1

    raw = np.fromfile(output_path, dtype=np.uint8)
    if raw.nbytes < DATA_OFFSET + ROWS * COLS * ELEM_SIZE:
        print(f"[ERROR] dump too small: {raw.nbytes} bytes")
        return 1
    payload = decode_header(raw)
    if payload is None:
        return 1

    golden = np.fromfile(golden_path, dtype=np.uint8)
    golden_payload = np.frombuffer(golden[DATA_OFFSET:].tobytes(), dtype="<f4")

    if payload.shape != golden_payload.shape:
        print(f"[ERROR] Shape mismatch: {payload.shape} vs {golden_payload.shape}")
        return 1
    if not np.allclose(payload, golden_payload, atol=eps, rtol=eps, equal_nan=True):
        abs_diff = np.abs(payload.astype(np.float64) - golden_payload.astype(np.float64))
        idx = int(np.argmax(abs_diff))
        print(
            f"[ERROR] Mismatch: max diff={float(abs_diff[idx])} at idx={idx} "
            f"(golden={float(golden_payload[idx])}, out={float(payload[idx])})"
        )
        return 1
    print("[INFO] compare passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
