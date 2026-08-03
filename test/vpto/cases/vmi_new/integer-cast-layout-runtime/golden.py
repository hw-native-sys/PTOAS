#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import argparse
from pathlib import Path

import numpy as np


SENTINEL = 0xA5


def repeat(values, size, dtype):
    values = np.asarray(values, dtype=dtype)
    return np.resize(values, size).astype(dtype)


def write_case(output_dir, first_file, source, outputs):
    source.tofile(output_dir / f"v{first_file}.bin")
    for offset, expected in enumerate(outputs, start=1):
        path = output_dir / f"v{first_file + offset}.bin"
        np.full(expected.nbytes, SENTINEL, dtype=np.uint8).tofile(path)
        expected.tofile(output_dir / f"golden_v{first_file + offset}.bin")


def generate(output_dir: Path) -> None:
    u8 = np.arange(256, dtype=np.uint8)
    i8 = np.arange(-128, 128, dtype=np.int16).astype(np.int8)
    u16 = repeat(
        [0, 1, 127, 128, 255, 256, 257, 32767, 32768, 65534, 65535,
         0x1234, 0xABCD, 0xFF80, 0x00FF, 0x0100], 128, np.uint16)
    i16 = repeat(
        [-32768, -32767, -1025, -257, -256, -255, -129, -128, -127,
         -1, 0, 1, 127, 128, 255, 256, 257, 32767], 128, np.int16)
    u32 = repeat(
        [0, 1, 127, 128, 255, 256, 65535, 65536, 0x12345678,
         0x80000000, 0xFFFFFF80, 0xFFFFFFFF, 0xA5A55A5A], 64, np.uint32)
    i32 = repeat(
        [-2147483648, -2147483647, -65537, -65536, -32769, -32768,
         -257, -256, -129, -128, -1, 0, 1, 127, 128, 255, 256, 32767,
         32768, 65535, 65536, 2147483647], 64, np.int32)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_case(output_dir, 1, u8, [u8.astype(np.uint16), u8.astype(np.uint32)])
    write_case(output_dir, 4, i8, [i8.astype(np.int16), i8.astype(np.int32)])
    write_case(output_dir, 7, u16, [u16.astype(np.uint32), u16.astype(np.uint8)])
    write_case(output_dir, 10, i16, [i16.astype(np.int32), i16.astype(np.int8)])
    write_case(output_dir, 13, u32, [u32.astype(np.uint16), u32.astype(np.uint8)])
    write_case(output_dir, 16, i32, [i32.astype(np.int16), i32.astype(np.int8)])

    zero_gap_u8 = u8[:128]
    zero_gap_u16 = u16[:64]
    zero_gap_u8.tofile(output_dir / "v19.bin")
    zero_gap_u16.tofile(output_dir / "v20.bin")
    for index, expected in {
        21: zero_gap_u8.astype(np.uint16),
        22: zero_gap_u8[:64].astype(np.uint32),
        23: zero_gap_u16.astype(np.uint32),
    }.items():
        np.full(expected.nbytes, SENTINEL, dtype=np.uint8).tofile(
            output_dir / f"v{index}.bin")
        expected.tofile(output_dir / f"golden_v{index}.bin")

    assert np.any(u16.astype(np.uint8) != np.minimum(u16, 255).astype(np.uint8))
    assert np.any(i16.astype(np.int8) != np.clip(i16, -128, 127).astype(np.int8))
    assert np.any(u32.astype(np.uint16) != np.minimum(u32, 65535).astype(np.uint16))
    assert np.any(i32.astype(np.int16) != np.clip(i32, -32768, 32767).astype(np.int16))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    generate(args.output_dir)


if __name__ == "__main__":
    main()
