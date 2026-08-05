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


ELEMENTS = 1024


def generate(output_dir: Path) -> None:
    input_data = (np.arange(ELEMENTS, dtype=np.float32) + 1024.25).astype(
        np.float32
    )
    initial = (-16384.0 - np.arange(ELEMENTS, dtype=np.float32)).astype(
        np.float32
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    input_data.tofile(output_dir / "input.bin")
    initial.tofile(output_dir / "initial.bin")
    initial.tofile(output_dir / "explicit_output.bin")
    initial.tofile(output_dir / "rewritten_output.bin")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate vldus/vstus soft post-update SIM probe data."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    generate(args.output_dir)


if __name__ == "__main__":
    main()
