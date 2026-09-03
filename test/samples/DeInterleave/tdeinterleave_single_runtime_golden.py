#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import numpy as np
from pathlib import Path
import sys

for search_root in (Path(__file__).resolve().parent, Path(__file__).resolve().parents[1]):
    if (search_root / "validation_runtime.py").is_file():
        sys.path.insert(0, str(search_root))
        break

from validation_runtime import default_buffers, float_values, load_case_meta, matrix32, rng, write_buffers, write_golden


def main():
    meta = load_case_meta()
    [src_name] = meta.inputs
    src = matrix32(
        float_values(rng(), meta.elem_counts[src_name], style="signed"),
        rows=16,
        cols=128,
    )
    buffers = default_buffers(meta)
    buffers[src_name] = src.reshape(-1)
    write_buffers(meta, buffers)
    write_golden(meta, {meta.outputs[0]: src[:, 0::2].reshape(-1), meta.outputs[1]: src[:, 1::2].reshape(-1)})


if __name__ == "__main__":
    main()
