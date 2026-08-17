#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


SAMPLES_DIR = Path(__file__).resolve().parents[2] / "samples"
SAMPLE_DIR = SAMPLES_DIR / "DeepseekV4DecodeA3"
sys.path.insert(0, str(SAMPLES_DIR))
sys.path.insert(0, str(SAMPLE_DIR))

from deepseek_v4_decode_golden_lib import _buffer_values  # noqa: E402


def main() -> None:
    meta = SimpleNamespace(
        elem_counts={
            "v1": 4,
            "v2": 8,
            "v3": 512 * 128,
            "v4": 512,
            "v5": 16 * 64,
            "v6": 8 * 4_096,
            "v7": 256,
            "v8": 128 * 128,
            "v9": 128,
            "v10": 16_384,
        },
        np_types={
            "v1": np.int32,
            "v2": np.int32,
            "v3": np.int8,
            "v4": np.float32,
            "v5": np.float32,
            "v6": np.float32,
            "v7": np.int32,
            "v8": np.int8,
            "v9": np.float32,
            "v10": np.float32,
        },
        outputs=["v6"],
    )
    buffers = {
        name: _buffer_values(meta, name, "score")
        for name in meta.elem_counts
    }
    assert np.all(buffers["v1"] == 2_048)
    assert np.all(buffers["v2"] == 2_047)
    assert np.count_nonzero(buffers["v3"]) == buffers["v3"].size
    assert np.count_nonzero(buffers["v8"]) == buffers["v8"].size
    assert np.all(buffers["v7"] == 0)
    assert np.all(buffers["v4"] > 0)
    assert np.all(buffers["v5"] > 0)
    assert np.all(buffers["v9"] > 0)
    assert np.all(buffers["v10"] == 0)
    assert np.all(buffers["v6"] == 0)


if __name__ == "__main__":
    main()
