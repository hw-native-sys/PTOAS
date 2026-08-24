#!/usr/bin/python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from pathlib import Path
import sys
import numpy as np
for root in (Path(__file__).resolve().parent, Path(__file__).resolve().parents[1]):
    if (root / "validation_runtime.py").is_file(): sys.path.insert(0, str(root)); break
from validation_runtime import default_buffers, load_case_meta, rng, single_output, write_buffers, write_golden
meta = load_case_meta()
out_name = single_output(meta)
src_name = meta.inputs[0]
src = rng().uniform(-2.0, 2.0, meta.elem_counts[src_name]).astype(np.float32)
buffers = default_buffers(meta)
buffers[src_name] = src
write_buffers(meta, buffers)
expected = np.zeros(meta.elem_counts[out_name], dtype=np.float32)
expected[: 16 * 16] = src[: 16 * 16]
write_golden(meta, {out_name: expected})
