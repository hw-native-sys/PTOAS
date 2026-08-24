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
src0_name, src1_name, idx0_name, idx1_name = meta.inputs
out_name = single_output(meta)
generator = rng()
src0 = generator.uniform(-2.0, 2.0, meta.elem_counts[src0_name]).astype(np.float32)
src1 = generator.uniform(-2.0, 2.0, meta.elem_counts[src1_name]).astype(np.float32)
idx0 = np.full(meta.elem_counts[idx0_name], 32, dtype=np.int32)
idx1 = np.full(meta.elem_counts[idx1_name], 32, dtype=np.int32)
buffers = default_buffers(meta)
buffers.update({src0_name: src0, src1_name: src1, idx0_name: idx0, idx1_name: idx1})
write_buffers(meta, buffers)
expected = np.zeros((8, 64), dtype=np.float32)
expected[:, :8] = src0.reshape(8, 64)[:, :8]
expected[:, 8:16] = src1.reshape(8, 64)[:, :8]
write_golden(meta, {out_name: expected})
