#!/usr/bin/python3
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
    if (search_root / 'validation_runtime.py').is_file():
        sys.path.insert(0, str(search_root))
        break

from validation_runtime import (
    default_buffers,
    float_values,
    load_case_meta,
    matrix32,
    pack_predicate_mask_for_buffer,
    rng,
    single_output,
    write_buffers,
    write_golden,
)


def main():
    meta = load_case_meta()
    [src_name] = meta.inputs
    generator = rng()
    src = float_values(generator, meta.elem_counts[src_name], style='cmp')
    pred = matrix32(src) > np.float32(1.0)
    buffers = default_buffers(meta)
    buffers[src_name] = src
    write_buffers(meta, buffers)
    out_name = single_output(meta)
    packed = pack_predicate_mask_for_buffer(pred, elem_count=meta.elem_counts[out_name], dtype=meta.np_types[out_name])
    write_golden(meta, {out_name: packed})


if __name__ == '__main__':
    main()
