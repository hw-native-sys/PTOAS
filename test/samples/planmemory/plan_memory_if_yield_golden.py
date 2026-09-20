#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Validate the branch-local tile allocation with an independent copy oracle."""

import numpy as np

from validation_runtime import default_buffers, load_case_meta, write_buffers, write_golden


def main():
    meta = load_case_meta()
    buffers = default_buffers(meta)
    values = np.arange(meta.elem_counts["v1"], dtype=np.float32)
    buffers["v1"] = ((values % 257 - 128) / 64).astype(np.float16)
    write_buffers(meta, buffers)
    write_golden(meta, {"v2": buffers["v1"].copy()})


if __name__ == "__main__":
    main()
