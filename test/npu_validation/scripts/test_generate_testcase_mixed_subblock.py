#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import importlib.util
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "generate_testcase",
    SCRIPT_DIR / "generate_testcase.py",
)
GENERATOR = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(GENERATOR)


SOURCE = r'''
AICORE void mixed_probe_aic(__gm__ float* v1, int32_t v2, int32_t v3) {
  (void)v1;
  (void)v2;
  (void)v3;
}

AICORE void mixed_probe_aiv(__gm__ float* v1, int32_t v2, int32_t v3, int32_t v4) {
  // pto: %subblock_idx
  if (v4 == 0) {
    v1[0] = (float)(v2 + v3);
  }
}
'''


def main() -> None:
    kernel_info = GENERATOR._describe_kernel_source(SOURCE)
    assert kernel_info["kind"] == "mixed"
    assert kernel_info["raw_params"] == [
        "__gm__ float* v1",
        "int32_t v2",
        "int32_t v3",
    ]
    assert kernel_info["aiv_local_param"] == "v4"

    wrapper = GENERATOR._append_mixed_kernel_wrapper(
        SOURCE,
        kernel_info["kernel_name"],
        kernel_info["raw_params"],
        kernel_info["aic_text"],
        kernel_info["aiv_text"],
        kernel_info["aiv_local_param"],
    )
    entry = wrapper.rsplit('extern "C" __global__ AICORE void mixed_probe', 1)[1]
    assert "int32_t v4" not in entry.split("{", 1)[0]
    assert "if (get_subblockid() == 0)" in entry


if __name__ == "__main__":
    main()
