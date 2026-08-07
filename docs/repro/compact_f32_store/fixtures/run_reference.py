#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Launch the native ASC/CCE reference shared library and check its output."""

from __future__ import annotations

import argparse
import ctypes
import os
from pathlib import Path

import numpy as np


WIDTH = 24
SCALE = 1.25
BIAS = 1.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--device", default=os.environ.get("NPU_TEST_DEVICE", "npu:0"))
    args = parser.parse_args()

    import torch
    import torch_npu  # noqa: F401

    torch.npu.config.allow_internal_format = False
    torch_npu.npu.set_compile_mode(jit_compile=False)
    torch.npu.set_device(args.device)
    host_input = np.linspace(-3.0, 4.0, WIDTH, dtype=np.float32)
    expected = host_input * np.float32(SCALE) + np.float32(BIAS)
    input_tensor = torch.from_numpy(host_input).to(args.device)
    output_tensor = torch.full((WIDTH,), float("nan"), dtype=torch.float32, device=args.device)

    library = ctypes.CDLL(str(args.library.resolve()))
    launch = library.launch_reference_compact_f32_store
    launch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    launch.restype = None
    stream = torch.npu.current_stream()._as_parameter_  # noqa: SLF001
    launch(
        ctypes.c_void_p(int(getattr(stream, "value", stream))),
        ctypes.c_void_p(input_tensor.data_ptr()),
        ctypes.c_void_p(output_tensor.data_ptr()),
    )
    torch.npu.synchronize()
    actual = output_tensor.cpu().numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    print("PASS native ASC/CCE compact-store numeric check")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
