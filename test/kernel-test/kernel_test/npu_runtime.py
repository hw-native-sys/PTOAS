# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Shared torch_npu runtime helpers for kernel-test backends."""

from __future__ import annotations

import os

_DEVICE = f"npu:{os.environ.get('NPU_DEVICE', '0')}"


def device_str() -> str:
    return _DEVICE


def init_torch_npu(device: str | None = None) -> None:
    global _DEVICE

    import torch
    import torch_npu

    _DEVICE = device or _DEVICE
    torch.npu.config.allow_internal_format = False
    torch_npu.npu.set_compile_mode(jit_compile=False)
    torch.npu.set_device(_DEVICE)


def ensure_runtime(component: str = "kernel-test") -> None:
    """Initialize torch_npu and normalize runtime initialization failures."""

    try:
        init_torch_npu()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"failed to initialize torch_npu for {component}; run direct correctness "
            "on a host with NPU access or use the simulator transport script for "
            "cannsim workflows"
        ) from exc


def empty_npu(shape, dtype):
    import torch

    return torch.empty(shape, dtype=dtype, device=_DEVICE)


def stream_ptr() -> int:
    import torch

    return torch.npu.current_stream()._as_parameter_


def sync() -> None:
    import torch

    torch.npu.synchronize()
