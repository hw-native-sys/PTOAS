"""Shared torch_npu helpers for store_pad8 cannsim runs."""

from __future__ import annotations

import os

import torch
import torch_npu

_DEVICE = f"npu:{os.environ.get('NPU_DEVICE', '0')}"


def device_str() -> str:
    return _DEVICE


def init_torch_npu(device: str | None = None) -> None:
    global _DEVICE
    if device is not None:
        _DEVICE = device
    torch.npu.config.allow_internal_format = False
    torch_npu.npu.set_compile_mode(jit_compile=False)
    torch.npu.set_device(_DEVICE)


def empty_npu(shape, dtype: torch.dtype) -> torch.Tensor:
    return torch.empty(shape, dtype=dtype, device=_DEVICE)


def stream_ptr() -> int:
    return torch.npu.current_stream()._as_parameter_


def sync() -> None:
    torch.npu.synchronize()
