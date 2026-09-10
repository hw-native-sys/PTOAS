# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Backend factory for the rope kernel."""

from __future__ import annotations

from kernel_test.backends import BackendAdapter


def create_backend(name: str) -> BackendAdapter:
    """Create one rope backend adapter from the local backend packages."""

    if name == "cce":
        try:
            from .cce.backend import RopeCceBackend
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "rope cce backend requires the rope runtime dependencies; use the "
                "`pto` conda environment or install numpy/torch/torch_npu first"
            ) from exc

        return RopeCceBackend()
    if name == "vmi":
        try:
            from .vmi.backend import RopeVmiBackend
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "rope vmi backend requires the rope runtime dependencies; use the "
                "`pto` conda environment or install ptodsl/torch/torch_npu first"
            ) from exc

        return RopeVmiBackend()
    if name == "mi":
        try:
            from .mi.backend import RopeMiBackend
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "rope mi backend requires the rope runtime dependencies; use the "
                "`pto` conda environment or install ptodsl/torch/torch_npu first"
            ) from exc

        return RopeMiBackend()
    raise ValueError(f"unknown rope backend: {name}")
