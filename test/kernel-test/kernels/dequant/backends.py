# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Backend factory for dequant."""

from __future__ import annotations

from kernel_test.backends import BackendAdapter


def create_backend(name: str) -> BackendAdapter:
    """Create one dequant backend adapter."""

    if name == "vmi":
        try:
            from .vmi.backend import AntiMxQuantTailAxisVmiBackend
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError(
                "dequant vmi backend requires the ptodsl/PTOAS "
                "Python environment; use the `pto` conda environment or install "
                "the local ptodsl dependencies first"
            ) from exc
        return AntiMxQuantTailAxisVmiBackend()
    raise ValueError(f"unknown dequant backend: {name}")
