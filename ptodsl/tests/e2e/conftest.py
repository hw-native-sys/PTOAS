# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Pytest configuration for e2e A3/A5 VPTO/EmitC tests."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest


# Ensure ptodsl package is importable.  When pytest is invoked from the repo
# root, ptodsl is already on sys.path; this handles direct invocation inside
# ptodsl/tests/e2e/ as well.
_bootstrap_dir = None
for _parent in Path(__file__).resolve().parents:
    if (_parent / "ptodsl" / "__init__.py").exists():
        _bootstrap_dir = _parent
        break
if _bootstrap_dir is not None and str(_bootstrap_dir) not in sys.path:
    sys.path.insert(0, str(_bootstrap_dir))


def pytest_addoption(parser):
    parser.addoption(
        "--target",
        default="a3",
        choices=["a3", "a5"],
        help="Target Ascend architecture (default: a3)",
    )
    parser.addoption(
        "--backend",
        default="vpto",
        choices=["vpto", "emitc"],
        help="PTOAS backend (default: vpto)",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "require_npu: mark test as requiring an NPU device with torch_npu installed",
    )
    config.addinivalue_line(
        "markers",
        "target(a3|a5): restrict test to a specific target architecture",
    )


@pytest.fixture(scope="session")
def torch():
    """Initialize torch_npu and return torch module (session-scoped)."""
    import torch
    import torch_npu  # noqa: F401

    torch.npu.config.allow_internal_format = False
    torch_npu.npu.set_compile_mode(jit_compile=False)
    torch.npu.set_device("npu:0")
    return torch


@pytest.fixture(scope="session")
def target_arch(request):
    return request.config.getoption("--target")


@pytest.fixture(scope="session")
def backend(request):
    return request.config.getoption("--backend")


# ---------------------------------------------------------------------------
# Lazy helpers (import torch/ptodsl only when needed)
# ---------------------------------------------------------------------------

def torch_dtype(name: str):
    import torch

    return getattr(torch, name)


def pto_dtype(name: str):
    from ptodsl import pto

    return getattr(pto, name)


def npu_stream(torch):
    return torch.npu.current_stream()._as_parameter_  # noqa: SLF001
