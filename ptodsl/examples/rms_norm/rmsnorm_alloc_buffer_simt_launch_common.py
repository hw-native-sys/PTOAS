# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Shared host-side setup for the RMSNorm alloc_buffer/SIMT launch examples."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


from rmsnorm_alloc_buffer_simt import rmsnorm_4096_alloc_buffer_simt_context_kernel


_DEVICE = "npu:0"
_HIDDEN_SIZE = 4096
_THREADS = 128
_ROUNDS = 8
_LANES = 4
_EPS = np.float32(1.0e-6)
_Y_GUARD_ELEMS = 1024
_RSTD_GUARD_ELEMS = 64
_SENTINEL = np.float32(123456.0)


@dataclass(frozen=True)
class Case:
    name: str
    n_cores: int
    tokens_per_core: int
    seed: int
    rtol: float = 1.0e-4
    y_atol: float = 1.0e-4
    rstd_atol: float = 1.0e-5

    @property
    def tokens(self) -> int:
        return self.n_cores * self.tokens_per_core


CASES = [
    Case("one_core_one_token", n_cores=1, tokens_per_core=1, seed=0x483001),
    Case("one_core_four_tokens", n_cores=1, tokens_per_core=4, seed=0x483004),
    Case("four_cores_two_tokens_each", n_cores=4, tokens_per_core=2, seed=0x483402),
]

FULL_CASE = Case("full_64_cores_64_tokens_each", n_cores=64, tokens_per_core=64, seed=0x483640)


def init_runtime():
    import torch
    import torch_npu  # noqa: F401

    torch.npu.config.allow_internal_format = False
    torch_npu.npu.set_compile_mode(jit_compile=False)
    torch.npu.set_device(_DEVICE)
    return torch


def npu_stream(torch):
    return torch.npu.current_stream()._as_parameter_  # noqa: SLF001


def make_inputs(case: Case) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(case.seed)
    x = rng.uniform(-0.75, 0.75, size=(case.tokens, _HIDDEN_SIZE)).astype(np.float32)
    w = rng.uniform(0.5, 1.5, size=(_HIDDEN_SIZE,)).astype(np.float32)

    # Make token/core addressing mistakes obvious in the output comparison.
    token_offsets = (np.arange(case.tokens, dtype=np.float32)[:, None] * np.float32(0.001))
    x = (x + token_offsets).astype(np.float32)
    return x, w


def rmsnorm_reference(x: np.ndarray, w: np.ndarray, eps: np.float32) -> tuple[np.ndarray, np.ndarray]:
    sum_sq = np.sum(x * x, axis=1, dtype=np.float32)
    rstd = (np.float32(1.0) / np.sqrt(sum_sq / np.float32(x.shape[1]) + eps)).astype(np.float32)
    y = (x * rstd[:, None] * w[None, :]).astype(np.float32)
    return y, rstd


def compile_kernel(case: Case):
    return rmsnorm_4096_alloc_buffer_simt_context_kernel.compile(
        threads=_THREADS,
        rounds=_ROUNDS,
        lanes=_LANES,
        hidden_size=_HIDDEN_SIZE,
        n_cores=case.n_cores,
        tokens_per_core=case.tokens_per_core,
    )


def assert_guard_unchanged(name: str, guard: np.ndarray) -> None:
    if not np.all(guard == _SENTINEL):
        bad = np.nonzero(guard != _SENTINEL)[0]
        first = int(bad[0])
        raise AssertionError(
            f"{name} guard overwritten at guard index {first}: got {guard[first]!r}, expected {_SENTINEL!r}"
        )
