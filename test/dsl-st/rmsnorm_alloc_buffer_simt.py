#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np

from common import assert_close, auto_main


REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_PATH = REPO_ROOT / "ptodsl" / "examples" / "rms_norm" / "rmsnorm_alloc_buffer_simt.py"
HIDDEN_SIZE = 4096
THREADS = 128
ROUNDS = 8
LANES = 4
N_CORES = 1
TOKENS_PER_CORE = 1
EPS = np.float32(1.0e-6)
SEED = 0x483001


def _load_rmsnorm_example():
    spec = spec_from_file_location("ptodsl_rmsnorm_alloc_buffer_simt_st", EXAMPLE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load RMSNorm example from {EXAMPLE_PATH}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_RMSNORM_EXAMPLE = _load_rmsnorm_example()


class _RMSNormSpecialization:
    def compile(self):
        return _RMSNORM_EXAMPLE.rmsnorm_4096_alloc_buffer_simt_context_kernel.compile(
            threads=THREADS,
            rounds=ROUNDS,
            lanes=LANES,
            hidden_size=HIDDEN_SIZE,
            n_cores=N_CORES,
            tokens_per_core=TOKENS_PER_CORE,
        )


RMSNORM_KERNEL = _RMSNormSpecialization()


def make_inputs():
    rng = np.random.RandomState(SEED)
    x = rng.uniform(-0.75, 0.75, size=(N_CORES * TOKENS_PER_CORE, HIDDEN_SIZE)).astype(np.float32)
    w = rng.uniform(0.5, 1.5, size=(HIDDEN_SIZE,)).astype(np.float32)
    y = np.zeros_like(x)
    rstd = np.zeros((N_CORES * TOKENS_PER_CORE,), dtype=np.float32)
    return x, y, w, rstd


def rmsnorm_reference(x: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sum_sq = np.sum(x * x, axis=1, dtype=np.float32)
    rstd = (np.float32(1.0) / np.sqrt(sum_sq / np.float32(x.shape[1]) + EPS)).astype(np.float32)
    y = (x * rstd[:, None] * w[None, :]).astype(np.float32)
    return y, rstd


def make_case():
    x, y, w, rstd = make_inputs()
    y_ref, rstd_ref = rmsnorm_reference(x, w)
    return [x, y, w, rstd], (y_ref, rstd_ref), [EPS]


def check_case(device_inputs, expected):
    y_ref, rstd_ref = expected
    y_out = device_inputs[1].cpu().numpy()
    rstd_out = device_inputs[3].cpu().numpy()
    assert_close(y_out, y_ref, rtol=1.0e-4, atol=1.0e-4)
    assert_close(rstd_out, rstd_ref, rtol=1.0e-4, atol=1.0e-5)


CASES = [
    {
        "name": "rmsnorm_alloc_buffer_simt",
        "kernel": RMSNORM_KERNEL,
        "make_case": make_case,
        "check": check_case,
    },
]


def emit_mlir():
    return RMSNORM_KERNEL.compile().mlir_text()


EMIT_MLIR_FN = emit_mlir


auto_main(globals())
