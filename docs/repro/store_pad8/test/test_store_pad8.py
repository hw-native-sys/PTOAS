#!/usr/bin/env python3
"""Correctness test for the store_pad8 microbench (CCE / VMI)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common.golden import LARGE, SMALL, extract_padded, generate_case  # noqa: E402
from common.launcher import launch  # noqa: E402
from common.torch_runtime import device_str, init_torch_npu  # noqa: E402


def _check(backend: str, case, variant: str) -> tuple[bool, float]:
    ref = generate_case(case)
    acc = __import__("torch").from_numpy(ref["acc"]).to(device_str())
    got = launch(acc, case.n_acc, backend=backend, variant=variant).cpu().numpy()
    if backend == "vmi" and variant == "pad8":
        got = extract_padded(got, case.n_acc)
    diff = float(np.max(np.abs(got.astype(np.float32) - ref["reduced"])))
    atol = float(os.environ.get("STORE_PAD8_ATOL", 2e-3))
    ok = diff <= atol
    tag = f"{backend}/{variant}" if backend == "vmi" else backend
    print(
        f"[{tag}] store_pad8 {case.name} (n_acc={case.n_acc}): maxDiff={diff:.6g} ok={ok}",
        flush=True,
    )
    return ok, diff


def main() -> int:
    init_torch_npu()
    backend = os.environ.get("TLVF_VMI_BACKEND", "vmi").lower()
    variant = os.environ.get("STORE_PAD8_VARIANT", "pad8").lower()
    case_name = os.environ.get("STORE_PAD8_CASE", "large").lower()
    case = LARGE if case_name == "large" else SMALL

    ok, _ = _check(backend, case, variant)
    tag = f"{backend}/{variant}" if backend == "vmi" else backend
    if ok:
        print(f"All store_pad8 [{tag}/{case.name}] tests PASSED", flush=True)
        os._exit(0)
    print(f"store_pad8 [{tag}/{case.name}] tests FAILED", flush=True)
    os._exit(1)


if __name__ == "__main__":
    raise SystemExit(main())
