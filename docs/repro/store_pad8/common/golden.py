"""Torch reference for the store_pad8 microbench.

Each of N_ACC live f32 accumulator vectors (length VL) is reduced to one
scalar. CCE stores compactly (vcadd + vsts ONEPT_B32). VMI pad-8 broadcasts
to 8 lanes under an 8-wide mask; VMI mask1 stores compactly under a 1-lane
mask (still not CCE ONEPT lowering on vmi-v0.1.3).

N_ACC=20 (LARGE) matches a typical residual-mix epilogue row count.
N_ACC=4 (SMALL) is a quick smoke case.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

VL = 64
REDUCE_PAD = 8
SEED = 123


@dataclass(frozen=True)
class StorePad8Case:
    name: str
    n_acc: int


LARGE = StorePad8Case(name="large", n_acc=20)
SMALL = StorePad8Case(name="small", n_acc=4)


def reduce_ref(acc: torch.Tensor) -> torch.Tensor:
    """acc: [N_ACC, VL] f32 -> [N_ACC] f32 (sum over lanes)."""
    return acc.sum(dim=-1)


def generate_case(case: StorePad8Case) -> dict:
    torch.manual_seed(SEED + case.n_acc)
    acc = torch.randn((case.n_acc, VL), dtype=torch.float32)
    reduced = reduce_ref(acc)
    return {
        "case": case,
        "acc": acc.numpy(),
        "reduced": reduced.numpy(),
    }


def extract_padded(padded: np.ndarray, n_acc: int, pad: int = REDUCE_PAD) -> np.ndarray:
    """VMI stores each scalar broadcast across pad lanes; take lane 0."""
    return np.asarray(padded, dtype=np.float32).reshape(n_acc, pad)[:, 0]
