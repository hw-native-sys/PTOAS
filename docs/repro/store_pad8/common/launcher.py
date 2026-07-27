"""Dispatch the store_pad8 microbench to CCE or VMI."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.torch_runtime import empty_npu, stream_ptr, sync  # noqa: E402

_COMPILED: dict[str, object] = {}


def launch_cce(acc: torch.Tensor, n_acc: int) -> torch.Tensor:
    import importlib.util

    path = ROOT / "cce" / "launcher.py"
    spec = importlib.util.spec_from_file_location("store_pad8_cce_launcher", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load CCE launcher from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.launch(acc, n_acc)


def launch_vmi(acc: torch.Tensor, n_acc: int) -> torch.Tensor:
    key = f"vmi:{n_acc}"
    if key not in _COMPILED:
        sys.path.insert(0, str(ROOT / "vmi"))
        from store_pad8_vmi import store_pad8_vmi_large, store_pad8_vmi_small

        kn = store_pad8_vmi_large if n_acc == 20 else store_pad8_vmi_small
        if n_acc not in (4, 20):
            raise ValueError(f"unsupported n_acc={n_acc}")
        _COMPILED[key] = kn.compile()
    compiled = _COMPILED[key]
    pad = empty_npu((n_acc * 8,), torch.float32)
    compiled[1, stream_ptr()](acc.data_ptr(), pad.data_ptr())
    sync()
    return pad


def launch(acc: torch.Tensor, n_acc: int, backend: str | None = None) -> torch.Tensor:
    backend = (backend or os.environ.get("TLVF_VMI_BACKEND", "vmi")).lower()
    if backend == "cce":
        return launch_cce(acc, n_acc)
    if backend == "vmi":
        return launch_vmi(acc, n_acc)
    raise ValueError(f"unsupported backend: {backend}")
