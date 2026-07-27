"""ctypes launcher for the store_pad8 CCE VF sim kernel."""

from __future__ import annotations

import ctypes
from pathlib import Path
import sys

import torch

_REPRO = Path(__file__).resolve().parent.parent
if str(_REPRO) not in sys.path:
    sys.path.insert(0, str(_REPRO))

from common.torch_runtime import empty_npu, stream_ptr, sync  # noqa: E402

_VL = 64

_LIB: ctypes.CDLL | None = None


def _lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is None:
        from common.cce_vf_build import build_cce_root

        path = build_cce_root(Path(__file__).resolve().parent)
        lib = ctypes.CDLL(str(path))
        for name in ("call_store_pad8_cce_large", "call_store_pad8_cce_small"):
            fn = getattr(lib, name)
            fn.argtypes = [ctypes.c_void_p] * 3
            fn.restype = None
        _LIB = lib
    return _LIB


def _vp(t: torch.Tensor) -> ctypes.c_void_p:
    return ctypes.c_void_p(t.data_ptr())


def launch(acc: torch.Tensor, n_acc: int) -> torch.Tensor:
    """acc: [n_acc, VL] f32 on npu -> [n_acc] f32 (compact ONEPT reduce)."""
    lib = _lib()
    name = "call_store_pad8_cce_large" if n_acc == 20 else "call_store_pad8_cce_small"
    if n_acc not in (4, 20):
        raise ValueError(f"unsupported n_acc={n_acc} (build only has large=20 / small=4)")
    reduced = empty_npu((n_acc,), torch.float32)
    getattr(lib, name)(stream_ptr(), _vp(acc), _vp(reduced))
    sync()
    return reduced
