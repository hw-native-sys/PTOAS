# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Small host-side tensor factory helpers used by PTODSL wrappers."""

from __future__ import annotations


def empty_like(tensor):
    """Allocate one host-side tensor with the same logical metadata as *tensor*."""
    new_empty = getattr(tensor, "new_empty", None)
    if callable(new_empty):
        return new_empty(tensor.shape)

    try:
        import torch  # type: ignore
    except Exception:
        torch = None
    if torch is not None and isinstance(tensor, torch.Tensor):
        return torch.empty_like(tensor)

    try:
        import numpy as np  # type: ignore
    except Exception:
        np = None
    if np is not None and isinstance(tensor, np.ndarray):
        return np.empty_like(tensor)

    raise TypeError(
        "pto.empty_like(...) could not infer how to allocate an output tensor for "
        f"{type(tensor)!r}; provide O= explicitly or use a tensor type exposing "
        ".new_empty(...), torch.empty_like, or numpy.empty_like support"
    )


__all__ = [
    "empty_like",
]
