# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Runtime data preparation for the rope kernel."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from kernel_test.npu_runtime import device_str, empty_npu

from .tile_config import sim_fn_name

if TYPE_CHECKING:
    import numpy as np
    import torch


@dataclass(frozen=True)
class RopeLaunchArgs:
    """Prepared runtime arguments shared by all rope backends."""

    dtype: str
    mode: str
    mode_value: int
    fn_name: str
    x: torch.Tensor
    cos: torch.Tensor
    sin: torch.Tensor
    y: torch.Tensor
    params: torch.Tensor
    s_count: int
    n_count: int


def torch_dtype(dtype: str) -> torch.dtype:
    import torch

    if dtype == "f16":
        return torch.float16
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "f32":
        return torch.float32
    raise ValueError(f"unknown dtype: {dtype}")


def params_tensor(params: np.ndarray) -> torch.Tensor:
    import numpy as np
    import torch

    return torch.from_numpy(np.asarray(params, dtype=np.int32)).to(device_str())


def artifact_case_dir_name(case: dict[str, object]) -> str:
    """Build a stable per-case artifact directory name for rope."""

    tile = case["tile"]
    return f"{case['dtype']}_{case['mode']}_s{tile.s}_n{tile.n}"


def artifact_case_dir(root: Path, case: dict[str, object], *, backend_name: str) -> Path:
    """Return the backend-specific rope artifact directory for one case."""

    return root / backend_name / artifact_case_dir_name(case)


def prepare_launch_args(case: dict, *, cycle: bool = False) -> RopeLaunchArgs:
    """Convert one rope case into prepared device tensors and launch metadata."""

    import torch

    dtype = case["dtype"]
    x_dtype = torch_dtype(dtype)
    cs_dtype = torch_dtype("f16") if dtype == "bf16" else x_dtype
    mode = case["mode"]
    mode_value = 0 if mode == "half" else 1
    dev = device_str()

    x = torch.from_numpy(case["x"]).to(x_dtype).to(dev)
    cos = torch.from_numpy(case["cos"]).to(cs_dtype).to(dev)
    sin = torch.from_numpy(case["sin"]).to(cs_dtype).to(dev)
    y = empty_npu(case["y"].shape, x_dtype)
    params = params_tensor(case["params"])
    s_count, n_count = [int(value) for value in case["params"]]

    return RopeLaunchArgs(
        dtype=dtype,
        mode=mode,
        mode_value=mode_value,
        fn_name=sim_fn_name(mode, dtype, cycle=cycle),
        x=x,
        cos=cos,
        sin=sin,
        y=y,
        params=params,
        s_count=s_count,
        n_count=n_count,
    )
