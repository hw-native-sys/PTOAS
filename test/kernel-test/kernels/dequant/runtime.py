# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Shared runtime and artifact helpers for the dequant kernel."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from kernel_test.npu_runtime import device_str, empty_npu

if TYPE_CHECKING:
    import torch

ENTRY_SYMBOL = "anti_mx_quant_tail_axis_compute_data_probe"
DEFAULT_NAMED_STEM = "dequant"


@dataclass(frozen=True)
class DequantCompileArgs:
    """Prepared compile inputs shared by the dequant backends."""

    src_fmt: str
    scale_fmt: str
    dst_fmt: str
    row_block_num: int
    col_block_num: int
    loop_num2vf: int
    entry_symbol: str
    case_dir_name: str
    named_stem: str
    default_alias: bool


@dataclass(frozen=True)
class DequantLaunchArgs:
    """Prepared runtime tensors and metadata for one dequant launch."""

    src_fmt: str
    dst_fmt: str
    row_block_num: int
    col_block_num: int
    loop_num2vf: int
    x: "torch.Tensor"
    scale: "torch.Tensor"
    y: "torch.Tensor"


def case_dir_name(case: dict[str, object]) -> str:
    return (
        f"fp8_{case['src_fmt']}_scale_{case['scale_fmt']}_out_{case['dst_fmt']}"
        f"_rb{case['row_block_num']}_cb{case['col_block_num']}"
    )


def prepare_compile_args(case: dict[str, object]) -> DequantCompileArgs:
    """Normalize one dequant case dict into stable compile metadata."""

    return DequantCompileArgs(
        src_fmt=str(case["src_fmt"]),
        scale_fmt=str(case["scale_fmt"]),
        dst_fmt=str(case["dst_fmt"]),
        row_block_num=int(case["row_block_num"]),
        col_block_num=int(case["col_block_num"]),
        loop_num2vf=int(case.get("loop_num2vf", 1)),
        entry_symbol=str(case.get("entry_symbol", ENTRY_SYMBOL)),
        case_dir_name=case_dir_name(case),
        named_stem=str(case.get("named_stem", DEFAULT_NAMED_STEM)),
        default_alias=bool(case.get("default_alias", False)),
    )


def artifact_case_dir(root: Path, case: dict[str, object]) -> Path:
    """Return the per-case artifact directory."""

    return root / case_dir_name(case)


def torch_dtype(dst_fmt: str) -> torch.dtype:
    import torch

    if dst_fmt == "f32":
        return torch.float32
    if dst_fmt == "bf16":
        return torch.bfloat16
    if dst_fmt == "f16":
        return torch.float16
    raise ValueError(f"unknown dst_fmt: {dst_fmt}")


def prepare_launch_args(case: dict[str, object]) -> DequantLaunchArgs:
    """Convert one dequant case into device tensors and launch metadata."""

    import torch

    dev = device_str()
    dst_dtype = torch_dtype(str(case["dst_fmt"]))
    x = torch.from_numpy(case["x_bits"]).to(torch.uint8).to(dev)
    scale = torch.from_numpy(case["scale_bits"]).to(torch.uint8).to(dev)
    y = empty_npu(case["y_expected"].shape, dst_dtype)
    return DequantLaunchArgs(
        src_fmt=str(case["src_fmt"]),
        dst_fmt=str(case["dst_fmt"]),
        row_block_num=int(case["row_block_num"]),
        col_block_num=int(case["col_block_num"]),
        loop_num2vf=int(case.get("loop_num2vf", 1)),
        x=x,
        scale=scale,
        y=y,
    )
