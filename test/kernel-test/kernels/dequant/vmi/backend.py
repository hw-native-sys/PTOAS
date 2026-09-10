# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Runtime VMI backend for dequant."""

from __future__ import annotations

import os
from pathlib import Path

from kernel_test.backends import ArtifactPlan, RunPurpose
from kernel_test.npu_runtime import ensure_runtime, stream_ptr, sync
from kernel_test.pto_artifacts import materialize_artifact_plan

from .anti_mx_quant_tail_axis_vmi import dequant_vmi_bf16, dequant_vmi_f16, dequant_vmi_f32
from ..runtime import artifact_case_dir, prepare_compile_args, prepare_launch_args

_VMI_ROOT = Path(__file__).resolve().parent
_KERNEL_ROOT = _VMI_ROOT.parent
_GENERATED_DIR = _KERNEL_ROOT / "generated"
_COMPILED: dict[tuple[str, str, int, int], object] = {}


def _kernel_for_dst(dst_fmt: str):
    if dst_fmt == "f32":
        return dequant_vmi_f32
    if dst_fmt == "bf16":
        return dequant_vmi_bf16
    if dst_fmt == "f16":
        return dequant_vmi_f16
    raise ValueError(f"unsupported dst_fmt: {dst_fmt}")


def _prepare_runtime_kernel(case: dict) -> object:
    compile_args = prepare_compile_args(case)
    cache_key = (
        compile_args.src_fmt,
        compile_args.dst_fmt,
        compile_args.row_block_num,
        compile_args.col_block_num,
    )
    compiled = _COMPILED.get(cache_key)
    if compiled is None:
        compiled = _kernel_for_dst(compile_args.dst_fmt).compile(
            SRC_FMT=compile_args.src_fmt,
            ROW_BLOCK_NUM=compile_args.row_block_num,
            COL_BLOCK_NUM=compile_args.col_block_num,
        )
        _COMPILED[cache_key] = compiled
    return compiled


def _build_artifact_plan(case: dict[str, object]) -> ArtifactPlan:
    compile_args = prepare_compile_args(case)
    compiled = _prepare_runtime_kernel(case)

    case_dir = artifact_case_dir(_GENERATED_DIR, case)
    return ArtifactPlan(
        generated_dir=_GENERATED_DIR,
        case_dir=case_dir,
        vmi_text=compiled.mlir_text(),
        alias_stem=compile_args.named_stem,
    )


class AntiMxQuantTailAxisVmiBackend:
    """Runtime VMI backend for dequant."""

    name = "vmi"

    def is_supported(self, case: object, *, purpose: RunPurpose) -> tuple[bool, str | None]:
        if purpose == "cycle":
            return False, "backend=vmi has correctness launch only; no cycle probe yet"
        supported = case["src_fmt"] in {"e4m3", "e5m2"} and case["dst_fmt"] in {"f32", "bf16", "f16"}
        if supported:
            return True, None
        return False, "backend=vmi not wired for this case"

    def launch(self, case: object, *, purpose: RunPurpose) -> object:
        if purpose != "correctness":
            raise ValueError(f"unsupported purpose for dequant vmi backend: {purpose}")

        ensure_runtime("dequant")
        launch_args = prepare_launch_args(case)
        compiled = _prepare_runtime_kernel(case)
        artifacts = materialize_artifact_plan(
            "correctness",
            _build_artifact_plan(case),
            root_alias=prepare_compile_args(case).default_alias,
        )

        compiled[1, stream_ptr()](
            launch_args.x.data_ptr(),
            launch_args.scale.data_ptr(),
            launch_args.y.data_ptr(),
        )
        sync()

        return {
            "y": launch_args.y,
            **dict(artifacts.paths),
        }

    def cache_tag(self) -> str:
        backend_py = _VMI_ROOT / "backend.py"
        kernel_py = _VMI_ROOT / "anti_mx_quant_tail_axis_vmi.py"
        return (
            f"vmi:{backend_py}:{os.path.getmtime(backend_py):.0f}:"
            f"{kernel_py}:{os.path.getmtime(kernel_py):.0f}"
        )

    def build_artifact_plan(self, case_id: str, case: object) -> ArtifactPlan:
        del case_id
        return _build_artifact_plan(case)
