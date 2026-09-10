# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Helpers for writing kernel-test PTO artifacts and lowering VMI to MI."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

from .backends import ArtifactOutputs, ArtifactPlan

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PTOAS_BIN = Path(
    os.environ.get(
        "PTOAS_BIN",
        str(_REPO_ROOT / "build" / "tools" / "ptoas" / "ptoas"),
    )
)


def resolve_ptoas_bin() -> Path:
    """Return the ptoas binary used for kernel-test artifact lowering."""

    if _PTOAS_BIN.is_file():
        return _PTOAS_BIN

    fallback = _REPO_ROOT / "install" / "bin" / "ptoas"
    if fallback.is_file():
        return fallback

    raise FileNotFoundError(f"ptoas not found: {_PTOAS_BIN}")


def lower_vmi_to_mi(vmi_path: Path, mi_path: Path) -> str:
    """Lower one VMI PTO artifact to MI PTO text with ptoas."""

    result = subprocess.run(
        [
            str(resolve_ptoas_bin()),
            "--pto-arch=a5",
            "--pto-backend=vpto",
            "--pto-level=level3",
            "--emit-vpto",
            "-o",
            str(mi_path),
            str(vmi_path),
        ],
        cwd=_REPO_ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "failed to lower VMI artifact with ptoas:\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return mi_path.read_text(encoding="utf-8")


def materialize_artifact_plan(
    case_id: str,
    plan: ArtifactPlan,
    *,
    root_alias: bool = False,
) -> ArtifactOutputs:
    """Write one backend artifact plan into the kernel-local generated tree."""

    plan.generated_dir.mkdir(parents=True, exist_ok=True)
    plan.case_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, str] = {"case_dir": str(plan.case_dir)}
    written_names: list[str] = []

    vmi_path = plan.case_dir / "vmi.pto"
    mi_path = plan.case_dir / "mi.pto"
    if plan.vmi_text is not None:
        vmi_path.write_text(plan.vmi_text, encoding="utf-8")
        paths["vmi_path"] = str(vmi_path)
        written_names.append("vmi.pto")

    if plan.mi_text is not None:
        mi_path.write_text(plan.mi_text, encoding="utf-8")
    elif plan.vmi_text is not None:
        lower_vmi_to_mi(vmi_path, mi_path)

    if plan.mi_text is not None or plan.vmi_text is not None:
        if mi_path.is_file():
            paths["mi_path"] = str(mi_path)
            if "mi.pto" not in written_names:
                written_names.append("mi.pto")

    if plan.alias_stem:
        for artifact_name in tuple(written_names):
            legacy_case_alias = plan.case_dir / f"{plan.alias_stem}.{artifact_name}"
            if legacy_case_alias.exists():
                legacy_case_alias.unlink()

    if root_alias:
        for artifact_name in tuple(written_names):
            src = plan.case_dir / artifact_name
            dst = plan.generated_dir / artifact_name
            shutil.copyfile(src, dst)
            paths[f"root_{artifact_name.replace('.', '_')}"] = str(dst)
            if plan.alias_stem:
                legacy_root_alias = plan.generated_dir / f"{plan.alias_stem}.{artifact_name}"
                if legacy_root_alias.exists():
                    legacy_root_alias.unlink()

    rendered = " and ".join(written_names) if written_names else "no PTO artifacts"
    return ArtifactOutputs(
        case_dir=str(plan.case_dir),
        message=f"wrote {rendered} under {plan.case_dir} for {case_id}",
        paths=paths,
    )
