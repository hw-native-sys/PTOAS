# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Backend interfaces for the kernel-test framework."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

RunPurpose = Literal["correctness", "cycle"]


@dataclass(frozen=True)
class ArtifactOutputs:
    """Normalized artifact emission result for one kernel-test case."""

    case_dir: str
    message: str
    paths: Mapping[str, str]


@dataclass(frozen=True)
class ArtifactPlan:
    """Backend-provided compile result that the framework can materialize."""

    generated_dir: Path
    case_dir: Path
    vmi_text: str | None = None
    mi_text: str | None = None
    alias_stem: str | None = None


class BackendAdapter(Protocol):
    """Stable interface shared by all framework backends."""

    name: str

    def is_supported(self, case: object, *, purpose: RunPurpose) -> tuple[bool, str | None]:
        """Return support status and an optional human-readable reason."""

    def launch(self, case: object, *, purpose: RunPurpose) -> object:
        """Launch one case and return backend-specific outputs."""


class ArtifactBackend(Protocol):
    """Optional extension for backends that can emit PTO artifacts."""

    def build_artifact_plan(self, case_id: str, case: object) -> ArtifactPlan:
        """Build the backend-specific PTO artifact plan for one case."""
