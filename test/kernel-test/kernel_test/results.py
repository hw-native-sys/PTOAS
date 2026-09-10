# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Result models shared by framework runners."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CaseResult:
    """Normalized result for one case execution."""

    ok: bool
    message: str
    skipped: bool = False


@dataclass(frozen=True)
class RunSummary:
    """Aggregate summary for one correctness run."""

    total: int
    passed: int
    failed: int
    skipped: int

    @property
    def all_passed(self) -> bool:
        return self.failed == 0
