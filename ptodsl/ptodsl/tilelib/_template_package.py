# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Resolve the canonical TileOps package in source and packaged environments."""

from __future__ import annotations

from functools import lru_cache
from importlib import import_module


_SOURCE_PACKAGE = "TileOps"
_PACKAGED_PACKAGE = "ptoas._runtime.share.ptoas.TileOps"


def _is_missing_package(error: ModuleNotFoundError, package: str) -> bool:
    missing = error.name
    return bool(missing) and (missing == package or package.startswith(f"{missing}."))


@lru_cache(maxsize=1)
def tileops_package():
    """Return TileOps from a source root or the bundled PTOAS resources."""

    try:
        return import_module(_SOURCE_PACKAGE)
    except ModuleNotFoundError as source_error:
        if not _is_missing_package(source_error, _SOURCE_PACKAGE):
            raise

    try:
        return import_module(_PACKAGED_PACKAGE)
    except ModuleNotFoundError as packaged_error:
        if not _is_missing_package(packaged_error, _PACKAGED_PACKAGE):
            raise
        raise ModuleNotFoundError(
            "unable to locate TileOps in either the Python path or the "
            "packaged PTOAS runtime resources",
            name=_SOURCE_PACKAGE,
        ) from packaged_error


def load_template(op: str, target: str) -> bool:
    """Load a template through the resolved canonical TileOps package."""

    return tileops_package().load_template(op, target)


__all__ = ["load_template", "tileops_package"]
