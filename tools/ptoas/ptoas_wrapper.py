#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Run the regular PTOAS CLI from a configured CMake package tree."""

from __future__ import annotations

import sys
from pathlib import Path


_PYTHON_ROOT_MODE = "@PTOAS_WRAPPER_PYTHON_ROOT_MODE@"
_PYTHON_ROOT = Path(r"@PTOAS_WRAPPER_PYTHON_ROOT@")
_ARCHIVE_PYTHON_REQUIREMENT_FILE = ".ptoas-python-version"


def _resolve_wrapper_path(argv0: str | None = None) -> Path:
    candidate = argv0 if argv0 is not None else sys.argv[0]
    wrapper = Path(candidate)
    if wrapper.exists():
        return wrapper.resolve()

    raise SystemExit(f"unable to locate the ptoas tree wrapper: {candidate}")


def _require_python_root(python_root: Path, *, context: str) -> Path:
    helper = python_root / "ptoas" / "_cli.py"
    if helper.is_file():
        return python_root.resolve()
    raise SystemExit(
        f"unable to locate the {context} ptoas Python package root: "
        f"expected {python_root}/ptoas/_cli.py"
    )


def _configured_python_root(wrapper: Path) -> Path:
    if _PYTHON_ROOT_MODE == "absolute":
        return _PYTHON_ROOT
    if _PYTHON_ROOT_MODE == "wrapper-relative":
        return wrapper.parent / _PYTHON_ROOT
    raise SystemExit(f"unsupported PTOAS wrapper mode: {_PYTHON_ROOT_MODE}")


def _add_configured_python_root(wrapper: Path) -> Path:
    python_root = _require_python_root(
        _configured_python_root(wrapper), context="configured"
    )

    python_root_text = str(python_root)
    if python_root_text not in sys.path:
        sys.path.insert(0, python_root_text)
    return python_root


def _disable_editable_import_redirects() -> None:
    # Editable installs register meta-path finders before the configured
    # build-tree root can be imported.  Those finders would otherwise resolve
    # ``ptoas._core`` from an unrelated installed checkout, bypassing the
    # native extension and staged resources belonging to this wrapper.
    sys.meta_path[:] = [
        finder
        for finder in sys.meta_path
        if "editable" not in repr(finder).lower()
    ]


def main() -> None:
    wrapper = _resolve_wrapper_path()
    requirement_file = wrapper.parent.parent / _ARCHIVE_PYTHON_REQUIREMENT_FILE
    if requirement_file.is_file():
        required_python_major_minor = requirement_file.read_text(
            encoding="utf-8"
        ).strip()
        actual = f"{sys.version_info.major}.{sys.version_info.minor}"
        if actual != required_python_major_minor:
            raise SystemExit(
                "this ptoas compiler archive requires CPython "
                f"{required_python_major_minor}, but the active interpreter is "
                f"{actual} ({sys.executable})"
            )
    _add_configured_python_root(wrapper)
    _disable_editable_import_redirects()
    from ptoas import _cli

    raise SystemExit(_cli.launch(sys.argv[1:], wrapper=wrapper))


if __name__ == "__main__":
    main()
