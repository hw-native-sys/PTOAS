# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Standard Python console entry point for PTOAS."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Sequence


def _resolve_wrapper_path(argv0: str | None = None) -> Path:
    candidate = argv0 if argv0 is not None else sys.argv[0]
    wrapper = Path(candidate)
    if wrapper.exists():
        return wrapper.resolve()

    raise SystemExit(f"unable to locate the active ptoas entry point: {candidate}")


def _load_native_module():
    from ptoas import _core

    return _core


def _resolve_tileops_dir(native_module) -> Path:
    module_file = getattr(native_module, "__file__", None)
    if not module_file:
        raise SystemExit("ptoas._core does not expose a module file")

    package_root = Path(module_file).resolve().parent
    runtime_root = package_root / "_runtime"
    tileops_dir = runtime_root / "share" / "ptoas" / "TileOps"
    if not tileops_dir.is_dir():
        raise SystemExit(
            "unable to locate packaged PTOAS TileOps resources: expected "
            f"{tileops_dir}"
        )
    return tileops_dir.resolve()


def launch(user_args: Sequence[str], *, wrapper: Path | None = None) -> int:
    native_module = _load_native_module()
    tileops_dir = _resolve_tileops_dir(native_module)
    wrapper = wrapper.resolve() if wrapper is not None else _resolve_wrapper_path()

    os.environ["PTOAS_BIN"] = str(wrapper)
    argv = [str(wrapper)]
    argv.extend(user_args)

    tileops_python_root = str(tileops_dir.parent)
    inserted_tileops_root = tileops_python_root not in sys.path
    if inserted_tileops_root:
        sys.path.insert(0, tileops_python_root)
    try:
        return int(native_module.main(argv))
    finally:
        if inserted_tileops_root:
            sys.path.remove(tileops_python_root)


def main() -> int:
    return launch(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
