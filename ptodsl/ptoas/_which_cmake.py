# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Locate the system-level cmake binary (not the pip console-script shim)."""

import os
from pathlib import Path
import shutil
from typing import Optional

# Native-executable magic numbers. The pip `cmake` wheel installs a Python
# console-script shim whose first bytes are `#!` (0x23 0x21); a real system
# cmake is an ELF (Linux) or Mach-O (macOS) binary.
_ELF_MAGIC = b"\x7fELF"
_MACHO_MAGICS = (
    b"\xfe\xed\xfa\xce",  # Mach-O 32-bit
    b"\xfe\xed\xfa\xcf",  # Mach-O 64-bit
    b"\xce\xfa\xed\xfe",  # Mach-O 32-bit, byte-swapped
    b"\xcf\xfa\xed\xfe",  # Mach-O 64-bit, byte-swapped
    b"\xca\xfe\xba\xbe",  # Mach-O universal (fat) binary
    b"\xbe\xba\xfe\xca",  # Mach-O universal (fat), byte-swapped
)


def _is_native_executable(path: Path) -> bool:
    if not path.exists() or not path.is_file():
        return False
    if path.stat().st_size <= 4:
        return False
    with open(path, "rb") as fh:
        header = fh.read(4)
    return header == _ELF_MAGIC or header in _MACHO_MAGICS


def which_cmake() -> Optional[Path]:
    """Return the first native cmake found on PATH, skipping the pip shim.

    :return: path to a system cmake executable, or None if none is found
    :rtype: Optional[Path]
    """
    path_dir_lst = [d.strip() for d in os.environ.get("PATH", "").split(os.pathsep) if d.strip()]
    seen = []
    for path_dir in path_dir_lst:
        if path_dir in seen:
            continue
        seen.append(path_dir)
        cmake_str = shutil.which("cmake", path=path_dir)
        if not cmake_str:
            continue
        cmake_file = Path(cmake_str).resolve()
        if _is_native_executable(cmake_file):
            return cmake_file
    return None
