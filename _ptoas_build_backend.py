# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""
PEP 517 build backend for ptoas.

Runs the CMake build (assuming LLVM is already built), then delegates
wheel packaging to docker/create_wheel.sh.

Environment variables (all optional):
  LLVM_BUILD_DIR               Path to LLVM build dir
                               (default: /llvm-workspace/llvm-project/build-shared)
  PTO_BUILD_DIR                Path to PTOAS build dir (default: <repo>/build)
  PTO_INSTALL_DIR              Install prefix (default: <repo>/install)
  CMAKE_C_COMPILER             Optional C compiler passed through to CMake
  CMAKE_CXX_COMPILER           Optional C++ compiler passed through to CMake
  PTOAS_PYTHON_PACKAGE_VERSION Wheel version override
"""
from __future__ import annotations

import base64
import glob
import hashlib
import io
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

_REPO = Path(__file__).parent.resolve()
_LLVM_BUILD_DIR = Path(
    os.environ.get("LLVM_BUILD_DIR",
                   "/llvm-workspace/llvm-project/build-shared")
)
_PTO_INSTALL_DIR = Path(
    os.environ.get("PTO_INSTALL_DIR", str(_REPO / "install"))
)
_BUILD_DIR = Path(os.environ.get("PTO_BUILD_DIR", str(_REPO / "build")))
_PTODSL_SOURCE_ROOT = _REPO / "ptodsl"
_MLIR_PY_PKG = (
    _LLVM_BUILD_DIR / "tools" / "mlir" / "python_packages" / "mlir_core"
)
_WHEEL_DIST_DIR = _BUILD_DIR / "wheel-dist"
_PROJECT_VERSION_RE = re.compile(
    r"project\s*\(\s*ptoas\s+VERSION\s+([0-9]+\.[0-9]+)\s*\)"
)


def _assert_installed_ptodsl_payload() -> None:
    """Fail fast if the root install tree did not stage the PTODSL package."""
    installed_init = _PTO_INSTALL_DIR / "ptodsl" / "__init__.py"
    if installed_init.exists():
        return
    raise RuntimeError(
        "PTODSL is missing from the PTOAS install tree. "
        f"Expected to find {installed_init}. "
        "Root CMake install rules must stage the public 'ptodsl' package "
        "before wheel assembly or installed-environment validation."
    )


def _assert_editable_ptodsl_source() -> None:
    """Fail fast if the editable install cannot point at the PTODSL source."""
    source_init = _PTODSL_SOURCE_ROOT / "ptodsl" / "__init__.py"
    if source_init.exists():
        return
    raise RuntimeError(
        "PTODSL editable source is missing from the repository checkout. "
        f"Expected to find {source_init}. "
        "Root editable installs must expose the in-repo PTODSL sources."
    )


def _default_ptoas_version() -> str:
    version = os.environ.get("PTOAS_PYTHON_PACKAGE_VERSION", "").strip()
    if version:
        return version
    cmake_file = _REPO / "CMakeLists.txt"
    match = _PROJECT_VERSION_RE.search(cmake_file.read_text(encoding="utf-8"))
    if not match:
        raise RuntimeError(f"could not find PTOAS version in {cmake_file}")
    return match.group(1)


def get_requires_for_build_wheel(config_settings=None):
    return ["setuptools>=68", "wheel", "pybind11<3"]


def get_requires_for_build_editable(config_settings=None):
    return ["setuptools>=68", "wheel", "pybind11<3"]


def get_requires_for_build_sdist(config_settings=None):
    return []


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    """Return wheel metadata without running the full build."""
    import email.message

    version = _default_ptoas_version()
    dist_info = Path(metadata_directory) / f"ptoas-{version}.dist-info"
    dist_info.mkdir(parents=True, exist_ok=True)

    meta = email.message.Message()
    meta["Metadata-Version"] = "2.1"
    meta["Name"] = "ptoas"
    meta["Version"] = version
    meta["Summary"] = "PTO Assembler & Optimizer"
    meta["Requires-Python"] = ">=3.9"
    meta["License"] = "Apache-2.0"
    meta["Requires-Dist"] = "numpy"
    (dist_info / "METADATA").write_text(str(meta))
    (dist_info / "WHEEL").write_text(
        "Wheel-Version: 1.0\nGenerator: _ptoas_build_backend\n"
        "Root-Is-Purelib: True\nTag: py3-none-any\n"
    )
    return dist_info.name


prepare_metadata_for_build_editable = prepare_metadata_for_build_wheel


def build_sdist(sdist_directory, config_settings=None):
    raise NotImplementedError(
        "ptoas does not support sdist. Use `pip install .` to build a wheel."
    )


def _should_use_linux_hardening_cache() -> bool:
    return sys.platform.startswith("linux")


def _cmake_configure_and_build():
    """CMake configure + build + install."""
    _BUILD_DIR.mkdir(parents=True, exist_ok=True)

    pybind11_dir = subprocess.check_output(
        [sys.executable, "-m", "pybind11", "--cmakedir"], text=True
    ).strip()

    cmake_cmd = [
        "cmake", "-GNinja",
        f"-S{_REPO}", f"-B{_BUILD_DIR}",
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DLLVM_DIR={_LLVM_BUILD_DIR}/lib/cmake/llvm",
        f"-DMLIR_DIR={_LLVM_BUILD_DIR}/lib/cmake/mlir",
        f"-DPython3_ROOT_DIR={sys.prefix}",
        f"-DPython3_EXECUTABLE={sys.executable}",
        "-DPython3_FIND_STRATEGY=LOCATION",
        f"-Dpybind11_DIR={pybind11_dir}",
        f"-DMLIR_PYTHON_PACKAGE_DIR={_MLIR_PY_PKG}",
        f"-DCMAKE_INSTALL_PREFIX={_PTO_INSTALL_DIR}",
    ]

    for compiler_var in ("CMAKE_C_COMPILER", "CMAKE_CXX_COMPILER"):
        compiler = os.environ.get(compiler_var, "").strip()
        if compiler:
            cmake_cmd.append(f"-D{compiler_var}={compiler}")

    release_version = os.environ.get("PTOAS_RELEASE_VERSION_OVERRIDE", "")
    if release_version:
        cmake_cmd.append(f"-DPTOAS_RELEASE_VERSION_OVERRIDE={release_version}")

    hardening_cache = _REPO / "cmake" / "LinuxHardeningCache.cmake"
    if _should_use_linux_hardening_cache() and hardening_cache.exists():
        cmake_cmd.insert(1, f"-C{hardening_cache}")

    subprocess.check_call(cmake_cmd)
    subprocess.check_call(["cmake", "--build", str(_BUILD_DIR)])
    subprocess.check_call(
        ["cmake", "--build", str(_BUILD_DIR), "--target", "install"]
    )
    _assert_installed_ptodsl_payload()


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    _cmake_configure_and_build()

    env = os.environ.copy()
    env.update({
        "PTO_SOURCE_DIR": str(_REPO),
        "PTO_INSTALL_DIR": str(_PTO_INSTALL_DIR),
        "LLVM_BUILD_DIR": str(_LLVM_BUILD_DIR),
        "PTO_WHEEL_DIST_DIR": str(_WHEEL_DIST_DIR),
        # Keep wheel packaging on the same interpreter pip used to invoke the
        # backend. This avoids drifting to a different `python3` on PATH in
        # conda/self-hosted CI environments.
        "PYTHON": sys.executable,
    })
    subprocess.check_call(
        ["bash", str(_REPO / "docker" / "create_wheel.sh")],
        env=env,
    )

    wheels = sorted(
        glob.glob(str(_WHEEL_DIST_DIR / "ptoas-*.whl")),
        key=os.path.getmtime,
    )
    if not wheels:
        raise RuntimeError(
            f"No ptoas-*.whl found in {_WHEEL_DIST_DIR} after build."
        )

    wheel_path = Path(wheels[-1])
    dest = Path(wheel_directory) / wheel_path.name
    shutil.copy2(wheel_path, dest)
    return dest.name


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    """PEP 660 editable install.

    Builds the C++ extensions in-place, then produces a minimal wheel that
    installs a .pth file pointing sys.path at the local PTODSL sources plus
    the installed/runtime build outputs. No files are copied into
    site-packages except the .pth file itself.
    """
    _cmake_configure_and_build()
    _assert_editable_ptodsl_source()

    version = _default_ptoas_version()

    # Paths that must be on sys.path for the package to be importable
    pth_paths = [
        # mlir.* namespace + _pto.so (installed there by CMake)
        str(_MLIR_PY_PKG),
        # Prefer the repository PTODSL sources so editable installs pick up
        # local Python edits instead of staged/install-tree copies.
        str(_PTODSL_SOURCE_ROOT),
        # Installed PTOAS runtime overlay (mlir.dialects.pto, _pto, TileOps).
        str(_PTO_INSTALL_DIR),
        # Keep the in-tree extension/staging output last as a fallback.
        str(_BUILD_DIR / "python"),
    ]

    # Use executable .pth content so editable installs can deterministically
    # prepend the repository PTODSL sources ahead of pre-existing build-tree
    # entries in the caller's Python environment.
    pth_content = (
        "import sys; "
        f"pth_paths = {pth_paths!r}; "
        "[sys.path.remove(path) for path in pth_paths if path in sys.path]; "
        "sys.path[:0] = pth_paths\n"
    )
    pth_filename = "ptoas-editable.pth"

    # ---- Build the editable wheel (a zip with .pth + dist-info) ----
    tag = f"py3-none-any"
    wheel_name = f"ptoas-{version}-{tag}.whl"
    wheel_path = Path(wheel_directory) / wheel_name

    dist_info_dir = f"ptoas-{version}.dist-info"

    def _sha256_record(data: bytes) -> str:
        digest = hashlib.sha256(data).digest()
        b64 = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
        return f"sha256={b64}"

    pth_bytes = pth_content.encode()
    wheel_meta = (
        "Wheel-Version: 1.0\n"
        "Generator: _ptoas_build_backend\n"
        "Root-Is-Purelib: True\n"
        f"Tag: {tag}\n"
        "Build: editable\n"
    ).encode()
    entry_points = (
        "[console_scripts]\n"
        "ptoas=ptoas._launcher:main\n"
    ).encode()
    metadata_content = (
        "Metadata-Version: 2.1\n"
        "Name: ptoas\n"
        f"Version: {version}\n"
        "Summary: PTO Assembler & Optimizer\n"
        "Requires-Python: >=3.9\n"
        "License: Apache-2.0\n"
        "Requires-Dist: numpy\n"
    ).encode()

    record_lines = [
        f"{pth_filename},{_sha256_record(pth_bytes)},{len(pth_bytes)}",
        f"{dist_info_dir}/WHEEL,{_sha256_record(wheel_meta)},{len(wheel_meta)}",
        f"{dist_info_dir}/entry_points.txt,{_sha256_record(entry_points)},{len(entry_points)}",
        f"{dist_info_dir}/METADATA,{_sha256_record(metadata_content)},{len(metadata_content)}",
        f"{dist_info_dir}/RECORD,,",
    ]
    record_content = "\n".join(record_lines).encode()

    with zipfile.ZipFile(wheel_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(pth_filename, pth_bytes)
        zf.writestr(f"{dist_info_dir}/WHEEL", wheel_meta)
        zf.writestr(f"{dist_info_dir}/entry_points.txt", entry_points)
        zf.writestr(f"{dist_info_dir}/METADATA", metadata_content)
        zf.writestr(f"{dist_info_dir}/RECORD", record_content)

    return wheel_name
