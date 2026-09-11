#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Online-compilation infrastructure for the version-sensitive pybind11 extensions.

The Python-version-independent ``libPTOASCompiler`` DSO is shipped prebuilt in the
wheel. The version-sensitive pybind11 extensions cannot be abi3 (pybind11 is
interpreter-specific), so on an interpreter that does not match the prebuilt ABI
they are (re)compiled here from the shipped sources + header closure under
``ptoas/_online/`` and cached.

A single online CMake invocation builds every version-sensitive extension at once:

  * ``ptoas._core``                                 -> package root
  * ``ptoas.mlir._mlir_libs._mlir``                 -> ``mlir/_mlir_libs``
  * ``ptoas.mlir._mlir_libs._mlirDialectsLLVM``     -> ``mlir/_mlir_libs``
  * ``ptoas.mlir._mlir_libs._site_initialize_0``    -> ``mlir/_mlir_libs``

This module must NOT import any of those native modules (its whole job is to
produce them). Loading the freshly built binaries is the meta path finder's job
(see ``_loader``); here we only compile and locate the ``.so`` files.
"""

import dataclasses
import fcntl
import importlib.machinery
import logging
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import threading
import time
from typing import List, Optional, Tuple

_log = logging.getLogger(__name__)

_QUALIFIED_MODULE = "ptoas._core"
_DSO_STEMS = ("libPTOASCompiler",)


@dataclasses.dataclass(frozen=True)
class _MemberSpec:
    """A version-sensitive native extension produced by the online build.

    ``stem`` is the module base name (e.g. ``_mlir``); ``subdir`` is the install
    location relative to both the package dir and the online build's install
    prefix (``.`` for the package root, ``mlir/_mlir_libs`` for the family).
    """

    stem: str
    subdir: str


# All native modules served by online compilation. A single CMake invocation
# (ptoas/_online/CMakeLists.txt) produces every one of them.
_MEMBERS = {
    "ptoas._core": _MemberSpec("_core", "."),
    "ptoas.mlir._mlir_libs._mlir": _MemberSpec("_mlir", "mlir/_mlir_libs"),
    "ptoas.mlir._mlir_libs._mlirDialectsLLVM": _MemberSpec(
        "_mlirDialectsLLVM", "mlir/_mlir_libs"
    ),
    "ptoas.mlir._mlir_libs._site_initialize_0": _MemberSpec(
        "_site_initialize_0", "mlir/_mlir_libs"
    ),
}


@dataclasses.dataclass(frozen=True)
class _BuildGroup:
    """A set of native extensions compiled and installed together.

    The online CMake project configures every target, but each build request
    only builds+installs the group it needs (``cmake --build --target`` +
    ``cmake --install --component``). This decouples the CLI's ``_core`` build
    from the ``ptoas.mlir`` family build: ``_core`` compiles cleanly against any
    supported pybind11, whereas the family fails against pybind11 >= 3.0.2 (its
    upstream bindings use ``def_property`` + ``keep_alive``). Building them
    separately keeps a ``_core``-only request from being aborted by a family
    compile error, and lets the family failure surface as the actionable
    pybind11 gate (see ``_PythonContext``) instead of a raw C++ static_assert.
    Note the CLI still imports ``ptoas.mlir.ir`` at ``_core`` load time, so it
    remains a runtime dependency of the family regardless of this build split.

    ``members`` are the intercepted fullnames; ``targets`` the CMake target
    names; ``component`` the install COMPONENT; ``requires_family_pybind11``
    gates the family's stricter pybind11 upper bound.
    """

    members: Tuple[str, ...]
    targets: Tuple[str, ...]
    component: str
    requires_family_pybind11: bool


_CORE_GROUP = _BuildGroup(
    members=("ptoas._core",),
    targets=("_core",),
    component="core",
    requires_family_pybind11=False,
)
_FAMILY_GROUP = _BuildGroup(
    members=(
        "ptoas.mlir._mlir_libs._mlir",
        "ptoas.mlir._mlir_libs._mlirDialectsLLVM",
        "ptoas.mlir._mlir_libs._site_initialize_0",
    ),
    targets=("_mlir", "_mlirDialectsLLVM", "_site_initialize_0"),
    component="family",
    requires_family_pybind11=True,
)


# The ``ptoas.mlir`` family is a set of extensions that must be observed as a
# coherent unit: they are built by one CMake invocation and an importer loads
# all three, so a reader must never mix members from two different builds (an
# ABI/symbol mismatch or init failure). Rather than publish them member by
# member (three separate os.replace, with an observable half-published window),
# a family build is installed into a per-build *generation* directory and made
# live by a single atomic rename of a marker file. Readers resolve every family
# member through the one marker, so they always see a single, fully-present
# generation. ``_core`` is a single member and stays on the flat layout (its
# lone os.replace is already atomic).
_FAMILY_GEN_ROOT = ".ptoas_family"   # subdir (under a base dir) holding generations
_FAMILY_GEN_MARKER = "current"       # file naming the live generation dir
_FAMILY_GEN_PREFIX = "gen."          # generation dir name prefix (see mkdtemp)
_FAMILY_SPECS = frozenset(_MEMBERS[m] for m in _FAMILY_GROUP.members)


def _group_for(fullname: str) -> _BuildGroup:
    """Return the build group that produces ``fullname``."""
    return _CORE_GROUP if fullname == _QUALIFIED_MODULE else _FAMILY_GROUP


def _package_dir() -> Path:
    """Directory of the installed (or editable) ``ptoas`` package."""
    return Path(__file__).parent.resolve()


def _find_shipped_lib_dir(pkg_dir: Path) -> Optional[Path]:
    """Locate the directory holding libPTOASCompiler.{so,dylib}.

    The DSO is installed alongside the prebuilt extensions with the MLIR shared
    libs; probe the package dir and the standard MLIR libs subdir.
    """
    candidates = [pkg_dir, pkg_dir / "mlir" / "_mlir_libs"]
    exts = (".so", ".dylib")
    for cand in candidates:
        if not cand.is_dir():
            continue
        if any((cand / f"{stem}{ext}").exists() for stem in _DSO_STEMS for ext in exts):
            return cand
    return None


def _find_shipped_llvmsupport(pkg_dir: Path) -> Optional[Path]:
    """Return the full path to the shipped shared libLLVMSupport, or ``None``.

    The family extensions link LLVMSupport, but a name-based ``-lLLVMSupport``
    breaks once ``auditwheel``/``delocate`` relocates and hash-mangles the
    external DSO out of the package into a sibling ``ptoas.libs`` / ``.dylibs``
    dir (e.g. ``libLLVMSupport-<hash>.so.19.1``). Discover the real file so the
    online build can link it by full path. Probe, in order: the in-package MLIR
    libs dir and package root (unmangled, e.g. an editable build), then the
    relocation dirs with a glob (mangled).
    """
    exact = [
        pkg_dir / "mlir" / "_mlir_libs" / "libLLVMSupport.so",
        pkg_dir / "mlir" / "_mlir_libs" / "libLLVMSupport.dylib",
        pkg_dir / "libLLVMSupport.so",
        pkg_dir / "libLLVMSupport.dylib",
    ]
    for cand in exact:
        if cand.exists():
            return cand
    # auditwheel (Linux) -> <dist>.libs next to the package; delocate (macOS) ->
    # <pkg>/.dylibs. Names are hash-mangled, so glob.
    glob_dirs = [pkg_dir.parent / "ptoas.libs", pkg_dir / ".dylibs"]
    for d in glob_dirs:
        if not d.is_dir():
            continue
        for pattern in ("libLLVMSupport*.so*", "libLLVMSupport*.dylib"):
            hits = sorted(d.glob(pattern))
            if hits:
                return hits[0]
    return None


class _CMakeContext:
    """Locate cmake and drive its configure/build/install phases."""

    @dataclasses.dataclass
    class CompileContext:
        src_dir: Path
        tmp_dir: Path
        install_prefix: Path
        cfg_args: Tuple[str, ...] = ()
        build_type: str = "Release"
        build_job_num: int = 32
        capture_output: bool = True
        build_targets: Tuple[str, ...] = ()
        install_component: str = ""

        def run_cmd(self, cmd: List[str]):
            ret = subprocess.run(
                cmd,
                text=True,
                encoding="utf-8",
                capture_output=self.capture_output,
                check=not self.capture_output,
            )
            if ret.returncode != 0 and self.capture_output:
                _log.error("cmd: %s, ret: %s", shlex.join(cmd), ret.returncode)
                _log.error("stdout:\n%s", ret.stdout)
                _log.error("stderr:\n%s", ret.stderr)
            ret.check_returncode()

    def __init__(self):
        self.cmake = self._which_cmake()
        if self.cmake is None:
            raise RuntimeError(
                "Can not find cmake, please check your environment.\n"
                "Hint: install system-level cmake (e.g. `apt install cmake` or `yum install cmake`), "
                "the cmake pip package is NOT a valid substitute."
            )

    @staticmethod
    def _which_cmake() -> Optional[Path]:
        from ._which_cmake import which_cmake

        return which_cmake()

    def compile(self, ctx: "CompileContext") -> Path:
        build_dir = Path(ctx.tmp_dir, "build")
        if build_dir.exists():
            shutil.rmtree(build_dir)
        build_dir.mkdir(parents=True)
        cmd = [
            str(self.cmake),
            "-S", str(ctx.src_dir),
            "-B", str(build_dir),
            f"-DCMAKE_BUILD_TYPE={ctx.build_type}",
            f"-DCMAKE_INSTALL_PREFIX={ctx.install_prefix}",
            *ctx.cfg_args,
        ]
        ctx.run_cmd(cmd=cmd)
        cmd = [str(self.cmake), "--build", str(build_dir)]
        if ctx.build_targets:
            cmd += ["--target", *ctx.build_targets]
        if ctx.build_job_num:
            cmd += ["-j", str(ctx.build_job_num)]
        ctx.run_cmd(cmd=cmd)
        cmd = [str(self.cmake), "--install", str(build_dir), "--prefix", str(ctx.install_prefix)]
        if ctx.install_component:
            cmd += ["--component", ctx.install_component]
        ctx.run_cmd(cmd=cmd)
        return ctx.install_prefix


class _PythonContext:
    """Validate the target interpreter and locate its pybind11 cmake package."""

    _PYBIND11_MIN_VERSION = (2, 13, 6)
    # pybind11 3.0.2 added a static_assert forbidding keep_alive on the
    # def_property family. The upstream MLIR bindings (IRCore.cpp et al., staged
    # into the family sources) rely on that pattern, so the family fails to
    # compile against pybind11 >= 3.0.2. _core is unaffected. See
    # gen_mlir_family_headers.sh / online_template for how the family is built.
    _PYBIND11_KEEP_ALIVE_BROKEN = (3, 0, 2)
    # Python 3.14 support landed in pybind11 3.0.0; earlier releases fail to
    # compile against a 3.14 interpreter. Combined with the family's < 3.0.2
    # upper bound, the ptoas.mlir family only builds against pybind11
    # 3.0.0 / 3.0.1 on 3.14. Used to warn (not hard-fail) since the effective
    # pybind11 floor is version-specific.
    _PYBIND11_PY314_MIN_VERSION = (3, 0, 0)
    _PY314_MINOR = 14

    def __init__(self, require_family_pybind11: bool = False):
        self.require_family_pybind11 = require_family_pybind11
        self.minor: int = 0
        self.pybind11_cmake_dir: Optional[Path] = None
        self._init_minor_version()
        self._init_development_component()
        self._init_pip_mod_pybind11()

    @staticmethod
    def _init_development_component():
        python_h = Path(sysconfig.get_path("include")) / "Python.h"
        if not python_h.exists():
            raise RuntimeError(
                f"Python development headers not found (expected {python_h}).\n"
                "Hint: install python3-dev (e.g. `apt install python3-dev` or `yum install python3-devel`)."
            )

    def _init_minor_version(self):
        minor = int(sys.version_info.minor)
        if minor < 8:
            raise RuntimeError(
                f"Python version 3.{minor} is not supported for online compilation, require >= 3.8.\n"
                "Hint: use a Python 3.8+ interpreter."
            )
        self.minor = minor

    def _init_pip_mod_pybind11(self):
        try:
            import pybind11
        except ImportError as e:
            raise RuntimeError(
                "pybind11 pip package not found.\nHint: install it with `pip install pybind11>=2.13.6`."
            ) from e
        import re

        ver_match = re.match(r"(\d+)\.(\d+)\.(\d+)", pybind11.__version__)
        if not ver_match:
            raise RuntimeError(
                f"Can't parse pybind11 version: {pybind11.__version__}.\n"
                "Hint: install it with `pip install pybind11>=2.13.6`."
            )
        current_ver = tuple(int(x) for x in ver_match.groups())
        if current_ver < self._PYBIND11_MIN_VERSION:
            raise RuntimeError(
                f"pybind11 version {pybind11.__version__} is too old, require >= 2.13.6.\n"
                "Hint: upgrade it with `pip install pybind11>=2.13.6`."
            )
        if (
            self.minor >= self._PY314_MINOR
            and current_ver < self._PYBIND11_PY314_MIN_VERSION
        ):
            _log.warning(
                "Python 3.%d requires a pinned pybind11 to build the online "
                "extensions: only 3.0.0 or 3.0.1 work. It needs pybind11 >= 3.0.0, "
                "while the ptoas.mlir family additionally requires pybind11 < 3.0.2 "
                "(def_property + keep_alive is rejected from 3.0.2 on), leaving 3.0.0 "
                "/ 3.0.1 as the only versions that build the full Python API. Detected "
                "pybind11 %s will likely fail to compile.\n"
                "Hint: pin it with `pip install 'pybind11==3.0.1'` (or 3.0.0).",
                self.minor,
                pybind11.__version__,
            )
        if self.require_family_pybind11 and current_ver >= self._PYBIND11_KEEP_ALIVE_BROKEN:
            if self.minor >= self._PY314_MINOR:
                downgrade_hint = (
                    "Hint: on Python 3.%d pin pybind11 to the only working versions with "
                    "`pip install 'pybind11==3.0.1'` (or 3.0.0); versions < 3.0.0 do not "
                    "support 3.14 and >= 3.0.2 break the bindings." % self.minor
                )
            else:
                downgrade_hint = (
                    "Hint: install a compatible pybind11 with "
                    "`pip install 'pybind11>=2.13.6,<3'` "
                    "(or the last working 3.x, `pip install pybind11==3.0.1`)."
                )
            raise RuntimeError(
                f"pybind11 version {pybind11.__version__} is incompatible with the "
                "ptoas.mlir Python bindings.\n"
                "The upstream MLIR bindings use def_property + keep_alive, which "
                "pybind11 >= 3.0.2 rejects at compile time (\"def_property family does "
                "not currently support keep_alive\").\n"
                f"{downgrade_hint}\n"
                "Note: the ptoas CLI itself does NOT need these bindings and works on any "
                "supported pybind11; this only affects the ptodsl / ptoas.mlir Python API."
            )
        pybind11_dir = Path(pybind11.get_cmake_dir()).resolve()
        if not pybind11_dir or not pybind11_dir.exists():
            raise RuntimeError("pybind11 cmake dir empty.\nHint: install it with `pip install pybind11>=2.13.6`.")
        self.pybind11_cmake_dir = pybind11_dir


class BuildOnlineCoreManager:
    """Process-wide singleton that (re)builds the online pybind11 extensions.

    Fast-path callers should let the meta path finder in ``_loader`` serve the
    prebuilt in-package binary; this manager is the fallback taken on an ABI
    mismatch. ``get_member_so`` compiles the whole family once (guarded by a
    cross-thread lock and a cross-process flock) and returns the requested
    member's ``.so`` from the build cache.
    """

    _instances: dict = {}
    _new_lock: threading.Lock = threading.Lock()
    _compile_lock: threading.Lock = threading.Lock()
    _FLOCK_TIMEOUT: int = 300  # seconds a process waits for a peer's compile

    def __new__(cls):
        if cls not in cls._instances:
            with cls._new_lock:
                if cls not in cls._instances:
                    cls._instances[cls] = super().__new__(cls)
        return cls._instances[cls]

    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        from importlib import metadata

        self.pkg_dir: Path = _package_dir()
        try:
            self.version: str = metadata.version("ptoas")
        except Exception:
            self.version = ""
        self._online_dir: Path = self.pkg_dir / "_online"
        self._cache_dir_value: Optional[Path] = None
        ver_info = sys.version_info
        self._lock_prefix: str = f".ptoas_online_build.cp{ver_info.major}{ver_info.minor}"
        self._initialized = True

    # -- stateless helpers -------------------------------------------------

    @staticmethod
    def _try_acquire_lock(lock_fd) -> bool:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except BlockingIOError:
            return False

    @staticmethod
    def _member_in_dir(spec: _MemberSpec, base_dir: Path) -> Optional[Path]:
        d = base_dir / spec.subdir
        for suffix in importlib.machinery.EXTENSION_SUFFIXES:
            cand = d / f"{spec.stem}{suffix}"
            if cand.exists():
                return cand
        return None

    @staticmethod
    def _current_family_gen(base_dir: Path) -> Optional[Path]:
        """Return the live family generation dir under ``base_dir``, or ``None``.

        The marker records only a bare generation dir name; reject anything with
        a path separator (defence in depth against a tampered marker).
        """
        marker = base_dir / _FAMILY_GEN_ROOT / _FAMILY_GEN_MARKER
        try:
            name = marker.read_text(encoding="utf-8").strip()
        except OSError:
            return None
        if not name or name in (".", "..") or os.path.basename(name) != name:
            return None
        gen_dir = base_dir / _FAMILY_GEN_ROOT / name
        return gen_dir if gen_dir.is_dir() else None

    # -- public entry ------------------------------------------------------

    def get_member_so(self, fullname: str, force_rebuild: bool = False) -> Path:
        """Return a loadable ``.so`` for ``fullname``, compiling if necessary.

        Only the build group that produces ``fullname`` is compiled: a CLI
        (``ptoas._core``) request never triggers the ``ptoas.mlir`` family build,
        so the CLI works even on an interpreter whose pybind11 cannot compile the
        family.

        ``force_rebuild`` is set by the loader when an in-package prebuilt
        matched this interpreter's ABI suffix but failed to *load* (a corrupt or
        subtly-incompatible binary). In that case the pkg-dir copy is untrusted:
        we prefer a distinct online build from the cache dir, and otherwise force
        a fresh compile into the cache dir rather than handing back the broken
        pkg-dir binary that ``_find_member_so`` would return first.
        """
        spec = _MEMBERS[fullname]
        cache_dir = self._cache_dir()
        found = self._resolve_member(spec, cache_dir, force_rebuild)
        if found is not None:
            return found
        group = _group_for(fullname)
        with self._compile_lock:
            found = self._resolve_member(spec, cache_dir, force_rebuild)
            if found is not None:
                return found
            self._ensure_compiled_locked(group, force_rebuild=force_rebuild)
            # A forced rebuild always installs into the cache dir; the pkg-dir
            # copy is the very binary that failed to load, so resolve from the
            # cache dir only. The normal path keeps the pkg-dir-first search.
            if force_rebuild:
                so = self._published_member_in_dir(spec, cache_dir)
            else:
                so = self._find_member_so(spec, cache_dir)
            if so is None:
                raise RuntimeError(
                    f"Online compilation finished but {spec.stem} not found.\n"
                    f"Cache directory: {cache_dir} (subdir {spec.subdir!r})\n"
                    f"Searched suffixes: {importlib.machinery.EXTENSION_SUFFIXES}"
                )
            return so

    def _published_member_in_dir(self, spec: _MemberSpec, base_dir: Path) -> Optional[Path]:
        """Resolve a published member in ``base_dir``, honoring family generations.

        Family members are served from the live generation (so all three come
        from one coherent build); if no generation is published yet, fall back
        to the flat layout (a shipped prebuilt lives flat at
        ``mlir/_mlir_libs``). ``_core`` always uses the flat layout.
        """
        if spec in _FAMILY_SPECS:
            gen_dir = self._current_family_gen(base_dir)
            if gen_dir is not None:
                cand = self._member_in_dir(spec, gen_dir)
                if cand is not None:
                    return cand
        return self._member_in_dir(spec, base_dir)

    def _resolve_member(self, spec: _MemberSpec, cache_dir: Path, force_rebuild: bool) -> Optional[Path]:
        """Best-effort lookup honoring ``force_rebuild``'s distrust of pkg_dir.

        Non-forced: normal pkg-dir-first search. Forced: only accept a distinct
        cache-dir build (never the pkg-dir prebuilt that just failed to load);
        when the cache dir *is* the pkg dir, the only copy is the broken one, so
        report a miss to force a recompile.
        """
        if not force_rebuild:
            return self._find_member_so(spec, cache_dir)
        if cache_dir == self.pkg_dir:
            return None
        return self._published_member_in_dir(spec, cache_dir)

    def _lock_name_for(self, group: _BuildGroup) -> str:
        """Per-group flock filename so core and family builds don't block each other."""
        return f"{self._lock_prefix}.{group.component}.lock"

    # -- compile state machine (called holding self._compile_lock) ---------

    def _ensure_compiled_locked(self, group: _BuildGroup, force_rebuild: bool = False):
        if not self._online_dir.exists():
            raise RuntimeError(
                "The version-sensitive extensions are unavailable for this "
                f"interpreter and no online sources were shipped (expected {self._online_dir}).\n"
                "Hint: install a wheel built with PTOAS_ENABLE_ONLINE_CORE_COMPILE=ON, "
                "or use the matching prebuilt interpreter."
            )

        # Cross-user cooperation: if someone is compiling into pkg_dir, wait.
        # Skipped on a forced rebuild: the pkg-dir binaries are exactly the ones
        # that failed to load, so a peer's pkg-dir build is not what we want.
        if not force_rebuild and self._wait_for_pkg_dir_compilation(group):
            return

        target_dir = self._cache_dir()
        lock_path = target_dir / self._lock_name_for(group)
        lock_fd = None
        try:
            lock_fd = open(lock_path, "w")
            if self._try_acquire_lock(lock_fd):
                self._compile_into(target_dir, group)
            else:
                self._wait_and_compile(lock_fd, target_dir, lock_path, group, force_rebuild)
        finally:
            # Unlink while still holding the lock to avoid an inode-reuse race
            # (see the pypto original for the detailed reasoning).
            try:
                lock_path.unlink(missing_ok=True)
            except OSError:
                pass
            if lock_fd is not None:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                except OSError:
                    pass
                lock_fd.close()

    def _poll_until(self, predicate, timeout: Optional[float] = None, interval: float = 1.0) -> bool:
        if timeout is None:
            timeout = self._FLOCK_TIMEOUT
        deadline = time.monotonic() + timeout
        while True:
            if predicate():
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(interval)

    def _wait_for_pkg_dir_compilation(self, group: _BuildGroup) -> bool:
        lock_path = self.pkg_dir / self._lock_name_for(group)
        if not os.access(self.pkg_dir, os.R_OK):
            return False
        if not lock_path.exists():
            return False
        _log.info("Detected compilation in progress at %s, waiting...", lock_path)
        if not self._poll_until(lambda: not lock_path.exists()):
            _log.warning("Timeout waiting for pkg_dir compilation")
            return False
        return self._all_members_present(self.pkg_dir, group)

    def _wait_and_compile(
        self,
        lock_fd,
        target_dir: Path,
        lock_path: Path,
        group: _BuildGroup,
        force_rebuild: bool = False,
    ):
        _log.info("Waiting for compilation by another process (lock: %s)...", lock_path)
        if self._poll_until(lambda: self._try_acquire_lock(lock_fd)):
            _log.info("Acquired compilation lock after waiting")
            # A peer just finished. Accept its output only when we can trust it:
            # on a forced rebuild whose cache dir is the pkg dir, "present" means
            # the broken prebuilt, so recompile regardless.
            trust_present = not (force_rebuild and target_dir == self.pkg_dir)
            if trust_present and self._all_members_present(target_dir, group):
                return
            self._compile_into(target_dir, group)
            return
        _log.warning("Timeout waiting for compilation lock, will compile independently")
        self._compile_into(target_dir, group)

    def _compile_into(self, target_dir: Path, group: _BuildGroup):
        pyenv_ctx = _PythonContext(require_family_pybind11=group.requires_family_pybind11)
        cmake_ctx = _CMakeContext()

        shipped_lib_dir = _find_shipped_lib_dir(self.pkg_dir)
        if shipped_lib_dir is None:
            raise RuntimeError(
                "Can not locate the shipped libPTOASCompiler DSO under "
                f"{self.pkg_dir}; the wheel appears to be incomplete."
            )

        target_dir.mkdir(parents=True, exist_ok=True)
        # Install into a staging prefix on the SAME filesystem as the cache dir,
        # then publish atomically: a single-member group (``_core``) is moved
        # into place with one os.replace, while the multi-member family is
        # published as one generation via a single marker rename (see
        # _publish_group). A concurrent reader never observes a half-written .so
        # or a partially-published family.
        staging = target_dir / f".online-staging.{os.getpid()}"
        shutil.rmtree(staging, ignore_errors=True)
        try:
            with tempfile.TemporaryDirectory(prefix=f".ptoas_online_build.{os.getpid()}.") as tmp_dir:
                cfg_args = [
                    f"-DPython3_EXECUTABLE={sys.executable}",
                    f"-DPython3_EXECUTABLE_VERSION=3.{pyenv_ctx.minor}",
                    f"-DPython3_MOD_PYBIND11_CMAKE_DIR={pyenv_ctx.pybind11_cmake_dir}",
                    f"-DPTOAS_SHIPPED_LIB_DIR={shipped_lib_dir}",
                ]
                # The family extensions link libLLVMSupport by full path because
                # auditwheel/delocate hash-mangle the relocated DSO's name.
                llvmsupport = _find_shipped_llvmsupport(self.pkg_dir)
                if llvmsupport is not None:
                    cfg_args.append(f"-DPTOAS_LLVMSUPPORT_LIB={llvmsupport}")
                compile_ctx = _CMakeContext.CompileContext(
                    src_dir=self._online_dir,
                    tmp_dir=Path(tmp_dir),
                    install_prefix=staging,
                    cfg_args=tuple(cfg_args),
                    build_targets=group.targets,
                    install_component=group.component,
                )
                cmake_ctx.compile(ctx=compile_ctx)
            self._publish_group(staging, target_dir, group)
        finally:
            shutil.rmtree(staging, ignore_errors=True)

        _log.info("Compiled and installed online %s extensions to %s", group.component, target_dir)

    def _publish_group(self, staging: Path, target_dir: Path, group: _BuildGroup):
        """Publish a freshly built group from staging into ``target_dir``.

        The family is published as one atomic generation (all members visible at
        once, via a single marker rename); any other group (only ``_core`` today)
        uses the flat per-member move, which is already atomic for one member.
        """
        if group is _FAMILY_GROUP:
            self._publish_family_generation(staging, target_dir, group)
        else:
            self._publish_members(staging, target_dir, group)

    def _publish_members(self, staging: Path, target_dir: Path, group: _BuildGroup):
        """Atomically move each built member from the staging prefix into place.

        The staging prefix mirrors the install layout (``<name>`` for _core,
        ``mlir/_mlir_libs/<name>`` for the family), so each member is resolved
        with ``_member_in_dir`` and ``os.replace``d to its final path -- an
        atomic rename within the same filesystem. Used for single-member groups;
        the family goes through _publish_family_generation for atomicity across
        its three members.
        """
        for member in group.members:
            spec = _MEMBERS[member]
            built = self._member_in_dir(spec, staging)
            if built is None:
                raise RuntimeError(
                    f"Online compile did not produce {spec.stem} under {staging} "
                    f"(subdir {spec.subdir!r})."
                )
            dst_dir = target_dir / spec.subdir
            dst_dir.mkdir(parents=True, exist_ok=True)
            os.replace(str(built), str(dst_dir / built.name))

    def _publish_family_generation(self, staging: Path, target_dir: Path, group: _BuildGroup):
        """Publish the family as a single generation made live by one atomic rename.

        Every member is moved into a fresh, uniquely-named generation dir; only
        after all of them are in place is the ``current`` marker flipped with a
        single ``os.replace``. A reader resolving members through the marker (see
        ``_published_member_in_dir``) therefore sees either the previous
        generation (fully present) or the new one (fully present) -- never a mix,
        a missing member, or a half-written file.
        """
        family_root = target_dir / _FAMILY_GEN_ROOT
        family_root.mkdir(parents=True, exist_ok=True)
        # Drop superseded generations first, but keep whatever the marker names
        # now so a concurrent reader mid-import still resolves a complete set.
        self._gc_family_generations(family_root)
        gen_dir = Path(tempfile.mkdtemp(prefix=_FAMILY_GEN_PREFIX, dir=family_root))
        for member in group.members:
            spec = _MEMBERS[member]
            built = self._member_in_dir(spec, staging)
            if built is None:
                raise RuntimeError(
                    f"Online compile did not produce {spec.stem} under {staging} "
                    f"(subdir {spec.subdir!r})."
                )
            dst_dir = gen_dir / spec.subdir
            dst_dir.mkdir(parents=True, exist_ok=True)
            os.replace(str(built), str(dst_dir / built.name))
        # Flip the marker last: a single atomic rename over the same filesystem.
        marker = family_root / _FAMILY_GEN_MARKER
        tmp_marker = family_root / f".{_FAMILY_GEN_MARKER}.{os.getpid()}.tmp"
        tmp_marker.write_text(gen_dir.name, encoding="utf-8")
        os.replace(str(tmp_marker), str(marker))

    def _gc_family_generations(self, family_root: Path):
        """Best-effort removal of stale generation dirs, keeping the live one.

        The generation named by the marker is preserved so a lock-free reader
        still resolves a complete set; other generations are left over from an
        earlier build and safe to delete. A reader that grabbed a now-removed
        path self-heals via the loader's one-shot rebuild fallback.
        """
        try:
            keep = self._current_family_gen(family_root.parent)
        except OSError:
            keep = None
        keep_name = keep.name if keep is not None else None
        try:
            entries = list(family_root.iterdir())
        except OSError:
            return
        for old in entries:
            if not old.name.startswith(_FAMILY_GEN_PREFIX) or old.name == keep_name:
                continue
            if old.is_dir():
                shutil.rmtree(old, ignore_errors=True)

    # -- cache dir + discovery --------------------------------------------

    def _cache_dir(self) -> Path:
        if self._cache_dir_value is not None:
            return self._cache_dir_value
        self._cache_dir_value = self._compute_cache_dir()
        return self._cache_dir_value

    def _compute_cache_dir(self) -> Path:
        # Prefer the package dir when it belongs to us and is actually writable.
        try:
            if self.pkg_dir.stat().st_uid == os.getuid():
                test_file = self.pkg_dir / f".ptoas_writable_test.{os.getpid()}"
                try:
                    test_file.touch()
                    test_file.unlink()
                    return self.pkg_dir
                except OSError:
                    pass
        except OSError:
            pass

        cache_dir = None
        xdg_cache = os.environ.get("XDG_CACHE_HOME")
        if xdg_cache:
            if Path(xdg_cache).is_absolute():
                cache_dir = Path(xdg_cache)
            else:
                _log.warning("XDG_CACHE_HOME=%s is not absolute, fallback to ~/.cache", xdg_cache)
        cache_dir = cache_dir if cache_dir else Path.home() / ".cache"
        cache_dir = cache_dir / "cann" / "ptoas"
        if self.version:
            cache_dir = cache_dir / self.version
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    def _find_member_so(self, spec: _MemberSpec, cache_dir: Optional[Path] = None) -> Optional[Path]:
        found = self._published_member_in_dir(spec, self.pkg_dir)
        if found is not None:
            return found
        if cache_dir is not None and cache_dir != self.pkg_dir:
            return self._published_member_in_dir(spec, cache_dir)
        return None

    def _all_members_present(self, base_dir: Path, group: _BuildGroup) -> bool:
        return all(
            self._published_member_in_dir(_MEMBERS[member], base_dir) is not None
            for member in group.members
        )

    def _find_core_so(self, cache_dir: Optional[Path] = None) -> Tuple[bool, Optional[Path]]:
        so = self._find_member_so(_MEMBERS[_QUALIFIED_MODULE], cache_dir)
        if so is not None:
            _log.info("Found _core: %s", so)
            return True, so
        return False, None


def get_or_build_member(fullname: str, force_rebuild: bool = False) -> Path:
    """Return a loadable ``.so`` path for ``fullname`` (compiling if needed).

    ``force_rebuild`` distrusts the in-package prebuilt (used by the loader when
    an ABI-matching prebuilt failed to import) and resolves/compiles from the
    cache dir instead.
    """
    return BuildOnlineCoreManager().get_member_so(fullname, force_rebuild=force_rebuild)
