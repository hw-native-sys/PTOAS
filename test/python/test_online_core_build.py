#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Regression tests for the ptoas._core online-compilation helpers.

These load ``ptodsl/ptoas/_which_cmake.py`` and ``ptodsl/ptoas/_build_online.py``
in isolation (no ``ptoas`` package import, no native ``_core``, no cmake), so the
pure Python logic that selects a real cmake, locates the shipped DSO, computes
the cache dir, and discovers a built ``_core`` is exercised on any interpreter.
"""

import importlib.machinery
import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


_REPO_ROOT = Path(__file__).resolve().parents[2]
_PTOAS_PKG = _REPO_ROOT / "ptodsl" / "ptoas"


def _load_isolated(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_WHICH_CMAKE = _load_isolated("_ptoas_which_cmake_probe", _PTOAS_PKG / "_which_cmake.py")
_BUILD_ONLINE = _load_isolated("_ptoas_build_online_probe", _PTOAS_PKG / "_build_online.py")


class NativeExecutableDetectionTests(unittest.TestCase):
    def _write(self, directory: Path, name: str, data: bytes, executable: bool = True) -> Path:
        target = directory / name
        target.write_bytes(data)
        if executable:
            target.chmod(0o755)
        return target

    def test_elf_and_macho_magics_are_native(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            elf = self._write(tmp_dir, "elf_bin", b"\x7fELF" + b"\x00" * 16)
            macho = self._write(tmp_dir, "macho_bin", b"\xcf\xfa\xed\xfe" + b"\x00" * 16)
            self.assertTrue(_WHICH_CMAKE._is_native_executable(elf))
            self.assertTrue(_WHICH_CMAKE._is_native_executable(macho))

    def test_pip_console_script_shim_is_not_native(self):
        with tempfile.TemporaryDirectory() as tmp:
            shim = self._write(
                Path(tmp), "cmake", b"#!/usr/bin/env python\nprint('shim')\n"
            )
            self.assertFalse(_WHICH_CMAKE._is_native_executable(shim))

    def test_tiny_and_missing_files_are_not_native(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            tiny = self._write(tmp_dir, "tiny", b"\x7fEL")  # <= 4 bytes
            self.assertFalse(_WHICH_CMAKE._is_native_executable(tiny))
            self.assertFalse(_WHICH_CMAKE._is_native_executable(tmp_dir / "does_not_exist"))


class WhichCmakeSelectionTests(unittest.TestCase):
    def test_skips_shim_dir_and_returns_native_binary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shim_dir = root / "shimbin"
            native_dir = root / "realbin"
            shim_dir.mkdir()
            native_dir.mkdir()
            shim = shim_dir / "cmake"
            shim.write_bytes(b"#!/usr/bin/env python\n")
            shim.chmod(0o755)
            native = native_dir / "cmake"
            native.write_bytes(b"\x7fELF" + b"\x00" * 32)
            native.chmod(0o755)

            # Shim directory intentionally first on PATH: the selector must skip
            # it and return the native cmake behind it.
            new_path = os.pathsep.join([str(shim_dir), str(native_dir)])
            with mock.patch.dict(os.environ, {"PATH": new_path}):
                found = _WHICH_CMAKE.which_cmake()
            self.assertEqual(found, native.resolve())

    def test_returns_none_when_only_shim_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            shim_dir = Path(tmp)
            shim = shim_dir / "cmake"
            shim.write_bytes(b"#!/usr/bin/env python\n")
            shim.chmod(0o755)
            with mock.patch.dict(os.environ, {"PATH": str(shim_dir)}):
                self.assertIsNone(_WHICH_CMAKE.which_cmake())


class ShippedLibDirTests(unittest.TestCase):
    def test_finds_dso_in_package_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            pkg = Path(tmp)
            (pkg / "libPTOASCompiler.so").write_bytes(b"\x00")
            self.assertEqual(_BUILD_ONLINE._find_shipped_lib_dir(pkg), pkg)

    def test_finds_dso_in_mlir_libs_subdir(self):
        with tempfile.TemporaryDirectory() as tmp:
            pkg = Path(tmp)
            libs = pkg / "mlir" / "_mlir_libs"
            libs.mkdir(parents=True)
            (libs / "libPTOASCompiler.dylib").write_bytes(b"\x00")
            self.assertEqual(_BUILD_ONLINE._find_shipped_lib_dir(pkg), libs)

    def test_returns_none_when_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(_BUILD_ONLINE._find_shipped_lib_dir(Path(tmp)))


def _fresh_manager():
    """Return a BuildOnlineCoreManager with its process-wide singleton reset."""
    cls = _BUILD_ONLINE.BuildOnlineCoreManager
    cls._instances.pop(cls, None)
    return cls()


class CacheDirTests(unittest.TestCase):
    def test_prefers_writable_owned_package_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            pkg = Path(tmp)
            mgr = _fresh_manager()
            mgr.pkg_dir = pkg
            mgr._cache_dir_value = None
            self.assertEqual(mgr._compute_cache_dir(), pkg)

    def test_falls_back_to_xdg_cache_when_pkg_not_owned(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pkg = root / "pkg"
            xdg = root / "xdg"
            pkg.mkdir()
            xdg.mkdir()
            mgr = _fresh_manager()
            mgr.pkg_dir = pkg
            mgr.version = "9.9.9"
            mgr._cache_dir_value = None
            # Force the ownership check to fail so the pkg-dir branch is skipped.
            with mock.patch.object(_BUILD_ONLINE.os, "getuid", return_value=os.getuid() + 1), \
                    mock.patch.dict(os.environ, {"XDG_CACHE_HOME": str(xdg)}):
                result = mgr._compute_cache_dir()
            self.assertEqual(result, xdg / "cann" / "ptoas" / "9.9.9")
            self.assertTrue(result.is_dir())


class FindCoreSoTests(unittest.TestCase):
    def _core_name(self) -> str:
        return f"_core{importlib.machinery.EXTENSION_SUFFIXES[0]}"

    def test_finds_core_in_package_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            pkg = Path(tmp)
            so = pkg / self._core_name()
            so.write_bytes(b"\x00")
            mgr = _fresh_manager()
            mgr.pkg_dir = pkg
            found, path = mgr._find_core_so()
            self.assertTrue(found)
            self.assertEqual(path, so)

    def test_finds_core_in_cache_dir_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pkg = root / "pkg"
            cache = root / "cache"
            pkg.mkdir()
            cache.mkdir()
            so = cache / self._core_name()
            so.write_bytes(b"\x00")
            mgr = _fresh_manager()
            mgr.pkg_dir = pkg
            found, path = mgr._find_core_so(cache_dir=cache)
            self.assertTrue(found)
            self.assertEqual(path, so)

    def test_absent_core_reports_not_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = _fresh_manager()
            mgr.pkg_dir = Path(tmp)
            found, path = mgr._find_core_so()
            self.assertFalse(found)
            self.assertIsNone(path)


class ResolveMemberForceRebuildTests(unittest.TestCase):
    """`_resolve_member` must distrust the pkg-dir prebuilt on a forced rebuild."""

    def _core_spec(self):
        return _BUILD_ONLINE._MEMBERS[_BUILD_ONLINE._QUALIFIED_MODULE]

    def _core_name(self) -> str:
        return f"_core{importlib.machinery.EXTENSION_SUFFIXES[0]}"

    def test_non_forced_returns_pkg_dir_hit(self):
        with tempfile.TemporaryDirectory() as tmp:
            pkg = Path(tmp)
            (pkg / self._core_name()).write_bytes(b"\x00")
            mgr = _fresh_manager()
            mgr.pkg_dir = pkg
            got = mgr._resolve_member(self._core_spec(), pkg, force_rebuild=False)
            self.assertEqual(got, pkg / self._core_name())

    def test_forced_ignores_pkg_dir_when_cache_is_pkg(self):
        # The only copy lives in pkg_dir and it is the very binary that failed to
        # load; a forced resolve must report a miss so a recompile is triggered.
        with tempfile.TemporaryDirectory() as tmp:
            pkg = Path(tmp)
            (pkg / self._core_name()).write_bytes(b"\x00")
            mgr = _fresh_manager()
            mgr.pkg_dir = pkg
            got = mgr._resolve_member(self._core_spec(), pkg, force_rebuild=True)
            self.assertIsNone(got)

    def test_forced_prefers_distinct_cache_build_over_broken_pkg(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pkg = root / "pkg"
            cache = root / "cache"
            pkg.mkdir()
            cache.mkdir()
            (pkg / self._core_name()).write_bytes(b"\x00")  # broken prebuilt
            good = cache / self._core_name()
            good.write_bytes(b"\x00")
            mgr = _fresh_manager()
            mgr.pkg_dir = pkg
            got = mgr._resolve_member(self._core_spec(), cache, force_rebuild=True)
            self.assertEqual(got, good)


class PublishMembersTests(unittest.TestCase):
    """`_publish_members` atomically moves each built member into place."""

    def _core_name(self) -> str:
        return f"_core{importlib.machinery.EXTENSION_SUFFIXES[0]}"

    def test_moves_member_from_staging_to_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)
            group = _BUILD_ONLINE._group_for(_BUILD_ONLINE._QUALIFIED_MODULE)
            spec = _BUILD_ONLINE._MEMBERS[_BUILD_ONLINE._QUALIFIED_MODULE]
            staging = target / ".online-staging.test"
            (staging / spec.subdir).mkdir(parents=True)
            built = staging / spec.subdir / self._core_name()
            built.write_bytes(b"\x7fELF")
            mgr = _fresh_manager()
            mgr.pkg_dir = target
            mgr._publish_members(staging, target, group)
            published = target / spec.subdir / self._core_name()
            self.assertTrue(published.exists())
            self.assertEqual(published.read_bytes(), b"\x7fELF")
            self.assertFalse(built.exists())

    def test_raises_when_member_missing_from_staging(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)
            group = _BUILD_ONLINE._group_for(_BUILD_ONLINE._QUALIFIED_MODULE)
            staging = target / ".online-staging.test"
            staging.mkdir()
            mgr = _fresh_manager()
            mgr.pkg_dir = target
            with self.assertRaises(RuntimeError):
                mgr._publish_members(staging, target, group)


class FamilyGenerationPublishTests(unittest.TestCase):
    """The ptoas.mlir family publishes atomically as one generation."""

    def _family_group(self):
        return _BUILD_ONLINE._FAMILY_GROUP

    def _suffix(self) -> str:
        return importlib.machinery.EXTENSION_SUFFIXES[0]

    def _stage_family(self, staging: Path, payload: bytes):
        group = self._family_group()
        for member in group.members:
            spec = _BUILD_ONLINE._MEMBERS[member]
            d = staging / spec.subdir
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{spec.stem}{self._suffix()}").write_bytes(payload)

    def test_publishes_all_members_and_resolves_from_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)
            group = self._family_group()
            staging = target / ".online-staging.test"
            self._stage_family(staging, b"\x7fELFv1")
            mgr = _fresh_manager()
            mgr.pkg_dir = target
            mgr._publish_group(staging, target, group)

            # Every member resolves, all from the same live generation dir.
            gen = mgr._current_family_gen(target)
            self.assertIsNotNone(gen)
            for member in group.members:
                spec = _BUILD_ONLINE._MEMBERS[member]
                got = mgr._published_member_in_dir(spec, target)
                self.assertIsNotNone(got)
                # got == <gen>/mlir/_mlir_libs/<name>; its 3rd parent is <gen>.
                self.assertEqual(got.parent.parent.parent, gen)
                self.assertEqual(got.read_bytes(), b"\x7fELFv1")
            self.assertTrue(mgr._all_members_present(target, group))

    def test_flat_prebuilt_is_ignored_once_a_generation_is_live(self):
        # A flat (ABI-mismatched) prebuilt family may coexist with an online
        # generation; resolution must prefer the generation.
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)
            group = self._family_group()
            for member in group.members:
                spec = _BUILD_ONLINE._MEMBERS[member]
                flat = target / spec.subdir
                flat.mkdir(parents=True, exist_ok=True)
                (flat / f"{spec.stem}{self._suffix()}").write_bytes(b"PREBUILT")
            staging = target / ".online-staging.test"
            self._stage_family(staging, b"ONLINE")
            mgr = _fresh_manager()
            mgr.pkg_dir = target
            mgr._publish_group(staging, target, group)
            spec = _BUILD_ONLINE._MEMBERS[group.members[0]]
            got = mgr._published_member_in_dir(spec, target)
            self.assertEqual(got.read_bytes(), b"ONLINE")

    def test_republish_supersedes_and_gcs_old_generation(self):
        # The current + immediately-previous generation are retained (a lock-free
        # reader mid-import stays safe); older ones are GC'd on the next publish.
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)
            group = self._family_group()
            mgr = _fresh_manager()
            mgr.pkg_dir = target

            staging1 = target / ".online-staging.1"
            self._stage_family(staging1, b"GEN1")
            mgr._publish_group(staging1, target, group)
            gen1 = mgr._current_family_gen(target)

            staging2 = target / ".online-staging.2"
            self._stage_family(staging2, b"GEN2")
            mgr._publish_group(staging2, target, group)
            gen2 = mgr._current_family_gen(target)

            self.assertNotEqual(gen1, gen2)
            self.assertTrue(gen1.exists())  # previous generation retained
            spec = _BUILD_ONLINE._MEMBERS[group.members[0]]
            got = mgr._published_member_in_dir(spec, target)
            self.assertEqual(got.read_bytes(), b"GEN2")

            # A third publish supersedes gen2 and GCs the now-stale gen1.
            staging3 = target / ".online-staging.3"
            self._stage_family(staging3, b"GEN3")
            mgr._publish_group(staging3, target, group)
            self.assertFalse(gen1.exists())
            self.assertTrue(gen2.exists())  # still the immediately-previous one
            got = mgr._published_member_in_dir(spec, target)
            self.assertEqual(got.read_bytes(), b"GEN3")

    def test_tampered_marker_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)
            family_root = target / _BUILD_ONLINE._FAMILY_GEN_ROOT
            family_root.mkdir(parents=True)
            (family_root / _BUILD_ONLINE._FAMILY_GEN_MARKER).write_text("../evil")
            mgr = _fresh_manager()
            mgr.pkg_dir = target
            self.assertIsNone(mgr._current_family_gen(target))


if __name__ == "__main__":
    unittest.main()
