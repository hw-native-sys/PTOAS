#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Exercise CANN wheel delivery with real ELF dependencies and fresh processes."""

import hashlib
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import unittest
from unittest import mock

from wheel.wheelfile import WheelFile


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/package/harden_cann_wheel.py"
SPEC = importlib.util.spec_from_file_location("harden_cann_wheel", SCRIPT)
HARDENER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HARDENER)


@unittest.skipUnless(sys.platform.startswith("linux"), "ELF delivery is Linux-only")
class CannWheelHardeningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="ptoas-elf-test-")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        cls.payload = cls.root / "payload"
        cls.libs = cls.payload / "ptoas.libs"
        cls.native = cls.payload / "ptoas" / "native"
        cls.libs.mkdir(parents=True)
        cls.native.mkdir(parents=True)
        cls.cc = HARDENER._tool("cc")
        cls.platform_tag = sysconfig.get_platform().replace("-", "_").replace(".", "_")
        cls._build_fixture()
        cls.wheel = cls.root / f"ptoas-0.1-py3-none-{cls.platform_tag}.whl"
        HARDENER._pack(cls.payload, cls.wheel)

    @classmethod
    def _compile(cls, name, source, output, options):
        source_path = cls.root / name
        source_path.write_text(source, encoding="utf-8")
        subprocess.run([cls.cc, "-g", str(source_path), "-o", str(output), *options],
                       check=True, timeout=60)

    @classmethod
    def _build_fixture(cls):
        cls._compile("leaf.c", "int leaf(void) { return 41; }\n", cls.libs / "libleaf.so",
                     ["-shared", "-fPIC", "-Wl,-soname,libleaf.so", "-Wl,-rpath,/unavailable/build"])
        cls._compile("middle.c", "extern int leaf(void); int answer(void) { return leaf() + 1; }\n",
                     cls.libs / "libmiddle.so",
                     ["-shared", "-fPIC", "-Wl,-soname,libmiddle.so", "-L" + str(cls.libs),
                      "-lleaf", "-Wl,--disable-new-dtags,-rpath,$ORIGIN"])
        source = '''#include <Python.h>
extern int answer(void);
static PyObject *get_answer(PyObject *self, PyObject *args) { return PyLong_FromLong(answer()); }
static PyMethodDef methods[] = {{"answer", get_answer, METH_NOARGS, "Return an answer."}, {NULL}};
static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "_native", NULL, -1, methods};
PyMODINIT_FUNC PyInit__native(void) { return PyModule_Create(&module); }
'''
        cls._compile("extension.c", source, cls.native / "_native.so",
                     ["-shared", "-fPIC", "-I" + sysconfig.get_path("include"),
                      "-L" + str(cls.libs), "-lmiddle", "-Wl,-rpath,$ORIGIN/../../ptoas.libs"])
        (cls.payload / "ptoas" / "__init__.py").write_text("from .native._native import answer\n")
        cls._compile("cli.c", "extern int answer(void); int main(void) { return answer() != 42; }\n",
                     cls.payload / "ptoas" / "cli",
                     ["-L" + str(cls.libs), "-lmiddle", "-Wl,-rpath-link," + str(cls.libs),
                      "-Wl,-rpath,$ORIGIN/../ptoas.libs"])
        # A circular DT_NEEDED graph must remain loadable without an ordering
        # heuristic or Python preloader. The final A replaces its link-time stub.
        cycle = "int base(void) { return 40; }\n"
        cls._compile("a.c", cycle, cls.libs / "liba.so", ["-shared", "-fPIC", "-Wl,-soname,liba.so"])
        cls._compile("b.c", "extern int base(void); int b(void) { return base() + 2; }\n",
                     cls.libs / "libb.so", ["-shared", "-fPIC", "-Wl,-soname,libb.so",
                     "-L" + str(cls.libs), "-la", "-Wl,-rpath,$ORIGIN"])
        cls._compile("a.c", cycle + "extern int b(void); int a(void) { return b(); }\n",
                     cls.libs / "liba.so", ["-shared", "-fPIC", "-Wl,-soname,liba.so",
                     "-L" + str(cls.libs), "-lb", "-Wl,-rpath,$ORIGIN"])
        metadata = cls.payload / "ptoas-0.1.dist-info"
        metadata.mkdir()
        (metadata / "METADATA").write_text("Metadata-Version: 2.1\nName: ptoas\nVersion: 0.1\n")
        (metadata / "WHEEL").write_text(
            "Wheel-Version: 1.0\nRoot-Is-Purelib: false\nTag: py3-none-" + cls.platform_tag + "\n")

    def setUp(self):
        self.case = tempfile.TemporaryDirectory(dir=self.root)
        self.addCleanup(self.case.cleanup)
        self.output = Path(self.case.name) / "hardened"

    def _harden_and_extract(self):
        HARDENER.process_wheel(self.wheel, self.output)
        wheel = self.output / self.wheel.name
        installed = Path(self.case.name) / "install prefix with spaces"
        installed.mkdir()
        # WheelFile verifies all RECORD hashes while reading the repacked wheel.
        HARDENER._extract(wheel, installed)
        return installed

    def test_strip_relocate_and_load_without_environment(self):
        original_hash = hashlib.sha256(self.wheel.read_bytes()).digest()
        with self.assertRaisesRegex(ValueError, "unhardened ELF"):
            HARDENER.process_wheel(self.wheel)
        installed = self._harden_and_extract()
        self.assertEqual(hashlib.sha256(self.wheel.read_bytes()).digest(), original_hash)
        self.assertEqual(HARDENER.validate_tree(installed), 6)
        self.assertTrue((installed / "ptoas" / "cli").stat().st_mode & 0o111)
        env = {name: value for name, value in os.environ.items()
               if name not in ("LD_LIBRARY_PATH", "LD_PRELOAD", "PYTHONPATH")}
        code = ("import ctypes, sys; sys.path.insert(0, sys.argv[1]); "
                "import ptoas; assert ptoas.answer() == 42; "
                "assert ctypes.CDLL(sys.argv[1] + '/ptoas.libs/liba.so').a() == 42")
        subprocess.run([sys.executable, "-I", "-c", code, str(installed)],
                       check=True, env=env, cwd="/", timeout=30)
        subprocess.run([str(installed / "ptoas" / "cli")], check=True, env=env, cwd="/", timeout=30)
        # Both DT_RPATH and DT_RUNPATH occurred in the source fixture.
        self.assertIn("DT_RPATH", HARDENER.inspect_elf(self.libs / "libmiddle.so").search_paths)
        self.assertIn("DT_RUNPATH", HARDENER.inspect_elf(self.libs / "libleaf.so").search_paths)
        self.assertIn(".dynsym", HARDENER.inspect_elf(installed / "ptoas.libs/libmiddle.so").sections)

    def test_repeated_hardening_preserves_dependency_loading(self):
        HARDENER.process_wheel(self.wheel, self.output)
        second = Path(self.case.name) / "second"
        self.assertEqual(HARDENER.process_wheel(self.output / self.wheel.name, second), 6)
        self.assertEqual(HARDENER.process_wheel(second / self.wheel.name), 6)

    def test_missing_dependency_is_rejected(self):
        installed = self._harden_and_extract()
        (installed / "ptoas.libs/libleaf.so").unlink()
        with self.assertRaisesRegex(ValueError, "missing packaged dependency"):
            HARDENER.validate_tree(installed)

    def test_duplicate_soname_is_rejected(self):
        installed = self._harden_and_extract()
        shutil.copyfile(installed / "ptoas.libs/libleaf.so", installed / "ptoas/duplicate.so")
        with self.assertRaisesRegex(ValueError, "ambiguous packaged library"):
            HARDENER.validate_tree(installed)

    def test_materialized_library_aliases_remain_loadable(self):
        payload = Path(self.case.name) / "aliases"
        shutil.copytree(self.payload, payload)
        shutil.copyfile(payload / "ptoas.libs/libleaf.so", payload / "ptoas.libs/libleaf.so.1")
        wheel = Path(self.case.name) / self.wheel.name
        HARDENER._pack(payload, wheel)
        self.assertEqual(HARDENER.process_wheel(wheel, self.output), 7)
        installed = Path(self.case.name) / "installed"
        installed.mkdir()
        HARDENER._extract(self.output / wheel.name, installed)
        env = {name: value for name, value in os.environ.items()
               if name not in ("LD_LIBRARY_PATH", "LD_PRELOAD")}
        subprocess.run([str(installed / "ptoas/cli")], check=True, env=env, cwd="/", timeout=30)
        self.assertEqual(HARDENER.process_wheel(self.output / wheel.name), 7)

    def test_different_library_alias_is_rejected(self):
        installed = self._harden_and_extract()
        library = installed / "ptoas.libs/libleaf.so"
        (library.parent / "libleaf.so.1").write_bytes(library.read_bytes() + b"different")
        with self.assertRaisesRegex(ValueError, "ambiguous packaged library"):
            HARDENER.validate_tree(installed)

    def test_external_absolute_dependency_is_rejected(self):
        installed = self._harden_and_extract()
        binary = installed / "ptoas.libs/libmiddle.so"
        subprocess.run([HARDENER._tool("patchelf"), "--replace-needed", "$ORIGIN/libleaf.so",
                        "/outside/libleaf.so", str(binary)], check=True, timeout=30)
        with self.assertRaisesRegex(ValueError, "non-relocatable dependency"):
            HARDENER.validate_tree(installed)

    def test_dependency_escape_is_rejected(self):
        installed = self._harden_and_extract()
        binary = installed / "ptoas.libs/libmiddle.so"
        subprocess.run([HARDENER._tool("patchelf"), "--replace-needed", "$ORIGIN/libleaf.so",
                        "$ORIGIN/../../outside.so", str(binary)], check=True, timeout=30)
        with self.assertRaisesRegex(ValueError, "dependency escapes wheel"):
            HARDENER.validate_tree(installed)

    def _cmake_check(self, wheel):
        template = SCRIPT.parents[2] / "cmake/ValidateCannWheel.cmake.in"
        script = Path(self.case.name) / "validate.cmake"
        content = template.read_text(encoding="utf-8")
        for key, value in {"Python3_EXECUTABLE": sys.executable,
                           "PTOAS_ROOT_DIR": SCRIPT.parents[2], "PTOAS_WHEEL_FILE": wheel}.items():
            content = content.replace("@" + key + "@", str(value))
        script.write_text(content, encoding="utf-8")
        return subprocess.run([HARDENER._tool("cmake"), "-P", str(script)],
                              capture_output=True, text=True, timeout=120, check=False)

    def test_cmake_staging_gate_checks_the_actual_wheel(self):
        result = self._cmake_check(self.wheel)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("CANN wheel ELF hardening validation failed", result.stderr)
        self._harden_and_extract()
        result = self._cmake_check(self.output / self.wheel.name)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_tool_failure_leaves_source_and_output_unchanged(self):
        self.output.mkdir()
        destination = self.output / self.wheel.name
        destination.write_bytes(b"previous artifact")
        with mock.patch.object(HARDENER, "_run", side_effect=subprocess.CalledProcessError(1, "strip")):
            with self.assertRaises(subprocess.CalledProcessError):
                HARDENER.process_wheel(self.wheel, self.output)
        self.assertEqual(destination.read_bytes(), b"previous artifact")
        self.assertEqual(list(self.output.iterdir()), [destination])

    def test_dynamic_paths_are_detected_without_section_headers(self):
        binary = Path(self.case.name) / "no-section-headers.so"
        content = bytearray((self.libs / "libmiddle.so").read_bytes())
        offsets = (40, 48, 60, 64) if content[4] == 2 else (32, 36, 48, 52)
        start, end, count_start, count_end = offsets
        content[start:end] = bytes(end - start)
        content[count_start:count_end] = bytes(count_end - count_start)
        binary.write_bytes(content)
        info = HARDENER.inspect_elf(binary)
        self.assertEqual(info.sections, ())
        self.assertIn("DT_RPATH", info.search_paths)
        self.assertIn("libleaf.so", info.needed)

    def test_original_cannot_be_overwritten(self):
        with self.assertRaisesRegex(ValueError, "preserve the original"):
            HARDENER.process_wheel(self.wheel, self.wheel.parent)

    def test_archive_traversal_is_rejected(self):
        wheel = Path(self.case.name) / self.wheel.name
        with WheelFile(str(wheel), "w") as archive:
            archive.writestr("../escaped", "not allowed")
        with self.assertRaisesRegex(ValueError, "unsafe wheel member"):
            HARDENER.process_wheel(wheel, self.output)
        self.assertFalse((self.output / "escaped").exists())


if __name__ == "__main__":
    unittest.main()
