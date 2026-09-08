#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""``@pto.jit(native_options=...)`` normalization and native-build wiring.

Everything here runs without a toolchain: the option layer is pure Python, and
the build layer is exercised by capturing the commands it would run.
"""

import importlib.util
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from ptodsl._native_options import (
    EMPTY_NATIVE_OPTIONS,
    NativeBuildOptions,
    normalize_native_options,
)
from ptodsl._runtime import native_build

# KernelModuleSpec and @pto.jit pull MLIR bindings. The rest of this file
# only inspects option records and captured compiler command lines.
_HAS_PTOAS_MLIR = importlib.util.find_spec("ptoas.mlir") is not None


class NormalizeNativeOptionsTest(unittest.TestCase):
    def test_none_and_empty_mapping_are_empty(self):
        self.assertIs(normalize_native_options(None), EMPTY_NATIVE_OPTIONS)
        self.assertTrue(normalize_native_options({}).is_empty())

    def test_relative_paths_resolve_against_the_declaring_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            declaring = root / "pkg" / "kernel.py"
            declaring.parent.mkdir(parents=True)
            declaring.touch()

            options = normalize_native_options(
                {"host_sources": ["shim.cpp", "../other/helper.cpp"], "include_dirs": "inc"},
                declaring_file=str(declaring),
            )

            self.assertEqual(
                options.host_sources,
                (root / "pkg" / "shim.cpp", root / "other" / "helper.cpp"),
            )
            self.assertEqual(options.include_dirs, (root / "pkg" / "inc",))

    def test_absolute_paths_are_kept(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            absolute = Path(temp_dir).resolve() / "shim.cpp"
            options = normalize_native_options(
                {"host_sources": [absolute]},
                declaring_file="/somewhere/else/kernel.py",
            )
            self.assertEqual(options.host_sources, (absolute,))

    def test_a_single_path_does_not_have_to_be_wrapped_in_a_list(self):
        options = normalize_native_options({"host_sources": "/tmp/shim.cpp"})
        self.assertEqual(options.host_sources, (Path("/tmp/shim.cpp").resolve(),))

    def test_duplicate_entries_collapse(self):
        options = normalize_native_options(
            {"host_sources": ["/tmp/a.cpp", "/tmp/a.cpp"], "link_libraries": ["dl", "dl"]}
        )
        self.assertEqual(len(options.host_sources), 1)
        self.assertEqual(options.link_libraries, ("dl",))

    def test_options_are_hashable_so_a_kernel_spec_stays_frozen(self):
        options = normalize_native_options(
            {"host_sources": ["/tmp/a.cpp"], "link_libraries": ["dl"]}
        )
        self.assertEqual(hash(options), hash(options))
        self.assertEqual(options, normalize_native_options(
            {"host_sources": ["/tmp/a.cpp"], "link_libraries": ["dl"]}
        ))

    def test_unknown_keys_are_rejected_and_named(self):
        with self.assertRaises(ValueError) as caught:
            normalize_native_options({"host_source": ["/tmp/a.cpp"]})
        self.assertIn("host_source", str(caught.exception))
        self.assertIn("host_sources", str(caught.exception))

    def test_non_mapping_is_rejected(self):
        with self.assertRaises(TypeError):
            normalize_native_options(["/tmp/a.cpp"])

    def test_library_names_reject_paths_and_flags(self):
        for bad in ("/usr/lib/libdl.so", "-ldl", "dl; rm -rf /", "my lib"):
            with self.assertRaises(ValueError, msg=f"{bad!r} should be rejected"):
                normalize_native_options({"link_libraries": [bad]})

    def test_library_dirs_do_not_have_to_accompany_host_sources(self):
        # Linking against a prebuilt library needs no source of our own.
        options = normalize_native_options(
            {"link_libraries": ["dl"], "library_dirs": ["/opt/lib"]}
        )
        self.assertEqual(options.link_libraries, ("dl",))
        self.assertEqual(options.library_dirs, (Path("/opt/lib"),))

    def test_include_dirs_without_host_sources_is_rejected(self):
        # They would apply to nothing, so a quiet no-op is worse than an error.
        with self.assertRaises(ValueError) as caught:
            normalize_native_options({"include_dirs": ["/opt/include"]}, function_name="k")
        self.assertIn("host_sources", str(caught.exception))

    def test_empty_path_is_rejected(self):
        with self.assertRaises(ValueError):
            normalize_native_options({"host_sources": [""]})

    def test_wrong_entry_type_is_rejected(self):
        with self.assertRaises(TypeError):
            normalize_native_options({"host_sources": [123]})


def _module_spec(native_options=EMPTY_NATIVE_OPTIONS):
    return types.SimpleNamespace(
        function_name="k",
        target_arch="a3",
        kernel_kind="vector",
        mode="explicit",
        backend="vpto",
        insert_sync=None,
        jit_source=None,
        native_options=native_options,
    )


class HostSourceCacheIdentityTest(unittest.TestCase):
    """Editing a host source has to invalidate the cached library."""

    def _config_text(self, options):
        return native_build._compile_config_text(
            module_spec=_module_spec(options),
            effective_insert_sync=False,
            effective_pto_level="level3",
            ptoas_overrides={"backend": "vpto"},
        )

    def test_no_options_leaves_the_config_text_unchanged(self):
        text = self._config_text(EMPTY_NATIVE_OPTIONS)
        self.assertNotIn("host_source=", text)
        self.assertIn("target_arch=a3", text)

    def test_editing_a_host_source_changes_the_config_text(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "shim.cpp"
            source.write_text("int a() { return 1; }\n", encoding="utf-8")
            options = NativeBuildOptions(host_sources=(source,))

            before = self._config_text(options)
            source.write_text("int a() { return 2; }\n", encoding="utf-8")
            after = self._config_text(options)

            self.assertIn(str(source), before)
            self.assertNotEqual(before, after)

    def test_an_unreadable_host_source_is_recorded_rather_than_raising(self):
        options = NativeBuildOptions(host_sources=(Path("/nonexistent/shim.cpp"),))
        self.assertIn("unreadable", self._config_text(options))

    def test_include_dirs_take_part_in_the_identity(self):
        base = NativeBuildOptions(host_sources=(Path("/tmp/a.cpp"),))
        with_inc = NativeBuildOptions(
            host_sources=(Path("/tmp/a.cpp"),), include_dirs=(Path("/opt/inc"),)
        )
        self.assertNotEqual(self._config_text(base), self._config_text(with_inc))

    def test_a_module_spec_without_the_option_still_builds(self):
        legacy = types.SimpleNamespace(
            function_name="k", target_arch="a3", kernel_kind="vector", mode="explicit",
            backend="vpto", insert_sync=None, jit_source=None,
        )
        self.assertIs(native_build._native_options_of(legacy), EMPTY_NATIVE_OPTIONS)


@unittest.skipUnless(_HAS_PTOAS_MLIR, "needs ptoas.mlir bindings")
class KernelModuleSpecTest(unittest.TestCase):
    def test_the_spec_carries_the_options_and_stays_frozen(self):
        from ptodsl._tracing import KernelModuleSpec

        options = NativeBuildOptions(host_sources=(Path("/tmp/a.cpp"),))
        spec = KernelModuleSpec(
            function_name="k", target_arch="a3", kernel_kind="vector", native_options=options
        )
        self.assertIs(spec.native_options, options)
        self.assertIs(native_build._native_options_of(spec), options)
        with self.assertRaises(Exception):
            spec.native_options = EMPTY_NATIVE_OPTIONS

    def test_a_spec_without_the_option_defaults_to_none_of_it(self):
        from ptodsl._tracing import KernelModuleSpec

        spec = KernelModuleSpec(function_name="k", target_arch="a3", kernel_kind="vector")
        self.assertTrue(spec.native_options.is_empty())


@unittest.skipUnless(_HAS_PTOAS_MLIR, "needs ptoas.mlir bindings")
class JitDecoratorSurfaceTest(unittest.TestCase):
    """The decorator kwarg has to reach the spec, and reject bad input early."""

    SOURCE = (
        'module attributes {pto.target_arch = "a3"} {\n'
        "  func.func @native_option_probe(%arg0: !pto.ptr<f32, gm>)"
        " attributes {pto.kernel} {\n"
        "    return\n"
        "  }\n"
        "}\n"
    )

    def _kernel(self, native_options):
        from ptodsl import pto

        @pto.jit(
            name="native_option_probe",
            target="a3",
            backend="vpto",
            mode="explicit",
            source=self.SOURCE,
            native_options=native_options,
        )
        def native_option_probe(buf: pto.ptr(pto.f32, "gm")):
            pass

        return native_option_probe

    def test_options_reach_the_cache_signature(self):
        kernel = self._kernel({"host_sources": ["shim.cpp"], "link_libraries": ["dl"]})
        options = kernel.__ptodsl_cache_signature__()[9]
        self.assertEqual(options.link_libraries, ("dl",))
        self.assertEqual(len(options.host_sources), 1)
        # Resolved against this test file, which is what the option promises.
        self.assertEqual(options.host_sources[0], Path(__file__).resolve().parent / "shim.cpp")

    def test_omitting_the_option_changes_nothing(self):
        kernel = self._kernel(None)
        self.assertTrue(kernel.__ptodsl_cache_signature__()[9].is_empty())

    def test_a_bad_option_is_rejected_at_decoration_time(self):
        with self.assertRaises(ValueError):
            self._kernel({"host_sourcez": ["shim.cpp"]})


class ExtraLinkFlagsTest(unittest.TestCase):
    def test_no_options_contributes_nothing(self):
        self.assertEqual(native_build._extra_link_flags(EMPTY_NATIVE_OPTIONS), [])

    def test_library_dirs_become_search_paths_and_rpaths(self):
        options = NativeBuildOptions(
            link_libraries=("dl", "m"), library_dirs=(Path("/opt/lib"),)
        )
        self.assertEqual(
            native_build._extra_link_flags(options),
            ["-L/opt/lib", "-Wl,-rpath,/opt/lib", "-ldl", "-lm"],
        )

    def test_search_paths_precede_the_libraries_that_need_them(self):
        flags = native_build._extra_link_flags(
            NativeBuildOptions(link_libraries=("custom",), library_dirs=(Path("/opt/lib"),))
        )
        self.assertLess(flags.index("-L/opt/lib"), flags.index("-lcustom"))

    def test_host_sources_pull_libstdcxx(self):
        flags = native_build._extra_link_flags(
            NativeBuildOptions(host_sources=(Path("/tmp/shim.cpp"),), link_libraries=("dl",))
        )
        self.assertEqual(flags, ["-ldl", "-lstdc++"])

    def test_explicit_libstdcxx_is_not_duplicated(self):
        flags = native_build._extra_link_flags(
            NativeBuildOptions(
                host_sources=(Path("/tmp/shim.cpp"),), link_libraries=("dl", "stdc++")
            )
        )
        self.assertEqual(flags, ["-ldl", "-lstdc++"])


class HostSourceCompileTest(unittest.TestCase):
    def test_no_host_sources_runs_no_compiler(self):
        with mock.patch.object(native_build, "_run") as run:
            objects = native_build._compile_host_sources(
                EMPTY_NATIVE_OPTIONS, Path("/tmp/cache")
            )
        self.assertEqual(objects, [])
        run.assert_not_called()

    def test_a_missing_host_source_is_named_before_the_toolchain_is_resolved(self):
        # Deliberately without stubbing the toolchain: a mistyped path must report
        # itself, not whichever environment variable the compiler lookup wants.
        options = NativeBuildOptions(host_sources=(Path("/nonexistent/shim.cpp"),))
        with mock.patch.object(native_build, "_run") as run:
            with self.assertRaises(FileNotFoundError) as caught:
                native_build._compile_host_sources(options, Path("/tmp/cache"))
        self.assertIn("/nonexistent/shim.cpp", str(caught.exception))
        run.assert_not_called()

    def test_each_source_is_compiled_as_host_c_plus_plus_with_its_include_dirs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "cache"
            source = Path(temp_dir) / "shim.cpp"
            source.write_text("int a() { return 1; }\n", encoding="utf-8")
            options = NativeBuildOptions(
                host_sources=(source,), include_dirs=(Path("/opt/inc"),)
            )

            with mock.patch.object(native_build, "resolve_bisheng", return_value="bisheng"):
                with mock.patch.object(
                    native_build, "common_include_flags", return_value=["-I/cann/include"]
                ):
                    with mock.patch.object(native_build, "_run") as run:
                        objects = native_build._compile_host_sources(options, cache_dir)

            self.assertEqual(run.call_count, 1)
            cmd = run.call_args[0][0]
            self.assertEqual(cmd[0], "bisheng")
            # Host code, not device code: no -xcce and no aicore arch.
            self.assertIn("-xc++", cmd)
            self.assertNotIn("-xcce", cmd)
            self.assertFalse([arg for arg in cmd if arg.startswith("--cce-aicore-arch")])
            self.assertIn("-I/opt/inc", cmd)
            self.assertIn("-I/cann/include", cmd)
            self.assertIn("-fPIC", cmd)
            self.assertEqual(cmd[-4:], ["-c", str(source), "-o", str(objects[0])])

    def test_sources_sharing_a_stem_get_distinct_objects(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = root / "a" / "shim.cpp"
            second = root / "b" / "shim.cpp"
            for path in (first, second):
                path.parent.mkdir(parents=True)
                path.write_text("int a() { return 1; }\n", encoding="utf-8")
            options = NativeBuildOptions(host_sources=(first, second))

            with mock.patch.object(native_build, "resolve_bisheng", return_value="bisheng"):
                with mock.patch.object(native_build, "common_include_flags", return_value=[]):
                    with mock.patch.object(native_build, "_run"):
                        objects = native_build._compile_host_sources(options, root / "cache")

            self.assertEqual(len(set(objects)), 2)


class LinkLineTest(unittest.TestCase):
    def _link(self, **kwargs):
        with mock.patch.object(native_build, "resolve_bisheng", return_value="bisheng"):
            with mock.patch.object(
                native_build, "runtime_library_flags", return_value=["-lruntime"]
            ):
                with mock.patch.object(native_build, "_run") as run:
                    native_build._link_shared_library(
                        Path("/c/launch.o"),
                        Path("/c/kernel.o"),
                        Path("/c/libk.so"),
                        kernel_kind="vector",
                        **kwargs,
                    )
        return run.call_args[0][0]

    def test_without_options_the_link_line_is_unchanged(self):
        cmd = self._link()
        self.assertEqual(cmd[-3:], ["/c/launch.o", "/c/kernel.o", "-lruntime"])

    def test_host_objects_are_linked_and_extra_flags_come_last(self):
        cmd = self._link(
            host_objects=[Path("/c/host_0_shim.o")],
            extra_link_flags=["-L/opt/lib", "-ldl"],
        )
        self.assertIn("/c/host_0_shim.o", cmd)
        # The host object precedes the libraries that resolve its references.
        self.assertLess(cmd.index("/c/host_0_shim.o"), cmd.index("-ldl"))
        self.assertLess(cmd.index("-lruntime"), cmd.index("-ldl"))
        self.assertEqual(cmd[-2:], ["-L/opt/lib", "-ldl"])
        # Undefined symbols stay an error, so a missing entry fails the build.
        self.assertIn("-Wl,--no-undefined", cmd)


if __name__ == "__main__":
    unittest.main()
