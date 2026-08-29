// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

# PTOAS MLIR Python Namespace and Native Isolation Design

## Status

This document records the agreed migration direction and its constraints. The
design is **proposed and not yet implemented**. Until the migration lands, the
current public imports remain `mlir.ir` and `mlir.dialects.pto`.

The intended public API is:

```python
from ptoas.mlir import ir
from ptoas.mlir.ir import Context, Module
from ptoas.mlir.dialects import pto
```

The migration must remove the PTOAS-owned top-level `mlir` package rather than
installing a compatibility alias that would recreate the namespace collision.

## Goals

1. Place the complete PTOAS-owned MLIR Python runtime under `ptoas.mlir`.
2. Allow PTOAS and another self-contained MLIR distribution, such as IREE,
   CIRCT, torch-mlir, or JAX, to coexist in one Python environment.
3. Give all PTOAS MLIR extensions one project-specific nanobind domain.
4. Keep one native MLIR runtime inside PTOAS so that `ptoas.mlir`, the PTO
   dialect facade, PTODSL, and `ptoas._core` share object identity.
5. Use MLIR's exported CMake source graph and standard wheel repair tools
   instead of copying package trees or implementing a custom loader.
6. Define and test the supported boundary between independently packaged MLIR
   runtimes.
7. Keep the repaired wheel as the official Python distribution, while retaining
   `ptoas-bin` as a CI and board-validation compatibility artifact. Generate
   that artifact from a deterministic CMake install-tree staging directory;
   do not make it a second Python package distribution.

## Non-Goals

- Do not make `ptoas.mlir.ir.Module` interchangeable with an IREE, CIRCT,
  torch-mlir, JAX, or system-MLIR `Module`.
- Do not install or populate a top-level `mlir` compatibility package.
- Do not scan repositories, neighboring build trees, `PYTHONPATH`, or library
  directories at Python import time.
- Do not preload LLVM/MLIR libraries with `RTLD_GLOBAL` or a custom bootstrap
  loader.
- Do not switch the LLVM SDK from shared to static libraries as part of the
  namespace migration.
- Do not modify LLVM/MLIR sources to implement the package layout.
- Do not make the compiler archive a second PyPI-style distribution with its
  own metadata, Python dependency resolver, or import bootstrap.
- Do not scan the source tree or arbitrary build directories while assembling
  the archive. Its contents must come from an explicit install prefix.
- Do not pretend that `auditwheel repair` or `delocate` is a generic tarball
  relocation tool. They remain the standard repair tools for wheels; any
  archive-native relocation must be a small, explicit packaging step.

## Industry and Upstream References

The design separates mechanisms with direct upstream/industry precedent from
the PTOAS-specific compatibility artifact required by the current board-test
workflow.

### Directly Referenced Mechanisms

- LLVM's standalone example uses `MLIR_PYTHON_PACKAGE_PREFIX`,
  `add_mlir_python_common_capi_library`, and `add_mlir_python_modules` to embed
  MLIR bindings in a downstream namespace.
- torch-mlir explicitly vendors MLIR in the `torch_mlir` namespace and defines
  `MLIR_PYTHON_PACKAGE_PREFIX=torch_mlir.`.
- IREE defines `MLIR_PYTHON_PACKAGE_PREFIX=iree.compiler.`, installs its MLIR
  tree below `iree/compiler`, and makes the Python bindings consume one
  project-owned shared compiler implementation. Its compiler tools are found
  relative to the installed Python package rather than by scanning build or
  source trees.
- CIRCT defines `MLIR_PYTHON_PACKAGE_PREFIX=circt.`, installs below `circt`, and
  uses a project-owned `CIRCTBindingsPythonCAPI` library.
- JAX packages its MLIR bindings below `jaxlib.mlir` with a private
  `_mlir_libs` directory.
- Linux native wheels conventionally use `auditwheel repair`; auditwheel copies
  external libraries, gives them content-derived names, rewrites `DT_NEEDED`,
  and installs package-relative RPATH entries.
- macOS native wheels conventionally use `delocate`, which copies dependent
  dylibs and rewrites install names and RPATH entries.
- CMake install components and CPack archive an explicit install graph into a
  deterministic prefix. They are the standard boundary for a native install
  tree, although they do not replace wheel-specific dependency repair.

These references directly support the `ptoas.mlir` namespace, project-owned
common CAPI, project-specific nanobind domain, and repaired-wheel architecture.

### Distribution and Archive Boundary

There is no universal MLIR or Python packaging standard for publishing a
compiler-only tarball alongside a Python wheel. Comparable projects either
publish wheels only or maintain a project-specific compiler/binary package.
The standard Python distribution remains the repaired wheel; `auditwheel` and
`delocate` are intentionally wheel-oriented tools.

PTOAS must currently retain a `ptoas-bin` archive because the PR board-test
monitor downloads the Linux artifact before running compiler cases. This is a
compatibility artifact, not a second public Python distribution. The archive
will be assembled from a dedicated CMake install-tree staging prefix, using
the same `ptoas` package layout and runtime contract as the wheel. CPack/TGZ or
an equivalent archive of that install prefix is the appropriate packaging
boundary; a script must not reconstruct a package by scanning the repository.

The archive and wheel have different native packaging constraints. Wheel
repair can be delegated to `auditwheel`/`delocate`. A relocatable install-tree
archive may still need a narrowly scoped platform step to ensure that bundled
non-system libraries use relative RPATHs or install names. That step must not
also copy Python sources, invent a second package layout, or run a second
general-purpose dependency-discovery pipeline hidden inside the launcher.

Relevant upstream sources:

- `llvm-project/mlir/examples/standalone/python/CMakeLists.txt`
- `llvm-project/mlir/cmake/modules/AddMLIRPython.cmake`
- `llvm/torch-mlir/python/CMakeLists.txt`
- `iree-org/iree/compiler/bindings/python/CMakeLists.txt`
- `iree-org/iree/compiler/bindings/python/iree/compiler/tools/binaries.py`
- `llvm/circt/lib/Bindings/Python/CMakeLists.txt`
- `cmake/Help/module/CPack.rst`
- `pypa/auditwheel`
- `matthew-brett/delocate`

## Target Package Layout

```text
ptoas/
├── __init__.py
├── _cli.py
├── _core.<abi>.so
├── _runtime/
└── mlir/
    ├── __init__.py
    ├── ir.py
    ├── passmanager.py
    ├── dialects/
    │   ├── arith.py
    │   ├── func.py
    │   ├── llvm.py
    │   ├── math.py
    │   ├── memref.py
    │   ├── pto.py
    │   └── scf.py
    └── _mlir_libs/
        ├── _mlir.<abi>.so
        ├── _site_initialize_0.<abi>.so
        └── libPTOASPythonCAPI.so
```

The corresponding native ownership graph is:

```text
ptoas.mlir.ir
      │
      ▼
ptoas/mlir/_mlir_libs/_mlir.so
      │
      ├──────────────────────┐
      ▼                      ▼
libPTOASPythonCAPI.so ◄── ptoas/_core.so
      │
      ▼
repaired, content-named LLVM/MLIR shared libraries
```

All PTOAS binding modules must resolve MLIR objects through the same
`PTOASPythonCAPI` and the same underlying LLVM/MLIR shared libraries.

## Python Namespace Configuration

Moving files under `ptoas/mlir` is not sufficient. Native MLIR bindings also
use the package prefix to locate Python classes and capsules. The binding
targets must therefore be compiled with:

```cmake
add_compile_definitions(
  "MLIR_PYTHON_PACKAGE_PREFIX=ptoas.mlir."
)
```

The definition must apply to the MLIR-generated extension targets and to
`PTOASPythonCore`. Since `_core` is built in a separate CMake directory, it may
need an explicit target-scoped compile definition in addition to the binding
directory definition.

The staged and installed package roots should become:

```cmake
set(PTOAS_MLIR_PYTHON_PACKAGE_DIR
    "${CMAKE_BINARY_DIR}/python/ptoas/mlir")

add_mlir_python_common_capi_library(PTOASPythonCAPI
  INSTALL_DESTINATION "ptoas/mlir/_mlir_libs"
  OUTPUT_DIRECTORY "${PTOAS_MLIR_PYTHON_PACKAGE_DIR}/_mlir_libs"
  ...
)

add_mlir_python_modules(PTOAS_Python
  ROOT_PREFIX "${PTOAS_MLIR_PYTHON_PACKAGE_DIR}"
  INSTALL_PREFIX "ptoas/mlir"
  ...
)
```

The generated package remains owned by the `PTOAS_Python` CMake install
component. It should not be copied from an LLVM build tree or added as a second
`tool.scikit-build.wheel.packages` source tree.

## Nanobind Domain Isolation

MLIR's nanobind domain determines which extension modules share native Python
type registrations. All modules inside one PTOAS MLIR runtime must use one
domain, while other MLIR distributions must use different domains.

Use:

```cmake
set(MLIR_BINDINGS_PYTHON_NB_DOMAIN "ptoas_mlir")
```

### Current LLVM Fork Constraint

The LLVM fork currently used by PTOAS has an older `AddMLIRPython.cmake` API.
Its `add_mlir_python_modules` function accepts `ROOT_PREFIX` and
`INSTALL_PREFIX`, but does not accept the newer per-call
`MLIR_BINDINGS_PYTHON_NB_DOMAIN` argument. It reads the CMake variable when it
creates nanobind extension targets.

Therefore the current implementation must set the variable before calling
`add_mlir_python_modules`; passing an extra argument to that function is not a
valid implementation for the current SDK. When the LLVM fork adopts the newer
upstream API, this can be converted to a per-package argument without changing
the public layout.

`ptoas._core` currently uses pybind11, so the nanobind domain does not directly
configure `_core`. Its interoperability with MLIR objects instead depends on
the package prefix, MLIR's adaptor/C API boundary, and the shared common CAPI.

## Common CAPI and Runtime Identity

`PTOASPythonCAPI` is already a project-owned common CAPI name and should remain
the single native binding runtime for PTOAS. The migration must preserve these
relationships:

- `_mlir` links `PTOASPythonCAPI`.
- `_site_initialize_0` links the same common CAPI.
- `ptoas._core` links the same common CAPI.
- `ptoas._core` initializes the matching Python runtime with:

  ```cpp
  py::module_::import("ptoas.mlir.ir");
  ```

- `ptoas.mlir.dialects.pto` imports `ptoas._core` for project-owned PTO types
  and enums.
- No PTOAS component imports or links a second system `MLIRPythonCAPI`.

`ptoas.__init__` should stay minimal and must not eagerly import `_core` or
`ptoas.mlir`, avoiding an import cycle between `_core`, `ir`, and the PTO
dialect facade.

## Dynamic Library Isolation

### Linux Repaired Wheels

PTOAS currently builds its LLVM SDK with `BUILD_SHARED_LIBS=ON`. This produces
many `libLLVM*.so` and `libMLIR*.so` dependencies. The release wheel must
continue through `auditwheel repair`.

Auditwheel provides the distribution isolation for those external libraries:

1. Copy each non-system dependency into the wheel-owned `ptoas.libs` tree.
2. Rename it with a content-derived hash.
3. Rewrite its ELF SONAME.
4. Rewrite consumers' `DT_NEEDED` entries.
5. Install package-relative RPATH entries.

PTOAS's repaired wheel already demonstrates this pattern with names such as:

```text
ptoas.libs/libMLIRArithDialect-00278e1e.so.21.1
ptoas.libs/libLLVMTransformUtils-2cda0d9b.so.21.1
```

The project should not manually rename or enumerate all LLVM/MLIR libraries.
The internal common CAPI name is already project-specific; auditwheel owns the
external dependency names.

### macOS Repaired Wheels

The macOS wheel must continue through `delocate`. Delocate owns copying dylibs
and rewriting install names and RPATH entries. PTOAS should not add a parallel
`install_name_tool` implementation.

### `_core` Package-Local RPATH

After nesting MLIR under `ptoas`, `_core` and its common CAPI are laid out as:

```text
ptoas/_core.so
ptoas/mlir/_mlir_libs/libPTOASPythonCAPI.so
```

The package-local install RPATH must therefore be:

```text
Linux: $ORIGIN;$ORIGIN/mlir/_mlir_libs
macOS: @loader_path;@loader_path/mlir/_mlir_libs
```

The old `$ORIGIN/../mlir/_mlir_libs` path belongs to the current sibling
packages and must not survive the migration.

For the common CAPI under `ptoas/mlir/_mlir_libs`, the
`RELATIVE_INSTALL_ROOT` value must be recomputed from the new location rather
than copied from the old configuration. From this location back to the Python
install prefix is normally three levels (`../../..`). The final value must be
verified from installed ELF/Mach-O metadata because auditwheel and delocate
will subsequently replace distribution-wheel dependency paths.

### Editable and Build-Tree Installs

Editable and direct build-tree installations do not pass through wheel repair.
They may use RPATH entries that refer to the selected developer LLVM build.
This is sufficient for a convenient PTOAS development environment, but it does
not guarantee same-process coexistence with an unrelated developer MLIR build.

The supported isolation contract applies most strongly to repaired release
wheels. Editable multi-MLIR coexistence is best effort unless both projects
were intentionally built against one compatible runtime.

### Compiler Compatibility Archive

The `ptoas-bin` archive is compiler-oriented and includes PTODSL as an internal
compiler runtime dependency. PTOAS defaults to the PTODSL TileLib backend, so
excluding `ptodsl/` would make an extracted archive unable to compile `.pto`
input containing unexpanded TileOps with the normal CLI defaults. The archive
must therefore contain both PTODSL and the installed nested MLIR package:

```text
ptodsl/
ptoas/_core.so
ptoas/mlir/
ptoas/mlir/_mlir_libs/_mlir.so
ptoas/mlir/_mlir_libs/libPTOASPythonCAPI.so
```

This does not make the archive a general PTODSL development distribution.
`ptodsl/` and `ptoas.mlir` are present so the compiler's default TileLib path
works without changing CLI semantics. The repaired wheel remains the supported
Python installation for authoring and importing PTODSL applications.

The repaired wheel remains the canonical Python distribution. Wheel and
archive packaging share the same CMake target graph but use different install
contracts. The archive is produced from an explicit compiler-runtime install
component so that its contents are owned by CMake rather than reconstructed
from source paths:

```text
configured CMake build
        │
        ├── wheel install components
        │       └── raw wheel
        │             └── auditwheel / delocate
        │                   └── repaired Python wheel
        │
        └── PTOAS_Python + compiler-runtime install components
                └── archive staging prefix
                      └── native relocation, if required
                            └── tar/CPack board artifact
```

The staging prefix must contain only the compiler archive contract:

```text
bin/ptoas
ptodsl/
ptoas/
ptoas/mlir/
ptoas/_runtime/share/ptoas/TileOps/
tilelang_dsl/
lib/                         # only if native dependencies are bundled
```

The archive exposes PTODSL only as part of the compiler runtime contract. It
does not include wheel metadata or claim to satisfy normal `pip` dependency
resolution. Board validation invokes the packaged compiler rather than using
the archive as an installable Python distribution.

Archive assembly is limited to install-tree and relocation operations:

- install `PTOAS_Python` and then `PTOAS_CompilerArchive` with an explicit,
  shared staging prefix;
- preserve the package-relative `ptoas/`, `ptoas/mlir/`, and `ptodsl/` layout;
- place any bundled non-system native dependencies below the archive prefix
  and use `$ORIGIN`/`@loader_path`-relative linkage;
- create the tarball from that prefix with CPack/TGZ or an equivalent simple
  archive command;
- smoke test the extracted archive with a clean environment.

The archive's `lib/` directory, when needed, is an install-tree convention and
is distinct from auditwheel's wheel-local `*.libs` sidecar. We must not copy a
wheel sidecar into an unrelated layout without also preserving the native
linkage that refers to it.

Native dependency discovery for the archive should use CMake's install-time
`file(GET_RUNTIME_DEPENDENCIES)`, which is available within PTOAS's existing
CMake 3.20 minimum, rather than handwritten recursive `ldd` or `otool`
parsers. Package-owned files such as `PTOASPythonCAPI`, `_mlir`, and site
initializers remain under `ptoas/mlir/_mlir_libs` and must be excluded from the
external dependency copy into `lib/`.

The archive relocation contract is:

```text
Linux ptoas/_core.so:
  $ORIGIN;$ORIGIN/mlir/_mlir_libs;$ORIGIN/../lib
Linux ptoas/mlir/_mlir_libs/*:
  $ORIGIN;$ORIGIN/../../../lib
Linux lib/*.so*:
  $ORIGIN

macOS ptoas/_core.so:
  @loader_path;@loader_path/mlir/_mlir_libs;@loader_path/../lib
macOS ptoas/mlir/_mlir_libs/*:
  @loader_path;@loader_path/../../../lib
macOS lib/*.dylib:
  @loader_path
```

Linux uses `patchelf` only to apply this staging-tree RPATH contract after
CMake resolves and copies dependencies. macOS uses `delocate-path` for the
equivalent directory relocation instead of maintaining a project-owned Mach-O
dependency parser. Neither platform searches for libraries at Python import
time.

The current compatibility archive is built from the CPython 3.11 workflow and
contains CPython-ABI-specific extension modules. It therefore requires CPython
3.11 at runtime. The existing `ptoas-bin-<platform>` artifact names remain
unchanged for the external board-test monitor; the build and extracted-archive
tests must validate the interpreter version explicitly. Moving to abi3 or
publishing per-Python archive names is outside this migration.

The single-runtime invariant is inherited from the shared CMake target graph:

- exactly one package-owned `PTOASPythonCAPI` is retained;
- `_core`, `_mlir`, and site initializers link the same project-owned common
  CAPI before the wheel and archive take their platform-specific packaging
  paths;
- LLVM/MLIR dependencies in the archive use relative install-tree RPATHs or
  install names and contain no build-tree absolute paths;
- any platform relocation helper is limited to the archive staging prefix and
  is separately validated from the wheel repair path.

The standalone archive is intended to launch a fresh compiler process. Unlike
the repaired wheel, it does not promise arbitrary same-process import
coexistence with another MLIR distribution. Its required invariant is narrower
but strict: one archive invocation loads one internally consistent PTOAS native
runtime built from the same CMake targets as the wheel and relocated only for
the archive's own install-tree layout.

## Why Not Switch Directly to Static LLVM

Changing only `BUILD_SHARED_LIBS=OFF` can make `_mlir.so` and `_core.so`
statically embed separate copies of MLIR. That would duplicate registries,
TypeIDs, command-line options, and native global state inside one process and
could break PTOAS's own cross-module object identity.

A future single-DSO design must instead follow an IREE-style ownership model:

```text
libPTOASCompiler.so
├── LLVM/MLIR implementation
├── PTO dialect and passes
├── compiler driver
└── exported C API

_mlir.so ──► libPTOASCompiler.so
_core.so ──► libPTOASCompiler.so
```

The follow-on implementation, including the shared development SDK and static
wheel SDK profiles, is specified in
`docs/designs/ptoas-compiler-dso-wheel-linking.md`.

That architectural change now provides the exported bridge and symbol-
visibility policy described there. This section is retained as the historical
namespace-migration baseline; the follow-on design supersedes its native DSO
topology.

## Public API Migration

Production imports, tests, samples, and documentation must move together:

```python
# Old
from mlir import ir
from mlir.ir import Module
from mlir.dialects import pto

# New
from ptoas.mlir import ir
from ptoas.mlir.ir import Module
from ptoas.mlir.dialects import pto
```

The affected surfaces include:

- PTO dialect Python facade and generated source graph;
- PTODSL implementation and public MLIR object documentation;
- TileLang DSL pybind backend;
- PTOAS Python tests;
- sample programs;
- README installation and Python API examples;
- wheel payload validation and import smoke tests;
- native `mlir.ir` import strings.

Because README and wheel tests currently promise top-level `mlir`, removing it
is a public API break and requires an explicit release note. A default alias is
not an acceptable compatibility mechanism because it defeats package
isolation.

## Supported Coexistence Boundary

| Scenario | Contract |
|---|---|
| Install PTOAS and another namespaced MLIR wheel in one environment | Supported target |
| Import both repaired wheels in one interpreter | Must be validated |
| Create independent contexts and modules in each runtime | Supported target |
| Pass a PTOAS `Module`, `Type`, or `Value` into another runtime | Unsupported |
| Exchange textual MLIR, bytecode, or files | Supported integration boundary |
| Mix editable builds backed by unrelated LLVM trees | Best effort, not guaranteed |

No public function should imply that independently packaged MLIR native objects
are interchangeable. Cross-project integration must serialize the IR or use a
subprocess unless the projects deliberately share one compatible runtime.

## Validation Plan

### Package Contract

- The wheel contains `ptoas/mlir/ir.py` and
  `ptoas/mlir/dialects/pto.py`.
- The wheel does not contain a top-level `mlir/` tree.
- `ptoas._core` and exactly one PTOAS `_mlir` core extension are present.
- Importing PTOAS does not create `sys.modules["mlir"]`.

### Functional Imports

```python
from ptoas.mlir import ir
from ptoas.mlir.dialects import pto
import ptodsl
from ptodsl import pto as dsl_pto
```

The test must create a context and module, register the PTO dialect, compile a
minimal PTODSL kernel, and invoke the installed `ptoas` console script.

### Runtime Identity

- PTO types created through `ptoas._core` work with
  `ptoas.mlir.ir.Context`.
- PTODSL returns `ptoas.mlir.ir.Module` objects.
- PTOAS binding modules all resolve the same `PTOASPythonCAPI`.
- Import order between `ptoas._core`, `ptoas.mlir.ir`, and
  `ptoas.mlir.dialects.pto` is deterministic and cycle-free.

### Linux Wheel Dependencies

- `auditwheel show` reports an accepted manylinux policy.
- Repaired LLVM/MLIR dependencies have content-derived names.
- `readelf -d` contains no absolute LLVM build path.
- No non-system `libLLVM*` or `libMLIR*` dependency remains outside the wheel.

### macOS Wheel Dependencies

- `delocate-listdeps` reports no external developer LLVM path.
- `otool -L` contains only system or wheel-owned dylibs.
- No absolute LLVM build or install path remains.

### Compiler Archive Dependencies

- The archive contains `ptoas/mlir` and `ptodsl` so the default PTODSL TileLib
  backend works without extra CLI options.
- The archive is produced by an explicit CMake compiler-runtime install
  component, not by scanning the source tree or an arbitrary install tree.
- Package-relative paths are preserved and the archive contains no source,
  build, or install-prefix absolute paths.
- Exactly one package-owned `PTOASPythonCAPI` exists.
- `_core`, `_mlir`, site initializers, and the common CAPI resolve through the
  archive's own relative native paths.
- Any native dependency relocation is performed only on the staging prefix and
  is not implemented by Python import-time path scanning.
- Runtime dependency discovery uses CMake
  `file(GET_RUNTIME_DEPENDENCIES)` rather than recursive shell parsing.
- Package-owned MLIR extensions and `PTOASPythonCAPI` remain under
  `ptoas/mlir/_mlir_libs`; only external native dependencies are copied to
  the archive `lib/` directory.
- The archive reports a clear error unless it is launched with CPython 3.11.
- The extracted archive runs `ptoas --version` and compiles a representative
  `.pto` input containing an unexpanded TileOp through the default PTODSL
  backend with `PYTHONPATH`, `LD_LIBRARY_PATH`, and `DYLD_LIBRARY_PATH` unset.
- PR workflows continue to upload the artifact name consumed by the external
  board-test monitor until that consumer is explicitly migrated.

### Coexistence

A packaging or release integration job should install one representative
namespaced MLIR distribution, preferably IREE, and test both import orders:

```python
import ptoas.mlir.ir
import iree.compiler.ir
```

and:

```python
import iree.compiler.ir
import ptoas.mlir.ir
```

Each runtime should create and verify its own context/module. The test must not
pass objects between runtimes. This heavier check belongs in wheel packaging or
release validation rather than every source-only PR job.

### Repository Regression

- Python binding and PTODSL tests;
- TileLang DSL backend tests;
- PTOAS lit tests;
- VMI lit and simulator regression where Python samples are involved;
- Linux and macOS wheel build, repair, clean-install, and CLI smoke tests.

## Implementation Sequence

1. Add `ptoas.mlir` package prefix and staging/install destinations.
2. Set the current LLVM fork's global nanobind domain to `ptoas_mlir`.
3. Keep and relocate the single `PTOASPythonCAPI`.
4. Update `_core` package import and package-local RPATH.
5. Add a dedicated compiler-runtime install component and explicit archive
   staging prefix; remove source-tree and arbitrary-prefix discovery from the
   collectors.
6. Include PTODSL in the compiler-runtime component so the default TileLib
   backend remains functional.
7. Replace recursive native dependency parsing with install-time CMake
   `file(GET_RUNTIME_DEPENDENCIES)`, then apply the documented Linux/macOS
   staging-tree relocation contract.
8. Update the PTO dialect facade and all production Python imports.
9. Update samples, tests, README, workflow artifacts, and wheel/archive
   contract validation without renaming the board-test artifact consumed by
   the external monitor.
10. Remove top-level `mlir` from the wheel without a default alias.
11. Run build-tree, editable, lit, simulator, repaired-wheel, and extracted
   standalone-archive regressions.
12. Validate dynamic dependencies and representative same-process wheel
   coexistence.
13. Consider a monolithic `libPTOASCompiler` only as a later, separately
    justified optimization for wheel size, startup time, or stronger symbol
    isolation.

## Decision Summary

- Public namespace: `ptoas.mlir`.
- Public PTO dialect: `ptoas.mlir.dialects.pto`.
- Native compiler module: `ptoas._core`.
- MLIR Python package prefix: `ptoas.mlir.`.
- Nanobind domain: `ptoas_mlir`.
- Common native runtime: one `PTOASPythonCAPI`.
- Linux distribution isolation: `auditwheel repair`.
- macOS distribution isolation: `delocate`.
- Official Python distribution: repaired wheel with the `ptoas` console script.
- Board-validation artifact: compiler-only tarball from the CMake
  compiler-runtime install component, preserving the nested `ptoas.mlir`
  package layout, including PTODSL for the default TileLib backend, and using
  relative native paths.
- Compiler archive Python ABI: CPython 3.11 until a separate abi3 or
  per-Python archive design is adopted.
- Archive dependency discovery: CMake `file(GET_RUNTIME_DEPENDENCIES)`;
  platform tools perform only staging-tree relocation.
- Board-test compatibility: retain the current `ptoas-bin` artifact contract
  until the external monitor is migrated deliberately.
- Cross-runtime native objects: unsupported.
- Top-level `mlir` compatibility alias: not installed.
- Static or monolithic LLVM redesign: deferred to a separate proposal.
