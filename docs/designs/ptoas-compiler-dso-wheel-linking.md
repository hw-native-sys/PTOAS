// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

# PTOAS Compiler DSO and Wheel Linking Design

## Status

This design is **implemented**. Development and normal regression builds keep
the shared LLVM/MLIR SDK profile; Linux and macOS wheel workflows select the
static PIC SDK profile described below.

The central rule is:

> PTOAS has one CMake target graph and one package-owned compiler DSO. Local
> development and normal regression CI supply that graph with a shared
> LLVM/MLIR SDK; wheel workflows supply it with a static PIC LLVM/MLIR SDK.

The SDK profile changes the implementation behind the same imported CMake
target names. It must not introduce a second, wheel-only list of LLVM or MLIR
libraries.

## Motivation

Before this design, the Linux and macOS wheel workflows built LLVM with
`BUILD_SHARED_LIBS=ON`. The Python extensions and `PTOASPythonCAPI` therefore
referred to a large graph of small LLVM/MLIR component libraries. Wheel repair
had to discover, inspect, copy, rewrite, and often sign that entire graph.

This is especially expensive on macOS. Recent successful runs observed roughly
42 minutes for arm64 and 72 minutes for x86_64 in `delocate-wheel`. In the same
runs, an approximately 6.3 MB raw wheel became approximately 19 MB after
repair. These are observations rather than committed performance thresholds,
but they identify the shape of the problem.

The dependency closure also includes libraries that do not describe the
intended PTOAS distribution boundary, for example:

```text
libMLIRMlirOptMain.dylib
libMLIRAffineTransformOps.dylib
libMLIRAMDGPUTransforms.dylib
...
```

The problem is not the existence of a `.dylib` or `.so`. PTOAS needs a shared
native boundary because `_core`, MLIR's generated `_mlir` modules, and
`_site_initialize_0` must use one MLIR runtime. The problem is exposing the
fine-grained LLVM/MLIR component graph as hundreds of runtime dependencies.

The implemented distribution boundary is one project-owned
`libPTOASCompiler.{so,dylib}`. In wheel builds, the linker's ordinary static
archive selection pulls the needed LLVM/MLIR implementation into that DSO. In
development builds, the same DSO continues to depend on shared LLVM/MLIR
components, retaining fast incremental links and the existing debugging model.

## Goals

1. Install exactly one project-owned compiler DSO named `PTOASCompiler`.
2. Make all PTOAS Python native modules use the MLIR runtime owned by that DSO.
3. Use static PIC LLVM/MLIR only for Linux and macOS wheel builds.
4. Keep local development, normal CI, simulator CI, and the development Docker
   image on shared LLVM/MLIR.
5. Keep one semantic CMake dependency graph in PTOAS. Select static versus
   shared implementations by selecting the LLVM SDK profile.
6. Let the static linker select archive members from CMake targets. Do not
   enumerate installed library paths or copy dependencies by hand.
7. Continue to run `auditwheel` and `delocate` for platform policy checks and
   any remaining non-system dependencies.
8. Preserve the standalone compiler archive, built from the same PTOAS targets
   as the wheel.

## Non-Goals

- Requiring `-dead_strip`, `--gc-sections`, LTO, or a custom linker in the
  first implementation.
- Applying `--whole-archive` or `-all_load` to all LLVM/MLIR libraries.
- Treating upstream's all-inclusive `libMLIR` as PTOAS's final distribution
  boundary. It does not include the PTOAS compiler implementation and does not
  by itself enforce a single runtime across PTOAS extensions.
- Creating a wheel-only list of LLVM/MLIR components in workflow YAML or
  `CMAKE_ARGS`.
- Providing a stable external ABI for `PTOASCompiler` in the first version.
- Changing development and ordinary regression builds to static LLVM/MLIR.
- Removing wheel repair or replacing it with project-owned dependency-copying
  logic.

## Previous Native Topology

The previous targets were split between
`lib/Bindings/Python/CMakeLists.txt` and `tools/ptoas/CMakeLists.txt`:

```text
PTOASPythonCAPI (shared)
├── embedded MLIR Python CAPI objects
└── shared LLVM/MLIR component dependencies

generated _mlir modules ───────► PTOASPythonCAPI
_site_initialize_0 ────────────► PTOASPythonCAPI

_core
├── NativeModule.cpp
├── PTOModule.cpp
├── ptoas_runtime_objects
│   ├── ptoas.cpp
│   ├── driver.cpp
│   ├── VPTOHostStubEmission.cpp
│   └── ObjectEmission.cpp
├── PTOCAPI
└── PTOASPythonCAPI
```

`ptoas_runtime_objects` carries the compiler's LLVM/MLIR link dependencies,
while MLIR's Python CAPI objects live in `PTOASPythonCAPI`. Consequently,
changing LLVM to static without changing this ownership boundary could embed
independent MLIR implementations in multiple Python modules.

## Implemented Native Topology

Evolve the existing MLIR common CAPI DSO into the compiler ownership boundary:

```text
PTOASCompiler (shared)
├── embedded MLIR Python CAPI objects
├── embedded PTOCAPI objects
├── embedded PTOIR and PTOTransforms objects
├── embedded ptobc_lib objects
├── PTOASCompilerImplementation objects
│   ├── current ptoas_runtime_objects sources
│   └── PTOModule.cpp
└── required LLVM/MLIR CMake targets

generated _mlir modules ───────► PTOASCompiler
_site_initialize_0 ────────────► PTOASCompiler
_core (NativeModule.cpp only) ─► PTOASCompiler
```

The installed location remains the private MLIR native directory because all
generated MLIR modules already know how to resolve their common CAPI there:

```text
ptoas/mlir/_mlir_libs/libPTOASCompiler.so       # Linux
ptoas/mlir/_mlir_libs/libPTOASCompiler.dylib    # macOS
```

`libPTOASPythonCAPI` is no longer installed. `PTOASCompiler` replaces it as
both the AddMLIRPython common CAPI and the PTOAS compiler implementation DSO.
The package-private target deliberately has no CMake `VERSION` or `SOVERSION`:
otherwise a wheel archive dereferences both the versioned file and its symlink
and stores the same DSO twice. The DSO still has the unversioned
`libPTOASCompiler` SONAME/install name used by the extensions in that wheel.

### Why `PTOModule.cpp` Belongs in the DSO

If `_core` retained `PTOModule.cpp`, the DSO would need to export a broad set of
MLIR and PTO C++ symbols used by the binding implementation. Moving it into
`PTOASCompiler` lets `_core` remain a thin Python entry point. The cross-target
surface can initially be limited to bridge functions equivalent to:

```cpp
namespace mlir::pto {
PTOAS_COMPILER_EXPORT int runPTOAS(int argc, char **argv);
}

namespace mlir::pto::python {
PTOAS_COMPILER_EXPORT void
populatePTODialectBindings(pybind11::module_ &module);
}
```

These declarations need a small visibility header and platform export macro.
For version 1 they are an internal, same-build C++ ABI between artifacts in one
wheel. A stable public C API can be designed separately if external consumers
need one.

This bridge intentionally depends on pybind11 and therefore makes
`PTOASCompiler` a package-internal implementation library, not a reusable
public SDK library. `PTOModule.cpp` must be compiled with RTTI and exceptions
enabled, matching the current pybind11 extension translation unit. The target
must also acquire pybind11's headers and Python link contract without adding a
second Python module initializer. A future public compiler ABI should be a
separate C API layered over this implementation.

Python extensions conventionally leave CPython API symbols to be resolved by
the interpreter. Moving pybind11 code into this private DSO therefore requires
one deliberate exception to `add_mlir_aggregate`'s Linux `-z defs` default and
pybind11's normal macOS dynamic-lookup link option. Do not link a release wheel
to an absolute `libpython` or Python framework merely to satisfy that check.
Instead, audit the final undefined-symbol set and allow only the expected
CPython API in addition to platform runtime symbols.

## One PTOAS Graph, Two LLVM SDK Profiles

PTOAS always links logical CMake targets such as `MLIRIR`, `MLIRParser`,
`MLIRMlirOptMain`, and `LLVMOption`. It never names their `.a`, `.so`, or
`.dylib` files. The LLVM build determines what implementation those targets
represent.

### Shared Development Profile

This remains the default for:

- local editable and developer builds;
- `.github/workflows/ci.yml`;
- `.github/workflows/ci_sim.yml`;
- `docker/Dockerfile`;
- `docker/Dockerfile.dev`;
- other normal regression configurations.

The relevant setting when building the LLVM SDK remains:

```cmake
-DBUILD_SHARED_LIBS=ON
```

PTOAS itself still creates `PTOASCompiler` explicitly as `SHARED`; the SDK's
`BUILD_SHARED_LIBS` value does not change that. Its private runtime dependencies
are the shared LLVM/MLIR component libraries from the developer SDK.
PTOAS-owned compiler and C API objects are embedded in the DSO in both
profiles. This preserves the current LLVM rebuild and incremental link
behavior without exposing separate PTOAS implementation DSOs.

### Static Wheel Profile

Only `.github/workflows/build_wheel.yml` and
`.github/workflows/build_wheel_mac.yml` build and consume this LLVM SDK. The
following are arguments to the LLVM source-tree configure, not a new PTOAS
mode switch:

```cmake
-DBUILD_SHARED_LIBS=OFF
-DLLVM_BUILD_LLVM_DYLIB=OFF
-DLLVM_LINK_LLVM_DYLIB=OFF
-DMLIR_LINK_MLIR_DYLIB=OFF
-DLLVM_ENABLE_PIC=ON
-DMLIR_INSTALL_AGGREGATE_OBJECTS=ON
```

The pinned LLVM 21 tree defines these options; `LLVM_ENABLE_PIC` and
`MLIR_INSTALL_AGGREGATE_OBJECTS` already default to `ON`, but wheel workflows
set them explicitly because they are part of the distribution contract. The
required contract is: LLVM/MLIR component targets are static and
position-independent; PTOAS does not silently redirect them to an upstream
LLVM or MLIR dylib. `MLIR_INSTALL_AGGREGATE_OBJECTS` is required only when
PTOAS consumes an installed LLVM SDK. The current workflows consume its build
tree, but keeping the option explicit makes an installed SDK equivalent.

When `PTOASCompiler` links those same target names, ordinary archive member
selection embeds the referenced implementation in the project DSO. The Python
extensions then have one direct PTOAS runtime dependency rather than direct
dependencies on the external LLVM build tree.

Static and shared LLVM SDKs must use separate build directories and cache keys.
Before enabling the profile, both wheel workflows must bump
`LLVM_CACHE_FLAVOR`, for example to `llvm21-vpto-wheel-static-v1`. Reusing a
cache produced with `BUILD_SHARED_LIBS=ON` is invalid even when the source SHA
is identical.

The resulting matrix is:

| Property | Development and normal CI | Wheel workflows |
| --- | --- | --- |
| PTOAS target graph | One common graph | The same graph |
| `PTOASCompiler` | Shared DSO | Shared DSO |
| PTOAS compiler objects | Embedded | Embedded |
| LLVM/MLIR components | Shared dependencies | Static PIC members in the DSO |
| Wheel repair | Not applicable | Policy check and residual dependencies |

## Concrete CMake Changes

### 1. Extend the AddMLIRPython Common CAPI Target

In `lib/Bindings/Python/CMakeLists.txt`, rename the existing target while
retaining MLIR's standard source collection and module wiring:

```cmake
add_mlir_python_common_capi_library(PTOASCompiler
  INSTALL_COMPONENT PTOAS_Python
  INSTALL_DESTINATION "ptoas/mlir/_mlir_libs"
  OUTPUT_DIRECTORY "${PTOAS_MLIR_PYTHON_PACKAGE_DIR}/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../"
  DECLARED_HEADERS
    MLIRPythonCAPI.HeaderSources
  DECLARED_SOURCES
    ${PTOAS_MLIR_PYTHON_SOURCE_SETS}
    PTOASPythonSources
  EMBED_LIBS
    MLIRCAPITransforms
    PTOCAPI
    PTOIR
    PTOTransforms
    ptobc_lib
    PTOASCompilerImplementation
)

set_property(TARGET PTOASCompiler PROPERTY LINKER_LANGUAGE CXX)

add_mlir_python_modules(PTOAS_Python
  # Existing arguments remain unchanged.
  COMMON_CAPI_LINK_LIBS
    PTOASCompiler
)
```

Using `add_mlir_python_common_capi_library` remains important: it is the
upstream mechanism that creates an `add_mlir_aggregate(... SHARED ...)`,
collects the selected Python CAPI objects, and wires the generated `_mlir` and
site-initializer modules to one common runtime. `PTOCAPI` is already declared
with `add_mlir_public_c_api_library`, so it is aggregation-capable and belongs
in `EMBED_LIBS`; adding it only through `target_link_libraries` would leave its
C API in a separate shared library under a downstream shared configuration.

Add `ENABLE_AGGREGATION` to the existing `PTOIR` and `PTOTransforms`
declarations before naming them in `EMBED_LIBS`. This embeds those
project-owned objects in both profiles while their LLVM/MLIR dependencies
continue to follow the selected SDK. This is a short list of PTOAS ownership
roots, not a wheel-specific enumeration of upstream libraries.

Likewise, convert `ptobc_lib` from a plain static library to a non-installed
`add_mlir_library(... OBJECT ENABLE_AGGREGATION ...)` target and embed it. Its
source list remains unchanged. This prevents its public `PTOIR` dependency
from reintroducing a separate PTOIR DSO after the outer aggregate has embedded
PTOIR.

`PTOASCompilerImplementation` is declared later under `tools/ptoas`. MLIR's
`add_mlir_aggregate` explicitly supports this in-tree target ordering through
deferred generator expressions, so `lib` can continue to precede `tools` in
the top-level build.

### 2. Make the Compiler Implementation Aggregation-Capable

In `tools/ptoas/CMakeLists.txt`, evolve the current
`ptoas_runtime_objects` target into a non-installed aggregation root. The
source list remains authoritative. The object target receives compile settings,
while its link dependencies are passed when the aggregation root is created:

```cmake
add_mlir_library(PTOASCompilerImplementation
  OBJECT
  ENABLE_AGGREGATION
  DISABLE_INSTALL
  ${PTOAS_RUNTIME_SOURCES}
  "${CMAKE_SOURCE_DIR}/lib/Bindings/Python/PTOModule.cpp"
  LINK_LIBS PUBLIC
  ${PTOAS_RUNTIME_LINK_LIBS}
)

ptoas_configure_runtime_compile_target(
  obj.PTOASCompilerImplementation)
```

`add_mlir_library(... OBJECT ...)` creates
`obj.PTOASCompilerImplementation`, which owns compilation, plus a normal
library target that carries its link interface. Applying the current compile
definitions only to the latter would not configure the object files. The
split is:

- `ptoas_configure_runtime_compile_target` owns release-version/default-path
  definitions and generated-header dependencies;
- `PTOAS_RUNTIME_LINK_LIBS` owns the existing single dependency list and is
  supplied to `add_mlir_library`;
- the object target also receives
  `MLIR_PYTHON_PACKAGE_PREFIX=ptoas.mlir.` because `PTOModule.cpp` moved out of
  the original extension target's directory.

`add_mlir_library(... ENABLE_AGGREGATION ...)` publishes the implementation's
object files and filtered transitive link dependencies to `PTOASCompiler`.
Dependencies already embedded by the outer aggregate, including `PTOIR` and
`PTOTransforms`, are removed from its runtime link interface. This is why a
plain object library plus a raw `target_link_libraries(PTOASCompiler ...)`
should not be the final design: those raw usage requirements could reintroduce
an embedded project library as a separate DSO in the shared profile.

`PTOAS_RUNTIME_LINK_LIBS` remains the single authoritative LLVM/MLIR component
list. Pass it through `add_mlir_library(... LINK_LIBS ...)` when creating the
aggregation root: adding dependencies later with `target_link_libraries` does
not retroactively place them in `MLIR_AGGREGATE_DEPS`. Do not copy that list to
`lib/Bindings/Python/CMakeLists.txt` or either wheel workflow.

Any PTOAS static or object target that is linked into `PTOASCompiler` must have
`POSITION_INDEPENDENT_CODE ON` when the corresponding LLVM/MLIR helper does not
already guarantee it.

The implementation still links the logical `ptobc_lib` target. Aggregate
filtering, rather than a second dependency list, ensures that its objects and
transitive dependencies enter the final DSO exactly once.

Apply per-source compile options to `PTOModule.cpp` after
the MLIR helper creates the object target: it must retain
`-frtti -fexceptions` (or the MSVC equivalent), while the LLVM-based compiler
sources retain LLVM's normal compile mode. Link
`obj.PTOASCompilerImplementation` privately to `pybind11::module`, not merely
`pybind11::headers`: the object compilation needs both pybind11 and Python
include usage requirements. Link `PTOASCompiler` privately to the same target
for Python's platform link contract, including macOS dynamic lookup.
`PTOASCompiler` does not become a Python module or gain a `PyInit_*` entry. On
Linux, add a scoped `-z undefs` override for the aggregate's default `-z defs`,
then validate the remaining undefined symbols as described above.

### 3. Make `_core` a Thin Entry Point

`PTOASPythonCore` retains only `NativeModule.cpp` as a source and links the
common DSO:

```cmake
add_mlir_python_extension(PTOASPythonCore _core
  # Existing install and Python binding arguments remain unchanged.
  SOURCES
    NativeModule.cpp
  LINK_LIBS
    PTOASCompiler
)
```

`NativeModule.cpp` calls only the exported compiler and PTO binding bridges.
It must not link `PTOASCompilerImplementation`, `PTOCAPI`, `PTOIR`,
`PTOTransforms`, or LLVM/MLIR components directly.

This arrangement does not create cyclic CMake subdirectories: the common DSO
is declared under `lib` with a deferred reference to the implementation target
declared later under `tools`. It also avoids creating a second implementation
target for wheel builds.

### 4. Control Symbol Visibility

`PTOASCompiler` should default to hidden C/C++ symbol visibility, while the
MLIR Python C API and the narrow PTOAS bridges retain their required exports.
`add_mlir_public_c_api_library` already compiles C API objects with hidden
visibility plus `MLIR_CAPI_BUILDING_LIBRARY`; preserve that mechanism. Add
explicit visibility only for the two PTOAS bridges and avoid a blanket export
list.

This is where PTOAS deliberately differs from a single Python extension such
as Triton's `libtriton`: the generated AddMLIRPython modules require the MLIR C
API to remain exported from the common DSO. The export surface is therefore
"MLIR/PTO C API plus two private bridges", not only one `PyInit_*` symbol.

On Linux, validate the dynamic symbol table with `readelf --dyn-syms`. On
macOS, validate it with `nm -gU`. A version script or exported-symbols list is
optional follow-up work if CMake visibility properties do not produce a small
enough surface.

## Static Selection and Wheel Size

The first size-control mechanism is normal static archive selection:

```text
PTOASCompiler undefined references
              │
              ▼
linker extracts only satisfying members from LLVM/MLIR archives
              │
              ▼
one package-owned compiler DSO
```

There is no project-maintained file or symbol allowlist. There is also no
global `--whole-archive`, `-all_load`, or `-force_load` over LLVM/MLIR, because
that would defeat archive selection and can greatly increase wheel size.

Static registration is the exception that needs explicit testing. Prefer
explicit dialect, pass, translation, and target registration roots. If one
specific upstream archive relies on static constructors and its members are
otherwise unreferenced, retain only that archive with a narrowly scoped
whole-archive mechanism and document why it is required.

Dead stripping and LTO are follow-up optimizations, not correctness
requirements. After the baseline static profile works, linker maps can show
which archive members entered `PTOASCompiler`. Only then should the project
evaluate `-dead_strip`, `--gc-sections`, thin LTO, or further registration
reduction against measured wheel size and link cost.

## Workflow Changes

The PTOAS configure and build command remains structurally identical in every
workflow. It receives `LLVM_DIR` and `MLIR_DIR` from the selected SDK and does
not receive a wheel-specific dependency list.

The following producers remain shared:

```text
local development
.github/workflows/ci.yml
.github/workflows/ci_sim.yml
docker/Dockerfile
docker/Dockerfile.dev
```

The following producers switch their LLVM build and cache to static PIC:

```text
.github/workflows/build_wheel.yml
.github/workflows/build_wheel_mac.yml
```

`auditwheel repair` and `delocate-wheel` remain in place. Their work should
become small because LLVM/MLIR components are no longer external dependencies,
but they still provide platform policy validation and relocate any legitimate
third-party shared dependencies.

## Installation and Standalone Archive

`_core` already uses an install RPATH that includes
`ptoas/mlir/_mlir_libs`, and AddMLIRPython supplies the corresponding layout
for generated modules. Preserve that contract while replacing the common DSO
name.

Update `cmake/RelocatePTOASCompilerArchive.cmake.in` in the same change:

- replace `$<TARGET_FILE_NAME:PTOASPythonCAPI>` with
  `$<TARGET_FILE_NAME:PTOASCompiler>`;
- replace the dependency exclusion matching `libPTOASPythonCAPI` with
  `libPTOASCompiler`;
- continue discovering generated AddMLIRPython modules from their owned
  install directory;
- continue deriving external dependencies from binary metadata rather than a
  handwritten library list.

Update `docker/validate_wheel_payload.py` and any native payload tests to
require `PTOASCompiler` and reject the obsolete DSO.

The archive and wheel must use the same native targets. Packaging may still
stage or relocate those targets differently, but it must not compile a second
compiler implementation.

## Required Invariants

Every supported build must satisfy:

1. Exactly one installed `libPTOASCompiler` exists in the PTOAS package.
2. No installed `libPTOASPythonCAPI` exists.
3. `_core`, generated `_mlir` modules, and `_site_initialize_0` resolve
   `PTOASCompiler`.
4. Python-created MLIR objects and PTOAS compiler code use one MLIR runtime.
5. The compiler archive and wheel originate from the same PTOAS CMake targets.
6. Static and shared LLVM SDK build trees and cache keys never overlap.

For the static wheel profile, additionally:

1. Package extensions do not directly depend on external `libLLVM*`,
   `libMLIR*`, `libPTOIR*`, or `libPTOTransforms*` shared libraries.
2. `PTOASCompiler` does not depend on external LLVM/MLIR component shared
   libraries.
3. A static library is not accidentally bundled as a separate wheel payload
   file.

AddMLIRPython's generated extensions may still privately embed archive members
from upstream-declared `PRIVATE_LINK_LIBS`, currently `LLVMSupport`. Upstream
defines these as hidden, extension-local support code. This exception must not
include MLIR CAPI, IR, context, dialect, pass, or PTOAS compiler implementation
objects: those remain owned by `PTOASCompiler`. It is also not a wheel payload
copy list and must not produce an additional shared-library dependency.

For the shared development profile, LLVM/MLIR component dependencies are
allowed, but they must occur below the common `PTOASCompiler` ownership
boundary rather than being independently linked into `_core`.

## Validation Plan

### Target and Dependency Graph

- Configure both profiles and inspect the generated link commands.
- Assert that all Python native modules link `PTOASCompiler`.
- Assert that `_core` has no direct compiler or LLVM/MLIR implementation
  libraries.
- Inspect raw and repaired wheels with `readelf -d` on Linux and `otool -L` on
  macOS.
- Reject project-external `libLLVM*`, `libMLIR*`, `libPTOIR*`, and
  `libPTOTransforms*` dependencies in static-profile wheel binaries.
- Permit only upstream-declared, hidden `PRIVATE_LINK_LIBS` objects in generated
  extensions; verify that MLIR CAPI/runtime objects are not duplicated there.
- Inspect loader resolution from a clean environment, not from the LLVM build
  directory.

### Functional Coverage

- Import `ptoas`, `ptoas._core`, `ptoas.mlir.ir`, all packaged MLIR dialects,
  and `ptoas.mlir.dialects.pto` in different orders.
- Parse and print representative MLIR modules.
- Construct PTO types, attributes, and enums through the Python bindings.
- Exercise the Python CLI, installed `ptoas` launcher, and compiler paths.
- Run the existing wheel import/compiler suite and standalone archive suite.
- Cover both Linux architectures and both macOS architectures built by the
  release matrix.
- Run the same behavior checks against a shared-profile editable build to
  protect developer workflows.

### Measurements

Record before and after values for:

- cold LLVM SDK build time;
- PTOAS link time and peak memory;
- raw and repaired wheel size;
- uncompressed native payload size;
- number of packaged DSOs and their direct dependency count;
- wheel repair time on every release architecture.

The first rollout is accepted based on correctness, a materially smaller
runtime dependency graph, and no material wheel-size regression. Do not set a
hard repair-time target until at least two uncached runs establish stable
baselines.

## Risks and Mitigations

### Missing Static Registration

Normal archive selection can omit code reachable only through static
constructors. Use explicit registration roots and functional tests. Apply
whole-archive semantics only to a proven, narrowly identified archive.

### Duplicate MLIR Runtime

An extension can accidentally retain a direct static MLIR dependency. Make
binary dependency checks and cross-module Python object tests release gates.

### Hidden Required Symbols

Default-hidden visibility can hide the common C API or PTOAS bridge exports.
Use explicit export macros and inspect the dynamic symbol table on both
platforms.

### Static Link Resource Cost

Static LLVM/MLIR can increase link time and peak memory. Limit this profile to
wheel workflows, cache the LLVM SDK independently, avoid LTO initially, and
measure the final link.

### Unexpected Wheel Size

Broad registration roots or whole-archive flags can retain too much code. Use
the linker map to identify the responsible archives, then reduce roots or
narrow the exceptional retention. Do not replace this analysis with a manual
copy allowlist.

## Rollout

1. Rename and extend the common DSO to `PTOASCompiler`, move the compiler and
   PTO binding implementation into it, and thin `_core` while LLVM remains
   shared everywhere.
2. Validate local development, normal CI, simulator CI, wheel behavior, and
   the standalone archive against the new ownership boundary.
3. Add independent static-PIC LLVM cache profiles to the two wheel workflows
   only.
4. Add binary dependency assertions, switch wheel builds to the static SDK,
   and compare size and repair time with the recorded baseline.
5. Inspect linker maps and consider dead stripping or LTO only if the measured
   result justifies the additional complexity.

Keeping step 1 separate from step 3 makes failures attributable: first prove
the single-runtime ownership change, then change how its implementation is
linked.

## Industry Reference Patterns

No referenced project is an exact template for PTOAS, but several established
patterns support the individual choices in this design:

- [IREE compiler Python bindings](https://github.com/iree-org/iree/blob/main/compiler/bindings/python/CMakeLists.txt)
  link tools and Python-facing code through the project-owned
  `iree_compiler_API_Impl` boundary.
- [Triton's top-level CMake](https://github.com/triton-lang/triton/blob/main/CMakeLists.txt)
  builds one `libtriton` shared Python library, links its compiler and
  LLVM/MLIR targets privately, clears transitive interface libraries, and
  constrains exports. Its LLVM discovery also requests `llvm-config
  --link-static`, making static LLVM a deliberate distribution input rather
  than a copied runtime component graph.
- [TVM's LLVM integration](https://github.com/apache/tvm/blob/main/cmake/utils/FindLLVM.cmake)
  supports LLVM as a build-selected compiler implementation behind a small
  number of project-owned runtime/compiler libraries.
- [torch-mlir's Python CMake](https://github.com/llvm/torch-mlir/blob/main/python/CMakeLists.txt)
  and [CIRCT's Python CMake](https://github.com/llvm/circt/blob/main/lib/Bindings/Python/CMakeLists.txt)
  use `add_mlir_python_common_capi_library` so generated extension modules
  share one native MLIR runtime.

These references support project ownership, a common MLIR runtime, and
build-profile selection. They do **not** establish a universal requirement to
enable `-dead_strip`, LTO, or fully static development builds. Those remain
project-specific optimizations driven by measurement.
