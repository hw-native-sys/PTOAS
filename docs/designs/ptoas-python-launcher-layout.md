// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

# PTOAS Python Launcher Layout Design

## Purpose

This document records the internal launcher contract for the Python-backed
`ptoas` command. The user-facing README should describe how to install and run
PTOAS without exposing these implementation details.

The PTOAS-owned MLIR Python runtime lives under `ptoas.mlir`; its namespace and
native-library isolation requirements are recorded separately in
[`ptoas-mlir-namespace-and-native-isolation.md`](ptoas-mlir-namespace-and-native-isolation.md).

## Entry Model

The wheel launcher uses the standard Python console-script and native-extension
model. Build-tree and install-tree entrypoints are narrow CMake adapters around
the same Python CLI and native extension:

```text
wheel console script -> ptoas._cli.main()
CMake tree wrapper   -> add its own Python root -> ptoas._cli.launch()
both                 -> import ptoas._core -> ptoas._core.main(argv)
```

`ptoas._core` is the single PTOAS-owned native extension. It provides the
compiler entry point and the native PTO dialect bindings used by the public
`ptoas.mlir.dialects.pto` facade. CMake and Python own the platform and ABI-specific
filename. Launcher and packaging code refer to the import name and never
construct `.so`, `.dylib`, or `.pyd` paths.

The LLVM-based driver implementation is compiled as an object library so it
can use LLVM's RTTI and exception settings independently from pybind11. Those
objects are linked into the Python extension; no separate native executable is
produced. Wheel and standalone-archive entrypoints therefore use the same
extension and CLI module.

## Wheel Layout

Wheel console scripts point directly to `ptoas._cli:main`. The CLI imports the
native extension through Python's normal package machinery and resolves TileOps
package data relative to the installed extension. It does not re-execute itself,
load modules by file path, rewrite `PYTHONPATH`, or override Python's standard
package precedence rules.

Editable installs use scikit-build-core's redirect mode. The backend maps the
Python sources to the checkout and the CMake-installed native extension to the
editable build output without package-local path manipulation.

Auditwheel and delocate discover the native extension through the standard
wheel binary scan, bundle its dependencies, and rewrite its runtime search
paths. The launcher does not preload or enumerate those libraries.

Wheel and editable builds use `scikit-build-core` directly as the PEP 517/660
backend. Project metadata and the console entry point live in `pyproject.toml`;
wheel tags, metadata, RECORD generation, CMake configure/build/install, and
editable redirects are owned by the backend rather than repository scripts.

The top-level `pyproject.toml` is the single canonical project configuration.
For the mutually exclusive `ptoas-vmi` distribution,
`packaging/ptoas-vmi/prepare_source.py` is the single staging entry point. It
exports every tracked file from one Git revision and applies
`pyproject.toml.patch`; the wheel is built directly from that tree. The patch
changes only the distribution name, static VMI version, description, CLI label,
and sdist inclusion mode; all CMake and Python package paths remain rooted in
the staged source tree. The checkout's top-level metadata is never modified.

Git owns source selection instead of a hand-maintained directory list.
scikit-build-core then owns PEP 517 metadata and wheel assembly. If an sdist is
requested independently, manual inclusion makes the backend include the complete
tracked snapshot without applying `.gitignore` a second time.
Submodule contents and untracked/generated working-tree files are intentionally
outside this source-distribution contract. Ordinary gates and releases do not
generate or publish a VMI sdist.

CMake's `PTOAS_Python` install component contains only the generated/native
wheel payload. Python source packages are declared through
`tool.scikit-build.wheel.packages`, which also lets editable installs redirect
imports to the source tree without namespace-package or custom `.pth` logic.

## Build-Tree Layout

The build-tree wrapper resolves only its own generated outputs:

- wrapper: `<build>/tools/ptoas/ptoas`
- Python root: `<build>/python`
- native module: importable as `ptoas._core` from the Python root
- TileOps: `<build>/python/ptoas/_runtime/share/ptoas/TileOps`

Missing Python packages or TileOps resources are hard layout errors. The
wrapper only adds `<build>/python` to `sys.path`; `ptoas._cli` owns the common
runtime-resource resolution and native invocation.

## Install-Tree Layout

The install-tree wrapper resolves only files under the same prefix:

- wrapper: `<prefix>/bin/ptoas`
- Python root: `<prefix>`
- native module: importable as `ptoas._core` from the Python root
- TileOps: `<prefix>/ptoas/_runtime/share/ptoas/TileOps`

The install-tree wrapper only adds `<prefix>` to `sys.path`, then delegates to
the same `ptoas._cli` module used by wheels.

Both tree wrappers are CMake packaging adapters, not general runtime discovery
helpers. Each wrapper adds exactly the Python root belonging to its own tree;
it never scans repositories, neighboring build directories, or environment
variables for another installation.

## Standalone Archive Layout

Standalone compiler archives contain the installed Python wrapper and package:

```text
bin/ptoas
ptoas/_cli.py
ptoas/_core.<abi>.so
ptoas/_runtime/share/ptoas/TileOps
ptoas/mlir/
ptodsl/
lib/<native dependencies>
tilelang_dsl/
```

The archive is assembled by installing `PTOAS_Python` and then
`PTOAS_CompilerArchive` into one staging prefix. The second component owns the
wrapper and compiler-time Python resources and performs relocation against the
already staged native payload. Packaging code does not scan source trees or
assemble Python packages from unrelated build directories.

The current archive is built against CPython 3.11 and requires a CPython 3.11
interpreter. `bin/ptoas` adds the archive root to `sys.path`, then uses the same
`ptoas._cli -> ptoas._core` path as the install tree. The packaged `ptodsl/`
tree supports the compiler's PTODSL TileLib implementation; it does not turn
the archive into a normal pip-installable PTODSL distribution.

Linux archives use package-relative and archive-relative `$ORIGIN` RPATHs;
macOS archives use the equivalent `@loader_path` install names. Package-owned
MLIR extensions remain under `ptoas/mlir/_mlir_libs`, while only external
native dependencies are collected under the archive `lib/` directory.

## PTODSL and TileLib Runtime

PTODSL imports MLIR and the PTO dialect through normal Python package
resolution. Wheel and editable installs provide those packages through their
declared installation layout. CTest and direct developer-tree runs must set an
explicit matching `PYTHONPATH`; PTODSL does not guess repository, LLVM build,
or PTOAS install paths at import time.

TileOp expansion runs in the CLI's existing Python process. `_core.main` creates
one Python-owned MLIR context for the compilation session, and the native driver
borrows that exact context. The in-process TileLib service materializes a source
module in the shared context and clones it into native ownership before the
Python module owner can be released. The packaged MLIR bindings and PTOAS must
therefore remain one ABI-matched runtime rather than independently replaceable
components.
