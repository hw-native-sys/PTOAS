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

## CANN Run-Package Layout

The CANN component package (`pto_as` run package and its rpm/deb equivalents)
does not ship a private Python tree. Instead it aligns with the sibling `pypto`
component: the self-contained PTOAS wheel is `pip install --no-deps --target`ed
into the CANN shared site-packages, and the command is exposed through the
version-level `bin/` directory that the toolkit `set_env.sh` already puts on
`PATH`.

- wheel source: `<version>/tools/ptoas/wheels/ptoas*.whl`
- Python payload: `<version>/python/site-packages/{ptoas,ptodsl,TileOps,SoftOps,ptoas.libs}`
  (+ the single `ptoas-<version>.dist-info`, and pip's `bin/` holding the
  console script that `[project.scripts]` generates under `--target`)
- interpreter record: `<version>/tools/ptoas/.ptoas-python.path`
- launcher (component-owned): `<version>/tools/ptoas/bin/ptoas`
- command on PATH: `<version>/bin/ptoas` — a relative symlink to the launcher,
  created with the same `createrelativelysoftlink` idiom used for the opapi
  softlinks

After the standard `source <cann>/set_env.sh`, `set_env.sh` prepends
`<version>/python/site-packages` to `PYTHONPATH` and `<version>/bin` to `PATH`,
so both `import ptoas` / `import ptodsl` and the `ptoas` command work with no
PTOAS-specific environment script. The launcher also prepends the shared
site-packages to `PYTHONPATH` itself, so invoking it by absolute path works even
before `set_env.sh` is sourced. It resolves its own symlink chain before
deriving `<version>`, so reaching it through the `<version>/bin/ptoas` symlink
computes the same layout as a direct invocation.

Install (`pto_install_wheel`) unpacks the wheel into a private staging tree
(`tools/ptoas/.ptoas-wheel-staging`) and then migrates only PTOAS's own entries
into the shared site-packages. Staging is required because
`pip install --upgrade --target` deletes and recreates the generated-script
directory of its target: pointing pip straight at the shared site-packages would
wipe the `bin/` commands of every sibling component. The migration removes only
same-named entries, so foreign files in `site-packages/` and in
`site-packages/bin/` survive both a first install and a reinstall. Install also
deletes the pre-site-packages-era private tree `tools/ptoas/python` when
upgrading over an older install, so the legacy payload is not stranded.

Exposing the command needs one more care: `<version>/bin` is a symlink to
`<version>/<arch>/bin`, and the filelist creates that per-architecture directory
with mode 550, which denies its owner the write permission needed to add the
`ptoas` symlink. `pto_install_wheel` therefore reads the directory mode through
the symlink (`stat -L`), adds the owner write bit for the link operation, and
restores the recorded mode afterwards. Reading it without `-L` would capture the
symlink's own 777 and the restore step would then leave the shared directory
world-writable.

Because the payload spans several top-level entries, replacing it is not one
atomic step. Before migrating, the version already installed is moved into
`tools/ptoas/.ptoas-wheel-backup`; if any later step fails, the new entries are
removed and the parked payload is put back. Failure semantics are uniform:

- Every failure returns non-zero. `ptoas` is never reported as installed while
  the command is missing, so a missing `bin_dir`, a missing launcher, or a
  failed symlink is an install failure like any other.
- A failure after the previous payload was parked restores it, so a failed
  upgrade leaves the previously installed version working and runnable.
- The interpreter record is written only after the command is published, so a
  failed install cannot leave a launcher pointing at a runtime that is absent.
- If the restore itself cannot complete, the backup tree is deliberately kept —
  it is then the only copy of the previous install — and its path is reported.

Uninstall (`pto_uninstall_wheel`) mirrors `pypto`: it removes the full PTOAS
payload — `ptoas/`, `ptodsl/`, `TileOps/`, `SoftOps/`, `ptoas.libs/`,
`ptoas-*.dist-info`, and PTOAS's own command in the shared `bin/` — drops the
`<version>/bin/ptoas` symlink, deletes the interpreter record plus any staging or
backup tree, and finally removes the `python/` tree itself when PTOAS was its
last occupant. Sibling components in the shared site-packages are never touched:
the empty-dir `rmdir` steps fail harmlessly while other components still own
entries. The rpm/deb prerm hook falls back to an inline removal only when
`pto_common.sh` is unreadable, and repeats the same name list there, including
the legacy private tree.

The shared command is matched in both of its forms. It is a symlink in the
normal layout, but the console script that pip generates for the wheel lands in
the same directory as a regular file, so every check and removal treats "exists
or is a symlink" as present; a symlink test alone would miss an undeletable
regular file and report the removal as successful.

`pto_remove_site_payload` and `pto_uninstall_wheel` return non-zero when any of
PTOAS's own entries survived, instead of reporting the status of whichever step
ran last. Without that, a payload deletion that failed would be followed by the
successful removal of the interpreter record and the caller would read the whole
removal as successful. Both callers — `pto_uninstall.sh` and the rpm/deb prerm
hook — propagate that status to their own callers.

### Uninstall entry point

The user-facing entry point `share/info/pto_as/script/uninstall.sh` (invoked
directly, or through the toolkit's `cann_uninstall.sh`) hands the component
installer the *version root* — the directory four levels above the script, i.e.
`script -> pto_as -> info -> share -> <version>`. The component installer
reconstructs the version directory from that argument and only honours the name
it is given when the path itself already looks like a version directory
(`is_version_dirpath`: it contains `share/info`). Passing the parent directory
instead makes it fall back to the hardcoded default name `cann`, so an install
whose version directory is called something else — the multi-version layout
`<prefix>/cann-9.2.0/ascend-toolkit`, for example — would be looked up at
`<parent>/cann`, found missing, and left completely in place; that branch also
exits successfully, so the removal reports success while every file stays on
disk.

Two checks make that class of silent no-op impossible, because a removal must
never be reported as successful while files are still installed:

- The component installer propagates its own removal status, and its caller
  exits with that status. Reporting the status of an unrelated log call instead
  let a failed removal look successful.
- Afterwards the entry point checks that none of the paths PTOAS owns still
  exist — the component info tree, `tools/ptoas` (wheels, launcher and
  interpreter record), the arch version header, the shared `site-packages`
  payload and the command entry in either of its forms — and fails loudly naming
  the first leftover, rather than trusting an exit status. Every path checked
  belongs to PTOAS alone, so a sibling component sharing the version directory
  cannot make the check fire.

The cleaned-up skeleton is bounded by what the installer actually created. The
installer records that list (`.ptoas-created-dirs` under `share/info/pto_as`)
before it creates anything: it walks up from the version directory and stops at
the first ancestor that already exists, so every recorded directory is one that
did not exist before. Removal deletes exactly those entries and stops there —
there is deliberately no walk up the tree. An unrelated empty directory such as
`/data/team/Ascend` is therefore never a candidate even when it happens to be
empty, which `rmdir` alone cannot distinguish from a directory the installer
made.

Removal sweeps the recorded list repeatedly until a pass removes nothing, rather
than once in the order the file happens to be written, so a list that names an
ancestor before its descendant still converges: the ancestor is not empty on the
first pass and is empty on the next. The installer writes the list deepest first
for the common case; the sweep is what makes the removal independent of that
ordering, including for a record an earlier revision wrote the other way round.
Only empty directories are ever removed by either path, so the first pass that
finds nothing to do is the correct place to stop.

Two details make that record trustworthy across the install lifecycle. Because
the metadata directory is installed read-only, writing the record restores the
owner write bit for the write and puts it back, and the outcome is checked —
an unwritten record would silently disable the cleanup. Because a reinstall over
an existing install creates no directories at all, its chain is empty and the
previous record is carried forward, so the directories the *first* install
created are not forgotten. The inner removal script reads the record from the
module directory and derives the version root from it: in that script the path
named `TARGET_VERSION_DIR` is the version directory's `share/info` subtree, not
the version directory itself.

The shared command is also checked and removed under both of its spellings:
`<version>/bin/ptoas` and `<version>/<arch>-linux/bin/ptoas`. `<version>/bin` is
a symlink into the architecture directory, and the link really lives in the
latter, so looking only at the former would leave a dangling command link behind
once the toolkit has torn the shared symlink down. Both directories are created
read-only, so the write bit is restored per entry around its own parent
directory rather than once for the shared path.

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
