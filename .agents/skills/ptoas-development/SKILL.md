---
name: ptoas-development
description: Prepare, build, test, compile with, or troubleshoot a PTOAS checkout on Linux, WSL, or Ubuntu. Use for any PTOAS development request when the workspace state is unknown, including initial setup or repair, Python frontend and native incremental builds, CLI compilation, VPTO output checks, and LLVM/Python runtime failures.
---

# PTOAS Development

Use this skill as the single entry point for PTOAS development. First probe the
checkout, then use the matching path below. Do not assume a global `ptoas`, a
conda environment, a build directory, or that the checkout has been initialized.

## Workspace contract

A prepared source workspace has one isolated virtual environment and one PTOAS
build tree. Its runtime commands use this contract:

```bash
export PTO_SOURCE_DIR=<workspace>/PTOAS
export LLVM_BUILD_DIR=<workspace>/llvm-project/build-shared
export VENV_DIR=<workspace>/.venv
export PYTHON_BIN="$VENV_DIR/bin/python"
export PTOAS_BIN="$VENV_DIR/bin/ptoas"
export PTO_BUILD_DIR="$PTO_SOURCE_DIR/build"
```

`<workspace>` is a placeholder, not a literal directory or shell syntax. Probe
the current checkout or source its generated `env.sh` to obtain the real paths.

For an isolated feature worktree created by `ptoas-workspace-manager`, source
its generated `env.sh`. It is authoritative: it selects the worktree's venv,
build tree, LLVM build, and optional CANN environment. Do not substitute a
parent venv or share its `PTO_BUILD_DIR` with another worktree.

## Probe before acting

Run this from the intended checkout. Prefer explicit valid environment
variables and workspace metadata; use local defaults only when unambiguous.

```bash
PTO_SOURCE_DIR="$(git rev-parse --show-toplevel)"
if [[ -f "$PTO_SOURCE_DIR/.ptoas-workspace.json" ]]; then
  # The manager generated env.sh alongside this metadata.
  source "$PTO_SOURCE_DIR/env.sh"
else
  : "${PTO_BUILD_DIR:=$PTO_SOURCE_DIR/build}"
  if [[ -z "${PYTHON_BIN:-}" && -x "$PTO_SOURCE_DIR/.venv/bin/python" ]]; then
    PYTHON_BIN="$PTO_SOURCE_DIR/.venv/bin/python"
  elif [[ -z "${PYTHON_BIN:-}" && -x "$(dirname "$PTO_SOURCE_DIR")/.venv/bin/python" ]]; then
    # This is the standard layout in the repository README.
    PYTHON_BIN="$(dirname "$PTO_SOURCE_DIR")/.venv/bin/python"
  fi
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    VENV_DIR="$(dirname "$(dirname "$PYTHON_BIN")")"
    PTOAS_BIN="$(dirname "$PYTHON_BIN")/ptoas"
  fi
fi

test -n "${LLVM_BUILD_DIR:-}" && test -d "$LLVM_BUILD_DIR/lib/cmake/llvm"
test -n "${LLVM_BUILD_DIR:-}" && test -d "$LLVM_BUILD_DIR/lib/cmake/mlir"
test -n "${PYTHON_BIN:-}" && test -x "$PYTHON_BIN"
test -f "$PTO_BUILD_DIR/CMakeCache.txt" || true
test -x "${PTOAS_BIN:-/nonexistent}" || true
```

If the normal-checkout probe finds a CMake tree but no explicit workspace
contract, read its cache only to report prior selections. It is evidence, not
permission to reuse its Python or LLVM path:

```bash
if [[ -f "$PTO_BUILD_DIR/CMakeCache.txt" ]]; then
  sed -n -E \
    -e 's|^(LLVM_DIR|MLIR_DIR):[^=]*=|cache \1=|' \
    -e 's|^(_Python3_EXECUTABLE|_Python_EXECUTABLE):[^=]*=|cache \1=|' \
    "$PTO_BUILD_DIR/CMakeCache.txt"
fi
```

Validate each reported candidate before choosing it. In particular, do not
reuse a Python recorded in an old cache as the workspace Python: create or
select the workspace venv first, then verify that its Python ABI matches the
chosen LLVM bindings.

Route from the results:

- Complete workspace: use the daily loop.
- LLVM exists but the venv, editable install, or PTOAS CMake tree is absent:
  initialize against that LLVM build.
- LLVM itself is missing or lacks the required CMake packages: bootstrap it.
- A cache-only, incomplete, or multiple-candidate workspace: report the
  candidates and ask which to use. Never bind a build tree to a guessed LLVM
  or erase a mismatched CMake cache.

The editable install records the chosen external LLVM library directory in its
native-extension RPATH. Do not require users to set `LD_LIBRARY_PATH` for the
normal CLI or Python workflow. If a direct, non-packaged binary later reports a
missing LLVM shared library, diagnose that artifact and its RPATH rather than
adding a global shell setting by default.

## Initialize or repair PTOAS

Use this path after resolving a compatible LLVM build and the Python intended
for the workspace. For a normal checkout, create it at the workspace root
(`$PTO_SOURCE_DIR/../.venv`), as in the repository README; manager-owned
worktrees already have one.

```bash
test -d "$LLVM_BUILD_DIR/lib/cmake/llvm"
test -d "$LLVM_BUILD_DIR/lib/cmake/mlir"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  VENV_DIR="$(dirname "$PTO_SOURCE_DIR")/.venv"
  python3 -m venv "$VENV_DIR"
  PYTHON_BIN="$VENV_DIR/bin/python"
fi
PTOAS_BIN="$(dirname "$PYTHON_BIN")/ptoas"
: "${PTO_BUILD_DIR:=$PTO_SOURCE_DIR/build}"

LLVM_BUILD_DIR="$LLVM_BUILD_DIR" \
PTO_BUILD_DIR="$PTO_BUILD_DIR" \
PYTHON_BIN="$PYTHON_BIN" \
  "$PTO_SOURCE_DIR/quick_install.sh"

"$PTOAS_BIN" --version
"$PYTHON_BIN" -c 'from ptoas.mlir.dialects import pto; print("ptoas import OK")'
```

`quick_install.sh` is the configuration and editable-install entrypoint. It
automatically uses `ccache` when available. Do not run it after every native
edit: retain the configured tree and build the affected target with Ninja.
If CMake reports a generator or source-path mismatch, preserve the build tree
and ask before deleting or reconfiguring its generated state.

## Bootstrap LLVM on Linux or WSL

Use only when no compatible LLVM build exists. Prefer a Linux filesystem over
`/mnt/c` for substantial WSL builds. Install missing prerequisites, including
venv support:

```bash
sudo apt-get update
sudo apt-get install -y git cmake ninja-build build-essential \
  python3 python3-dev python3-pip python3-venv
```

Set the workspace contract, create the venv, and install LLVM build Python
dependencies before configuring LLVM. PTOAS requires
`vpto-dev/llvm-project:feature-vpto`, shared libraries, and MLIR Python
bindings:

```bash
"$PYTHON_BIN" -m pip install \
  'scikit-build-core>=0.12.2,<2' 'pybind11<3' numpy

git clone https://github.com/vpto-dev/llvm-project.git "$LLVM_SOURCE_DIR"
git -C "$LLVM_SOURCE_DIR" checkout feature-vpto
cmake -G Ninja -S "$LLVM_SOURCE_DIR/llvm" -B "$LLVM_BUILD_DIR" \
  -DLLVM_ENABLE_PROJECTS='mlir;clang' -DBUILD_SHARED_LIBS=ON \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON -DLLVM_ENABLE_ASSERTIONS=ON \
  -DPython3_EXECUTABLE="$PYTHON_BIN" -DPython_EXECUTABLE="$PYTHON_BIN" \
  -Dpybind11_DIR="$("$PYTHON_BIN" -m pybind11 --cmakedir)" \
  -DCMAKE_BUILD_TYPE=Release -DLLVM_TARGETS_TO_BUILD=host
ninja -C "$LLVM_BUILD_DIR"
```

Then follow **Initialize or repair PTOAS**. Reuse an existing compatible LLVM
build; do not rebuild it solely because PTOAS source changed.

## Daily build, test, and compile loop

Always use the selected workspace tools, not `python`, `ptoas`, or `ninja`
from an unrelated environment.

```bash
# Python-only frontend change: normally no native rebuild is required.
"$PYTHON_BIN" path/to/focused_test.py

# Native source, CMake, or generated binding change.
ninja -C "$PTO_BUILD_DIR" ptoas

# Full configured build or project test suite.
ninja -C "$PTO_BUILD_DIR"
ninja -C "$PTO_BUILD_DIR" check-pto

# Compile a supplied PTO input.
"$PTOAS_BIN" input.pto -o output.cpp
```

Run `quick_install.sh` again only when establishing/repairing the editable
installation or changing its Python, LLVM, or build-directory selection.

## VPTO smoke check

For a known end-to-end frontend and VPTO check, use Abs with explicit paths.
`--emit-vpto` writes textual VPTO output; do not use `--vpto-print-ir` as a
file-output substitute.

```bash
PTOAS_BIN="$PTOAS_BIN" \
PYTHON_BIN="$PYTHON_BIN" \
PTOAS_OUT_DIR=/tmp/ptoas-abs-vpto \
PTOAS_FLAGS='--pto-arch a5 --pto-backend=vpto --emit-vpto' \
  "$PTO_SOURCE_DIR/test/samples/runop.sh" -t Abs

sed -n '1,260p' /tmp/ptoas-abs-vpto/Abs/abs-pto.cpp
```

Expect `pto.backend = "vpto"`, `pto.target_arch = "a5"`, the Abs kernel, and
the GM/UB copy plus vector operations. Report the generated path and the first
concrete failure class: missing LLVM packages, editable installation, CMake
generator mismatch, dynamic-linker failure, build error, or test failure.
