# Workspace layout and safety contract

The manager deliberately separates mutable state:

```text
shared-root/
├── llvm-project/build-vpto21-py311/   # shared LLVM_BUILD_DIR
├── .ccache/                           # optional shared compiler cache
├── workspaces/feature-a/              # Git worktree + .venv + env.sh
└── builds/feature-a/                  # PTOAS CMake/Ninja tree
```

The workspace metadata file contains absolute paths and the branch name. It is
an operational file and is not intended to be committed to PTOAS. The generated
`env.sh` exports `LLVM_BUILD_DIR`, `PTO_BUILD_DIR`, `PTO_SOURCE_DIR`, and
`PYTHON_BIN`; it exports `CCACHE_DIR` when supplied and sources CANN only if
the creator provided a script path. The editable PTOAS installation records its
external LLVM runtime path, so ordinary CLI and Python use does not require an
`LD_LIBRARY_PATH` export.

Compatibility rules:

- LLVM build and Python ABI must match the PTOAS build. A Python 3.11 LLVM
  binding build should not be reused with a Python 3.12 workspace.
- A CANN environment is shared by reference, never cloned by this tool.
- `PTO_BUILD_DIR` must not equal the repository path, the workspace path, or
  `LLVM_BUILD_DIR`.
- `destroy` treats a missing PR, a non-merged PR, an unavailable `gh`, a dirty
  worktree, an unexpected untracked file, a local commit not contained in the
  merged PR head, and an invalid worktree registration as unsafe. Only the
  manager-owned `.venv/`, metadata, and `env.sh` are ignored by the
  source-change check.
- `destroy --yes` is still gated by the same checks; `--yes` only confirms the
  already-validated deletion.
