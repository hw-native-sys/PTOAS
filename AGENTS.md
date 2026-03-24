# Repository Guidelines

## Project Structure & Module Organization
Core dialect definitions, IR types, and pass declarations live in `include/PTO/`; implementations live in `lib/PTO/`. User-facing tools are under `tools/`, especially `tools/ptoas`. Python bindings and generated dialect helpers are in `python/`. Regression inputs and executable samples live in `test/`, with operator samples in `test/samples/`, backend checks in `test/phase2/` and `test/phase3/`, and NPU validation helpers in `test/npu_validation/`. Design notes and specs are in `docs/`, including `docs/vpto-spec.md`.

## Build, Test, and Development Commands
Source the repo environment before building or running samples:

```bash
source env.sh
bash do_cmake.sh --llvm "$LLVM_ROOT"
cmake --build build -j
```

- `source env.sh`: exports LLVM, Python, and PTOAS paths for local work.
- `bash do_cmake.sh --llvm "$LLVM_ROOT"`: configures the in-tree `build/` directory.
- `cmake --build build -j`: builds libraries, `ptoas`, and Python bindings.
- `./test/samples/runop.sh -t Abs`: runs a focused sample family.
- `./test/samples/runop.sh all`: runs the broader sample sweep.
- `bash test/samples/run_a5vm_acceptance_checks.sh`: executes targeted A5VM regressions.

## Coding Style & Naming Conventions
Use C++17 and follow nearby MLIR/LLVM conventions. Match existing indentation: 2 spaces in TableGen, and the surrounding file’s style in C++. Prefer descriptive helper names such as `lowerTLOAD`, `build...Scope`, or `extract...Contract`. Keep source ASCII unless a file already uses Unicode. New sample files should generally use lowercase snake case unless extending an existing sample family.

## Testing Guidelines
Start with the smallest relevant test, then expand coverage. For backend work, prefer `--pto-backend=a5vm --a5vm-print-ir` so you can inspect raw A5VM IR before textual emission. Keep sample expectations aligned with the active backend output. Mark intentionally unsupported cases as `SKIP` or `XFAIL` with a concrete reason.

## A5 Semantic Source Of Truth
When changing A5 lowering, LLVM emission, sample semantics, or validation
oracles, inspect the installed PTO implementation under `ASCEND_HOME_PATH`
first and treat it as the semantic baseline.

Do not infer A5 behavior from repo-local lowering or emitter code when the
installed PTO headers can answer it. Only allow a repo-local intrinsic
replacement after confirming that the replacement relationship exists at the
intrinsic/compiler-contract layer as well.

If the installed PTO headers are not enough, trace the real frontend-produced
artifacts with the current Bisheng toolchain, for example via testcase build
flags plus `-v` and `-save-temps`, before changing behavior.

## Commit & Pull Request Guidelines
Use short, imperative commit subjects, for example `Fix explicit StringAttr bool conversion`. Keep each commit scoped to one coherent change. PRs should describe the affected lowering path, list the exact validation commands run, call out samples newly passing or intentionally deferred, and include relevant IR or output snippets for backend-facing changes.

## Configuration Notes
Build in the repository’s `build/` directory, not ad hoc paths under `/tmp`. Avoid committing local-only environment helpers unless they are meant to be shared project scripts.
