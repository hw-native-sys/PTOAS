---
name: ptoas-test-framework-guidance
description: Guidance for adding moving reviewing or validating PTOAS tests across lit VPTO runtime and TileLang ST frameworks.
---

# PTOAS Test Framework Guidance

Use this skill before adding or relocating PTOAS tests. The goal is to put each test in the framework that actually runs it and validates the intended behavior.

## Framework Selection

- Use `test/lit` for compiler regression tests: parser/printing, verifier diagnostics, pass output, IR rewrites, CLI behavior, and `FileCheck`-style checks.
- Use `test/lit/pto` for generic PTO/EmitC/DSL-lowering lit tests that do not run the VPTO backend.
- Use `test/lit/vpto` for lit tests whose `RUN:` line uses `--pto-backend=vpto`, `--emit-vpto`, VPTO pass dumps, or VPTO-specific diagnostics.
- Use `test/lit/vpto/cube` for focused cube VPTO lit tests when grouping with existing cube tests is clearer.
- Use `test/vpto` for runtime validation of VPTO-generated fatobj/kernel behavior on SIM or NPU. These tests must compile and run, generate data, and compare results.
- Use `test/tilelang_st` for TileLang DSL system tests that validate DSL rendering plus build/run/compare behavior through the TileLang ST harness.

## Do Not

- Do not add new `.pto` tests under `test/basic`; lit does not discover that directory.
- Do not place VPTO backend lit tests under `test/lit/pto`; put them under `test/lit/vpto`.
- Do not add runtime/simulator expectations to lit tests. If correctness depends on executing a kernel and comparing data, use `test/vpto` or `test/tilelang_st`.
- Do not add test path or coverage notes to ISA/spec manuals. `docs/isa` and `docs/vpto-spec.md` should describe op semantics and interfaces, not test locations.
- Do not fix unrelated compiler behavior while only moving tests. If relocation exposes stale tests, report the stale category separately before broad rewrites.

## `test/lit` Rules

- Every lit `.pto` file needs at least one `// RUN:` line.
- Prefer small single-purpose checks with `FileCheck`.
- Use `not ptoas ... 2>&1 | FileCheck %s` for negative verifier/parser tests.
- Use `--emit-pto-ir` when checking PTO IR output.
- Use `--pto-backend=vpto --emit-vpto` or `--mlir-print-ir-*` when checking VPTO IR/pass output.
- Keep output checks meaningful; do not reduce a test to only a function-name check if the old test verified richer behavior.

Validation commands:

```bash
lit --show-tests build/test/lit
lit -v build/test/lit --filter '<case-name>'
cmake --build build -j64 --target check-pto
```

If `lit` or `FileCheck` is missing, use the `llvm-test-tool-fallback` skill before treating it as a test failure.

## `test/vpto` Runtime Rules

Use `test/vpto/cases` when the test must prove generated VPTO code executes correctly.

A case directory is discovered only when it contains:

- `kernel.pto`
- `launch.cpp`
- `main.cpp`
- `golden.py`
- `compare.py`

Current VPTO runtime cases should use the unified fatobj flow emitted by `ptoas`; do not add per-case `stub.cpp` or split `cube.pto`/`kernel.pto` unless the framework explicitly changes. For mixed cube/vector kernels, keep the code in `kernel.pto` using the current module/section form expected by PTOAS.

Validation commands:

```bash
WORK_SPACE=/tmp/pto-vpto CASE_NAME='<relative-case>' \
  test/vpto/scripts/run_host_vpto_validation.sh

WORK_SPACE=/tmp/pto-vpto JOBS=64 \
  test/vpto/scripts/run_host_vpto_validation_parallel.sh
```

Required environment normally includes `ASCEND_HOME_PATH`; SIM runs may need `SIM_LIB_DIR` if auto-detection fails.

## `test/tilelang_st` Rules

Use `test/tilelang_st` when the behavior starts from TileLang DSL and must be verified through DSL-generated `.pto`, build, run, data generation, and comparison.

Testcases live under:

```text
test/tilelang_st/npu/<soc>/src/st/testcase/<testcase>/
```

The batch runner discovers a testcase when `<testcase>.pto` exists in that testcase directory. Keep testcase data generation and case definitions with the existing ST structure, usually including `cases.py` when multiple parameterized cases are needed.

Validation commands:

```bash
python3 test/tilelang_st/script/run_all_st.py --list
python3 test/tilelang_st/script/run_all_st.py -r sim -v a5 -t '<testcase>' --smoke -j 1
python3 test/tilelang_st/script/run_all_st.py -r sim -v a5 --smoke -j 64
```

Use the ST harness rather than ad-hoc scripts, so CI and local validation exercise the same path.

## Finishing Checklist

- The test is in the framework that matches its assertion: compile/IR, VPTO runtime, or TileLang ST runtime.
- New lit tests are visible in `lit --show-tests build/test/lit`.
- VPTO backend lit tests live under `test/lit/vpto`, not `test/lit/pto`.
- Runtime tests are discoverable by their framework without special-case script logic.
- The smallest relevant validation command was run, and failures are reported by framework and category.
