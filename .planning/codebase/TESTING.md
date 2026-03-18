# Testing Patterns

**Analysis Date:** 2026-03-18

## Test Framework

**Runner:**
- Primary native runner: CTest via `include(CTest)` in `CMakeLists.txt`.
- CTest config lives in `tools/ptobc/tests/CMakeLists.txt`.
- Additional test execution is orchestrated by shell scripts and GitHub Actions rather than `pytest` or `unittest`.

**Assertion Library:**
- No Python assertion framework is used for repo-wide tests.
- MLIR textual checks use `FileCheck` directives embedded in `.mlir` and `.pto` files under `test/basic/` and `test/samples/SCF/`.
- Shell tests assert via process exit codes, `cmp`, and `test -s` in `tools/ptobc/tests/stage9_e2e.sh` and `tools/ptobc/tests/ptobc_to_ptoas_smoke.sh`.

**Run Commands:**
```bash
ctest --test-dir build --output-on-failure     # Run configured CTest tests
ninja -C build test                            # Build-tree test target via CMake/CTest
bash test/samples/runop.sh --enablebc all      # Run sample Python/PTO/ptobc/ptoas pipeline
```

## Test File Organization

**Location:**
- Use separate test trees rather than co-located unit tests.
- MLIR/FileCheck regression tests live in `test/basic/`.
- Sample-driven end-to-end tests live in `test/samples/`.
- CTest wrapper scripts and opcode checks live in `tools/ptobc/tests/`.
- NPU validation generators and templates live in `test/npu_validation/scripts/` and `test/npu_validation/templates/`.

**Naming:**
- Shell/CTest smoke tests use descriptive snake_case names. Examples: `tools/ptobc/tests/stage9_e2e.sh`, `tools/ptobc/tests/ptobc_to_ptoas_smoke.sh`, `tools/ptobc/tests/opcode_coverage_check.py`.
- Sample regressions often use `test_*.py` or scenario names inside feature directories. Examples: `test/samples/Sync/test_barrier_sync.py`, `test/samples/Sync/test_inject_sync_loop.py`.
- Expected-failure samples use suffixes like `_invalid` or `_xfail`; `test/samples/runop.sh` interprets those suffixes as expected failures.

**Structure:**
```text
test/
├── basic/                  # FileCheck-style MLIR regressions
├── samples/<Feature>/      # Python/PT0/CPP sample cases and generated artifacts
└── npu_validation/         # Generated remote-board validation support

tools/ptobc/tests/
├── CMakeLists.txt          # CTest definitions
├── stage9_e2e.sh           # encode -> decode -> re-encode bytecode roundtrip
├── ptobc_to_ptoas_smoke.sh # bytecode -> ptoas smoke test
└── opcode_coverage_check.py
```

## Test Structure

**Suite Organization:**
```cmake
add_test(NAME ptobc_stage9_e2e
  COMMAND ${CMAKE_COMMAND} -E env
    PTOBC_BIN=$<TARGET_FILE:ptobc>
    PTOBC_ALLOW_GENERIC=1
    TESTDATA_DIRS=${PTObc_TESTDATA_DIR}:${CMAKE_SOURCE_DIR}/test/samples
    ${CMAKE_CURRENT_LIST_DIR}/stage9_e2e.sh
)
```

**Patterns:**
- Define CTest suites as black-box command invocations with required env vars in `tools/ptobc/tests/CMakeLists.txt`.
- Prefer artifact-pipeline testing over fine-grained unit tests:
  - Python sample emits `.pto`
  - `ptobc` optionally round-trips bytecode
  - `ptoas` lowers to `.cpp`
  - remote scripts may compile and run on target hardware
- Keep feature-specific golden checks inside the source file when using FileCheck. Examples: `test/basic/plan_memory_reuse_sequential.mlir` and `test/samples/SCF/scf_while_break.pto`.
- Preserve multi-case execution even if some samples fail. `test/samples/runop.sh` aggregates failures instead of exiting on first error.

## Mocking

**Framework:** None detected.

**Patterns:**
```typescript
Not applicable. The codebase does not use a mocking library such as pytest-mock,
unittest.mock, gmock, or a dependency-injection test seam.
```

**What to Mock:**
- Not applicable for current patterns. Tests prefer running real CLI binaries (`ptoas`, `ptobc`) and real sample inputs.

**What NOT to Mock:**
- Do not replace the compiler pipeline with stubs in repo-native tests. Existing tests validate emitted PTO/bytecode/C++ artifacts using actual binaries from `build/tools/ptoas/ptoas` and `build/tools/ptobc/ptobc`.

## Fixtures and Factories

**Test Data:**
```typescript
# FileCheck-style embedded expectation
// RUN: ptoas %s 2>&1 1>/dev/null | FileCheck %s
// CHECK: end PTO plan Mem!
// CHECK: func.func @reuse_sequential

# Roundtrip test data roots
tools/ptobc/testdata/add_static_multicore.pto
tools/ptobc/testdata/matmul_static_singlecore.pto
test/samples/<Feature>/*.pto
```

**Location:**
- Static bytecode fixture inputs live in `tools/ptobc/testdata/`.
- Compiler sample fixtures live in `test/samples/`.
- Generated NPU validation code is templated from `test/npu_validation/templates/` by `test/npu_validation/scripts/generate_testcase.py`.

## Coverage

**Requirements:** None enforced by a coverage tool.

**View Coverage:**
```bash
Not detected. No coverage command or coverage config file is present.
```

## Test Types

**Unit Tests:**
- Minimal. The nearest unit-style check is `tools/ptobc/tests/opcode_coverage_check.py`, which compares PTO mnemonics from `include/PTO/IR/PTOOps.td` against `tools/ptobc/generated/ptobc_opcodes_v0.h`.

**Integration Tests:**
- Primary test style.
- `tools/ptobc/tests/stage9_e2e.sh` validates `ptobc encode -> decode -> re-encode` determinism over both `tools/ptobc/testdata/` and `test/samples/`.
- `tools/ptobc/tests/ptobc_to_ptoas_smoke.sh` validates `ptobc encode -> ptoas lower` on a representative sample.
- `test/samples/runop.sh` validates Python binding generation and compiler lowering across many feature directories.

**E2E Tests:**
- Remote-board/NPU validation is used as hardware-level E2E coverage.
- CI workflow `.github/workflows/ci.yml` builds a payload, uploads it, and optionally runs `remote-npu-validation` using generated `npu_validation/` directories.
- `test/npu_validation/scripts/generate_testcase.py` creates per-sample `main.cpp`, `golden.py`, `compare.py`, and `run.sh`.

## Common Patterns

**Async Testing:**
```typescript
Not applicable. No asynchronous runtime or event-loop test framework is used.
```

**Error Testing:**
```bash
case "$base" in
  *_invalid|*_xfail) expect_fail=1 ;;
esac

if ! "$python" "$f" > "$mlir"; then
  if [[ $expect_fail -eq 1 ]]; then
    echo -e "${A}(${base}.py)\tXFAIL\tpython failed as expected"
    continue
  fi
fi
```
- Expected-failure semantics are encoded in filenames and handled by `test/samples/runop.sh`.
- Shell tests treat missing tools or fixtures as explicit errors with exit code `2`.
- FileCheck regressions assert specific emitted IR/text fragments rather than exact whole-file matches.

## Prescriptive Guidance

- Add new compiler regression tests to `test/basic/` when the assertion is textual IR or pass output and can be expressed with `// RUN:` plus `FileCheck`.
- Add new pipeline/sample tests to `test/samples/<Feature>/` when the scenario is best exercised through Python bindings, `.pto` assembly, bytecode roundtrips, or emitted C++.
- Register new `ptobc`-level smoke/roundtrip tests in `tools/ptobc/tests/CMakeLists.txt` when the test depends on built binaries and should run under `ctest`.
- Use real sample artifacts rather than mocks. Reuse `tools/ptobc/testdata/` or add new `.pto` fixtures there for stable bytecode-focused coverage.
- Mark intentional failure cases with `_invalid` or `_xfail` so `test/samples/runop.sh` classifies them correctly.
- Keep hardware validation generation under `test/npu_validation/`; do not hand-check in bespoke per-sample runners when the templated flow can cover the case.

---

*Testing analysis: 2026-03-18*
