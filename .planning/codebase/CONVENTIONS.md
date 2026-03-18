# Coding Conventions

**Analysis Date:** 2026-03-18

## Naming Patterns

**Files:**
- Use `PascalCase` for dialect/public C++ headers and MLIR-facing source files in `include/PTO/IR/`, `include/PTO/Transforms/`, `lib/PTO/IR/`, and `lib/PTO/Transforms/`. Examples: `include/PTO/IR/PTODialect.h`, `lib/PTO/Transforms/PTOPlanMemory.cpp`, `lib/PTO/Transforms/InsertSync/MemoryDependentAnalyzer.cpp`.
- Use lowercase tool entrypoint names for CLI code in `tools/ptoas/ptoas.cpp` and `tools/ptobc/src/main.cpp`.
- Use snake_case for Python helper modules and generated scripts in `python/pto/dialects/pto.py`, `tools/ptobc/tests/opcode_coverage_check.py`, and `test/npu_validation/scripts/generate_testcase.py`.
- Use lowercase or mixed historical sample names under `test/samples/`; preserve the directory’s existing name rather than normalizing it. Examples: `test/samples/MatMul/tmatmulk.py`, `test/samples/DataMovement/dataMovement.py`, `test/samples/Matmul_transpose/Matmul_transpose.py`.

**Functions:**
- Use `camelCase` for C++ functions and methods. Examples: `printPTOASVersion` in `tools/ptoas/ptoas.cpp`, `parsePTOFile` in `tools/ptobc/src/main.cpp`, `RecursionIR` and `UpdateLinearOperation` in `lib/PTO/Transforms/PTOPlanMemory.cpp`.
- Use leading underscores for Python internal helpers. Examples: `_load_local_pto_ext`, `_ensure_sync_attr`, `_infer_void_gm_pointee_type` in `python/pto/dialects/pto.py` and `test/npu_validation/scripts/generate_testcase.py`.
- Use shell helper verbs in lowercase with underscores for Bash functions. Examples: `resolve_ptoas_bin`, `process_one_dir`, `usage` in `test/samples/runop.sh`.

**Variables:**
- Use trailing underscores for C++ member fields. Example: `func_` in `lib/PTO/Transforms/PTOPlanMemory.cpp`.
- Use `lowerCamelCase` for local C++ variables and options. Examples: `oneShotOptions`, `outputFilename`, `target_arch`.
- Use uppercase snake case for compile-time macros and CMake options. Examples: `PTOAS_RELEASE_VERSION`, `PTO_ENABLE_PYTHON_BINDING`, `DEBUG_TYPE` in `tools/ptoas/ptoas.cpp`, `CMakeLists.txt`, and `lib/PTO/Transforms/PTOPlanMemory.cpp`.
- Use snake_case for Python locals and module constants. Examples: `lib_dir`, `INCLUDE_REPLACEMENT`, `pointer_param_names` in `python/pto/dialects/pto.py` and `test/npu_validation/scripts/generate_testcase.py`.

**Types:**
- Use `PascalCase` for dialect types, attrs, and enums exposed from C++ and Python bindings. Examples: `TensorViewType`, `AddressSpaceAttr`, `SyncOpType`, `PTOBuildLevel` in `include/PTO/IR/PTO.h`, `lib/Bindings/Python/PTOModule.cpp`, and `tools/ptoas/ptoas.cpp`.
- Use MLIR/LLVM typedef conventions rather than STL aliases when working inside dialect code. Examples: `LogicalResult`, `ParseResult`, `SmallVectorImpl<int64_t>` in `lib/PTO/IR/PTO.cpp`.

## Code Style

**Formatting:**
- No dedicated formatter config is detected at repo root: no `.clang-format`, `clang-format` config, `.editorconfig`, `.prettierrc`, `black`, or `ruff` configuration files were found.
- Follow LLVM/MLIR source formatting in C++ files:
  - File headers use LLVM banner comments. See `lib/PTO/IR/PTO.cpp` and `tools/ptoas/ptoas.cpp`.
  - Namespaces are explicitly closed with comments. See `include/PTO/IR/PTO.h` and `lib/PTO/Transforms/PTOPlanMemory.cpp`.
  - `using namespace mlir; using namespace pto;` is used in implementation files, not public headers. See `lib/PTO/IR/PTO.cpp` and `tools/ptoas/ptoas.cpp`.
- Python formatting is PEP 8 leaning but not tool-enforced:
  - Imports are often grouped in tuples and wrapped manually. See `test/samples/MatMul/tmatmulk.py`.
  - Helper-heavy modules keep short blank-line-separated functions. See `python/pto/dialects/pto.py`.
- Bash scripts use strict modes and shellcheck-friendly style. See `test/samples/runop.sh` and `tools/ptobc/tests/stage9_e2e.sh`.

**Linting:**
- No ESLint, Ruff, Flake8, MyPy, Pylint, or clang-tidy config is detected at repo root.
- The effective “lint” standard is compile/test success plus MLIR/FileCheck expectations in `test/basic/*.mlir`, CTest definitions in `tools/ptobc/tests/CMakeLists.txt`, and CI execution in `.github/workflows/ci.yml`.

## Import Organization

**Order:**
1. Project-local/public headers first in C++ implementation files. Examples: `#include "PTO/IR/PTO.h"` in `lib/PTO/IR/PTO.cpp`, `#include "ptobc/ptobc_format.h"` in `tools/ptobc/src/main.cpp`.
2. MLIR/LLVM headers next, generally grouped by subsystem. See `lib/PTO/IR/PTO.cpp` and `tools/ptoas/ptoas.cpp`.
3. Standard library headers last. Examples: `<algorithm>`, `<numeric>`, `<optional>` in `lib/PTO/IR/PTO.cpp`; `<string>` in `tools/ptoas/ptoas.cpp`.

**Path Aliases:**
- No custom import alias system is detected.
- C++ relies on include roots added by `CMakeLists.txt`, so include paths are project-relative such as `PTO/IR/PTO.h` and `ptobc/ptobc_format.h`.
- Python uses package-relative imports inside `python/pto/dialects/pto.py` (`from . import _pto_ops_gen`, `from .._mlir_libs import _pto as _pto_mod`).

## Error Handling

**Patterns:**
- Use MLIR result-based error propagation in dialect/parser code. Examples: `return failure();`, `failed(parser.parseLess())`, `LogicalResult` in `lib/PTO/IR/PTO.cpp`.
- Use assertions and `llvm_unreachable` for invariants that should never fail after verification. Examples: `assert(...)` and `llvm_unreachable("PlanMemory Traverse IR Failed! ")` in `lib/PTO/Transforms/PTOPlanMemory.cpp`.
- Use exceptions only in standalone Python or CLI utility layers. Examples: `throw std::runtime_error(...)` paths exposed via bindings in `lib/Bindings/Python/PTOModule.cpp`, and `catch (const std::exception& e)` in `tools/ptobc/src/main.cpp`.
- CLI and shell scripts return explicit exit codes with stderr messages. See `usage()` / `return 2` in `tools/ptobc/src/main.cpp` and `set -euo pipefail` plus `echo "error: ..."` in `tools/ptobc/tests/stage9_e2e.sh`.
- Sample runners prefer aggregation over fail-fast when validating many cases. `test/samples/runop.sh` deliberately uses `set -uo pipefail` without `-e`, tracks `overall=1`, and continues after individual failures.

## Logging

**Framework:** LLVM debug streams, `std::cerr`, Python exceptions/prints, and shell stdout/stderr.

**Patterns:**
- Use LLVM debug macros for pass-level diagnostics. Example: `#define DEBUG_TYPE "pto-plan-memory"` and `LDBG(X)` in `lib/PTO/Transforms/PTOPlanMemory.cpp`.
- Use `llvm::raw_ostream` for CLI user-facing output in `tools/ptoas/ptoas.cpp`.
- Use `std::cerr` for command-line errors in `tools/ptobc/src/main.cpp`.
- Use plain `print()` in Python utilities and sample generators when the script’s output is the test artifact itself. Examples: `test/samples/Sync/test_barrier_sync.py` and `tools/ptobc/tests/opcode_coverage_check.py`.

## Comments

**When to Comment:**
- Add explanatory block comments before non-obvious MLIR parsing/rewriting logic. See the member-call marker rewrite description in `tools/ptoas/ptoas.cpp` and custom asm notes in `lib/PTO/IR/PTO.cpp`.
- Keep historical or design notes near build/test plumbing where behavior is surprising. See cache/runtime notes in `.github/workflows/ci.yml` and wheel-build notes in `pyproject.toml`.
- Comment expected failure or environment-sensitive test behavior inline in runners. See `alloc_tile_addr` and `test_a5_buf_sync` handling in `test/samples/runop.sh`.

**JSDoc/TSDoc:**
- Not applicable.
- Public C++ APIs use concise line comments and LLVM banner blocks instead of Doxygen-heavy headers. See `include/PTO/IR/PTO.h`.

## Function Design

**Size:** 
- Core C++ pass and parser files contain large, stateful functions and methods. Accept this pattern when extending existing MLIR algorithms in `lib/PTO/IR/PTO.cpp` and `lib/PTO/Transforms/PTOPlanMemory.cpp`; keep new logic adjacent to the owning pass/type implementation rather than forcing unrelated abstractions.
- Python utility modules prefer many short helper functions around one orchestration flow. See `python/pto/dialects/pto.py` and `test/npu_validation/scripts/generate_testcase.py`.

**Parameters:**
- Prefer typed MLIR/LLVM signatures in C++. Examples: `parseShape(AsmParser &parser, SmallVectorImpl<int64_t> &shape)` in `lib/PTO/IR/PTO.cpp`.
- Python helpers use typed hints sparingly for public-ish utilities but not everywhere. Examples: `_idx_const(v: int)` in `test/samples/MatMul/tmatmulk.py` and `Optional[str]` in `test/npu_validation/scripts/generate_testcase.py`.
- CLI/Bash entrypoints read from env vars and command-line flags instead of global config files. See `.github/workflows/ci.yml`, `test/samples/runop.sh`, and `tools/ptobc/src/main.cpp`.

**Return Values:**
- Return MLIR success/failure objects in dialect code.
- Return explicit numeric process codes in tools and scripts.
- Return `py::none()` or raise exceptions in pybind wrappers when conversion fails. See `lib/Bindings/Python/PTOModule.cpp`.

## Module Design

**Exports:**
- Public C++ surface is declared from `include/PTO/**` and generated `.inc` includes collected through umbrella headers such as `include/PTO/IR/PTO.h`.
- Python package exports are centralized via assignment plus `__all__` in `python/pto/dialects/pto.py`.
- Command tools keep a single `main` with local helpers rather than splitting into many translation units for small CLIs. See `tools/ptobc/src/main.cpp` and `tools/ptoas/ptoas.cpp`.

**Barrel Files:**
- Use umbrella headers in C++ rather than JS-style barrel modules. `include/PTO/IR/PTO.h` is the main aggregation point for dialect enums, attrs, types, and ops.
- Python uses the generated `_pto_ops_gen` module plus the hand-written wrapper `python/pto/dialects/pto.py` as the effective barrel.

---

*Convention analysis: 2026-03-18*
