# Architecture

**Analysis Date:** 2026-03-18

## Pattern Overview

**Overall:** Out-of-tree MLIR/LLVM compiler toolchain with a layered dialect library, transformation pipeline, CLI frontends, C API surface, and Python bindings.

**Key Characteristics:**
- Use TableGen definitions in `include/PTO/IR/PTOOps.td`, `include/PTO/IR/PTOTypeDefs.td`, `include/PTO/IR/PTOAttrs.td`, and `include/PTO/Transforms/Passes.td` as the source of truth for dialect IR and pass declarations.
- Keep the public interface in `include/` and the implementation in `lib/`, with tooling in `tools/`, Python packaging in `python/`, and verification assets in `test/`.
- Build all compiler-facing features around MLIR modules: parse PTO text or PTOBC bytecode into `mlir::ModuleOp`, run passes, then emit C++ or bytecode.

## Layers

**Build and composition layer:**
- Purpose: Define the project graph, discover LLVM/MLIR/Python dependencies, and assemble the subdirectories that form the toolchain.
- Location: `CMakeLists.txt`, `include/CMakeLists.txt`, `lib/CMakeLists.txt`, `tools/CMakeLists.txt`, `python/CMakeLists.txt`
- Contains: top-level CMake options, dependency discovery, install/export logic, and subdirectory inclusion.
- Depends on: external LLVM/MLIR CMake packages and optional Python/pybind11 packages.
- Used by: every library, CLI tool, install artifact, and test target.

**IR definition layer:**
- Purpose: Define the PTO dialect, custom operations, types, attributes, enums, and interfaces.
- Location: `include/PTO/IR/`, especially `include/PTO/IR/PTO.h`, `include/PTO/IR/PTODialect.h`, `include/PTO/IR/PTOOps.td`, `include/PTO/IR/PTOInterfaces.td`
- Contains: ODS/TableGen definitions plus the public C++ headers consumed by tools, passes, the C API, and Python bindings.
- Depends on: MLIR core IR, type system, and interfaces.
- Used by: `lib/PTO/IR/`, `lib/PTO/Transforms/`, `tools/ptoas/ptoas.cpp`, `tools/ptobc/src/main.cpp`, `lib/CAPI/Dialect/PTO.cpp`, `lib/Bindings/Python/PTOModule.cpp`

**IR implementation layer:**
- Purpose: Materialize the dialect declared in the IR headers and generated `.inc` files.
- Location: `lib/PTO/IR/`
- Contains: dialect/type/attribute implementation in `lib/PTO/IR/PTO.cpp`, `lib/PTO/IR/PTOAttrs.cpp`, `lib/PTO/IR/PTOTypeDefs.cpp`, and sync utilities in `lib/PTO/IR/PTOSyncUtils.cpp`
- Depends on: `include/PTO/IR/` and generated headers from `PTOOpsIncGen`.
- Used by: all transformation libraries and all user-facing interfaces.

**Transformation layer:**
- Purpose: Define optimization, analysis, lowering, memory planning, and sync insertion passes over PTO and adjacent MLIR dialects.
- Location: `include/PTO/Transforms/`, `include/PTO/Transforms/InsertSync/`, `lib/PTO/Transforms/`, `lib/PTO/Transforms/InsertSync/`
- Contains: pass declarations in `include/PTO/Transforms/Passes.h`, pass TableGen declarations in `include/PTO/Transforms/Passes.td`, and pass implementations such as `lib/PTO/Transforms/PTOPlanMemory.cpp`, `lib/PTO/Transforms/PTOToEmitC.cpp`, `lib/PTO/Transforms/InferPTOLayout.cpp`, `lib/PTO/Transforms/LoweringSyncToPipe.cpp`, and the InsertSync subsystem under `lib/PTO/Transforms/InsertSync/`
- Depends on: `PTOIR`, MLIR pass infrastructure, and standard MLIR dialects such as Func, MemRef, Arith, SCF, EmitC, and Tensor.
- Used by: `tools/ptoas/ptoas.cpp`

**Bytecode codec layer:**
- Purpose: Encode PTO MLIR into PTOBC and decode PTOBC back into PTO text or an in-memory MLIR module.
- Location: `tools/ptobc/include/ptobc/`, `tools/ptobc/src/`, `tools/ptobc/generated/`
- Contains: format definitions, LEB128 helpers, MLIR encode/decode helpers, and canonical printer support.
- Depends on: `PTOIR`, selected MLIR dialects, and LLVM support libraries.
- Used by: `tools/ptobc/src/main.cpp` and `tools/ptoas/ptoas.cpp` via `ptobc_lib`

**CLI tool layer:**
- Purpose: Provide command-line entry points for compilation and encoding/decoding workflows.
- Location: `tools/ptoas/ptoas.cpp`, `tools/ptobc/src/main.cpp`
- Contains: argument parsing, dialect registry setup, file I/O, pass pipeline assembly, and final output generation.
- Depends on: `PTOIR`, `PTOTransforms`, `ptobc_lib`, MLIR parser/EmitC/C++ translation support.
- Used by: developers, tests, sample workflows, and downstream automation.

**Interop layer:**
- Purpose: Expose PTO to non-C++ consumers through a stable C API and a Python extension that integrates with MLIR Python bindings.
- Location: `include/pto-c/Dialect/PTO.h`, `lib/CAPI/Dialect/PTO.cpp`, `lib/Bindings/Python/PTOModule.cpp`, `python/pto/dialects/pto.py`
- Contains: C wrappers for PTO types/attrs/dialect registration and a pybind11 module exposing enums, attrs, and dialect registration helpers.
- Depends on: `PTOIR`, `PTOCAPI`, MLIR C API, and pybind11.
- Used by: Python-based PTO generation in `test/samples/*.py` and external MLIR Python consumers.

**Verification and example layer:**
- Purpose: Exercise the dialect and tools through MLIR snippets, Python-generated programs, C++ outputs, and NPU validation templates.
- Location: `test/basic/`, `test/samples/`, `test/compile_cpp/`, `test/npu_validation/`, `tools/ptobc/tests/`
- Contains: focused `.mlir` pass tests, many sample PTO programs, compiler smoke tests, and remote validation scaffolding.
- Depends on: the built CLI tools and Python bindings.
- Used by: manual verification and CTest-driven smoke/e2e coverage.

## Data Flow

**PTO-to-C++ compilation flow:**

1. `tools/ptoas/ptoas.cpp` builds an `mlir::DialectRegistry`, loads PTO plus required MLIR dialects, and reads the input file.
2. `tools/ptoas/ptoas.cpp` detects the input format: textual `.pto` is parsed with `parseSourceFile`, while binary PTOBC is decoded through `ptobc::decodePTOBCToModule`.
3. `tools/ptoas/ptoas.cpp` validates CLI-level invariants such as `--pto-level` and `AllocTileOp` address expectations before transformation.
4. `tools/ptoas/ptoas.cpp` constructs a `PassManager` and runs the pipeline: `createLoweringSyncToPipePass`, optional `createInferPTOLayoutPass`, `createPTOViewToMemrefPass`, optional `createPlanMemoryPass`, optional `createPTOInsertSyncPass`, `createCSEPass`, and architecture-specific `createEmitPTOManualPass`.
5. `tools/ptoas/ptoas.cpp` runs EmitC cleanup and translation steps, then rewrites marker calls in the emitted C++ string before writing the final file.

**PTOBC encode/decode flow:**

1. `tools/ptobc/src/main.cpp` accepts `encode` or `decode`.
2. For `encode`, `tools/ptobc/src/main.cpp` registers PTO and core MLIR dialects, parses PTO text into a module, and passes it to `ptobc::encodeFromMLIRModule`.
3. `tools/ptobc/src/main.cpp` serializes the resulting `PTOBCFile` and writes the raw bytes with `ptobc::writeFile`.
4. For `decode`, `tools/ptobc/src/main.cpp` delegates directly to `ptobc::decodeFileToPTO`, producing textual PTO output.

**Python authoring flow:**

1. `lib/Bindings/Python/PTOModule.cpp` exposes `_pto`, including `register_dialect` and PTO enums/attr helpers.
2. `lib/Bindings/Python/CMakeLists.txt` copies `python/pto/dialects/pto.py` and generated `_pto_ops_gen.py` into the MLIR-style Python package layout.
3. Sample programs under `test/samples/` import `mlir.dialects.pto`, build PTO IR in Python, and emit `.pto` files that feed the CLI tools.

**State Management:**
- State lives primarily inside `mlir::MLIRContext`, `mlir::ModuleOp`, and pass-local analyses. There is no application-level runtime state outside the current process; tools read input, build IR, transform it, and emit artifacts.

## Key Abstractions

**PTO dialect and ODS definitions:**
- Purpose: Define the PTO language contract once and generate the repetitive C++/Python support from it.
- Examples: `include/PTO/IR/PTOOps.td`, `include/PTO/IR/PTOTypeDefs.td`, `include/PTO/IR/PTOAttrs.td`, `include/PTO/IR/PTOInterfaces.td`
- Pattern: TableGen-first MLIR dialect design.

**`PTOIR` library:**
- Purpose: Provide the concrete implementation of dialect registration, parsing, printing, type logic, and attr logic.
- Examples: `lib/PTO/IR/PTO.cpp`, `lib/PTO/IR/PTOAttrs.cpp`, `lib/PTO/IR/PTOTypeDefs.cpp`
- Pattern: Generated declarations in `include/` paired with handwritten C++ implementations in `lib/`.

**Pass factory API:**
- Purpose: Keep pass creation behind stable factory functions and generated pass registration rather than exposing pass classes directly.
- Examples: `include/PTO/Transforms/Passes.h`, `lib/PTO/Transforms/PTOPlanMemory.cpp`, `lib/PTO/Transforms/PTOToEmitC.cpp`
- Pattern: `create...Pass()` factories declared in a shared header and linked through `PTOTransforms`.

**InsertSync subsystem:**
- Purpose: Break a complex synchronization feature into multiple analysis and rewrite components instead of one monolithic pass file.
- Examples: `include/PTO/Transforms/InsertSync/InsertSyncAnalysis.h`, `include/PTO/Transforms/InsertSync/MemoryDependentAnalyzer.h`, `lib/PTO/Transforms/InsertSync/PTOInsertSync.cpp`, `lib/PTO/Transforms/InsertSync/SyncCodegen.cpp`
- Pattern: feature-local submodule under `Transforms/InsertSync/` with several collaborating helpers.

**`ptobc_lib`:**
- Purpose: Centralize PTOBC format handling so both the `ptobc` CLI and `ptoas` can reuse the same codec logic.
- Examples: `tools/ptobc/src/mlir_encode.cpp`, `tools/ptobc/src/ptobc_format.cpp`, `tools/ptobc/src/ptobc_decode_print.cpp`
- Pattern: static support library plus a thin CLI wrapper.

**C API wrappers:**
- Purpose: Convert between MLIR C API handles and PTO C++ types/attrs without exposing C++ templates to downstream consumers.
- Examples: `include/pto-c/Dialect/PTO.h`, `lib/CAPI/Dialect/PTO.cpp`
- Pattern: `wrap`/`unwrap` bridge over `PTOIR`.

## Entry Points

**Primary compiler CLI:**
- Location: `tools/ptoas/ptoas.cpp`
- Triggers: direct execution of the built `ptoas` binary.
- Responsibilities: register dialects, parse PTO or PTOBC input, enforce level/arch options, run passes, translate to C++, and post-process output.

**PTOBC CLI:**
- Location: `tools/ptobc/src/main.cpp`
- Triggers: direct execution of the built `ptobc` binary with `encode` or `decode`.
- Responsibilities: register a minimal dialect set, parse PTO, serialize to PTOBC, or decode PTOBC back to PTO text.

**Python extension module:**
- Location: `lib/Bindings/Python/PTOModule.cpp`
- Triggers: importing `mlir._mlir_libs._pto` through the installed Python package layout.
- Responsibilities: register the PTO dialect with a context and expose PTO enums, attrs, and type helpers into Python.

**Build system entry point:**
- Location: `CMakeLists.txt`
- Triggers: `cmake -S . -B build`
- Responsibilities: locate LLVM/MLIR, enable Python bindings when configured, add subdirectories, export install targets, and wire test targets.

## Error Handling

**Strategy:** Fail fast at tool boundaries using MLIR/LLVM error results, explicit CLI validation, and exception-based guards in the PTOBC CLI.

**Patterns:**
- `tools/ptoas/ptoas.cpp` checks file-open, parse, pass-run, arch/level validation, and C++ emission failures explicitly and returns non-zero on error.
- `tools/ptobc/src/main.cpp` wraps encode/decode operations in `try/catch` and reports exceptions to `stderr`.

## Cross-Cutting Concerns

**Logging:** Use LLVM/MLIR diagnostics and `llvm::errs()` in CLI tools. Do not add an application-specific logging layer; compiler diagnostics belong in passes or tool frontends.

**Validation:** Put structural IR validation in dialect/op definitions under `include/PTO/IR/` and `lib/PTO/IR/`. Put pipeline-specific or mode-specific validation in CLI code such as `tools/ptoas/ptoas.cpp`.

**Authentication:** Not applicable. This repository is a local compiler toolchain and test harness without a service-side auth boundary.

---

*Architecture analysis: 2026-03-18*
