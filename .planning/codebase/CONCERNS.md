# Codebase Concerns

**Analysis Date:** 2026-03-18

## Tech Debt

**Monolithic lowering and transform passes:**
- Issue: Core compiler behavior is concentrated in a few very large translation units, which makes local reasoning, review, and targeted regression testing difficult.
- Files: `lib/PTO/Transforms/PTOToEmitC.cpp`, `lib/PTO/Transforms/PTOViewToMemref.cpp`, `lib/PTO/Transforms/PTOPlanMemory.cpp`, `lib/PTO/IR/PTO.cpp`
- Impact: Small feature changes can create wide behavioral regressions across lowering, memory planning, and IR semantics. Build times and debug cycles also increase because the code is not decomposed by concern.
- Fix approach: Split passes by operation family or lowering stage, move reusable helpers into smaller headers/implementation files, and add pass-local tests before further feature work.

**String-based post-processing in code generation:**
- Issue: `ptoas` rewrites generated C++ by scanning and replacing marker calls in raw strings instead of representing member calls in IR.
- Files: `tools/ptoas/ptoas.cpp`, `lib/PTO/Transforms/PTOToEmitC.cpp`
- Impact: Output correctness depends on string shape, balanced parentheses, and marker formatting. EmitC or printer changes can silently break generated C++.
- Fix approach: Replace the text rewrite layer with a first-class lowering representation or a dedicated EmitC extension for member access and pointer operations.

**Large amount of commented-out and partially implemented code:**
- Issue: Several transform files contain disabled implementations, abandoned experiments, and TODO placeholders mixed into production code.
- Files: `lib/PTO/Transforms/BufferizableOpInterfaceImpl.cpp`, `lib/PTO/Transforms/PTOPlanMemory.cpp`, `tools/ptoas/ptoas.cpp`, `include/PTO/Transforms/Passes.h`, `include/PTO/Transforms/Passes.td`
- Impact: It is hard to tell which paths are intentionally unsupported versus temporarily disabled. Future changes are likely to reactivate stale code accidentally.
- Fix approach: Delete dead code, move deferred work into tracked issues, and keep only compilable paths that are still intended to ship.

**Checked-in generated/build artifacts:**
- Issue: Generated Python bindings and build outputs are present alongside source.
- Files: `build/python/mlir/dialects/_pto_ops_gen.py`, `build/lib/Bindings/Python/_pto_ops_gen.py`, `install/mlir/dialects/_pto_ops_gen.py`, `build/python/mlir/dialects/pto.py`, `install/mlir/dialects/pto.py`
- Impact: Source of truth becomes unclear, diffs become noisy, and stale generated files can mask whether a change actually came from codegen or manual edits.
- Fix approach: Treat generated artifacts as build outputs only, remove them from version control, and document the regeneration path in one maintained script.

## Known Bugs

**InferPTOMemScope pass is not production-ready:**
- Symptoms: The pass prints full-operation dumps to stderr, emits `"Hello PTO Infer Mem Scope!"`, and explicitly does not handle UB arguments passed to `gpu.func`.
- Files: `lib/PTO/Transforms/InferPTOMemScope.cpp`, `tools/ptoas/ptoas.cpp`
- Trigger: Enabling `createInferPTOMemScopePass()` or reintroducing it into the default pipeline.
- Workaround: Keep the pass disabled in `tools/ptoas/ptoas.cpp` until GPU-function argument handling and debug output are cleaned up.

**Known failing or flaky sample cases are masked in CI:**
- Symptoms: CI and sample harnesses encode expected failures and default skip lists for specific cases instead of enforcing a clean green suite.
- Files: `.github/workflows/ci.yml`, `test/samples/runop.sh`
- Trigger: Running full sample validation or remote-board validation with the default workflow settings.
- Workaround: Override `skip_cases`/`RUN_ONLY_CASES` during investigation and treat `mix_kernel`, `vadd_validshape`, `vadd_validshape_dynamic`, and `print` as unstable until fixed.

**PTOBC maintenance path is partly manual and partly heuristic:**
- Symptoms: PTO opcode coverage is checked via regex heuristics, while maintenance notes and test output refer to a generator path that is not present in the repository.
- Files: `tools/ptobc/tests/opcode_coverage_check.py`, `tools/ptobc/MAINTENANCE.md`, `tools/ptobc/generated/ptobc_opcodes_v0.h`
- Trigger: Adding or renaming PTO ops in `include/PTO/IR/PTOOps.td`.
- Workaround: Manually update `tools/ptobc/generated/ptobc_opcodes_v0.h` and rerun the ptobc tests after each dialect change.

## Security Considerations

**Remote validation executes ambient shell startup files and external env scripts:**
- Risk: The remote validation entrypoint sources `~/.bash_profile`, `~/.bashrc`, and Ascend toolkit setup scripts before building or running generated code.
- Files: `test/npu_validation/scripts/run_remote_npu_validation.sh`
- Current mitigation: The script logs what it sources and keeps `set -euo pipefail` around most of the workflow.
- Recommendations: Replace ambient sourcing with an explicit allowlist of required environment variables, and document a deterministic non-interactive environment contract for CI and remote hosts.

**Remote validation can pull unpinned code by default:**
- Risk: The remote workflow clones `pto-isa` and checks out `origin/HEAD` when `PTO_ISA_COMMIT` is unset.
- Files: `test/npu_validation/scripts/run_remote_npu_validation.sh`, `.github/workflows/ci.yml`
- Current mitigation: The scheduled GitHub workflow provides a pinned default commit in workflow inputs.
- Recommendations: Make `PTO_ISA_COMMIT` mandatory in the script, fail closed when it is absent, and avoid defaulting to a moving branch head.

**Installer downloads release assets without integrity verification:**
- Risk: `install.sh` fetches release metadata and tarballs from GitHub but does not verify checksums or signatures before extraction and launcher installation.
- Files: `install.sh`
- Current mitigation: Downloads use HTTPS and fail on transport errors.
- Recommendations: Publish checksums with releases, verify them in `install.sh`, and fail installation on digest mismatch.

## Performance Bottlenecks

**Repeated whole-module debug dumping in transform code:**
- Problem: Some passes print or dump entire operations/modules directly during normal execution paths.
- Files: `lib/PTO/Transforms/InferPTOMemScope.cpp`, `lib/PTO/Transforms/PTOVFloopGather.cpp`, `lib/PTO/Transforms/PTOToEmitC.cpp`
- Cause: Debug statements are not consistently guarded behind LLVM debug flags or build-time toggles.
- Improvement path: Move diagnostics behind `LLVM_DEBUG`, `DEBUG_TYPE`, or explicit verbose flags so release runs do not pay I/O and formatting costs.

**Metadata recovery is repeatedly recomputed through SSA backtracking:**
- Problem: Lowering logic walks backwards through `bind_tile`, casts, and subviews to recover tile metadata and valid dimensions.
- Files: `lib/PTO/Transforms/PTOViewToMemref.cpp`
- Cause: Important metadata is erased from types and recovered ad hoc with recursive helper functions.
- Improvement path: Cache derived metadata per value or preserve it in a more structured attribute/dataflow form before heavy lowering begins.

**Regex-based testcase generation parses generated C++ at scale:**
- Problem: Validation project generation infers kernel signatures and buffer roles by scanning generated C++ text with regexes.
- Files: `test/npu_validation/scripts/generate_testcase.py`
- Cause: The validation pipeline consumes emitted C++ text rather than structured compiler metadata.
- Improvement path: Emit machine-readable sidecar metadata from `ptoas` and let testcase generation consume that instead of reparsing source text.

## Fragile Areas

**Sync event allocation logic is assert-heavy and mutation-heavy:**
- Files: `lib/PTO/Transforms/InsertSync/SyncEventIdAllocation.cpp`, `lib/PTO/Transforms/InsertSync/InsertSyncAnalysis.cpp`, `lib/PTO/Transforms/InsertSync/PTOIRTranslator.cpp`
- Why fragile: Allocation, widening, reallocation, and downgrade behavior mutate shared state across several passes and rely on many internal invariants enforced only by `assert`.
- Safe modification: Change this area only with targeted tests that cover resource exhaustion, widening retries, and block-sync reservations together.
- Test coverage: Coverage exists for sample scenarios in `test/samples/Sync/`, but there are no focused unit tests for allocator internals.

**CLI parsing is intentionally permissive around dialect registration:**
- Files: `tools/ptoas/ptoas.cpp`, `tools/ptobc/src/main.cpp`, `tools/ptobc/src/ptobc_decode_print.cpp`
- Why fragile: Both tools enable `allowUnregisteredDialects(true)`, which reduces early validation pressure and can let malformed or partially supported IR travel deeper into the pipeline.
- Safe modification: Tighten dialect registration only with migration tests for existing `.pto` and `.ptobc` inputs; otherwise changes may break compatibility unexpectedly.
- Test coverage: Smoke and sample tests exist in `tools/ptobc/tests/` and `test/samples/`, but they do not prove strict parse compatibility boundaries.

**Shell-driven sample execution is path- and convention-dependent:**
- Files: `test/samples/runop.sh`, `docker/test_ptoas_cli.sh`, `docker/test_wheel_imports.sh`
- Why fragile: Tool discovery, expected-failure handling, architecture-specific skips, and environment assumptions are encoded in shell logic rather than in structured test definitions.
- Safe modification: Keep path resolution and skip logic centralized; if this harness changes, rerun both CI sample tests and the wheel/CLI smoke scripts.
- Test coverage: The harness itself is exercised indirectly by CI, but there is no separate validation of harness behavior.

## Scaling Limits

**Remote board validation is effectively single-threaded:**
- Current capacity: One remote validation job at a time because `.github/workflows/ci.yml` uses `concurrency.group: remote-npu-validation`.
- Limit: As the sample corpus grows, nightly validation time grows linearly and blocks other remote-board runs.
- Scaling path: Shard cases across hosts or devices, or split build-only and run-on-hardware phases more aggressively.

**Sample validation scales with the full repository corpus:**
- Current capacity: `test/samples/runop.sh` and `tools/ptobc/tests/stage9_e2e.sh` walk entire sample trees serially.
- Limit: Adding more `.py` and `.pto` samples increases runtime and output volume for every full validation run.
- Scaling path: Add manifest-based test selection, parallel execution, and per-directory ownership so developers can run only affected sample families.

## Dependencies at Risk

**LLVM/MLIR version lock is strict and expensive to change:**
- Risk: The build and documentation assume `llvmorg-19.1.7`, and much of the out-of-tree integration is coupled to that exact version.
- Impact: Upgrades will be invasive across CMake, bindings, dialect interfaces, and generated code.
- Migration plan: Isolate version-specific compatibility code, add a documented upgrade checklist, and test a second LLVM version in CI before changing the project baseline.

**pybind11 compatibility is narrower than top-level docs suggest:**
- Risk: CI and wheel builds explicitly pin `pybind11<3`, while the main README only states `pybind11`.
- Impact: Local builds can fail or diverge from CI if developers install current pybind11 releases.
- Migration plan: Document the exact supported constraint in `README.md` and `pyproject.toml`, then remove the pin only after the binding issues are fixed.

## Missing Critical Features

**No structured metadata output for downstream tooling:**
- Problem: Validation and packaging scripts infer ABI, buffer roles, and unsupported constructs from generated C++ or ad hoc file layouts.
- Blocks: Reliable external tooling, stable validation generation, and safer refactors of C++ codegen output.

**No focused unit-test layer for core passes:**
- Problem: Most verification is sample-driven end-to-end testing rather than small pass-level tests.
- Blocks: Safe refactors inside `lib/PTO/Transforms/` and faster diagnosis when a single transformation regresses.

## Test Coverage Gaps

**Pass internals are mostly untested in isolation:**
- What's not tested: Internal helper behavior in memory planning, sync event allocation, mem-scope inference, and EmitC conversion.
- Files: `lib/PTO/Transforms/PTOPlanMemory.cpp`, `lib/PTO/Transforms/InsertSync/SyncEventIdAllocation.cpp`, `lib/PTO/Transforms/InferPTOMemScope.cpp`, `lib/PTO/Transforms/PTOToEmitC.cpp`
- Risk: Regressions surface late as generated C++ differences or remote-board failures rather than as narrow failing tests.
- Priority: High

**Packaging and installation paths have only smoke coverage:**
- What's not tested: Installer integrity checks, release asset correctness beyond smoke import/CLI checks, and upgrade/downgrade scenarios.
- Files: `install.sh`, `docker/test_ptoas_cli.sh`, `docker/test_wheel_imports.sh`, `.github/workflows/build_wheel.yml`, `.github/workflows/build_wheel_mac.yml`
- Risk: Broken release artifacts or unsafe installer behavior can reach users without deeper validation.
- Priority: Medium

**Remote validation infrastructure has weak self-test coverage:**
- What's not tested: The control flow inside remote orchestration, host environment assumptions, and failure handling when dependencies or devices are unavailable.
- Files: `test/npu_validation/scripts/run_remote_npu_validation.sh`, `test/npu_validation/scripts/generate_testcase.py`
- Risk: Infrastructure failures can be mistaken for compiler regressions, and script changes can silently break nightly validation.
- Priority: Medium

---

*Concerns audit: 2026-03-18*
