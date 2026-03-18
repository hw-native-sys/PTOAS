---
phase: 01-a5vm-foundation
plan: 02
subsystem: ir
tags: [mlir, a5vm, tablegen, ptoas, dialect]
requires:
  - phase: 01-a5vm-foundation
    provides: corrected Phase 1 fixture contracts for copy and vabs primitives
provides:
  - corrected `mlir::a5vm` dialect namespace and primitive op contracts
  - handwritten `!a5vm.vec<...>` parsing and verification for copy and register ops
  - `ptoas` registration and build-order wiring for generated A5VM headers
affects: [01-a5vm-foundation, 02-pto-lowering, a5vm-backend]
tech-stack:
  added: []
  patterns: [TableGen dialect contracts with handwritten MLIR verifiers, generated-header dependency wiring]
key-files:
  created: [.planning/phases/01-a5vm-foundation/01-a5vm-foundation-02-SUMMARY.md]
  modified: [include/PTO/IR/A5VMDialect.td, include/PTO/IR/A5VMOps.td, lib/PTO/IR/A5VM.cpp, tools/ptoas/ptoas.cpp, tools/ptoas/CMakeLists.txt, lib/PTO/Transforms/PTOToA5VM.cpp, lib/PTO/Transforms/A5VMTextEmitter.cpp, include/PTO/Transforms/Passes.td]
key-decisions:
  - "Keep copy-op transfer attrs optional in parsing but mandatory in handwritten verification so invalid fixtures fail with the planned diagnostic instead of a parser error."
  - "Derive copy-op burst and stride metadata from the existing lowering contract instead of widening the public Phase 2 lowering structs in this plan."
  - "Add `A5VMOpsIncGen` as a direct `ptoas` build dependency because the CLI includes generated A5VM headers before linking against PTOIR."
patterns-established:
  - "A5VM stays in `mlir::a5vm` while preserving `a5vm` assembly spelling."
  - "Primitive A5VM copy/register ops use verifier-enforced metadata instead of PTO-shaped pseudo-ops."
requirements-completed: [A5VM-01, A5VM-02, A5VM-03, A5VM-04]
duration: 25min
completed: 2026-03-19
---

# Phase 01 Plan 02: A5VM Foundation Summary

**Corrected `mlir::a5vm` primitive IR with 256-byte vector typing, GM/UB copy ops, and `vlds`/`vabs`/`vsts` verification through `ptoas`**

## Performance

- **Duration:** 25 min
- **Started:** 2026-03-18T20:09:00Z
- **Completed:** 2026-03-18T20:34:12Z
- **Tasks:** 2
- **Files modified:** 8

## Accomplishments
- Replaced the obsolete `a5vm.load` / `a5vm.abs` / `a5vm.store` TableGen surface with corrected copy and register primitives under `::mlir::a5vm`.
- Rewrote `lib/PTO/IR/A5VM.cpp` to preserve `!a5vm.vec<...>` syntax, enforce the exact 256-byte rule, and verify copy plus register op contracts.
- Rewired `ptoas`, pass metadata, lowering helpers, and the A5VM text emitter to compile and recognize the corrected namespace and op classes.

## Task Commits

Each task was committed atomically:

1. **Task 1: Redefine the A5VM dialect contracts around corrected primitives** - `197386f` (feat)
2. **Task 2: Implement corrected A5VM parsing, printing, and verification** - `4194d7f` (feat)

## Files Created/Modified
- `include/PTO/IR/A5VMDialect.td` - switches the dialect C++ namespace to `::mlir::a5vm`
- `include/PTO/IR/A5VMOps.td` - defines copy and register primitive ops with verifier-owned metadata checks
- `lib/PTO/IR/A5VM.cpp` - implements vector type parse/print, verifier logic, and memory-effect hooks
- `tools/ptoas/ptoas.cpp` - registers and loads `mlir::a5vm::A5VMDialect`
- `tools/ptoas/CMakeLists.txt` - ensures A5VM generated headers are built before compiling `ptoas`
- `lib/PTO/Transforms/PTOToA5VM.cpp` - updates lowering helpers to create corrected A5VM primitive ops
- `lib/PTO/Transforms/A5VMTextEmitter.cpp` - updates emission logic to the corrected op classes and namespace
- `include/PTO/Transforms/Passes.td` - updates pass dependent dialect metadata to `a5vm::A5VMDialect`

## Decisions Made

- Kept transfer attrs parser-optional and verifier-required so invalid copy fixtures report the plan-locked op diagnostic instead of failing earlier in assembly parsing.
- Derived copy transfer metadata from the existing lowering contract fields to avoid an architectural change to the public lowering structs during this plan.
- Fixed the `ptoas` build dependency on `A5VMOpsIncGen` because the corrected CLI now directly includes A5VM generated headers.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Added direct generated-header dependency for `ptoas`**
- **Found during:** Task 2
- **Issue:** `ptoas.cpp` includes `A5VMDialect.h.inc`, but the executable only depended on `PTOOpsIncGen`, so the generated A5VM headers were missing at compile time.
- **Fix:** Added `A5VMOpsIncGen` to `tools/ptoas/CMakeLists.txt` and rebuilt through the configured Makefile generator.
- **Files modified:** `tools/ptoas/CMakeLists.txt`
- **Verification:** `make -C build pto-opt -j1` completed and the rebuilt binary passed all four Phase 1 fixture checks.
- **Committed in:** `4194d7f`

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Required for correctness of the corrected dialect wiring. No scope creep.

## Issues Encountered

- The existing build directory is configured for `Unix Makefiles` while a stale `build.ninja` was also present, so verification had to use `make -C build` rather than `ninja -C build`.
- `FileCheck` was not on `PATH`; verification used `/data/mouliangyu/projects/github.com/llvm/llvm-project/build/bin/FileCheck`.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Phase 2 lowering can now target first-class `mlir::a5vm` copy and register primitives without carrying the obsolete pseudo-op model.
- The rebuilt `ptoas` binary parses and verifies the corrected Phase 1 fixtures, so downstream lowering and emission work can rely on the new dialect boundary.

## Self-Check: PASSED

- FOUND: `.planning/phases/01-a5vm-foundation/01-a5vm-foundation-02-SUMMARY.md`
- FOUND: `197386f`
- FOUND: `4194d7f`

---
*Phase: 01-a5vm-foundation*
*Completed: 2026-03-19*
