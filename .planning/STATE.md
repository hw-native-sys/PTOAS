# Project State

**Updated:** 2026-03-19
**Status:** Phase 2 in progress

## Project Reference

See: `.planning/PROJECT.md` (updated 2026-03-18)

**Core value:** Preserve PTO library semantics and template-driven behavior inside PTOAS so backend lowering retains enough information to enable optimization instead of losing it during library instantiation.
**Current focus:** Phase 2 - PTO Lowering

## Current Position

- Project initialized
- Workflow preferences captured
- Research completed
- Requirements defined
- Roadmap created
- Phase 1 plans created
- Plan `01-01` executed and summarized
- Plan `01-02` executed and summarized
- Plan `01-03` executed and summarized
- Plan `02-01` executed and summarized
- Next execution target: `02-02-PLAN.md`

## Active Milestone

**Name:** Initial A5VM backend bring-up
**Goal:** Replace the `emitc` backend slot with a PTOAS-native `a5vm` path that can compile the `Abs` sample and produce textual LLVM HIVM intrinsic IR.

## Phase Status

| Phase | Name | Status |
|-------|------|--------|
| 1 | A5VM Foundation | Complete |
| 2 | PTO Lowering | In Progress |
| 3 | HIVM Emission | Pending |
| 4 | Abs Validation | Pending |

## Requirements Snapshot

- Total v1 requirements: 16
- Complete: 10
- In Progress: 0
- Pending: 6
- Blocked: 0

## Key Decisions Snapshot

- Introduce `a5vm` as the hardware-facing backend dialect.
- Replace the current `emitc` slot rather than redesigning the pass pipeline.
- Keep v1 limited to the `Abs` sample and the minimum PTO interface set it requires.
- Emit textual LLVM HIVM intrinsic IR first, then confirm final intrinsic spellings externally.
- Use committed MLIR `RUN:`/`FileCheck` fixtures as the Phase 1 backend contract before implementation starts.
- Use a standalone Bash runner for Phase 1 verification instead of relying on external lit configuration.
- Use a handwritten A5VM vector type parser/printer to preserve the exact `!a5vm.vec<...>` syntax under the local MLIR toolchain.
- Keep `emitc` as the default backend while exposing `a5vm` through an explicit `--pto-backend` selector.
- Treat raw A5VM textual fixtures as already-lowered backend IR on the A5VM path so debug IR preserves shared dialects and A5VM ops.
- Report unresolved A5VM mappings through explicit comments, diagnostics, and optional sidecar files instead of guessing intrinsic spellings.
- Use committed Phase 2 MLIR/FileCheck fixtures as the PTO semantic-lowering contract before implementing the lowering pass.
- Use a standalone Bash runner for Phase 2 verification instead of relying on external lit configuration.

## Recent Progress

- Committed Phase 1 fixtures for `a5vm` vector types, core ops, backend switching, and shared dialect preservation
- Added executable `test/phase1/run_phase1_checks.sh` with unresolved-report and intrinsic-tracing coverage
- Created `.planning/phases/01-a5vm-foundation/01-a5vm-foundation-01-SUMMARY.md`
- Added first-class A5VM IR headers, TableGen contracts, and `lib/PTO/IR/A5VM.cpp`
- Registered A5VM in `ptoas` with parse-only textual fixture handling and passing Phase 1 A5VM checks
- Created `.planning/phases/01-a5vm-foundation/01-a5vm-foundation-02-SUMMARY.md`
- Marked BACK-01, BACK-02, and A5VM-01 through A5VM-04 complete in `.planning/REQUIREMENTS.md`
- Added the standalone `A5VMTextEmitter` with LLVM-like text output, intrinsic tracing, and unresolved-report support
- Wired `ptoas --pto-backend=a5vm` to emit textual A5VM output while keeping `emitc` as the default path
- Created `.planning/phases/01-a5vm-foundation/01-a5vm-foundation-03-SUMMARY.md`
- Added committed Phase 2 fixtures for TLOAD, TSTORE, TABS, and unary lowering metadata contracts
- Added executable `test/phase2/run_phase2_checks.sh` with explicit ptoas and `ctest` invocations
- Created `.planning/phases/02-pto-lowering/02-pto-lowering-01-SUMMARY.md`

## Open Questions

- Which exact LLVM HIVM intrinsic spellings correspond to each builtin variant exercised by the final `Abs` path
- Whether the implemented `Abs` path needs only the currently expected load/abs/store intrinsic families or additional helper intrinsics

## Session Continuity

- Next recommended command: `/gsd:execute-phase 02-pto-lowering`
- Next plan to execute: `02-02-PLAN.md`
- Current blocker status: none

---
*Last updated: 2026-03-19 after completing plan 02-01*
