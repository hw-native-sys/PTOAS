# Project State

**Updated:** 2026-03-19
**Status:** Phase 1 execution in progress

## Project Reference

See: `.planning/PROJECT.md` (updated 2026-03-18)

**Core value:** Preserve PTO library semantics and template-driven behavior inside PTOAS so backend lowering retains enough information to enable optimization instead of losing it during library instantiation.
**Current focus:** Phase 1 - A5VM Foundation

## Current Position

- Project initialized
- Workflow preferences captured
- Research completed
- Requirements defined
- Roadmap created
- Phase 1 plans created
- Plan `01-01` executed and summarized
- Next execution target: `01-02-PLAN.md`

## Active Milestone

**Name:** Initial A5VM backend bring-up
**Goal:** Replace the `emitc` backend slot with a PTOAS-native `a5vm` path that can compile the `Abs` sample and produce textual LLVM HIVM intrinsic IR.

## Phase Status

| Phase | Name | Status |
|-------|------|--------|
| 1 | A5VM Foundation | In Progress |
| 2 | PTO Lowering | Pending |
| 3 | HIVM Emission | Pending |
| 4 | Abs Validation | Pending |

## Requirements Snapshot

- Total v1 requirements: 16
- Complete: 6
- In Progress: 0
- Pending: 10
- Blocked: 0

## Key Decisions Snapshot

- Introduce `a5vm` as the hardware-facing backend dialect.
- Replace the current `emitc` slot rather than redesigning the pass pipeline.
- Keep v1 limited to the `Abs` sample and the minimum PTO interface set it requires.
- Emit textual LLVM HIVM intrinsic IR first, then confirm final intrinsic spellings externally.
- Use committed MLIR `RUN:`/`FileCheck` fixtures as the Phase 1 backend contract before implementation starts.
- Use a standalone Bash runner for Phase 1 verification instead of relying on external lit configuration.

## Recent Progress

- Committed Phase 1 fixtures for `a5vm` vector types, core ops, backend switching, and shared dialect preservation
- Added executable `test/phase1/run_phase1_checks.sh` with unresolved-report and intrinsic-tracing coverage
- Created `.planning/phases/01-a5vm-foundation/01-a5vm-foundation-01-SUMMARY.md`
- Marked BACK-01, BACK-02, and A5VM-01 through A5VM-04 complete in `.planning/REQUIREMENTS.md`

## Open Questions

- Which exact LLVM HIVM intrinsic spellings correspond to each builtin variant exercised by the final `Abs` path
- Whether the implemented `Abs` path needs only the currently expected load/abs/store intrinsic families or additional helper intrinsics

## Session Continuity

- Next recommended command: `/gsd:execute-phase 01-a5vm-foundation`
- Next plan to execute: `01-02-PLAN.md`
- Current blocker status: none

---
*Last updated: 2026-03-19 after completing plan 01-01*
