---
phase: 02-pto-lowering
plan: 01
subsystem: testing
tags: [mlir, filecheck, a5vm, ptoas, bash]
requires:
  - phase: 01-a5vm-foundation
    provides: Phase 1 A5VM dialect, backend selector, and print-ir path used by the new fixture contracts
provides:
  - Committed Phase 2 MLIR/FileCheck fixtures for TLOAD, TSTORE, TABS, and unary lowering shape
  - Executable Phase 2 runner with explicit ptoas and ctest invocation order
affects: [02-02-PLAN.md, 02-03-PLAN.md, testing]
tech-stack:
  added: []
  patterns: [Committed lowering contract fixtures, direct Bash verification runner]
key-files:
  created:
    - test/phase2/tload_contract_trace.mlir
    - test/phase2/tstore_branch_shape.mlir
    - test/phase2/tabs_precheck.mlir
    - test/phase2/unary_template_shape.mlir
    - test/phase2/run_phase2_checks.sh
  modified: []
key-decisions:
  - "Capture Phase 2 PTO lowering expectations as committed MLIR/FileCheck contracts before lowering code exists."
  - "Use a direct Bash runner with explicit ptoas invocations instead of lit discovery for this wave."
patterns-established:
  - "Phase contracts live as source fixtures that lock future metadata keys and exact diagnostic text."
  - "Phase verification scripts print each fixture name before invoking the corresponding compiler command."
requirements-completed: [PTO-01, PTO-02, PTO-03, PTO-04]
duration: 9min
completed: 2026-03-19
---

# Phase 2 Plan 1: PTO Lowering Summary

**Phase 2 semantic-lowering contracts for TLOAD, TSTORE, TABS, and unary abs shape with an executable verification runner**

## Performance

- **Duration:** 9 min
- **Started:** 2026-03-18T17:49:00Z
- **Completed:** 2026-03-18T17:58:33Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments

- Added four committed `test/phase2/*.mlir` fixtures that lock the exact metadata keys, diagnostic strings, and unsupported-branch behavior required for PTO Phase 2.
- Added `test/phase2/run_phase2_checks.sh` as the direct verification entrypoint for the new fixtures plus `ctest`.
- Replaced the Wave 0 placeholders in Phase 2 with concrete repository artifacts that future lowering work can target.

## Task Commits

Each task was committed atomically:

1. **Task 1: Create Phase 2 MLIR/FileCheck fixtures** - `193d906` (feat)
2. **Task 2: Add a documented runner for all Phase 2 checks** - `26cc8ae` (chore)

## Files Created/Modified

- `test/phase2/tload_contract_trace.mlir` - TLOAD contract fixture for layout, shape, trace, and padding/init metadata.
- `test/phase2/tstore_branch_shape.mlir` - TSTORE vec-path contract plus explicit ACC and MAT TODO diagnostics.
- `test/phase2/tabs_precheck.mlir` - TABS precheck contract for vec-domain, row-major, valid-shape, and dtype restrictions.
- `test/phase2/unary_template_shape.mlir` - Unary abs lowering contract that distinguishes A5VM lowering from legacy EmitC lowering.
- `test/phase2/run_phase2_checks.sh` - Executable runner spelling out the exact Phase 2 verification commands.

## Decisions Made

- Used committed source fixtures as the primary contract for PTO Phase 2 metadata and diagnostics so later lowering work has explicit textual targets.
- Kept verification as a direct shell script with ordered compiler invocations and a missing-binary guard to match the phase validation strategy.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- `CLAUDE.md` was not present at the repository root, so execution proceeded with the plan, state, and skill instructions already in the workspace.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Phase 2 now has committed contract fixtures and a runnable entrypoint for semantic-lowering work.
- Plans `02-02` and `02-03` can implement the lowering helpers and backend wiring against these exact checks.

## Self-Check: PASSED

---
*Phase: 02-pto-lowering*
*Completed: 2026-03-19*
