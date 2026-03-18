---
phase: 01-a5vm-foundation
plan: 01
subsystem: testing
tags: [mlir, filecheck, ptoas, a5vm, bash]
requires: []
provides:
  - Phase 1 source fixtures for the a5vm vector type, load, abs, and store behaviors
  - Backend-switch and shared-dialect executable expectations for the future a5vm path
  - An executable Bash runner that documents the Phase 1 verification commands
affects: [01-02-PLAN.md, 01-03-PLAN.md, phase-1-validation]
tech-stack:
  added: []
  patterns: [mlir RUN/FileCheck fixtures, standalone bash verification runner]
key-files:
  created:
    - test/phase1/a5vm_vec_type.mlir
    - test/phase1/a5vm_load_op.mlir
    - test/phase1/a5vm_abs_op.mlir
    - test/phase1/a5vm_store_op.mlir
    - test/phase1/a5vm_backend_switch.mlir
    - test/phase1/a5vm_shared_dialects.mlir
    - test/phase1/run_phase1_checks.sh
  modified: []
key-decisions:
  - "Use committed MLIR RUN/FileCheck fixtures as the Phase 1 contract before backend implementation starts."
  - "Ship a standalone Bash runner instead of relying on an external lit configuration for Wave 0 verification."
patterns-established:
  - "Source-first backend planning: write fixture coverage before implementing the a5vm dialect and backend path."
  - "Phase verification helpers should fail fast on missing tool binaries and print the active case before each compiler call."
requirements-completed: [BACK-01, BACK-02, A5VM-01, A5VM-02, A5VM-03, A5VM-04]
duration: 18min
completed: 2026-03-18
---

# Phase 01 Plan 01: Phase 1 Fixture Contracts Summary

**Phase 1 A5VM MLIR/FileCheck fixtures and a documented runner covering type, op, backend-switch, and shared-dialect expectations**

## Performance

- **Duration:** 18 min
- **Started:** 2026-03-18T16:43:00Z
- **Completed:** 2026-03-18T17:00:58Z
- **Tasks:** 2
- **Files modified:** 7

## Accomplishments
- Added six committed Phase 1 fixtures that encode the planned a5vm vector-type, load, abs, store, backend-switch, and shared-dialect expectations.
- Added an executable `test/phase1/run_phase1_checks.sh` helper that documents the exact verification order and debug-sidecar checks for the wave.
- Converted the previous Wave 0 validation gap into concrete repository artifacts that later implementation plans can target directly.

## Task Commits

Each task was committed atomically:

1. **Task 1: Create Phase 1 MLIR/FileCheck fixtures** - `6b10d65` (feat)
2. **Task 2: Add a documented runner for all Phase 1 checks** - `9fc9307` (feat)

**Plan metadata:** pending docs commit

## Files Created/Modified
- `test/phase1/a5vm_vec_type.mlir` - Covers legal and illegal 256-byte a5vm vector type cases.
- `test/phase1/a5vm_load_op.mlir` - Pins `a5vm.load` syntax and metadata expectations.
- `test/phase1/a5vm_abs_op.mlir` - Pins `a5vm.abs` syntax and mismatched-type diagnostics.
- `test/phase1/a5vm_store_op.mlir` - Pins `a5vm.store` syntax and metadata expectations.
- `test/phase1/a5vm_backend_switch.mlir` - Defines the future `--pto-backend=a5vm` text-emission contract.
- `test/phase1/a5vm_shared_dialects.mlir` - Ensures shared `arith` and `scf` ops remain visible alongside `a5vm.abs`.
- `test/phase1/run_phase1_checks.sh` - Documents and automates the full Phase 1 static verification path.

## Decisions Made
- Used committed source fixtures as the contract for BACK-01, BACK-02, and A5VM-01 through A5VM-04 so later implementation work has executable acceptance targets.
- Used a standalone Bash runner rather than introducing a committed lit config in this plan, which matches the validation note that current lit wiring is external or build-generated.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- Git index writes were blocked by the sandbox, so the task commits were executed with elevated git permissions only for `git add` and `git commit`.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Phase 1 implementation plans can now target committed fixture files and a documented runner instead of placeholder validation notes.
- The runner is intentionally static at this stage; full execution depends on the a5vm backend work from later Phase 1 plans.

## Self-Check: PASSED

- Found `.planning/phases/01-a5vm-foundation/01-a5vm-foundation-01-SUMMARY.md`
- Found commit `6b10d65`
- Found commit `9fc9307`

---
*Phase: 01-a5vm-foundation*
*Completed: 2026-03-18*
