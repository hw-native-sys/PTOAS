---
phase: 02-pto-lowering
plan: 06
subsystem: testing
tags: [mlir, filecheck, validation, runner, a5vm]
requires:
  - phase: 02-pto-lowering
    provides: "Phase 2 fixtures and runner preflight guards from plans 02-01 through 02-05"
provides:
  - "Deterministic FileCheck discovery in the Phase 2 verification runner"
  - "Validation docs aligned to the committed runner entrypoint"
  - "Deferred blocker log cleaned of the resolved tabs_abs_loop_shape preflight mismatch"
affects: [phase-02-validation, phase-03-hivm-emission, ptoas-runner]
tech-stack:
  added: []
  patterns: ["workspace-safe tool discovery in shell runners", "runner-first validation commands in planning docs"]
key-files:
  created: [.planning/phases/02-pto-lowering/02-pto-lowering-06-SUMMARY.md]
  modified: [test/phase2/run_phase2_checks.sh, .planning/phases/02-pto-lowering/02-VALIDATION.md, .planning/phases/02-pto-lowering/deferred-items.md]
key-decisions:
  - "Resolve FileCheck in the runner through a deterministic workspace-specific search order before any fixture execution."
  - "Make the committed Phase 2 runner the primary documented quick and full validation entrypoint."
patterns-established:
  - "Shell verification entrypoints should surface missing external tools through explicit checked-path errors instead of implicit PATH failures."
  - "Validation docs should name the committed runner when that runner already owns the suite contract."
requirements-completed: [PTO-01, PTO-02, PTO-03, PTO-04]
duration: 12min
completed: 2026-03-19
---

# Phase 2 Plan 06: Summary

**Phase 2 verification now resolves FileCheck deterministically, documents the runner as the canonical entrypoint, and removes the obsolete tabs preflight blocker note**

## Performance

- **Duration:** 12 min
- **Started:** 2026-03-19T03:58:30Z
- **Completed:** 2026-03-19T04:10:10Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments

- Added a deterministic `FileCheck` search order to `test/phase2/run_phase2_checks.sh` and routed every fixture through the resolved binary.
- Updated Phase 2 validation guidance to use `bash test/phase2/run_phase2_checks.sh` as both the quick-run and full-suite command.
- Removed the stale `tabs_abs_loop_shape.mlir` runner mismatch from deferred blockers while preserving the unrelated generated-header and build-regeneration issues.

## Task Commits

Each task was committed atomically:

1. **Task 1: Add a workspace-safe FileCheck discovery path to the Phase 2 runner** - `9d7c455` (fix)
2. **Task 2: Align validation docs and deferred notes with the runnable runner contract** - `5f34845` (docs)

**Plan metadata:** pending

## Files Created/Modified

- `test/phase2/run_phase2_checks.sh` - Resolves `FileCheck` through `FileCheck`, `FileCheck-19`, or `/usr/lib/llvm-19/bin/FileCheck` before executing the Phase 2 fixture suite.
- `.planning/phases/02-pto-lowering/02-VALIDATION.md` - Names the committed runner as the validation entrypoint and documents the same FileCheck discovery contract used by the script.
- `.planning/phases/02-pto-lowering/deferred-items.md` - Drops the resolved stale runner/fixture mismatch while retaining still-real build-tree blockers.

## Decisions Made

- Resolve `FileCheck` before checking runtime fixtures so missing-tool failures are deterministic and actionable.
- Keep the broader generated-header and stale-build metadata blockers in deferred items because this plan did not fix the build graph itself.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- `bash test/phase2/run_phase2_checks.sh` cannot complete in this workspace because no `FileCheck` candidate is installed at `FileCheck`, `FileCheck-19`, or `/usr/lib/llvm-19/bin/FileCheck`.
- The current workspace also lacks an executable `./build/tools/ptoas/ptoas`, so end-to-end Phase 2 suite execution still depends on an available build tree after the external tooling prerequisite is satisfied.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- The Phase 2 runner contract is now explicit and documented, so future verification runs will use the same `FileCheck` resolution strategy as the committed entrypoint.
- Full end-to-end verification still requires one of the documented `FileCheck` binaries and a runnable `./build/tools/ptoas/ptoas` build tree.

## Self-Check: PASSED

- Found `.planning/phases/02-pto-lowering/02-pto-lowering-06-SUMMARY.md`.
- Verified task commits `9d7c455` and `5f34845` exist in git history.
