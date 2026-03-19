---
phase: 02-pto-lowering
plan: 05
subsystem: api
tags: [mlir, a5vm, pto, scf, diagnostics]
requires:
  - phase: 02-pto-lowering
    provides: vec-scope contracts, A5VM unary/copy lowering scaffolding
provides:
  - explicit dummy-loop AIV vec-scope carrier in lowered TABS IR
  - full-fixture diagnostic collection for unsupported TABS and TSTORE cases
affects: [phase-02, a5vm-lowering, verification]
tech-stack:
  added: []
  patterns: [explicit loop-carrier lowering, full-module diagnostic collection]
key-files:
  created: []
  modified:
    - lib/PTO/Transforms/PTOToA5VMLowering.cpp
    - lib/PTO/Transforms/PTOToA5VM.cpp
    - test/phase2/tabs_abs_loop_shape.mlir
    - test/phase2/tabs_precheck_a5.mlir
    - test/phase2/tstore_domain_todos.mlir
key-decisions:
  - "Keep `__VEC_SCOPE__` observable as a one-trip `scf.for` carrier with its attr-dict printed on the carrier loop, while the inner loop chunks linearly by vector width."
  - "Replace fail-fast partial conversion with an explicit PTO-op walk so dedicated lowering diagnostics accumulate before the pass signals failure."
patterns-established:
  - "Loop-carrier contract: lock both the carrier `scf.for` shape and its closing attr-dict when MLIR custom printing moves attrs off the header line."
  - "Diagnostic contract: emit dedicated per-op errors during lowering, continue visiting remaining PTO ops, and only fail the pass after the walk completes."
requirements-completed: [PTO-02, PTO-03, PTO-04]
duration: 12min
completed: 2026-03-19
---

# Phase 02 Plan 05: Vec-Scope Carrier And Diagnostic Coverage Summary

**Explicit `__VEC_SCOPE__` dummy-loop carrier in lowered TABS IR plus full unsupported-case diagnostic coverage for Phase 2 fixtures**

## Performance

- **Duration:** 12 min
- **Started:** 2026-03-19T03:51:29Z
- **Completed:** 2026-03-19T04:03:29Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments
- Reworked TABS lowering so the observable IR now uses a one-trip outer `scf.for` as the AIV vec-scope carrier and a separate inner chunk loop over vector-width slices.
- Updated the loop-shape fixture to lock the actual printed carrier contract, including `cce_aiv_loop_hint`, `llvm.loop.aivector_scope`, and the ordered `a5vm.vlds -> a5vm.vabs -> a5vm.vsts` sequence.
- Replaced fail-fast PTO-to-A5VM conversion with a full-module lowering walk so all locked TABS precheck and TSTORE TODO diagnostics surface in one run before the pass fails overall.

## Task Commits

Each task was committed atomically:

1. **Task 1: Preserve the explicit dummy `__VEC_SCOPE__` carrier in lowered TABS IR** - `ffc8a8e` (feat)
2. **Task 2: Restore full locked diagnostics for TABS prechecks and TSTORE TODO branches** - `6a604d2` (fix)

## Files Created/Modified
- `lib/PTO/Transforms/PTOToA5VMLowering.cpp` - Simplified unary vec-scope lowering to an outer dummy carrier loop plus one inner chunk loop.
- `lib/PTO/Transforms/PTOToA5VM.cpp` - Swapped partial conversion for an explicit PTO-op walk that collects diagnostics across the whole module.
- `test/phase2/tabs_abs_loop_shape.mlir` - Locked the printed carrier-loop attr-dict and ordered vector primitive sequence.
- `test/phase2/tabs_precheck_a5.mlir` - Locked the four dedicated TABS diagnostics while forbidding the old legalization-failure contract.
- `test/phase2/tstore_domain_todos.mlir` - Locked both ACC and MAT TODO diagnostics in one run under the planned `--a5vm-print-ir` invocation.

## Decisions Made
- Kept the vec-scope semantics attached to a printed loop carrier instead of moving them onto only the vector ops, because the user-confirmed semantics require a loop-shaped owner.
- Matched the fixture to MLIR's actual `scf.for` printing behavior by checking the carrier loop header and closing attr-dict separately.
- Used a manual lowering walk instead of dialect conversion so dedicated diagnostics remain observable across multiple PTO ops in the same module.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- The workspace did not contain the plan's `./build/tools/ptoas/ptoas` binary, so verification used a throwaway build at `/tmp/ptoas-02-05-build`.
- `FileCheck` was not on `PATH`; verification used `/data/mouliangyu/projects/github.com/llvm/llvm-project/build/bin/FileCheck`.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- The vec-scope carrier and multi-diagnostic contracts are now locked in committed fixtures, so later Phase 2 work can build on a stable unary-lowering and unsupported-case baseline.
- Remaining Phase 2 verification gaps are outside this plan's scope, primarily the exact TLOAD/TSTORE metadata preservation and the committed runner path assumptions.

## Self-Check

PASSED

---
*Phase: 02-pto-lowering*
*Completed: 2026-03-19*
