---
phase: 02-pto-lowering
plan: 02
subsystem: api
tags: [mlir, a5vm, pto, lowering, cmake]
requires:
  - phase: 02-pto-lowering
    provides: Phase 2 fixture contracts and runner that lock the lowering metadata and diagnostics implemented here
provides:
  - Public PTO-to-A5VM lowering contract structs and entrypoint declarations
  - Shared PTOToA5VM helper implementation for contract extraction and placeholder A5VM op emission
  - Explicit TABS prechecks and visible TSTORE ACC/MAT TODO branches for backend follow-up
affects: [02-03-PLAN.md, ptoas, a5vm]
tech-stack:
  added: []
  patterns: [Explicit lowering contracts, helper-based PTO semantic extraction, placeholder backend branch diagnostics]
key-files:
  created:
    - include/PTO/Transforms/A5VMLowering.h
    - lib/PTO/Transforms/PTOToA5VM.cpp
  modified:
    - lib/PTO/Transforms/CMakeLists.txt
key-decisions:
  - "Keep the lowering surface split into public contracts plus a helper implementation file before pass wiring."
  - "Use explicit metadata attachment helpers so fixture-locked attribute names stay readable and reusable."
  - "Preserve unsupported TSTORE ACC and MAT paths as dedicated TODO diagnostics instead of collapsing them into a generic failure."
patterns-established:
  - "PTO semantic extraction happens in named helpers per op family before A5VM ops are constructed."
  - "Unary lowering reuses a shared helper that attaches family, domain, layout, and valid-shape attributes."
requirements-completed: [PTO-01, PTO-02, PTO-03, PTO-04]
duration: 7min
completed: 2026-03-19
---

# Phase 2 Plan 2: PTO Lowering Summary

**Reusable PTO-to-A5VM lowering contracts with extracted TLOAD/TABS/TSTORE metadata, unary abs scaffolding, and explicit unsupported store-domain diagnostics**

## Performance

- **Duration:** 7 min
- **Started:** 2026-03-18T18:04:00Z
- **Completed:** 2026-03-18T18:11:00Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments

- Added `include/PTO/Transforms/A5VMLowering.h` with the exact shared Phase 2 contracts and lowering entrypoints the plan required.
- Implemented `lib/PTO/Transforms/PTOToA5VM.cpp` with named extraction helpers, contract attribute attachment, unary abs scaffolding, and locked TABS/TSTORE diagnostics.
- Wired `PTOToA5VM.cpp` into `PTOTransforms` and verified the target builds successfully with `CCACHE_DISABLE=1`.

## Task Commits

Each task was committed atomically:

1. **Task 1: Define public PTO-to-A5VM lowering contracts and entrypoints** - `4ecc41b` (feat)
2. **Task 2: Implement shared contract extraction helpers and placeholder branch behavior** - `dbd994d` (feat)

## Files Created/Modified

- `include/PTO/Transforms/A5VMLowering.h` - Public contract layer for Phase 2 load, unary, and store metadata plus lowering entrypoints.
- `lib/PTO/Transforms/PTOToA5VM.cpp` - Shared extraction helpers, contract attribute attachment, unary placeholder lowering, and explicit TODO branch diagnostics.
- `lib/PTO/Transforms/CMakeLists.txt` - Adds the new PTO-to-A5VM helper implementation to `PTOTransforms`.

## Decisions Made

- Kept helper extraction functions as named per-op routines so Phase 2 metadata remains inspectable instead of being folded into one opaque matcher.
- Emitted placeholder A5VM ops with attached contract attributes to preserve the fixture-locked metadata surface before backend pass wiring lands in the next plan.
- Used exact, dedicated ACC and MAT store failure helpers so future work has visible completion points in code and diagnostics.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- `CLAUDE.md` was not present at the repository root, so execution proceeded using the phase plan, planning state, and workspace skill instructions.
- The local build environment failed under `ccache` because `/home/mouliangyu/.ccache/tmp` was read-only in the sandbox. Verification succeeded by rebuilding `PTOTransforms` with `CCACHE_DISABLE=1`.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Plan `02-03` can now wire these lowering helpers into the `a5vm` backend path instead of inventing contracts during pass integration.
- The fixture-locked metadata keys and diagnostics from `02-01` now have matching helper implementations and a build-wired source file.

## Self-Check: PASSED

---
*Phase: 02-pto-lowering*
*Completed: 2026-03-19*
