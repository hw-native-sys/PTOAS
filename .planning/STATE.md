---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
current_phase: 2
current_phase_name: PTO Lowering
current_plan: 6
status: executing
stopped_at: Completed 02-06-PLAN.md
last_updated: "2026-03-19T04:11:20.719Z"
last_activity: 2026-03-19
progress:
  total_phases: 4
  completed_phases: 2
  total_plans: 13
  completed_plans: 9
  percent: 69
---

# Project State

**Updated:** 2026-03-19
**Status:** Ready to execute
**Current Phase:** 2
**Current Phase Name:** PTO Lowering
**Total Phases:** 4
**Current Plan:** 6
**Total Plans in Phase:** 6
**Progress:** [███████░░░] 69%
**Last Activity:** 2026-03-19
**Last Activity Description:** Executed plan 02-06 to make the Phase 2 runner use deterministic FileCheck discovery and align validation docs

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
- Revised plan `02-01` executed and summarized
- Plan `02-02` executed and summarized
- Plan `02-03` executed and summarized
- Plan `02-04` executed and summarized
- Plan `02-05` executed and summarized
- Plan `02-06` executed and summarized
- Next execution target: execute `03-01-PLAN.md`

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
- Report unsupported A5VM seam cases through explicit comments, diagnostics, and optional sidecar files instead of guessing later-stage emission behavior.
- Use committed Phase 2 MLIR/FileCheck fixtures as the PTO semantic-lowering contract before implementing the lowering pass.
- Use a standalone Bash runner for Phase 2 verification instead of relying on external lit configuration.

## Recent Progress

- Rewrote `include/PTO/Transforms/A5VMLowering.h` around explicit A5-only TLOAD, TABS, and TSTORE lowering contracts
- Added `lib/PTO/Transforms/PTOToA5VMLowering.cpp` and split pass wiring away from shared contract extraction, copy-loop programming, and unary vec-scope lowering
- Switched `lib/PTO/Transforms/PTOToA5VM.cpp` to partial conversion so failed helper lowerings now fail the pass instead of being skipped
- Preserved the existing `tools/ptoas/ptoas.cpp` A5VM branch wiring and recorded task completion with an explicit empty commit because the branch already satisfied the plan contract
- Made `lib/PTO/Transforms/PTOToA5VM.cpp` show explicit dispatch through `lowerTLOAD`, `lowerTABS`, and `lowerTSTORE` while keeping the pass boundary thin
- Recovered copy-family stride metadata through `memref.subview`, `memref.reinterpret_cast`, and `memref.cast` chains in `PTOToA5VMLowering.cpp`
- Restored exact `trace_offsets` / `trace_sizes` from lowered subview form and tightened the TLOAD/TSTORE fixtures to require `src_strides` / `dst_strides = [32, 1]`

## Open Questions

- Which exact LLVM HIVM intrinsic spellings correspond to each builtin variant exercised by the final `Abs` path
- Whether the implemented `Abs` path needs only the currently expected load/abs/store intrinsic families or additional helper intrinsics

## Pending Todos

- None currently recorded for this phase execution.

## Session Continuity

- Next recommended command: `/gsd:execute-phase`
- Next plan to execute: `03-01-PLAN.md`
- Current blocker status: Phase 2 planning is complete, but end-to-end runner verification still needs a runnable `./build/tools/ptoas/ptoas` build tree and one of the documented FileCheck binaries.

## Performance Metrics

| Phase | Duration | Tasks | Files |
|-------|----------|-------|-------|
| Phase 02-pto-lowering P01 (replay) | 20min | 2 tasks | 11 files |
| Phase 01-a5vm-foundation P03 (refresh) | 22min | 2 tasks | 4 files |
| Phase 02 P02 | 7min | 2 tasks | 3 files |
| Phase 02-pto-lowering P03 | 24min | 2 tasks | 6 files |
| Phase 01 P01 | 21min | 2 tasks | 10 files |
| Phase 01-a5vm-foundation P02 | 25min | 2 tasks | 8 files |
| Phase 02-pto-lowering P02 | 24min | 2 tasks | 5 files |
| Phase 02-pto-lowering P03 | 12min | 2 tasks | 2 files |
| Phase 02-pto-lowering P01 | 8min | 2 tasks | 3 files |
| Phase 02-pto-lowering P02 | 14min | 2 tasks | 1 files |
| Phase 02-pto-lowering P04 | 75min | 2 tasks | 4 files |
| Phase 02-pto-lowering P05 | 12min | 2 tasks | 5 files |
| Phase 02-pto-lowering P06 | 12min | 2 tasks | 3 files |

## Decisions Made


- [Phase 02]: Keep the lowering surface split into public contracts plus a helper implementation file before pass wiring.
- [Phase 02]: Use explicit metadata attachment helpers so fixture-locked attribute names stay readable and reusable.
- [Phase 02]: Preserve unsupported TSTORE ACC and MAT paths as dedicated TODO diagnostics instead of collapsing them into a generic failure.
- [Phase 02]: Run PTO-to-A5VM only on the --pto-backend=a5vm branch after the shared pre-backend passes.
- [Phase 02]: Extract tile layout, valid dims, and address-space metadata from bind_tile and pointer_cast SSA chains because the A5VM boundary sees memref-backed tile values.
- [Phase 02]: Use an explicit rewrite walk instead of greedy pattern application so single-op Phase 2 fixtures retain visible a5vm.load and a5vm.abs ops in debug IR.
- [Phase 02]: Gate revised Wave 0 fixture replay on the landed A5VM primitive names instead of silently tolerating stale Phase 1 assumptions.
- [Phase 02]: Represent `__VEC_SCOPE__` structurally in Phase 2 fixtures by checking loop nesting and ordered `vlds` / `vabs` / `vsts`.
- [Phase 02]: Revisit the current `__VEC_SCOPE__` representation because the user confirmed it corresponds to `cce_aiv_loop_hint` / `llvm.loop.aivector_scope`, not just ordinary loop structure.
- [Phase 02]: Keep obsolete pseudo-op name rejection in the Phase 2 runner so fixture files stay focused on the corrected lowering shape.
- [Phase 01-a5vm-foundation]: Keep the Phase 1 A5VM seam at raw corrected backend text and defer llvm.hivm emission to the later HIVM phase.
- [Phase 01]: Keep the no-legacy-name regression check in the standalone runner rather than in the MLIR fixtures so file-level validation can forbid obsolete spellings entirely.
- [Phase 01-a5vm-foundation]: Keep copy-op transfer attrs parser-optional and verifier-required so invalid fixtures fail with the planned diagnostic instead of a parser error.
- [Phase 01-a5vm-foundation]: Derive copy transfer metadata from existing lowering contract fields instead of widening the public Phase 2 lowering structs in this plan.
- [Phase 01-a5vm-foundation]: Add A5VMOpsIncGen as a direct ptoas build dependency because the CLI includes generated A5VM headers before linking against PTOIR.
- [Phase 02]: Keep the public Phase 2 surface limited to lowerTLOAD, lowerTABS, and lowerTSTORE plus truthful A5-only contracts.
- [Phase 02]: Represent copy-family set_loop programming as explicit attached metadata so the PTO branch structure stays visible before dedicated loop ops exist.
- [Phase 02]: Build unary Abs lowering as structural SCF vec-scope loops around vlds, vabs, and vsts, and register SCF in the pass dependency list.
- [Phase 02-pto-lowering]: Use partial conversion so helper lowering failures surface as pass failures instead of being silently skipped.
- [Phase 02-pto-lowering]: Factor shared pre-backend passes into a helper so the A5VM branch stays structurally separate from EmitC.
- [Phase 02-pto-lowering]: Expose AIV vec-scope ownership as an explicit unary lowering contract instead of implying it from plain scf.for nesting.
- [Phase 02-pto-lowering]: Enforce the corrected vec-scope semantics in the Phase 2 runner so stale pseudo-op checks and bare loop-only fixtures fail fast.
- [Phase 02]: Treat an already-satisfied task as an explicit empty commit when enforcing one commit per executed plan task.
- [Phase 02]: Keep __VEC_SCOPE__ as a dedicated dummy loop carrier so the chosen loop owns cce_aiv_loop_hint and llvm.loop.aivector_scope metadata.
- [Phase 02-pto-lowering]: Treat task 2 as an explicit empty commit because the A5VM branch wiring already satisfied the plan contract.
- [Phase 02]: Recover copy-family metadata by walking memref.subview, memref.reinterpret_cast, and memref.cast instead of relying on direct PartitionViewOp producers.
- [Phase 02]: Keep Task 2 focused on fixture contracts by locking only the exact stride and trace attrs the emitted IR now proves.
- [Phase 02-pto-lowering]: Keep __VEC_SCOPE__ as a one-trip scf.for carrier with its printed attr-dict locking cce_aiv_loop_hint and llvm.loop.aivector_scope.
- [Phase 02-pto-lowering]: Replace fail-fast partial conversion with a full PTO-op lowering walk so dedicated diagnostics accumulate before signaling pass failure.
- [Phase 02]: Resolve FileCheck in the Phase 2 runner through FileCheck, FileCheck-19, then /usr/lib/llvm-19/bin/FileCheck before executing fixtures.
- [Phase 02]: Document bash test/phase2/run_phase2_checks.sh as the canonical quick-run and full-suite validation entrypoint for Phase 2.

## Blockers

- Fresh `CCACHE_DISABLE=1 ninja -C build PTOTransforms ptoas` verification is blocked by pre-existing A5VM generated-header and build-regeneration issues; see `.planning/phases/02-pto-lowering/deferred-items.md`.
- `bash test/phase2/run_phase2_checks.sh` now has deterministic FileCheck discovery, but this workspace still lacks both a `FileCheck` candidate and an executable `./build/tools/ptoas/ptoas`.

## Session

**Last Date:** 2026-03-19T04:11:20.717Z
**Stopped At:** Completed 02-06-PLAN.md
**Resume File:** None

---
*Last updated: 2026-03-19 after completing plan 02-06*
