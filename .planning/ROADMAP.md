# Roadmap: PTOAS A5VM Backend

**Created:** 2026-03-18
**Granularity:** standard
**Mode:** yolo
**Requirements Source:** `.planning/REQUIREMENTS.md`

## Overview

**4 phases** | **16 v1 requirements mapped** | All v1 requirements covered ✓

| # | Phase | Goal | Requirements | Success Criteria |
|---|-------|------|--------------|------------------|
| 1 | A5VM Foundation | Introduce the corrected backend boundary and the minimum hardware-facing `a5vm` primitive IR needed for the `Abs` path | BACK-01, BACK-02, A5VM-01, A5VM-02, A5VM-03, A5VM-04 | 4 |
| 2 | PTO Lowering | Lower `TLOAD`, `TABS`, and `TSTORE` from PTO into `a5vm` using the real PTO-library structure rather than pseudo load/store abstractions | PTO-01, PTO-02, PTO-03, PTO-04 | 4 |
| 3 | HIVM Emission | Replace the `emitc` output path with textual LLVM HIVM emission for the implemented `a5vm` subset | HIVM-01, HIVM-02, HIVM-03 | 3 |
| 4 | Abs Validation | Compile the `Abs` sample end to end through the new path and extract the required HIVM intrinsic inventory | BACK-03, VAL-01, VAL-02 | 3 |

## Phase Details

### Phase 1: A5VM Foundation

**Goal**

Create the corrected backend entry point and the minimum hardware-facing `a5vm` primitive IR needed for the `Abs` path.

**Requirements**

- BACK-01
- BACK-02
- A5VM-01
- A5VM-02
- A5VM-03
- A5VM-04

**Success Criteria**

1. A new backend path exists at the current `emitc` boundary without redesigning the overall pass pipeline.
2. `a5vm` defines legal fixed-width 256-byte vector typing and rejects illegal widths.
3. `a5vm` uses the `mlir::a5vm` namespace and a primitive op surface that stays close to the CCE builtin layer rather than to PTO-interface-shaped pseudo-ops.
4. General control flow and scalar arithmetic remain handled by shared dialects rather than moving into `a5vm`.

**Plans:** 3 plans

Plans:
- [ ] `01-01-PLAN.md` — Replan required: existing fixtures assume the wrong A5VM op surface
- [ ] `01-02-PLAN.md` — Replan required: existing dialect implementation uses the wrong namespace and pseudo-op model
- [x] `01-03-PLAN.md` — Preserve raw corrected A5VM text at the backend seam and lock the explicit `--pto-backend=a5vm` contract

### Phase 2: PTO Lowering

**Goal**

Implement PTO-to-A5VM lowering that preserves the real PTO-library control structure and semantic decisions for the `Abs` path.

**Requirements**

- PTO-01
- PTO-02
- PTO-03
- PTO-04

**Success Criteria**

1. `TLOAD` lowers according to the PTO library’s GM-to-UB copy behavior rather than to a pseudo-load abstraction.
2. `TABS` lowers according to the PTO library’s real vector pipeline and loop structure, including `vld`, `vabs`, and `vst`.
3. `TSTORE` lowers according to the PTO library’s UB-to-GM copy behavior while preserving the source tile domain and destination layout decisions needed for backend code selection.
4. The lowering structure is reusable for future PTO ops without replacing the architecture established for `Abs`.

**Plans:** 6 plans

Plans:
- [x] `02-01-PLAN.md` — Rewrite the Phase 2 public contracts, fixtures, and runner around explicit AIV loop-scope carriage for `TABS`
- [x] `02-02-PLAN.md` — Implement truthful PTO-to-A5VM helper lowering for copy families and explicit AIV-scoped unary lowering
- [x] `02-03-PLAN.md` — Register and wire the corrected PTO-to-A5VM execution path through `ptoas --pto-backend=a5vm`
- [x] `02-04-PLAN.md` — Recover exact TLOAD and TSTORE stride plus partition-trace metadata in observable lowered IR
- [x] `02-05-PLAN.md` — Preserve the explicit `__VEC_SCOPE__` dummy-loop carrier and restore full locked diagnostics
- [x] `02-06-PLAN.md` — Make the committed Phase 2 verification runner runnable with a workspace-safe FileCheck strategy

### Phase 3: HIVM Emission

**Goal**

Emit textual LLVM HIVM intrinsic IR from `a5vm` and fully remove the `emitc` output dependency for the implemented subset.

**Requirements**

- HIVM-01
- HIVM-02
- HIVM-03

**Success Criteria**

1. The implemented backend subset emits textual LLVM HIVM intrinsic IR instead of `emitc` C++.
2. Intrinsic spellings are derived from op/type/variant information rather than a single hardcoded string path.
3. The textual output for the implemented subset is structurally legal and suitable for downstream verification on another machine.

**Plans:** 4 plans

Plans:
- [ ] `03-01-PLAN.md` — Create the Phase 3 FileCheck fixtures and committed runner for HIVM emission, naming, unresolved reporting, and llvm-as parsing
- [ ] `03-02-PLAN.md` — Build the shared HIVM intrinsic naming and unresolved-selection helper layer
- [ ] `03-03-PLAN.md` — Implement the LLVM-like A5VM text printer and unresolved sidecar serialization
- [ ] `03-04-PLAN.md` — Wire `ptoas` to replace the final EmitC output slot with the new HIVM text emitter and run the full Phase 3 suite

### Phase 4: Abs Validation

**Goal**

Use the `Abs` sample as the first acceptance case for the new backend and extract the exact intrinsic inventory required by the implemented path.

**Requirements**

- BACK-03
- VAL-01
- VAL-02

**Success Criteria**

1. `./test/samples/runop.sh -t Abs` can be compiled through the new backend path.
2. The emitted `Abs` path can be inspected to enumerate the exact LLVM HIVM intrinsic names required by the implementation.
3. The resulting path establishes a concrete baseline for adding more PTO operations later.

## Sequencing Notes

- Phase 1 must be corrected before PTO semantic lowering has a trustworthy target IR.
- Phase 2 must be corrected before the final intrinsic inventory can be trusted.
- Phase 3 should be implemented before full `Abs` validation because the sample acceptance target is the new textual HIVM output path.
- Phase 4 is the acceptance and inventory-extraction phase, not a separate architecture redesign.

---
*Last updated: 2026-03-19 after completing plan 02-06 execution*
