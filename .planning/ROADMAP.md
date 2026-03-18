# Roadmap: PTOAS A5VM Backend

**Created:** 2026-03-18
**Granularity:** standard
**Mode:** yolo
**Requirements Source:** `.planning/REQUIREMENTS.md`

## Overview

**4 phases** | **16 v1 requirements mapped** | All v1 requirements covered ✓

| # | Phase | Goal | Requirements | Success Criteria |
|---|-------|------|--------------|------------------|
| 1 | A5VM Foundation | Introduce the new backend boundary and the minimum `a5vm` IR model needed for the `Abs` path | BACK-01, BACK-02, A5VM-01, A5VM-02, A5VM-03, A5VM-04 | 4 |
| 2 | PTO Lowering | Lower `TLOAD`, `TABS`, and `TSTORE` from PTO into `a5vm` while preserving PTO semantics and template behavior | PTO-01, PTO-02, PTO-03, PTO-04 | 4 |
| 3 | HIVM Emission | Replace the `emitc` output path with textual LLVM HIVM emission for the implemented `a5vm` subset | HIVM-01, HIVM-02, HIVM-03 | 3 |
| 4 | Abs Validation | Compile the `Abs` sample end to end through the new path and extract the required HIVM intrinsic inventory | BACK-03, VAL-01, VAL-02 | 3 |

## Phase Details

### Phase 1: A5VM Foundation

**Goal**

Create the new backend entry point and the minimum `a5vm` dialect surface required to represent the `Abs` vector path without relying on `emitc`.

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
3. `a5vm` contains the minimum load, abs, and store style operations needed for the `Abs` path.
4. General control flow and scalar arithmetic remain handled by shared dialects rather than moving into `a5vm`.

**Plans:** 3 plans

Plans:
- [x] `01-01-PLAN.md` — Create the Phase 1 FileCheck fixtures and committed runner for all Wave 0 checks
- [x] `01-02-PLAN.md` — Add the first-class `a5vm` dialect, 256-byte vector type, and minimum `load`/`abs`/`store` ops
- [x] `01-03-PLAN.md` — Wire explicit backend selection, A5VM text emission, and unresolved-report diagnostics at the current `emitc` boundary

### Phase 2: PTO Lowering

**Goal**

Implement PTO-to-A5VM lowering helpers that preserve existing PTO-side semantic decisions for the `Abs` path.

**Requirements**

- PTO-01
- PTO-02
- PTO-03
- PTO-04

**Success Criteria**

1. `TLOAD` lowers into `a5vm` using data derived from PTO tile/global semantics rather than a hardcoded sample-only path.
2. `TABS` lowers into `a5vm` in a way that matches existing PTO unary behavior and source/destination compatibility expectations.
3. `TSTORE` lowers into `a5vm` while preserving the source tile domain and destination layout decisions needed for backend code selection.
4. The lowering structure is reusable for future PTO ops without replacing the architecture established for `Abs`.

**Plans:** 3 plans

Plans:
- [x] `02-01-PLAN.md` — Create the Phase 2 FileCheck fixtures and committed runner for PTO semantic-lowering checks
- [ ] `02-02-PLAN.md` — Define the shared PTO-to-A5VM lowering contracts, helper layer, and explicit unsupported-branch behavior
- [ ] `02-03-PLAN.md` — Register the PTO-to-A5VM pass and wire it into the `--pto-backend=a5vm` path in `ptoas`

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

- Phase 1 must complete before PTO semantic lowering has a target IR.
- Phase 2 must complete before the final intrinsic inventory can be trusted.
- Phase 3 should be implemented before full `Abs` validation because the sample acceptance target is the new textual HIVM output path.
- Phase 4 is the acceptance and inventory-extraction phase, not a separate architecture redesign.

---
*Last updated: 2026-03-19 after completing plan 02-01*
