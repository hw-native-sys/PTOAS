# Requirements: PTOAS A5VM Backend

**Defined:** 2026-03-18
**Core Value:** Preserve PTO library semantics and template-driven behavior inside PTOAS so backend lowering retains enough information to enable optimization instead of losing it during library instantiation.

## v1 Requirements

### Backend Integration

- [ ] **BACK-01**: Developer can run the existing PTOAS compilation flow with a backend path that replaces the current `emitc` generation slot without requiring a pass-pipeline redesign.
- [ ] **BACK-02**: Developer can keep ordinary control flow and scalar arithmetic in shared dialects such as `scf` and `arith` while only hardware-facing PTO operations enter the new backend path.
- [ ] **BACK-03**: Developer can compile the `Abs` sample through the new backend path using `./test/samples/runop.sh -t Abs`.

### A5VM Dialect

- [ ] **A5VM-01**: Developer can represent legal `a5vm` vector types whose total width is always exactly 256 bytes.
- [ ] **A5VM-02**: Developer can represent the `Abs` load path with an `a5vm` load operation whose result type is a legal `a5vm` vector type.
- [ ] **A5VM-03**: Developer can represent the `Abs` compute path with an `a5vm` absolute-value operation whose operand and result types are legal `a5vm` vector types.
- [ ] **A5VM-04**: Developer can represent the `Abs` store path with an `a5vm` store operation that consumes a legal `a5vm` vector value and backend-specific addressing inputs.

### PTO Lowering

- [ ] **PTO-01**: Developer can lower PTO `TLOAD` on the `Abs` path into `a5vm` operations while preserving the PTO-side layout, shape, and valid-region decisions needed for backend code selection.
- [ ] **PTO-02**: Developer can lower PTO `TABS` on the `Abs` path into `a5vm` operations in a way that matches existing PTO parameter behavior and unary-op template dispatch intent.
- [ ] **PTO-03**: Developer can lower PTO `TSTORE` on the `Abs` path into `a5vm` operations while preserving the PTO-side source tile domain and destination layout behavior needed for code selection.
- [ ] **PTO-04**: Developer can add new PTO-to-A5VM lowerings through the same framework without changing the backend architecture established for `Abs`.

### HIVM Textual Emission

- [ ] **HIVM-01**: Developer can lower `a5vm` operations into textual LLVM HIVM intrinsic IR instead of `emitc` C++.
- [ ] **HIVM-02**: Developer can derive intrinsic spellings from operation family, vector type, and variant metadata rather than hardcoding a single `Abs` string path.
- [ ] **HIVM-03**: Developer can emit structurally legal and reasonable textual IR for the `Abs` path even though final downstream validation happens on another machine.

### Validation

- [ ] **VAL-01**: Developer can inspect the `Abs` backend output and identify the exact LLVM HIVM intrinsic names needed by the implemented path.
- [ ] **VAL-02**: Developer can use the `Abs` path as the first acceptance case for future expansion of the A5VM backend.

## v2 Requirements

### PTO Coverage Expansion

- **PTOX-01**: Developer can lower additional PTO unary and binary vector operations beyond `TABS`.
- **PTOX-02**: Developer can support more `TLOAD` and `TSTORE` layout and data-type variants beyond the minimum `Abs` path.
- **PTOX-03**: Developer can support accumulator and mat-tile backend flows through `a5vm` or adjacent backend abstractions.

### Backend Maturity

- **A5VX-01**: Developer can replace textual HIVM emission with first-class LLVM intrinsic ops when the target environment exposes them.
- **A5VX-02**: Developer can validate emitted HIVM IR locally in CI or tool-based tests.

## Out of Scope

| Feature | Reason |
|---------|--------|
| Full PTO library reimplementation | v1 only needs the minimum interfaces required to compile the `Abs` sample |
| Broad PTO op coverage unrelated to `Abs` | would expand scope before the backend architecture is proven |
| Downstream execution validation on local machine | target environment for full HIVM validation is not available locally |
| Pass-pipeline redesign | the user requires replacing only the `emitc` position, not reworking the overall pipeline |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| BACK-01 | Phase 1 | Pending |
| BACK-02 | Phase 1 | Pending |
| BACK-03 | Phase 4 | Pending |
| A5VM-01 | Phase 1 | Pending |
| A5VM-02 | Phase 1 | Pending |
| A5VM-03 | Phase 1 | Pending |
| A5VM-04 | Phase 1 | Pending |
| PTO-01 | Phase 2 | Pending |
| PTO-02 | Phase 2 | Pending |
| PTO-03 | Phase 2 | Pending |
| PTO-04 | Phase 2 | Pending |
| HIVM-01 | Phase 3 | Pending |
| HIVM-02 | Phase 3 | Pending |
| HIVM-03 | Phase 3 | Pending |
| VAL-01 | Phase 4 | Pending |
| VAL-02 | Phase 4 | Pending |

**Coverage:**
- v1 requirements: 16 total
- Mapped to phases: 16
- Unmapped: 0 ✓

---
*Requirements defined: 2026-03-18*
*Last updated: 2026-03-18 after initial definition*
