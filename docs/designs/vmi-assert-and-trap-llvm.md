# Proposal: Fix `pto.trap` VPTO→LLVM + Add `pto.vmi.assert`

Status: **proposal** (design only; no functional change in this PR)

Date: 2026-08-06

Motivation: TileLang / MoE kernels (e.g. Ascend `group_count`) need fail-fast
SDC checks of the form `-1 <= expert_idx < num_groups`. Scalar SimtVF already
has `T.device_assert` (printf + fake OOB store). VMI / VPTO kernels need an
equivalent that survives the A5 fatobj pipeline.

## Summary

| Item | Proposal |
| --- | --- |
| **P0 fix** | Lower `pto.trap` in VPTO→LLVM (EmitC already works) |
| **P1 surface** | Add `pto.vmi.assert %fail_mask` |
| **P1 lowering** | Implement assert as masked `vscatter` of a dummy value to UB base with signed index `-1` (deliberate UB OOB, same *class* of fault as SIMT fake `-1` store) |

Camodel evidence (Ascend950PR_9599, 2026-08-06): live-lane
`RV_VSCATTER` with index `-1` against UB base `0` reports
`vec_err_ub_addr_overflow_t0` / `ub_addr_overflow`. Empty mask does not.

---

## Part 1 — Fix `pto.trap` on the VPTO LLVM path

### Current state

| Backend | Status |
| --- | --- |
| EmitC (`PTOToEmitC.cpp`) | **Works**: `pto.trap` → opaque call `trap()` |
| Sample | `test/samples/Trap/trap.py` covers EmitC IR construction |
| VPTO→LLVM (`VPTOLLVMEmitter.cpp`, `VPTOCANN900LLVMEmitter.cpp`) | **Missing**: no `LowerTrapOpPattern`; `pto.trap` survives into device LLVM translation and fails with `missing LLVMTranslationDialectInterface for pto.trap` |

`TrapOp` is defined in `include/PTO/IR/PTOOps.td` as a nullary micro-op /
SIMT op. Manual (`docs/PTO_IR_manual.md` §`pto.trap`) already says it should
lower to a device trap/abort intrinsic.

TileLang PTO codegen emits `pto.TrapOp()` (or historically `_llvm_dialect.TrapOp()`)
under `pto.if_(cond == 0)` for `tl.device_assert`. That path is blocked today
on A5 VPTO fatobj builds.

### Proposed fix

Add a nullary lowering pattern next to other sync/config patterns in both
emitters:

```text
pto.trap
  → VPTO LLVM: func.call @trap()   # preferred, matches EmitC name
  OR llvm.intr.trap                 # fallback if bisheng rejects @trap
```

Concrete steps:

1. **`LowerTrapOpPattern`** in `VPTOLLVMEmitter.cpp` and
   `VPTOCANN900LLVMEmitter.cpp`:
   - match `pto::TrapOp`
   - emit `func.call` to callee `"trap"` with empty args (same string EmitC uses)
   - register planned decl `() -> ()`
   - erase the op
2. Register the pattern in the conversion pattern set (near
   `LowerMemBarOpPattern` / nullary config ops).
3. Confirm whether Ascend device LLVM / bisheng resolves `trap` as an
   intrinsic or needs an explicit declaration / `llvm.intr.trap`. Prefer the
   EmitC-compatible name first; switch only if board/camodel compile fails.
4. **Tests**
   - lit: VPTO round-trip / LLVM emit contains the trap call (extend
     `test/samples/Trap/` or add `test/vpto/.../trap`).
   - Optional camodel: scalar `if` + `pto.trap` under false condition still
     completes; under true condition, model reports a fault or host ACL error.

### Out of scope for the trap fix

- Message / printf payloads (SIMT `device_assert_with_msg` stays SimtVF-only
  until a separate print story exists on VPTO).
- Changing EmitC behaviour.

### Acceptance

- A kernel containing only `pto.trap` compiles through `ptoas --pto-backend=vpto`
  without `LLVMTranslationDialectInterface` errors.
- EmitC path remains green.

---

## Part 2 — `pto.vmi.assert(mask)` via OOB `vscatter`

### Why not only `pto.trap`?

`pto.trap` is **scalar / control-flow**. VMI range checks produce a
**lane predicate**. Pattern today:

```text
%ok = ...cmp range...
# need: if any lane is bad, fault — without scalarizing every lane
```

Scalarizing with `pto.if_` + `pto.trap` either needs a lane reduction that is
not yet a first-class VMI surface, or forces SimtVF. For vector counting /
histogram kernels, a masked side-effecting assert is the natural fit.

Tried and rejected as assert mechanisms (camodel + `.39` board):

| Idea | Result |
| --- | --- |
| `vdup(-1)` + `vcvt si32→ui16` NOSAT | No trap; wraps to `0xFFFF` |
| Illegal arithmetic / overflow flags | Not a reliable hard fault under live mask |

Working probe (same class as SIMT fake OOB write):

```text
# dest = first UB alloc (base address 0)
# values = any matching dtype vector (e.g. loaded from dest)
# idx = vbrc(i32(-1))
# mask = fail lanes (active ⇒ should fault)

pto.vmi.vscatter %values, %ub_base, %idx, %fail_mask
```

Camodel:

| Mode | Mask | Observation |
| --- | --- | --- |
| clean | 0 active | no `ub_addr_overflow`; ACL PASS |
| poison | 1 active | `vec_err_ub_addr_overflow_t0` on `RV_VSCATTER` (addr ~`0xfffffffc` for B32) |
| force | all active | same overflow, many times |

Host ACL still returned 0 on camodel (soft-fault log). Board behaviour for
this probe is TBD; SIMT assert on `.39` already surfaces ACL `507035`.

### Proposed IR

```mlir
// Active lanes in %fail are assertion failures.
// Empty / all-false mask is a no-op (must not fault).
pto.vmi.assert %fail : !pto.vmi.mask<Nxpred>
```

Optional sugar (frontend only):

```python
pto.vmi.assert(fail_mask)
# or TileLang: T.vmi.device_assert(fail_mask)
```

**Semantics (fail-mask):**

- For every active lane in `%fail`, the implementation must induce a
  device-visible fault before later side effects that depend on those lanes
  being valid.
- Inactive lanes must not fault.
- No results. Not `Pure`. Ordered with surrounding VMI memory / VF effects.

**Naming note:** fail-mask matches the OOB scatter mask directly. If reviewers
prefer “ok-mask” (`assert` when false), the surface can invert once in
lowering; document the choice in ODS and PTODSL clearly.

### Suggested ODS sketch

```tablegen
def VMIAssertOp : VMI_Op<"assert", []> {
  let summary = "Fault if any lane of the fail mask is active.";
  let arguments = (ins VMI_Mask:$fail);
  let results = (outs);
  let assemblyFormat = [{
    $fail attr-dict `:` type($fail)
  }];
}
```

Constraints:

- `%fail` is `!pto.vmi.mask<Nxpred>` (layout-assigned form after layout pass).
- Prefer contiguous physical chunks for the first implementation (same
  restriction family as early `vscatter` lowering).

### Lowering plan (`vmi-to-vpto`)

For each physical chunk of `%fail`:

1. Materialize a UB destination pointer that is known to be at (or near)
   addressable UB base for the assert scratch — options:
   - **A (preferred for kernels):** reuse an existing live UB pointer already
     in the VF (e.g. hist buffer / tile base). Document that callers may pass
     an optional base in a later revision if needed.
   - **B (self-contained):** emit a private 1-element (or 64-lane) UB scratch
     allocated at known offset 0 in the assert helper region.
2. Build `%idx = vbrc i32 -1` (or VPTO equivalent after layout split).
3. Build `%dummy` with element type matching `vscatter` contract
   (A5 today: **32-bit values + i32 indices + b32 mask**, or the documented
   16/8-bit variants). Simplest: `vload` from the scratch / hist base.
4. Emit `pto.vmi.vscatter %dummy, %ub, %idx, %fail_chunk` (→ existing
   `pto.vscatter` path).

Do **not** require `pto.trap` for the VMI assert path; trap remains the
scalar/control-flow primitive.

### PTODSL / TileLang surface

```python
# PTODSL
pto.vmi.assert(fail_mask)

# TileLang (follow-up)
T.vmi.device_assert(fail_mask)  # or T.vmi.assert(...)
# group_count example:
ok = (-1 <= idx) & (idx < num_groups)   # vector cmp chain
T.vmi.device_assert(~ok)                 # fail-mask
# then continue counting on idx >= 0
```

### Validation plan

1. **Unit / lit:** `assert` with empty mask → no scatter or scatter under
   zero mask; IR verifies.
2. **Camodel:** poison one lane → `ub_addr_overflow`; clean → quiet.
3. **Board (`.39` / `.52`):** confirm host-visible error (ACL / device error)
   for poison; clean PASS. Compare to SIMT `device_assert` ACL `507035`.
4. **Integration:** `group_count` VMI path uses `pto.vmi.assert` for
   `-1 <= idx < E` instead of silently counting OOR indices.

### Non-goals

- Printf / source location messages on the VMI path (phase 2).
- Replacing SIMT `__assert_fail` (keep SimtVF path as-is).
- Using arithmetic overflow / `vcvt` as the fault mechanism.

### Open questions

1. Fail-mask vs ok-mask naming — recommend **fail-mask** for 1:1 scatter mask.
2. Does board escalate `ub_addr_overflow` to a hard ACL error, or only a
   model log like camodel? Must verify before declaring production-ready.
3. Is a dedicated scratch UB required, or is “any UB pointer at base 0”
   enough when the first `alloc` in the VF is the hist buffer?
4. Should `pto.vmi.assert` also be legal under non-contiguous layouts in v1,
   or contiguous-only?

---

## Suggested implementation PR sequence

1. **This proposal PR** — design doc only (review API + lowering).
2. **Trap fix PR** — `LowerTrapOpPattern` + lit (unblocks scalar
   `tl.device_assert` → `pto.trap` on VPTO).
3. **Assert feature PR** — ODS + PTODSL + `vmi-to-vpto` scatter lowering +
   camodel/board probes.
4. **Consumer PR** — TileLang `group_count` / other MoE kernels switch to
   `T.vmi.assert`.

## References

- Camodel / board probes (TileLang tree):
  - `examples/ascend/vmi/group_count/test_vmi_scatter_neg1_ub.py`
  - `examples/ascend/vmi/group_count/test_vmi_s32_to_u16_trap.py` (negative result)
- EmitC trap: `lib/PTO/Transforms/PTOToEmitC.cpp` (`PTOTrapOpToEmitC`)
- Op def: `include/PTO/IR/PTOOps.td` (`TrapOp`)
- SIMT assert behaviour: TileLang Ascend `debug.h` + SimtVF `__assert_fail`
