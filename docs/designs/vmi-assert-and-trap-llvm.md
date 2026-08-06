# Proposal: Fix `pto.trap` VPTO→LLVM + Add `pto.vmi.assert`

Status: **proposal** (design only; no functional change in this PR)

Date: 2026-08-06

## Summary

| Item | Proposal |
| --- | --- |
| **P0a** | Lower `pto.trap` in VPTO→LLVM (unblocks A5 fatobj scalar assert) |
| **P0b** | Fix EmitC / SIMT lowering of `pto.trap` (today emits AICore-only `trap()`) |
| **P1 surface** | Add `pto.vmi.assert %fail_mask` |
| **P1 lowering** | Masked `vscatter` of a dummy value to UB base with signed index `-1` (deliberate UB OOB; same fault *class* as SIMT fake `-1` store) |

Camodel evidence (Ascend950PR_9599, 2026-08-06): live-lane
`RV_VSCATTER` with index `-1` against UB base `0` reports
`vec_err_ub_addr_overflow_t0` / `ub_addr_overflow`. Empty mask does not.

---

## Background — why this is required

### Product requirement: SDC range check in `group_count`

MoE `group_count` consumes expert indices of shape `(tokens, topk)`. Contract
(shared with CUDA TileKernels):

```text
allow expert_idx == -1          # padding / unused slot
require -1 <= expert_idx < E    # otherwise silent wrong hist = SDC
if expert_idx >= 0: count++
```

CUDA reference (`TileKernels/.../group_count_kernel.py`):

```python
expert_idx = T.int32(group_idx[i, j])
T.device_assert(-1 <= expert_idx < num_groups)
if expert_idx >= 0:
    T.atomic_add(out_shared[expert_idx], 1)
```

Ascend ports need the same fail-fast behaviour for both:

1. **SimtVF baselines** (`group_count_simt.py` / DMA variant)
2. **VMI / VPTO production path** (`group_count_vmi.py`)

Without a device-visible fault, an OOR index quietly corrupts the histogram.

### How Ascend classic SIMT supports this today (works)

TileLang `target="ascend"` + `T.SimtVF` lowers:

```text
T.device_assert(cond)
  → device_assert(cond)          # tl_templates/ascend/debug.h
  → assert(cond)                 # AscendC assert macro
  → __assert_fail(...)           # asc_assert_simt_impl.h
       printf DUMP_SIMT_ASSERT
       __trap():  *((uint8_t __gm__*)-1) = 0   # deliberate GM OOB
```

Notes:

- SIMT has **no** AICore `trap()` builtin. Fail-fast is **fake OOB store**.
- Validated on camodel + board `.39`: inject `expert_idx=E` → host ACL
  `507035` + `Device assert failed` / `[ASSERT] ... Assertion false`.
- AICore AscendC assert uses real `trap()` (`asc_aicore_assert_impl.h`).

So the working Ascend SIMT story is **outside PTOAS** (classic AscendC /
TileLang Ascend codegen).

### Does current PTO lowering support the same SIMT assert?

**Short answer: not correctly / not end-to-end.**

| Path | What happens today | SDC assert usable? |
| --- | --- | --- |
| TileLang Ascend SimtVF (`target=ascend`) | `device_assert` → `__assert_fail` → `__trap()` OOB | **Yes** (shipped) |
| TileLang PTO scalar (`target=pto`) | `device_assert` → `pto.if_(cond==0)` + `pto.trap` | EmitC: emits `trap()` (AICore). VPTO: **compile fails** (`missing LLVMTranslationDialectInterface for pto.trap`) |
| PTO `pto.simt_entry` + `pto.trap` | ODS marks `TrapOp` with `SimtOpInterface`, but EmitC always lowers to opaque call **`trap()`** | **No** — `trap()` is `CCE_INTRINSIC[aicore]` / `__builtin_cce_trap`. SIMT runtime exposes `__trap()` (OOB write), not `trap()` |
| VMI / VPTO vector range check | no `pto.vmi.assert`; scalar reduce + `pto.trap` blocked on VPTO | **No** |

Implications for this proposal:

1. **VPTO→LLVM `pto.trap`** must be fixed for AICore / vector-side scalar
   asserts (and for TileLang PTO `device_assert` on fatobj).
2. **EmitC SIMT** must not blindly emit AICore `trap()` inside
   `pto.simt_entry`. It should emit the SIMT fail-fast sequence (at least
   `__asc_simt_vf::__trap()` / equivalent OOB store; printf optional).
3. **VMI** still needs a masked assert (`pto.vmi.assert`) because a lane
   predicate cannot be expressed as one scalar `pto.trap` without a
   reduction that is not yet a first-class VMI surface.

Until (2) lands, “rewrite group_count SIMT onto PTO SIMT + `pto.trap`”
would regress relative to classic Ascend SimtVF.

---

## Part 1 — Fix `pto.trap` lowering

### Current state

| Backend | Status |
| --- | --- |
| EmitC AICore | Emits opaque `trap()` — matches CANN AICore intrinsic |
| EmitC inside `pto.simt_entry` | **Incorrect**: same `trap()` call; SIMT needs `__trap()` / OOB store |
| Sample | `test/samples/Trap/trap.py` covers EmitC IR construction only |
| VPTO→LLVM (`VPTOLLVMEmitter.cpp`, `VPTOCANN900LLVMEmitter.cpp`) | **Missing**: no `LowerTrapOpPattern`; fails with `missing LLVMTranslationDialectInterface for pto.trap` |

`TrapOp` is defined in `include/PTO/IR/PTOOps.td` as
`[MicroOpInterface, SimtOpInterface]`. Manual (`docs/PTO_IR_manual.md`
§`pto.trap`) says it should lower to a device trap/abort intrinsic — but
does not distinguish AICore vs SIMT.

TileLang PTO codegen emits `pto.TrapOp()` under `pto.if_(cond == 0)` for
`tl.device_assert`. That path is blocked on A5 VPTO fatobj today.

### Proposed fix

#### 1a. VPTO→LLVM (AICore / vector)

```text
pto.trap
  → func.call @trap()     # preferred, matches EmitC / AICore intrinsic
  OR llvm.intr.trap       # fallback if bisheng rejects @trap
```

Concrete steps:

1. **`LowerTrapOpPattern`** in `VPTOLLVMEmitter.cpp` and
   `VPTOCANN900LLVMEmitter.cpp`:
   - match `pto::TrapOp`
   - emit `func.call` to callee `"trap"` with empty args
   - register planned decl `() -> ()`
   - erase the op
2. Register near `LowerMemBarOpPattern` / nullary config ops.
3. Confirm bisheng resolves `trap`; fall back only if needed.
4. Lit + optional camodel for scalar `if` + `pto.trap`.

#### 1b. EmitC SIMT context (required for PTO SIMT parity)

```text
pto.trap in AICore / non-simt func  →  trap()
pto.trap in pto.simt_entry          →  SIMT __trap() / *((__gm__ uint8_t*)-1)=0
                                      (+ optional printf like AscendC)
```

Acceptance for SIMT EmitC: a `pto.simt_entry` containing `pto.trap` compiles
with bisheng SIMT and faults on board the same way as AscendC `__assert_fail`
(host-visible device error). Do **not** call AICore `trap()` from SIMT.

### Out of scope for the trap fix alone

- Full message / printf payloads on VPTO (SIMT printf can stay AscendC-side
  until a PTO print story exists).
- VMI masked assert (Part 2).

### Acceptance

- A kernel with only `pto.trap` compiles through `ptoas --pto-backend=vpto`.
- EmitC AICore path remains green (`trap()`).
- EmitC SIMT path emits SIMT-legal fail-fast, not AICore `trap()`.

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
2. **Trap fix PR**
   - VPTO→LLVM `LowerTrapOpPattern` + lit (unblocks scalar
     `tl.device_assert` → `pto.trap` on fatobj).
   - EmitC SIMT context: `pto.trap` in `pto.simt_entry` → SIMT `__trap()` /
     OOB store, **not** AICore `trap()`.
3. **Assert feature PR** — ODS + PTODSL + `vmi-to-vpto` scatter lowering +
   camodel/board probes.
4. **Consumer PR** — TileLang VMI `group_count` switches to `T.vmi.assert`;
   optional PTO SIMT rewrite can then match classic Ascend SimtVF behaviour.

## References

- Requirement / CUDA reference:
  `TileKernels/tile_kernels/moe/group_count_kernel.py`
- Ascend SimtVF baseline (working today via AscendC, not PTO):
  TileLang `group_count_simt.py` + `tl_templates/ascend/debug.h`
- CANN SIMT fail-fast:
  `asc_assert_simt_impl.h` (`__assert_fail` → `__trap()` OOB GM write)
- CANN AICore fail-fast:
  `asc_aicore_assert_impl.h` (`__assert_fail` → `trap()`)
- Camodel / board probes (TileLang tree):
  - `examples/ascend/vmi/group_count/test_vmi_scatter_neg1_ub.py`
  - `examples/ascend/vmi/group_count/test_vmi_s32_to_u16_trap.py` (negative result)
- EmitC trap: `lib/PTO/Transforms/PTOToEmitC.cpp` (`PTOTrapOpToEmitC`)
- Op def: `include/PTO/IR/PTOOps.td` (`TrapOp`, `SimtOpInterface`)
