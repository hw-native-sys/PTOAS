# 15. SCF (Shared MLIR Dialect)

> **Category:** Shared structured control flow around PTO regions
> **Dialect:** `scf`
> **Upstream Reference:** https://mlir.llvm.org/docs/Dialects/SCFDialect/

The upstream MLIR `scf` dialect defines structured control flow operations with regions, including counted loops, conditional regions, and while-style loops. In VPTO programs, `scf` is the control shell around PTO ops: it sequences DMA, vector, and tile operations; carries scalar or tile state across iterations; and preserves analyzable control flow for PTO-specific analyses and lowerings.

These ops are part of the supported VPTO source surface, but they are shared MLIR control-flow constructs rather than PTO ISA instructions.

---

## Supported Ops

| Op | Role in VPTO Programs | Notes |
|----|------------------------|-------|
| `scf.for` | counted loops and loop-carried values | also used for `__VEC_SCOPE__` dummy-loop form |
| `scf.if` | structured conditional execution | may yield values or act as side-effect-only branch |
| `scf.yield` | region terminator for `for` / `if` / `while` bodies | carries loop or branch results |
| `scf.while` | break-like or stateful loops | useful for source-level structured control |
| `scf.condition` | loop-continue / loop-exit decision for `scf.while` | placed in the "before" region |

Ops such as `scf.execute_region`, `scf.forall`, or `scf.index_switch` are not part of the documented VPTO shared-dialect surface here.

---

## Current PTOAS Coverage

- `scf.for`, `scf.if`, and `scf.yield` are directly exercised in the shared-dialect A5VM fixture and appear widely across PTO samples
- the VPTO `__VEC_SCOPE__` contract is modeled as a specialized `scf.for` annotated with `llvm.loop.aivector_scope`
- PTO synchronization and memory analyses explicitly reason about `scf.for`, `scf.if`, `scf.yield`, and `scf.while`
- `scf.while` and `scf.condition` appear in control-flow samples and are handled in PTO-to-EmitC control-flow lowering, but they are less broadly exercised than `for` / `if` on all backend paths

---

## Typical Patterns

### Counted Loop

```mlir
scf.for %i = %c0 to %c4 step %c1 {
  %offset = arith.muli %i, %c32 : index
  %v = pto.vlds %ub[%offset] : !llvm.ptr<6> -> !pto.vreg<64xf32>
  %abs = pto.vabs %v : !pto.vreg<64xf32> -> !pto.vreg<64xf32>
  pto.vsts %abs, %ub_out[%offset] : !pto.vreg<64xf32>, !llvm.ptr<6>
}
```

### Counted Loop with Loop-Carried State

```mlir
%final_alive = scf.for %i = %c0 to %c4 step %c1
    iter_args(%alive = %true) -> (i1) {
  %break_now = arith.cmpi eq, %i, %c2 : index
  %next_alive = scf.if %break_now -> (i1) {
    scf.yield %false : i1
  } else {
    scf.yield %alive : i1
  }
  scf.yield %next_alive : i1
}
```

### Structured Conditional Region

```mlir
%is_mode_a = arith.cmpi eq, %mode, %c0_i32 : i32
scf.if %is_mode_a {
  pto.tmuls ins(%data, %scale_a : !pto.tile_buf<...>, f32) outs(%data : !pto.tile_buf<...>)
} else {
  pto.tadds ins(%data, %bias_b : !pto.tile_buf<...>, f32) outs(%data : !pto.tile_buf<...>)
}
```

### While-Style Break Loop

```mlir
%final:2 = scf.while (%i = %c0, %alive = %true) : (index, i1) -> (index, i1) {
  %lt = arith.cmpi slt, %i, %c4 : index
  %go = arith.andi %lt, %alive : i1
  scf.condition(%go) %i, %alive : index, i1
} do {
^bb0(%i2: index, %alive2: i1):
  %next_i = arith.addi %i2, %c1 : index
  scf.yield %next_i, %alive2 : index, i1
}
```

---

## Authoring Guidance

- use `scf.for` for regular counted loops and loop-carried scalar/tile state
- use `scf.if` for structured branching around PTO regions instead of inventing PTO-specific branch ops
- keep region results explicit with `scf.yield`; this is important for PTO analyses that track carried buffers and aliasing
- use `scf.while` only when a counted loop cannot express the control cleanly; `scf.for` remains the more common and better-exercised form in the current repository
- build branch predicates and loop conditions with `arith` ops, not PTO vector masks, unless the control decision truly comes from a scalarized value
