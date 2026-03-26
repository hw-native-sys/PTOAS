# 14. Arith (Shared MLIR Dialect)

> **Category:** Shared scalar and index operations used around PTO ops
> **Dialect:** `arith`
> **Upstream Reference:** https://mlir.llvm.org/docs/Dialects/ArithOps/

The upstream MLIR `arith` dialect defines primitive arithmetic, comparison, select, and cast operations over signless integer, index, and floating-point values. In VPTO programs, these ops are used as the scalar and index layer around PTO instructions: they build constants, compute offsets and loop bounds, derive valid-shape metadata, and form predicates for `scf` control flow.

These ops are part of the supported VPTO source surface, but they are not PTO ISA instructions and do not map to CCE builtins directly.

---

## Role in VPTO Programs

- materialize scalar constants used by PTO scalar operands and loop bounds
- compute scalar/index offsets for tensor views, partitioning, and dynamic shapes
- derive scalar predicates that guard `scf.if` or `scf.while`
- select between scalar values without introducing PTO-specific control ops

Prefer PTO ops for vector or tile payload math. Use `arith` for scalar/index bookkeeping that surrounds PTO regions.

---

## Supported Author-Facing Ops

| Op | Typical VPTO Use | Notes |
|----|------------------|-------|
| `arith.constant` | index, integer, boolean, and float constants | ubiquitous in samples and tests |
| `arith.addi` | scalar/index addition | loop-carried counters, offsets, break-like state |
| `arith.subi` | scalar/index subtraction | remainder or tail-size computation |
| `arith.muli` | scalar/index multiplication | row-stride and chunk-offset computation |
| `arith.cmpi` | integer/index comparison | branch predicates and loop conditions |
| `arith.select` | scalar select | choose scalar/index values without branching |
| `arith.index_cast` | cast integer to `index` or back | dynamic shape and ABI glue |
| `arith.index_castui` | unsigned cast into `index`/integer | lowering and copy-shape helpers |
| `arith.andi` | boolean / integer AND | combine loop-continue flags |
| `arith.xori` | boolean / integer XOR | negate or flip boolean state |
| `arith.remui` | unsigned remainder | parity and chunk splitting helpers |
| `arith.ceildivsi` | signed ceil division | tile-count and chunk-count setup |

---

## Lowering-Generated Helper Ops

The PTO lowering and LLVM emission paths may also introduce a few additional `arith` ops as internal helpers even when authors do not write them directly:

- `arith.extui`
- `arith.extsi`
- `arith.trunci`
- `arith.shli`
- `arith.ori`
- `arith.divui`

These helper ops are useful for integer width adjustment or packed-parameter construction, but they are not the primary author-facing surface documented for VPTO kernels.

---

## Current PTOAS Coverage

- the A5VM shared-dialect fixture explicitly exercises `arith.addi` together with `scf.for` and `scf.yield`
- the A5VM text emitter currently has explicit handling for `arith.constant`, `arith.index_castui`, `arith.muli`, `arith.addi`, `arith.subi`, `arith.cmpi`, and `arith.select`
- the broader PTO lowering and view/materialization helpers also rely on `arith.index_cast`, `arith.extui`, `arith.extsi`, `arith.trunci`, `arith.shli`, `arith.ori`, and related scalar integer helpers

This means the author-facing subset above is intentionally larger than the one A5VM text emission test exercises in a single path, but it stays within the scalar/index patterns already present in the repository.

---

## Typical Patterns

### Scalar Setup

```mlir
%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : index
%scale = arith.constant 2.0 : f32
```

### Dynamic Offset Computation

```mlir
%vrow = arith.index_cast %valid_row : i32 to index
%chunk = arith.muli %row, %c32 : index
%tail = arith.subi %limit, %chunk : index
```

### Scalar Predicate and Selection

```mlir
%is_first = arith.cmpi eq, %i, %c0 : index
%active = arith.select %is_first, %first_count, %steady_count : index
```

### Break-Like Boolean Update

```mlir
%break_now = arith.cmpi eq, %i, %c2 : index
%alive_next = arith.xori %break_now, %true : i1
```

---

## Authoring Guidance

- keep `arith` values scalar or `index` typed; do not use `arith` as a substitute for PTO vector/tile compute
- use `arith.cmpi` plus `scf.if` / `scf.while` for control flow, not ad hoc control intrinsics
- prefer `arith.index_cast` / `arith.index_castui` at ABI or shape boundaries where `index` is required
- keep branch predicates and loop bounds explicit; this helps PTO analyses reason about memory scope and synchronization
