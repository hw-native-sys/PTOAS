# 14. Arith (Shared MLIR Dialect)

> **Category:** Shared full scalar `arith` surface used around PTO ops
> **Dialect:** `arith`
> **Upstream Reference:** https://mlir.llvm.org/docs/Dialects/ArithOps/

The upstream MLIR `arith` dialect defines primitive arithmetic, comparison, select, and cast operations over signless integer, index, floating-point, and boolean-compatible scalar values. Within PTO micro Instruction code, the full scalar operation surface of `arith` is supported. These ops are used around PTO instructions to build constants, compute offsets and loop bounds, perform general scalar math, derive valid-shape metadata, and form predicates for `scf` control flow.

These ops are part of the documented PTO micro Instruction surface, but they are not PTO ISA instructions.

---

## Role in PTO micro Instruction Code

- materialize scalar constants used by PTO scalar operands and loop bounds
- compute scalar/index offsets for tensor views, partitioning, and dynamic shapes
- perform general scalar integer and floating-point math outside PTO vector/tile payload operations
- derive scalar predicates that guard `scf.if` or `scf.while`
- apply scalar casts, width changes, bitwise ops, and selects without introducing PTO-specific control ops

Prefer PTO ops for vector or tile payload math. Use `arith` for scalar computation and bookkeeping that surrounds PTO regions.

---

## Supported Surface

The documented PTO micro Instruction surface supports the full scalar operation surface of upstream `arith`. The upstream `arith` dialect reference remains authoritative for the exhaustive op-by-op syntax and semantics. The categories below summarize how that support is used in PTO micro Instruction code.

| Category | Representative Ops | Typical Use in PTO micro Instruction Code |
|----------|--------------------|------------------|
| Constants | `arith.constant` | integer, floating-point, boolean, and `index` constants |
| Integer / Index Arithmetic | `arith.addi`, `arith.subi`, `arith.muli`, `arith.divsi`, `arith.divui`, `arith.ceildivsi`, `arith.ceildivui`, `arith.floordivsi`, `arith.remsi`, `arith.remui` | offsets, bounds, chunk sizes, scalar math |
| Floating-Point Arithmetic | `arith.addf`, `arith.subf`, `arith.mulf`, `arith.divf`, `arith.negf`, `arith.maximumf`, `arith.minimumf`, `arith.maxnumf`, `arith.minnumf` | scalar math around PTO regions |
| Bitwise / Shift Ops | `arith.andi`, `arith.ori`, `arith.xori`, `arith.shli`, `arith.shrsi`, `arith.shrui` | flags, masks, packed scalar fields |
| Comparisons / Select | `arith.cmpi`, `arith.cmpf`, `arith.select`, `arith.maxsi`, `arith.minui` | predicates, clamps, scalar muxes |
| Casts / Width Changes | `arith.index_cast`, `arith.index_castui`, `arith.extsi`, `arith.extui`, `arith.trunci`, `arith.sitofp`, `arith.uitofp`, `arith.fptosi`, `arith.fptoui`, `arith.extf`, `arith.truncf`, `arith.bitcast` | ABI glue, dynamic-shape plumbing, scalar type adaptation |

---

## Current PTOAS Coverage

- the current repository examples are still dominated by constants, casts, integer/index arithmetic, compares, and selects because those are the most common surrounding-scalar patterns in existing kernels
- backend-specific tests such as the PTO shared-dialect fixture visibly exercise only a representative subset of `arith` ops in a single path
- the documented PTO micro Instruction source-level contract is nevertheless the full scalar `arith` surface, not just the index-heavy subset that appears most often in current samples

This section therefore uses representative categories and examples instead of pretending that the supported `arith` surface is limited to the currently most common sample patterns.

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

### General Scalar Arithmetic

```mlir
%sum_i = arith.addi %lhs_i, %rhs_i : i32
%sum_f = arith.addf %lhs_f, %rhs_f : f32
%prod_f = arith.mulf %sum_f, %scale : f32
```

### Scalar Predicate and Selection

```mlir
%is_first = arith.cmpi eq, %i, %c0 : index
%active = arith.select %is_first, %first_count, %steady_count : index
```

### Bitwise / Width Adaptation

```mlir
%flags = arith.andi %flags0, %flags1 : i32
%wide = arith.extui %flags : i32 to i64
%shrunk = arith.trunci %wide : i64 to i16
```

---

## Authoring Guidance

- treat upstream `arith` scalar semantics as the source of truth for supported scalar ops
- keep `arith` values scalar or `index` typed; do not use `arith` as a substitute for PTO vector/tile compute
- use `arith` for general scalar math, scalar comparisons, bitwise operations, and casts around PTO regions, not just for `index` arithmetic
- use `arith.cmpi` / `arith.cmpf` plus `scf.if` / `scf.while` for control flow, not ad hoc control intrinsics
- prefer `arith.index_cast` / `arith.index_castui` at ABI or shape boundaries where `index` is required, but do not read that as a restriction on the rest of scalar `arith`
