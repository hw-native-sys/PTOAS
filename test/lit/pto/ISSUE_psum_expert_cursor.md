# Issue: PTOAS cannot lower psum expert-cursor `scf.while` pattern

## In plain language

Packed per-group token layouts need a tiny device loop:

1. Walk tokens in order.
2. Advance a monotonic **group cursor** with a **data-dependent `while`**
   against an aligned prefix-sum threshold (`psum`).
3. Compute a **boolean `valid`** (real token vs pad).
4. Conditionally store a result.

**What works today (related samples):**

- Top-level `scf.while` with an `i1` alive flag (`test/samples/SCF/scf_while_break.pto`)
- `scf.for` carrying an index + nested `scf.while` with a **simple** condition
  (no `scf.if` inside the while condition)

**What fails (this repro):**

- Full algorithm in `psum_expert_cursor_while.pto`:
  `error: Unsupported SCF op remained after pre-lowering`
- Reduced trigger in `psum_for_while_if_condition.pto`:
  `scf.for` + nested `scf.while` whose condition contains `scf.if`
  → pinned `ptoas` **segfaults** (SIGSEGV) or fails SCF pre-lowering

## Out of scope

- Host layout generators
- Any compute beyond the keep-mask / store

## How to run

Use a working `ptoas` (pin tip `8a3e3def` or equivalent). Avoid shadowing
`PYTHONPATH` with an incomplete build tree (that can break `import ptoas._core`).

```bash
# Full expert-cursor algorithm (expects the Unsupported SCF diagnostic)
ptoas --pto-arch=a5 test/lit/pto/psum_expert_cursor_while.pto -o -

# Reduced for+while+if-in-condition trigger
ptoas --pto-arch=a5 test/lit/pto/psum_for_while_if_condition.pto -o -
```

Or via lit (FileCheck expects the failure until fixed):

```bash
# from a configured ptoas build / lit setup
llvm-lit test/lit/pto/psum_expert_cursor_while.pto
```

## Observed results (2026-08-06)

```
# psum_expert_cursor_while.pto
loc("eid_after"(...psum_expert_cursor_while.pto":40:20)): \
  error: Unsupported SCF op remained after pre-lowering
Error: Pass execution failed.

# psum_for_while_if_condition.pto (same machine)
Segmentation fault (exit 139)
```

Control: `test/samples/SCF/scf_while_break.pto` still lowers to EmitC.

## Desired fix

Lower (or rewrite during pre-lowering) nested `scf.while` with `scf.if` in the
condition / body so the expert-cursor pattern emits valid vector-kernel CCE.
