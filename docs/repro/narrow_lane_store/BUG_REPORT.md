# Bug report — 1-lane `vstore` needs 32-byte aligned UB address

Audience: PTOAS maintainers.
Tag / tree: `vmi-v0.1.3` (this worktree).

## Problem

A 1-lane `vstore` whose destination is 4-byte aligned but **not** 32-byte
aligned faults the vector cores:

```
RuntimeError: aclrtLaunchKernelWithHostArgs failed ... ACL error 507035
[Error]: The vector core execution is abnormal.
```

The AscendC peer stores to the same packed addresses with
`vsts(..., dist="ONEPT_B32")` and is fine.

The failure mode is especially nasty: the fault can poison the process so the
*next* unrelated kernel fails, and the traceback points somewhere innocent.

## Minimal algorithm

```text
for k in 0..7:
    vstore(val[k], out_ub[k], mask=1-lane)   # packed (8,) f32
```

For odd `k`, `&out_ub[k]` is only 4-byte aligned. That faults on VMI.

## Fixtures

| Fixture | Role | Expected on current build |
|---------|------|---------------------------|
| `fixtures/faulting_vmi.py` | Packed 1-lane stores | **FAIL** launch (ACL 507035) |
| `fixtures/padded_workaround_vmi.py` | `(B, 8)` padded slots | **PASS** numeric |
| `fixtures/reference_asc_cce.asc` | AscendC `ONEPT_B32` peer | **PASS** compile |

## Ask

Either:

1. Support 1-lane `vstore` to 4-byte-aligned UB addresses (matching AscendC
   `ONEPT_B32`), or
2. Emit a **build-time diagnostic** when a 1-lane store target is not 32-byte
   aligned, instead of a deferred device fault.

## Criteria

1. `faulting_vmi.py` launches without ACL 507035 and returns `out == vals`, or
2. Compiling a misaligned 1-lane store fails loudly at build time with a clear
   message naming the alignment requirement.

Until then, the padded `(B, 8)` layout in `padded_workaround_vmi.py` remains
the working path (32 bytes of UB per row, no runtime cost beyond the pad).

## Non-goals

- Frontend / TileLang emission (this package is hand-written pure ptodsl)
- Changing AscendC `ONEPT_B32` semantics
