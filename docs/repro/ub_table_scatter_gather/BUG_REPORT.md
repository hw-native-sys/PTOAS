# Bug report — UB-resident `vscatter` / `vgather` returns stale data

Audience: PTOAS maintainers.
Tag / tree: `vmi-v0.1.3` (this worktree).

## Problem

A small lookup table kept in UB is written with two `vscatter` passes and read
back with `vgather`. The kernel compiles, launches, and completes without an
ACL error, but every gathered lane returns the **seed sentinel** (`-inf`) as if
the second scatter never landed.

The AscendC peer with the same schedule (`vscatter` → `mem_bar` → `vscatter` →
`mem_bar` → `vgather2`) is exact.

## Minimal algorithm

```text
pool[0..63]  <-  -inf          # vscatter seed
barrier V
pool[0..63]  <-  vals[0..63]   # vscatter payload (identity offsets)
barrier V
out[0..63]   <-  pool[0..63]   # vgather
```

Expected: `out == vals`. Observed: `out == [-inf] * 64`.

## Fixtures

| Fixture | Role | Expected on current build |
|---------|------|---------------------------|
| `fixtures/failing_vmi.py` | Minimal VMI repro | **FAIL** numeric (all `-inf`) |
| `fixtures/reference_asc_cce.asc` | AscendC peer | **PASS** compile (`bisheng`) |
| `fixtures/variant_no_barrier.py` | No `pipe_barrier` | still all `-inf` (ordering ruled out) |
| `fixtures/variant_uint_offsets.py` | `ui32` offsets | still wrong (signedness ruled out) |
| `fixtures/variant_byte_offsets.py` | offsets `* 4` | still wrong (units ruled out) |

## Ruled out

1. **Missing ordering.** Removing barriers does not change the result; adding
   `pipe_barrier("V")` between phases does not either. AscendC's
   `mem_bar(VST_VST)` / `mem_bar(VST_VLD)` have no VMI twin; the pipe barrier
   is the closest available and is strictly more conservative.
2. **Offset signedness.** Spec types the offset as `!pto.vmi.vreg<L×i32>`.
   Reinterpreting as `ui32` was tried; still stale.
3. **Offset units.** Spec says per-lane element offset. Scaling by 4 (byte
   offsets) was tried; also wrong.

## Criteria

1. `failing_vmi.py` launched on device returns `out == vals` (identity).
2. Until then, please treat UB-resident scatter/gather tables as broken on VMI
   and keep the AscendC path as the reference.

## Non-goals

- Frontend / TileLang emission (this package is hand-written pure ptodsl)
- SoftPipeline stage count
- Simulator-only claims as the sole proof
