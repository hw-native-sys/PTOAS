# 02 — Compact inverse `vbrc` (no pad + BRC)

## Algorithm need

Quant kernels compute per-group reciprocal scales, then broadcast each inverse
across the group's lanes before the multiply. A **compact** inverse (one slot
per group, no pad) fan-out is what ASC-level BW wants; padding + BRC reload
adds UB traffic and ops on the hot path.

## Current slow path

Materialize recip scales as `#pto.vmi.layout<num_groups=8, slots=8>`,
`group_store` them, then reload with `dist_mode = "brc"`. Checked-in dump:
`lowered_vpto.pto` — key ops `pto.vsts` with `PAT_VL8`, then
`pto.vlds {dist = "BRC_B32"}`.

Reproduce:

```bash
$PTO_TEST_OPT current_slow_vmi.pto $PASS -o lowered_vpto.pto
```

## Desired VMI + current failure

Compact `vbrc` from `num_groups=8, slots=1` (see `desired_vmi.pto`):

```
error: VMI-LAYOUT-CONTRACT: pto.vmi.group_broadcast has no registered layout support:
operand#0=...num_groups = 8, slots = 1... result#0=...contiguous...
```

## Target MI + bisheng status

`target_mi.pto` is a `BRC_B32` load (plus a consumer store inside `vecscope`).

| Step | Result |
|---|---|
| `ptoas ... --emit-vpto` | OK |
| `ptoas ... --emit-vpto-llvm-ir` | OK |
| `bisheng ... -c -x ir` → `.o` | OK |
