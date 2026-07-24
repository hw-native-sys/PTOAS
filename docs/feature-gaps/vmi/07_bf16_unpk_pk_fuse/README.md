# 07 — bf16 UNPK / PK fuse (full strip)

## Algorithm need

Residual-mix **forward** strips (mix-apply / post combine) are dense bf16↔f32
lanes: load bf16, widen, scale/accumulate, narrow, store. ASC hits
`UNPK_B16` + part-aware widen and `PK_B32` narrow store on both halves of the
unpacked register. VMI authors write continuous `vload` + `vcvt`; that path
must lower to the same full unpack/pack form for ASC-level BW.

## Current slow path

Continuous L=64 strip (`current_slow_vmi.pto`). Checked-in dump:
`lowered_vpto.pto` — already emits `pto.vlds {dist = "UNPK_B16"}` and
`pto.vsts {dist = "PK_B32"}`, but only feeds **`part = "EVEN"`** after UNPK
(half of the unpacked load is unused). That half-utilized form is what residual-mix
fwd pays today versus ASC’s EVEN+ODD strip.

Reproduce:

```bash
$PTO_TEST_OPT current_slow_vmi.pto $PASS -o lowered_vpto.pto
```

## Desired VMI + current failure

Explicit unpack load (`desired_vmi.pto`) so authors can request UNPK without
relying on continuous-load heuristics:

```
error: failed to legalize operation 'pto.vmi.vload' that was explicitly marked illegal
    %x = pto.vmi.vload %src[%off] {dist_mode = "unpack"}
...
error: VMI-RESIDUAL-OP: failed to convert all VMI ops/types to VPTO
```

Also want continuous L=64 (or L=128) strips to lower with **both** EVEN and ODD
parts filled, matching `target_mi.pto`.

## Target MI + bisheng status

`target_mi.pto` — `UNPK_B16` load, EVEN+ODD `vcvt`, dual `vmul`, `vor` +
`PK_B32` store.

| Step | Result |
|---|---|
| `ptoas ... --emit-vpto` | OK |
| `ptoas ... --emit-vpto-llvm-ir` | OK |
| `bisheng ... -c -x ir` → `.o` | **OK** |
