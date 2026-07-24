# 03 — Direct scalar / 1PT reduce store

## Algorithm need

Residual-mix post-bwd reduces a tile to **one f32 scalar per group**. ASC uses
a ONEPT/1PT-style store. If VMI still pads that result to 8 slots before store,
extra UB traffic shows up on the residual-mix path and keeps VMI below ASC BW.

## Working-today workaround

Group-reduce with `{group = 8}` and store the 8-slot result (`buggy_vmi.pto`).
High-level kernels also keep a pad-to-8 habit because dense adjacent size-1
stores have been unreliable in practice. Checked-in dump: `lowered_vpto.pto` —
key ops `pto.vcgadd` + `pto.vsts` with `PAT_VL8`.

Reproduce:

```bash
$PTO_TEST_OPT buggy_vmi.pto $PASS -o lowered_vpto.pto
```

## Desired VMI + current failure

Idiomatic mask1 store after `vcadd` (mirrors
`test/lit/vmi_new/vmi_to_vpto_reduce_addf_store.pto` in `desired_vmi.pto`).
**This lowers today** to `pto.vcadd` + `pto.vsts` under a `PAT_VL1` mask
(no explicit `dist = "1PT_B32"` in the dump). The remaining gap is practical:
residual-mix authoring still pads to 8, and ASC's ONEPT path is not what the
padded workaround emits.

## Target MI + bisheng status

`target_mi.pto` asks for `pto.vsts {dist = "1PT_B32"}` after `pto.vcadd`.

| Step | Result |
|---|---|
| `ptoas ... --emit-vpto` | OK |
| `ptoas ... --emit-vpto-llvm-ir` | OK |
| `bisheng ... -c -x ir` → `.o` | **FAIL** — bisheng crashes (exit 139) on the emitted HIVM IR (`vcadd` / `vstsx1`) |
