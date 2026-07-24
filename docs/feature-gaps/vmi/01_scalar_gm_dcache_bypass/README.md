# 01 — Scalar GM dcache-bypass store/load

## Algorithm need

Per-block quantization over **32×32** groups produces one f32 scale factor (SF)
per group. That scalar must land in GM (and later be re-read). Today the tiny
write still goes through bulk MTE, so setup/alignment/sync dominate and keep
VMI short of ASC bandwidth on the SF path.

## Working-today workaround

Stage the SF in UB, then `pto.copy_ubuf_to_gm` with a padded burst (here 32B
for a 4B payload). Checked-in dump: `lowered_vpto.pto` — key op is
`pto.copy_ubuf_to_gm` (no scalar GM path).

Reproduce:

```bash
PTO_TEST_OPT=.../pto-test-opt
PASS='-vmi-lower-unified-to-legacy -vmi-mask-granularity-assignment -vmi-layout-assignment -vmi-to-vpto'
$PTO_TEST_OPT buggy_vmi.pto $PASS -o lowered_vpto.pto
```

## Desired VMI + current failure

Direct `pto.vmi.vstore` of a slots=1 vector to `!pto.ptr<f32, gm>` with
`dcache_bypass` (see `desired_vmi.pto`):

```
error: 'pto.vmi.group_store' op requires memory destination to be UB-backed
```

## Target MI + bisheng status

`target_mi.pto` uses `pto.store_scalar ... {dcache_bypass}` on GM.

| Step | Result |
|---|---|
| `ptoas ... --emit-vpto` | OK |
| `ptoas ... --emit-vpto-llvm-ir` | OK |
| `bisheng ... -c -x ir` → `.o` | OK |
