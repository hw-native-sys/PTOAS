# 06 — Small-L `ui8→ui16` widen

## Algorithm need

Fused quant→dequant needs a **small-L** UE8M0 widen (`ui8→ui16` at L=8) on the
compact SF path. L=256 widen already works; L=8 illegalizes to residual
`pto.vmi.extui`, so fused round-trip cannot match ASC BW and falls back to
compose / larger tiles.

Reference for the working L=256 shape:
`test/lit/vmi_new/vmi_to_vpto_integer_casts.pto`.

## Working-today workaround

L=256 `pto.vmi.vcvt` ui8→ui16 (`buggy_vmi.pto`). Checked-in dump:
`lowered_vpto.pto` — key ops `pto.vcvt ... {part = "EVEN"}` and `{part = "ODD"}`.

Reproduce:

```bash
$PTO_TEST_OPT buggy_vmi.pto $PASS -o lowered_vpto.pto
```

## Desired VMI + current failure

L=8 group-slot widen (`desired_vmi.pto`):

```
error: failed to legalize operation 'pto.vmi.extui' that was explicitly marked illegal
...
error: VMI-RESIDUAL-OP: failed to convert all VMI ops/types to VPTO
```

## Target MI + bisheng status

`target_mi.pto` is the MI shape the L=256 path already reaches (`vcvt` EVEN +
ui16 store). Small-L VMI should lower here instead of leaving `extui`.

| Step | Result |
|---|---|
| `ptoas ... --emit-vpto` | OK |
| `ptoas ... --emit-vpto-llvm-ir` | OK |
| `bisheng ... -c -x ir` → `.o` | **FAIL** — bisheng crashes (exit 139) on the emitted HIVM IR |
