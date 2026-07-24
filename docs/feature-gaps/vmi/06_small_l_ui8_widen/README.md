# 06 — Small-L `ui8→ui16` widen

## Algorithm need

Fused quant→dequant needs a **small-L** UE8M0 widen (`ui8→ui16` at L=8) on the
compact SF path. L=256 widen already works; L=8 illegalizes to residual
`pto.vmi.extui`, so fused round-trip cannot match ASC BW and falls back to
compose / larger tiles.

Reference for the working L=256 shape:
`test/lit/vmi_new/vmi_to_vpto_integer_casts.pto`.

## Current slow path

L=256 `pto.vmi.vcvt` ui8→ui16 (`current_slow_vmi.pto` / `current_slow_vmi.py`). Checked-in dump:
`lowered_vpto.pto` — key ops `pto.vcvt ... {part = "EVEN"}` and `{part = "ODD"}`.

Reproduce:

```bash
$PTO_TEST_OPT current_slow_vmi.pto $PASS -o lowered_vpto.pto
```

## Desired VMI + current failure

L=8 group-slot widen (`desired_vmi.pto` / `desired_vmi.py`):

```
error: failed to legalize operation 'pto.vmi.extui' that was explicitly marked illegal
...
error: VMI-RESIDUAL-OP: failed to convert all VMI ops/types to VPTO
```

## Target MI + bisheng status

`target_mi.pto` is the small-L MI shape fused quant→dequant needs: load,
`vcvt` `{part = "EVEN"}` under `PAT_VL8`, then ui16 store. Small-L VMI
(`desired_vmi.pto`) should lower here instead of leaving `extui`.

| Step | Result |
|---|---|
| `ptoas ... --emit-vpto` | OK |
| `ptoas ... --emit-vpto-llvm-ir` | OK |
| `bisheng ... -c -x ir` → `.o` | **OK** |

Remaining gap is **VMI legalization** of L=8 (desired → residual `extui`), not
HIVM/bisheng on this MI shape.

## ASC/CCE working baseline

`reference_asc_cce.asc` widens `ui8→ui16` under `PAT_VL8` with `vcvt` /
`PART_EVEN` then stores. It compiles to a non-empty `.o` with
`bisheng --cce-aicore-only` (CCE twin of the MI path above).

```bash
BISHENG=${BISHENG:-/usr/local/Ascend/cann-9.0.0/tools/bisheng_compiler/bin/bisheng}
ASCEND=${ASCEND:-/usr/local/Ascend/cann-9.0.0}
"$BISHENG" -O2 -fPIC -std=c++17 --npu-arch=dav-3510 --cce-aicore-only -c \
  reference_asc_cce.asc -o /tmp/ref_gap06.o \
  -I"$ASCEND/include" \
  -I"$ASCEND/compiler/tikcpp/tikcfw" \
  -I"$ASCEND/compiler/tikcpp/tikcfw/impl" \
  -I"$ASCEND/compiler/tikcpp/tikcfw/interface"
test -s /tmp/ref_gap06.o
```
