# 03 — Direct scalar / 1PT reduce store

## Algorithm need

Residual-mix post-bwd reduces a tile to **one f32 scalar per group**. ASC uses
a ONEPT/1PT-style store. If VMI still pads that result to 8 slots before store,
extra UB traffic shows up on the residual-mix path and keeps VMI below ASC BW.

## Current slow path

Group-reduce with `{group = 8}` and store the 8-slot result (`current_slow_vmi.pto` / `current_slow_vmi.py`).
High-level kernels also keep a pad-to-8 habit because dense adjacent size-1
stores have been unreliable in practice. Checked-in dump: `lowered_vpto.pto` —
key ops `pto.vcgadd` + `pto.vsts` with `PAT_VL8`.

Reproduce:

```bash
$PTO_TEST_OPT current_slow_vmi.pto $PASS -o lowered_vpto.pto
```

## Desired VMI + current failure

Idiomatic mask1 store after `vcadd` (mirrors
`test/lit/vmi_new/vmi_to_vpto_reduce_addf_store.pto` in `desired_vmi.pto` / `desired_vmi.py`).
**This lowers today** to `pto.vcadd` + `pto.vsts` under a `PAT_VL1` mask
(no explicit `dist = "1PT_B32"` in the dump). The remaining gap is practical:
residual-mix authoring still pads to 8, and ASC's ONEPT path is not what the
padded slow path emits.

## Target MI + bisheng status

`target_mi.pto` asks for `pto.vcadd` then `pto.vsts {dist = "1PT_B32"}` —
the MI form of ASC `ONEPT_B32` (matches `reference_asc_cce.asc`).

| Step | Result |
|---|---|
| `ptoas ... --emit-vpto` | OK |
| `ptoas ... --emit-vpto-llvm-ir` | OK |
| `bisheng ... -c -x ir` → `.o` | **FAIL** — bisheng crashes (exit 139) on the emitted HIVM IR (`vcadd` / `vstsx1`) |

## ASC/CCE working baseline

`reference_asc_cce.asc` is a minimal hand-written Ascend CCE kernel:
`vcadd` then `vsts(..., ONEPT_B32)`. It compiles to a non-empty `.o` with
`bisheng --cce-aicore-only` (same CCE path ASC uses), proving the instruction
pattern is legal while `target_mi` HIVM still crashes.

```bash
BISHENG=${BISHENG:-/usr/local/Ascend/cann-9.0.0/tools/bisheng_compiler/bin/bisheng}
ASCEND=${ASCEND:-/usr/local/Ascend/cann-9.0.0}
"$BISHENG" -O2 -fPIC -std=c++17 --npu-arch=dav-3510 --cce-aicore-only -c \
  reference_asc_cce.asc -o /tmp/ref_gap03.o \
  -I"$ASCEND/include" \
  -I"$ASCEND/compiler/tikcpp/tikcfw" \
  -I"$ASCEND/compiler/tikcpp/tikcfw/impl" \
  -I"$ASCEND/compiler/tikcpp/tikcfw/interface"
test -s /tmp/ref_gap03.o
```
