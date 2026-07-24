# 04 — Packed UE8M0 scale factors

## Algorithm need

Per-token MXFP8 stores UE8M0 scales with **pack_factor=2**: two `ui8` exponents
share one `ui16` element. Unpacked `ui8` SF doubles scale-factor traffic versus
packed storage and blocks ASC-level BW on that pipe.

## Current slow path

Store SF as unpacked `ui8` only (`current_slow_vmi.pto`). Checked-in dump:
`lowered_vpto.pto` — key op `pto.vsts ... {dist = "PK4_B32"}` into `!pto.ptr<ui8, ub>`
(byte store of the unpacked tensor, not a packed `ui16` SF layout).

Reproduce:

```bash
$PTO_TEST_OPT current_slow_vmi.pto $PASS -o lowered_vpto.pto
```

## Desired VMI + current failure

First-class pack into `ui16` then store (see `desired_vmi.pto`):

```
error: custom op 'pto.vmi.vpack' is unknown
```

## Target MI + bisheng status

`target_mi.pto` uses `pto.vsts {dist = "PK_B16"}` into `!pto.ptr<ui16, ub>`
(packed SF word store; ASC baseline also exercises `ONEPT_B16` / `PK_B16`).

| Step | Result |
|---|---|
| `ptoas ... --emit-vpto` | OK |
| `ptoas ... --emit-vpto-llvm-ir` | OK |
| `bisheng ... -c -x ir` → `.o` | **FAIL** — bisheng crashes (exit 139) on the emitted HIVM IR |

## ASC/CCE working baseline

`reference_asc_cce.asc` stores packed `ui16` SF with `ONEPT_B16` and `PK_B16`.
It compiles to a non-empty `.o` with `bisheng --cce-aicore-only`, proving the
packed-SF store shape is legal on the Ascend CCE path while `target_mi` HIVM
still crashes.

```bash
BISHENG=${BISHENG:-/usr/local/Ascend/cann-9.0.0/tools/bisheng_compiler/bin/bisheng}
ASCEND=${ASCEND:-/usr/local/Ascend/cann-9.0.0}
"$BISHENG" -O2 -fPIC -std=c++17 --npu-arch=dav-3510 --cce-aicore-only -c \
  reference_asc_cce.asc -o /tmp/ref_gap04.o \
  -I"$ASCEND/include" \
  -I"$ASCEND/compiler/tikcpp/tikcfw" \
  -I"$ASCEND/compiler/tikcpp/tikcfw/impl" \
  -I"$ASCEND/compiler/tikcpp/tikcfw/interface"
test -s /tmp/ref_gap04.o
```
