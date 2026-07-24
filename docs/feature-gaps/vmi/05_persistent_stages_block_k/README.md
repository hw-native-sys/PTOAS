# 05 — Persistent stages > 1 / `block_k` > 512

## Algorithm need

Long-K quant / **cast-back (dequant)** Persistent loops want multi-buffer
stages (e.g. stages=2) and `block_k > 512` so MTE and vector stay overlapped.
Caps at stages=1 / `block_k ≤ 512` leave the pipe underfed versus ASC on large
hidden sizes. Raising stages in the host schedule has also faulted in practice.

Consumers today:

- **Per-block quant** — stuck on stages=1 and `block_k ≤ 512` on large shapes.
- **Cast-back / dequant** — sweet spot is stages=1 + tile_k=512; tile_k=1024
  regresses until larger K / deeper Persistent is legal and fast. Closing this
  gap is the main lever to push cast-back toward ASC BW.

## Current slow path

`block_k = 512`, single-stage Persistent tile (`current_slow_vmi.pto` / `current_slow_vmi.py`). Checked-in
dump: `lowered_vpto.pto` — chunked `pto.vlds` / `pto.vcvt` / `pto.vmul` /
`pto.vsts` over the 512-wide tile.

Reproduce:

```bash
$PTO_TEST_OPT current_slow_vmi.pto $PASS -o lowered_vpto.pto
```

## Desired VMI + current failure

`stages = 2` and `block_k = 1024` with dual UB buffers (`desired_vmi.pto` / `desired_vmi.py`).
Layout assignment rejects the 1024-wide multiply:

```
error: VMI-UNSUPPORTED: pto.vmi.mulf ... deinterleaved = 2 ... requires ... deinterleaved = 4;
pto.vmi.ensure_layout cannot materialize this conversion
```

## Target MI + bisheng status

`target_mi.pto` expresses the full stages=2 / `block_k=1024` Persistent
schedule (aligned with `desired_vmi.pto` + ASC HardEvent shell):

- dual UB args (`x_ub0/1`, `y_ub0/1`) selected by `slot = i % 2`
- prime / drain `pto.set_flag` / `pto.wait_flag` on `EVENT_ID0/1`
- in-loop `pto.wait_flag_dyn` / `pto.set_flag_dyn` on `%slot`
- VF body: `UNPK_B16` load → `vmul` → `PK4_B32` store

| Step | Result |
|---|---|
| `ptoas ... --emit-vpto` | OK (schedule + flags preserved) |
| `ptoas ... --emit-vpto-llvm-ir` | OK |
| `bisheng ... -c -x ir` → `.o` | **FAIL** — bisheng crashes on the emitted HIVM IR |

## ASC/CCE working baseline

`reference_asc_cce.asc` is the same schedule on the Ascend CCE path:
stages=2 ping-pong (`SetFlag`/`WaitFlag` on slots `w & 1`), `block_k=1024`,
VF body `UNPK_B16` / `vmul` / `PK_B32`. It compiles to a non-empty `.o` with
`bisheng --cce-aicore-only`, proving the pattern is legal while `target_mi`
HIVM still crashes.

```bash
BISHENG=${BISHENG:-/usr/local/Ascend/cann-9.0.0/tools/bisheng_compiler/bin/bisheng}
ASCEND=${ASCEND:-/usr/local/Ascend/cann-9.0.0}
"$BISHENG" -O2 -fPIC -std=c++17 --npu-arch=dav-3510 --cce-aicore-only -c \
  reference_asc_cce.asc -o /tmp/ref_gap05.o \
  -I"$ASCEND/include" \
  -I"$ASCEND/compiler/tikcpp/tikcfw" \
  -I"$ASCEND/compiler/tikcpp/tikcfw/impl" \
  -I"$ASCEND/compiler/tikcpp/tikcfw/interface"
test -s /tmp/ref_gap05.o
```
