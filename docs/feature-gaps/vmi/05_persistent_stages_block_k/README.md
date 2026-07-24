# 05 — Persistent stages > 1 / `block_k` > 512

## Algorithm need

Long-K quant / cast-back Persistent loops want multi-buffer stages (e.g.
stages=2) and `block_k > 512` so MTE and vector stay overlapped. Caps at
stages=1 / `block_k ≤ 512` leave the pipe underfed versus ASC on large hidden
sizes. Raising stages in the host schedule has also faulted in practice.

## Working-today workaround

`block_k = 512`, single-stage Persistent tile (`buggy_vmi.pto`). Checked-in
dump: `lowered_vpto.pto` — chunked `pto.vlds` / `pto.vcvt` / `pto.vmul` /
`pto.vsts` over the 512-wide tile.

Reproduce:

```bash
$PTO_TEST_OPT buggy_vmi.pto $PASS -o lowered_vpto.pto
```

## Desired VMI + current failure

`stages = 2` and `block_k = 1024` with dual UB buffers (`desired_vmi.pto`).
Layout assignment rejects the 1024-wide multiply:

```
error: VMI-UNSUPPORTED: pto.vmi.mulf ... deinterleaved = 2 ... requires ... deinterleaved = 4;
pto.vmi.ensure_layout cannot materialize this conversion
```

## Target MI + bisheng status

`target_mi.pto` sketches a stages=2 / block_k=1024 module with UNPK load,
mul, and PK4 store inside `vecscope`.

| Step | Result |
|---|---|
| `ptoas ... --emit-vpto` | OK |
| `ptoas ... --emit-vpto-llvm-ir` | OK |
| `bisheng ... -c -x ir` → `.o` | **FAIL** — bisheng crashes (exit 139) on the emitted HIVM IR |
