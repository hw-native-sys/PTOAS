# 04 — Packed UE8M0 scale factors

## Algorithm need

Per-token MXFP8 stores UE8M0 scales with **pack_factor=2**: two `ui8` exponents
share one `ui16` element. Unpacked `ui8` SF doubles scale-factor traffic versus
packed storage and blocks ASC-level BW on that pipe.

## Working-today workaround

Store SF as unpacked `ui8` only (`buggy_vmi.pto`). Checked-in dump:
`lowered_vpto.pto` — key op `pto.vsts ... {dist = "PK4_B32"}` into `!pto.ptr<ui8, ub>`
(byte store of the unpacked tensor, not a packed `ui16` SF layout).

Reproduce:

```bash
$PTO_TEST_OPT buggy_vmi.pto $PASS -o lowered_vpto.pto
```

## Desired VMI + current failure

First-class pack into `ui16` then store (see `desired_vmi.pto`):

```
error: custom op 'pto.vmi.vpack' is unknown
```

## Target MI + bisheng status

`target_mi.pto` uses `pto.vsts {dist = "PK_B16"}` into `!pto.ptr<ui16, ub>`.

| Step | Result |
|---|---|
| `ptoas ... --emit-vpto` | OK |
| `ptoas ... --emit-vpto-llvm-ir` | OK |
| `bisheng ... -c -x ir` → `.o` | **FAIL** — bisheng crashes (exit 139) on the emitted HIVM IR |
