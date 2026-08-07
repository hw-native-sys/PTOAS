# Compact f32 group-store device reproducer

This package isolates a PTOAS layout/lowering defect: a logical scalar carried
in a 64-lane broadcast f32 vector is not converted to compact group-slot layout
before a `group=1` store.

The intended operation is:

```python
pto.vmi.vstore(value, destination, offset, group=1, stride=1)
```

The equivalent native ASC/CCE instruction is `vsts(..., ONEPT_B32, ...)`.
Both should write lane 0 to one f32 element. Current PTOAS instead keeps the
broadcast result in contiguous layout and emits a dense, full-mask `vsts`
without `1PT_B32`; launching that code raises device error 507035.

## Algorithm

All three fixtures implement the same neutral computation for 24 values:

```text
for i in 0..23:
    lanes = broadcast(input[i])
    result = lanes * 1.25 + 1.0
    output[i] = result[0]
```

They run as independent processes because a vector-core fault can poison the
current device context.

| Fixture | Purpose | Expected result |
|---|---|---|
| `fixtures/reference_compact_store.asc` | Native ASC/CCE `ONEPT_B32` baseline | Numeric pass |
| `fixtures/desired_compact_store.py` | Direct VMI `group=1, stride=1` after broadcast arithmetic | Currently mislowers to dense `vsts` and faults with 507035 |
| `fixtures/workaround_padded_store.py` | 64-lane VMI stores followed by scalar extraction | Numeric pass, but non-optimal |

The workaround allocates 24 × 64 f32 values (6144 bytes) instead of 24 f32
values (96 bytes), then launches a separate SIMT extraction phase. It is proof
that broadcast arithmetic and normal vector stores work, not a suitable
replacement for compact group-store support.

## Run

The scripts default to CANN 9.1.0.beta3 and the `cann91_dev` conda environment.
They use only local source/build artifacts.

```bash
cd docs/repro/compact_f32_store
./scripts/check_compile.sh
./scripts/run_device.sh 0
```

`run_device.sh` uses `task-submit --device` when available. Generated objects,
MLIR and logs are written below the gitignored `outputs/` directory. The
compile check records the dense-store mislowering in emitted VPTO. The device
case is classified as a reproduced bug only when its isolated log contains
`507035`; any other failure is reported as unexpected. A numeric pass is
reported honestly as a fixed issue.

If this worktree has not been built in place, point `PTOAS_BIN` at a PTOAS
binary built from the same commit. The Python path is intentionally rooted at
this source worktree so the fixture itself remains PTODSL/PTOAS-only.

## Relationship to nearby reports

- `narrow_lane_store` is directly related but not identical. Its fault was on
  a one-slot load lowered to a 32-byte-aligned block load. Commit `823ec908`
  changed that load to `BRC_B32`, and this branch contains the fix. Here the
  load is already explicitly `BRC_B32`; the remaining defect is propagation
  from a contiguous broadcast result into a compact `group=1` store.
- `pad_brc` concerns legalization of grouped broadcast values in registers.
  The desired fixture here already lowers; its decisive test is device launch.
- `quant_vf_parity` covers quantization/vector-fusion parity and does not
  report this compact f32 store pattern.
- `l8_widen` covers widening/layout behavior and is not the same operation.

See [`FEATURE_REQUEST.md`](FEATURE_REQUEST.md) for the precise acceptance
criteria and [`recorded/DEVICE_RESULTS.md`](recorded/DEVICE_RESULTS.md) for the
result recorded on the development host.
