# Indirect buffer schedule: direct pointers versus stacked VMI

Some fused kernels consume a runtime list of same-shaped GM buffers. Direct CCE
loads an address from a GM table and DMAs the pointed-to payload. Current
PTODSL can only expose each buffer as a separate formal argument; a generic
implementation must first stack every source and later unstack every result.

This package makes both facts executable:

- `stacked_pipeline_vmi.py` is the complete 72-core, `8192 x 4096`, 10-layer
  stacked-buffer VMI schedule.
- `indirect_api_negative.py` attempts the natural nested-pointer type and is
  required to fail on current PTODSL.
- `reference_device.asc` is the complete address-table kernel with direct DMA;
  `reference_launch.cpp` supplies only its stream-first host ABI.
- `desired_pointer_table.py` states the requested surface.

Requirements are CANN `9.1.0-beta.3`, the pinned PTODSL/PTOAS installation,
and PyTorch with NPU support. Source that CANN release first, or set
`CANN_ENV` to its `set_env.sh`; set `CONDA_ENV` when the active environment is
not the default `cann91_dev`.

```bash
source /path/to/cann-9.1.0-beta.3/set_env.sh
ACL_DEVICE_ID=0 bash check.sh compile
ACL_DEVICE_ID=0 bash check.sh benchmark
```

With `ACL_DEVICE_ID`, benchmark mode compiles and launches the direct CCE
pointer-table kernel and the production-shaped stacked-buffer VMI kernel
through `torch_npu`, checking both paths on the same stream. It also retains
the nested-pointer API rejection test as the ABI-specific part of the issue.

| Verified device-0 measurement | Direct CCE us | Stacked VMI us | CCE/VMI |
|---|---:|---:|---:|
| event median, 256 MB L2 flush, 20 samples | 1485.899 | 4630.336 | 0.3209 |
| FFTS device time, 20 repetitions | 1460.010 | 4594.875 | 0.3177 |

The live harness allocates each layer at the generated block stride and stores
the six table entries as device addresses, so the CCE side is an actual
pointer-table launch. The VMI side materializes the same layer-major buffers
and includes all six stack copies plus the two result unstack copies in the
timed call. Its nonzero-data comparison fails fast
(`layer_input_maxabs=0.1875`, `residual_maxabs=0.06244755`), so it cannot be
used as a valid throughput comparison until the VMI body is repaired. Requested ABI
behavior remains a bounds-checkable load from
`ptr<ptr<T, gm>, gm>` yielding `ptr<T, gm>`, correct memory effects, and host
launcher support for an address-table tensor; the current production-shaped
comparison demonstrates the staging-copy performance penalty while that ABI
is unavailable.
