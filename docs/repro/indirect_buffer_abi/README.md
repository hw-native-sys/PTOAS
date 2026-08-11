# Request: typed arrays of GM pointers in the PTODSL launch ABI

Some fused kernels consume a runtime list of same-shaped GM buffers. Direct CCE
loads an address from a GM table and DMAs the pointed-to payload. Current
PTODSL can only expose each buffer as a separate formal argument; a generic
implementation must first stack every source and later unstack every result.

This package makes both facts executable:

- `fixed_arguments.py` is the compileable fixed-argument control.
- `indirect_api_negative.py` attempts the natural nested-pointer type and is
  required to fail on current PTODSL.
- `reference_cce.cpp` is a complete address-table kernel with direct DMA.
- `desired_pointer_table.py` states the requested surface.

Run `bash check.sh compile`, `bash check.sh benchmark`, or `bash check.sh`.
With `ACL_DEVICE_ID`, benchmark mode compiles and launches
the fixed-argument workaround through `torch_npu`, checks its copy result, and
also retains the API rejection test for nested GM pointers.

| Representative case | Direct us | Stacked workaround us | Direct/workaround | extra copied bytes |
|---|---:|---:|---:|---:|
| payload 2560, 10 stages | 925.5155 | 3887.6199 | 0.2381 | 2,481,848,320 |
| payload 4096, 10 stages | 1348.9992 | 4203.7893 | 0.3209 | 3,966,631,936 |

Requested behavior: a bounds-checkable load from
`ptr<ptr<T, gm>, gm>` yielding `ptr<T, gm>`, correct memory effects, and host
launcher support for an address-table tensor. It must remove the bulk staging
copies and reach `direct_us / indirect_us >= 0.98`.
