# Request: typed arrays of GM pointers in the PTODSL launch ABI

Some fused kernels consume a runtime list of same-shaped GM buffers. Direct CCE
loads an address from a GM table and DMAs the pointed-to payload. Current
PTODSL can only expose each buffer as a separate formal argument; a generic
implementation must first stack every source and later unstack every result.

This package makes both facts executable:

- `fixed_arguments.py` is the compileable fixed-argument control.
- `indirect_api_negative.py` attempts the natural nested-pointer type and is
  required to fail on current PTODSL.
- `reference_device.asc` is the complete address-table kernel with direct DMA;
  `reference_launch.cpp` supplies only its stream-first host ABI.
- `desired_pointer_table.py` states the requested surface.

Run `bash check.sh compile`, `bash check.sh benchmark`, or `bash check.sh`.
With `ACL_DEVICE_ID`, benchmark mode compiles and launches
the direct CCE pointer-table kernel and the fixed-argument VMI control through
`torch_npu`, checks both host goldens, and retains the API rejection test.

| Verified device-0 event median | Direct CCE us | Fixed-argument VMI us | CCE/VMI |
|---|---:|---:|---:|
| 10 transfers, 64-launch synchronized batches | see live output | see live output | per-launch |

The live harness allocates each layer at the generated block stride and stores
the six table entries as device addresses, so this is an actual pointer-table
launch rather than a host-only ABI sketch. The VMI value is an ABI-only control, not the full pointer-table workload. Requested behavior: a
bounds-checkable load from
`ptr<ptr<T, gm>, gm>` yielding `ptr<T, gm>`, correct memory effects, and host
launcher support for an address-table tensor. It must remove the bulk staging
copies and reach `direct_us / indirect_us >= 0.98`.
