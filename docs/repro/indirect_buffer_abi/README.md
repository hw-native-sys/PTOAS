# Request: indirect GM buffer arrays in the PTODSL launch ABI

## Summary

Some fused kernels iterate over a runtime list of same-shaped global-memory
buffers. A direct CCE kernel accepts a GM table of addresses, loads one address
per iteration, casts it to a typed GM pointer, and performs DMA directly. The
current PTODSL public launch surface accepts typed GM pointers but has no clean,
portable way to represent and dereference an array of GM pointers.

The only current workaround is to copy every input into a stacked workspace,
run the kernel, then copy every output back. For memory-bound multi-stage
elementwise chains this adds enough traffic to make parity impossible and makes
the high-level code more complicated than the direct implementation.

## Reproduce

```bash
bash docs/repro/indirect_buffer_abi/check.sh
```

`fixtures/fixed_arguments.py` is the compileable control with two explicit GM
arguments. `fixtures/desired_pointer_table.py` documents the desired generic
surface. It is intentionally not executable today because PTODSL has no pointer
element type / pointer-table load operation to express it safely.

## Requested behavior

Add a typed, bounds-checkable way to pass a GM pointer table and load an entry as
`!pto.ptr<T, gm>` inside a kernel, with correct memory effects and codegen. The
host launcher must accept an address-table tensor without requiring a copy into
a stacked data tensor. Success means the indirect form has no extra bulk data
copies and reaches at least 98% of the equivalent fixed/direct CCE kernel.
