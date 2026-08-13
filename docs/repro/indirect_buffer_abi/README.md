# PTOAS performance and ABI issue: indirect GM buffer tables

The portable PTO ABI for an indirect GM buffer table is a GM array of 64-bit
device addresses.  Declare it as `pto.ptr(pto.i64, "gm")`, load an address
with `pto.load_scalar(..., bypass_l1=True)`, and convert it at the use site
with `pto.castptr(addr, pto.ptr(T, "gm"))`.  Nested pointer types and
`pto.load_ptr` are intentionally not part of the API.

`fixtures/pointer_table_abi.py` is the runnable regression for this ABI.  It
loads a dynamic table entry and uses the resulting typed GM pointer in DMA.

The full recurrence's stacked-buffer VMI implementation remains **3.15x
slower** than direct CCE. This reproducer now separates that performance work
from the resolved type-system/ABI concern.

The complete workload is a 72-core, `8192 x 4096`, 10-layer four-lane recurrence. `reference_device.asc` loads runtime addresses from GM tables and uses direct DMA. `stacked_pipeline_vmi.py` still receives dense layer-major buffers and performs wrapper stack/unstack copies. `pointer_table_abi.py` provides the supported address-table spelling.

## Reproduce

Use CANN `9.1.0-beta.3` and the environment containing the matching pinned PTODSL/PTOAS installation. `ACL_DEVICE_ID` is the only device selector.

```bash
source /path/to/cann-9.1.0-beta.3/set_env.sh
ACL_DEVICE_ID=0 bash check.sh compile
ACL_DEVICE_ID=0 bash check.sh benchmark
```

`check.sh compile` builds both runnable libraries and verifies the supported address-table ABI emits `pto.ld_dev`, `pto.castptr`, and GM-to-UB DMA. `check.sh benchmark` launches both through the same ctypes stream ABI, validates both result tensors, and then measures the complete operations.

| Verified device-0 result | Direct CCE us | Stacked VMI us | CCE/VMI |
|---|---:|---:|---:|
| Event median, 256 MB L2 flush | 1484.662 | 4629.356 | 0.3205 |
| FFTS device time | 1457.808 | 4592.626 | 0.3174 |

The direct side receives actual device-address tables; it is not a mock of indirection. The VMI side includes every staging copy required by the current fixed-formal interface. Both layer-input and residual outputs are checked before timing, in float32 with a `0.25` absolute tolerance appropriate for BF16 conversion. The VMI stage intentionally computes all four output lanes before storing any of them: the recurrence reads the old four-lane residual, so an earlier in-place store must not become an input to a later lane.

## Remaining performance work

The current dense workaround is numerically equivalent, but it still stacks
and unstacks the runtime buffer lists. The remaining task is to apply the
supported address-table ABI to the complete persistent VMI recurrence and
approach the direct CCE baseline (`CCE/VMI >= 0.98`).
