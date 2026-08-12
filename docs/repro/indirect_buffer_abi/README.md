# PTOAS performance and ABI issue: indirect GM buffer tables

PTO DSL cannot express a typed GM pointer table. Its correct stacked-buffer VMI workaround is **3.15x slower** than direct CCE for this full recurrence. The requested outcome is direct pointer-table support and a VMI implementation no slower than the CCE baseline (`CCE/VMI >= 0.98`).

The complete workload is a 72-core, `8192 x 4096`, 10-layer four-lane recurrence. `reference_device.asc` loads runtime addresses from GM tables and uses direct DMA. `stacked_pipeline_vmi.py` must instead receive dense layer-major buffers and the wrapper stack/unstack copies. The natural DSL form is retained in `indirect_api_negative.py`; it fails before lowering with `pto.ptr(...) does not support element type !pto.ptr<f32, gm>`.

## Reproduce

Use CANN `9.1.0-beta.3` and the environment containing the matching pinned PTODSL/PTOAS installation. `ACL_DEVICE_ID` is the only device selector.

```bash
source /path/to/cann-9.1.0-beta.3/set_env.sh
ACL_DEVICE_ID=0 bash check.sh compile
ACL_DEVICE_ID=0 bash check.sh benchmark
```

`check.sh compile` builds both runnable libraries and confirms the expected pointer-table API rejection. `check.sh benchmark` launches both through the same ctypes stream ABI, validates both result tensors, and then measures the complete operations.

| Verified device-0 result | Direct CCE us | Stacked VMI us | CCE/VMI |
|---|---:|---:|---:|
| Event median, 256 MB L2 flush | 1484.662 | 4629.356 | 0.3205 |
| FFTS device time | 1457.808 | 4592.626 | 0.3174 |

The direct side receives actual device-address tables; it is not a mock of indirection. The VMI side includes every staging copy required by the current fixed-formal interface. Both layer-input and residual outputs are checked before timing, in float32 with a `0.25` absolute tolerance appropriate for BF16 conversion. The VMI stage intentionally computes all four output lanes before storing any of them: the recurrence reads the old four-lane residual, so an earlier in-place store must not become an input to a later lane.

## Requested PTOAS work

Add a typed GM pointer-table surface: a bounds-checkable load from `ptr<ptr<T, gm>, gm>` yielding `ptr<T, gm>`, correct memory effects, DMA from a dynamically loaded GM pointer, and matching host-launcher ABI support. The current dense workaround is retained because it is numerically equivalent, but it must stack and unstack all runtime buffer lists. PTOAS should remove that materialization and approach the direct CCE baseline (`CCE/VMI >= 0.98`).
