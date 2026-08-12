# PTOAS correctness and ABI issue: indirect GM buffer tables

PTO DSL cannot express a typed GM pointer table, and the available stacked VMI workaround fails numerical validation for the full recurrence. This is first a **correctness and ABI blocker**, not an accepted VMI throughput comparison. The direct CCE implementation is correct and demonstrates that the intended address-table algorithm has a fast device implementation.

The complete workload is a 72-core, `8192 x 4096`, 10-layer four-lane recurrence. `reference_device.asc` loads runtime addresses from GM tables and uses direct DMA. `stacked_pipeline_vmi.py` must instead receive dense layer-major buffers and the wrapper stack/unstack copies. The natural DSL form is retained in `indirect_api_negative.py`; it fails before lowering with `pto.ptr(...) does not support element type !pto.ptr<f32, gm>`.

## Reproduce

Use CANN `9.1.0-beta.3` and the environment containing the matching pinned PTODSL/PTOAS installation. `ACL_DEVICE_ID` is the only device selector.

```bash
source /path/to/cann-9.1.0-beta.3/set_env.sh
ACL_DEVICE_ID=0 bash check.sh compile
ACL_DEVICE_ID=0 bash check.sh benchmark
```

`check.sh compile` builds both runnable libraries and confirms the expected pointer-table API rejection. `check.sh benchmark` launches both through the same ctypes stream ABI, then intentionally exits nonzero when the full VMI recurrence disagrees with CCE. That failure is the expected current result.

| Diagnostic device-0 measurement | Direct CCE us | Stacked VMI us | CCE/VMI |
|---|---:|---:|---:|
| Event median, 256 MB L2 flush | 1485.899 | 4630.336 | 0.3209 |
| FFTS device time | 1460.010 | 4594.875 | 0.3177 |

The direct side receives actual device-address tables; it is not a mock of indirection. The VMI side includes every staging copy required by the current fixed-formal interface. Its nonzero-data check fails with approximately `layer_input_maxabs=0.1875` and `residual_maxabs=0.06244755`, so the displayed VMI time is diagnostic evidence of the workaround cost only, not a valid performance result.

## Requested PTOAS work

Add a typed GM pointer-table surface: a bounds-checkable load from `ptr<ptr<T, gm>, gm>` yielding `ptr<T, gm>`, correct memory effects, DMA from a dynamically loaded GM pointer, and matching host-launcher ABI support. Repair the full VMI recurrence before accepting a throughput result. Once correct, the implementation should avoid stack/unstack materialization and aim for no material deficit versus the direct CCE baseline (`CCE/VMI >= 0.98`).
