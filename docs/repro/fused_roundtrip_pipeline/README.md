# PTOAS performance issue: fused FP8 round-trip is slower through VMI

The VMI implementation of this fused BF16-to-FP8-to-BF16 round trip is **25% slower** than the algorithm-equivalent CCE implementation. The requested outcome is VMI no slower than the CCE baseline (`CCE/VMI >= 0.98`) for this single-kernel, persistent vector pattern.

Both paths process the same contiguous `8192 x 2048` BF16 tensor with the same 72-core grid and three-stage persistent schedule. The CCE fixture expresses SIMD/DMA overlap directly. [`full_roundtrip_vmi.py`](fixtures/full_roundtrip_vmi.py) retains the complete VMI group reduction, scale encoding, FP8 conversion, reverse scaling, and DMA pipeline. It is not a small-tile proxy.

## Reproduce

Use CANN `9.1.0-beta.3` and the environment containing the matching pinned PTODSL/PTOAS installation. `ACL_DEVICE_ID` is the only device selector.

```bash
source /path/to/cann-9.1.0-beta.3/set_env.sh
ACL_DEVICE_ID=0 bash check.sh compile
ACL_DEVICE_ID=0 bash check.sh benchmark
```

The benchmark builds both libraries, launches them with the same ctypes stream ABI, and checks finite output plus CCE/VMI agreement within one FP8 quantization step. It reports cold-L2 event timing and device-only FFTS time; FFTS aggregates AIC records across 30 repetitions, so it measures device execution rather than the launch floor.

| Verified device-0 result | Direct CCE us | VMI us | CCE/VMI |
|---|---:|---:|---:|
| Cold-L2 event median | 30.913 | 31.980 | 0.9666 |
| FFTS device time | 21.555 | 26.990 | 0.7986 |

Cold-L2 timing is diagnostic only. FFTS is the accepted result; absolute latency varies with device load, so rerunning the checked-in command is authoritative.

## Why this is a PTOAS issue

The CCE result proves a high-throughput implementation is reachable for the same algorithm, shape, grid, and persistent pipeline. Unlike the other reports, this case already uses one full VMI kernel, so the deficit cannot be explained by Python composition or an added adapter. VMI retains the complete algorithm but currently lowers/schedules the grouped reduction, conversion, and vector pipeline less efficiently than CCE.

Please improve VMI scheduling and lowering for grouped reduction, conversion, and SIMD/DMA pipeline overlap until it reaches the CCE baseline. Replacing the fixture with hand-written CCE-like instructions would avoid, rather than exercise, the VMI compiler path that needs improvement.
