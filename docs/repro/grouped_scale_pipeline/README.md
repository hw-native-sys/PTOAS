# PTOAS performance issue: grouped scale and FP8 conversion

The VMI implementation of this complete grouped reduction, scale, and FP8 conversion is **3.66x slower** than the algorithm-equivalent direct CCE implementation. The requested outcome is VMI no slower than the CCE baseline (`CCE/VMI >= 0.98`).

Both fixtures process the same ragged `8001 x 16384` BF16 tensor with 72 cores, double-buffered UB, group-of-256 maxima, broadcast scale application, and FP8 conversion. `production_group_cce.asc` is a direct single-kernel CCE implementation. `production_group_vmi.py` is the corresponding complete VMI schedule, including the static `8032`-row adapter required by the current specialization. This is deliberately a full device workload, not a small-tile proxy.

## Reproduce

Use CANN `9.1.0-beta.3` and the environment containing the matching PTODSL/PTOAS installation. `ACL_DEVICE_ID` is the only device selector.

```bash
source /path/to/cann-9.1.0-beta.3/set_env.sh
ACL_DEVICE_ID=0 bash check.sh compile
ACL_DEVICE_ID=0 bash check.sh benchmark
```

The benchmark builds both shared libraries, launches both through the same ctypes stream ABI, validates the outputs, and writes generated artifacts under `outputs/`. Cold-L2 event timing is a sanity check; device-only FFTS is the accepted comparison because it excludes Python and launch overhead and includes every device operation required by each path.

| Verified device-0 result | Direct CCE us | VMI us | CCE/VMI |
|---|---:|---:|---:|
| Cold-L2 event median | 183.656 | 494.581 | 0.3713 |
| FFTS device time | 134.517 | 492.079 | 0.2734 |

Absolute latency varies with device load; rerunning the checked-in command is authoritative. The numerical comparison passes before timing.

## Why this is a PTOAS issue

CCE proves that this exact shape and algorithm have a high-throughput, single-72-core implementation on the device. The VMI fixture retains the same reduction, conversion, tiling, and pipeline intent. Its padded input/output copies are not an artificial penalty: the current VMI static specialization cannot directly own the ragged extent, so they are part of the deployed VMI operation and are included fairly in FFTS.

Please improve VMI lowering/scheduling for this pattern: preserve group reductions and tail predicates across broadcast and conversion; avoid UB spill/reload and layout materialization; and support a direct ragged-tile, single-launch pipeline. Replacing the VMI fixture with a hand-written CCE-like implementation would hide the missing lowering capability rather than test it.
