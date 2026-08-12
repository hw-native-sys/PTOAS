# PTOAS performance issue: FP8 requantization is slower through VMI

The complete VMI requantization operation is **3.19x slower** than the algorithm-equivalent direct CCE operation. The requested outcome is a VMI lowering with no material performance deficit (`CCE/VMI >= 0.98`) for this packed FP8 decode, grouped reduction, and FP8 re-encode pattern.

Both implementations operate on `8064 x 7168` packed FP8 values and per-32 input scales, produce FP8 output and per-32-row float scales, and use 72 cores. `production_reference.asc` fuses decode, reduction, scale generation, and conversion in one GM-to-UB-to-GM CCE kernel. The VMI workload uses eight static 1024-row unpack launches followed by the complete requant launch. That composition is intentionally retained: it is the current VMI plan, and hand-fusing it in Python would not demonstrate that PTOAS can generate a correct or efficient fused plan.

## Reproduce

Use CANN `9.1.0-beta.3` and the environment containing the matching pinned PTODSL/PTOAS installation. `ACL_DEVICE_ID` is the only device selector.

```bash
source /path/to/cann-9.1.0-beta.3/set_env.sh
ACL_DEVICE_ID=0 bash check.sh compile
ACL_DEVICE_ID=0 bash check.sh benchmark
```

The harness uses the same ctypes stream ABI for both sides, validates output and scales over the defined `8064`-row extent, then times the complete callable. The final VMI strip has 128 inactive padded rows; it is zero-padded because the static kernel reads it, and is outside the output contract. FFTS device time is primary: it excludes host launch cost and sums all VMI stages and adapters required by the current plan.

| Verified device-0 result | Direct CCE us | VMI us | CCE/VMI |
|---|---:|---:|---:|
| Cold-L2 event median | 47.041 | 302.386 | 0.1556 |
| FFTS device time | 45.430 | 145.072 | 0.3132 |

The numerical comparison passes before timing. Cold-L2 timing is retained as a sanity check; FFTS is the accepted comparison. Absolute latency varies with device load, so a fresh run is authoritative.

## Why this is a PTOAS issue

The direct CCE fixture demonstrates that a fast, fused implementation of this algorithm and shape is reachable on the device. The VMI fixture preserves the same decode, per-group maximum, scale calculation, and re-encode result. Its multiple strip launches and full BF16 intermediate are the current lowering/scheduling limitation, not a voluntary benchmark choice.

Please add or improve fused VMI lowering across FP8 decode, grouped reduction, and requantization; keep intermediates in UB; and schedule the full operation without static strip launches or extra materialization. A VMI source manually rewritten to look like the CCE assembly would not establish that PTOAS can lower the intended composition, which is why the current plan remains visible.
