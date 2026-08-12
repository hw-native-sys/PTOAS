# VMI performance issue: BF16 widen and multiply

This standalone reproducer compares direct CCE and PTODSL implementations of one vector operation. Both paths use BF16 inputs, an FP32 output, 64-lane vector chunks, and the same 64-core direct grid.

```text
fp32_out[i] = float(bf16_a[i]) * float(bf16_b[i])
```

The CCE source is [`bf16_widen_multiply_cce.asc`](fixtures/bf16_widen_multiply_cce.asc). The VMI source is [`bf16_widen_multiply_vmi.py`](fixtures/bf16_widen_multiply_vmi.py) and imports only `ptodsl.pto`. `benchmark.py` compiles, links, launches, checks both outputs against `float(a) * float(b)`, and event-times the same ctypes stream ABI.

## Reproduce

Use CANN `9.1.0-beta.3` and PTOAS revision `83e3799f`.

```bash
source /path/to/cann/set_env.sh
ACL_DEVICE_ID=0 bash check.sh compile
ACL_DEVICE_ID=0 bash check.sh benchmark
```

## Observed result

Production measurements for the same primitive were:

| Elements | Direct CCE us | VMI us | CCE/VMI |
|---:|---:|---:|---:|
| 8192 | 1.421 | 1.423 | 0.999 |
| 28672 | 1.488 | 1.455 | 1.023 |
| 131072 | 1.835 | 2.007 | 0.915 |
| 262144 | 2.101 | 2.207 | 0.952 |

The requested target is `CCE/VMI >= 0.98` for the larger contiguous cases.
