# Feature request 3 — AscendC vs VMI emit / pipe notes

## Primary evidence is the full kernel

Wall-clock gap and pipe diagnosis live in
[`PERF_FINDINGS.md`](PERF_FINDINGS.md) for
`reference_asc_fp32_block_quant.asc` vs
`current_vmi_fp32_block_quant_8192x2048.ptodsl.py`.

At **8192×2048**, msopprof Task Duration is ~27.0 µs (AscendC) vs ~31.5 µs (VMI)
→ AscendC/VMI ≈ **0.86**. PipeUtilization: VMI higher `aiv_vec_ratio` /
`aiv_scalar_ratio` and UB vector read BW; AscendC higher `aiv_mte3_ratio`.

Strip fixtures below are **VF fragments only** — they do not reproduce the gap.

## Strip fragment (optional compile / emit detail)

| Side | Fixture | Role |
|------|---------|------|
| AscendC | `reference_asc_fp32_strip_amax.asc` | Dense MicroAPI abs-max strip |
| VMI | `current_vmi_fp32_strip_amax.pto` | abs + `vcmax` with `group = 1` |

### Status today

- AscendC strip compiles with `bisheng`.
- VMI strip lowers with `pto-test-opt` (check-script pass list).
- The one-element reduce+store path is already legal; closing FR3 does **not**
  wait on further strip legalization.
- Optional: lower the strip and list extra VMI moves vs AscendC MicroAPI if that
  helps motivate a denser emit. That alone is not enough — the full-kernel
  numbers in `PERF_FINDINGS.md` are the claim.

### How to inspect the strip emit

1. Lower `current_vmi_fp32_strip_amax.pto` (check-script passes, or wrap the
   body in `pto.vecscope` and run `ptoas --emit-vpto`).
2. Compile `reference_asc_fp32_strip_amax.asc` and inspect its vector / memory
   sequence.
3. List any extra VMI moves, spills, or layout reshapes relative to AscendC.

## Strip emit findings

_(optional detail — none required to keep FR3 open; see PERF_FINDINGS.md)_
