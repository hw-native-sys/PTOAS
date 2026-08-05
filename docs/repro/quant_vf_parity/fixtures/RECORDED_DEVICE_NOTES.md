# Recorded on-device results (AscendC vs VMI)

These notes support feature requests 1–3. The compile check in
`scripts/check_vmi_asc_residual.sh` only covers AscendC `bisheng` and VMI
`pto-test-opt` / `ptoas`. On-device comparisons (AscendC result vs VMI result)
were run outside this package.

## Feature request 1 — block index controlling a vector body

Minimized VMI that fails `ptoas` → `bisheng` with
`Do not know how to expand the result of this operator!`:

- `current_vmi_block_idx_vf.pto`
- Full dump: `current_vmi_multibuf_expand_full.pto`

AscendC reference: `reference_asc_block_idx_vf.asc`.

While shrinking the dump: dynamic `set_flag` / `wait_flag` alone do **not**
reproduce the expand error; `get_block_idx` feeding a compare that **requires**
a vector body to run does.

## Feature request 2 — dequant double-buffer numerics

Comparison: **AscendC result** vs **VMI result** (maximum absolute difference).

| Program | Buffering / strip | Result |
|---------|-------------------|--------|
| Wide schedule (two on-chip faces, K-chunk 512 / 256-lane halves) | `broken_vmi_dequant_dblbuf.ptodsl.py` (full kernel dump) | maxabs ≠ 0 vs AscendC |
| Narrow schedule (two on-chip faces, K-chunk 256) | same algorithm, smaller K face | maxabs = 0 vs AscendC |

Vector bodies that lower on their own (bf16 scale expand + multiply):

- Wide strip: `broken_vmi_dequant_dblbuf.pto` (256 lanes, group 8)
- Narrow strip: `working_vmi_dequant_narrow.pto` (128 lanes, group 4)

## Feature request 3 — FP32 strip abs-max performance

Both fixtures compile / lower today:

- AscendC: `reference_asc_fp32_strip_amax.asc`
- VMI: `current_vmi_fp32_strip_amax.pto`

Wall-clock and generated-code density vs AscendC are tracked in
`emit_compare_note.md`. A PTOAS compiler change is only justified after that
note records a concrete AscendC-vs-VMI emit difference (extra memory ops or
layout work on the VMI side).
