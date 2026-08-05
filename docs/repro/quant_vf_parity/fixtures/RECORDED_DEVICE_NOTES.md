# Recorded on-device results (AscendC vs VMI)

These notes support the **open** feature requests (1–2). Former request 3 is
**solved — not a blocker now** and is kept only as backlog. The compile check in
`scripts/check_vmi_asc_residual.sh` only covers AscendC `bisheng` and VMI
`pto-test-opt` / `ptoas`. On-device comparisons (AscendC result vs VMI result)
were run outside this package.

## Feature request 1 — block index controlling a vector body (**open — primary**)

Minimized VMI that fails `ptoas` → `bisheng` with
`Do not know how to expand the result of this operator!`:

- `current_vmi_block_idx_vf.pto`
- Full dump: `current_vmi_multibuf_expand_full.pto`

AscendC reference: `reference_asc_block_idx_vf.asc`.

While shrinking the dump: dynamic `set_flag` / `wait_flag` alone do **not**
reproduce the expand error; `get_block_idx` feeding a compare that **requires**
a vector body to run does.

## Feature request 2 — dequant double-buffer numerics (**open**)

Comparison: **AscendC result** vs **VMI result** (maximum absolute difference).

| Program | Buffering / strip | Result |
|---------|-------------------|--------|
| Wide schedule (two on-chip faces, K-chunk 512 / 256-lane halves) | `broken_vmi_dequant_dblbuf.ptodsl.py` (full kernel dump) | maxabs ≠ 0 vs AscendC |
| Narrow schedule (two on-chip faces, K-chunk 256) | same algorithm, smaller K face | maxabs = 0 vs AscendC |

Vector bodies that lower on their own (bf16 scale expand + multiply):

- Wide strip: `broken_vmi_dequant_dblbuf.pto` (256 lanes, group 8)
- Narrow strip: `working_vmi_dequant_narrow.pto` (128 lanes, group 4)

## Backlog — former feature request 3 (solved; not a blocker)

**Issue solved. Not a blocker now. Kept as backlog.**

Full-kernel wall-clock (~0.98× at 8192×2048) is in `PERF_FINDINGS.md`.
Strip fixtures compile / lower today but are not an open ask:

- AscendC: `reference_asc_fp32_strip_amax.asc`
- VMI: `current_vmi_fp32_strip_amax.pto`
