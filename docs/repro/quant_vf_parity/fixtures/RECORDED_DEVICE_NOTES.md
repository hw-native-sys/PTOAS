# Recorded device results (AscendC vs VMI)

These notes support Ask 1–3. Compile checks in `scripts/check_vmi_asc_residual.sh`
cover AscendC `bisheng` and VMI `pto-test-opt` / `ptoas` only. Device A/B was run
outside this package; re-run commands are listed for maintainers with NPU access.

## Ask 1 — block-idx VF expand

Minimized VMI that fails `ptoas` → `bisheng` with
`Do not know how to expand the result of this operator!`:

- `current_vmi_block_idx_vf.pto`
- Full dump: `current_vmi_multibuf_expand_full.pto`

AscendC baseline: `reference_asc_block_idx_vf.asc`.

Minimization finding: dyn `set_flag`/`wait_flag` alone do **not** reproduce
expand; `get_block_idx` feeding a compare that guards VF does.

## Ask 2 — dequant double-buffer numerics

| Program | Buffering / strip | Result vs AscendC |
|---------|-------------------|-------------------|
| Wide schedule (two UB faces, K-chunk 512 / 256-lane halves) | `broken_vmi_dequant_dblbuf.ptodsl.py` (full kernel dump) | **maxabs ≠ 0** |
| Narrow schedule (two UB faces, K-chunk 256) | same algorithm, smaller K face | **maxabs = 0** |

VF bodies that lower in isolation (bf16 `vbrc` expand + `vmul`):

- Wide strip: `broken_vmi_dequant_dblbuf.pto` (256 lanes, G=8)
- Narrow strip: `working_vmi_dequant_narrow.pto` (128 lanes, G=4)

## Ask 3 — FP32 strip density

VMI `group=1` reduce/store already legalizes (`current_vmi_fp32_strip_amax.pto`).
Wall-time and emit density comparison vs AscendC MicroAPI is tracked in
`emit_compare_note.md`. This is **not** filed as a missing legalization until
emit A/B shows redundant MTE or layout on the VMI side.
