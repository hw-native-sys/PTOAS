# Feature request — AscendC vs VMI residuals after bf16 abs / G=4 broadcast

Audience: PTOAS maintainers.

Background already landed elsewhere (`zjw/bf16_vmi_fixes`): bf16 `vabs` via
sign-bit clear, and bf16 group-4 `group_broadcast`. This package keeps open
residuals only. Each ask pairs an AscendC program that works with a nearly
equivalent VMI/PTO program that is slower, wrong, or fails to emit.

Pipe sync ops in the VMI dumps are already present as `set_flag` /
`wait_flag` (and dyn forms). Turning on PTOAS InsertSync is not the request.

---

## Ask 1 — `get_block_idx` bound check around VF fails ptoas → bisheng

### Problem

Multi-buffer kernels often gate a vector-face body on a bound derived from
`get_block_idx` (for example `block_idx / 8 < N`). AscendC expresses that with
`GetBlockIdx` and compiles with `bisheng`.

On the VMI/PTO path, the same pattern dies in ptoas → bisheng with:

```text
Do not know how to expand the result of this operator!
```

Minimization notes (important):

- `set_flag_dyn` / `wait_flag_dyn` alone **do not** reproduce the expand failure.
- `get_block_idx` used only for MTE addressing can succeed.
- The failure appears when `get_block_idx` feeds a compare that guards a VF
  (`create_mask` / `vload` / `vstore`).

### What AscendC already does

See [`fixtures/reference_asc_block_idx_vf.asc`](fixtures/reference_asc_block_idx_vf.asc).

### What VMI does today (broken emit)

| Fixture | Role |
|---------|------|
| [`current_vmi_block_idx_vf.pto`](fixtures/current_vmi_block_idx_vf.pto) | Minimized repro (expand FAIL) |
| [`current_vmi_multibuf_expand_full.pto`](fixtures/current_vmi_multibuf_expand_full.pto) | Full multi-buffer dump that first showed the failure |

### What we need

`get_block_idx` (and the integer ops that form a tile bound from it) must expand
through VPTO / bisheng when they control a VF region, the same way they already
do for plain MTE addressing.

### Criteria

1. AscendC reference still compiles with `bisheng`.
2. After the fix, `current_vmi_block_idx_vf.pto` completes `ptoas` device
   emission without the expand error above.
3. Until then, the check script records the current FAIL with that error string.

### Non-goals

- Asking PTOAS to invent the multi-buffer schedule
- Treating InsertSync enablement as the fix
- Filing dyn pipe-event ops as the primary bug (they are present in the full
  dump but are not sufficient to reproduce expand)

---

## Ask 2 — Double-buffered dequant wrong numerics vs AscendC

### Problem

Dequant of e4m3 with UE8M0 scales (expand scale across a strip, multiply, store)
is correct on AscendC under multi-buffer MTE+V. On VMI, the same algorithm with
**two UB faces and a wide K-chunk (512 / 256-lane halves)** disagrees with
AscendC (maxabs ≠ 0). The **narrower** schedule (smaller K face / 128-lane
strips) matches AscendC.

So this is not “VMI cannot dequant”. It is shape- and buffering-sensitive wrong
results on the VMI legalization / emission path. The checked-in `.pto` bodies
are the bf16 expand+mul strip (legal contiguous path); the full double-buffer
kernel dump is the `.ptodsl.py` used for the device mismatch.

### What AscendC already does

See [`fixtures/reference_asc_dequant_dblbuf.asc`](fixtures/reference_asc_dequant_dblbuf.asc).

### What VMI does today

| Fixture | Role |
|---------|------|
| [`broken_vmi_dequant_dblbuf.pto`](fixtures/broken_vmi_dequant_dblbuf.pto) | Wide-strip VF body (lowers) |
| [`working_vmi_dequant_narrow.pto`](fixtures/working_vmi_dequant_narrow.pto) | Narrow-strip VF body (lowers; matched AscendC on device) |
| [`broken_vmi_dequant_dblbuf.ptodsl.py`](fixtures/broken_vmi_dequant_dblbuf.ptodsl.py) | Full double-buffer kernel dump used for the device mismatch |

Recorded numbers: [`fixtures/RECORDED_DEVICE_NOTES.md`](fixtures/RECORDED_DEVICE_NOTES.md).

### What we need

The wide, two-face schedule must produce the same results as AscendC (and as the
narrow VMI twin) for this dequant algorithm — fix in VMI layout / `vmi-to-vpto` /
emission around multi-buffer, not by forcing every kernel onto the narrow shape.

### Criteria

1. Wide and narrow VF bodies keep lowering with `pto-test-opt`.
2. After the fix, the wide full-kernel schedule matches AscendC (maxabs = 0) on
   the recorded shape family in `RECORDED_DEVICE_NOTES.md`.

### Non-goals

- Product schedule knobs presented as the PTOAS fix
- Claiming the narrow path is unacceptable forever (it is the control that works)

---

## Ask 3 — FP32 strip reduce/store density (emit evidence required)

### Problem

After `vcmax` / `vstore` with `group = 1` already legalizes to `1PT_B32`, a
representative FP32 strip abs-amax + scale store is still slower on VMI than a
nearly equivalent AscendC MicroAPI path in product measurements. That alone is
not enough to demand a PTOAS change.

### What AscendC already does

See [`fixtures/reference_asc_fp32_strip_amax.asc`](fixtures/reference_asc_fp32_strip_amax.asc).

### What VMI does today (works)

See [`fixtures/current_vmi_fp32_strip_amax.pto`](fixtures/current_vmi_fp32_strip_amax.pto).

### What we need

Only if emit A/B shows redundant MTE, layout reshapes, or extra moves on the VMI
side relative to AscendC: remove that tax in VMI layout / emission. If emit looks
comparable, this is **not** a PTOAS legalization bug.

Track progress in [`fixtures/emit_compare_note.md`](fixtures/emit_compare_note.md).

### Criteria

1. VMI fixture continues to lower (`group = 1` path stays valid).
2. A concrete emit diff is attached before accepting a PTOAS code change for
   this ask.

### Non-goals

- Wall-time ratios as the sole proof
- Re-opening pad-32 + `dist_mode="brc"` as the primary ask
