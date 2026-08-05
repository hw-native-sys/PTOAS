# Feature requests — remaining quant gaps on VMI after bf16 abs / group-4 broadcast

Audience: PTOAS maintainers reviewing this reproducer package.

On top of [PR #1145](https://github.com/hw-native-sys/PTOAS/pull/1145)
(bf16 absolute value via clearing the sign bit, and bf16 group-4 broadcast),
several issues are still unfixed. Each section below pairs a working AscendC
program with a nearly equivalent VMI/PTO program that fails to compile, produces
wrong results, or is slower for a reason we want PTOAS to investigate.

Pipe synchronization (`set_flag` / `wait_flag`, including dynamic forms) is
already present in the VMI dumps. Enabling InsertSync is not what these
requests ask for.

---

## Feature request 1 — `get_block_idx` used to decide whether a vector body runs fails in ptoas → bisheng

### Problem

Multi-buffer kernels often take the hardware block index, form a simple bound
(for example `block_idx / 8 < N`), and only then run a short vector load/store
body. AscendC does this with `GetBlockIdx` and compiles with `bisheng`.

The same pattern on the VMI/PTO path dies during device emission with:

```text
Do not know how to expand the result of this operator!
```

What we checked while shrinking the dump:

- Dynamic `set_flag_dyn` / `wait_flag_dyn` alone do **not** cause this error.
- Using `get_block_idx` only to compute memory addresses for MTE can succeed.
- The failure shows up when `get_block_idx` feeds a compare that **requires** a
  vector body to run (`create_mask` / `vload` / `vstore` under that `scf.if`).

### Working AscendC reference

[`fixtures/reference_asc_block_idx_vf.asc`](fixtures/reference_asc_block_idx_vf.asc)
— `bisheng` compiles this today.

### Broken VMI / PTO today

| Fixture | What it is |
|---------|------------|
| [`current_vmi_block_idx_vf.pto`](fixtures/current_vmi_block_idx_vf.pto) | Smallest program that still hits the expand error |
| [`current_vmi_multibuf_expand_full.pto`](fixtures/current_vmi_multibuf_expand_full.pto) | Larger multi-buffer dump where the failure was first seen |

### Desired behavior

`get_block_idx` and the integer arithmetic that builds a tile bound from it must
lower through ptoas and bisheng when that bound **controls** a vector region —
the same way those ops already work when they only feed MTE addressing.

After a fix:

1. The AscendC reference still compiles with `bisheng`.
2. `current_vmi_block_idx_vf.pto` finishes `ptoas` device emission without the
   expand error above.
3. Until then, `scripts/check_vmi_asc_residual.sh` should keep recording FAIL
   and match that error string.

### Out of scope

- Having PTOAS invent the multi-buffer schedule
- Treating InsertSync as the fix
- Treating dynamic pipe-event ops as the main bug (they appear in the full dump
  but are not enough to reproduce the expand failure)

---

## Feature request 2 — Wide double-buffered dequant disagrees with AscendC

### Problem

Dequantizing e4m3 data with UE8M0 scales (expand the scale across a strip,
multiply, store) is correct on AscendC with multi-buffer copy + vector work.
On VMI, the **same algorithm** with two on-chip buffer faces and a **wide**
K-chunk (512 elements, worked as 256-lane halves) disagrees with AscendC
(maximum absolute difference ≠ 0). The **narrower** schedule (smaller K face /
128-lane strips) matches AscendC.

So VMI can dequant; the bug is sensitive to strip width and buffering depth on
the VMI lower / emit path.

Checked-in pieces:

- Small `.pto` bodies: bf16 scale expand + multiply (legal contiguous path).
- Full double-buffer kernel dump: `.ptodsl.py`, used for the on-device mismatch.

### Working AscendC reference

[`fixtures/reference_asc_dequant_dblbuf.asc`](fixtures/reference_asc_dequant_dblbuf.asc)
— `bisheng` compiles this today.

### VMI today

| Fixture | What it is |
|---------|------------|
| [`broken_vmi_dequant_dblbuf.pto`](fixtures/broken_vmi_dequant_dblbuf.pto) | Wide-strip vector body (lowers; part of the failing schedule) |
| [`working_vmi_dequant_narrow.pto`](fixtures/working_vmi_dequant_narrow.pto) | Narrow-strip vector body (lowers; matched AscendC on device) |
| [`broken_vmi_dequant_dblbuf.ptodsl.py`](fixtures/broken_vmi_dequant_dblbuf.ptodsl.py) | Full double-buffer kernel dump used for the on-device mismatch |

Recorded on-device numbers:
[`fixtures/RECORDED_DEVICE_NOTES.md`](fixtures/RECORDED_DEVICE_NOTES.md).

### Desired behavior

The wide, two-face schedule must produce the same results as AscendC (and as the
narrow VMI twin) for this dequant algorithm. Prefer a fix in VMI layout /
`vmi-to-vpto` / emission around multi-buffering — not a rule that every kernel
must use the narrow shape.

After a fix:

1. Wide and narrow vector bodies still lower with `pto-test-opt`.
2. The wide full-kernel schedule matches AscendC (maximum absolute difference
   = 0) on the shapes listed in `RECORDED_DEVICE_NOTES.md`.

### Out of scope

- Presenting product schedule knobs as the PTOAS fix
- Claiming the narrow path must be removed (it is the control case that works)

---

## Feature request 3 — FP32 strip abs-max: VMI is slower than AscendC; need emit proof before a compiler change

### What the programs do

Both sides compute an absolute maximum over an FP32 strip and write a scale
value — a common building block for FP32 quant.

- **AscendC (working reference):**
  [`fixtures/reference_asc_fp32_strip_amax.asc`](fixtures/reference_asc_fp32_strip_amax.asc)
  loads two 64-lane FP32 chunks, takes abs, reduces with `vmax`, stores the
  result. Compiles with `bisheng` today.
- **VMI (already compiles):**
  [`fixtures/current_vmi_fp32_strip_amax.pto`](fixtures/current_vmi_fp32_strip_amax.pto)
  does `vabs` then `vcmax` with `group = 1`. `pto-test-opt` lowers this today
  (the one-element / one-part reduce+store path already works).

### What is wrong today

On real kernels that use this strip pattern, the VMI path is still slower than
the AscendC MicroAPI path in wall-clock measurements. Correctness and basic
lowering are **not** the open issue here.

### What kind of PTOAS issue this might be

This is a **performance** question, not a “does not compile” bug:

1. If the machine code that PTOAS emits for the VMI fixture does extra memory
   traffic, layout reshapes, or moves that the AscendC reference does not, then
   that is a concrete PTOAS emit / layout problem to fix.
2. If the two emits look similarly dense, the gap may be ISA choice or how the
   larger kernel is scheduled — not something this small fixture can justify as
   a PTOAS legalization change.

A wall-clock ratio by itself does not tell which of those it is. That is why
this package does not yet ask for a specific code change in PTOAS: we need a
side-by-side look at the generated code first.

### Evidence in this package

| Side | Fixture | What the check script shows today |
|------|---------|-----------------------------------|
| AscendC | `reference_asc_fp32_strip_amax.asc` | Compiles (`bisheng`) |
| VMI | `current_vmi_fp32_strip_amax.pto` | Lowers to VPTO (`pto-test-opt`) |

How to compare the generated code (AscendC vs VMI):

1. Lower the VMI fixture (`pto-test-opt` with the check-script passes, or
   `ptoas --emit-vpto` with the body wrapped in `pto.vecscope`).
2. Compile the AscendC fixture and inspect its vector / memory sequence.
3. Record any extra VMI moves, spills, or layout reshapes in
   [`fixtures/emit_compare_note.md`](fixtures/emit_compare_note.md).

That note is empty of a real diff today — so this feature request stays as
“reproducer + open performance question,” not “land this compiler patch.”

### Desired behavior

- Keep the current VMI fixture compiling (the `group = 1` reduce/store path
  must remain valid).
- If the emit comparison shows clear redundant work on the VMI side relative to
  AscendC, remove that redundancy in VMI layout or emission so the strip is as
  dense as the AscendC reference.
- Do not accept a PTOAS code change for this item until that AscendC-vs-VMI emit
  comparison is written down in `emit_compare_note.md`.

### Out of scope

- Using wall-clock ratios alone as proof of a compiler bug
- Re-opening pad-32 + broadcast load mode as the main request
