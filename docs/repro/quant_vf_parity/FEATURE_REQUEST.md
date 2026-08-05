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

## Feature request 3 — Full FP32 block-quant: VMI is slower than AscendC at large shape

### What the programs do (same algorithm both sides)

FP32 matrix → e4m3 + per-block scales (`32×32`), UE8M0-style scale round:

1. Persistent tiles over the matrix with **two on-chip faces** overlapping copy
   and vector work.
2. Tile face: **32 rows × 512 cols** of FP32 in UB.
3. Vector body, eight times per tile (each **64 cols**): abs-max over 32 rows;
   two independent one-element reduces (low/high 32-lane halves) → two scales;
   keep reciprocals in registers; store two scale floats; per row multiply +
   convert to e4m3 and store.
4. Copy quantized tile and scales back to GM.

Public name: `fp32_block_quant` (“double-buffered block quantize”).

### Primary evidence (full kernel — reproduces the gap)

| Side | Artifact | Role |
|------|----------|------|
| AscendC | [`fixtures/reference_asc_fp32_block_quant.asc`](fixtures/reference_asc_fp32_block_quant.asc) + [`fixtures/fp32_block_quant_artifact/`](fixtures/fp32_block_quant_artifact/) | Double-buffered tile schedule; VF uses `__simd_callee__` return wrappers over MicroAPI (same pattern as the timed `.so`); `bisheng` / prebuilt `.so` |
| VMI | [`fixtures/current_vmi_fp32_block_quant_8192x2048.ptodsl.py`](fixtures/current_vmi_fp32_block_quant_8192x2048.ptodsl.py) (and `…_512x2048…`) | Same schedule in VMI/PTO |

Host (CANN + `torch_npu` / ACL + this PTOAS tree only — no product `PYTHONPATH`):

```bash
PTOAS_ROOT=/path/to/PTOAS NPU_DEVICE=1 ./scripts/run_fp32_block_quant_device.sh
./scripts/run_msopprof_fp32_block_quant.sh both
```

Checked-in numbers and pipe interpretation:
[`fixtures/PERF_FINDINGS.md`](fixtures/PERF_FINDINGS.md).

| Shape | AscendC / VMI (msopprof or host) | Meaning |
|-------|----------------------------------|---------|
| 512×2048 | ~0.97 | Near parity (control) |
| 8192×2048 | **~0.86** (msopprof ~27.0 vs ~31.5 µs) | VMI slower — primary claim |

Correctness: scale and output lane mismatches vs AscendC = 0 before trusting timing.

### Strip fixtures (VF fragment only — do **not** claim the gap)

| Side | Fixture | What they prove |
|------|---------|-----------------|
| AscendC | [`reference_asc_fp32_strip_amax.asc`](fixtures/reference_asc_fp32_strip_amax.asc) | Dense MicroAPI abs-max strip compiles |
| VMI | [`current_vmi_fp32_strip_amax.pto`](fixtures/current_vmi_fp32_strip_amax.pto) | `vabs` + `vcmax(group=1)` lowers today |

These compile/lower checks only show the one-element reduce path is legal. They
do **not** time anything and do **not** reproduce the large-shape gap.

Emit notes for the strip (optional detail):
[`fixtures/emit_compare_note.md`](fixtures/emit_compare_note.md).

### What kind of PTOAS issue this is

A **performance** residual on the full double-buffered schedule, not a “does not
compile” bug. From `PERF_FINDINGS.md` (PipeUtilization @ 8192×2048):

- VMI spends more core-time in vector and scalar; higher UB read bandwidth from
  the vector unit.
- AscendC shows higher MTE3 overlap while finishing sooner.
- Gap is large-shape only → bandwidth / overlap / emit density, not a small-N
  functional bug.
- Strip-only `group=1` legalization is **not** the missing piece.

Likely residual classes: denser AscendC MicroAPI packing of the 64-lane strip
vs VMI dual `vcmax` + `vsel` recip; and/or MTE/VF overlap tax in lowered VMI
double-buffer emit. Prefer an emit or pipe fix justified by `PERF_FINDINGS.md`
(+ optional strip emit compare), not a wall-clock ratio alone.

### Desired behavior

1. VMI within noise of AscendC at **8192×2048** (ratio near 1.0), **or**
2. A concrete emit/pipe change in PTOAS justified by the findings above.

Keep the strip `group=1` path compiling. Do not treat strip-only PASS as closing
this request.

### Out of scope

- Declaring a PTOAS patch before reading `PERF_FINDINGS.md`
- Claiming the tiny strip fixtures reproduce the wall-clock gap
- bf16 per-block, packed SF, fused roundtrip, cast_back (other asks)
