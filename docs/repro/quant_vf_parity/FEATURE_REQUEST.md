# Feature requests — AscendC vs VMI residuals after PR #1145

Audience: PTOAS maintainers reviewing this reproducer package.

On top of [PR #1145](https://github.com/hw-native-sys/PTOAS/pull/1145)
(bf16 absolute value via clearing the sign bit, and bf16 group-4 broadcast),
these AscendC-vs-VMI residuals remain. Each open request pairs a working
AscendC program with a nearly equivalent VMI/PTO program that fails to compile
or produces wrong results.

Pipe synchronization (`set_flag` / `wait_flag`, including dynamic forms) is
already present in the VMI dumps. Enabling InsertSync is not what these
requests ask for.

## Status (read this first)

| Priority | Topic | Status |
|----------|--------|--------|
| **Open — primary** | Feature request 1: `get_block_idx` bound cannot control a vector body | **Blocker** — VMI emit fails; blocks multi-buffer schedules that need that pattern |
| **Open** | Feature request 2: wide double-buffered dequant disagrees with AscendC | **Still open** — numeric mismatch on the wide schedule |
| Backlog | Former request 3: FP32 double-buffered block-quant wall-clock | **Solved as a product / schedule issue — not a PTOAS blocker now.** Kept only as backlog (fixtures + notes). |

---

## Feature request 1 — `get_block_idx` used to decide whether a vector body runs fails in ptoas → bisheng

**This is the main remaining ask.** Multi-buffer kernels that form a tile bound
from the hardware block index and only then run vector work cannot finish
device emission on VMI. AscendC does the same pattern today. Until this lowers,
schedules that need that guard (including offset-style multi-buffer pipelines
at large shapes) cannot ship on the VMI path.

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

**Still open** (correctness, not the primary compile blocker).

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

## Backlog — solved / not a blocker (kept for history)

### Former feature request 3 — FP32 double-buffered block-quant wall-clock

**Issue solved. Not a blocker now. Kept only as backlog.**

Wall-clock at large shape (8192×2048) is within noise of AscendC (~0.98× on
msopprof and product zero-copy) after the VMI side uses **serial** abs-max row
loops (`range` → `scf.for`). The earlier ~0.88× gap was from **explicitly
unrolling** that abs-max strip (huge IR / slower), not from a missing PTOAS
legalization of `vcmax(group=1)`.

Fixtures and notes remain for regression and history:

| Artifact | Role |
|----------|------|
| [`reference_asc_fp32_block_quant.asc`](fixtures/reference_asc_fp32_block_quant.asc) + [`fp32_block_quant_artifact/`](fixtures/fp32_block_quant_artifact/) | AscendC full kernel + ctypes launch lib |
| [`current_vmi_fp32_block_quant_*.ptodsl.py`](fixtures/) | VMI full kernel (serial abs-max / pair / quantize) |
| [`PERF_FINDINGS.md`](fixtures/PERF_FINDINGS.md) | Host + msopprof numbers after the fix |
| Strip amax fixtures / [`emit_compare_note.md`](fixtures/emit_compare_note.md) | Fragment-only compile checks (never claimed the wall gap) |

Optional re-run (not required to close an open ask):

```bash
PTOAS_ROOT=/path/to/PTOAS NPU_DEVICE=1 ./scripts/run_fp32_block_quant_device.sh
./scripts/run_msopprof_fp32_block_quant.sh both
```

Do **not** treat strip-only PASS, or small residual pipe-metric differences while
wall-clock is ~parity, as a reason to reopen this as a primary blocker.
