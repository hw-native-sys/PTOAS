# Backlog — former feature request 3 (solved; not a blocker)

**Issue solved. Not a blocker now. Kept as backlog.**

Wall-clock at 8192×2048 is ~parity (~0.98× AscendC/VMI on msopprof) after
serial abs-max rows. See [`PERF_FINDINGS.md`](PERF_FINDINGS.md).

Strip fixtures below are **VF fragments only** — they never reproduced the old
wall gap and are not an open ask.

## Strip fragment (optional compile / emit detail)

| Side | Fixture | Role |
|------|---------|------|
| AscendC | `reference_asc_fp32_strip_amax.asc` | Dense MicroAPI abs-max strip |
| VMI | `current_vmi_fp32_strip_amax.pto` | abs + `vcmax` with `group = 1` |

### Status today

- AscendC strip compiles with `bisheng`.
- VMI strip lowers with `pto-test-opt` (check-script pass list).
- The one-element reduce+store path is already legal; no further strip
  legalization is needed to keep this off the open-issue list.

### How to inspect the strip emit

1. Lower `current_vmi_fp32_strip_amax.pto` (check-script passes, or wrap the
   body in `pto.vecscope` and run `ptoas --emit-vpto`).
2. Compile `reference_asc_fp32_strip_amax.asc` and inspect its vector / memory
   sequence.
3. List any extra VMI moves, spills, or layout reshapes relative to AscendC
   (optional; not required to reopen this ask).

## Strip emit findings

_(optional detail — see PERF_FINDINGS.md for the closed wall-clock claim)_
