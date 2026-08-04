# Narrow 1-lane store — 32-byte alignment fault

A 1-lane `vstore` to a 4-byte-aligned (but not 32-byte-aligned) UB address
faults the vector cores. AscendC `vsts(..., ONEPT_B32)` at the same address is
fine. This package is a self-contained bug report and numeric repro for PTOAS.

Branch: `repro/narrow-store-align`. Details: [`BUG_REPORT.md`](BUG_REPORT.md).

## Algorithm

```text
# faulting (packed destination; input staged padded so loads stay aligned)
for k in 0..7:
    vstore(val[k, 0], out_ub[k], mask=1)       # out_ub: (8,) f32

# workaround (padded destination; GM out is also (B, 8), host checks col 0)
for k in 0..7:
    vstore(val[k, 0], pad_ub[k, 0], mask=1)    # pad_ub: (8, 8) f32
```

| Fixture | Role | Expected on current build |
|---------|------|---------------------------|
| `fixtures/faulting_vmi.py` | Packed 1-lane stores | **FAIL** launch (ACL 507035) |
| `fixtures/padded_workaround_vmi.py` | 32-byte padded slots | **PASS** numeric |
| `fixtures/reference_asc_cce.asc` | AscendC peer | **PASS** (`bisheng`) |

## Layout

```text
narrow_lane_store/
  BUG_REPORT.md
  README.md
  fixtures/
  scripts/
  outputs/          # produced by check script (gitignored)
```

## Prerequisites

- CANN toolkit with `bisheng` and Ascend headers (`ASCEND_HOME_PATH`)
- A Python env with `ptodsl`
- An NPU device for the numeric check (`NPU_TEST_DEVICE`, default `npu:1`)

## Run

From this directory:

```bash
source scripts/env.sh
./scripts/check_narrow_lane_store.sh
```

Results: `outputs/check_narrow_lane_store/results.txt`.

Run the two VMI fixtures in **separate processes** (the check script does);
a 507035 fault can poison the NPU context for later launches in the same
process.

## How to read the result

- **faulting COMPILE PASS + LAUNCH FAIL (507035)** together with **padded
  NUMERIC PASS** is the bug this package reports.
- If faulting also passes numerically, the criteria in
  [`BUG_REPORT.md`](BUG_REPORT.md) are met.
- If the host has no NPU / driver, the script records `LAUNCH SKIP` and still
  scores the compile side.
