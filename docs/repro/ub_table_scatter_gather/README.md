# UB table scatter/gather — stale `vgather` reads

A UB-resident lookup table written with `vscatter` and read back with
`vgather` returns the seed sentinel on every lane. AscendC with the same
schedule is exact. This package is a self-contained bug report and numeric
repro for PTOAS.

Branch: `repro/ub-pool-scatter-gather`. Details: [`BUG_REPORT.md`](BUG_REPORT.md).

## Algorithm

```text
seed pool[i] = -inf          for i in 0..63   (vscatter)
barrier
pool[i]      = vals[i]       for i in 0..63   (vscatter)
barrier
out[i]       = pool[i]       for i in 0..63   (vgather)
```

| Fixture | Role | Expected on current build |
|---------|------|---------------------------|
| `fixtures/failing_vmi.py` | Failing VMI numeric repro | **FAIL** (all `-inf`) |
| `fixtures/reference_asc_cce.asc` | AscendC peer | **PASS** (`bisheng`) |
| `fixtures/variant_*.py` | Ruled-out hypotheses | still wrong |

## Layout

```text
ub_table_scatter_gather/
  BUG_REPORT.md
  README.md
  fixtures/
  scripts/
  outputs/          # produced by check script (gitignored)
```

## Prerequisites

- CANN toolkit with `bisheng` and Ascend headers (`ASCEND_HOME_PATH`)
- A Python env with `ptodsl` (this tree's `ptodsl/` or an installed wheel)
- An NPU device for the numeric check (`NPU_TEST_DEVICE`, default `npu:1`)

## Run

From this directory:

```bash
source scripts/env.sh
./scripts/check_ub_table_scatter_gather.sh
```

Results: `outputs/check_ub_table_scatter_gather/results.txt`.

## How to read the result

- **COMPILE PASS** on `failing_vmi.py` plus **NUMERIC FAIL** (all `-inf`) is
  the bug this package reports.
- If **NUMERIC PASS**, the criteria in [`BUG_REPORT.md`](BUG_REPORT.md) are met.
- If the host has no NPU / driver, the script records `LAUNCH SKIP` and still
  scores the compile side.
