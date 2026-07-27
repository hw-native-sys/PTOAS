# pipelining — VMI Persistent stages vs CCE ping-pong repro

Self-contained compile repro for PTOAS gap 05: long-K quant / cast-back
Persistent loops want multi-buffer stages (e.g. stages=2) and `block_k > 512`
so MTE and vector stay overlapped. Ascend CCE can express stages=2 ping-pong
with `block_k=1024`; VMI on tag **vmi-v0.1.3** cannot reach the same schedule
— layout assignment rejects wide tiles, and the hand-written MI path crashes
bisheng even after ptoas emits LLVM IR.

Fixtures are copied from `docs/feature-gaps/vmi/05_persistent_stages_block_k/`.

## What breaks

| Path | stages | block_k | Fault mode |
|------|-------:|--------:|------------|
| CCE (`fixtures/reference_asc_cce.asc`) | 2 | 1024 | **OK** — bisheng `--cce-aicore-only` produces a non-empty `.o` |
| VMI desired (`fixtures/desired_vmi.*`) | 2 | 1024 | **Layout reject** — `pto.vmi.mulf` on 1024-wide tile: deinterleaved=2 vs contiguous |
| VMI target MI (`fixtures/target_mi.pto`) | 2 | 1024 | **Bisheng crash** — ptoas `--emit-vpto` and LLVM IR OK; object compile exit 70 |
| VMI current slow (`fixtures/current_slow_vmi.*`) | 1 | 512 | **Frontend OK** — ptodsl emits MLIR; idiomatic VMI `.pto` hits same layout issue |
| VMI lowered MI (`fixtures/lowered_vpto.pto`) | 1 | 512 | **OK to LLVM IR** — chunked 128-wide `vlds`/`vmul`/`vsts` path lowers |

Run `./scripts/check_stages.sh` to refresh the table below.

## Check results (vmi-v0.1.3, 2026-07-27)

Measured by `./scripts/check_stages.sh`:

| Step | Result |
|------|--------|
| `reference_asc_cce.asc` → bisheng `--cce-aicore-only` | **PASS** |
| `desired_vmi.py` → MLIR frontend | **PASS** (frontend only) |
| `desired_vmi.py` MLIR → LLVM IR | **FAIL** — layout reject on 1024-wide vmul |
| `desired_vmi.pto` → `--emit-vpto` | **FAIL** — same layout reject |
| `target_mi.pto` → `--emit-vpto` | **PASS** |
| `target_mi.pto` → LLVM IR → bisheng `.o` | **FAIL** — bisheng crash (exit 70) |
| `current_slow_vmi.py` → MLIR frontend | **PASS** (frontend only) |
| `lowered_vpto.pto` → LLVM IR | **PASS** — stages=1 chunked MI path |

Full log: `sim_outputs/check_stages/compile_results.txt`

## Layout

```
pipelining/
  fixtures/     IR dumps and reference ASC from gap 05
  scripts/      env.sh, check_stages.sh
```

CCE stages=2 baseline lives in `fixtures/reference_asc_cce.asc` (compile-only
is enough for this repro; on-device overlap profiling is TODO below).

## Prerequisites

- CANN 9.0 (`bisheng`)
- Built PTOAS at tag **vmi-v0.1.3** (branch `zjw/pipelining_issue`)

Push remote hint: https://github.com/learning-chip/PTOAS.git

## Run

From this directory (`docs/repro/pipelining/`):

```bash
source scripts/env.sh
./scripts/check_stages.sh
```

## Pin

- PTOAS tag: **vmi-v0.1.3**
- Branch: `zjw/pipelining_issue`
- `PTOAS_ROOT` defaults to the repo root (`../../..` from this repro directory).

## TODO

- On-device run and **msopprof** bandwidth study for stages=2 CCE vs stages=1
  VMI on large hidden sizes.
- Guide: https://gitcode.com/Ascend/msopprof/blob/master/docs/en/user_guide/msopprof_user_guide.md

## Done when

VMI can compile and run stages=2 / `block_k=1024` Persistent tiles with
dual-buffer ping-pong — same overlap pattern as CCE and `target_mi.pto`.
