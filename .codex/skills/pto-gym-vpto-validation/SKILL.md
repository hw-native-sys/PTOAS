---
name: pto-gym-vpto-validation
description: Run PTO-Gym validation from this PTOAS repo. Use when the user asks to run PTO-Gym SIM or board validation from the current source tree. Always force PTOAS onto the VPTO LLVM path instead of relying on the repo default backend.
---

# PTO-Gym VPTO Validation

Use this skill when the task is specifically about:
- running `3rdparty/PTO-Gym/examples/pto/scripts/run_host_vpto_validation.sh`
- running `3rdparty/PTO-Gym/examples/pto/scripts/run_host_vpto_validation_parallel.sh`
- validating PTO-Gym cases from this PTOAS source tree

## Required Rule

When PTO-Gym is run from this repo, do not rely on the default PTOAS backend.

Always pass PTOAS flags that force the VPTO LLVM path.
The current `ptoas` CLI spellings in this repo are `--pto-backend=vpto` and
`--vpto-emit-hivm-llvm`; do not shorten `--pto-backend` to `--backend`.

Use:

```bash
PTOAS_FLAGS='--pto-backend=vpto --vpto-emit-hivm-llvm --pto-arch a5'
```

If the caller already provides `PTOAS_FLAGS`, make sure these options are still
present. Do not silently fall back to the repo default backend.

## Canonical Environment

Use `.work/` under the repo for all scratch output and temp files:

```bash
mkdir -p .work/tmp .work/runs
export TMPDIR=$PWD/.work/tmp
export TMP=$TMPDIR
export TEMP=$TMPDIR
```

Typical simulator environment:

```bash
source /home/mouliangyu/.local/ascend/beta.2/cann-9.0.0-beta.2/set_env.sh
export ASCEND_HOME_PATH=/home/mouliangyu/.local/ascend/beta.2/cann-9.0.0-beta.2
export PTOAS_BIN=$PWD/build/tools/ptoas/ptoas
export PTOAS_FLAGS='--pto-backend=vpto --vpto-emit-hivm-llvm --pto-arch a5'
```

## Canonical Commands

Single case:

```bash
WORK_SPACE=$PWD/.work/runs/pto-gym-single \
ASCEND_HOME_PATH=$ASCEND_HOME_PATH \
PTOAS_BIN=$PTOAS_BIN \
PTOAS_FLAGS="$PTOAS_FLAGS" \
CASE_NAME=micro-op/binary-vector/vadd \
DEVICE=SIM \
bash 3rdparty/PTO-Gym/examples/pto/scripts/run_host_vpto_validation.sh
```

Parallel micro-op sweep:

```bash
WORK_SPACE=$PWD/.work/runs/pto-gym-microop \
ASCEND_HOME_PATH=$ASCEND_HOME_PATH \
PTOAS_BIN=$PTOAS_BIN \
PTOAS_FLAGS="$PTOAS_FLAGS" \
CASE_PREFIX=micro-op \
DEVICE=SIM \
JOBS=64 \
bash 3rdparty/PTO-Gym/examples/pto/scripts/run_host_vpto_validation_parallel.sh
```

## Reporting Back

Report:
- the exact `PTOAS_FLAGS` used
- the final `PASS/FAIL` counts
- the summary file path under `.work/runs/...`

If a run fails, identify the first failing case from `parallel-summary.tsv` and
then inspect that case directory under `WORK_SPACE`.
