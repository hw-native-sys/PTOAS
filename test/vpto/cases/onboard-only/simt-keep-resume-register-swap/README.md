# SIMT Keep/Resume Register Swap

This probe checks non-immediate parallel copies between persistent 32-bit and
64-bit SIMT register slots.

## Data Flow

For each SIMT lane `tid`:

1. `stage0` computes runtime values and keeps them:
   - slot 0 / `TPER4 = tid + 100`
   - slot 1 / `TPER5 = tid + 200`
   - slot 2 / `TPERL3` / `R6:R7 = 0x1122334400000000 + tid`
   - slot 4 / `TPERL4` / `R8:R9 = 0x5566778800000000 + tid`
2. `stage1` resumes both values and keeps them in reverse order:
   - `TPER4 <- old TPER5`
   - `TPER5 <- old TPER4`
   - `TPERL3 <- old TPERL4`
   - `TPERL4 <- old TPERL3`
3. `stage2` resumes all slots and writes the swapped 32-bit and 64-bit values
   to separate GM buffers.

The second keep must be treated as a parallel copy. A destructive sequence
loses one side of a cycle and fails the host check. The backend must break each
cycle with temporary registers or spills. The 64-bit values have different,
nonzero high halves so the test covers both words of each register pair.

## Generate LLVM IR

```bash
PTOAS_ROOT=/path/to/PTOAS
"${PTOAS_ROOT}/build/tools/ptoas/ptoas" \
  --pto-arch=a5 \
  --pto-backend=vpto \
  \
  --emit-vpto-llvm-ir \
  "${PTOAS_ROOT}/test/vpto/cases/onboard-only/simt-keep-resume-register-swap/kernel.pto" \
  -o kernel.ll
```

The relevant `stage1` constraint is expected to be:

```llvm
%swapped = call { i32, i32, i64, i64 } asm sideeffect "",
  "={TPER4},={TPER5},={TPERL3},={TPERL4},0,1,2,3"
  (i32 %old_tper5, i32 %old_tper4, i64 %old_tperl4, i64 %old_tperl3)
```

## Hardware Validation

This test requires hardware because the simulator does not model keep/resume
state across SIMT entries. Build and run it through PTOAS's validation script:

```bash
cd /path/to/PTOAS
CASE_NAME=onboard-only/simt-keep-resume-register-swap \
DEVICE=HW \
WORK_SPACE=/tmp/ptoas-register-swap-validation \
ASCEND_HOME_PATH=/path/to/CANN \
test/vpto/scripts/run_host_vpto_validation.sh
```
