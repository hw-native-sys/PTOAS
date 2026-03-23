---
name: ptoas-npu-validation-a5
description: Build and validate PTOAS-generated A5 kernels with test/npu_validation, especially the Abs sample. Use when the user wants to generate npu_validation testcases, compile Abs for A5 with bisheng, or run/check the A5 NPU or simulator validation flow with PTO_ISA_ROOT.
---

# PTOAS NPU Validation A5

Use this skill when the task is specifically about:
- generating `test/npu_validation` projects from PTOAS output
- compiling the `Abs` sample through the `npu_validation` flow
- validating A5 targets such as `dav-c310-vec`
- using an external `pto-isa` checkout via `PTO_ISA_ROOT`

## Preconditions

Before running `npu_validation`, make sure:
- `ptoas` is already built in `./build`
- `bisheng` is in `PATH` or available through CANN `set_env.sh`
- `PTO_ISA_ROOT` points to a `pto-isa` checkout with:
  - `include/`
  - `tests/common/`

Canonical external repo path on this machine:

```bash
export PTO_ISA_ROOT=/home/mouliangyu/projects/github.com/PTO-ISA/pto-isa
```

## Canonical Flow For Abs On A5

### 1. Generate the PTOAS kernel

Use the default EmitC-style output, because `npu_validation` consumes `*-pto.cpp`.

```bash
source env.sh
PTOAS_BIN="$PWD/build/tools/ptoas/ptoas" \
PTOAS_OUT_DIR=/tmp/ptoas-abs-emitc \
./test/samples/runop.sh -t Abs
```

Expected output:
- `/tmp/ptoas-abs-emitc/Abs/abs-pto.cpp`

### 2. Generate the `npu_validation` testcase

For A5 `Abs`, use `dav-c310-vec`.

```bash
python3 test/npu_validation/scripts/generate_testcase.py \
  --input /tmp/ptoas-abs-emitc/Abs/abs-pto.cpp \
  --testcase abs \
  --output-root /tmp/ptoas-npu-validation-run \
  --run-mode sim \
  --soc-version dav_3102 \
  --aicore-arch dav-c310-vec
```

Expected output directory:
- `/tmp/ptoas-npu-validation-run/Abs/abs`

### 3. Configure and build

```bash
export PTO_ISA_ROOT=/home/mouliangyu/projects/github.com/PTO-ISA/pto-isa
source /usr/local/Ascend/cann/set_env.sh
cmake -S /tmp/ptoas-npu-validation-run/Abs/abs \
  -B /tmp/ptoas-npu-validation-run/Abs/abs/build \
  -DSOC_VERSION=dav_3102 \
  -DENABLE_SIM_GOLDEN=ON
cmake --build /tmp/ptoas-npu-validation-run/Abs/abs/build --parallel
```

Build expectations observed on this machine:
- `libabs_kernel.so` builds
- `abs` builds
- `abs_sim` may fail to link if simulator runtime symbols are missing

### 4. Generate golden inputs

```bash
cd /tmp/ptoas-npu-validation-run/Abs/abs
python3 ./golden.py
```

Expected files:
- `v1.bin`
- `v2.bin`

## Running

### NPU run

Only attempt this on a shell that can actually see `/dev/davinci*`.

```bash
export PTO_ISA_ROOT=/home/mouliangyu/projects/github.com/PTO-ISA/pto-isa
source /usr/local/Ascend/cann/set_env.sh
cd /tmp/ptoas-npu-validation-run/Abs/abs
LD_LIBRARY_PATH="/usr/local/Ascend/cann-9.0.0/lib64:${LD_LIBRARY_PATH:-}" \
  ./build/abs
```

### Simulator run

If `abs_sim` links successfully, run it with simulator libraries in `LD_LIBRARY_PATH`.

```bash
export PTO_ISA_ROOT=/home/mouliangyu/projects/github.com/PTO-ISA/pto-isa
source /usr/local/Ascend/cann/set_env.sh
cd /tmp/ptoas-npu-validation-run/Abs/abs
LD_LIBRARY_PATH="/usr/local/Ascend/cann-9.0.0/aarch64-linux/simulator/dav_3102/lib:/usr/local/Ascend/cann-9.0.0/lib64:${LD_LIBRARY_PATH:-}" \
  ./build/abs_sim
```

## Known Failure Modes

- If `generate_testcase.py` fails, the input is usually not a PTOAS EmitC `*-pto.cpp` kernel.
- If configure fails, `PTO_ISA_ROOT` is usually unset or points to the wrong checkout.
- If `abs_sim` fails to link, common missing symbols are:
  - `rtDevBinaryRegister`
  - `rtDevBinaryUnRegister`
  - `rtLaunch`
  - `rtKernelLaunch`
  - `rtKernelLaunchWithFlagV2`
  - `rtFunctionRegister`
- If `./build/abs` fails at `aclInit(nullptr)`, the shell likely does not have usable Ascend device access.
- In sandboxed agent environments, `/dev/davinci*` may be invisible even if the host shell can see them.

## Reporting Back

When you use this skill, report:
- the generated testcase directory
- whether `libabs_kernel.so`, `abs`, and `abs_sim` built
- whether `golden.py` generated input/output bins
- the first concrete blocker for NPU or simulator execution
