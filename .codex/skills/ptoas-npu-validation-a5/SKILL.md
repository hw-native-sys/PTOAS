---
name: ptoas-npu-validation-a5
description: Generate and run PTOAS-based A5 test/npu_validation testcases, build the testcase binaries, and validate runtime output on NPU or simulator. Use when the user wants NPU run validation, golden/compare checks, or runtime troubleshooting for A5.
---

# PTOAS NPU Validation A5

Use this skill when the task is specifically about:
- generating `test/npu_validation` projects from PTOAS output
- building testcase binaries for A5
- running NPU or simulator validation
- generating golden inputs and checking results with `compare.py`
- diagnosing runtime blockers such as missing device access or `aclrtSetDevice`

This skill is the main entry point for runtime validation.

Do not use this skill as the primary entry point when the task is only about:
- exporting LLVM IR or LLVM bitcode
- validating the `bisheng` handoff
- assembling a fat object or replacement kernel library from the LLVM path

When this validation flow needs a custom LLVM IR or LLVM BC artifact, use
`ptoas-a5vm-llvm-artifacts` first to build that artifact, then return here to
run the testcase.

## Preconditions

Before running `npu_validation`, make sure:
- `ptoas` is already built in `./build`
- `bisheng` is in `PATH` or available through CANN `set_env.sh`
- `PTO_ISA_ROOT` points to a `pto-isa` checkout with:
  - `include/`
  - `tests/common/`
- the shell can read `/dev/davinci*` if you intend to execute on real hardware

Example:

```bash
export PTO_ISA_ROOT=/path/to/pto-isa
```

Useful runtime check:

```bash
source /usr/local/Ascend/cann/set_env.sh
python3 - <<'PY'
import ctypes
lib = ctypes.cdll.LoadLibrary('libascendcl.so')
aclInit = lib.aclInit; aclInit.argtypes=[ctypes.c_char_p]; aclInit.restype=ctypes.c_int
aclrtGetDeviceCount = lib.aclrtGetDeviceCount; aclrtGetDeviceCount.argtypes=[ctypes.c_void_p]; aclrtGetDeviceCount.restype=ctypes.c_int
aclrtSetDevice = lib.aclrtSetDevice; aclrtSetDevice.argtypes=[ctypes.c_int]; aclrtSetDevice.restype=ctypes.c_int
cnt = ctypes.c_uint(0)
print('aclInit', aclInit(None))
print('aclrtGetDeviceCount', aclrtGetDeviceCount(ctypes.byref(cnt)), cnt.value)
print('aclrtSetDevice', aclrtSetDevice(0))
PY
```

Interpretation:
- `aclInit` succeeds
- `aclrtGetDeviceCount` should report at least one device if the runtime can enumerate hardware
- if `aclrtSetDevice(0)` fails with `507033` (`ACL_ERROR_RT_DEV_SETUP_ERROR`), the user context can see a device but cannot open a usable runtime context

## Canonical Flow

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
export PTO_ISA_ROOT=/path/to/pto-isa
source /usr/local/Ascend/cann/set_env.sh
cmake -S /tmp/ptoas-npu-validation-run/Abs/abs \
  -B /tmp/ptoas-npu-validation-run/Abs/abs/build \
  -DSOC_VERSION=dav_3102 \
  -DENABLE_SIM_GOLDEN=ON
cmake --build /tmp/ptoas-npu-validation-run/Abs/abs/build --parallel
```

Typical build expectations:
- `libabs_kernel.so` builds
- `abs` builds
- `abs_sim` may also build if the simulator runtime is available

If you need to replace the default `libabs_kernel.so` with one assembled from
an LLVM IR or LLVM BC path, build that artifact with
`ptoas-a5vm-llvm-artifacts` and place it first in `LD_LIBRARY_PATH` when
running `./build/abs`.

### 4. Generate golden inputs

```bash
cd /tmp/ptoas-npu-validation-run/Abs/abs
python3 ./golden.py
```

Expected files:
- `v1.bin`
- `v2.bin`

For the generated `Abs` testcase, `golden.py` does not emit `golden_v2.bin`,
but `compare.py` expects it. Build the oracle explicitly from the input:

```bash
cd /tmp/ptoas-npu-validation-run/Abs/abs
python3 - <<'PY'
import numpy as np
v1 = np.fromfile('v1.bin', dtype=np.float32)
np.abs(v1).astype(np.float32).tofile('golden_v2.bin')
PY
```

Expected additional file:
- `golden_v2.bin`

## Running

### NPU run

Only attempt this on a shell that can actually see `/dev/davinci*`.

```bash
export PTO_ISA_ROOT=/path/to/pto-isa
source /usr/local/Ascend/cann/set_env.sh
cd /tmp/ptoas-npu-validation-run/Abs/abs
LD_LIBRARY_PATH="${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH:-}" \
  ./build/abs
```

If the current user cannot open the device context but `sudo` is acceptable in
your environment, retry with:

```bash
sudo bash -lc '
  cd /tmp/ptoas-npu-validation-run/Abs/abs
  source /usr/local/Ascend/cann/set_env.sh >/dev/null 2>&1
  LD_LIBRARY_PATH="${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH:-}" \
    ./build/abs
'
```

Observed runtime result on this machine for the `Abs` testcase:
- normal user run failed at `aclrtSetDevice(0)` with `507033`
- `sudo` run of the existing `./build/abs` binary succeeded
- `python3 ./compare.py` then reported `[INFO] compare passed`

### Simulator run

If `abs_sim` links successfully, run it with simulator libraries in `LD_LIBRARY_PATH`.

```bash
export PTO_ISA_ROOT=/path/to/pto-isa
source /usr/local/Ascend/cann/set_env.sh
cd /tmp/ptoas-npu-validation-run/Abs/abs
LD_LIBRARY_PATH="${ASCEND_HOME_PATH}/aarch64-linux/simulator/dav_3102/lib:${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH:-}" \
  ./build/abs_sim
```

Treat simulator execution as optional. Depending on the local CANN install, the
simulator binary may link successfully but still fail at runtime due to missing
simulator services or runtime symbols.

## Compare

After generating `golden_v2.bin` and running the NPU binary, compare with:

```bash
cd /tmp/ptoas-npu-validation-run/Abs/abs
python3 ./compare.py
```

Expected success output:
- `[INFO] compare passed`

## Known Failure Modes

- `generate_testcase.py` fails because the input is not a PTOAS EmitC `*-pto.cpp` kernel
- configure fails because `PTO_ISA_ROOT` is unset or points to the wrong checkout
- `abs_sim` fails to link because simulator runtime symbols are missing
- `./build/abs` fails at `aclInit(nullptr)` because the shell does not have usable Ascend runtime access
- non-`sudo` `./build/abs` fails at `aclrtSetDevice(0)` with `507033`, meaning the user context sees the device but cannot open a usable runtime context
- `compare.py` reports `golden_v2.bin` missing because the testcase generation did not create it automatically

## Reporting Back

When you use this skill, report:
- the generated testcase directory
- whether `libabs_kernel.so`, `abs`, and `abs_sim` built
- whether `golden.py` generated input bins and whether `golden_v2.bin` had to be created explicitly
- whether NPU execution worked directly or required elevated privileges
- whether `compare.py` passed
- the first concrete blocker for NPU or simulator execution
