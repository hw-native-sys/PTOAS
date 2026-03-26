# VPTO Host Validation

`test/vpto` provides an end-to-end A5 validation path for hand-curated VPTO
cases whose input `kernel.pto` is already A5VM MLIR.

The driver script is:

```bash
bash test/vpto/scripts/run_host_vpto_validation.sh
```

## Required Environment

Source the CANN environment first so `ASCEND_HOME_PATH` and the toolchain are
available:

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export WORK_SPACE=/path/to/workspace
export PTOAS_BIN=$PWD/build/tools/ptoas/ptoas
```

`test/vpto` uses the installed PTO headers under `${ASCEND_HOME_PATH}/include`.
It does not require sourcing the repo-local `env.sh`.

Optional overrides:

```bash
export PTOAS_FLAGS="--pto-arch a5"
export A5VM_FLAGS="--pto-backend=a5vm --a5vm-emit-hivm-llvm"
export AICORE_ARCH=dav-c310-vec
export HOST_RUNNER="ssh root@localhost"
export CASE_NAME=abs
```

## Case Discovery

The runner automatically discovers every first-level subdirectory under
`test/vpto/cases/`.

- If `CASE_NAME` is unset, all cases are run.
- If `CASE_NAME=<name>` is set, only `test/vpto/cases/<name>/` is run.

## Case Layout

Each case directory must use these fixed file names:

```text
test/vpto/cases/<case-name>/
  kernel.pto
  stub.cpp
  launch.cpp
  main.cpp
  golden.py
  compare.py
```

File roles:

- `kernel.pto`: A5VM MLIR input consumed by `ptoas`
- `stub.cpp`: host-side fatobj stub that exports the final kernel symbol
- `launch.cpp`: kernel launch wrapper
- `main.cpp`: host executable entry for validation
- `golden.py`: generates testcase inputs and expected outputs
- `compare.py`: compares runtime outputs against golden data

## Flow

For each case, the runner performs:

1. Lower `kernel.pto` to LLVM IR with `ptoas`
2. Compile LLVM IR to device object with `bisheng`
3. Build `launch.cpp` and `stub.cpp`
4. Repack and link the kernel `.so`
5. Build the host executable and generate golden data
6. Run validation on the configured host and compare outputs

## Usage

Run all cases:

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export WORK_SPACE=$(mktemp -d /tmp/vpto.XXXXXX)
export PTOAS_BIN=$PWD/build/tools/ptoas/ptoas
bash test/vpto/scripts/run_host_vpto_validation.sh
```

Run a single case:

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export WORK_SPACE=$(mktemp -d /tmp/vpto-abs.XXXXXX)
export PTOAS_BIN=$PWD/build/tools/ptoas/ptoas
export CASE_NAME=abs
bash test/vpto/scripts/run_host_vpto_validation.sh
```
