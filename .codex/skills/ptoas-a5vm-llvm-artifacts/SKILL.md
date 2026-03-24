---
name: ptoas-a5vm-llvm-artifacts
description: Inspect PTOAS A5VM intermediate MLIR, export A5VM kernels as LLVM IR or LLVM bitcode, validate the bisheng handoff, and assemble device objects, fat objects, or shared kernel libraries. Use when the user asks for A5VM MLIR, an LLVM IR path build, LLVM BC export, bisheng compilation, or fatobj/lib.so assembly for A5.
---

# PTOAS A5VM LLVM Artifacts

Use this skill when the task is specifically about:
- printing or inspecting A5VM intermediate MLIR
- exporting PTOAS A5 kernels as LLVM IR or LLVM bitcode through the A5VM backend
- checking whether the export is textual LLVM IR or real LLVM bitcode
- compiling the exported artifact with `bisheng`
- assembling a device object, fat relocatable object, or shared kernel library from the LLVM path
- helping with an "LLVM IR path build", "LLVM IR path compile", or "A5VM MLIR" request

## Strong Rule

If an LLVM-path code change is needed for an A5 op, first inspect the installed
PTO implementation under `ASCEND_HOME_PATH` and treat it as the semantic
baseline. Repo-local lowering and emitter code are not authoritative.

Only accept an intrinsic-level replacement after confirming it from installed
PTO headers or from real frontend-generated compiler artifacts such as
`-save-temps` outputs. Do not invent intrinsic mappings from memory.

This is not the primary entry point for:
- generating `test/npu_validation` testcases
- running on hardware, handling `aclrtSetDevice`, or deciding whether `sudo` is needed
- `golden.py` / `compare.py` result checks

If the end goal is runtime validation, use `ptoas-npu-validation-a5` as the main
skill and call this skill only when that flow needs a custom LLVM IR or LLVM BC
kernel artifact.

## Preconditions

Before using this path, make sure:
- `ptoas` is already built in `./build`
- `bisheng` is available through CANN `set_env.sh`
- `env.sh` can be sourced from the repo root
- for the fatobj path, you already have a generated testcase directory that
  contains a wrapper source such as `abs_kernel.cpp` and a built `launch.cpp.o`

Load the repo environment before running examples:

```bash
set +u
source env.sh
set -u
```

Use the `set +u` form when the caller shell has `set -u`, because `env.sh`
appends to variables such as `PYTHONPATH` and `LD_LIBRARY_PATH`.

## Inspect A5VM MLIR

Use this when you need to look at the A5VM-stage IR before deciding whether to
continue to textual LLVM IR, LLVM bitcode, or the full artifact assembly flow.

Canonical flag:

```bash
--a5vm-print-ir
```

Example:

```bash
source env.sh
PTOAS_BIN="$PWD/build/tools/ptoas/ptoas" \
PTOAS_OUT_DIR=/tmp/ptoas-a5vm-ir \
PTOAS_FLAGS='--pto-arch a5 --pto-backend=a5vm --a5vm-print-ir' \
./test/samples/runop.sh -t Abs
```

Use this output to:
- confirm the lowering has reached the A5VM dialect you expect
- inspect whether a transformation issue appears before LLVM export
- compare the A5VM MLIR path against the later LLVM IR or bitcode output

## Export Paths

### LLVM bitcode export

Use:

```bash
--pto-backend=a5vm --a5vm-emit-hivm-bc
```

Example:

```bash
source env.sh
PTOAS_BIN="$PWD/build/tools/ptoas/ptoas" \
PTOAS_OUT_DIR=/tmp/ptoas-a5vm-hivm-bc \
PTOAS_FLAGS='--pto-arch a5 --pto-backend=a5vm --a5vm-emit-hivm-bc' \
./test/samples/runop.sh -t Abs
```

Typical outputs:
- `/tmp/ptoas-a5vm-hivm-bc/Abs/abs-pto-ir.pto`
- `/tmp/ptoas-a5vm-hivm-bc/Abs/abs-pto.cpp`

Important:
- the payload is written to `*-pto.cpp` even in bitcode mode
- that file is LLVM bitcode, not C++ source

Bitcode checks:

```bash
file /tmp/ptoas-a5vm-hivm-bc/Abs/abs-pto.cpp
xxd -l 16 /tmp/ptoas-a5vm-hivm-bc/Abs/abs-pto.cpp
"$LLVM_ROOT/bin/llvm-dis" /tmp/ptoas-a5vm-hivm-bc/Abs/abs-pto.cpp -o - | sed -n '1,80p'
```

Expected signs:
- `file` reports `LLVM IR bitcode`
- the header starts with `42 43 c0 de`
- `llvm-dis` shows HiVM/LLVM content

### Textual LLVM IR export

Use:

```bash
--pto-backend=a5vm --a5vm-emit-hivm-llvm
```

Example:

```bash
source env.sh
PTOAS_BIN="$PWD/build/tools/ptoas/ptoas" \
PTOAS_OUT_DIR=/tmp/ptoas-a5vm-hivm-llvm \
PTOAS_FLAGS='--pto-arch a5 --pto-backend=a5vm --a5vm-emit-hivm-llvm' \
./test/samples/runop.sh -t Abs
```

Typical output:
- `/tmp/ptoas-a5vm-hivm-llvm/Abs/abs-pto.cpp`

Important:
- despite the `.cpp` suffix, this file is textual LLVM IR
- compile it with `-x ir`

Suggested progression:
- start with `--a5vm-print-ir` when the user wants the intermediate A5VM form
- use `--a5vm-emit-hivm-llvm` when the user wants textual LLVM IR
- use `--a5vm-emit-hivm-bc` when the user wants real LLVM bitcode

## Compile The Export With Bisheng

Load the CANN environment first:

```bash
source /usr/local/Ascend/cann/set_env.sh
```

### Compile bitcode to a device object

Preferred:

```bash
bisheng \
  --target=hiipu64-hisilicon-cce \
  -march=dav-c310-vec \
  --cce-aicore-arch=dav-c310-vec \
  --cce-aicore-only \
  -c -x ir /tmp/ptoas-a5vm-hivm-bc/Abs/abs-pto.cpp \
  -o /tmp/ptoas-a5vm-hivm-bc/Abs/abs-pto.o
```

Alternative:
- copy or rename the payload to `.bc`
- compile without relying on the misleading `.cpp` suffix

### Compile textual LLVM IR to a device object

```bash
bisheng \
  --target=hiipu64-hisilicon-cce \
  -march=dav-c310-vec \
  --cce-aicore-arch=dav-c310-vec \
  --cce-aicore-only \
  -c -x ir /tmp/ptoas-a5vm-hivm-llvm/Abs/abs-pto.cpp \
  -o /tmp/abs_ir_path_artifacts/kernel_from_llvm_ir.o
```

Checks:
- keep `-march` and `--cce-aicore-arch` aligned with the intended testcase arch
- for the LLVM IR path, the resulting object should not retain unresolved
  `llvm.hivm.*` symbols

## If You Need The Real Compiler-Expected Intrinsic Shape

Do not infer the final `llvm.hivm.*` operand contract from repo-local emitter
code alone.

When a hand-written LLVM IR path fails in instruction selection or appears to
miscompile, use this trace order:

1. confirm the installed PTO wrapper path first with `pto-a5-installed-impl-trace`
2. generate the normal testcase kernel source through the working emitc path
3. inspect testcase compile flags from:
   - `<testcase>/build/CMakeFiles/<target>.dir/flags.make`
   - `<testcase>/build/CMakeFiles/<target>.dir/build.make`
4. rerun that same `bisheng` compile with `-v` and `-save-temps`
5. inspect:
   - `*.ccei` to confirm the wrapper builtin sequence
   - `strings *.bc | rg 'llvm.hivm\\.'` to see which HIVM intrinsics survive
6. if builtin names still are not enough, extract the exact frontend-produced
   LLVM IR by replaying the `cc1` invocation from `-v` with `-emit-llvm -S`

Use this when you need to answer questions such as:
- is the intrinsic name correct but the mask form wrong
- did the compiler expect a `plt/pset` result instead of a literal mask
- is the LLVM IR path missing hidden frontend-generated structure or attrs

This is the preferred way to align repo-local LLVM emission with the real
compiler contract.

## Assemble Fat Objects And Shared Libraries

Use this only when the validation flow needs a replacement kernel library built
from the LLVM path. The canonical example below uses the generated `Abs`
testcase, but the pattern is the same for other testcases: take the testcase
wrapper source, embed the device object, pack it with `cce-ld`, then link the
shared kernel library.

Required testcase artifacts:
- a wrapper source such as `/tmp/ptoas-npu-validation-run/Abs/abs/abs_kernel.cpp`
- a built launch object such as
  `/tmp/ptoas-npu-validation-run/Abs/abs/build/CMakeFiles/abs_kernel.dir/launch.cpp.o`

### 1. Build the host stub object

```bash
/usr/local/Ascend/cann-9.0.0/tools/bisheng_compiler/bin/bisheng -cc1 \
  -triple aarch64-unknown-linux-gnu \
  -fcce-is-host \
  -fcce-fatobj-compile \
  -fcce-include-aibinary /tmp/abs_ir_path_artifacts/kernel_from_llvm_ir.o \
  -fcce-device-module-id a55ab1efc0defeed \
  -fcce-aicore-arch dav-c310-vec \
  -x cce /tmp/ptoas-npu-validation-run/Abs/abs/abs_kernel.cpp \
  -o /tmp/abs_ir_path_artifacts/kernel_host_stub.o
```

### 2. Pack the fat relocatable object

```bash
/usr/local/Ascend/cann-9.0.0/bin/cce-ld \
  /usr/local/Ascend/cann-9.0.0/bin/ld.lld \
  -x \
  -cce-lite-bin-module-id a55ab1efc0defeed \
  -cce-aicore-arch=dav-c310-vec \
  -r \
  -o /tmp/abs_ir_path_artifacts/kernel_fat.o \
  -cce-stub-dir /usr/local/Ascend/cann-9.0.0/tools/bisheng_compiler/lib/clang/15.0.5/include/cce_stub \
  -cce-install-dir /usr/local/Ascend/cann-9.0.0/tools/bisheng_compiler/bin \
  -cce-inputs-number 1 \
  /tmp/abs_ir_path_artifacts/kernel_host_stub.o
```

The module id must match between:
- `-fcce-device-module-id`
- `-cce-lite-bin-module-id`

### 3. Link the shared kernel library

```bash
mkdir -p /tmp/abs_ir_path_artifacts/link_try
cd /tmp/abs_ir_path_artifacts/link_try
/usr/local/Ascend/cann-9.0.0/bin/bisheng \
  -fPIC -s -Wl,-z,relro -Wl,-z,now --cce-fatobj-link \
  -shared -Wl,-soname,libabs_kernel.so \
  -o libabs_kernel.so \
  /tmp/abs_ir_path_artifacts/kernel_fat.o \
  /tmp/ptoas-npu-validation-run/Abs/abs/build/CMakeFiles/abs_kernel.dir/launch.cpp.o
```

This skill stops at producing the replacement artifact. To run the testcase
with that library and validate outputs, switch back to `ptoas-npu-validation-a5`.

## Failure Modes

Report the first concrete blocker:
- `--a5vm-print-ir`, `--a5vm-emit-hivm-bc`, or `--a5vm-emit-hivm-llvm` used without `--pto-backend=a5vm`
- `--a5vm-emit-hivm-bc` or `--a5vm-emit-hivm-llvm` used without `--pto-backend=a5vm`
- `env.sh` was not sourced, or failed under `set -u`
- `bisheng` was not found or CANN environment was not loaded
- a bitcode payload was treated as source because it kept a misleading suffix
- the testcase wrapper or `launch.cpp.o` is missing for the fatobj path
- the module ids used for stub creation and `cce-ld` packing do not match

## Reporting Back

When you use this skill, report:
- whether the user-facing artifact of interest was A5VM MLIR, textual LLVM IR, or LLVM bitcode
- the exact `ptoas` flags used
- whether the export was A5VM MLIR, LLVM bitcode, or textual LLVM IR
- the exact output path that contains the exported payload
- whether `llvm-dis`, `file`, or direct inspection confirmed the payload type
- whether `bisheng` produced a device object
- whether the flow also produced a fat relocatable object or shared kernel library
- which step was the first blocker, if the full artifact chain did not complete
