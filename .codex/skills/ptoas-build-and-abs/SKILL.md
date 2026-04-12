---
name: ptoas-build-and-abs
description: Rebuild PTOAS in the repo build directory and compile the Abs sample to inspect generated VPTO output. Use when the user asks to build ptoas, rebuild the current build tree, or run/check the Abs sample output.
---

# PTOAS Build And Abs

Use this skill when the task is specifically about:
- rebuilding `ptoas` in this repo
- doing a full repo build in the repo-local `build/` directory
- compiling `test/samples/Abs/abs.py`
- inspecting the generated VPTO text for `Abs`

## Canonical Commands

### 1. Configure the repo-local build directory

`do_cmake.sh` is the canonical entrypoint. It always targets `./build`.

```bash
./do_cmake.sh --llvm /data/mouliangyu/projects/github.com/llvm/llvm-project/install
```

If `do_cmake.sh` fails because `build/` has a generator mismatch between old Makefiles/Ninja metadata, do not guess. State that `build/` is inconsistent and ask before cleaning the generated build metadata in `build/`.

### 2. Build

For just the CLI:

```bash
CCACHE_DISABLE=1 ninja -C build ptoas
```

For a full repo build:

```bash
CCACHE_DISABLE=1 ninja -C build
```

If the user asked for "full build", prefer the full command above. If they only want to run `Abs`, building `ptoas` is usually enough.

### 3. Prepare runtime environment

Before running `runop.sh`, always:

```bash
source env.sh
```

This sets `PYTHONPATH`, `LD_LIBRARY_PATH`, and the MLIR/PTO python roots needed by the samples.

### 4. Compile `Abs` to VPTO text

Use `runop.sh` with explicit `PTOAS_BIN`, explicit output directory, and A5 backend flags:

```bash
source env.sh
PTOAS_BIN="$PWD/build/tools/ptoas/ptoas" \
PTOAS_OUT_DIR=/tmp/ptoas-abs-vpto \
PTOAS_FLAGS='--pto-arch a5 --pto-backend=vpto --vpto-print-ir' \
./test/samples/runop.sh -t Abs
```

Expected outputs:
- `/tmp/ptoas-abs-vpto/Abs/abs-pto-ir.pto`
- `/tmp/ptoas-abs-vpto/Abs/abs-pto.cpp`

Despite the `.cpp` suffix, on the VPTO backend this file contains the emitted VPTO textual IR.

## Inspection

The main file to show the user is:

```bash
sed -n '1,260p' /tmp/ptoas-abs-vpto/Abs/abs-pto.cpp
```

For quick sanity checks, look for:
- `vpto.copy_gm_to_ubuf`
- `src_strides = [32, 1]`
- `trace_offsets = [0, 0]`
- `trace_sizes = [32, 32]`
- `cce_aiv_loop_hint`
- `llvm.loop.aivector_scope`
- `vpto.vlds`
- `vpto.vabs`
- `vpto.vsts`
- `vpto.copy_ubuf_to_gm`

## Reporting Back

When you ran `Abs`, report:
- whether `ptoas` had to be rebuilt
- the exact generated file path for the VPTO text
- whether the output contains the expected copy-family metadata and vec-scope carrier attrs

If the build fails, include the first concrete blocker:
- generator mismatch in `build/`
- link failure in `ptoas`
- missing runtime env because `env.sh` was not sourced
- missing sample output file
