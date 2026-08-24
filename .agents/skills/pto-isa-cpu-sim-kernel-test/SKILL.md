---
name: pto-isa-cpu-sim-kernel-test
description: Validate a `.pto` program or `ptoas`-generated C++ kernel with the local `pto-isa` CPU simulator in this repository. Covers generating `.pto` files from PTOAS samples, running `ptoas` to emit C++, grafting the emitted kernel into a testcase under `.downloads/pto-isa/tests/cpu/st/testcase/`, updating `main.cpp`, `gen_data.py`, and CMake wiring as needed, then running `tests/run_cpu.py` for functional testing and debug on Windows, WSL, or Linux.
---

# PTO ISA CPU Sim Kernel Test

## Overview

Use this skill for functional validation with the local `pto-isa` CPU simulator, not CA model or board execution. Prefer the local checkout at `.downloads/pto-isa`; if that path is missing, find another local clone before editing or running tests.

When both Windows PowerShell and WSL are available, prefer WSL first for `ptoas`, Python, CMake, and CPU sim test execution. Use Windows-native execution only when WSL is unavailable or the task is explicitly Windows-specific.

## Workflow

1. Confirm the source artifact and execution target.

- If the user already has a `.pto` file, use it directly.
- If the user starts from a PTOAS sample or Python generator, produce the `.pto` file first.
- If the goal is only quick logic sanity checking and the emitted kernel does not expose clean GM inputs or outputs, prefer a standalone CPU sim harness instead of forcing the standard ST structure.

2. Generate C++ from PTOAS.

- Resolve the workspace through `ptoas-development` before compiling. It selects the isolated venv CLI and the matching build tree; do not assume a global `ptoas` or `build/tools/ptoas/ptoas`.
- Save generated artifacts in a scratch folder in the current repository, for example `cpu_sim_debug/<case>/`, so repeated iterations do not immediately dirty the `pto-isa` checkout.
- Typical flow:

```bash
# example: produce .pto from a PTOAS sample
"$PYTHON_BIN" test/samples/MatMul/tmatmulk.py > /tmp/tmatmulk.pto

# emit C++
"$PTOAS_BIN" /tmp/tmatmulk.pto -o /tmp/tmatmulk.cpp
```

3. Choose the integration path.

- Preferred path: integrate the emitted kernel into `.downloads/pto-isa/tests/cpu/st/testcase/<case>` and run `tests/run_cpu.py`.
- Fast debug path: use a standalone CPU sim harness when the emitted kernel has no GM interface, has fixed internal tile addresses only, or is still too unstable for a full ST test.
- In this repository, `cpu_sim_debug/subview_tmatmul/` is a good local example of scratch-space experimentation. The emitted `generated_kernel.cpp` there shows the kind of wrapper-free output that often needs adaptation before it fits the CPU ST harness.

4. Reuse an existing CPU ST testcase before creating a new one.

- Start by inspecting a nearby testcase in `.downloads/pto-isa/tests/cpu/st/testcase/`, such as `tadd` or `tmatmul`.
- Reusing an existing testcase is usually the fastest route because `main.cpp`, `gen_data.py`, and the launch wrapper pattern are already wired.
- Create a brand-new testcase only when the host-side buffer layout or validation logic is too different to fit an existing one.

5. Match the emitted kernel to the CPU ST harness.

- The standard CPU ST layout expects these files:
  - `main.cpp`
  - `<case>_kernel.cpp`
  - `gen_data.py`
  - `CMakeLists.txt`
- Keep the emitted kernel body as intact as possible. Prefer adding a thin launch wrapper around it instead of rewriting the generated function.
- Inspect the emitted entry signature before editing `main.cpp`.
- If the emitted function does not match the existing host launcher, adapt the wrapper or the host code so that buffer ownership, argument order, and output file names are consistent.
- If the emitted kernel has no GM inputs or outputs, do not blindly copy a `tadd`-style harness. Either:
  - write a purpose-built `main.cpp` for that testcase, or
  - move to the standalone CPU sim path for quick verification.

6. Respect `run_cpu.py` assumptions.

- `tests/run_cpu.py` copies `gen_data.py` into the build directory and executes it from there.
- `gen_data.py` must write files relative to the current working directory, not relative to its original source directory.
- The usual `main.cpp` pattern resolves data via `../<TestSuite>.<case>/...`, so `gen_data.py` must create directories named after the actual gtest suite and case names.
- On Windows, `tests/run_cpu.py` requires an explicit `--generator` such as `Ninja` or `MinGW Makefiles`.

7. Wire CMake correctly.

- Existing testcase directories normally contain a one-line `CMakeLists.txt`:

```cmake
pto_cpu_sim_st(<case>)
```

- If creating a new testcase directory, also add the case name to `.downloads/pto-isa/tests/cpu/st/testcase/CMakeLists.txt`.
- If reusing an existing testcase name, top-level testcase registration may not need changes.

8. Run the CPU sim test and iterate.

Linux or WSL:

```bash
cd .downloads/pto-isa
python3 tests/run_cpu.py --testcase <case> --clean --verbose
```

Windows native:

```powershell
cd .downloads\pto-isa
python3 tests\run_cpu.py --testcase <case> --generator Ninja --clean --verbose
```

Useful variants:

```bash
# reuse an existing build
python3 tests/run_cpu.py --testcase <case> --no-build

# run a single gtest
python3 tests/run_cpu.py --testcase <case> --gtest_filter 'Suite.case'
```

9. Debug failures by category.

- Configure or compiler failure: confirm the generator, compiler, and CMake toolchain.
- Build failure in emitted C++: inspect missing macros, wrapper mismatches, or host/kernel signature drift.
- Runtime file-not-found failure: check whether `gen_data.py` wrote to the build directory with the expected `Suite.case` folder names.
- Numeric mismatch: verify golden generation first, then inspect the wrapper and emitted kernel assumptions.

## Local Paths

- Current PTOAS repository: repository root
- Local `pto-isa` checkout: `.downloads/pto-isa`
- CPU ST runner: `.downloads/pto-isa/tests/run_cpu.py`
- CPU ST testcase root: `.downloads/pto-isa/tests/cpu/st/testcase`
- Scratch area example: `cpu_sim_debug/subview_tmatmul/`
- Existing local compile note for emitted C++: `test/compile_cpp/README.md`

## Read Next

Read `references/cpu-st-patterns.md` when you need file-level examples, a checklist for adapting `main.cpp` and `gen_data.py`, or a reminder of the two preferred integration patterns.
