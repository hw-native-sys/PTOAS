# ptoas (PTO Assembler & Optimizer)

## 1. Introduction

**ptoas** is a specialized compiler toolchain built on top of the **LLVM19 VPTO branch (`vpto-dev/llvm-project:feature-vpto`)**, designed specifically for **PTO Bytecode** (Programming Tiling Operator Bytecode).

Acting as the bridge between upper-level AI frameworks and underlying NPU/GPGPU/CPU hardware, `ptoas` is built in an **Out-of-Tree** architecture and provides complete C++ and Python interfaces. Its primary responsibilities include:

1. **IR Parsing & Verification**: Parses `.pto` input files and verifies the semantic correctness of PTO Dialect operations (Ops).
2. **Compilation & Optimization (Passes)**: Executes optimization passes targeting the Da Vinci Architecture, such as operator fusion and automatic synchronization insertion.
3. **Code Generation (Lowering)**: Supports lowering PTO IR to `EmitC` / `Linalg` dialects, ultimately generating code that calls the `pto-isa` C++ library.
4. **Python Bindings**: Provides seamlessly integrated Python modules. Through integration with MLIR Core bindings, frameworks such as **PyPTO**, **PTODSL**, and **CuTile** can build, manipulate, and compile PTO Bytecode directly from Python.

---

## 2. Directory Structure

```text
PTOAS/
├── include/
│   └── PTO/               # PTO Dialect headers and TableGen definitions (.td)
├── lib/
│   ├── PTO/               # Dialect core implementation (IR) and Pass logic (Transforms)
│   ├── CAPI/              # C language interface exposure
│   └── Bindings/Python/   # Python Binding C++ implementation (Pybind11)
├── python/                # Python module build scripts and helper code
├── test/
│   └── samples/           # Test cases
├── tools/
│   ├── ptoas/             # ptoas command-line tool entry point (Output: ptoas)
│   └── ptobc/             # ptobc command-line tool entry point (Output: ptobc)
└── CMakeLists.txt         # Top-level build configuration
```

---

## 3. Build Instructions

⚠️ **Important**: This project strictly requires the **LLVM19 VPTO branch `vpto-dev/llvm-project:feature-vpto`**.

### 3.0 Environment Variable Configuration

To simplify the build process, **first modify and run the following commands according to your environment**. Subsequent steps reference these variables directly.

```bash
# ================= Configuration (edit here) =================
# Set your workspace root directory
# (recommended: a dedicated directory for LLVM and PTOAS)
export WORKSPACE_DIR=$HOME/llvm-workspace

# LLVM source and build paths
export LLVM_SOURCE_DIR=$WORKSPACE_DIR/llvm-project
export LLVM_BUILD_DIR=$LLVM_SOURCE_DIR/build-shared

# PTOAS source path
export PTO_SOURCE_DIR=$WORKSPACE_DIR/PTOAS
# =============================================================

# Create the workspace directory
mkdir -p $WORKSPACE_DIR

# Clone PTOAS first: the venv lives inside the PTOAS checkout
# ($PTO_SOURCE_DIR/.venv) to keep the workspace self-contained, so
# $PTO_SOURCE_DIR must exist before creating it.
cd $WORKSPACE_DIR
git clone https://github.com/hw-native-sys/PTOAS.git PTOAS
cd $PTO_SOURCE_DIR

# Use an isolated environment. LLVM and PTOAS must use the same Python.
python3 -m venv "$PTO_SOURCE_DIR/.venv"
source "$PTO_SOURCE_DIR/.venv/bin/activate"
export PYTHON_BIN="$(command -v python3)"
```

### 3.1 Prerequisites

* **OS**: Linux (Ubuntu 20.04+ recommended)
* **Compiler**: GCC >= 9 or Clang (C++17 support required)
* **Build System**: CMake >= 3.20, Ninja
* **Python**: 3.10+
* **Python Packages**: `scikit-build-core`, `pybind11<3`, `numpy`

```bash
"$PYTHON_BIN" -m pip install "scikit-build-core>=0.12.2,<2" "pybind11<3" numpy
```

> **Note**: The current LLVM/MLIR Python bindings are not compatible with `pybind11` 3.x.
> If you encounter errors like `def_property family does not currently support keep_alive`
> when building LLVM, run the downgrade command above first.

### 3.2 Step 1: Build LLVM/MLIR (Dependency)

Download the VPTO-adapted LLVM source, check out the `feature-vpto` branch, and build with **shared libraries** to ensure correct linking for Python bindings.

```bash
# 1. Clone LLVM
cd $WORKSPACE_DIR
git clone https://github.com/vpto-dev/llvm-project.git
cd $LLVM_SOURCE_DIR

# 2. [Critical] Check out the VPTO adaptation branch
git checkout feature-vpto

# 3. Configure CMake (build shared libs with Python bindings enabled)
cmake -G Ninja -S llvm -B $LLVM_BUILD_DIR \
    -DLLVM_ENABLE_PROJECTS="mlir;clang" \
    -DBUILD_SHARED_LIBS=ON \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DPython3_EXECUTABLE="$PYTHON_BIN" \
    -DPython_EXECUTABLE="$PYTHON_BIN" \
    -Dpybind11_DIR="$("$PYTHON_BIN" -m pybind11 --cmakedir)" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_TARGETS_TO_BUILD="host"

# 4. Build LLVM (this step takes a long time)
ninja -C $LLVM_BUILD_DIR
```

### 3.3 Step 2: Build PTOAS (Out-of-Tree)

PTOAS was already cloned in section 3.0; build it against the LLVM 19 you
just compiled.

```bash
# 1. Enter the PTOAS source directory (cloned in section 3.0)
cd $PTO_SOURCE_DIR

# 2. Install into the current Python environment while keeping a persistent,
#    incrementally reusable build tree.
PYTHON_BIN="$PYTHON_BIN" \
LLVM_BUILD_DIR="$LLVM_BUILD_DIR" \
PTO_BUILD_DIR="$PTO_SOURCE_DIR/build" \
  ./quick_install.sh
```

`quick_install.sh` uses an editable install with build isolation disabled so a
temporary build environment's pybind11 path is not persisted in
`CMakeCache.txt`. The `ptoas` command is installed into the environment that
owns `PYTHON_BIN`. Activating the virtual environment created above puts its
`bin` directory on `PATH`.

After installation, configure the runtime environment in section 4 before
running either `ptoas` or `check-pto`.

### 3.4 Step 3: Supported Python Install Flows

If you want to use Python bindings or PTODSL, prefer the repository-root
`ptoas` package contract instead of manually patching `PYTHONPATH`.

```bash
# 1) Released or CI-built wheel: installs PTOAS + PTODSL together
"$PYTHON_BIN" -m pip install /path/to/ptoas*.whl

# 2) Non-editable source install from the repository root
cd $PTO_SOURCE_DIR
"$PYTHON_BIN" -m pip install . --no-build-isolation

# 3) Editable install for PTOAS / PTODSL developers
cd $PTO_SOURCE_DIR
"$PYTHON_BIN" -m pip install -e . --no-build-isolation
```

After installation, the following imports should work directly:

```python
import ptodsl
from ptodsl import pto, scalar
from ptoas.mlir.dialects import pto as mlir_pto
```

> Notes:
> - The `ptoas` wheel also installs PTODSL.
> - VMI release wheels publish under the Python project name `ptoas-vmi`; the wheel filename is normalized to `ptoas_vmi-*.whl`.
> - The VMI release line keeps the `ptoas` CLI name; `ptoas --version` prints `ptoas vmi A.B.C`.
> - The `ptoas` and `ptoas-vmi` release wheels are **mutually exclusive**. They both install the same top-level `ptoas` Python package and `ptoas` console script, so do **not** install them into the same Python environment. Mixing them will overwrite files, and uninstalling one can break the other.
> - `ptoas-bin-*.tar.gz` compiler-only tarballs provide CLI/toolchain bits, not a PTODSL-capable Python distribution.
> - Release tags use `ptoas-vX.Y` for the main toolchain and `vmi-vA.B.C` for the `ptoas-vmi` distribution. Before creating a VMI release tag, update the version in `packaging/ptoas-vmi/pyproject.toml.patch` to the matching `A.B.C` through the release PR. A VMI build exports the current Git revision, applies that metadata patch only in a staging tree, and builds the wheel directly from that tree. It neither modifies the checkout's top-level `pyproject.toml` nor generates or publishes an sdist in ordinary gates or releases.
> - `--no-build-isolation` keeps pip from baking a temporary pybind11 path into `CMakeCache.txt`, which would break later `ninja` reconfigure runs after the temporary virtual environment is removed.

If you previously ran `pip install -e .` without the flag and your build is now broken, fix the existing `CMakeCache.txt` with:

```bash
cmake -B build -Dpybind11_DIR="$("$PYTHON_BIN" -m pybind11 --cmakedir)"
```

---

## 4. Runtime Environment

In every new shell, restore the path variables from section 3.0 and reactivate
the Python environment that owns the PTOAS installation. Source and editable
installs use shared libraries from the external LLVM build tree, so add that
directory to the runtime library search path:

```bash
# Re-export WORKSPACE_DIR, LLVM_BUILD_DIR, and the other paths from section 3.0.
source "$PTO_SOURCE_DIR/.venv/bin/activate"
export PYTHON_BIN="$(command -v python3)"
export LD_LIBRARY_PATH="$LLVM_BUILD_DIR/lib:${LD_LIBRARY_PATH:-}"

command -v ptoas
ptoas --version
```

Source developers can reuse the retained build tree after completing the
runtime setup above:

```bash
ninja -C "$PTO_SOURCE_DIR/build" check-pto
```

Release wheels carry their runtime dependencies and do not require this
`LD_LIBRARY_PATH` when no external LLVM build tree is used. Neither installation
flow requires manually assembling `PYTHONPATH`.

Load CANN's public environment setup only when CANN, Bisheng, the simulator, or
an NPU is required. Select the path that exists in your environment:

```bash
source /usr/local/Ascend/cann/set_env.sh
# or
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
```

Without a virtual environment, if pip uses the user installation scheme, set
`PATH` before running `ptoas` and then configure the same `LD_LIBRARY_PATH`
shown above:

```bash
export PATH="$(python3 -m site --user-base)/bin:$PATH"
hash -r
export LD_LIBRARY_PATH="$LLVM_BUILD_DIR/lib:${LD_LIBRARY_PATH:-}"
command -v ptoas
ptoas --version
```

---

## 5. Usage

### 5.1 Command-Line Interface (CLI)

```bash
# Parse and print PTO IR
ptoas test/lit/pto/empty_func.pto

# Run the AutoSyncInsert pass
ptoas test/lit/pto/empty_func.pto --enable-insert-sync -o outputfile.cpp

# Specify target hardware architecture (A3 / A5)
ptoas test/lit/pto/empty_func.pto --pto-arch=a5 -o outputfile.cpp

# Specify build level (level3 disables PlanMemory/InsertSync)
ptoas test/lit/pto/empty_func.pto --pto-level=level3 -o outputfile.cpp

# Print the current ptoas release version
ptoas --version
```

### 5.2 Python API

In a supported `ptoas` install environment, both the PTO Dialect and PTODSL
can be imported directly.

```python
from ptoas.mlir.ir import Context, Module, Location
# PTOAS ships its MLIR Python API in the ptoas.mlir namespace.
from ptoas.mlir.dialects import pto
from ptodsl import pto as jit_pto, scalar

with Context() as ctx, Location.unknown():
    pto.register_dialect(ctx, load=True)
    module = Module.create()
    print("PTO Dialect registered successfully!")
    print("PTODSL imported successfully!", jit_pto, scalar)
```

### 5.3 Running Tests

```bash
# Recommended: enter a supported PTOAS / PTODSL install environment first
cd $PTO_SOURCE_DIR
"$PYTHON_BIN" -m pip install -e . --no-build-isolation

# Run Python binding tests
cd $PTO_SOURCE_DIR/test/samples/MatMul/
"$PYTHON_BIN" ./tmatmulk.py > ./tmatmulk.pto

# Run ptoas tests
ptoas ./tmatmulk.pto -o ./tmatmulk.cpp
```

### 5.4 On-Board Validation

This flow generates NPU validation test cases from the `.cpp` files produced by ptoas (under `test/samples/`) and runs them on an NPU. The example below reuses `MatMul/tmatmulk.cpp` generated in section 5.3.

> For compile-only validation on a machine without an NPU card, see [docs/no_npu_compile_only_guide_zh.md](docs/no_npu_compile_only_guide_zh.md).

```bash
# The relative paths below start at the repository root.
cd "$PTO_SOURCE_DIR"

# 1) Generate the npu_validation test directory
#    (creates npu_validation/ under the current sample directory)

# A2/A3 example:
python3 test/npu_validation/scripts/generate_testcase.py \
  --input test/samples/MatMul/tmatmulk.cpp \
  --run-mode npu \
  --soc-version Ascend910B1

# A5 example:
python3 test/npu_validation/scripts/generate_testcase.py \
  --input test/samples/MatMul/tmatmulk.cpp \
  --run-mode npu \
  --soc-version Ascend950

# 2) Run validation (run.sh requires no additional arguments)
test/samples/MatMul/npu_validation/tmatmulk/run.sh
```

Notes:
- `test/samples/MatMul/npu_validation/tmatmulk/` will contain `tmatmulk_kernel.cpp`, `main.cpp`, `golden.py`, `compare.py`, `run.sh`, and `CMakeLists.txt`.
- `golden.py` generates random inputs by default; outputs default to all zeros (only the count, shape, and data type of inputs/outputs match the kernel parameters).
- `compare.py` compares `golden*.bin` against `output*.bin` and reports an error if they differ.
