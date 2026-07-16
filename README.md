# ptoas (PTO Assembler & Optimizer)

## 1. 项目简介 (Introduction)

**ptoas** (`ptoas`) 是一个基于 **LLVM/MLIR LLVM21 VPTO 分支 (`vpto-dev/llvm-project:feature-vpto-llvm21`)** 框架构建的专用编译器工具链，专为 **PTO Bytecode** (Programming Tiling Operator Bytecode) 设计。

作为连接上层 AI 框架与底层各类NPU/GPGPU/CPU硬件，`ptoas` 采用 **Out-of-Tree** 架构构建，提供了完整的 C++ 与 Python 接口，主要职责包括：

1. **IR 解析与验证**：解析 `.pto` 输入文件，验证 PTO Dialect 操作（Ops）的语义正确性。
2. **编译优化 (Passes)**：执行针对达芬奇架构（Da Vinci Architecture）的特定优化 Pass，如算子融合、自动同步插入策略等。
3. **代码生成 (Lowering)**：支持将 PTO IR 下降（Lowering）到 `EmitC` / `Linalg` Dialect，最终生成可调用 `pto-isa` C++ 库的代码。
4. **Python 绑定 (Python Bindings)**：提供无缝集成的 Python 模块。通过与 MLIR Core 绑定集成，支持 **PyPTO**、**PTODSL**、**CuTile** 等框架在 Python 端直接构建、操作和编译 PTO Bytecode。

---

## 2. 目录结构 (Directory Structure)

```text
PTOAS/
├── include/
│   └── PTO/               # PTO Dialect 的头文件与 TableGen 定义 (.td)
├── lib/
│   ├── PTO/               # Dialect 核心实现 (IR) 与 Pass 逻辑 (Transforms)
│   ├── CAPI/              # C 语言接口暴露
│   └── Bindings/Python/   # Python Binding C++ 实现 (Pybind11)
├── python/                # Python 模块构建脚本与辅助代码
├── test/
│   └── samples/           # 测试用例
├── tools/
│   ├── ptoas/             # ptoas 命令行工具入口 (Output: ptoas)
│   └── ptobc/             # ptobc 命令行工具入口 (Output: ptobc)
└── CMakeLists.txt         # 顶级构建配置

```

---

## 3. 构建指南 (Build Instructions)

⚠️ **重要提示**：本项目严格依赖 **LLVM21 VPTO 分支 `vpto-dev/llvm-project:feature-vpto-llvm21`**。


### 3.0 环境变量配置 (Configuration)

为了简化构建流程，**请首先根据您的实际环境修改并运行以下命令**。后续步骤将直接引用这些变量。

```bash
# ================= 配置区域 (请修改这里) =================
# 设置您的工作根目录 (建议创建一个专门的目录存放 LLVM 和 PTOAS)
export WORKSPACE_DIR=$HOME/llvm-workspace

# LLVM 源码与构建路径
export LLVM_SOURCE_DIR=$WORKSPACE_DIR/llvm-project
export LLVM_BUILD_DIR=$LLVM_SOURCE_DIR/build-shared

# PTOAS 源码与安装路径
export PTO_SOURCE_DIR=$WORKSPACE_DIR/PTOAS
export PTO_INSTALL_DIR=$PTO_SOURCE_DIR/install
# =======================================================

# 创建工作目录
mkdir -p $WORKSPACE_DIR

```

### 3.1 环境准备 (Prerequisites)

* **OS**: Linux (Ubuntu 20.04+ 推荐)
* **Compiler**: GCC >= 9 或 Clang (支持 C++17)
* **Build System**: CMake >= 3.20, Ninja
* **Python**: 3.8+
* **Python Packages**: `pybind11<3`, `nanobind`, `numpy`
```bash
python3 -m pip install 'pybind11<3' nanobind numpy

```

> 说明：当前 PTOAS Python 扩展继续使用 `pybind11`，LLVM21 的 MLIR Python 绑定构建需要 `nanobind`。
> 当前 LLVM/MLIR Python 绑定与 `pybind11` 3.x 不兼容。
> 如果编译 LLVM 时遇到 `def_property family does not currently support keep_alive` 等报错，
> 请确认使用上面的 `pybind11<3` 依赖。



### 3.2 第一步：构建 LLVM/MLIR (Dependency)

我们需要下载 VPTO 适配后的 LLVM 源码，切换到 `feature-vpto-llvm21` 分支，并以**动态库 (Shared Libs)** 模式编译，以确保 Python Binding 的正确链接。

```bash
# 1. 下载 LLVM 源码
cd $WORKSPACE_DIR
git clone https://github.com/vpto-dev/llvm-project.git
cd $LLVM_SOURCE_DIR

# 2. [关键] 切换到 VPTO 适配分支
git checkout feature-vpto-llvm21

# 3. 配置 CMake (构建动态库并启用 Python 绑定)
cmake -G Ninja -S llvm -B $LLVM_BUILD_DIR \
    -DLLVM_ENABLE_PROJECTS="mlir;clang" \
    -DBUILD_SHARED_LIBS=ON \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DPython3_EXECUTABLE=$(which python3) \
    -DPython_EXECUTABLE=$(which python3) \
    -Dpybind11_DIR=$(python3 -m pybind11 --cmakedir) \
    -Dnanobind_DIR=$(python3 -m nanobind --cmake_dir) \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_TARGETS_TO_BUILD="host"

# 4. 编译 LLVM (这一步耗时较长)
ninja -C $LLVM_BUILD_DIR

```

### 3.3 第二步：构建 PTOAS (Out-of-Tree)

下载 PTOAS 源码并基于刚刚编译好的 LLVM 21 进行构建。

```bash
# 1. 下载 PTOAS 源码
cd $WORKSPACE_DIR
git clone https://gitcode.com/cann/pto-as.git PTOAS
cd $PTO_SOURCE_DIR

# 2. 获取 pybind11 的 CMake 路径
export PYBIND11_CMAKE_DIR=$(python3 -m pybind11 --cmakedir)

# 3. 配置 CMake
# 注意：此处直接使用了 3.0 章节中定义的变量，无需手动修改
cmake -G Ninja \
    -S . \
    -B build \
    -DLLVM_DIR=$LLVM_BUILD_DIR/lib/cmake/llvm \
    -DMLIR_DIR=$LLVM_BUILD_DIR/lib/cmake/mlir \
    -DPython3_EXECUTABLE=$(which python3) \
    -DPython3_FIND_STRATEGY=LOCATION \
    -Dpybind11_DIR="${PYBIND11_CMAKE_DIR}" \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DMLIR_PYTHON_PACKAGE_DIR=$LLVM_BUILD_DIR/tools/mlir/python_packages/mlir_core \
    -DCMAKE_INSTALL_PREFIX="$PTO_INSTALL_DIR"

# 4. 编译并安装
ninja -C build-llvm21
ninja -C build-llvm21 install

# 5. 检查构建产物
# build 输出（便于本地开发/调试）
$PTO_SOURCE_DIR/build-llvm21/python/
├── mlir
│   ├── _mlir_libs
│   │   └── _pto.cpython-*.so
│   └── dialects
│       ├── pto.py
│       └── _pto_ops_gen.py

# install 输出（Python 方言文件和原生扩展）
$PTO_INSTALL_DIR/
└── mlir
    ├── dialects
    │   ├── pto.py
    │   └── _pto_ops_gen.py
    └── _mlir_libs
        └── _pto.cpython-*.so

# CLI 工具
$PTO_SOURCE_DIR/build-llvm21/tools/ptoas/ptoas
$PTO_SOURCE_DIR/build-llvm21/tools/ptobc/ptobc

```

### 3.4 Python 安装合同 (Python Distribution Contract)

如果你要使用 Python 绑定、PTODSL资源，推荐使用仓库根目录
`ptoas` 包的安装合同，而不是手动拼 `PYTHONPATH`：

```bash
# 非 editable 的源码安装
cd $PTO_SOURCE_DIR
pip install . --no-build-isolation

# PTOAS / PTODSL 开发者的 editable 安装
cd $PTO_SOURCE_DIR
pip install -e . --no-build-isolation
```

发布或 CI 产出的 `ptoas` wheel 也遵循同一合同：

```bash
pip install /path/to/ptoas-*.whl
```

安装完成后，以下导入应直接可用：

```python
import ptodsl
from ptodsl import pto, scalar
from mlir.dialects import pto as mlir_pto
```

> 说明：
> - `ptoas` wheel 会同时安装 PTODSL，并提供可直接调用的 `ptoas` CLI。
> - `ptoas-bin-*.tar.gz` 这类 compiler-only 二进制 tarball 只提供 CLI/toolchain，
>   **不是** PTODSL-capable Python distribution；仅解压 tarball 不能保证
>   `import ptodsl` 可用。
> - release tag 约定：`ptoas-vX.Y` 发布主工具链，`vmi-vA.B.C` 发布 VMI 文档/规范。

---

## 4. 运行环境配置 (Runtime Environment)

如果你已经通过 `pip install .`、`pip install -e .` 或 `pip install ptoas-*.whl`
完成安装，那么 `import ptodsl` / `from mlir.dialects import pto` / `ptoas`
都不应再依赖手动设置 `PYTHONPATH`。

下面这组环境变量主要用于**直接消费 build/install tree** 的场景，例如：

- 不走 pip 安装，直接调试 CMake install 输出
- 调试 `ptoas` CLI、动态库搜索路径或 MLIR Python overlay
- 复用仓库脚本做 compile-only / simulator / sample 生成

您可以将以下命令添加到 `.bashrc` 或启动脚本中。

```bash
# --- 运行时变量配置 (基于之前定义的路径) ---

# 1. Python Path: 拼接 MLIR Core 和 PTO Core
#    这样在 python 中 import mlir.dialects.pto 时能正确找到
export MLIR_PYTHON_ROOT=$LLVM_BUILD_DIR/tools/mlir/python_packages/mlir_core
export PTO_PYTHON_ROOT=$PTO_INSTALL_DIR/
export PYTHONPATH=$PTO_PYTHON_ROOT:$MLIR_PYTHON_ROOT:$PYTHONPATH

# 2. Library Path: 确保能加载 LLVM 和 PTO 的动态库
export LD_LIBRARY_PATH=$LLVM_BUILD_DIR/lib:$PTO_INSTALL_DIR/lib:$LD_LIBRARY_PATH

# 3. PATH: 将 ptoas / ptobc 添加到命令行路径
export PATH=$PTO_SOURCE_DIR/build-llvm21/tools/ptoas:$PTO_SOURCE_DIR/build-llvm21/tools/ptobc:$PATH

```

---

## 5. 使用方法 (Usage)

### 5.1 命令行工具 (CLI)

```bash
# 解析并打印 PTO IR
ptoas test/lit/pto/empty_func.pto

# 运行 AutoSyncInsert Pass
ptoas test/lit/pto/empty_func.pto --enable-insert-sync -o outputfile.cpp

# 指定目标硬件架构（A3 / A5）
ptoas test/lit/pto/empty_func.pto --pto-arch=a5 -o outputfile.cpp

# 指定构建 Level（level3 会禁用 PlanMemory/InsertSync）
ptoas test/lit/pto/empty_func.pto --pto-level=level3 -o outputfile.cpp

# VPTO backend 默认启用 VMI -> VPTO 语义 pipeline
# 可使用 --enable-vmi=false 临时关闭
# public function signature 不能直接暴露 !pto.vmi.* 类型
ptoas test/lit/vmi/vmi_ptoas_cli_pipeline.pto --pto-arch=a5 --pto-backend=vpto --emit-vpto -o -

# 查看当前 ptoas release 版本号
ptoas --version

```

### 5.2 Python 接口 (Python API)

在支持的 `ptoas` 安装环境中，PTO Dialect 与 PTODSL 都可以直接导入。

```python
from mlir.ir import Context, Module, Location
# [关键] 从 mlir.dialects 导入 pto，这是 Out-of-tree 绑定的标准用法
from mlir.dialects import pto
from ptodsl import pto as jit_pto, scalar

with Context() as ctx, Location.unknown():
    pto.register_dialect(ctx, load=True)
    module = Module.create()
    print("PTO Dialect registered successfully!")
    print("PTODSL imported successfully!", jit_pto, scalar)

```

### 5.3 运行测试

```bash
# 建议先进入支持的 PTOAS / PTODSL 安装环境
cd $PTO_SOURCE_DIR
pip install -e . --no-build-isolation

# 运行python binding 测试
cd $PTO_SOURCE_DIR/test/samples/MatMul/
python3 ./tmatmulk.py > ./tmatmulk.pto

# 运行ptoas 测试
$PTO_SOURCE_DIR/build/tools/ptoas/ptoas ./tmatmulk.pto -o ./tmatmulk.cpp
```

### 5.4 上板验证

该流程用于将 `test/samples` 下生成的 `.cpp`（ptoas 输出）自动生成 NPU 验证用例，并在 NPU 上运行。下面示例直接复用 5.3 里生成的 `MatMul/tmatmulk.cpp`。

> 只想在无卡机器上做 host-side compile-only，请先看 [docs/no_npu_compile_only_guide_zh.md](docs/no_npu_compile_only_guide_zh.md)。


```bash
# 1) 生成 npu_validation 测试目录（会在当前 sample 目录下创建 npu_validation/）
# A2/A3 示例：
python3 test/npu_validation/scripts/generate_testcase.py \
  --input test/samples/MatMul/tmatmulk.cpp \
  --run-mode npu \
  --soc-version Ascend910B1

# A5 示例:
python3 test/npu_validation/scripts/generate_testcase.py \
  --input test/samples/MatMul/tmatmulk.cpp \
  --run-mode npu \
  --soc-version Ascend950

# 2) 运行验证（run.sh 无需额外参数）
test/samples/MatMul/npu_validation/tmatmulk/run.sh
```

说明：
- `test/samples/MatMul/npu_validation/tmatmulk/` 下会生成 `tmatmulk_kernel.cpp / main.cpp / golden.py / compare.py / run.sh / CMakeLists.txt`
- `golden.py` 默认生成随机输入，输出默认全零（只保证输入/输出数量、shape、datatype 与 kernel 参数一致）
- `compare.py` 负责对比 `golden*.bin` 与 `output*.bin`，不一致时会报错

---
