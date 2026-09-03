# ptoas (PTO Assembler & Optimizer)

## 1. 项目简介 (Introduction)

**ptoas** (`ptoas`) 是一个基于 **LLVM/MLIR LLVM19 VPTO 分支 (`vpto-dev/llvm-project:feature-vpto`)** 框架构建的专用编译器工具链，专为 **PTO Bytecode** (Programming Tiling Operator Bytecode) 设计。

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

⚠️ **重要提示**：本项目严格依赖 **LLVM19 VPTO 分支 `vpto-dev/llvm-project:feature-vpto`**。


### 3.0 环境变量配置 (Configuration)

为了简化构建流程，**请首先根据您的实际环境修改并运行以下命令**。后续步骤将直接引用这些变量。

```bash
# ================= 配置区域 (请修改这里) =================
# 设置您的工作根目录 (建议创建一个专门的目录存放 LLVM 和 PTOAS)
export WORKSPACE_DIR=$HOME/llvm-workspace

# LLVM 源码与构建路径
export LLVM_SOURCE_DIR=$WORKSPACE_DIR/llvm-project
export LLVM_BUILD_DIR=$LLVM_SOURCE_DIR/build-shared

# PTOAS 源码路径
export PTO_SOURCE_DIR=$WORKSPACE_DIR/PTOAS
# =======================================================

# 创建工作目录
mkdir -p $WORKSPACE_DIR

# 先拉取 PTOAS 源码：venv 将创建在 PTOAS 工作区内（$PTO_SOURCE_DIR/.venv），
# 保持工作区自包含，所以需要先有 $PTO_SOURCE_DIR 目录。
cd $WORKSPACE_DIR
git clone https://github.com/hw-native-sys/PTOAS.git PTOAS
cd $PTO_SOURCE_DIR

# 推荐使用独立虚拟环境。后续 LLVM 和 PTOAS 构建必须使用同一个 Python。
python3 -m venv "$PTO_SOURCE_DIR/.venv"
source "$PTO_SOURCE_DIR/.venv/bin/activate"
export PYTHON_BIN="$(command -v python3)"

```

### 3.1 环境准备 (Prerequisites)

* **OS**: Linux (Ubuntu 20.04+ 推荐)
* **Compiler**: GCC >= 9 或 Clang (支持 C++17)
* **Build System**: CMake >= 3.20, Ninja
* **Python**: 3.10+
* **Python Packages**: `scikit-build-core`, `pybind11<3`, `numpy`
```bash
"$PYTHON_BIN" -m pip install 'scikit-build-core>=0.12.2,<2' 'pybind11<3' numpy

```

> 说明：PTOAS 与 LLVM 19 的 MLIR Python 绑定均使用 `pybind11`。
> 当前 LLVM/MLIR Python 绑定与 `pybind11` 3.x 不兼容。
> 如果编译 LLVM 时遇到 `def_property family does not currently support keep_alive` 等报错，
> 请确认使用上面的 `pybind11<3` 依赖。



### 3.2 第一步：构建 LLVM/MLIR (Dependency)

我们需要下载 VPTO 适配后的 LLVM 源码，切换到 `feature-vpto` 分支，并以**动态库 (Shared Libs)** 模式编译，以确保 Python Binding 的正确链接。

```bash
# 1. 下载 LLVM 源码
cd $WORKSPACE_DIR
git clone https://github.com/vpto-dev/llvm-project.git
cd $LLVM_SOURCE_DIR

# 2. [关键] 切换到 VPTO 适配分支
git checkout feature-vpto

# 3. 配置 CMake (构建动态库并启用 Python 绑定)
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

# 4. 编译 LLVM (这一步耗时较长)
ninja -C $LLVM_BUILD_DIR

```

### 3.3 第二步：构建 PTOAS (Out-of-Tree)

PTOAS 源码已在 3.0 拉取，基于刚刚编译好的 LLVM 19 进行构建。

```bash
# 1. 进入 PTOAS 源码目录（3.0 已 clone）
cd $PTO_SOURCE_DIR

# 2. 安装到当前 Python 环境，并保留可增量构建的 build tree
PYTHON_BIN="$PYTHON_BIN" \
LLVM_BUILD_DIR="$LLVM_BUILD_DIR" \
PTO_BUILD_DIR="$PTO_SOURCE_DIR/build" \
  ./quick_install.sh

```

`quick_install.sh` 使用 editable install，并关闭 build isolation，避免把临时
构建环境中的 pybind11 路径写入持久化 `CMakeCache.txt`。`ptoas` 会直接安装到
`PYTHON_BIN` 对应的当前环境中。激活上面创建的虚拟环境后，它的 `bin` 目录已经
位于 `PATH` 中。

安装完成后，必须先按第 4 节配置运行环境，再执行 `ptoas` 或 `check-pto`。

### 3.4 Python 安装合同 (Python Distribution Contract)

如果你要使用 Python 绑定、PTODSL资源，推荐使用仓库根目录
`ptoas` 包的安装合同，而不是手动拼 `PYTHONPATH`：

```bash
# 非 editable 的源码安装
cd $PTO_SOURCE_DIR
"$PYTHON_BIN" -m pip install . --no-build-isolation

# PTOAS / PTODSL 开发者的 editable 安装
cd $PTO_SOURCE_DIR
"$PYTHON_BIN" -m pip install -e . --no-build-isolation
```

发布或 CI 产出的 `ptoas` wheel 也遵循同一合同：

```bash
"$PYTHON_BIN" -m pip install /path/to/ptoas*.whl
```

安装完成后，以下导入应直接可用：

```python
import ptodsl
from ptodsl import pto, scalar
from ptoas.mlir.dialects import pto as mlir_pto
```

> 说明：
> - `ptoas` wheel 会同时安装 PTODSL，并提供可直接调用的 `ptoas` CLI。
> - VMI release 线仍然复用 `ptoas` CLI 名称；对应的 `ptoas --version` 会显示
>   `ptoas vmi A.B.C`。
> - `ptoas` 与 `ptoas-vmi` 两个 release wheel **互斥**：它们都会安装同名的顶层
>   `ptoas` Python 包和 `ptoas` console script，**不要**在同一个 Python 环境里同时安装；
>   混装会互相覆盖文件，卸载其中一个也可能破坏另一个。
> - `ptoas-bin-*.tar.gz` 这类 compiler-only 二进制 tarball 只提供 CLI/toolchain，
>   **不是** PTODSL-capable Python distribution；仅解压 tarball 不能保证
>   `import ptodsl` 可用。
> - release tag 约定：`ptoas-vX.Y` 发布主工具链，`vmi-vA.B.C` 发布
>   `ptoas-vmi` distribution。创建 VMI release tag 前，应通过发布 PR 将
>   `packaging/ptoas-vmi/pyproject.toml.patch` 中的版本更新为相同的 `A.B.C`。
>   VMI 发布流程会从当前 Git revision 导出完整源码快照，只在 staging tree
>   中应用该 metadata patch，再直接从 staging tree 构建 wheel；不会修改工作区
>   根目录的 `pyproject.toml`，也不会在普通门禁或 release 中生成、发布 sdist。

---

## 4. 运行环境配置 (Runtime Environment)

每次打开新 shell 时，先恢复 3.0 中配置的路径变量并重新激活安装 PTOAS 的
Python 环境。editable 安装会将选定 LLVM 构建目录记录为 native extension 的
运行时搜索路径，不需要手工设置 `LD_LIBRARY_PATH`：

```bash
# 先重新导出 WORKSPACE_DIR、LLVM_BUILD_DIR 等 3.0 中的路径变量
source "$PTO_SOURCE_DIR/.venv/bin/activate"
export PYTHON_BIN="$(command -v python3)"

command -v ptoas
ptoas --version
```

源码开发者完成上述配置后，可以复用安装时保留的 build tree 运行测试：

```bash
ninja -C "$PTO_SOURCE_DIR/build" check-pto
```

发布 wheel 自带运行时依赖。无论哪种安装方式，都不需要手工拼接
`PYTHONPATH` 或 `LD_LIBRARY_PATH`。

### Daily wheel

定时构建会将最新 wheel 发布到 GitHub 的 `nightly` release。开发者可以查看
[Nightly Build](https://github.com/hw-native-sys/PTOAS/releases/tag/nightly)，
或在仓库 checkout 中运行下面的命令自动选择当前 Python 和平台对应的 wheel：

```bash
python tools/install_nightly_wheel.py
```

脚本使用当前 Python 自带的 pip packaging 支持选择 wheel，无需预先单独安装
`packaging`。

daily workflow 的实际 Python 版本、平台和架构以 nightly release 中当前发布的
wheel 为准。

如需先查看将要安装的文件，可以加上 `--dry-run`。脚本使用当前 Python
环境执行安装，不会自动重装 wheel 的运行时依赖，并会替换该环境中已安装的同名
nightly wheel。GitHub Release 提供 asset digest 时脚本会自动校验 SHA-256，也可
通过 `--sha256` 显式指定摘要。nightly wheel 来自 GitHub Release，使用前请确认
下载来源和当前环境符合预期。若选中的 asset 超过 48 小时未更新，脚本会给出警告。
Linux 和 macOS nightly release 同时附带 manifest，其中记录各平台 wheel 对应的
源码 commit 和构建任务，跨平台版本不一致时可据此核对。

需要 CANN、Bisheng、simulator 或 NPU 时，再加载 CANN 对外提供的环境脚本。
常见安装位置如下，按实际环境选择一个：

```bash
source /usr/local/Ascend/cann/set_env.sh
# 或
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
```

如果没有使用虚拟环境，并且 pip 将软件包安装到了用户目录，请在运行 `ptoas`
之前先配置 `PATH`：

```bash
export PATH="$(python3 -m site --user-base)/bin:$PATH"
hash -r
command -v ptoas
ptoas --version
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

# VPTO backend 总是启用 VMI -> VPTO 语义 pipeline
# public function signature 不能直接暴露 !pto.vmi.* 类型
ptoas test/lit/vmi_new/vmi_ptoas_cli_pipeline.pto --pto-arch=a5 --pto-backend=vpto --emit-vpto -o -

# 输出 VPTO 调度分析，不改变 IR
ptoas input.pto --pto-arch=a5 --pto-backend=vpto --emit-vpto \
  --vpto-scheduler=analyze -o output.cpp

# off（默认）完全禁用；on 执行相同分析，不重排 IR
ptoas input.pto --pto-arch=a5 --pto-backend=vpto --emit-vpto \
  --vpto-scheduler=on -o output.cpp

# 查看当前 ptoas release 版本号
ptoas --version

```

`--vpto-scheduler` 在 VPTO 发射流水线的最终 CSE 之后、发射合法性校验之前运行。
目标模型仅支持 A5 Vector kernel；规范化后的 Cube 子模块会被跳过，不生成
调度分析报告。在非 A5 架构上显式启用该选项会报错并终止编译。
`analyze`/`on` 的确定性报告写入标准错误，包括区域边界、依赖 DAG、关键路径、
目标资源占用和寄存器压力；生成代码仍写入正常输出。设计与分析格式详见
[`docs/designs/vpto-scheduler-framework.md`](docs/designs/vpto-scheduler-framework.md)。

### 5.2 Python 接口 (Python API)

在支持的 `ptoas` 安装环境中，PTO Dialect 与 PTODSL 都可以直接导入。

```python
from ptoas.mlir.ir import Context, Module, Location
# PTOAS 自带的 MLIR Python API 位于 ptoas.mlir 命名空间。
from ptoas.mlir.dialects import pto
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
"$PYTHON_BIN" -m pip install -e . --no-build-isolation

# 运行python binding 测试
cd $PTO_SOURCE_DIR/test/samples/MatMul/
"$PYTHON_BIN" ./tmatmulk.py > ./tmatmulk.pto

# 运行ptoas 测试
ptoas ./tmatmulk.pto -o ./tmatmulk.cpp
```

### 5.4 上板验证

该流程用于将 `test/samples` 下生成的 `.cpp`（ptoas 输出）自动生成 NPU 验证用例，并在 NPU 上运行。下面示例直接复用 5.3 里生成的 `MatMul/tmatmulk.cpp`。

> 只想在无卡机器上做 host-side compile-only，请先看 [docs/no_npu_compile_only_guide_zh.md](docs/no_npu_compile_only_guide_zh.md)。


```bash
# 以下相对路径均以仓库根目录为起点
cd "$PTO_SOURCE_DIR"

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
