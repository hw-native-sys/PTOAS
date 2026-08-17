# 基于已安装 LLVM 的 PTOAS 构建说明

本文档按 [README.md](../README.md) 第 3 章的逻辑整理，适用于：

- LLVM/MLIR LLVM19 VPTO 分支 `vpto-dev/llvm-project:feature-vpto` 已经构建并安装完成。
- LLVM 安装路径固定为 `/opt/llvm`。
- `/opt/llvm` 是共享目录，不希望 `ptoas` 的安装步骤写入其中。

## 3.0 环境变量配置

先按 README 第 3.0 节的思路把变量定好。区别是这里不再使用 LLVM 源码目录和 LLVM build tree，而是直接使用 LLVM install tree。

```bash
# ================= 配置区域 (请按实际环境调整) =================
export WORKSPACE_DIR=$HOME/llvm-workspace

# LLVM 已安装完成，直接指向 install 根目录
export LLVM_INSTALL_DIR=/opt/llvm

# 为兼容仓库内部分脚本 / lit 变量命名，这里额外保留 LLVM_BUILD_DIR
export LLVM_BUILD_DIR=$LLVM_INSTALL_DIR

# ptoas 源码与安装路径
export PTO_SOURCE_DIR=$WORKSPACE_DIR/PTOAS
export PTO_INSTALL_DIR=$PTO_SOURCE_DIR/install-optllvm
# ============================================================

mkdir -p "$WORKSPACE_DIR"
```

说明：

- 这里的 `LLVM_BUILD_DIR` 只是为了兼容仓库内已有变量名，实际指向的是 LLVM install 根目录 `/opt/llvm`。
- `PTO_INSTALL_DIR` 建议单独放到 PTOAS 自己目录下，避免与共享 LLVM 安装混用。

## 3.1 环境准备

沿用 README 第 3.1 节即可，重点确认这些依赖已经满足：

- Linux
- GCC >= 9 或 Clang
- CMake >= 3.20
- Ninja
- Python 3.8+
- `pybind11<3`
- `numpy`

```bash
pip3 install "pybind11<3" numpy
```

## 跳过 3.2

README 第 3.2 节是 LLVM/MLIR 的下载和编译步骤。当前场景下 LLVM 已经安装在 `/opt/llvm`，这一节可以直接跳过。

已验证：

```bash
/opt/llvm/bin/llvm-config --version
```

输出为：

```text
19.1.7
```

实际源码基线应来自 `https://github.com/vpto-dev/llvm-project.git` 的 `feature-vpto` 分支。

## 3.3 第二步：构建 ptoas

这里沿用 README 第 3.3 节的流程，但 `LLVM_DIR` 和 `MLIR_DIR` 需要改为
`/opt/llvm/lib/cmake/...`。

PTOAS 使用 MLIR 导出的 Python source-set 在自己的 build/install tree 中
组装完整的 `mlir` package，不再复制或 overlay LLVM build tree 中已经生成的
Python package。

```bash
cd "$PTO_SOURCE_DIR"

# 1. 安装 MLIR Python binding 的构建依赖
python3 -m pip install 'pybind11<3'

# 2. 配置 CMake
cmake -G Ninja \
    -S . \
    -B build \
    -DLLVM_DIR=$LLVM_INSTALL_DIR/lib/cmake/llvm \
    -DMLIR_DIR=$LLVM_INSTALL_DIR/lib/cmake/mlir \
    -DPython3_EXECUTABLE=$(which python3) \
    -DPython3_FIND_STRATEGY=LOCATION \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DCMAKE_INSTALL_PREFIX="$PTO_INSTALL_DIR"

# 3. 编译并安装
ninja -C build
cmake --install build
```

## 构建后关键产物

按上面的配置，关键产物位置如下：

- build 目录：
  - `$PTO_SOURCE_DIR/build/tools/ptoas/ptoas`
  - `$PTO_SOURCE_DIR/build/tools/ptobc/ptobc`
  - `$PTO_SOURCE_DIR/build/python/ptoas/_core.cpython-*.so`
  - `$PTO_SOURCE_DIR/build/python/mlir/dialects/pto.py`
- install 目录：
  - `$PTO_INSTALL_DIR/bin/ptoas`
  - `$PTO_INSTALL_DIR/ptoas/_core.cpython-*.so`
  - `$PTO_INSTALL_DIR/ptoas/mlir/dialects/pto.py`
  - `$PTO_INSTALL_DIR/share/ptoas/oplib/level3`

## 补充：运行环境

### 使用 build 目录中的 `ptoas`

```bash
export PATH=$PTO_SOURCE_DIR/build/tools/ptoas:$PATH
export PYTHONPATH=$PTO_SOURCE_DIR/build/python:$PYTHONPATH
export LD_LIBRARY_PATH=$LLVM_INSTALL_DIR/lib:$PTO_SOURCE_DIR/build/lib:$LD_LIBRARY_PATH
```

### 使用 install 目录中的 `ptoas`

```bash
export PATH=$PTO_INSTALL_DIR/bin:$PATH
export PYTHONPATH=$PTO_INSTALL_DIR:$PYTHONPATH
export LD_LIBRARY_PATH=$LLVM_INSTALL_DIR/lib:$PTO_INSTALL_DIR/lib:$LD_LIBRARY_PATH
```

注意：

- install 版 `ptoas` 仍然需要从 `/opt/llvm/lib` 加载 LLVM/MLIR 共享库。
- 如果直接运行 `$PTO_INSTALL_DIR/bin/ptoas` 而没有设置 `LD_LIBRARY_PATH=$LLVM_INSTALL_DIR/lib:...`，会报缺少 `libMLIR*.so`。

## 本地验证结果

当前仓库已验证通过以下组合：

- `LLVM_DIR=/opt/llvm/lib/cmake/llvm`
- `MLIR_DIR=/opt/llvm/lib/cmake/mlir`
- `CMAKE_INSTALL_PREFIX=$PTO_INSTALL_DIR`

最小验证结果：

- build 版 `ptoas --version` 输出 `ptoas 0.22`
- build 版 `ptoas` 可成功处理 `test/lit/pto/empty_func.pto`
- install 版 Python 绑定可在 `PYTHONPATH=$PTO_INSTALL_DIR` 下正常导入
- 若 install 版 `ptoas` 配合 `LD_LIBRARY_PATH=/opt/llvm/lib:$PTO_INSTALL_DIR/lib`，可正常执行
