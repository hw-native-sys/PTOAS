# 端侧 Kirin9030 架构 PTOAS 使用指南

本文档描述在端侧 Kirin9030 架构上使用 PTOAS 的完整流程，涵盖从源码构建、IR 生成、代码生成到最终编译 kernel 二进制的全部步骤。

## 1. 概述

端侧流程与标准 A3/A5 流程的主要区别：

- 使用 `--pto-arch=kirin9030` 指定目标架构
- 需要安装 CANN@Kirin 工具包获取 bisheng 编译器
- 编译 kernel 时需指定 `--cce-aicore-arch=dav-l311`

完整流程分为以下阶段：

```text
Python 样例 → .pto 文件 → ptoas 生成 .cpp → bisheng 编译 → kernel 二进制 (.o)
```

## 2. 获取并构建 PTOAS

### 2.1 下载源码

```bash
git clone https://github.com/gtest-rgb/PTOAS.git
cd PTOAS
git checkout feature/kirin9030-arch-support
```

### 2.2 构建源码

按照 [README.md](../README.md) 中第 3 章「构建指南」完成 LLVM/MLIR 依赖编译和 PTOAS 的 Out-of-Tree 构建。

简要步骤：

```bash
# 1. 构建 LLVM/MLIR (llvmorg-19.1.7)
cmake -G Ninja -S llvm -B $LLVM_BUILD_DIR \
    -DLLVM_ENABLE_PROJECTS="mlir;clang" \
    -DBUILD_SHARED_LIBS=ON \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DPython3_EXECUTABLE=$(which python3) \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_TARGETS_TO_BUILD="host"
ninja -C $LLVM_BUILD_DIR

# 2. 构建 PTOAS
cmake -G Ninja -S . -B build \
    -DLLVM_DIR=$LLVM_BUILD_DIR/lib/cmake/llvm \
    -DMLIR_DIR=$LLVM_BUILD_DIR/lib/cmake/mlir \
    -DPython3_EXECUTABLE=$(which python3) \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DCMAKE_INSTALL_PREFIX="$PTO_INSTALL_DIR"
ninja -C build
```

构建完成后确认 `ptoas` 工具可用：

```bash
ls build/tools/ptoas/ptoas
```

## 3. 生成 .pto 文件

以仓库自带的 `test/samples/Adds/adds.py` 为例，执行 Python 脚本生成 `.pto` 文件：

```bash
cd test/samples/Adds
python3 adds.py > adds.pto
```

该脚本通过 PTO Python 绑定构建 PTO IR 模块，输出 `.pto` 格式的文本 IR。

## 4. 使用 ptoas 生成 kernel C++ 源码

使用 `ptoas` 工具指定 `kirin9030` 架构，生成调用 `pto-isa` 的 C++ kernel 源码：

```bash
ptoas ./adds.pto --pto-arch=kirin9030 --enable-insert-sync -o ./adds.cpp
```

参数说明：

| 参数 | 说明 |
|------|------|
| `--pto-arch=kirin9030` | 指定目标架构为 Kirin9030 |
| `--enable-insert-sync` | 启用自动同步插入，确保硬件流水线正确同步 |
| `-o ./adds.cpp` | 输出 C++ kernel 源码文件 |

## 5. 编译 kernel 二进制文件

将生成的 `.cpp` 文件编译为 kernel 二进制需要 bisheng 编译器和 pto-isa 头文件。

### 5.1 下载 pto-isa 源码

```bash
git clone https://gitcode.com/cann/pto-isa.git
```

### 5.2 安装 CANN@Kirin 工具包

CANN@Kirin 工具包包含 bisheng 编译器，是编译 kernel 的必要依赖。

下载地址：

```text
https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/legacy/20260528120328736/Ascend-cann-toolkit_9.1.0_linux-x86_64.run
```

安装：

```bash
chmod +x Ascend-cann-toolkit_9.1.0_linux-x86_64.run
./Ascend-cann-toolkit_9.1.0_linux-x86_64.run --install --install-path=<path-to-cann>
```

### 5.3 配置环境变量

```bash
source <path-to-cann>/cann/set_env.sh
```

验证 bisheng 编译器可用：

```bash
which bisheng
bisheng --version
```

### 5.4 使用 bisheng 编译 kernel 二进制

```bash
bisheng -c -O3 -g -x cce -std=c++17 \
    --cce-aicore-only \
    --cce-aicore-arch=dav-l311 \
    -I<pto-isa源码路径>/include \
    -mllvm -cce-aicore-jump-expand=true \
    -mllvm -cce-aicore-function-stack-size=16384 \
    -mllvm -cce-aicore-record-overflow=false \
    -mllvm -cce-aicore-addr-transform \
    -mllvm -cce-aicore-dcci-insert-for-scalar=false \
    --cce-aicore-input-parameter-size=4096 \
    -o adds.o adds.cpp
```

参数说明：

| 参数 | 说明 |
|------|------|
| `-c` | 只编译，不链接 |
| `-O3` | 最高优化等级 |
| `-x cce` | 指定语言为 CCE (Cube Compute Engine) |
| `--cce-aicore-only` | 仅编译 AI Core 部分 |
| `--cce-aicore-arch=dav-l311` | 指定 AI Core 微架构为 dav-l311 (Kirin9030) |
| `-I<pto-isa源码路径>/include` | 包含 pto-isa 头文件路径 |
| `-o adds.o` | 输出目标文件 |

编译完成后，`adds.o` 即为可在 Kirin9030 上运行的 kernel 二进制文件。

## 6. 完整流程示例

以下为一个从头到尾的完整操作示例：

```bash
# === 1. 获取 PTOAS ===
git clone https://github.com/gtest-rgb/PTOAS.git
cd PTOAS
git checkout feature/kirin9030-arch-support

# === 2. 构建 PTOAS (假设 LLVM 已构建完成) ===
cmake -G Ninja -S . -B build \
    -DLLVM_DIR=$LLVM_BUILD_DIR/lib/cmake/llvm \
    -DMLIR_DIR=$LLVM_BUILD_DIR/lib/cmake/mlir \
    -DPython3_EXECUTABLE=$(which python3) \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON
ninja -C build

# === 3. 配置运行环境 ===
source <path-to-cann>/cann/set_env.sh

# === 4. 生成 .pto 文件 ===
cd test/samples/Adds
python3 adds.py > adds.pto

# === 5. 生成 kernel C++ 源码 ===
../../build/tools/ptoas/ptoas ./adds.pto \
    --pto-arch=kirin9030 \
    --enable-insert-sync \
    -o ./adds.cpp

# === 6. 编译 kernel 二进制 ===
bisheng -c -O3 -g -x cce -std=c++17 \
    --cce-aicore-only \
    --cce-aicore-arch=dav-l311 \
    -I<pto-isa源码路径>/include \
    -mllvm -cce-aicore-jump-expand=true \
    -mllvm -cce-aicore-function-stack-size=16384 \
    -mllvm -cce-aicore-record-overflow=false \
    -mllvm -cce-aicore-addr-transform \
    -mllvm -cce-aicore-dcci-insert-for-scalar=false \
    --cce-aicore-input-parameter-size=4096 \
    -o adds.o adds.cpp

echo "Kernel 二进制生成完成: adds.o"
```

## 7. 排障建议

| 问题 | 排查方向 |
|------|----------|
| `ptoas` 报 `invalid --pto-arch` | 确认已切换到 `feature/kirin9030-arch-support` 分支 |
| `bisheng` 命令未找到 | 检查是否执行了 `source <path-to-cann>/cann/set_env.sh` |
| 编译报头文件找不到 | 确认 `-I<pto-isa源码路径>/include` 路径正确 |
| 编译报 `dav-l311` 相关错误 | 确认 CANN@Kirin 版本为 9.1.0，支持 Kirin9030 架构 |
