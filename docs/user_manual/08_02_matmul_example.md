# 8.2 MatMul 示例

## 1. 概述

本节对应一个可直接编译运行的 MatMul 工程：

- [08_02_matmul_project](../../test/user_manual_examples/08_02_matmul_project/)

工程内包含：

- `kernel.pto`
- `run.sh`
- `CMakeLists.txt`
- `launch.cpp`
- `main.cpp`

## 2. 入口文件

- PTO ISA 输入：[`08_02_matmul_project/kernel.pto`](../../test/user_manual_examples/08_02_matmul_project/kernel.pto)
- Host 入口：[`08_02_matmul_project/main.cpp`](../../test/user_manual_examples/08_02_matmul_project/main.cpp)
- 启动封装：[`08_02_matmul_project/launch.cpp`](../../test/user_manual_examples/08_02_matmul_project/launch.cpp)

## 3. 运行方式

在工程目录下执行：

```bash
bash run.sh
```

`run.sh` 会完成以下步骤：

1. 调用 `ptoas` 把 `kernel.pto` 编译成 `kernel.cpp`
2. 使用 CCE 编译器完成构建
3. 运行 host 程序

## 4. 示例说明

这个工程使用单个 MatMul kernel，输入为：

- `A`: `32 x 32`
- `B`: `32 x 32`

输出为：

- `C`: `32 x 32`

当前示例直接验证基础矩阵乘 `A x B`。
