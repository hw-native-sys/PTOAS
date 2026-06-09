# 8.3 Softmax 示例

## 1. 概述

本节对应一个可直接编译运行的 Softmax 工程：

- [08_03_softmax_project](../../test/user_manual_examples/08_03_softmax_project/)

工程内包含：

- `kernel.pto`
- `run.sh`
- `CMakeLists.txt`
- `launch.cpp`
- `main.cpp`

## 2. 入口文件

- PTO ISA 输入：[`08_03_softmax_project/kernel.pto`](../../test/user_manual_examples/08_03_softmax_project/kernel.pto)
- Host 入口：[`08_03_softmax_project/main.cpp`](../../test/user_manual_examples/08_03_softmax_project/main.cpp)
- 启动封装：[`08_03_softmax_project/launch.cpp`](../../test/user_manual_examples/08_03_softmax_project/launch.cpp)

## 3. 运行方式

在工程目录下执行：

```bash
bash run.sh
```

`run.sh` 会先调用 `ptoas` 生成 `kernel.cpp`，然后完成构建和运行。

## 4. 示例说明

这个工程使用单个向量路径 kernel，输入为：

- `scores`: `32 x 32`
- `group_scale`: `32 x 32`

输出为：

- `softmax`: `32 x 32`

示例中的 Softmax 使用多项式近似指数形式，便于直接观察 PTO ISA 中逐元素计算的组织方式。
