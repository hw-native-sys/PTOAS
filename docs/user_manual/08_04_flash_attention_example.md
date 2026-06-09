# 8.4 Flash Attention 示例

## 1. 概述

本节对应一个可直接编译运行的 Flash Attention 分阶段工程：

- [08_04_flash_attention_project](../../test/user_manual_examples/08_04_flash_attention_project/)

工程内包含：

- `qk.pto`
- `softmax.pto`
- `sv.pto`
- `run.sh`
- `CMakeLists.txt`
- `launch_qk.cpp`
- `launch_softmax.cpp`
- `launch_sv.cpp`
- `main.cpp`

## 2. 入口文件

- QK 阶段：[`08_04_flash_attention_project/qk.pto`](../../test/user_manual_examples/08_04_flash_attention_project/qk.pto)
- Softmax 阶段：[`08_04_flash_attention_project/softmax.pto`](../../test/user_manual_examples/08_04_flash_attention_project/softmax.pto)
- SV 阶段：[`08_04_flash_attention_project/sv.pto`](../../test/user_manual_examples/08_04_flash_attention_project/sv.pto)

## 3. 运行方式

在工程目录下执行：

```bash
bash run.sh
```

`run.sh` 会先生成三个阶段对应的 `kernel.cpp`，再完成构建和运行。

## 4. 示例说明

这个工程采用三阶段串联方式：

1. `QK`: 计算 `Q x K^T`
2. `Softmax`: 对 `scores` 做缩放、截断和归一化
3. `SV`: 计算 `softmax x V`

Host 侧通过三个 launch wrapper 顺序调用三个 kernel，并在 GM 中显式传递 `scores` 和 `softmax` 中间结果。
