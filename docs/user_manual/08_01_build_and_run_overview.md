# 8.1 从 PTO 到输出代码

## 1. 概述

对用户而言，一条完整链路通常包括 5 步：

1. 编写 `.pto`
2. 使用 `ptoas` 生成 kernel C++
3. 准备 `launch.cpp`，把 kernel entry 暴露成 host 可调用 wrapper
4. 使用 CCE 编译器把 kernel C++ 与 `launch.cpp` 编译成 host 可链接的 fatobj
5. 在 host 程序中分配内存、调用 launch wrapper，并通过 ACL 运行

如果你的生成代码需要调用 PTO 指令 API，则它会通过 `pto-isa` 提供的统一入口头：

```cpp
#include <pto/pto-inst.hpp>
```

`pto-isa` 的使用参考：

- [https://gitcode.com/cann/pto-isa](https://gitcode.com/cann/pto-isa)

## 2. 第一步：使用 ptoas 生成 kernel C++

典型命令如下：

```bash
ptoas input.pto --pto-arch=a3 --enable-insert-sync -o input_kernel.cpp
```

常见参数包括：

- `--pto-arch=a3|a5`：指定目标代际
- `--pto-level=level2|level3`：控制 PTO ISA 所处层级
- `--enable-insert-sync`：启用常规自动同步

## 3. 第二步：生成代码如何与 pto-isa 结合

`ptoas` 生成的 C++ 通常会直接包含 PTO 指令头文件，并把 PTO ISA 中的操作映射为 PTO 指令 API。

典型形态如下：

```cpp
#include "pto/pto-inst.hpp"
using namespace pto;

__global__ AICORE void kernel_name(__gm__ float* a, __gm__ float* b, __gm__ float* c) {
  Tile<TileType::Left, float, 32, 32, ...> lhs;
  Tile<TileType::Right, float, 32, 32, ...> rhs;
  Tile<TileType::Acc, float, 32, 32, ...> acc;
  GlobalTensor<float, ...> gA(...);
  GlobalTensor<float, ...> gB(...);
  GlobalTensor<float, ...> gC(...);

  TLOAD(lhs, gA);
  TLOAD(rhs, gB);
  TMATMUL(acc, lhs, rhs);
  TSTORE(gC, acc);
}
```

这里的结合关系可以概括为：

- PTO ISA 描述 tile 级语义
- `ptoas` 把这些语义翻译成 PTO 指令 API 调用
- `pto-isa` 提供这些 API 对应的类型系统、指令声明和目标相关实现

从用户视角看，`ptoas` 生成代码后，下一步并不是再手写 tile 指令主体，而是让 CCE 编译器继续编译这些已经调用了 PTO 指令 API 的 C++ 文件。

## 4. 第三步：准备 launch.cpp

`launch.cpp` 的作用是把设备侧 kernel entry 包装成 host 侧可调用函数。最小形式如下：

```cpp
#include <pto/pto-inst.hpp>
#include "acl/acl.h"

__global__ AICORE void matmul_block(__gm__ float* a, __gm__ float* b, __gm__ float* c);

void LaunchMatmul_block(float *a, float *b, float *c, void *stream) {
  matmul_block<<<1, nullptr, stream>>>(a, b, c);
}
```

如果是多阶段 Flash Attention，则可以准备多个 wrapper，例如：

- `LaunchFlash_attention_qk_block`
- `LaunchFlash_attention_softmax_block`
- `LaunchFlash_attention_sv_block`

然后由 host 程序按顺序调用。

## 5. 第四步：通过 CCE 编译器生成 fatobj

### 5.1 fatobj 的角色

这里的 fatobj 可以理解为：

- 对 host 可链接
- 内部携带 device binary
- 能被 host 可执行程序或共享库直接链接

因此，用户最终链接的通常不是裸设备目标，而是带有设备镜像的 host 可链接产物。

### 5.2 典型编译方式

使用 `bisheng` 编译 kernel C++ 与 `launch.cpp` 时，常见形式如下：

```bash
bisheng -shared -fPIC \
  -xcce \
  --cce-aicore-arch=dav-c220-cube \
  --cce-fatobj-link \
  input_kernel.cpp launch.cpp \
  -I${PTO_ISA_ROOT}/include \
  -I${ASCEND_HOME_PATH}/include \
  -I${ASCEND_HOME_PATH}/pkg_inc \
  -o libinput_kernel.so
```

在 CMake 中，最关键的一项通常是：

```cmake
target_link_options(your_kernel PRIVATE --cce-fatobj-link)
```

## 6. 第五步：host 程序如何运行 fatobj

host 程序的基本流程通常如下：

1. `aclInit`
2. `aclrtSetDevice`
3. `aclrtCreateStream`
4. 申请 host / device 内存
5. 把输入从 host 拷到 device
6. 调用 `LaunchXXX(...)`
7. `aclrtSynchronizeStream`
8. 把输出从 device 拷回 host
9. 释放资源

对应的最小骨架如下：

```cpp
#include "acl/acl.h"

void LaunchMatmul_block(float *a, float *b, float *c, void *stream);

int main() {
  aclInit(nullptr);
  aclrtSetDevice(0);

  aclrtStream stream = nullptr;
  aclrtCreateStream(&stream);

  // 申请 host / device 内存，并准备输入
  // aclrtMemcpy(..., ACL_MEMCPY_HOST_TO_DEVICE)

  LaunchMatmul_block(devA, devB, devC, stream);
  aclrtSynchronizeStream(stream);

  // aclrtMemcpy(..., ACL_MEMCPY_DEVICE_TO_HOST)

  aclrtDestroyStream(stream);
  aclrtResetDevice(0);
  aclFinalize();
  return 0;
}
```

如果是 Flash Attention 的分阶段版本，则 host 侧通常按以下顺序调用：

```cpp
LaunchFlash_attention_qk_block(..., stream);
LaunchFlash_attention_softmax_block(..., stream);
LaunchFlash_attention_sv_block(..., stream);
aclrtSynchronizeStream(stream);
```

## 7. 推荐的工程组织

对单个样例，推荐保持如下结构：

```text
example/
├── kernel.pto
├── kernel.cpp
├── launch.cpp
├── main.cpp
└── CMakeLists.txt
```

其中：

- `kernel.pto`：PTO ISA 输入
- `kernel.cpp`：`ptoas` 输出
- `launch.cpp`：kernel wrapper
- `main.cpp`：host 侧运行入口
- `CMakeLists.txt`：调用 `bisheng` 生成 fatobj 并链接 host 程序

当前手册中的对应工程目录为：

- [`08_02_matmul_project`](./08_02_matmul_project/)
- [`08_03_softmax_project`](./08_03_softmax_project/)
- [`08_04_flash_attention_project`](./08_04_flash_attention_project/)

## 8. 常见注意事项

### 8.1 `--pto-arch` 要与目标代际匹配

如果 `.pto` 面向 A5，却按 A3 编译，常见结果是：

- 类型或操作约束不匹配
- 某些 op 不被目标支持

### 8.2 kernel 参数顺序要保持一致

`.pto` 中函数参数顺序、生成后的 kernel 形参顺序、`launch.cpp` wrapper 参数顺序、host 侧调用顺序，必须保持一致。

### 8.3 多阶段样例要明确中间缓冲区

对于 Flash Attention 这类分阶段场景，`scores`、`softmax` 等中间结果都需要 host 侧正确分配和串联传递。

### 8.4 编译通过不等于运行正确

`ptoas` 编译通过、`bisheng` 编译通过，只能说明：

- PTO ISA 结构基本成立
- 生成的 PTO 指令调用形式基本成立

是否符合预期，仍然需要用户根据自己的场景在板端进一步检查运行结果。
