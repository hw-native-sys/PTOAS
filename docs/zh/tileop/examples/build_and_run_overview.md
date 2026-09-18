# 从 PTO 到输出代码

## 概述

本文使用一个 A3 vec add kernel 贯穿编译与运行流程：两个输入向量 `a`、`b` 和输出向量 `c` 均包含 1024 个连续的 `float32` 元素，计算 `c[i] = a[i] + b[i]`（`0 <= i < 1024`）。kernel 将每个向量表示为 `1 x 1024` 的 tile，使用一个 block 完成计算。

对用户而言，一条完整链路通常包括 5 步：

1. 编写 `.pto`
2. 使用 `ptoas` 生成 kernel C++
3. 准备 `launch.cpp`，把 kernel entry 暴露成 host 可调用 wrapper
4. 使用 CCE 编译器把 kernel C++ 与 `launch.cpp` 编译成 host 可链接的 fatobj
5. 在 host 程序中分配内存、调用 launch wrapper，并通过 AscendCL（Ascend Computing Language，昇腾编程语言，简称ACL）运行

如果你的生成代码需要调用 PTO 指令 API，则它会通过 `pto-isa` 提供的统一入口头文件：

```cpp
#include <pto/pto-inst.hpp>
```

`pto-isa` 的使用参考：

- [https://gitcode.com/cann/pto-isa](https://gitcode.com/cann/pto-isa)

本文档中 `pto-isa` 头文件统一写作 `#include <pto/pto-inst.hpp>`（`pto-isa` 推荐的统一入口头形式）；CANN 提供的头文件按 CANN 惯例写作 `#include "acl/acl.h"`。注意 `ptoas` 自身生成的 C++ 会输出 `#include "pto/pto-inst.hpp"` 的引号形式，两者指向同一个头文件，可互换使用。

## 第一步：使用 ptoas 生成 kernel C++

完整输入文件为 [`vec_add.pto`](./vec_add.pto)。除文件头注释外，内容如下，可直接保存为 `vec_add.pto`：

```mlir
!vec_tile = !pto.tile_buf<vec, 1x1024xf32>

module attributes {pto.target_arch = "a3"} {
  func.func @vec_add(%a: !pto.ptr<f32>, %b: !pto.ptr<f32>, %c: !pto.ptr<f32>)
      attributes {pto.entry, pto.kernel_kind = #pto.kernel_kind<vector>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1024 = arith.constant 1024 : index

    %a_view = pto.make_tensor_view %a, shape = [%c1, %c1024], strides = [%c1024, %c1]
        : !pto.tensor_view<?x?xf32>
    %b_view = pto.make_tensor_view %b, shape = [%c1, %c1024], strides = [%c1024, %c1]
        : !pto.tensor_view<?x?xf32>
    %c_view = pto.make_tensor_view %c, shape = [%c1, %c1024], strides = [%c1024, %c1]
        : !pto.tensor_view<?x?xf32>

    %a_part = pto.partition_view %a_view, offsets = [%c0, %c0], sizes = [%c1, %c1024]
        : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<1x1024xf32>
    %b_part = pto.partition_view %b_view, offsets = [%c0, %c0], sizes = [%c1, %c1024]
        : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<1x1024xf32>
    %c_part = pto.partition_view %c_view, offsets = [%c0, %c0], sizes = [%c1, %c1024]
        : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<1x1024xf32>

    %a_tile = pto.alloc_tile : !vec_tile
    %b_tile = pto.alloc_tile : !vec_tile
    %c_tile = pto.alloc_tile : !vec_tile

    pto.tload ins(%a_part : !pto.partition_tensor_view<1x1024xf32>) outs(%a_tile : !vec_tile)
    pto.tload ins(%b_part : !pto.partition_tensor_view<1x1024xf32>) outs(%b_tile : !vec_tile)
    pto.tadd ins(%a_tile, %b_tile : !vec_tile, !vec_tile) outs(%c_tile : !vec_tile)
    pto.tstore ins(%c_tile : !vec_tile) outs(%c_part : !pto.partition_tensor_view<1x1024xf32>)
    return
  }
}
```

`shape`、`sizes` 和 `strides` 均以元素为单位；`strides = [1024, 1]` 表示连续行主序存储。`!vec_tile` 是本文件内的类型别名，表示存放在向量计算存储区中的 `1 x 1024` 个 `f32` 元素，整块 tile 都参与运算。

`pto.entry` 将 `vec_add` 标记为可从 host 启动的 kernel 入口；`pto.kernel_kind = #pto.kernel_kind<vector>` 指定它执行向量计算。

在 `vec_add.pto` 所在目录执行：

```bash
ptoas vec_add.pto --pto-arch=a3 --pto-backend=emitc \
  --enable-insert-sync -o vec_add_kernel.cpp
```

常见参数包括：

- `--pto-arch=a3|a5`：指定目标代际
- `--pto-backend=emitc`：生成调用 PTO 指令 API 的 C++，供本文的 CCE 编译流程使用
- `--enable-insert-sync`：为输入搬运、向量计算和输出搬运插入必要的同步；本文示例需要保留此选项

## 第二步：生成代码如何与 pto-isa 结合

`ptoas` 生成的 C++ 通常会直接包含 PTO 指令头文件，并把 PTO ISA 中的操作映射为 PTO 指令 API。

下面展示 vec add 对应的主要 PTO API 调用。为突出数据流，示意代码省略了地址绑定和自动同步；实际构建时使用第一步生成的完整 `vec_add_kernel.cpp`。

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

extern "C" __global__ AICORE void vec_add(__gm__ float* a, __gm__ float* b, __gm__ float* c) {
  using VecTile = Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, 1, 1024>;
  using VecShape = Shape<1, 1, 1, 1, 1024>;
  using VecStride = Stride<1024, 1024, 1024, 1024, 1>;
  using VecTensor = GlobalTensor<float, VecShape, VecStride>;

  VecTile tileA;
  VecTile tileB;
  VecTile tileC;
  VecTensor gA(a);
  VecTensor gB(b);
  VecTensor gC(c);

  // 完整生成代码会为 tile 绑定存储地址。
  TLOAD(tileA, gA);
  TLOAD(tileB, gB);
  // 完整生成代码会在计算前等待输入搬运完成。
  TADD(tileC, tileA, tileB);
  // 完整生成代码会在写回前等待计算完成。
  TSTORE(gC, tileC);
}
```

这里的结合关系可以概括为：

- PTO ISA 描述 tile 级语义
- `ptoas` 把这些语义翻译成 PTO 指令 API 调用
- `pto-isa` 提供这些 API 对应的类型系统、指令声明和目标相关实现

从用户视角看，`ptoas` 生成代码后，下一步并不是再手写 tile 指令主体，而是让 CCE 编译器继续编译这些已经调用了 PTO 指令 API 的 C++ 文件。

## 第三步：准备 launch.cpp

`launch.cpp` 的作用是把设备侧 kernel entry 包装成 host 侧可调用函数。最小形式如下：

```cpp
#include <pto/pto-inst.hpp>
#include "acl/acl.h"

extern "C" __global__ AICORE void vec_add(__gm__ float* a, __gm__ float* b, __gm__ float* c);

void LaunchVecAdd(float *a, float *b, float *c, aclrtStream stream) {
  vec_add<<<1, nullptr, stream>>>(a, b, c);
}
```

这里固定启动一个 block，与 `.pto` 中处理一整块 `1 x 1024` tile 的约定一致。`a`、`b`、`c` 必须是 device 地址，指向三个独立的缓冲区，每个缓冲区至少为 `1024 * sizeof(float)`，即 4096 字节。

kernel 声明中的 `extern "C"` 与 `pto.entry` 生成的入口保持一致，使 `launch.cpp` 能正确链接到 `vec_add`。

## 第四步：通过 CCE 编译器生成 fatobj

### fatobj 的角色

这里的 fatobj 可以理解为：

- 对 host 可链接
- 内部携带 device binary
- 能被 host 可执行程序或共享库直接链接

因此，用户最终链接的通常不是裸设备目标，而是带有设备镜像的 host 可链接产物。

### 典型编译方式

准备好 CANN 环境，将 `PTO_ISA_ROOT` 指向 `pto-isa` 源码目录，并确认 `ASCEND_HOME_PATH` 指向 CANN 安装目录。把第三步代码保存为 `launch.cpp` 后，与生成的 kernel C++ 一起编译：

```bash
bisheng -std=c++17 -shared -fPIC \
  -xcce \
  --cce-aicore-arch=dav-c220-vec \
  --cce-fatobj-link \
  vec_add_kernel.cpp launch.cpp \
  -I"${PTO_ISA_ROOT}/include" \
  -I"${ASCEND_HOME_PATH}/include" \
  -I"${ASCEND_HOME_PATH}/pkg_inc" \
  -o libvec_add_kernel.so
```

本文使用向量计算，因此选择 A3 对应的 `dav-c220-vec` 编译目标。

在 CMake 中，最关键的一项通常是：

```cmake
target_link_options(vec_add_kernel PRIVATE --cce-fatobj-link)
```

## 第五步：host 程序如何运行 fatobj

host 程序的基本流程通常如下：

1. `aclInit`
2. `aclrtSetDevice`
3. `aclrtCreateStream`
4. 申请 host / device 内存
5. 把输入从 host 拷到 device
6. 调用 `LaunchVecAdd(devA, devB, devC, stream)`
7. `aclrtSynchronizeStream`
8. 把输出从 device 拷回 host
9. 逐元素比较 `hostC[i]` 与 `hostA[i] + hostB[i]`
10. 释放 host / device 内存，销毁 stream，复位 device，并调用 `aclFinalize`

下面是第 6、7 步的调用片段，可放在 `main.cpp` 中。调用前需完成 ACL 初始化、stream 创建、三个 device 缓冲区的分配和输入拷贝；调用后检查返回值，再将 4096 字节输出拷回 host 进行比较。完整 host 程序还需检查各个 ACL 调用的返回值，并在退出前释放已申请的资源。

```cpp
#include "acl/acl.h"

void LaunchVecAdd(float *a, float *b, float *c, aclrtStream stream);

aclError RunVecAdd(float *devA, float *devB, float *devC, aclrtStream stream) {
  LaunchVecAdd(devA, devB, devC, stream);
  return aclrtSynchronizeStream(stream);
}
```

例如，令 `hostA[i] = float(i)`、`hostB[i] = 1.0f`，则预期 `hostC[i] = float(i) + 1.0f`。对一般浮点输入，应按所需精度设置比较容差。

## 推荐的工程组织

对单个样例，推荐保持如下结构：

```text
vec_add_example/
├── vec_add.pto
├── vec_add_kernel.cpp
├── launch.cpp
├── main.cpp
└── CMakeLists.txt
```

其中：

- `vec_add.pto`：完整 PTO ISA 输入，见[随文文件](./vec_add.pto)
- `vec_add_kernel.cpp`：`ptoas` 输出
- `launch.cpp`：kernel wrapper
- `main.cpp`：host 侧运行入口
- `CMakeLists.txt`：调用 `bisheng` 生成 fatobj 并链接 host 程序

## 常见注意事项

### `--pto-arch` 要与目标代际匹配

如果 `.pto` 面向 A5，却按 A3 编译，常见结果是：

- 类型或操作约束不匹配
- 某些 op 不被目标支持

### kernel 参数顺序要保持一致

`.pto` 中函数参数顺序、生成后的 kernel 形参顺序、`launch.cpp` wrapper 参数顺序、host 侧调用顺序，必须保持一致。

### 输入长度与启动规模要保持一致

本例只处理 1024 个元素，不包含按 block 划分数据或尾块处理逻辑。缩短缓冲区会越界，增加 block 数会让多个 block 重复写入同一输出区域。改变长度或并行规模时，需要同时修改 `.pto` 中的视图和 tile、launch 配置及 host 内存分配。

### 编译通过不等于运行正确

`ptoas` 编译通过、`bisheng` 编译通过，只能说明：

- PTO ISA 结构基本成立
- 生成的 PTO 指令调用形式基本成立

是否符合预期，仍然需要用户根据自己的场景在板端进一步检查运行结果。
