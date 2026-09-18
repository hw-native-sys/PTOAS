---
name: pto-st-debug
description: 端到端调试 PTO Tile 算子——算子分类→自动生成harness→插tprint→运行比对→二分定位→根因修复。当用户说"调试这个算子"、"debug ST"、"定位精度问题"、"对比golden"时触发。
license: MIT
---

# PTO ST Debug Skill

PTOAS 编译器调试工作流。给定一个 `.pto` 文件，AI 自动生成配套测试代码，插入 tprint 打桩，逐步骤比对定位根因。

---

## Quick Path（80% 场景：类别 A 元素级二元 op）

大多数调试任务是简单 vec 空间二元算子。最短路径：

```
1. 读 .pto → 确认是类别 A（2 in + 1 out，shape 对称，vec 空间）
2. 复制 test/tilelang_st/npu/a5/src/st/testcase/tadd/ 为模板
3. 全局替换：tadd→<op>, TADD→<OP>
4. 按 cases.py 调整 shape/dtype/eps
5. 修改 gen_data.py 的 golden 公式
6. 运行: python3 test/tilelang_st/script/run_st.py -r sim -v a5 -t <op>
7. 如果比对失败 → 阶段 2 TPrint 调试
```

---

## 前置

```bash
LLVM_BUILD_DIR=/path/to/llvm/build ./quick_install.sh
source "${ASCEND_HOME_PATH}/bin/setenv.bash"
```

所有命令从 repo root 执行。

### 流程入口

输入文件为 `.pto`，走阶段 0-5（标准 ptoas 流程）。

### 核心文件地图

| 资源 | 路径 |
|---|---|
| PTO IR 样本/测试 | `test/samples/<Op>/` |
| ST testcase | `test/tilelang_st/npu/a5/src/st/testcase/` |
| ST 运行脚本 | `test/tilelang_st/script/run_st.py` |
| ptoas 命令 | 当前 Python 环境安装的 `ptoas`（可用 `PTOAS_BIN` 覆盖） |
| ODS 定义 | `include/PTO/IR/PTOOps.td` |
| VPTO ODS 定义 | `include/PTO/IR/VPTOOps.td` |
| C++ IR/Verifier | `lib/PTO/IR/PTO.cpp` |
| ExpandTileOp | `lib/PTO/Transforms/TileOp/ExpandTileOp.cpp` |
| VPTO→LLVM lowering | `lib/PTO/Transforms/VPTOLLVMEmitter.cpp` |
| PTODSL TileLib 模板 | `lib/TileOps/` |
| PTODSL TileLib daemon | `ptodsl/ptodsl/tilelib/serving/` |
| Tile Op ISA 参考 | `docs/isa/tile-op/` |
| TPrint 格式/约束详参 | 本文档阶段 2.1 ~ 2.4 |

### ptoas 调试 flags

| Flag | 作用 |
|---|---|
| `--emit-pto-ir` | 输出 lower 后的 PTO IR |
| `--emit-vpto` | 输出 VPTO IR 到文件（Expand 后、lowering 前） |
| `--vpto-print-ir` | 打印 VPTO IR 到 stderr |
| `--enable-insert-sync` | 自动插入同步 barrier |
| `--enable-tile-op-expand` | 展开 tile op（调用 DSL 模板） |
| `--pto-backend=vpto` | 使用 VPTO 后端（生成 fatobj） |
| `--pto-backend=emitc` | 使用 EmitC 后端（生成 C++ 代码，默认） |

调试 lowering 问题时优先用 `--vpto-print-ir` 看 Expand 后的 IR。

### st_common API 参考

`test/tilelang_st/npu/a5/src/st/testcase/st_common.py` 提供以下函数：

```python
def setup_case_rng(case):
    """基于 case["name"] 的 hash 设置确定性随机种子。"""

def save_case_data(case_name, data_dict):
    """创建 case_name/ 子目录，将 data_dict 中每个 numpy array 写为 {key}.bin。
    Args:
        case_name: 子目录名 (如 "f32_16x64")
        data_dict: {"input1": arr, "input2": arr, "golden": arr}
    """

def validate_cases(cases):
    """校验每个 case 是否包含必需 keys: name, dtype, shape, valid_shape, eps。"""

def result_cmp(golden, output, eps):
    """比较 golden 和 output (numpy arrays)，返回 bool。
    内部使用 np.allclose(golden, output, atol=eps, rtol=eps, equal_nan=True)。
    打印首个 mismatch 位置和值。
    """
```

---

## 阶段 0：算子分类（先分类，后生成）

在生成任何代码之前，先读 `.pto` 文件，按**计算图特征**分类。不同类别对应不同的生成模板。

### 0.1 分类决策

从 `.pto` 文件提取以下信息：
- 入口函数参数个数和类型
- PTO op 类型：tadd / tcmp / tcolmax / tadds / texp / tcvt / tcolexpandadd 等
- 输入个数（读 .pto 的 op 有几个 `ins`）
- 输出 dtype 是否 = i8 且有 `cmpMode` 属性
- 输入输出 shape 是否对称
- 有无标量参数

```
┌─ 有标量参数（%scalar: T）？                      → 类别 B：标量 op
├─ 输出 dtype==i8 且有 cmpMode 属性？               → 类别 C：比较/tcmp
├─ 只有 1 个输入 op（一元）？                       → 类别 F：一元 op
├─ 输入 shape ≠ 输出 shape？
│   ├─ 输出 size 是 reduction（如 (R,C)→(1,C)）？    → 类别 D：Reduction
│   └─ 输出 size = 输入 size（broadcast）？          → 类别 G：Broadcast
├─ 涉及 cube 空间（mat/left/right/acc）？           → 类别 E：Cube+Vector
└─ 上述都不满足                                      → 类别 A：元素级二元
```

### 0.2 各类别特征与生成策略

| 类别 | 特征 | Op 示例 | 可自动生成？ |
|---|---|---|---|
| A: 元素级二元 | 2 in + 1 out，shape 对称，dtype 一致，vec 空间 | tadd, tsub, tmul, tdiv, tmax, tmin | 完全模板化 |
| B: 标量 op | 1 tensor in + scalar + 1 out，shape 对称 | tadds, tmuls, tshls | 模板 + 标量参数 |
| C: 比较 op | 2 in + 1 out，输出 dtype=i8，有 cmpMode 属性 | tcmp | 模板 + i8输出 |
| D: Reduction | 1 in + 1 out，shape 不对称（rows→1 或 cols→1） | tcolmax, tcolsum | 模板 + 非对称shape |
| E: Cube+Vector | 涉及 cube 空间，cube+vec 双 kernel | tmatmul, tpush/tpop | **不自动生成** |
| F: 一元 op | 1 in + 1 out，shape 对称，dtype 一致 | texp, tsqrt, trecip, tneg, trelu | 模板化（少1个输入） |
| G: Broadcast | 2 in（shape 不对称）+ 1 out | tcolexpandadd, trowexpandadd | **不自动生成** |
| H: 类型转换 | 1 in + 1 out，dtype 不同（输出≠i8） | tcvt | **不自动生成** |

### 0.3 跨文件命名约定

这是最常见的静默错误来源。命名约定如下：

```
.pto 中 kernel 名:   @<OP>_<dtype>_<shape_suffix>
  例: @TADD_f32_16x64  → 大写 op 名 + 下划线 + dtype + 下划线 + 行数x列数

launch.cpp 声明:     extern "C" __global__ AICORE void <OP>_<dtype>_<shape_suffix>(...)
launch.cpp 包装:     void Launch<OP>_<dtype>_<shape_suffix>(...)
                       → 在 kernel 名前加 "Launch" 前缀

main.cpp kCases[]:   .launch = Launch<OP>_<dtype>_<shape_suffix>
main.cpp 目录名:     ./<case_name>/
  case_name 来自 cases.py 的 "name" 字段，如 "f32_16x64"

cases.py "name":     f"<dtype_short>_{rows}x{cols}"
                       → 不含 op 名前缀，dtype 用简称（f32/f16/i32/i16）
```

**一致性检查**：生成完所有文件后，验证以下 4 处字符串是否一致：
1. `.pto` 的 `func.func @NAME`
2. `launch.cpp` 的 `extern ... void NAME(...)` 和 `void LaunchNAME(...)`
3. `main.cpp` 的 `LaunchNAME`
4. `cases.py` 的 `"name"`（用于子目录名）

---

## 阶段 1：自动生成测试 Harness

### 1.1 类别 A 黄金样例（tadd 完整实现）

以下是从项目中提取的**真实可运行代码**，作为生成其他类别 A 算子的基础。

#### `tadd.pto`（完整，无省略）

```mlir
module attributes {pto.target_arch = "a5", pto.kernel_kind = #pto.kernel_kind<vector>} {
  func.func @TADD_f32_16x64(%a_ptr: !pto.ptr<f32>, %b_ptr: !pto.ptr<f32>, %c_ptr: !pto.ptr<f32>) attributes {pto.kernel} {
    %c0    = arith.constant 0    : index
    %c1    = arith.constant 1    : index
    %c16   = arith.constant 16   : index
    %c64   = arith.constant 64   : index
    %c1024 = arith.constant 1024 : index

    %a_view = pto.make_tensor_view %a_ptr,
      shape = [%c1, %c1, %c1, %c16, %c64],
      strides = [%c1024, %c1024, %c1024, %c64, %c1]
      : !pto.tensor_view<1x1x1x16x64xf32>
    %b_view = pto.make_tensor_view %b_ptr,
      shape = [%c1, %c1, %c1, %c16, %c64],
      strides = [%c1024, %c1024, %c1024, %c64, %c1]
      : !pto.tensor_view<1x1x1x16x64xf32>
    %c_view = pto.make_tensor_view %c_ptr,
      shape = [%c1, %c1, %c1, %c16, %c64],
      strides = [%c1024, %c1024, %c1024, %c64, %c1]
      : !pto.tensor_view<1x1x1x16x64xf32>

    %a_part = pto.partition_view %a_view,
      offsets = [%c0, %c0, %c0, %c0, %c0],
      sizes = [%c1, %c1, %c1, %c16, %c64]
      : !pto.tensor_view<1x1x1x16x64xf32> -> !pto.partition_tensor_view<1x1x1x16x64xf32>
    %b_part = pto.partition_view %b_view,
      offsets = [%c0, %c0, %c0, %c0, %c0],
      sizes = [%c1, %c1, %c1, %c16, %c64]
      : !pto.tensor_view<1x1x1x16x64xf32> -> !pto.partition_tensor_view<1x1x1x16x64xf32>
    %c_part = pto.partition_view %c_view,
      offsets = [%c0, %c0, %c0, %c0, %c0],
      sizes = [%c1, %c1, %c1, %c16, %c64]
      : !pto.tensor_view<1x1x1x16x64xf32> -> !pto.partition_tensor_view<1x1x1x16x64xf32>

    %a = pto.alloc_tile
      : !pto.tile_buf<vec, 16x64xf32>
    %b = pto.alloc_tile
      : !pto.tile_buf<vec, 16x64xf32>
    %c = pto.alloc_tile
      : !pto.tile_buf<vec, 16x64xf32>

    pto.tload ins(%a_part : !pto.partition_tensor_view<1x1x1x16x64xf32>)
              outs(%a : !pto.tile_buf<vec, 16x64xf32>)
    pto.tload ins(%b_part : !pto.partition_tensor_view<1x1x1x16x64xf32>)
              outs(%b : !pto.tile_buf<vec, 16x64xf32>)

    pto.tadd ins(%a, %b : !pto.tile_buf<vec, 16x64xf32>,
                          !pto.tile_buf<vec, 16x64xf32>)
             outs(%c : !pto.tile_buf<vec, 16x64xf32>)

    pto.tstore ins(%c : !pto.tile_buf<vec, 16x64xf32>)
               outs(%c_part : !pto.partition_tensor_view<1x1x1x16x64xf32>)
    return
  }
}
```

关键规则：
- 5D shape：`[1, 1, 1, rows, cols]`，strides = `[rows*cols, rows*cols, rows*cols, cols, 1]`
- partition_view 的 sizes 后两维 = `valid_shape`（当 valid ≠ allocated 时不同）
- 类型统一：`!pto.tile_buf<vec, RxCx<dtype>>`

#### `launch.cpp`

```cpp
#include <stdint.h>

#ifndef AICORE
#define AICORE [aicore]
#endif

extern "C" __global__ AICORE void TADD_f32_16x64(__gm__ float *a, __gm__ float *b, __gm__ float *c);

void LaunchTADD_f32_16x64(float *a, float *b, float *c, void *stream) {
    TADD_f32_16x64<<<1, nullptr, stream>>>((__gm__ float *)a, (__gm__ float *)b, (__gm__ float *)c);
}
```

类型映射：`f32→float`, `f16→uint16_t`, `bf16→uint16_t`, `i32→int32_t`, `i16→int16_t`, `i8→int8_t`

#### `main.cpp`

```cpp
#include "acl/acl.h"
#include "test_common.h"
using namespace PtoTestCommon;

void LaunchTADD_f32_16x64(float *a, float *b, float *c, void *stream);
using LaunchFn = void (*)(float *, float *, float *, void *);

struct TestCase {
    const char *name; LaunchFn launch;
    size_t rows, cols, validRows, validCols, elemSize;
};
static const TestCase kCases[] = {
    {"f32_16x64", LaunchTADD_f32_16x64, 16, 64, 16, 64, sizeof(float)},
};
// RunCase: ReadFile(input*.bin) → H2D → tc.launch() → D2H → WriteFile(output.bin)
// main: aclInit → for each kCase → RunCase → aclFinalize
```

完整代码参考 `test/tilelang_st/npu/a5/src/st/testcase/tadd/main.cpp`。

#### `cases.py`

```python
import numpy as np

CASES = [
    {"name": "f32_16x64", "dtype": np.float32, "shape": (16, 64), "valid_shape": (16, 64), "eps": 1e-6},
]
```

#### `gen_data.py`

```python
import numpy as np
from cases import CASES
from st_common import validate_cases, setup_case_rng, save_case_data

validate_cases(CASES)

for case in CASES:
    setup_case_rng(case)
    dtype = case["dtype"]
    shape = case["shape"]
    vr, vc = case["valid_shape"]

    input1 = np.random.randint(1, 10, size=shape).astype(dtype)
    input2 = np.random.randint(1, 10, size=shape).astype(dtype)

    golden = np.zeros(shape, dtype=dtype)
    golden[:vr, :vc] = (input1[:vr, :vc] + input2[:vr, :vc]).astype(dtype, copy=False)

    save_case_data(case["name"], {"input1": input1, "input2": input2, "golden": golden})
```

#### `compare.py`

```python
import os, sys, numpy as np
from cases import CASES
from st_common import result_cmp, style_fail, style_pass, validate_cases

validate_cases(CASES)
for case in CASES:
    golden = np.fromfile(f"{case['name']}/golden.bin", dtype=case["dtype"]).reshape(case["shape"])
    output = np.fromfile(f"{case['name']}/output.bin", dtype=case["dtype"]).reshape(case["shape"])
    vr, vc = case["valid_shape"]
    ok = result_cmp(golden[:vr,:vc], output[:vr,:vc], case["eps"])
    print(style_pass(f"PASS {case['name']}") if ok else style_fail(f"FAIL {case['name']}"))
```

#### `CMakeLists.txt`

```cmake
pto_tilelang_vec_st(tadd)
```

### 1.2 其他类别与类别 A 的差异表

基于上面的黄金样例，以下列出各类别的**增量差异**（只列不同之处）：

| 差异点 | B: 标量 op | C: 比较 op | D: Reduction | F: 一元 op |
|---|---|---|---|---|
| `.pto` 参数 | 多 `%scalar: T` | 同 A | 同 A（只有 src, dst） | 只有 `%src, %dst` |
| `.pto` op | `pto.tadds ins(%src, %scalar)` | `pto.tcmp ins(%a, %b {cmpMode=...})` | `pto.tcolmax ins(%src)` | `pto.texp ins(%src)` |
| 输出 dtype | 同输入 | `i8`（固定） | 同输入 | 同输入 |
| 输出 shape | 同输入 | 同输入 | `(1, C)` 或 `(R, 1)` | 同输入 |
| `launch.cpp` 参数 | `src, dst, scalar` | `a, b(float*), c(int8_t*)` | `src, dst`（2 个 ptr） | `src, dst`（2 个 ptr） |
| `main.cpp` 输入文件 | `input.bin`（1 个） | `input1.bin + input2.bin` | `input.bin`（1 个） | `input.bin`（1 个） |
| `cases.py` 额外字段 | `"scalar": 2.5` | `"dst_dtype": np.int8, "cmp_mode": "eq"` | `"dst_shape", "dst_valid_shape"` | 无 |
| golden 公式 | `input + scalar` | 见 1.3 节（packed bitmask，不可直接用 astype） | `np.max(a, axis=0)` | `np.exp(a)` |

**不自动生成的类别**：E（Cube+Vector）、G（Broadcast）、H（tcvt）。参考现有 testcase 目录手工编写。

### 1.3 golden 公式速查

| Op | numpy golden |
|---|---|
| tadd | `a + b` |
| tsub | `a - b` |
| tmul | `a * b` |
| tdiv | `a / b` |
| tmax | `np.maximum(a, b)` |
| tmin | `np.minimum(a, b)` |
| tadds | `a + scalar` |
| tmuls | `a * scalar` |
| texp | `np.exp(a)` |
| tsqrt | `np.sqrt(a)` |
| trecip | `1.0 / a` |
| tlog | `np.log(a)` |
| tneg | `-a` |
| trelu | `np.maximum(0, a)` |
| tcolmax | `np.max(a[:vr,:vc], axis=0, keepdims=True)` |
| tcolsum | `np.sum(a[:vr,:vc], axis=0, keepdims=True)` |
| tcmp eq | 精确匹配 eps=0。输出是 packed bitmask（1 bit/element，每 byte 存 8 个比较结果）。参考 `tcmp/gen_data.py` |

---

## 阶段 2：TPrint 调试

### 2.1 TPrint 约束

| 可打印类型 | 约束 |
|---|---|
| `!pto.tile_buf<loc=vec, ...>` | dtype 限 `f16/f32/i8/i16/i32` |
| `!pto.partition_tensor_view` | 无额外约束 |
| `memref` | 无额外约束 |

**关键限制**：
- **仅 vec 空间可用**。mat/left/right/acc/bias 等 cube 空间不可用 tprint
- cube 内部不可直接观察，但可通过 tpop 后的 vec tile 间接验证 cube 计算结果
- Expand 后 `.pto` 中的 tile op 已被 DSL 模板替换，tprint 观察的是 Expand **前**的 tile 值
- 如需观察 Expand **后**的微指令级值，用 `--vpto-print-ir` dump VPTO IR

### 2.2 TPrint 输出格式详解

#### 基本格式

TPRINT 输出由 **header 行 + shape 行 + 数据行** 组成：

```
=== [TPRINT Tile] Data Type: <dtype>, Layout: <layout>, TileType: Vec ===
  Shape: [rows, cols], Valid Shape: [valid_rows, valid_cols]
  <row0 数据>
  <row1 数据>
  ...
```

#### dtype 决定数据格式

| dtype | 打印格式 | 示例 |
|---|---|---|
| f32 | 十进制浮点，2 位小数 | `0.00  1.00  2.00  10.00  127.00` |
| bf16 | hex 16 位 | `0x0000 0x3f80 0x4000 0x4040` |
| f16 | hex 16 位 | `0x0000 0x3c00 0x4000` |
| i32/i16/i8 | 十进制整数 | `0  1  2  -3  127` |

#### bf16 常用值速查表

bf16 取 f32 的高 16 位。调试时需要心算对照：

| f32 值 | bf16 hex | f32 值 | bf16 hex |
|---|---|---|---|
| 0.0 | 0x0000 | 10.0 | 0x4120 |
| 1.0 | 0x3f80 | 20.0 | 0x41a0 |
| 2.0 | 0x4000 | 30.0 | 0x41f0 |
| 3.0 | 0x4040 | 40.0 | 0x4220 |
| 4.0 | 0x4080 | 50.0 | 0x4248 |
| 5.0 | 0x40a0 | 60.0 | 0x4270 |
| 6.0 | 0x40c0 | 70.0 | 0x428c |
| 7.0 | 0x40e0 | 80.0 | 0x42a0 |
| 8.0 | 0x4100 | 90.0 | 0x42b4 |
| 9.0 | 0x4110 | 100.0 | 0x42c8 |

规律：整数 N 的 bf16 = `struct.pack('>f', float(N))[:2]` 的 hex。

#### subblock 分裂规则

当 tile 行数 > 8 时，AIV 硬件将 tile 分为多个 subblock（每组 8 行）：
- **subblock_idx=0**：rows 0..7
- **subblock_idx=1**：rows 8..15
- 两组日志在运行时**交错出现**（不保证顺序）

真实输出示例（16×128 tile，subblock_idx=0，输入公式 `row*10+col`）：
```
=== [TPRINT Tile] Data Type: float32, Layout: ND, TileType: Vec ===
  Shape: [8, 128], Valid Shape: [8, 128]
  0.00   1.00   2.00   3.00  ...  127.00
  10.00  11.00  12.00  13.00 ...  137.00
  ...
  70.00  71.00  72.00  73.00 ...  197.00
```

subblock_idx=1（rows 8..15）第 0 行为 `80.00 81.00 ... 207.00`。

#### Layout 字段含义

| Layout | 含义 | 何时出现 |
|---|---|---|
| ND | Normal Dense（行优先连续存储） | TLOAD 后、计算 op 后 |
| Nz | fractal/Z-order 布局 | TMOV 后（进入 cube 前的布局转换） |

**关键**：TMOV 只改 layout 不改逻辑值。如果 TMOV 前后数值不同 → TMOV lowering 有 bug。

### 2.3 固定公式输入法（核心调试方法论）

**优先使用固定公式输入**而非随机数据——可人工心算验证，100% 确定性复现，看一眼就知道对不对。

#### Step 1：选择输入公式

推荐公式（从简单到复杂）：

| 公式 | 特点 | 适用场景 |
|---|---|---|
| `row * 10 + col` | 每个元素唯一，心算容易 | **首选**，适用大多数场景 |
| `(row % 4) * 3 - (col % 5)` | 有正有负，值域小 | 测试有符号运算 |
| `1.0 if row == col else 0.0` | 单位矩阵 | **cube tmatmul 验证技巧** |
| `float(col)` | 纯列序 | 验证 reduction（row-wise） |

**cube 验证技巧（identity weight）**：将 tmatmul 的 weight 矩阵设为单位矩阵 I，则：
- `output = input @ I = input`（当 M=K=N 时）
- 如果 input 是 M×K，weight 是 K×N 的 I，output 就是 input 的前 N 列
- 这样无需理解矩阵乘法细节，直接验证 tpop 后的 vec tile 值 = 输入的对应列

#### Step 2：编写预期参考

对每个 TPRINT 点，预先推算预期输出。格式模板：

```
## 输入定义
- input_a[row, col] = <公式>, shape=R×C, dtype=<type>
- input_b[row, col] = <公式>, shape=R×C, dtype=<type>

## TPRINT(vN) — after <OP>
预期：dtype=<type>, Layout=<ND|Nz>, Shape=[8,C]
  subblock_idx=0: row0=[<公式代入 row=0>], row1=[<代入 row=1>] ...
验证要点：<与上一个 checkpoint 的数学关系>
```

具体示例见 2.4 节的 Cube+Vector 定位示例。

#### Step 3：插入 tprint

在 `.pto` 中：
```mlir
pto.tload ins(%a_part : ...) outs(%a : ...)
pto.tprint ins(%a : !pto.tile_buf<vec, 16x128xf32>)   // v22

pto.tcvt ins(%a : ...) outs(%a_bf16 : ...)
pto.tprint ins(%a_bf16 : !pto.tile_buf<vec, 16x128xbf16>)  // v26

pto.tmov ins(%a_bf16 : ...) outs(%a_nz : ...)
pto.tprint ins(%a_nz : !pto.tile_buf<vec, 16x128xbf16>)    // v27
```

#### Step 4：按 subblock 分段比对

从 simulator log 中提取 TPRINT 输出，按 `subblock_idx` 分段，与 Step 2 的预期**逐行比对**。

**第一个不匹配的 checkpoint = bug 所在的 op**。不需要猜测，直接定位。

### 2.4 逐 checkpoint 定位流程

按数据流变换边界系统打桩，每个 checkpoint 只验证一个变换阶段：

```
GM ─TLOAD─→ tile ─TCVT/TMOV─→ tile ─OP─→ tile ─TSTORE─→ GM
    [A]           [B]              [C]          [D=golden比对]
```

**打桩规则**：
1. 每个 TLOAD 之后 → 验证 DMA 加载
2. 每个 TCVT/TMOV 之后 → 验证类型转换/布局搬运
3. 每个计算 op 之后 → 验证计算逻辑
4. 跨单元传输后 → tpop_from_aic 后验证 cube 结果
5. TSTORE 之前 → 最终状态

**定位逻辑**（从前往后比对，首个出错点 = root cause）：
```
A 错误 → strides / tensor_view / partition_view 配置错
A 正确、B 错误 → TCVT 类型转换或 TMOV 布局搬运的 lowering 问题
B 正确、C 错误 → 计算 op 的 lowering/编码有问题
C 正确、D≠golden → tstore / partition_view strides 有问题
所有 TPRINT 正确但最终 output.bin 错误 → TSTORE 路径问题
```

#### 简单 vec 算子定位示例（tadd）

输入：`A[r,c] = r*10+c`，`B[r,c] = c`

```
v0: TLOAD(A)  → row0=[0,1,2,...,63], row1=[10,11,...,73] ✓
v1: TLOAD(B)  → row0=[0,1,2,...,63], row1=[0,1,...,63]   ✓
v2: TADD(A,B) → row0=[0,2,4,...,126], row1=[10,12,...,136] ✓/✗
v3: =golden   → 与 output.bin 比对
```

#### Cube+Vector 定位示例

数据流：`TLOAD(attn_out) → TCVT(→bf16) → TMOV(ND→Nz) → [cube:tmatmul] → tpop → TADD → TSTORE`

输入：`attn_out[r,c] = r*10+c (f32, 16×128)`，`wo = I (bf16, 128×64)`

```
v22: TLOAD 后  → 8×128 f32 ND    验证: row0=[0..127], row1=[10..137]
v26: TCVT 后  → 8×128 bf16 ND   验证: row0=[0x0000,0x3f80,...] = f32→bf16
v27: TMOV 后  → 8×128 bf16 Nz   验证: 值=v26(不变), layout ND→Nz
v28: tpop 后  → 8×64 f32 ND    验证: = attn_out前64列 (因为 wo=I)
v30: TADD 后  → 8×64 f32 ND    验证: = v28 + hidden_states
```

定位逻辑：
```
v22 错 → DMA / tensor_view strides 配错
v22 对、v26 错 → TCVT lowering（f32→bf16 编码错误）
v26 对、v27 错 → TMOV lowering（值变了说明搬运有问题）
v27 对、v28 错 → cube tmatmul 或 tpush/tpop 机制有问题
v28 对、v30 错 → TADD lowering
v30 对、output.bin 错 → TSTORE 路径
```

定位后**移除所有 tprint**（增加模拟时间）。

---

## 阶段 3：症状快速匹配

### 3.1 构建/编译失败决策树

```
什么阶段报错？
├─ ptoas crash/assert fail
│   ├─ 加了 --enable-tile-op-expand → DSL 模板 bug 或 daemon 缓存过期
│   │   → pkill -f tilelib; 检查 lib/TileOps/a5/<op>.py
│   └─ 没加 → VPTO lowering 或 pass pipeline bug
│       → --vpto-print-ir 看 VPTO IR; 检查 VPTOLLVMEmitter.cpp
│
├─ bisheng 编译/链接失败
│   ├─ "undefined reference to <KERNEL_NAME>" → launch.cpp extern 声明与 .pto kernel 名不匹配
│   ├─ "Broken module" → VPTO lowering 控制流问题
│   └─ "Intrinsic has incorrect argument type" → HIVM ABI 规范化缺失
│       → 检查 VPTOLLVMEmitter.cpp normalizeByteScalarOperandForHivmCall
│
├─ cmake/make 失败
│   ├─ ModuleNotFoundError → Python 依赖或 PYTHONPATH 未设
│   ├─ select_kernel() no kernel → DSL 模板约束不匹配
│   └─ ptoas 输出为空 → 检查 run_ptoas_to_file.cmake 日志
│
└─ 模拟器运行 crash
    ├─ output.bin 全零 → tstore 未执行或 partition_view 覆盖空区域
    ├─ output.bin 大小不对 → shape × elemSize 不一致
    └─ ACL invalid pointer → H2D/D2H copy 大小与 buffer 不匹配
```

### 3.2 数值比对决策树

```
数值比对结果？
├─ 全部通过               → 算子正确
├─ eq 通过、非 eq 全错     → cmpMode 属性漏传 → ExpandTileOp.cpp appendOpContextAttrs
├─ 全部有偏差              → 常量/编码反转（SAT/NOSAT/ROUND）→ VPTOLLVMEmitter.cpp immediate
├─ 大 tile 对、小 tile 错  → Mask 缺失 → 三层排查（3.3）
├─ 部分 case 对、部分错
│   ├─ 失败 case dtype 有规律 → DSL 模板约束问题
│   └─ 无规律 → 属性漏传
└─ output.bin 全零/未生成   → 回到 3.1
```

### 3.3 Mask 缺失三层排查

| 层 | 根因 | 检查方式 |
|---|---|---|
| ODS 定义 | `.td` 缺少 mask 操作数 | `grep mask include/PTO/IR/VPTOOps.td` 查对应 op |
| IR 使用 | ODS 有 mask 但 Expand 不传实参 | `ExpandTileOp.cpp` 对应 op 生成逻辑 |
| Lowering | mask 存在但值计算/传递错误 | `VPTOLLVMEmitter.cpp` 中 mask 编码 |

---

## 阶段 4：根因定位

TPrint 不匹配时，按以下顺序排查：

**4a. 先排查 `.pto` 自身**：strides / partition offsets / layout / dtype / op 语义

**4b. 分析不匹配模式**：

| 差异模式 | 指向 | 排查 |
|---|---|---|
| 所有元素偏移一致 | 常量/编码字段填错 | `VPTOLLVMEmitter.cpp` 编码 immediate |
| 仅尾部/边界元素错误 | Mask 缺失 | 3.3 三层排查 |
| 符号翻转/全零/错位 | 数据布局传递错误 | tmov/broadcast 的 lowering |
| 浮点误差 1e-3~1e-5 | 取整/饱和模式不对 | SAT/NOSAT/ROUND 编码 |

**4c. 查 Expand 后 VPTO IR**：`ptoas --vpto-print-ir --enable-tile-op-expand ...`

**4d. 查 lowering 函数**：`VPTOLLVMEmitter.cpp` 中的 `emit*/visit*`，对照 `docs/isa/tile-op/`

---

## 阶段 5：修复与验证

| Bug 层 | 文件 | 修复后动作 |
|---|---|---|
| ODS 定义 | `include/PTO/IR/*.td` | `ninja -C build ptoas` + 清 build |
| C++ IR/Verifier | `lib/PTO/IR/PTO.cpp` | `ninja -C build ptoas` + 清 build |
| Expand/属性转发 | `lib/PTO/Transforms/TileOp/ExpandTileOp.cpp` | `ninja -C build ptoas` + 清 build |
| VPTO→LLVM lowering | `lib/PTO/Transforms/VPTOLLVMEmitter.cpp` | `ninja -C build ptoas` + 清 build |
| PTODSL TileLib 模板 | `lib/TileOps/a5/*.py` | 清 build + 重启 TileLib daemon |
| `.pto` 文件 | `*.pto` | 仅重跑 ptoas |

**清 build**：`rm -rf test/tilelang_st/npu/a5/src/st/build`

**完整验证**：
```bash
ninja -C build ptoas && pip install -e . --no-build-isolation && \
  rm -rf test/tilelang_st/npu/a5/src/st/build && \
  python3 test/tilelang_st/script/run_st.py -r sim -v a5 -t <op>
```

### run_st.py 常用参数

```bash
# 运行单个 op 的所有 case
python3 test/tilelang_st/script/run_st.py -r sim -v a5 -t tadd

# 只运行特定 case
python3 test/tilelang_st/script/run_st.py -r sim -v a5 -t tadd -c f32_16x64

# 跳过构建（使用已有 build）
python3 test/tilelang_st/script/run_st.py -r sim -v a5 -t tadd --without-build
```

---

## 护栏

- 生成代码前先分类（阶段 0），不要对 E/G/H 类算子盲目生成
- 生成完所有文件后，交叉验证命名约定（0.3）——4 处字符串必须一致
- TPrint 仅 vec 空间可用，cube 空间通过 tpop 后的 vec tile 间接验证
- 修改 ptoas 或 DSL 层后必须清 build
- 调试期间 `pkill -f daemon_server` 避免 DSL 缓存干扰
- 定位后移除所有 tprint
- ST testcase 的 gen_data.py / compare.py 只处理最终输出 golden，中间值用阶段 2 的固定公式 + 预期推算方法验证
